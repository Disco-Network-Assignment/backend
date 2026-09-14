"""The campaign pipeline:

    intake -> route -> signals -> match + guards -> personas -> creatives (parallel, checked)
           -> config -> summary

Code owns the order of the stages; inside a stage the agent may call tools or hand off, but it
always returns a typed object. Each stage yields `PipelineEvent`s as it goes, which the API
streams as NDJSON so the UI fills in panel by panel."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.agents.context import RunContext
from app.agents.llm_stages import StageExecutor
from app.domain.catalog import CatalogRepository
from app.domain.config_builder import ConfigBuilder
from app.domain.creative_checks import check_lengths
from app.domain.fit_signals import SignalCalculator
from app.domain.guardrails import AssessmentGuard
from app.domain.input_policy import InputPolicy, RouteDecision
from app.enums import ConfigStatus, EventStatus, FailureKind, Stage, Verdict
from app.errors import StageError
from app.schemas import (
    AdvertiserBrief,
    CampaignConfig,
    CampaignPlan,
    CampaignSummary,
    ClarificationRequest,
    CreativeVariant,
    FitSignals,
    LintReport,
    PersonaPick,
    PersonaPickDraft,
    PersonaSelection,
    PipelineEvent,
    PlanRequest,
    PlanResponse,
    PublisherAssessment,
    RejectedPersona,
    StageMeta,
    StopResult,
)

logger = logging.getLogger(__name__)


@dataclass
class RunState:
    run_id: str
    request: PlanRequest
    ctx: RunContext
    trace: list[StageMeta] = field(default_factory=list)
    brief: AdvertiserBrief | None = None
    decision: RouteDecision | None = None
    signals: list[FitSignals] = field(default_factory=list)
    assessments: list[PublisherAssessment] = field(default_factory=list)
    personas: PersonaSelection | None = None
    creatives: list[CreativeVariant] = field(default_factory=list)
    config: CampaignConfig | None = None
    summary: CampaignSummary | None = None

    @property
    def recommended(self) -> list[PublisherAssessment]:
        return [a for a in self.assessments if a.verdict is Verdict.RECOMMEND]


class CampaignPipeline:
    def __init__(self, executor: StageExecutor, catalog: CatalogRepository,
                 signal_calculator: SignalCalculator, guard: AssessmentGuard, policy: InputPolicy,
                 config_builder: ConfigBuilder, summary_enabled: bool = True) -> None:
        self._executor = executor
        self._catalog = catalog
        self._signals = signal_calculator
        self._guard = guard
        self._policy = policy
        self._config_builder = config_builder
        self._summary_enabled = summary_enabled

    # ------------------------------------------------------------------ public API
    async def stream(self, request: PlanRequest) -> AsyncIterator[PipelineEvent]:
        state = RunState(run_id=uuid.uuid4().hex[:12], request=request,
                         ctx=self._executor.new_context(request.description))
        logger.info("[PIPELINE] run=%s start", state.run_id)
        try:
            async for event in self._run(state):
                yield event
        except StageError as e:
            logger.warning("[PIPELINE] run=%s stage=%s failed (%s): %s", state.run_id, e.stage, e.kind, e.message)
            yield PipelineEvent(stage=e.stage, status=EventStatus.FAILED, kind=e.kind, message=e.message)
        except Exception as e:  # noqa: BLE001 - the stream must end with a typed event
            logger.exception("[PIPELINE] run=%s unexpected failure", state.run_id)
            yield PipelineEvent(stage=Stage.ERROR, status=EventStatus.FAILED, kind=FailureKind.UNKNOWN,
                                message=f"unexpected error: {e}")

    async def run(self, request: PlanRequest) -> PlanResponse:
        """The same run without streaming: the last event decides the outcome."""
        last: PipelineEvent | None = None
        async for event in self.stream(request):
            last = event
        if last is None or last.status is EventStatus.FAILED:
            raise StageError(last.stage if last else Stage.ERROR, last.kind if last and last.kind
                             else FailureKind.UNKNOWN, last.message if last else "no events")
        if last.stage is Stage.STOPPED:
            return PlanResponse(status="stopped", stopped=StopResult.model_validate(last.data))
        return PlanResponse(status="done", plan=CampaignPlan.model_validate(last.data))

    # ------------------------------------------------------------------ stages
    async def _run(self, state: RunState) -> AsyncIterator[PipelineEvent]:
        yield _started(Stage.INTAKE)
        run = await self._executor.intake(state.ctx, state.request.options.session_id)
        state.trace.append(run.meta)
        if isinstance(run.output, ClarificationRequest):
            yield self._stopped(state, run.output.reason, run.output.questions)
            return
        state.brief = state.ctx.brief = run.output
        yield _completed(Stage.INTAKE, state, state.brief)
        state.decision = self._policy.route(state.brief)
        if state.decision.stop:  # the brief writer itself judged the input insufficient
            yield self._stopped(state, state.decision.reason or self._policy.STOP_REASON,
                                state.brief.clarifying_questions)
            return

        yield _started(Stage.SIGNALS)
        state.signals = self._signals.compute_all(state.brief)
        state.ctx.fit_signals = {s.publisher_id: s for s in state.signals}
        state.trace.append(StageMeta(stage=Stage.SIGNALS, ms=0))
        yield _completed(Stage.SIGNALS, state, state.signals)

        yield _started(Stage.MATCH)
        run = await self._executor.match(state.ctx)
        state.trace.append(run.meta)
        state.assessments = self._guard.apply(run.output, state.ctx.fit_signals, state.brief)
        state.ctx.recommended = state.recommended
        yield _completed(Stage.MATCH, state, state.assessments)

        if state.recommended or state.request.options.force_exploratory:
            yield _started(Stage.PERSONAS)
            state.personas = await self._personas(state)
            yield _completed(Stage.PERSONAS, state, state.personas)

            yield _started(Stage.CREATIVE)
            async for event in self._creatives(state):
                yield event
            yield _completed(Stage.CREATIVE, state, state.creatives)

        yield _started(Stage.CONFIG)
        state.config = self._config_builder.build(state.brief, state.assessments, state.personas,
                                                  state.creatives, state.decision)
        state.trace.append(StageMeta(stage=Stage.CONFIG, ms=0))
        yield _completed(Stage.CONFIG, state, state.config)

        if self._summary_enabled and state.config.status is ConfigStatus.DRAFT:
            yield _started(Stage.SUMMARY)
            run = await self._executor.summarize(state.ctx, self._plan_view(state))
            state.trace.append(run.meta)
            state.summary = run.output
            yield _completed(Stage.SUMMARY, state, state.summary)

        plan = CampaignPlan(run_id=state.run_id, description=state.request.description, brief=state.brief, publishers=state.assessments, personas=state.personas,
                            creatives=state.creatives, config=state.config, summary=state.summary,
                            trace=state.trace)
        logger.info("[PIPELINE] run=%s done recommended=%d creatives=%d status=%s", state.run_id,
                    len(state.recommended), len(state.creatives), state.config.status)
        yield PipelineEvent(stage=Stage.DONE, status=EventStatus.COMPLETED, data=_dump(plan))

    def _stopped(self, state: RunState, reason: str, questions: list[str]) -> PipelineEvent:
        stop = StopResult(run_id=state.run_id, description=state.request.description, reason=reason,
                          clarifying_questions=questions or ["What do you sell, and who buys it?",
                                                             "What does a typical order cost?"],
                          examples=self._catalog.example_descriptions(), trace=state.trace)
        return PipelineEvent(stage=Stage.STOPPED, status=EventStatus.COMPLETED, data=_dump(stop))

    async def _personas(self, state: RunState) -> PersonaSelection:
        assert state.decision
        candidates = state.recommended or state.assessments[:3]  # exploratory runs use the least-bad
        state.ctx.recommended = candidates
        run = await self._executor.select_personas(state.ctx, state.decision.persona_cap)
        state.trace.append(run.meta)
        name = lambda persona_id: self._catalog.persona(persona_id).name  # noqa: E731
        valid_ids = {c.publisher_id for c in candidates}
        selected = [PersonaPick(**p.model_dump(exclude={"best_publishers"}), persona_name=name(p.persona_id),
                                best_publishers=[i for i in p.best_publishers if i in valid_ids])
                    for p in run.output.selected if self._catalog.has_persona(p.persona_id)]
        rejected = [RejectedPersona(**r.model_dump(), persona_name=name(r.persona_id))
                    for r in run.output.rejected if self._catalog.has_persona(r.persona_id)]
        return PersonaSelection(selected=selected, rejected=rejected)

    async def _creatives(self, state: RunState) -> AsyncIterator[PipelineEvent]:
        assert state.personas
        picks = state.personas.selected
        tasks = [asyncio.create_task(self._creative(state, pick)) for pick in picks]
        finished = 0
        try:
            for future in asyncio.as_completed(tasks):
                variant = await future
                finished += 1
                state.creatives.append(variant)
                yield PipelineEvent(stage=Stage.CREATIVE, status=EventStatus.PROGRESS,
                                    completed=finished, total=len(tasks), data=_dump(variant))
        finally:
            for task in tasks:
                task.cancel()
        order = {pick.persona_id: i for i, pick in enumerate(picks)}
        state.creatives.sort(key=lambda v: order[v.persona_id])

    async def _creative(self, state: RunState, pick: PersonaPick) -> CreativeVariant:
        """One copywriter run; the agent checks its own draft with the check_creative tool, and
        code runs the same length checks once more on what came back."""
        assert state.brief
        persona = self._catalog.persona(pick.persona_id)
        targets = pick.best_publishers or [a.publisher_id for a in state.recommended[:3]]
        draft_pick = PersonaPickDraft(**pick.model_dump(exclude={"persona_name"}))
        checks_before = state.ctx.creative_checks
        run = await self._executor.write_creative(state.ctx, draft_pick, persona, targets)
        state.trace.append(run.meta)
        issues = check_lengths(run.output)
        return CreativeVariant(
            **run.output.model_dump(), id=f"creative-{pick.persona_id}", persona_id=pick.persona_id,
            persona_name=persona.name, target_publishers=targets,
            lint=LintReport(passed=not issues, issues=issues,
                            self_checks=state.ctx.creative_checks - checks_before),
        )

    @staticmethod
    def _plan_view(state: RunState) -> dict:
        """What the summary stage reads: the decisions, not the whole payload."""
        assert state.brief and state.config and state.personas is not None
        return {
            "business": state.brief.business_summary,
            "input_quality": state.brief.input_quality,
            "allocation": [a.model_dump(mode="json") for a in state.config.publisher_allocation],
            "personas": [{"persona_name": p.persona_name, "angle": p.angle} for p in state.personas.selected],
            "creatives": [{"persona": c.persona_name, "headline": c.headline, "lint_passed": c.lint.passed}
                          for c in state.creatives],
            "budget": state.config.budget.model_dump(mode="json"),
            "bid_strategy": state.config.bid_strategy.model_dump(mode="json"),
            "kpi": state.config.kpis.primary,
            "open_questions": state.config.open_questions,
        }


# ---------------------------------------------------------------------- helpers

def _started(stage: Stage) -> PipelineEvent:
    return PipelineEvent(stage=stage, status=EventStatus.STARTED)


def _completed(stage: Stage, state: RunState, data) -> PipelineEvent:
    meta = next((m for m in reversed(state.trace) if m.stage is stage), None)
    return PipelineEvent(stage=stage, status=EventStatus.COMPLETED, ms=meta.ms if meta else 0, data=_dump(data))


def _dump(value):
    if isinstance(value, list):
        return [_dump(v) for v in value]
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value
