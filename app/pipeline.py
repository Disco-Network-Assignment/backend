"""The campaign pipeline.

    intake -> route -> signals -> match + guards -> personas -> creatives (parallel, checked)
           -> config -> summary

Code owns the order of the stages. Inside a stage the agent may call tools or hand off, but
it always returns a typed object. Each stage yields `PipelineEvent`s as it goes, which the
API streams as NDJSON so the UI fills in panel by panel.
"""

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
from app.runs import RunStore, new_record
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
    RunError,
    StageMeta,
    StopResult,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUN_ID_LENGTH = 12                    # hex characters; enough to find a run in the logs
EXPLORATORY_PUBLISHER_COUNT = 3       # when nothing is recommended, personas use the least-bad N
DEFAULT_CREATIVE_TARGET_COUNT = 3     # creatives without a persona preference target the top N
DEFAULT_CLARIFYING_QUESTIONS = [
    "What do you sell, and who buys it?",
    "What does a typical order cost?",
]


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


@dataclass
class RunState:
    """Everything one request has produced so far; each stage fills in its own field."""

    run_id: str
    request: PlanRequest
    ctx: RunContext                     # the context every agent and tool in this run shares
    trace: list[StageMeta] = field(default_factory=list)
    brief: AdvertiserBrief | None = None
    decision: RouteDecision | None = None
    signals: list[FitSignals] = field(default_factory=list)
    assessments: list[PublisherAssessment] = field(default_factory=list)
    personas: PersonaSelection | None = None
    creatives: list[CreativeVariant] = field(default_factory=list)
    config: CampaignConfig | None = None
    summary: CampaignSummary | None = None
    # the terminal outcome, whichever one happened
    plan: CampaignPlan | None = None
    stop: StopResult | None = None

    @property
    def recommended(self) -> list[PublisherAssessment]:
        recommended = []
        for assessment in self.assessments:
            if assessment.verdict is Verdict.RECOMMEND:
                recommended.append(assessment)
        return recommended


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class CampaignPipeline:
    def __init__(self, executor: StageExecutor, catalog: CatalogRepository,
                 signal_calculator: SignalCalculator, guard: AssessmentGuard, policy: InputPolicy,
                 config_builder: ConfigBuilder, summary_enabled: bool = True,
                 store: RunStore | None = None):
        self.executor = executor            # answers the judgement questions (the agents)
        self.catalog = catalog
        self.signals = signal_calculator
        self.guard = guard
        self.policy = policy
        self.config_builder = config_builder
        self.summary_enabled = summary_enabled
        self.store = store                  # run history; None means nothing is kept

    # ---- public API ----

    async def stream(self, request: PlanRequest) -> AsyncIterator[PipelineEvent]:
        """
        Run every stage and yield an event as each starts, progresses and completes. The
        stream always ends with a typed event: `done`, `stopped`, or `failed`.
        """
        state = RunState(
            run_id=uuid.uuid4().hex[:RUN_ID_LENGTH],
            request=request,
            ctx=self.executor.new_context(request.description),
        )
        log.info("[PIPELINE] run=%s start", state.run_id)
        try:
            async for event in self._run_stages(state):
                # store the outcome before the client hears about it, so a history refresh
                # triggered by the terminal event already sees this run
                if event.stage in (Stage.DONE, Stage.STOPPED):
                    await self._save_history(state)
                yield event
        except StageError as e:
            log.warning("[PIPELINE] run=%s stage=%s failed (%s): %s", state.run_id, e.stage, e.kind, e.message)
            await self._save_history(state, RunError(stage=e.stage, kind=e.kind, message=e.message))
            yield PipelineEvent(stage=e.stage, status=EventStatus.FAILED, kind=e.kind, message=e.message)
        except Exception as e:  # noqa: BLE001 - the stream must end with a typed event, whatever broke
            log.exception("[PIPELINE] run=%s unexpected failure", state.run_id)
            error = RunError(stage=Stage.ERROR, kind=FailureKind.UNKNOWN, message=f"unexpected error: {e}")
            await self._save_history(state, error)
            yield PipelineEvent(stage=error.stage, status=EventStatus.FAILED, kind=error.kind, message=error.message)

    async def _save_history(self, state: RunState, error: RunError | None = None):
        """Record the outcome (plan, stop or failure) so the dashboard can list and reload it."""
        if self.store is None:
            return
        record = new_record(state.run_id, state.request.description, state.request.options.session_id)
        if state.plan is not None:
            record.status = "done"
            record.plan = state.plan
            record.input_quality = state.plan.brief.input_quality
            record.recommended = len(state.plan.recommended)
            record.personas = len(state.plan.personas.selected) if state.plan.personas else 0
            record.creatives = len(state.plan.creatives)
            record.budget_usd = state.plan.config.budget.total_usd
        elif state.stop is not None:
            record.status = "stopped"
            record.stopped = state.stop
        else:
            record.status = "failed"
            record.error = error
            if state.brief is not None:
                record.input_quality = state.brief.input_quality
        await self.store.save(record)

    async def run(self, request: PlanRequest) -> PlanResponse:
        """The same run without streaming: the last event decides the outcome."""
        last = None
        async for event in self.stream(request):
            last = event

        if last is None:
            raise StageError(Stage.ERROR, FailureKind.UNKNOWN, "no events")
        if last.status is EventStatus.FAILED:
            raise StageError(last.stage, last.kind or FailureKind.UNKNOWN, last.message or "failed")
        if last.stage is Stage.STOPPED:
            return PlanResponse(status="stopped", stopped=StopResult.model_validate(last.data))
        return PlanResponse(status="done", plan=CampaignPlan.model_validate(last.data))

    # ---- the stages, in order ----

    async def _run_stages(self, state: RunState) -> AsyncIterator[PipelineEvent]:
        # --- Stage 1: intake (triage agent -> brief writer or clarifier) ---
        yield started(Stage.INTAKE)
        intake = await self.executor.intake(state.ctx, state.request.options.session_id)
        state.trace.append(intake.meta)

        # the clarifier answered: there is nothing to plan with
        if isinstance(intake.output, ClarificationRequest):
            yield self._stopped_event(state, intake.output.reason, intake.output.questions)
            return

        state.brief = intake.output
        state.ctx.brief = state.brief
        yield completed(Stage.INTAKE, state, state.brief)

        # --- Stage 2: route (code decides the consequence of the input quality) ---
        state.decision = self.policy.route(state.brief)
        if state.decision.stop:
            # the brief writer itself judged the input insufficient
            reason = state.decision.reason or self.policy.STOP_REASON
            yield self._stopped_event(state, reason, state.brief.clarifying_questions)
            return

        # --- Stage 3: signals (deterministic evidence per publisher) ---
        yield started(Stage.SIGNALS)
        state.signals = self.signals.compute_all(state.brief)
        for signal in state.signals:
            state.ctx.fit_signals[signal.publisher_id] = signal
        state.trace.append(StageMeta(stage=Stage.SIGNALS, ms=0))
        yield completed(Stage.SIGNALS, state, state.signals)

        # --- Stage 4: match (agent scores every publisher; guards enforce the rules) ---
        yield started(Stage.MATCH)
        match = await self.executor.match(state.ctx)
        state.trace.append(match.meta)
        state.assessments = self.guard.apply(match.output, state.ctx.fit_signals, state.brief)
        state.ctx.recommended = state.recommended
        yield completed(Stage.MATCH, state, state.assessments)

        # --- Stages 5 and 6: personas and creatives, only when there is something to run on ---
        worth_writing_creatives = bool(state.recommended) or state.request.options.force_exploratory
        if worth_writing_creatives:
            yield started(Stage.PERSONAS)
            state.personas = await self._select_personas(state)
            yield completed(Stage.PERSONAS, state, state.personas)

            yield started(Stage.CREATIVE)
            async for event in self._write_creatives(state):
                yield event
            yield completed(Stage.CREATIVE, state, state.creatives)

        # --- Stage 7: config (code assembles the plan from everything above) ---
        yield started(Stage.CONFIG)
        state.config = self.config_builder.build(state.brief, state.assessments, state.personas,
                                                 state.creatives, state.decision)
        state.trace.append(StageMeta(stage=Stage.CONFIG, ms=0))
        yield completed(Stage.CONFIG, state, state.config)

        # --- Stage 8: summary (optional narrative, only for a plan we recommend) ---
        if self.summary_enabled and state.config.status is ConfigStatus.DRAFT:
            yield started(Stage.SUMMARY)
            summary = await self.executor.summarize(state.ctx, self._plan_view(state))
            state.trace.append(summary.meta)
            state.summary = summary.output
            yield completed(Stage.SUMMARY, state, state.summary)

        # --- Done ---
        plan = CampaignPlan(
            run_id=state.run_id, description=state.request.description, brief=state.brief,
            publishers=state.assessments, personas=state.personas, creatives=state.creatives,
            config=state.config, summary=state.summary, trace=state.trace,
        )
        state.plan = plan
        log.info("[PIPELINE] run=%s done recommended=%d creatives=%d status=%s", state.run_id,
                 len(state.recommended), len(state.creatives), state.config.status)
        yield PipelineEvent(stage=Stage.DONE, status=EventStatus.COMPLETED, data=as_json(plan))

    def _stopped_event(self, state: RunState, reason: str, questions: list[str]) -> PipelineEvent:
        """The terminal event when the input is too thin to plan anything."""
        stop = StopResult(
            run_id=state.run_id, description=state.request.description, reason=reason,
            clarifying_questions=questions or DEFAULT_CLARIFYING_QUESTIONS,
            examples=self.catalog.example_descriptions(), trace=state.trace,
        )
        state.stop = stop
        return PipelineEvent(stage=Stage.STOPPED, status=EventStatus.COMPLETED, data=as_json(stop))

    # ---- personas ----

    async def _select_personas(self, state: RunState) -> PersonaSelection:
        # an exploratory run (nothing recommended) works with the least-bad publishers
        candidates = state.recommended
        if not candidates:
            candidates = state.assessments[:EXPLORATORY_PUBLISHER_COUNT]
        state.ctx.recommended = candidates

        run = await self.executor.select_personas(state.ctx, state.decision.persona_cap)
        state.trace.append(run.meta)

        # keep only publisher ids the agent was actually shown
        candidate_ids = set()
        for candidate in candidates:
            candidate_ids.add(candidate.publisher_id)

        selected = []
        for pick in run.output.selected:
            if not self.catalog.has_persona(pick.persona_id):
                continue  # an invented persona id is dropped rather than failing the run
            best_publishers = []
            for publisher_id in pick.best_publishers:
                if publisher_id in candidate_ids:
                    best_publishers.append(publisher_id)
            selected.append(PersonaPick(
                **pick.model_dump(exclude={"best_publishers"}),
                persona_name=self.catalog.persona(pick.persona_id).name,
                best_publishers=best_publishers,
            ))

        rejected = []
        for rejection in run.output.rejected:
            if not self.catalog.has_persona(rejection.persona_id):
                continue
            rejected.append(RejectedPersona(
                **rejection.model_dump(),
                persona_name=self.catalog.persona(rejection.persona_id).name,
            ))

        return PersonaSelection(selected=selected, rejected=rejected)

    # ---- creatives ----

    async def _write_creatives(self, state: RunState) -> AsyncIterator[PipelineEvent]:
        """One copywriter per persona, in parallel; a progress event as each one lands."""
        picks = state.personas.selected
        tasks = []
        for pick in picks:
            tasks.append(asyncio.create_task(self._write_creative(state, pick)))

        finished = 0
        try:
            for next_done in asyncio.as_completed(tasks):
                variant = await next_done
                finished += 1
                state.creatives.append(variant)
                yield PipelineEvent(stage=Stage.CREATIVE, status=EventStatus.PROGRESS,
                                    completed=finished, total=len(tasks), data=as_json(variant))
        finally:
            # if the client went away mid-way, do not leave copywriters running
            for task in tasks:
                task.cancel()

        # they finished in arrival order; present them in persona order
        position = {}
        for index, pick in enumerate(picks):
            position[pick.persona_id] = index

        def by_persona_order(variant: CreativeVariant) -> int:
            return position[variant.persona_id]

        state.creatives.sort(key=by_persona_order)

    async def _write_creative(self, state: RunState, pick: PersonaPick) -> CreativeVariant:
        """
        One copywriter run. The agent checks its own draft with the check_creative tool;
        code runs the same length check once more on what came back and reports both.
        """
        persona = self.catalog.persona(pick.persona_id)

        # the ad runs where the persona fits best, or on the top recommended publishers
        target_publishers = pick.best_publishers
        if not target_publishers:
            target_publishers = []
            for assessment in state.recommended[:DEFAULT_CREATIVE_TARGET_COUNT]:
                target_publishers.append(assessment.publisher_id)

        draft_pick = PersonaPickDraft(**pick.model_dump(exclude={"persona_name"}))
        checks_before = state.ctx.creative_checks
        run = await self.executor.write_creative(state.ctx, draft_pick, persona, target_publishers)
        state.trace.append(run.meta)

        issues = check_lengths(run.output)
        self_checks = state.ctx.creative_checks - checks_before
        return CreativeVariant(
            **run.output.model_dump(),
            id=f"creative-{pick.persona_id}",
            persona_id=pick.persona_id,
            persona_name=persona.name,
            target_publishers=target_publishers,
            lint=LintReport(passed=not issues, issues=issues, self_checks=self_checks),
        )

    # ---- summary ----

    @staticmethod
    def _plan_view(state: RunState) -> dict:
        """What the summary agent reads: the decisions, not the whole payload."""
        allocation = []
        for row in state.config.publisher_allocation:
            allocation.append(row.model_dump(mode="json"))

        personas = []
        for pick in state.personas.selected:
            personas.append({"persona_name": pick.persona_name, "angle": pick.angle})

        creatives = []
        for creative in state.creatives:
            creatives.append({"persona": creative.persona_name, "headline": creative.headline,
                              "lint_passed": creative.lint.passed})

        return {
            "business": state.brief.business_summary,
            "input_quality": state.brief.input_quality,
            "allocation": allocation,
            "personas": personas,
            "creatives": creatives,
            "budget": state.config.budget.model_dump(mode="json"),
            "bid_strategy": state.config.bid_strategy.model_dump(mode="json"),
            "kpi": state.config.kpis.primary,
            "open_questions": state.config.open_questions,
        }


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def started(stage: Stage) -> PipelineEvent:
    return PipelineEvent(stage=stage, status=EventStatus.STARTED)


def completed(stage: Stage, state: RunState, data) -> PipelineEvent:
    """A completed event carries the stage output and how long the stage took."""
    elapsed_ms = 0
    for meta in reversed(state.trace):
        if meta.stage is stage:
            elapsed_ms = meta.ms
            break
    return PipelineEvent(stage=stage, status=EventStatus.COMPLETED, ms=elapsed_ms, data=as_json(data))


def as_json(value):
    """Pydantic models (or lists of them) as plain dicts, ready for the NDJSON line."""
    if isinstance(value, list):
        converted = []
        for item in value:
            converted.append(as_json(item))
        return converted
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value
