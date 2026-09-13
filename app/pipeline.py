"""The campaign pipeline:

    intake -> route -> signals -> match + guards -> personas -> creatives (parallel, linted)
           -> config -> summary

A workflow, not an agent: the steps are known in advance, so code owns the control flow and the
model only ever answers a typed question. Each stage yields `PipelineEvent`s as it goes, which
the API streams as NDJSON so the UI fills in panel by panel."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.agents.llm_stages import StageExecutor
from app.domain.catalog import CatalogRepository
from app.domain.config_builder import ConfigBuilder
from app.domain.creative_checks import CreativeLinter, LintContext
from app.domain.fit_signals import SignalCalculator
from app.domain.guardrails import AssessmentGuard
from app.domain.input_policy import InputPolicy, RouteDecision
from app.enums import (
    ConfigStatus,
    EventStatus,
    ExecutionMode,
    FailureKind,
    GenderSkew,
    InputQuality,
    LintSeverity,
    PriceTier,
    ProductCategory,
    PurchaseModel,
    Stage,
    Verdict,
)
from app.errors import StageError
from app.schemas import (
    AdvertiserBrief,
    CampaignConfig,
    CampaignPlan,
    CampaignSummary,
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
    TargetCustomer,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    run_id: str
    request: PlanRequest
    mode: ExecutionMode
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
                 signal_calculator: SignalCalculator, guard: AssessmentGuard, router: InputPolicy,
                 linter: CreativeLinter, planner: ConfigBuilder,
                 summary_enabled: bool = True) -> None:
        self._executor = executor
        self._catalog = catalog
        self._signals = signal_calculator
        self._guard = guard
        self._router = router
        self._linter = linter
        self._planner = planner
        self._summary_enabled = summary_enabled

    @property
    def mode(self) -> ExecutionMode:
        return self._executor.mode

    # ------------------------------------------------------------------ public API
    async def stream(self, request: PlanRequest) -> AsyncIterator[PipelineEvent]:
        ctx = PipelineContext(run_id=uuid.uuid4().hex[:12], request=request, mode=self.mode)
        logger.info("[PIPELINE] run=%s mode=%s start", ctx.run_id, ctx.mode)
        try:
            async for event in self._run(ctx):
                yield event
        except StageError as e:
            logger.warning("[PIPELINE] run=%s stage=%s failed (%s): %s", ctx.run_id, e.stage, e.kind, e.message)
            yield PipelineEvent(stage=e.stage, status=EventStatus.FAILED, kind=e.kind, message=e.message)
        except Exception as e:  # noqa: BLE001 - the stream must end with a typed event
            logger.exception("[PIPELINE] run=%s unexpected failure", ctx.run_id)
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
    async def _run(self, ctx: PipelineContext) -> AsyncIterator[PipelineEvent]:
        yield _started(Stage.INTAKE)
        ctx.brief = await self._intake(ctx)
        yield _completed(Stage.INTAKE, ctx, ctx.brief)

        ctx.decision = self._router.route(ctx.brief)
        if ctx.decision.stop:
            stop = StopResult(run_id=ctx.run_id, description=ctx.request.description, mode=ctx.mode,
                              brief=ctx.brief, reason=ctx.decision.reason or "",
                              clarifying_questions=ctx.brief.clarifying_questions,
                              examples=self._catalog.example_descriptions(), trace=ctx.trace)
            yield PipelineEvent(stage=Stage.STOPPED, status=EventStatus.COMPLETED, data=_dump(stop))
            return

        yield _started(Stage.SIGNALS)
        ctx.signals = self._signals.compute_all(ctx.brief)
        ctx.trace.append(StageMeta(stage=Stage.SIGNALS, ms=0, mode=ctx.mode))
        yield _completed(Stage.SIGNALS, ctx, ctx.signals)

        yield _started(Stage.MATCH)
        run = await self._executor.match(ctx.brief, ctx.signals)
        ctx.trace.append(run.meta)
        ctx.assessments = self._guard.apply(run.output, {s.publisher_id: s for s in ctx.signals}, ctx.brief)
        yield _completed(Stage.MATCH, ctx, ctx.assessments)

        if ctx.recommended or ctx.request.options.force_exploratory:
            yield _started(Stage.PERSONAS)
            ctx.personas = await self._personas(ctx)
            yield _completed(Stage.PERSONAS, ctx, ctx.personas)

            yield _started(Stage.CREATIVE)
            async for event in self._creatives(ctx):
                yield event
            yield _completed(Stage.CREATIVE, ctx, ctx.creatives)

        yield _started(Stage.CONFIG)
        ctx.config = self._planner.build(ctx.brief, ctx.assessments, ctx.personas, ctx.creatives, ctx.decision)
        ctx.trace.append(StageMeta(stage=Stage.CONFIG, ms=0, mode=ctx.mode))
        yield _completed(Stage.CONFIG, ctx, ctx.config)

        if self._summary_enabled and ctx.config.status is ConfigStatus.DRAFT:
            yield _started(Stage.SUMMARY)
            run = await self._executor.summarize(self._plan_view(ctx))
            ctx.trace.append(run.meta)
            ctx.summary = run.output
            yield _completed(Stage.SUMMARY, ctx, ctx.summary)

        plan = CampaignPlan(run_id=ctx.run_id, description=ctx.request.description, mode=ctx.mode,
                            brief=ctx.brief, publishers=ctx.assessments, personas=ctx.personas,
                            creatives=ctx.creatives, config=ctx.config, summary=ctx.summary,
                            trace=ctx.trace)
        logger.info("[PIPELINE] run=%s done recommended=%d creatives=%d status=%s", ctx.run_id,
                    len(ctx.recommended), len(ctx.creatives), ctx.config.status)
        yield PipelineEvent(stage=Stage.DONE, status=EventStatus.COMPLETED, data=_dump(plan))

    async def _intake(self, ctx: PipelineContext) -> AdvertiserBrief:
        description = ctx.request.description
        if self._router.is_trivially_insufficient(description):
            ctx.trace.append(StageMeta(stage=Stage.INTAKE, ms=0, mode=ctx.mode))
            return _insufficient_brief(description)
        run = await self._executor.intake(description)
        ctx.trace.append(run.meta)
        return run.output

    async def _personas(self, ctx: PipelineContext) -> PersonaSelection:
        assert ctx.brief and ctx.decision
        candidates = ctx.recommended or ctx.assessments[:3]  # exploratory runs use the least-bad
        run = await self._executor.select_personas(ctx.brief, candidates, ctx.decision.persona_cap)
        ctx.trace.append(run.meta)
        name = lambda persona_id: self._catalog.persona(persona_id).name  # noqa: E731
        valid_ids = {c.publisher_id for c in candidates}
        selected = [PersonaPick(**p.model_dump(exclude={"best_publishers"}), persona_name=name(p.persona_id),
                                best_publishers=[i for i in p.best_publishers if i in valid_ids])
                    for p in run.output.selected if self._catalog.has_persona(p.persona_id)]
        rejected = [RejectedPersona(**r.model_dump(), persona_name=name(r.persona_id))
                    for r in run.output.rejected if self._catalog.has_persona(r.persona_id)]
        return PersonaSelection(selected=selected, rejected=rejected)

    async def _creatives(self, ctx: PipelineContext) -> AsyncIterator[PipelineEvent]:
        assert ctx.personas
        picks = ctx.personas.selected
        tasks = [asyncio.create_task(self._creative(ctx, pick)) for pick in picks]
        finished = 0
        try:
            for future in asyncio.as_completed(tasks):
                variant = await future
                finished += 1
                ctx.creatives.append(variant)
                yield PipelineEvent(stage=Stage.CREATIVE, status=EventStatus.PROGRESS,
                                    completed=finished, total=len(tasks), data=_dump(variant))
        finally:
            for task in tasks:
                task.cancel()
        order = {pick.persona_id: i for i, pick in enumerate(picks)}
        ctx.creatives.sort(key=lambda v: order[v.persona_id])

    async def _creative(self, ctx: PipelineContext, pick: PersonaPick) -> CreativeVariant:
        """One copywriter call, linted; a hard lint failure buys exactly one rewrite."""
        assert ctx.brief
        persona = self._catalog.persona(pick.persona_id)
        targets = pick.best_publishers or [a.publisher_id for a in ctx.recommended[:3]]
        draft_pick = PersonaPickDraft(**pick.model_dump(exclude={"persona_name"}))
        run = await self._executor.write_creative(ctx.brief, draft_pick, persona, targets, feedback=[])
        ctx.trace.append(run.meta)
        issues = self._linter.lint(LintContext(run.output, persona, ctx.request.description))
        retried = not self._linter.passed(issues)
        if retried:
            feedback = [i.message for i in issues if i.severity is LintSeverity.HARD]
            run = await self._executor.write_creative(ctx.brief, draft_pick, persona, targets, feedback)
            ctx.trace.append(run.meta)
            issues = self._linter.lint(LintContext(run.output, persona, ctx.request.description))
        return CreativeVariant(
            **run.output.model_dump(), id=f"creative-{pick.persona_id}", persona_id=pick.persona_id,
            persona_name=persona.name, target_publishers=targets,
            lint=LintReport(passed=self._linter.passed(issues), issues=issues, retried=retried),
        )

    @staticmethod
    def _plan_view(ctx: PipelineContext) -> dict:
        """What the summary stage reads: the decisions, not the whole payload."""
        assert ctx.brief and ctx.config and ctx.personas is not None
        return {
            "business": ctx.brief.business_summary,
            "input_quality": ctx.brief.input_quality,
            "allocation": [a.model_dump(mode="json") for a in ctx.config.publisher_allocation],
            "personas": [{"persona_name": p.persona_name, "angle": p.angle} for p in ctx.personas.selected],
            "creatives": [{"persona": c.persona_name, "headline": c.headline, "lint_passed": c.lint.passed}
                          for c in ctx.creatives],
            "budget": ctx.config.budget.model_dump(mode="json"),
            "bid_strategy": ctx.config.bid_strategy.model_dump(mode="json"),
            "kpi": ctx.config.kpis.primary,
            "open_questions": ctx.config.open_questions,
        }


# ---------------------------------------------------------------------- helpers

def _started(stage: Stage) -> PipelineEvent:
    return PipelineEvent(stage=stage, status=EventStatus.STARTED)


def _completed(stage: Stage, ctx: PipelineContext, data) -> PipelineEvent:
    meta = next((m for m in reversed(ctx.trace) if m.stage is stage), None)
    return PipelineEvent(stage=stage, status=EventStatus.COMPLETED, ms=meta.ms if meta else 0,
                         data=_dump(data))


def _dump(value):
    if isinstance(value, list):
        return [_dump(v) for v in value]
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _insufficient_brief(description: str) -> AdvertiserBrief:
    """Junk input never reaches a model; this is the brief the stop result carries."""
    return AdvertiserBrief(
        business_summary="No usable business description.",
        product_category=ProductCategory.OTHER, secondary_categories=[], price_tier=PriceTier.MID,
        estimated_price_point_usd=None, purchase_model=PurchaseModel.ONE_TIME, brand_attributes=[],
        target_customer=TargetCustomer(age_range=None, gender_skew=GenderSkew.UNKNOWN,
                                       income_tier=None, life_stage=None),
        audience_signals=[], is_consumer_commerce=True, input_quality=InputQuality.INSUFFICIENT,
        confidence=0.0, assumptions=[],
        clarifying_questions=["What do you sell, and who buys it?",
                              "What does a typical order cost?"],
        interpretations=[],
    )
