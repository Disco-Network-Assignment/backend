"""End-to-end runs of the pipeline with a scripted executor standing in for the agents: routing,
guards, personas, parallel creatives with lint, config and the event protocol."""

from app.domain.config_builder import ConfigBuilder
from app.domain.fit_signals import SignalCalculator
from app.domain.guardrails import AssessmentGuard
from app.domain.input_policy import InputPolicy
from app.enums import (
    ConfigStatus,
    EventStatus,
    IncomeTier,
    InputQuality,
    PriceTier,
    ProductCategory,
    Stage,
    Verdict,
)
from app.pipeline import CampaignPipeline
from app.schemas import (
    CampaignPlan,
    ClarificationRequest,
    Interpretation,
    PlanOptions,
    PlanRequest,
    StopResult,
    TargetCustomer,
)
from tests.helpers import SENIOR_DOG_FOOD, ScriptedExecutor, create_sample_brief


def make_pipeline(catalog, executor: ScriptedExecutor) -> CampaignPipeline:
    return CampaignPipeline(executor, catalog, SignalCalculator(catalog), AssessmentGuard(catalog), InputPolicy(),
                            ConfigBuilder(catalog))


async def collect(pipeline, description=SENIOR_DOG_FOOD, **options):
    request = PlanRequest(description=description, options=PlanOptions(**options))
    return [event async for event in pipeline.stream(request)]


class TestHappyPath:
    async def test_dog_food_runs_every_stage_in_order(self, pipeline, executor):
        events = await collect(pipeline)
        completed = [e.stage for e in events if e.status is EventStatus.COMPLETED]
        assert completed == [Stage.INTAKE, Stage.SIGNALS, Stage.MATCH, Stage.PERSONAS,
                             Stage.CREATIVE, Stage.CONFIG, Stage.SUMMARY, Stage.DONE]
        assert executor.calls[:3] == [Stage.INTAKE, Stage.MATCH, Stage.PERSONAS]
        plan = CampaignPlan.model_validate(events[-1].data)
        recommended = [p.publisher_id for p in plan.recommended]
        assert {"pub_007", "pub_009"} <= set(recommended) and "pub_013" not in recommended
        assert plan.publishers[0].rank == 1 and len(plan.publishers) == 20
        assert all(p.exclusion_reason for p in plan.publishers if p.verdict is not Verdict.RECOMMEND)
        assert plan.personas and plan.personas.selected[0].persona_id == "persona_004"
        assert len(plan.creatives) == 3 and all(c.lint.passed for c in plan.creatives)
        assert plan.config.status is ConfigStatus.DRAFT and plan.summary is not None
        assert [m.agent for m in plan.trace if m.stage is Stage.INTAKE] == ["brief_writer"]
        assert sum(m.tool_calls for m in plan.trace) >= 4 and plan.trace[0].handoffs == 1

    async def test_creative_progress_events(self, pipeline):
        events = await collect(pipeline)
        progress = [e for e in events if e.stage is Stage.CREATIVE and e.status is EventStatus.PROGRESS]
        assert [e.completed for e in progress] == [1, 2, 3] and all(e.total == 3 for e in progress)

    async def test_run_wraps_the_stream_and_passes_the_session(self, pipeline, executor):
        response = await pipeline.run(PlanRequest(description=SENIOR_DOG_FOOD,
                                                  options=PlanOptions(session_id="abc")))
        assert response.status == "done" and response.plan
        assert response.plan.brief.input_quality is InputQuality.CLEAR and executor.session_id == "abc"


class TestMessyInput:
    async def test_clarifier_handoff_stops_the_run(self, catalog):
        clarification = ClarificationRequest(reason="Nothing says what is sold.", questions=["What do you sell?"])
        pipeline = make_pipeline(catalog, ScriptedExecutor(catalog, clarification=clarification))
        events = await collect(pipeline, "idk just try it")
        assert [e.stage for e in events] == [Stage.INTAKE, Stage.STOPPED]
        stop = StopResult.model_validate(events[-1].data)
        assert stop.reason == clarification.reason and stop.clarifying_questions == ["What do you sell?"]
        assert len(stop.examples) == 15 and stop.trace[0].agent == "clarifier"

    async def test_brief_writer_flagging_insufficient_also_stops(self, catalog):
        brief = create_sample_brief(input_quality=InputQuality.INSUFFICIENT, clarifying_questions=["Who buys it?"])
        pipeline = make_pipeline(catalog, ScriptedExecutor(catalog, brief=brief))
        response = await pipeline.run(PlanRequest(description="something"))
        assert response.status == "stopped" and response.stopped
        assert response.stopped.clarifying_questions == ["Who buys it?"]

    async def test_vague_input_continues_with_a_small_pilot(self, catalog):
        brief = create_sample_brief(input_quality=InputQuality.VAGUE, confidence=0.5,
                                    assumptions=["assumed dog food"], clarifying_questions=["Which animals?"])
        response = await make_pipeline(catalog, ScriptedExecutor(catalog, brief=brief)).run(
            PlanRequest(description="pet stuff"))
        assert response.plan and response.plan.config.budget.total_usd <= 1500
        assert response.plan.brief.assumptions and response.plan.brief.clarifying_questions

    async def test_ambiguous_input_caps_personas(self, catalog):
        brief = create_sample_brief(input_quality=InputQuality.AMBIGUOUS,
                                    interpretations=[Interpretation(label="Dog food", brief_patch="Sold for dogs.")])
        executor = ScriptedExecutor(catalog, brief=brief, picks=("persona_004", "persona_001", "persona_002", "persona_003"))
        response = await make_pipeline(catalog, executor).run(PlanRequest(description="food for pets or people"))
        plan = response.plan
        assert plan and plan.brief.interpretations and plan.personas and len(plan.personas.selected) == 3

    async def test_off_catalog_ends_not_recommended(self, catalog):
        pipeline = make_pipeline(catalog, ScriptedExecutor(catalog, brief=off_catalog_brief()))
        response = await pipeline.run(PlanRequest(description="B2B SaaS for dental practices"))
        plan = response.plan
        assert plan and plan.config.status is ConfigStatus.NOT_RECOMMENDED
        assert plan.recommended == [] and plan.personas is None and plan.creatives == []
        assert plan.summary is None and plan.config.budget.total_usd == 0
        assert all(p.score <= 40 for p in plan.publishers)

    async def test_force_exploratory_still_writes_creatives(self, catalog):
        pipeline = make_pipeline(catalog, ScriptedExecutor(catalog, brief=off_catalog_brief()))
        response = await pipeline.run(PlanRequest(description="B2B SaaS for dental practices",
                                                  options=PlanOptions(force_exploratory=True)))
        assert response.plan and response.plan.personas and response.plan.creatives

    async def test_luxury_never_lands_on_impulse_surfaces(self, catalog):
        brief = create_sample_brief(
            business_summary="Handcrafted Italian leather handbags.", product_category=ProductCategory.LUXURY_ACCESSORIES,
            secondary_categories=[], price_tier=PriceTier.LUXURY, estimated_price_point_usd=1800,
            target_customer=TargetCustomer(age_range="30-55", gender_skew="female", income_tier=IncomeTier.HIGH,
                                           life_stage=None))
        response = await make_pipeline(catalog, ScriptedExecutor(catalog, brief=brief)).run(
            PlanRequest(description="luxury handbags"))
        recommended = {p.publisher_id for p in response.plan.recommended}
        assert recommended and "pub_001" not in recommended
        assert all(catalog.publisher(pid).audience.income_tier == "high" for pid in recommended)


def off_catalog_brief():
    return create_sample_brief(business_summary="B2B SaaS for dental practices.",
                               product_category=ProductCategory.B2B_SOFTWARE, secondary_categories=[],
                               is_consumer_commerce=False, input_quality=InputQuality.OFF_CATALOG, confidence=0.4)
