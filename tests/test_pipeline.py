"""End-to-end runs of the pipeline in heuristic mode: routing, guards, personas, parallel
creatives with lint, config and the event protocol, over the sample advertisers."""

import pytest

from app.enums import ConfigStatus, EventStatus, InputQuality, Stage, Verdict
from app.schemas import CampaignPlan, PlanOptions, PlanRequest, StopResult


async def collect(pipeline, description, **options):
    request = PlanRequest(description=description, options=PlanOptions(**options))
    return [event async for event in pipeline.stream(request)]


def example(catalog, number):
    return next(e for e in catalog.examples if e.number == number).description


class TestHappyPath:
    async def test_dog_food_runs_every_stage_in_order(self, pipeline, catalog):
        events = await collect(pipeline, example(catalog, 1))
        completed = [e.stage for e in events if e.status is EventStatus.COMPLETED]
        assert completed == [Stage.INTAKE, Stage.SIGNALS, Stage.MATCH, Stage.PERSONAS,
                             Stage.CREATIVE, Stage.CONFIG, Stage.SUMMARY, Stage.DONE]
        assert events[0].stage is Stage.INTAKE and events[0].status is EventStatus.STARTED
        plan = CampaignPlan.model_validate(events[-1].data)
        recommended = [p.publisher_id for p in plan.recommended]
        assert {"pub_007", "pub_009"} <= set(recommended)
        assert "pub_013" not in recommended
        assert plan.publishers[0].rank == 1 and len(plan.publishers) == 20
        assert all(p.exclusion_reason for p in plan.publishers if p.verdict is not Verdict.RECOMMEND)
        assert plan.personas and plan.personas.selected[0].persona_id == "persona_004"
        assert 3 <= len(plan.creatives) <= 5 and all(c.lint.passed for c in plan.creatives)
        assert plan.config.status is ConfigStatus.DRAFT and plan.summary is not None
        assert {m.stage for m in plan.trace} >= {Stage.INTAKE, Stage.MATCH, Stage.PERSONAS, Stage.CREATIVE}

    async def test_creative_progress_events(self, pipeline, catalog):
        events = await collect(pipeline, example(catalog, 1))
        progress = [e for e in events if e.stage is Stage.CREATIVE and e.status is EventStatus.PROGRESS]
        assert [e.completed for e in progress] == list(range(1, len(progress) + 1))
        assert all(e.total == len(progress) for e in progress)

    async def test_run_wraps_the_stream(self, pipeline, catalog):
        response = await pipeline.run(PlanRequest(description=example(catalog, 12)))
        assert response.status == "done" and response.plan
        assert response.plan.brief.input_quality is InputQuality.CLEAR
        assert {p.publisher_id for p in response.plan.recommended} & {"pub_007", "pub_009", "pub_018"}


class TestMessyInput:
    async def test_junk_stops_before_any_model_call(self, pipeline):
        events = await collect(pipeline, "idk just try it")
        assert [e.stage for e in events][-1] is Stage.STOPPED
        stop = StopResult.model_validate(events[-1].data)
        assert stop.brief.input_quality is InputQuality.INSUFFICIENT
        assert stop.clarifying_questions and len(stop.examples) == 15

    async def test_vague_input_continues_with_a_small_pilot(self, pipeline, catalog):
        response = await pipeline.run(PlanRequest(description=example(catalog, 5)))
        assert response.plan and response.plan.brief.input_quality is InputQuality.VAGUE
        assert response.plan.config.budget.total_usd <= 1500
        assert response.plan.brief.assumptions and response.plan.brief.clarifying_questions

    async def test_ambiguous_input_caps_personas_and_offers_interpretations(self, pipeline, catalog):
        response = await pipeline.run(PlanRequest(description=example(catalog, 8)))
        plan = response.plan
        assert plan and plan.brief.input_quality is InputQuality.AMBIGUOUS
        assert plan.brief.interpretations and plan.personas and len(plan.personas.selected) <= 3

    async def test_off_catalog_ends_not_recommended(self, pipeline, catalog):
        response = await pipeline.run(PlanRequest(description=example(catalog, 7)))
        plan = response.plan
        assert plan and plan.config.status is ConfigStatus.NOT_RECOMMENDED
        assert plan.recommended == [] and plan.personas is None and plan.creatives == []
        assert plan.summary is None and plan.config.budget.total_usd == 0
        assert all(p.score <= 40 for p in plan.publishers)

    async def test_force_exploratory_still_writes_creatives(self, pipeline, catalog):
        response = await pipeline.run(PlanRequest(description=example(catalog, 7),
                                                  options=PlanOptions(force_exploratory=True)))
        assert response.plan and response.plan.personas and response.plan.creatives

    async def test_luxury_never_lands_on_impulse_surfaces(self, pipeline, catalog):
        response = await pipeline.run(PlanRequest(description=example(catalog, 10)))
        recommended = {p.publisher_id for p in response.plan.recommended}
        assert recommended and "pub_001" not in recommended
        assert all(catalog.publisher(pid).audience.income_tier == "high" for pid in recommended)


@pytest.mark.parametrize("number", range(1, 16))
async def test_every_example_ends_with_a_terminal_event(pipeline, catalog, number):
    events = await collect(pipeline, example(catalog, number))
    assert events[-1].stage in (Stage.DONE, Stage.STOPPED)
    assert not any(e.status is EventStatus.FAILED for e in events)
