"""The heuristic executor is the keyless fallback and the test double for the LLM stages, so
its readings of the 15 sample advertisers must match the assignment's intent."""

import pytest

from app.agents.heuristic import HeuristicStageExecutor
from app.domain.lint import BODY_MAX, CTA_MAX, HEADLINE_MAX
from app.domain.signals import SignalCalculator
from app.enums import InputQuality, PriceTier, ProductCategory, PurchaseModel
from app.schemas import PersonaPickDraft
from tests.helpers import SENIOR_DOG_FOOD, create_sample_brief


@pytest.fixture
def executor(catalog):
    return HeuristicStageExecutor(catalog, SignalCalculator(catalog))


EXPECTED = {
    1: (InputQuality.CLEAR, ProductCategory.PET_FOOD),
    2: (InputQuality.CLEAR, ProductCategory.ACTIVEWEAR),
    3: (InputQuality.CLEAR, ProductCategory.FUNCTIONAL_BEVERAGES),
    4: (InputQuality.CLEAR, ProductCategory.HOME_DECOR_CANDLES),
    5: (InputQuality.VAGUE, ProductCategory.WELLNESS_SERVICES),
    6: (InputQuality.OFF_CATALOG, ProductCategory.OUTDOOR_GEAR),
    7: (InputQuality.OFF_CATALOG, ProductCategory.B2B_SOFTWARE),
    8: (InputQuality.AMBIGUOUS, ProductCategory.KIDS_BABY),
    9: (InputQuality.CLEAR, ProductCategory.HOUSEHOLD_CLEANING),
    10: (InputQuality.CLEAR, ProductCategory.LUXURY_ACCESSORIES),
    11: (InputQuality.CLEAR, ProductCategory.SNACKS_PROTEIN),
    12: (InputQuality.CLEAR, ProductCategory.PET_SUPPLIES),
    13: (InputQuality.CLEAR, ProductCategory.SUPPLEMENTS_VITAMINS),
    14: (InputQuality.CLEAR, ProductCategory.HOME_TEXTILES_BEDDING),
    15: (InputQuality.INSUFFICIENT, ProductCategory.OTHER),
}


class TestIntake:
    @pytest.mark.parametrize("number", sorted(EXPECTED))
    async def test_reads_each_example(self, executor, catalog, number):
        example = next(e for e in catalog.examples if e.number == number)
        brief = (await executor.intake(example.description)).output
        quality, category = EXPECTED[number]
        assert brief.input_quality is quality
        assert brief.product_category is category

    async def test_price_and_purchase_model(self, executor):
        brief = (await executor.intake(SENIOR_DOG_FOOD)).output
        assert brief.purchase_model is PurchaseModel.SUBSCRIPTION
        assert brief.price_tier is PriceTier.PREMIUM
        handbags = (await executor.intake("Custom-fit leather handbags, Italian-made. Average price point $1,200.")).output
        assert handbags.estimated_price_point_usd == 1200 and handbags.price_tier is PriceTier.LUXURY

    async def test_ambiguous_input_offers_interpretations(self, executor):
        brief = (await executor.intake("A new kind of thing for moms.")).output
        assert len(brief.interpretations) == 3 and brief.clarifying_questions

    async def test_b2b_is_not_consumer_commerce(self, executor):
        brief = (await executor.intake("B2B SaaS for dental practices.")).output
        assert not brief.is_consumer_commerce and brief.purchase_model is PurchaseModel.B2B


class TestMatch:
    async def test_every_publisher_is_assessed_with_reasons(self, executor, catalog):
        brief = create_sample_brief()
        signals = SignalCalculator(catalog).compute_all(brief)
        out = (await executor.match(brief, signals)).output
        assert {a.publisher_id for a in out.assessments} == {p.id for p in catalog.publishers}
        assert all(a.reasons for a in out.assessments)
        assert all(a.exclusion_reason for a in out.assessments if a.verdict != "recommend")

    async def test_pet_publishers_lead_for_dog_food(self, executor, catalog):
        brief = create_sample_brief()
        out = (await executor.match(brief, SignalCalculator(catalog).compute_all(brief))).output
        best = sorted(out.assessments, key=lambda a: -a.score)[:2]
        assert {a.publisher_id for a in best} == {"pub_007", "pub_009"}


class TestPersonas:
    async def test_pet_parent_leads_and_gifter_is_out_for_a_subscription(self, executor):
        out = (await executor.select_personas(create_sample_brief(), [], persona_cap=5)).output
        assert out.selected[0].persona_id == "persona_004"
        assert "persona_010" not in {p.persona_id for p in out.selected}
        assert 3 <= len(out.selected) <= 5
        assert {p.persona_id for p in out.selected} | {r.persona_id for r in out.rejected} == {
            f"persona_{i:03d}" for i in range(1, 11)}

    async def test_persona_cap_is_respected(self, executor):
        out = (await executor.select_personas(create_sample_brief(), [], persona_cap=3)).output
        assert len(out.selected) <= 3


class TestCreative:
    async def test_lengths_are_within_the_unit(self, executor, catalog):
        persona = catalog.persona("persona_004")
        pick = PersonaPickDraft(persona_id=persona.id, fit_score=90, why_plausible="w",
                                angle="Lead with vet-recommended", watchouts=[], best_publishers=[])
        draft = (await executor.write_creative(create_sample_brief(), pick, persona, ["pub_007"], [])).output
        assert len(draft.headline) <= HEADLINE_MAX and len(draft.body) <= BODY_MAX
        assert len(draft.cta) <= CTA_MAX and draft.persona_reasoning

    async def test_feedback_changes_the_copy(self, executor, catalog):
        persona = catalog.persona("persona_004")
        pick = PersonaPickDraft(persona_id=persona.id, fit_score=90, why_plausible="w",
                                angle="a", watchouts=[], best_publishers=[])
        first = (await executor.write_creative(create_sample_brief(), pick, persona, [], [])).output
        second = (await executor.write_creative(create_sample_brief(), pick, persona, [], ["too long"])).output
        assert first.headline != second.headline and "rewritten after review" in second.persona_reasoning
