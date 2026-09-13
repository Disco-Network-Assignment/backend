"""The pure domain rules: fit signals, guardrails, allocation, economics, lint, routing, the
planner, the prompt registry, and the heuristic intake over the 15 sample advertisers."""

import pytest

from app.agents.heuristic_stages import HeuristicStageExecutor
from app.domain.budget_split import AllocationCandidate, BudgetAllocator
from app.domain.config_builder import ConfigBuilder
from app.domain.creative_checks import CreativeLinter, LintContext
from app.domain.economics import DEFAULT_ECONOMICS
from app.domain.fit_signals import SignalCalculator, age_overlap_pct, aov_fit, category_overlap
from app.domain.guardrails import AssessmentGuard
from app.domain.input_policy import InputPolicy
from app.enums import (
    BidModel,
    ConfigStatus,
    GuardrailTag,
    IncomeTier,
    InputQuality,
    LintSeverity,
    PriceTier,
    ProductCategory,
    PurchaseModel,
    Verdict,
)
from app.prompts.loader import PromptError
from app.schemas import ClarificationRequest
from tests.helpers import create_match_output, create_sample_brief, create_sample_creative


def all_ids(catalog):
    return [p.id for p in catalog.publishers]


def signals_for(catalog, brief):
    return {s.publisher_id: s for s in SignalCalculator(catalog).compute_all(brief)}


# ---------------------------------------------------------------- signals

class TestSignals:
    @pytest.mark.parametrize("target, publisher, expected", [("30-55", "25-55", 1.0), ("30-55", "18-34", 0.16),
                                                             (None, "18-34", 0.5), ("50-70", "18-34", 0.0)])
    def test_age_overlap(self, target, publisher, expected):
        assert age_overlap_pct(target, publisher) == expected

    def test_aov_fit_decays_outside_the_sweet_spot(self):
        assert aov_fit(1.0) == 1.0 and aov_fit(5.0) == 0.5 and aov_fit(0.15) == 0.5

    def test_category_overlap(self, catalog):
        assert category_overlap(create_sample_brief(), catalog.publisher("pub_007")) == 1.0  # Pawline
        assert category_overlap(create_sample_brief(), catalog.publisher("pub_013")) == 0.0  # Velvetline

    def test_prior_ranks_pet_publishers_first_for_dog_food(self, catalog):
        signals = SignalCalculator(catalog).compute_all(create_sample_brief())
        top = {s.publisher_id for s in sorted(signals, key=lambda s: -s.prior)[:2]}
        assert top == {"pub_007", "pub_009"}


# ---------------------------------------------------------------- guards

class TestGuards:
    def test_missing_publisher_is_filled_and_tagged(self, catalog):
        brief = create_sample_brief()
        out = create_match_output({pid: 80 for pid in all_ids(catalog) if pid != "pub_013"})
        rows = AssessmentGuard(catalog).apply(out, signals_for(catalog, brief), brief)
        velvetline = next(r for r in rows if r.publisher_id == "pub_013")
        assert velvetline.verdict is Verdict.EXCLUDE
        assert GuardrailTag.FILLED_MISSING in velvetline.guardrails_applied
        assert [r.rank for r in rows] == list(range(1, 21))

    def test_too_many_missing_publishers_is_a_validation_error(self, catalog):
        out = create_match_output({pid: 80 for pid in all_ids(catalog)[:10]})
        assert any("missing" in e for e in AssessmentGuard(catalog).validation_errors(out))

    def test_off_catalog_caps_scores_and_verdicts(self, catalog):
        brief = create_sample_brief(input_quality=InputQuality.OFF_CATALOG, is_consumer_commerce=False)
        out = create_match_output({pid: 95 for pid in all_ids(catalog)})
        rows = AssessmentGuard(catalog).apply(out, signals_for(catalog, brief), brief)
        assert all(r.score <= 40 and r.verdict is not Verdict.RECOMMEND for r in rows)

    def test_price_mismatch_cap_spares_high_income_publishers(self, catalog):
        brief = create_sample_brief(price_tier=PriceTier.LUXURY, estimated_price_point_usd=1200.0)
        out = create_match_output({pid: 90 for pid in all_ids(catalog)})
        rows = {r.publisher_id: r for r in AssessmentGuard(catalog).apply(out, signals_for(catalog, brief), brief)}
        assert rows["pub_001"].score == 55  # Swiftcart, mid income
        assert GuardrailTag.AOV_MISMATCH_CAP not in rows["pub_005"].guardrails_applied  # Linden Park

    def test_recommended_set_is_bounded_and_filled(self, catalog):
        brief = create_sample_brief()
        guard = AssessmentGuard(catalog)
        many = guard.apply(create_match_output({pid: 90 for pid in all_ids(catalog)}), signals_for(catalog, brief), brief)
        assert sum(r.verdict is Verdict.RECOMMEND for r in many) == 6
        scores = {pid: 30.0 for pid in all_ids(catalog)}
        scores.update({"pub_007": 90, "pub_009": 65, "pub_018": 62})
        few = guard.apply(create_match_output(scores), signals_for(catalog, brief), brief)
        assert {r.publisher_id for r in few if r.verdict is Verdict.RECOMMEND} == {"pub_007", "pub_009", "pub_018"}


# ---------------------------------------------------------------- allocation + economics

class TestAllocation:
    def test_shares_sum_to_one_in_five_percent_steps(self):
        shares = BudgetAllocator().allocate([AllocationCandidate("a", 94, 4_800_000),
                                             AllocationCandidate("b", 89, 62_000_000),
                                             AllocationCandidate("c", 81, 8_400_000)])
        assert sum(s.share for s in shares) == pytest.approx(1.0)
        assert all(round(s.share / 0.05) * 0.05 == pytest.approx(s.share) for s in shares)

    def test_floor_and_cap_hold(self):
        shares = BudgetAllocator().allocate([AllocationCandidate("a", 99, 80_000_000)]
                                            + [AllocationCandidate(str(i), 60, 3_000_000) for i in range(4)])
        assert max(s.share for s in shares) <= 0.40 + 1e-9 and min(s.share for s in shares) >= 0.10 - 1e-9

    def test_single_publisher_gets_everything(self):
        assert BudgetAllocator().allocate([AllocationCandidate("a", 80, 1_000_000)])[0].share == 1.0


class TestEconomics:
    def test_bands_pilot_and_cpa(self):
        assert DEFAULT_ECONOMICS.cpm_band(IncomeTier.HIGH, 90) == (20.7, 32.2)
        assert DEFAULT_ECONOMICS.pilot_for(0.9).total_usd == 5000 and DEFAULT_ECONOMICS.pilot_for(0.1).flight_days == 7
        assert DEFAULT_ECONOMICS.target_cpa(100, PurchaseModel.SUBSCRIPTION) == 60
        assert DEFAULT_ECONOMICS.target_cpa(100, PurchaseModel.B2B) is None

    def test_forecast_maths(self):
        forecast = DEFAULT_ECONOMICS.forecast(5000, 12.5)
        assert (forecast.impressions, forecast.clicks, forecast.conversions) == (400_000, 3600, 108)


# ---------------------------------------------------------------- lint

class TestLint:
    def issues(self, catalog, persona_id="persona_004", description="vet-formulated", **draft):
        ctx = LintContext(create_sample_creative(**draft), catalog.persona(persona_id), description)
        return CreativeLinter().lint(ctx)

    def test_clean_draft_passes(self, catalog):
        assert self.issues(catalog) == []

    def test_length_and_unstated_claims_are_hard(self, catalog):
        issues = self.issues(catalog, headline="x" * 61, body="Clinically proven to cure joint pain.")
        assert {i.rule for i in issues} == {"length", "claims"}
        assert all(i.severity is LintSeverity.HARD for i in issues)

    def test_claim_stated_by_the_advertiser_is_allowed(self, catalog):
        assert self.issues(catalog, description="clinically proven", body="Clinically proven nutrition.") == []

    def test_persona_disinterest_trips(self, catalog):
        issues = self.issues(catalog, persona_id="persona_005", headline="The viral bag everyone is obsessed with")
        assert any("trendy language" in i.message for i in issues)
        gifter = self.issues(catalog, persona_id="persona_010", cta="Subscribe now")
        assert any("subscription-only" in i.message for i in gifter)

    def test_shouting_is_only_a_warning(self, catalog):
        issues = self.issues(catalog, headline="HUGE DEAL TODAY!! Really!")
        assert issues and CreativeLinter.passed(issues)


# ---------------------------------------------------------------- router + planner

class TestRouter:
    def test_junk_and_insufficient_stop(self):
        router = InputPolicy()
        assert router.is_trivially_insufficient("idk") and not router.is_trivially_insufficient("We sell dog food")
        assert router.route(create_sample_brief(input_quality=InputQuality.INSUFFICIENT)).stop

    def test_vague_continues_with_caveats(self):
        decision = InputPolicy().route(create_sample_brief(input_quality=InputQuality.VAGUE))
        assert not decision.stop and decision.persona_cap == 3 and decision.confidence_multiplier == 0.7


class TestPlanner:
    def build(self, catalog, brief, top):
        scores = {p.id: 30.0 for p in catalog.publishers}
        scores.update(top)
        rows = AssessmentGuard(catalog).apply(create_match_output(scores), signals_for(catalog, brief), brief)
        return ConfigBuilder(catalog).build(brief, rows, None, [], InputPolicy().route(brief))

    def test_draft_allocation_and_cpa_bidding(self, catalog):
        config = self.build(catalog, create_sample_brief(), {"pub_007": 94, "pub_009": 89, "pub_018": 81})
        assert config.status is ConfigStatus.DRAFT
        assert sum(a.share_pct for a in config.publisher_allocation) == pytest.approx(100)
        assert config.bid_strategy.model is BidModel.CPA and config.bid_strategy.target_cpa_usd == 42.0
        assert any("CPM bands" in a for a in config.assumptions)

    def test_nothing_recommended_is_not_recommended(self, catalog):
        brief = create_sample_brief(input_quality=InputQuality.OFF_CATALOG, is_consumer_commerce=False)
        config = self.build(catalog, brief, {})
        assert config.status is ConfigStatus.NOT_RECOMMENDED and config.budget.total_usd == 0


# ---------------------------------------------------------------- prompts

PROMPT_VARIABLES = {
    "triage": {"description": "We sell dog food."},
    "intake": {"categories": "a", "attributes": "b"},
    "clarify": {},
    "match_publishers": {"catalog": [], "brief": {}, "publisher_count": "20"},
    "select_personas": {"personas": [], "persona_cap": "5", "brief": {}, "recommended": []},
    "write_creative": {"brief": {}, "persona": {}, "angle": "a", "watchouts": "w", "target_publishers": "p"},
    "campaign_summary": {"plan": {}},
    "validation_retry": {"errors": "- missing"},
}


class TestPrompts:
    @pytest.mark.parametrize("name", sorted(PROMPT_VARIABLES))
    def test_every_prompt_renders(self, prompts, name):
        rendered = prompts.render(name, **PROMPT_VARIABLES[name])
        assert "{{" not in rendered.input and "{{" not in (rendered.instructions or "")

    def test_missing_variable_raises(self, prompts):
        with pytest.raises(PromptError):
            prompts.render("intake", categories="a")


# ---------------------------------------------------------------- heuristic intake

EXPECTED_READING = {
    1: (InputQuality.CLEAR, ProductCategory.PET_FOOD), 2: (InputQuality.CLEAR, ProductCategory.ACTIVEWEAR),
    3: (InputQuality.CLEAR, ProductCategory.FUNCTIONAL_BEVERAGES), 4: (InputQuality.CLEAR, ProductCategory.HOME_DECOR_CANDLES),
    5: (InputQuality.VAGUE, ProductCategory.WELLNESS_SERVICES), 6: (InputQuality.OFF_CATALOG, ProductCategory.OUTDOOR_GEAR),
    7: (InputQuality.OFF_CATALOG, ProductCategory.B2B_SOFTWARE), 8: (InputQuality.AMBIGUOUS, ProductCategory.KIDS_BABY),
    9: (InputQuality.CLEAR, ProductCategory.HOUSEHOLD_CLEANING), 10: (InputQuality.CLEAR, ProductCategory.LUXURY_ACCESSORIES),
    11: (InputQuality.CLEAR, ProductCategory.SNACKS_PROTEIN), 12: (InputQuality.CLEAR, ProductCategory.PET_SUPPLIES),
    13: (InputQuality.CLEAR, ProductCategory.SUPPLEMENTS_VITAMINS), 14: (InputQuality.CLEAR, ProductCategory.HOME_TEXTILES_BEDDING),
    15: (InputQuality.INSUFFICIENT, ProductCategory.OTHER),
}


@pytest.mark.parametrize("number", sorted(EXPECTED_READING))
async def test_heuristic_intake_reads_each_sample_advertiser(catalog, number):
    executor = HeuristicStageExecutor(catalog, SignalCalculator(catalog), CreativeLinter())
    example = next(e for e in catalog.examples if e.number == number)
    output = (await executor.intake(executor.new_context(example.description), None)).output
    quality, category = EXPECTED_READING[number]
    if quality is InputQuality.INSUFFICIENT:
        assert isinstance(output, ClarificationRequest) and output.questions
    else:
        assert (output.input_quality, output.product_category) == (quality, category)
