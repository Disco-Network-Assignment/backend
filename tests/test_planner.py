import pytest

from app.domain.guards import AssessmentGuard
from app.domain.planner import CampaignPlanner
from app.domain.router import InputRouter
from app.domain.signals import SignalCalculator
from app.enums import BidModel, ConfigStatus, InputQuality, Objective, PurchaseModel
from tests.helpers import create_match_output, create_sample_brief


@pytest.fixture
def build(catalog):
    guard = AssessmentGuard(catalog)
    calc = SignalCalculator(catalog)
    planner = CampaignPlanner(catalog)

    def _build(brief, scores):
        signals = {s.publisher_id: s for s in calc.compute_all(brief)}
        rows = guard.apply(create_match_output(scores), signals, brief)
        return planner.build(brief, rows, None, [], InputRouter().route(brief))
    return _build


def scores_for(catalog, **top):
    scores = {p.id: 30.0 for p in catalog.publishers}
    scores.update(top)
    return scores


class TestCampaignPlanner:
    def test_draft_allocation_sums_to_100(self, build, catalog):
        config = build(create_sample_brief(), scores_for(catalog, pub_007=94, pub_009=89, pub_018=81))
        assert config.status is ConfigStatus.DRAFT
        assert sum(a.share_pct for a in config.publisher_allocation) == pytest.approx(100)
        assert sum(a.budget_usd for a in config.publisher_allocation) == pytest.approx(config.budget.total_usd)
        assert config.budget.total_usd == 5000 and config.budget.flight_days == 14

    def test_cpa_bidding_when_price_is_stated(self, build, catalog):
        config = build(create_sample_brief(), scores_for(catalog, pub_007=94, pub_009=89, pub_018=81))
        assert config.bid_strategy.model is BidModel.CPA
        assert config.bid_strategy.target_cpa_usd == 42.0  # 60% of $70 for a subscription
        assert config.kpis.primary == "cpa_usd"

    def test_cpm_bidding_without_a_price(self, build, catalog):
        brief = create_sample_brief(estimated_price_point_usd=None)
        config = build(brief, scores_for(catalog, pub_007=94, pub_009=89, pub_018=81))
        assert config.bid_strategy.model is BidModel.CPM
        assert any("No price stated" in a for a in config.assumptions)
        assert any("typical order value" in q for q in config.open_questions)

    def test_nothing_recommended_is_not_recommended(self, build, catalog):
        brief = create_sample_brief(input_quality=InputQuality.OFF_CATALOG, is_consumer_commerce=False)
        config = build(brief, scores_for(catalog))
        assert config.status is ConfigStatus.NOT_RECOMMENDED
        assert config.budget.total_usd == 0 and config.publisher_allocation == []
        assert config.forecast.impressions == 0
        assert any("consumer commerce" in q for q in config.open_questions)

    def test_confidence_scales_the_pilot(self, build, catalog):
        brief = create_sample_brief(input_quality=InputQuality.VAGUE, confidence=0.6)
        config = build(brief, scores_for(catalog, pub_007=94, pub_009=89, pub_018=81))
        assert config.confidence == pytest.approx(0.42)
        assert config.budget.total_usd == 1500

    def test_objective_follows_purchase_model(self, build, catalog):
        gifting = create_sample_brief(purchase_model=PurchaseModel.GIFTING)
        assert build(gifting, scores_for(catalog, pub_007=94)).objective is Objective.SEASONAL_GIFTING
        assert build(create_sample_brief(), scores_for(catalog, pub_007=94)).objective is Objective.ACQUISITION

    def test_targeting_is_derived_from_the_recommended_set(self, build, catalog):
        config = build(create_sample_brief(), scores_for(catalog, pub_007=94, pub_009=89))
        assert config.targeting.geo == ["nationwide"]  # Ruffco is nationwide
        assert "pet" in config.targeting.contextual
        assert set(config.targeting.demographics.income_tiers) == {"mid-high", "mid"}
        assert any(e.startswith("publisher: ") for e in config.targeting.exclusions)

    def test_economics_constants_are_surfaced(self, build, catalog):
        config = build(create_sample_brief(), scores_for(catalog, pub_007=94))
        assert any("CPM bands" in a for a in config.assumptions)
        assert any("Confidence" in a for a in config.assumptions)
