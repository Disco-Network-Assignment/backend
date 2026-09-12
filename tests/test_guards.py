import pytest

from app.domain.guards import AssessmentGuard, GuardConfig
from app.domain.signals import SignalCalculator
from app.enums import GuardrailTag, InputQuality, PriceTier, Verdict
from tests.helpers import create_match_output, create_sample_brief


@pytest.fixture
def guard(catalog):
    return AssessmentGuard(catalog)


@pytest.fixture
def signals(catalog):
    def build(brief):
        return {s.publisher_id: s for s in SignalCalculator(catalog).compute_all(brief)}
    return build


def all_ids(catalog):
    return [p.id for p in catalog.publishers]


class TestValidationErrors:
    def test_complete_output_has_no_errors(self, guard, catalog):
        out = create_match_output({pid: 80 for pid in all_ids(catalog)})
        assert guard.validation_errors(out) == []

    def test_many_missing_publishers_is_an_error(self, guard, catalog):
        out = create_match_output({pid: 80 for pid in all_ids(catalog)[:10]})
        assert any("missing" in e for e in guard.validation_errors(out))

    def test_few_missing_publishers_is_tolerated(self, guard, catalog):
        out = create_match_output({pid: 80 for pid in all_ids(catalog)[:18]})
        assert guard.validation_errors(out) == []

    def test_duplicate_ids_is_an_error(self, guard):
        out = create_match_output({"pub_001": 80})
        out.assessments.append(out.assessments[0])
        assert any("duplicate" in e for e in guard.validation_errors(out))

    def test_missing_exclusion_reason_is_an_error(self, guard):
        out = create_match_output({"pub_001": 30}, exclusion_reason=None)
        assert any("exclusion_reason" in e for e in guard.validation_errors(out))


class TestApply:
    def test_missing_publisher_is_filled_and_tagged(self, guard, signals, catalog):
        brief = create_sample_brief()
        out = create_match_output({pid: 80 for pid in all_ids(catalog) if pid != "pub_013"})
        rows = guard.apply(out, signals(brief), brief)
        velvetline = next(r for r in rows if r.publisher_id == "pub_013")
        assert velvetline.verdict is Verdict.EXCLUDE
        assert GuardrailTag.FILLED_MISSING in velvetline.guardrails_applied
        assert len(rows) == 20 and [r.rank for r in rows] == list(range(1, 21))

    def test_off_catalog_caps_scores_and_verdicts(self, guard, signals, catalog):
        brief = create_sample_brief(input_quality=InputQuality.OFF_CATALOG, is_consumer_commerce=False)
        out = create_match_output({pid: 95 for pid in all_ids(catalog)})
        rows = guard.apply(out, signals(brief), brief)
        assert all(r.score <= 40 and r.verdict is not Verdict.RECOMMEND for r in rows)
        assert all(GuardrailTag.OFF_CATALOG_CAP in r.guardrails_applied for r in rows)

    def test_aov_mismatch_cap_spares_high_income_publishers(self, guard, signals, catalog):
        brief = create_sample_brief(price_tier=PriceTier.LUXURY, estimated_price_point_usd=1200.0)
        out = create_match_output({pid: 90 for pid in all_ids(catalog)})
        rows = {r.publisher_id: r for r in guard.apply(out, signals(brief), brief)}
        assert GuardrailTag.AOV_MISMATCH_CAP in rows["pub_001"].guardrails_applied  # Swiftcart, mid
        assert rows["pub_001"].score == 55
        assert GuardrailTag.AOV_MISMATCH_CAP not in rows["pub_005"].guardrails_applied  # Linden Park

    def test_recommend_with_low_score_is_demoted(self, guard, signals, catalog):
        brief = create_sample_brief()
        out = create_match_output({pid: 50 for pid in all_ids(catalog)}, verdict=Verdict.RECOMMEND,
                                  exclusion_reason=None)
        rows = guard.apply(out, signals(brief), brief)
        assert all(r.verdict is not Verdict.RECOMMEND for r in rows)
        assert all(GuardrailTag.VERDICT_SCORE_MISMATCH in r.guardrails_applied for r in rows)

    def test_recommended_set_is_bounded_to_six(self, guard, signals, catalog):
        brief = create_sample_brief()
        out = create_match_output({pid: 90 for pid in all_ids(catalog)})
        rows = guard.apply(out, signals(brief), brief)
        recommended = [r for r in rows if r.verdict is Verdict.RECOMMEND]
        assert len(recommended) == 6
        assert any(GuardrailTag.BEYOND_TOP_N in r.guardrails_applied for r in rows)

    def test_minimum_is_filled_by_promotion(self, guard, signals, catalog):
        brief = create_sample_brief()
        scores = {pid: 30 for pid in all_ids(catalog)}
        scores.update({"pub_007": 90, "pub_009": 65, "pub_018": 62})
        rows = guard.apply(create_match_output(scores), signals(brief), brief)
        recommended = {r.publisher_id for r in rows if r.verdict is Verdict.RECOMMEND}
        assert recommended == {"pub_007", "pub_009", "pub_018"}
        promoted = next(r for r in rows if r.publisher_id == "pub_009")
        assert GuardrailTag.PROMOTED_TO_FILL_MINIMUM in promoted.guardrails_applied

    def test_divergence_from_signals_is_flagged(self, guard, signals, catalog):
        brief = create_sample_brief()
        out = create_match_output({pid: 95 for pid in all_ids(catalog)})
        rows = {r.publisher_id: r for r in guard.apply(out, signals(brief), brief)}
        assert GuardrailTag.DIVERGES_FROM_SIGNALS in rows["pub_013"].guardrails_applied

    def test_custom_config(self, catalog, signals):
        guard = AssessmentGuard(catalog, GuardConfig(max_recommended=2))
        brief = create_sample_brief()
        out = create_match_output({pid: 90 for pid in all_ids(catalog)})
        rows = guard.apply(out, signals(brief), brief)
        assert sum(r.verdict is Verdict.RECOMMEND for r in rows) == 2
