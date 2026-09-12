from app.domain.signals import (
    SignalCalculator,
    age_overlap_pct,
    aov_fit,
    category_overlap,
    gender_alignment,
    income_price_alignment,
)
from app.enums import GenderSkew, IncomeTier, PriceTier, ProductCategory
from tests.helpers import create_sample_brief


class TestAgeOverlap:
    def test_full_overlap(self):
        assert age_overlap_pct("30-55", "25-55") == 1.0

    def test_partial_overlap(self):
        assert age_overlap_pct("30-55", "18-34") == 0.16

    def test_unknown_target_is_neutral(self):
        assert age_overlap_pct(None, "18-34") == 0.5

    def test_no_overlap(self):
        assert age_overlap_pct("50-70", "18-34") == 0.0


class TestGenderAlignment:
    def test_female_target_uses_female_share(self):
        assert gender_alignment(GenderSkew.FEMALE, 0.9) == 0.9

    def test_balanced_target_prefers_balanced_publisher(self):
        assert gender_alignment(GenderSkew.BALANCED, 0.5) == 1.0
        assert gender_alignment(GenderSkew.BALANCED, 0.9) == 0.2

    def test_unknown_is_mildly_neutral(self):
        assert gender_alignment(GenderSkew.UNKNOWN, 0.1) == 0.6


class TestIncomePriceAlignment:
    def test_exact_match(self):
        assert income_price_alignment(IncomeTier.HIGH, PriceTier.LUXURY) == 1.0

    def test_two_steps_apart(self):
        assert income_price_alignment(IncomeTier.MID, PriceTier.LUXURY) == 0.4


class TestAovFit:
    def test_sweet_spot(self):
        assert aov_fit(1.0) == 1.0

    def test_too_expensive_decays(self):
        assert aov_fit(5.0) == 0.5
        assert aov_fit(10.0) == 0.25

    def test_too_cheap_decays(self):
        assert aov_fit(0.15) == 0.5


class TestCategoryOverlap:
    def test_same_shelf(self, catalog):
        assert category_overlap(create_sample_brief(), catalog.publisher("pub_007")) == 1.0

    def test_unrelated_shelf(self, catalog):
        assert category_overlap(create_sample_brief(), catalog.publisher("pub_013")) == 0.0

    def test_secondary_category_counts_less(self, catalog):
        brief = create_sample_brief(product_category=ProductCategory.BEAUTY_SKINCARE,
                                    secondary_categories=[ProductCategory.PET_FOOD])
        assert category_overlap(brief, catalog.publisher("pub_007")) == 0.7


class TestSignalCalculator:
    def test_dog_food_prior_ranks_pet_publishers_first(self, catalog):
        signals = SignalCalculator(catalog).compute_all(create_sample_brief())
        top = [s.publisher_id for s in sorted(signals, key=lambda s: -s.prior)[:2]]
        assert set(top) == {"pub_007", "pub_009"}

    def test_notes_keyword_hits(self, catalog):
        signals = SignalCalculator(catalog).compute(create_sample_brief(), catalog.publisher("pub_007"))
        assert "subscription" in signals.notes_keyword_hits
        assert "premium" in signals.notes_keyword_hits

    def test_reach_index_bounds(self, catalog):
        calc = SignalCalculator(catalog)
        reach = {s.publisher_id: s.reach_index for s in calc.compute_all(create_sample_brief())}
        assert reach["pub_001"] == 1.0  # 84M impressions, the largest
        assert reach["pub_014"] == 0.0  # 2.8M, the smallest
