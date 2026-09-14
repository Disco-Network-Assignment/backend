"""Deterministic fit evidence per publisher: the numbers a model must not "vibe".

For one advertiser and one publisher, code computes:

    category_overlap        same shelf (1.0), adjacent shelf (0.5) or unrelated (0.0)
    age_overlap_pct         how much of the target age range the publisher's audience covers
    gender_alignment        how well the publisher's gender split matches the target
    income_price_alignment  publisher income tier vs the product's price tier
    aov_ratio / aov_fit     product price vs the publisher's average order value
    reach_index             monthly impressions on a 0-1 log scale within the catalog
    prior                   a weighted blend of the above on a 0-100 scale

The matcher agent gets these as evidence (it reads the publisher's free-text notes itself),
the UI shows them next to the agent's reasons, and the guard flags a verdict that wildly
disagrees with the prior.
"""

import math
from dataclasses import dataclass

from app.domain.catalog import CatalogRepository
from app.domain.categories import terms_for
from app.domain.economics import DEFAULT_ECONOMICS, Economics
from app.enums import GenderSkew, IncomeTier, PriceTier
from app.schemas import AdvertiserBrief, FitSignals, Publisher

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# price / AOV band in which post-purchase offers convert best: a $30 add-on after a $100
# basket is easy, a $600 product after a $40 basket is not
AOV_SWEET_SPOT_LOW = 0.3
AOV_SWEET_SPOT_HIGH = 2.5

# neutral values when the brief does not say
UNKNOWN_AGE_OVERLAP = 0.5
UNKNOWN_GENDER_ALIGNMENT = 0.6

INCOME_RANK = {IncomeTier.MID: 1, IncomeTier.MID_HIGH: 2, IncomeTier.HIGH: 3}
PRICE_RANK = {PriceTier.BUDGET: 0, PriceTier.MID: 1, PriceTier.PREMIUM: 2, PriceTier.LUXURY: 3}


@dataclass(frozen=True)
class PriorWeights:
    """How much each signal counts in the blended prior; they add up to 1."""

    category: float = 0.35
    age: float = 0.20
    gender: float = 0.10
    income: float = 0.15
    aov: float = 0.20


# ---------------------------------------------------------------------------
# Individual signals (pure functions, unit-tested on their own)
# ---------------------------------------------------------------------------


def parse_range(text: str | None) -> tuple[int, int] | None:
    """'25-45' (or with an en dash) -> (25, 45); anything else is unknown."""
    low, _, high = (text or "").replace("–", "-").partition("-")
    if low.strip().isdigit() and high.strip().isdigit():
        return int(low), int(high)
    return None


def age_overlap_pct(target_range: str | None, publisher_range: str) -> float:
    """
    Share of the advertiser's target age range covered by the publisher's audience.
    Target 30-60 against a 25-45 audience: 15 of 30 years covered -> 0.5.
    An unknown target is neutral (0.5) rather than a penalty.
    """
    target = parse_range(target_range)
    publisher = parse_range(publisher_range)
    if target is None or publisher is None:
        return UNKNOWN_AGE_OVERLAP

    target_low, target_high = target
    publisher_low, publisher_high = publisher
    overlap_low = max(target_low, publisher_low)
    overlap_high = min(target_high, publisher_high)
    overlap_years = max(0, overlap_high - overlap_low)
    target_years = max(1, target_high - target_low)
    return round(overlap_years / target_years, 2)


def gender_alignment(skew: GenderSkew, female_share: float) -> float:
    """How well the publisher's audience (given as its female share) matches the target."""
    if skew is GenderSkew.FEMALE:
        return round(female_share, 2)
    if skew is GenderSkew.MALE:
        return round(1 - female_share, 2)
    if skew is GenderSkew.BALANCED:
        # 1.0 at a 50/50 split, falling to 0 at 100/0
        return round(1 - abs(2 * female_share - 1), 2)
    return UNKNOWN_GENDER_ALIGNMENT


def income_price_alignment(income: IncomeTier, tier: PriceTier) -> float:
    """1.0 when the tiers line up, minus 0.3 for every tier of distance."""
    distance = abs(INCOME_RANK[income] - PRICE_RANK[tier])
    return round(max(0.0, 1 - 0.3 * distance), 2)


def aov_fit(ratio: float) -> float:
    """1.0 inside the sweet spot, decaying outside it (5x the basket -> 0.5, 10x -> 0.25)."""
    if ratio < AOV_SWEET_SPOT_LOW:
        return round(ratio / AOV_SWEET_SPOT_LOW, 2)
    if ratio > AOV_SWEET_SPOT_HIGH:
        return round(AOV_SWEET_SPOT_HIGH / ratio, 2)
    return 1.0


def category_overlap(brief: AdvertiserBrief, publisher: Publisher) -> float:
    """
    1.0 same shelf, 0.5 adjacent shelf, 0 unrelated. The advertiser's secondary categories
    count less (0.7 / 0.35) and only ever raise the result.
    """
    publisher_terms = set(publisher.subcategories)
    publisher_terms.add(publisher.category)

    def publisher_has_any(terms: tuple[str, ...]) -> bool:
        for term in terms:
            if term in publisher_terms:
                return True
        return False

    primary = terms_for(brief.product_category)
    if publisher_has_any(primary.direct):
        score = 1.0
    elif publisher_has_any(primary.adjacent):
        score = 0.5
    else:
        score = 0.0

    for category in brief.secondary_categories:
        secondary = terms_for(category)
        if publisher_has_any(secondary.direct):
            score = max(score, 0.7)
        elif publisher_has_any(secondary.adjacent):
            score = max(score, 0.35)
    return score


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class SignalCalculator:
    def __init__(self, catalog: CatalogRepository, economics: Economics = DEFAULT_ECONOMICS,
                 weights: PriorWeights = PriorWeights()):
        self.catalog = catalog
        self.economics = economics
        self.weights = weights

    def price_point(self, brief: AdvertiserBrief) -> float:
        """The stated price, or the tier default when the advertiser gave none."""
        return self.economics.price_point(brief.estimated_price_point_usd, brief.price_tier)

    def compute(self, brief: AdvertiserBrief, publisher: Publisher) -> FitSignals:
        # --- price vs the publisher's basket ---
        aov_ratio = self.price_point(brief) / publisher.avg_order_value_usd

        # --- reach: where this publisher sits between the smallest and largest in the catalog ---
        smallest_log, largest_log = self.catalog.log_reach_bounds
        spread = max(largest_log - smallest_log, 1e-9)
        reach = (math.log10(publisher.monthly_impressions) - smallest_log) / spread

        # --- the individual signals ---
        category = category_overlap(brief, publisher)
        age = age_overlap_pct(brief.target_customer.age_range, publisher.audience.age_skew)
        gender = gender_alignment(brief.target_customer.gender_skew, publisher.audience.gender_split.female)
        income = income_price_alignment(publisher.audience.income_tier, brief.price_tier)
        aov = aov_fit(aov_ratio)

        # --- the blended prior on a 0-100 scale ---
        w = self.weights
        prior = 100 * (w.category * category
                       + w.age * age
                       + w.gender * gender
                       + w.income * income
                       + w.aov * aov)

        return FitSignals(
            publisher_id=publisher.id,
            category_overlap=category,
            age_overlap_pct=age,
            gender_alignment=gender,
            income_price_alignment=income,
            aov_ratio=round(aov_ratio, 2),
            aov_fit=aov,
            reach_index=round(reach, 2),
            prior=round(prior),
        )

    def compute_all(self, brief: AdvertiserBrief) -> list[FitSignals]:
        signals = []
        for publisher in self.catalog.publishers:
            signals.append(self.compute(brief, publisher))
        return signals
