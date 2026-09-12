"""Deterministic fit evidence per publisher - the numbers a model must not "vibe".

Category overlap through the taxonomy, demographic overlap, income tier vs price tier, the
post-purchase AOV ratio, reach, and which brand attributes the publisher's notes mention. The
matcher agent gets them as evidence, the UI shows them next to the model's reasons, and the
guard uses the blended prior to flag a verdict that wildly disagrees with the data."""

import math
import re
from dataclasses import dataclass

from app.domain.catalog import CatalogRepository
from app.domain.economics import DEFAULT_ECONOMICS, Economics
from app.domain.taxonomy import terms_for
from app.enums import BrandAttribute, GenderSkew, IncomeTier, PriceTier
from app.schemas import AdvertiserBrief, FitSignals, Publisher

AOV_SWEET_SPOT = (0.3, 2.5)  # price / AOV band in which post-purchase offers convert best

_RANGE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")
_INCOME_RANK = {IncomeTier.MID: 1, IncomeTier.MID_HIGH: 2, IncomeTier.HIGH: 3}
_PRICE_RANK = {PriceTier.BUDGET: 0, PriceTier.MID: 1, PriceTier.PREMIUM: 2, PriceTier.LUXURY: 3}

# brand attribute -> what its evidence looks like in a publisher's free-text notes
NOTE_KEYWORDS: dict[BrandAttribute, re.Pattern[str]] = {
    BrandAttribute.SUSTAINABLE: re.compile(r"sustainab|eco|recycl|refill", re.I),
    BrandAttribute.PREMIUM: re.compile(r"premium|quality|affluent|high aov", re.I),
    BrandAttribute.LUXURY: re.compile(r"affluent|conservative|high aov|quality", re.I),
    BrandAttribute.VALUE: re.compile(r"value|deal|budget|impulse", re.I),
    BrandAttribute.SCIENCE_BACKED: re.compile(r"science|evidence|clinical|skeptical", re.I),
    BrandAttribute.SUBSCRIPTION: re.compile(r"subscription|repeat", re.I),
    BrandAttribute.GIFTING: re.compile(r"gift", re.I),
    BrandAttribute.PLAYFUL: re.compile(r"playful|fun", re.I),
    BrandAttribute.HERITAGE: re.compile(r"heritage|craft|classic|conservative", re.I),
    BrandAttribute.CONVENIENCE: re.compile(r"convenience|impulse|late-night|frequency", re.I),
    BrandAttribute.INCLUSIVE: re.compile(r"inclusive|representation", re.I),
    BrandAttribute.NATURAL_CLEAN: re.compile(r"clean|natural|non-toxic|organic", re.I),
    BrandAttribute.PERSONALIZED: re.compile(r"personaliz", re.I),
    BrandAttribute.PERFORMANCE: re.compile(r"fitness|performance|athlet", re.I),
}


@dataclass(frozen=True)
class PriorWeights:
    category: float = 0.35
    age: float = 0.20
    gender: float = 0.10
    income: float = 0.15
    aov: float = 0.20


def parse_range(text: str | None) -> tuple[int, int] | None:
    match = _RANGE.search(text or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def age_overlap_pct(target_range: str | None, publisher_range: str) -> float:
    """Share of the advertiser's target range covered by the publisher's age skew.
    An unknown target is neutral (0.5) rather than a penalty."""
    target = parse_range(target_range)
    publisher = parse_range(publisher_range)
    if target is None or publisher is None:
        return 0.5
    low, high = max(target[0], publisher[0]), min(target[1], publisher[1])
    return round(max(0, high - low) / max(1, target[1] - target[0]), 2)


def gender_alignment(skew: GenderSkew, female_share: float) -> float:
    if skew is GenderSkew.FEMALE:
        return round(female_share, 2)
    if skew is GenderSkew.MALE:
        return round(1 - female_share, 2)
    if skew is GenderSkew.BALANCED:
        return round(1 - abs(2 * female_share - 1), 2)
    return 0.6  # unknown: mildly neutral


def income_price_alignment(income: IncomeTier, tier: PriceTier) -> float:
    distance = abs(_INCOME_RANK[income] - _PRICE_RANK[tier])
    return round(max(0.0, 1 - 0.3 * distance), 2)


def aov_fit(ratio: float) -> float:
    """1.0 inside the sweet spot, decaying outside it (5x -> 0.5, 10x -> 0.25)."""
    low, high = AOV_SWEET_SPOT
    if ratio < low:
        return round(ratio / low, 2)
    if ratio > high:
        return round(high / ratio, 2)
    return 1.0


def category_overlap(brief: AdvertiserBrief, publisher: Publisher) -> float:
    """1.0 same shelf, 0.5 adjacent shelf, 0 unrelated; secondary categories count less."""
    publisher_terms = {publisher.category, *publisher.subcategories}

    def hit(candidates: tuple[str, ...]) -> bool:
        return any(term in publisher_terms for term in candidates)

    primary = terms_for(brief.product_category)
    score = 1.0 if hit(primary.direct) else 0.5 if hit(primary.adjacent) else 0.0
    for category in brief.secondary_categories:
        secondary = terms_for(category)
        if hit(secondary.direct):
            score = max(score, 0.7)
        elif hit(secondary.adjacent):
            score = max(score, 0.35)
    return score


class SignalCalculator:
    def __init__(self, catalog: CatalogRepository, economics: Economics = DEFAULT_ECONOMICS,
                 weights: PriorWeights = PriorWeights()) -> None:
        self._catalog = catalog
        self._economics = economics
        self._weights = weights

    def price_point(self, brief: AdvertiserBrief) -> float:
        return self._economics.price_point(brief.estimated_price_point_usd, brief.price_tier)

    def compute(self, brief: AdvertiserBrief, publisher: Publisher) -> FitSignals:
        ratio = self.price_point(brief) / publisher.avg_order_value_usd
        low_log, high_log = self._catalog.log_reach_bounds
        reach = (math.log10(publisher.monthly_impressions) - low_log) / max(high_log - low_log, 1e-9)
        category = category_overlap(brief, publisher)
        age = age_overlap_pct(brief.target_customer.age_range, publisher.audience.age_skew)
        gender = gender_alignment(brief.target_customer.gender_skew,
                                  publisher.audience.gender_split.female)
        income = income_price_alignment(publisher.audience.income_tier, brief.price_tier)
        aov = aov_fit(ratio)
        w = self._weights
        prior = 100 * (w.category * category + w.age * age + w.gender * gender
                       + w.income * income + w.aov * aov)
        return FitSignals(
            publisher_id=publisher.id,
            category_overlap=category,
            age_overlap_pct=age,
            gender_alignment=gender,
            income_price_alignment=income,
            aov_ratio=round(ratio, 2),
            aov_fit=aov,
            reach_index=round(reach, 2),
            notes_keyword_hits=[attr.value for attr in brief.brand_attributes
                                if NOTE_KEYWORDS[attr].search(publisher.notes)],
            prior=round(prior),
        )

    def compute_all(self, brief: AdvertiserBrief) -> list[FitSignals]:
        return [self.compute(brief, publisher) for publisher in self._catalog.publishers]
