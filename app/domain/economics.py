"""Every money constant in one place, with the reasoning next to it.

All of these are heuristics - the exercise ships no auction or performance data - which is why
the planner copies the ones it used into `config.assumptions`: a draft that hides its
assumptions is not reviewable. Changing a number here changes the whole plan consistently."""

from dataclasses import dataclass, field

from app.enums import IncomeTier, PriceTier, PurchaseModel
from app.schemas import Forecast


@dataclass(frozen=True)
class PilotBudget:
    min_confidence: float
    total_usd: float
    flight_days: int


@dataclass(frozen=True)
class Economics:
    # starting CPM bands by the publisher's income tier (USD per 1,000 impressions)
    cpm_band_usd: dict[IncomeTier, tuple[float, float]] = field(default_factory=lambda: {
        IncomeTier.MID: (8.0, 12.0),
        IncomeTier.MID_HIGH: (12.0, 18.0),
        IncomeTier.HIGH: (18.0, 28.0),
    })
    # pilot size scales with how much of the brief is stated rather than assumed
    pilot_tiers: tuple[PilotBudget, ...] = (
        PilotBudget(min_confidence=0.75, total_usd=5000.0, flight_days=14),
        PilotBudget(min_confidence=0.5, total_usd=3000.0, flight_days=10),
        PilotBudget(min_confidence=0.0, total_usd=1500.0, flight_days=7),
    )
    # post-purchase offer units: shoppers are in buying mode, so CTR runs above display
    assumed_ctr: float = 0.009
    assumed_cvr: float = 0.03
    # allowable acquisition cost as a share of the price; subscriptions may pay more per
    # first order because the value keeps coming
    target_cpa_share: dict[PurchaseModel, float] = field(default_factory=lambda: {
        PurchaseModel.ONE_TIME: 0.30,
        PurchaseModel.SUBSCRIPTION: 0.60,
        PurchaseModel.GIFTING: 0.30,
        PurchaseModel.B2B: 0.0,
    })
    # used only when the advertiser states no price
    price_by_tier: dict[PriceTier, float] = field(default_factory=lambda: {
        PriceTier.BUDGET: 15.0,
        PriceTier.MID: 45.0,
        PriceTier.PREMIUM: 120.0,
        PriceTier.LUXURY: 600.0,
    })
    frequency_cap_impressions: int = 3
    frequency_cap_days: int = 7
    optimize_after_impressions: int = 50_000

    def fit_multiplier(self, score: float) -> float:
        """Bid up a little for an obvious home, down a little for a stretch."""
        if score >= 85:
            return 1.15
        if score >= 70:
            return 1.0
        return 0.85

    def cpm_band(self, income_tier: IncomeTier, score: float) -> tuple[float, float]:
        low, high = self.cpm_band_usd[income_tier]
        m = self.fit_multiplier(score)
        return (round(low * m, 2), round(high * m, 2))

    def pilot_for(self, confidence: float) -> PilotBudget:
        for tier in self.pilot_tiers:
            if confidence >= tier.min_confidence:
                return tier
        return self.pilot_tiers[-1]

    def price_point(self, stated_price: float | None, tier: PriceTier) -> float:
        return stated_price if stated_price is not None else self.price_by_tier[tier]

    def target_cpa(self, price: float, purchase_model: PurchaseModel) -> float | None:
        share = self.target_cpa_share[purchase_model]
        return round(price * share, 2) if share > 0 else None

    def max_cpc(self, target_cpa: float | None) -> float | None:
        return round(target_cpa * self.assumed_cvr, 2) if target_cpa is not None else None

    def forecast(self, budget_usd: float, cpm_usd: float) -> Forecast:
        if budget_usd <= 0 or cpm_usd <= 0:
            return Forecast(impressions=0, clicks=0, conversions=0, cpa_usd=None)
        impressions = round(budget_usd / cpm_usd * 1000)
        clicks = round(impressions * self.assumed_ctr)
        conversions = round(clicks * self.assumed_cvr)
        cpa = round(budget_usd / conversions, 2) if conversions else None
        return Forecast(impressions=impressions, clicks=clicks, conversions=conversions,
                        cpa_usd=cpa)

    def describe(self) -> list[str]:
        """The constants in words, for `config.assumptions`."""
        bands = ", ".join(f"{tier}: ${lo:.0f}-{hi:.0f}" for tier, (lo, hi) in self.cpm_band_usd.items())
        return [
            f"Starting CPM bands by publisher income tier ({bands}), x1.15 for scores >= 85 and "
            f"x0.85 for scores under 70.",
            f"Forecast assumes {self.assumed_ctr:.1%} CTR and {self.assumed_cvr:.0%} click-to-conversion "
            f"for post-purchase offer units.",
            "Target CPA is 30% of the price for one-time purchases and 60% for subscriptions "
            "(first order is worth more than its price).",
            "Pilot budget scales with confidence: $5,000/14 days when the brief is mostly stated, "
            "$3,000/10 days when partly assumed, $1,500/7 days when mostly assumed.",
        ]


DEFAULT_ECONOMICS = Economics()
