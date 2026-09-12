"""Assembles the campaign config from everything upstream. Deterministic on purpose: every
number in the config can be traced to a score, a catalog field, or a constant in economics.py,
and the constants used are echoed into `assumptions` so the reviewer sees them."""

from app.domain.allocation import AllocationCandidate, BudgetAllocator
from app.domain.catalog import CatalogRepository
from app.domain.economics import DEFAULT_ECONOMICS, Economics
from app.domain.router import RouteDecision
from app.domain.signals import parse_range
from app.enums import (
    BidModel,
    ConfigStatus,
    GenderSkew,
    Objective,
    PriceTier,
    PurchaseModel,
    RouteFlag,
    Verdict,
)
from app.schemas import (
    AdvertiserBrief,
    BidStrategy,
    Budget,
    CampaignConfig,
    CreativeRotation,
    CreativeVariant,
    Demographics,
    FrequencyCap,
    Kpis,
    PersonaSelection,
    PublisherAllocation,
    PublisherAssessment,
    Targeting,
)


class CampaignPlanner:
    def __init__(self, catalog: CatalogRepository, economics: Economics = DEFAULT_ECONOMICS,
                 allocator: BudgetAllocator | None = None) -> None:
        self._catalog = catalog
        self._economics = economics
        self._allocator = allocator or BudgetAllocator()

    def build(self, brief: AdvertiserBrief, assessments: list[PublisherAssessment],
              personas: PersonaSelection | None, creatives: list[CreativeVariant],
              decision: RouteDecision) -> CampaignConfig:
        recommended = [a for a in assessments if a.verdict is Verdict.RECOMMEND]
        status = ConfigStatus.DRAFT if recommended else ConfigStatus.NOT_RECOMMENDED
        confidence = round(min(1.0, brief.confidence * decision.confidence_multiplier), 2)
        pilot = self._economics.pilot_for(confidence)
        total = pilot.total_usd if status is ConfigStatus.DRAFT else 0.0
        price = self._economics.price_point(brief.estimated_price_point_usd, brief.price_tier)

        allocation = self._allocation(recommended, total)
        weighted_cpm = self._weighted_cpm(allocation)
        bid = self._bid_strategy(brief, price, weighted_cpm)
        forecast = self._economics.forecast(total, weighted_cpm)

        return CampaignConfig(
            status=status,
            objective=self._objective(brief),
            confidence=confidence,
            targeting=self._targeting(brief, assessments, personas),
            publisher_allocation=allocation,
            bid_strategy=bid,
            budget=Budget(total_usd=total,
                          daily_cap_usd=round(total / pilot.flight_days, 2) if total else 0.0,
                          flight_days=pilot.flight_days),
            creative_rotation=CreativeRotation(
                optimize_after_impressions=self._economics.optimize_after_impressions),
            frequency_cap=FrequencyCap(impressions=self._economics.frequency_cap_impressions,
                                       per_days=self._economics.frequency_cap_days),
            kpis=self._kpis(bid, forecast.conversions),
            forecast=forecast,
            assumptions=self._assumptions(brief, confidence, pilot.total_usd, pilot.flight_days),
            open_questions=self._open_questions(brief, status, decision, creatives),
        )

    # ---- pieces ----
    @staticmethod
    def _objective(brief: AdvertiserBrief) -> Objective:
        if brief.purchase_model is PurchaseModel.GIFTING:
            return Objective.SEASONAL_GIFTING
        if brief.purchase_model is PurchaseModel.B2B or brief.price_tier is PriceTier.LUXURY:
            return Objective.CONSIDERATION
        return Objective.ACQUISITION

    def _targeting(self, brief: AdvertiserBrief, assessments: list[PublisherAssessment],
                   personas: PersonaSelection | None) -> Targeting:
        recommended = [a for a in assessments if a.verdict is Verdict.RECOMMEND]
        picks = personas.selected if personas else []
        persona_rows = [self._catalog.persona(p.persona_id) for p in picks]

        age_range = brief.target_customer.age_range or self._union_age_range(
            [p.age_range for p in persona_rows])
        gender = brief.target_customer.gender_skew
        if gender is GenderSkew.UNKNOWN and persona_rows:
            gender = self._majority_gender([p.gender_skew for p in persona_rows])

        publishers = [self._catalog.publisher(a.publisher_id) for a in recommended]
        income_tiers = _unique([p.audience.income_tier for p in publishers])
        contextual = _unique([p.category for p in publishers]
                             + [s for p in publishers for s in p.subcategories])
        geos = _unique([g for p in publishers for g in p.audience.top_geos])
        if "nationwide" in geos:
            geos = ["nationwide"]

        exclusions = _unique([f"messaging: {d}" for p in persona_rows for d in p.disinterested_in]
                             + [f"publisher: {a.publisher_name}" for a in assessments
                                if a.verdict is Verdict.EXCLUDE][:8])
        return Targeting(
            demographics=Demographics(age_range=age_range, gender_skew=gender,
                                      income_tiers=income_tiers),
            interests=_unique([a for p in persona_rows for a in p.category_affinities]),
            contextual=contextual,
            geo=geos,
            persona_ids=[p.persona_id for p in picks],
            exclusions=exclusions,
        )

    def _allocation(self, recommended: list[PublisherAssessment],
                    total: float) -> list[PublisherAllocation]:
        candidates = [AllocationCandidate(a.publisher_id, a.score,
                                          self._catalog.publisher(a.publisher_id).monthly_impressions)
                      for a in recommended]
        by_id = {a.publisher_id: a for a in recommended}
        rows = []
        for share in self._allocator.allocate(candidates):
            assessment = by_id[share.publisher_id]
            publisher = self._catalog.publisher(share.publisher_id)
            band = self._economics.cpm_band(publisher.audience.income_tier, assessment.score)
            budget = round(total * share.share, 2)
            mid = (band[0] + band[1]) / 2
            lead_reason = assessment.reasons[0] if assessment.reasons else "fit judged by the matcher"
            rows.append(PublisherAllocation(
                publisher_id=publisher.id,
                publisher_name=publisher.name,
                share_pct=round(share.share * 100, 1),
                budget_usd=budget,
                suggested_cpm_range_usd=band,
                est_impressions=int(budget / mid * 1000) if budget else 0,
                rationale=(f"Rank #{assessment.rank}, score {assessment.score:.0f}; "
                           f"{lead_reason} Reach {publisher.monthly_impressions / 1e6:.1f}M/mo, "
                           f"{publisher.audience.income_tier} income -> CPM ${band[0]:.0f}-{band[1]:.0f}."),
            ))
        return rows

    @staticmethod
    def _weighted_cpm(allocation: list[PublisherAllocation]) -> float:
        total_share = sum(a.share_pct for a in allocation)
        if not total_share:
            return 0.0
        return round(sum((a.suggested_cpm_range_usd[0] + a.suggested_cpm_range_usd[1]) / 2
                         * a.share_pct for a in allocation) / total_share, 2)

    def _bid_strategy(self, brief: AdvertiserBrief, price: float,
                      weighted_cpm: float) -> BidStrategy:
        target_cpa = self._economics.target_cpa(price, brief.purchase_model)
        stated = brief.estimated_price_point_usd is not None
        model = BidModel.CPA if target_cpa is not None and stated else BidModel.CPM
        if model is BidModel.CPA:
            rationale = (f"Optimise to a ${target_cpa:.0f} CPA "
                         f"({self._economics.target_cpa_share[brief.purchase_model]:.0%} of the "
                         f"${price:.0f} price point); start CPM bids at ${weighted_cpm:.0f} so the "
                         f"learning phase buys enough impressions to find converters.")
        else:
            rationale = (f"No stated price point, so bid a flat CPM of ${weighted_cpm:.0f} "
                         f"(fit-weighted average of the publisher bands) and review after the "
                         f"pilot; switch to a CPA target once conversions are measurable.")
        return BidStrategy(model=model, starting_cpm_usd=weighted_cpm, target_cpa_usd=target_cpa,
                           max_cpc_usd=self._economics.max_cpc(target_cpa), rationale=rationale)

    def _kpis(self, bid: BidStrategy, conversions: int) -> Kpis:
        targets: dict[str, float] = {"ctr": self._economics.assumed_ctr}
        if bid.target_cpa_usd is not None:
            targets["cpa_usd"] = bid.target_cpa_usd
        if conversions:
            targets["conversions"] = float(conversions)
        primary = "cpa_usd" if bid.model is BidModel.CPA else "ctr"
        secondary = [k for k in ("ctr", "cpa_usd", "conversions") if k in targets and k != primary]
        return Kpis(primary=primary, secondary=secondary, targets=targets)

    def _assumptions(self, brief: AdvertiserBrief, confidence: float, pilot_total: float,
                     flight_days: int) -> list[str]:
        notes = list(brief.assumptions)
        if brief.estimated_price_point_usd is None:
            notes.append(f"No price stated; the {brief.price_tier} tier default of "
                         f"${self._economics.price_by_tier[brief.price_tier]:.0f} drives the AOV "
                         f"and CPA maths.")
        notes.append(f"Confidence {confidence:.2f} -> pilot of ${pilot_total:,.0f} over "
                     f"{flight_days} days.")
        return notes + self._economics.describe()

    @staticmethod
    def _open_questions(brief: AdvertiserBrief, status: ConfigStatus, decision: RouteDecision,
                        creatives: list[CreativeVariant]) -> list[str]:
        questions = list(brief.clarifying_questions)
        if status is ConfigStatus.NOT_RECOMMENDED:
            questions.append("This catalog is consumer commerce (apparel, pet, home, grocery, "
                             "beauty, wellness); no publisher is a defensible fit. Which consumer "
                             "audience, if any, buys this product?")
        if RouteFlag.OFF_CATALOG in decision.flags and status is ConfigStatus.DRAFT:
            questions.append("The fit is weak; treat the allocation as an exploratory test, not a "
                             "plan.")
        if brief.estimated_price_point_usd is None:
            questions.append("What is the typical order value? It changes which publishers' "
                             "shoppers can afford the product.")
        failed = [c.persona_name for c in creatives if not c.lint.passed]
        if failed:
            questions.append(f"Creative for {', '.join(failed)} did not pass review after a "
                             f"retry; edit before launch.")
        return questions

    # ---- helpers ----
    @staticmethod
    def _union_age_range(ranges: list[str]) -> str | None:
        parsed = [r for r in (parse_range(x) for x in ranges) if r]
        if not parsed:
            return None
        return f"{min(r[0] for r in parsed)}-{max(r[1] for r in parsed)}"

    @staticmethod
    def _majority_gender(skews: list[str]) -> GenderSkew:
        female = sum("female" in s for s in skews)
        male = sum(s == "male" or "male-leaning" in s for s in skews)
        if female > len(skews) / 2:
            return GenderSkew.FEMALE
        if male > len(skews) / 2:
            return GenderSkew.MALE
        return GenderSkew.BALANCED


def _unique(items: list) -> list:
    seen: set = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
