"""Assembles the campaign config from everything upstream.

Deterministic on purpose: every number in the config can be traced to a score, a catalog
field, or a constant in economics.py, and the constants used are echoed into `assumptions`
so the reviewer sees them. The steps:

    1. status and confidence (draft, or not recommended when nothing fits)
    2. pilot budget from confidence
    3. targeting from the brief, the personas and the recommended publishers
    4. budget split across the recommended publishers, with a CPM band each
    5. bid strategy (CPA when a price is known, else flat CPM)
    6. KPIs, forecast, assumptions and open questions
"""

from app.domain.budget_split import AllocationCandidate, BudgetAllocator
from app.domain.catalog import CatalogRepository
from app.domain.economics import DEFAULT_ECONOMICS, Economics
from app.domain.fit_signals import parse_range
from app.domain.input_policy import RouteDecision
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

MAX_PUBLISHER_EXCLUSIONS_LISTED = 8   # the targeting exclusions list names at most this many publishers

NOT_RECOMMENDED_QUESTION = ("This catalog is consumer commerce (apparel, pet, home, grocery, beauty, "
                            "wellness); no publisher is a defensible fit. Which consumer audience, if "
                            "any, buys this product?")
WEAK_FIT_NOTE = "The fit is weak; treat the allocation as an exploratory test, not a plan."
PRICE_QUESTION = ("What is the typical order value? It changes which publishers' shoppers can "
                  "afford the product.")


class ConfigBuilder:
    def __init__(self, catalog: CatalogRepository, economics: Economics = DEFAULT_ECONOMICS,
                 allocator: BudgetAllocator | None = None):
        self.catalog = catalog
        self.economics = economics
        self.allocator = allocator or BudgetAllocator()

    def build(self, brief: AdvertiserBrief, assessments: list[PublisherAssessment],
              personas: PersonaSelection | None, creatives: list[CreativeVariant],
              decision: RouteDecision) -> CampaignConfig:
        recommended = recommended_only(assessments)

        # --- Step 1: status and confidence ---
        if recommended:
            status = ConfigStatus.DRAFT
        else:
            status = ConfigStatus.NOT_RECOMMENDED
        # the router lowers confidence for vague or off-catalog input
        confidence = round(min(1.0, brief.confidence * decision.confidence_multiplier), 2)

        # --- Step 2: pilot budget; nothing is spent on a plan we do not recommend ---
        pilot = self.economics.pilot_for(confidence)
        if status is ConfigStatus.DRAFT:
            total_usd = pilot.total_usd
            daily_cap_usd = round(total_usd / pilot.flight_days, 2)
        else:
            total_usd = 0.0
            daily_cap_usd = 0.0

        # --- Step 3 to 5: targeting, allocation, bids ---
        targeting = self._targeting(brief, assessments, personas)
        allocation = self._allocation(recommended, total_usd)
        weighted_cpm = self._weighted_cpm(allocation)
        price = self.economics.price_point(brief.estimated_price_point_usd, brief.price_tier)
        bid = self._bid_strategy(brief, price, weighted_cpm)

        # --- Step 6: what to measure, what to expect, what was assumed, what is still open ---
        forecast = self.economics.forecast(total_usd, weighted_cpm)
        kpis = self._kpis(bid, forecast.conversions)
        assumptions = self._assumptions(brief, confidence, pilot.total_usd, pilot.flight_days)
        open_questions = self._open_questions(brief, status, decision, creatives)

        return CampaignConfig(
            status=status,
            objective=self._objective(brief),
            confidence=confidence,
            targeting=targeting,
            publisher_allocation=allocation,
            bid_strategy=bid,
            budget=Budget(total_usd=total_usd, daily_cap_usd=daily_cap_usd, flight_days=pilot.flight_days),
            creative_rotation=CreativeRotation(
                optimize_after_impressions=self.economics.optimize_after_impressions),
            frequency_cap=FrequencyCap(impressions=self.economics.frequency_cap_impressions,
                                       per_days=self.economics.frequency_cap_days),
            kpis=kpis,
            forecast=forecast,
            assumptions=assumptions,
            open_questions=open_questions,
        )

    # ---- objective ----

    @staticmethod
    def _objective(brief: AdvertiserBrief) -> Objective:
        if brief.purchase_model is PurchaseModel.GIFTING:
            return Objective.SEASONAL_GIFTING
        # expensive or B2B products are not impulse buys; the ad's job is consideration
        if brief.purchase_model is PurchaseModel.B2B or brief.price_tier is PriceTier.LUXURY:
            return Objective.CONSIDERATION
        return Objective.ACQUISITION

    # ---- targeting ----

    def _targeting(self, brief: AdvertiserBrief, assessments: list[PublisherAssessment],
                   personas: PersonaSelection | None) -> Targeting:
        picks = personas.selected if personas else []
        persona_rows = []
        for pick in picks:
            persona_rows.append(self.catalog.persona(pick.persona_id))

        publishers = []
        for assessment in recommended_only(assessments):
            publishers.append(self.catalog.publisher(assessment.publisher_id))

        # demographics: the brief first, the chosen personas fill in what it left unknown
        age_range = brief.target_customer.age_range
        if age_range is None:
            persona_ranges = []
            for persona in persona_rows:
                persona_ranges.append(persona.age_range)
            age_range = self._union_age_range(persona_ranges)

        gender = brief.target_customer.gender_skew
        if gender is GenderSkew.UNKNOWN and persona_rows:
            persona_skews = []
            for persona in persona_rows:
                persona_skews.append(persona.gender_skew)
            gender = self._majority_gender(persona_skews)

        income_tiers = []
        for publisher in publishers:
            income_tiers.append(publisher.audience.income_tier)

        # interests come from the personas; contextual terms and geos from the publishers
        interests = []
        for persona in persona_rows:
            interests.extend(persona.category_affinities)

        contextual = []
        for publisher in publishers:
            contextual.append(publisher.category)
        for publisher in publishers:
            contextual.extend(publisher.subcategories)

        geos = []
        for publisher in publishers:
            geos.extend(publisher.audience.top_geos)
        geos = unique(geos)
        if "nationwide" in geos:
            geos = ["nationwide"]

        # exclusions: what the personas dislike, and the publishers the matcher ruled out
        exclusions = []
        for persona in persona_rows:
            for dislike in persona.disinterested_in:
                exclusions.append(f"messaging: {dislike}")
        excluded_publishers = []
        for assessment in assessments:
            if assessment.verdict is Verdict.EXCLUDE:
                excluded_publishers.append(f"publisher: {assessment.publisher_name}")
        exclusions.extend(excluded_publishers[:MAX_PUBLISHER_EXCLUSIONS_LISTED])

        persona_ids = []
        for pick in picks:
            persona_ids.append(pick.persona_id)

        return Targeting(
            demographics=Demographics(age_range=age_range, gender_skew=gender,
                                      income_tiers=unique(income_tiers)),
            interests=unique(interests),
            contextual=unique(contextual),
            geo=geos,
            persona_ids=persona_ids,
            exclusions=unique(exclusions),
        )

    @staticmethod
    def _union_age_range(ranges: list[str]) -> str | None:
        """The span covering every persona range: 25-45 and 30-55 -> 25-55."""
        lows = []
        highs = []
        for text in ranges:
            parsed = parse_range(text)
            if parsed is not None:
                lows.append(parsed[0])
                highs.append(parsed[1])
        if not lows:
            return None
        return f"{min(lows)}-{max(highs)}"

    @staticmethod
    def _majority_gender(skews: list[str]) -> GenderSkew:
        """Persona skews are free text ("female-leaning", "male", "balanced"); majority wins."""
        female_count = 0
        male_count = 0
        for skew in skews:
            if "female" in skew:
                female_count += 1
            elif skew == "male" or "male-leaning" in skew:
                male_count += 1
        if female_count > len(skews) / 2:
            return GenderSkew.FEMALE
        if male_count > len(skews) / 2:
            return GenderSkew.MALE
        return GenderSkew.BALANCED

    # ---- allocation ----

    def _allocation(self, recommended: list[PublisherAssessment], total_usd: float) -> list[PublisherAllocation]:
        candidates = []
        assessments_by_id = {}
        for assessment in recommended:
            publisher = self.catalog.publisher(assessment.publisher_id)
            candidates.append(AllocationCandidate(assessment.publisher_id, assessment.score,
                                                  publisher.monthly_impressions))
            assessments_by_id[assessment.publisher_id] = assessment

        rows = []
        for share in self.allocator.allocate(candidates):
            assessment = assessments_by_id[share.publisher_id]
            publisher = self.catalog.publisher(share.publisher_id)

            # the CPM band depends on who the shoppers are and how good the fit is
            cpm_low, cpm_high = self.economics.cpm_band(publisher.audience.income_tier, assessment.score)
            cpm_mid = (cpm_low + cpm_high) / 2

            budget_usd = round(total_usd * share.share, 2)
            if budget_usd:
                est_impressions = int(budget_usd / cpm_mid * 1000)
            else:
                est_impressions = 0

            if assessment.reasons:
                lead_reason = assessment.reasons[0]
            else:
                lead_reason = "fit judged by the matcher"

            rows.append(PublisherAllocation(
                publisher_id=publisher.id,
                publisher_name=publisher.name,
                share_pct=round(share.share * 100, 1),
                budget_usd=budget_usd,
                suggested_cpm_range_usd=(cpm_low, cpm_high),
                est_impressions=est_impressions,
                rationale=(f"Rank #{assessment.rank}, score {assessment.score:.0f}; {lead_reason} "
                           f"Reach {publisher.monthly_impressions / 1e6:.1f}M/mo, "
                           f"{publisher.audience.income_tier} income -> CPM ${cpm_low:.0f}-{cpm_high:.0f}."),
            ))
        return rows

    @staticmethod
    def _weighted_cpm(allocation: list[PublisherAllocation]) -> float:
        """The mid-band CPM of each publisher, weighted by its share of the budget."""
        total_share = 0.0
        weighted_sum = 0.0
        for row in allocation:
            cpm_low, cpm_high = row.suggested_cpm_range_usd
            cpm_mid = (cpm_low + cpm_high) / 2
            weighted_sum += cpm_mid * row.share_pct
            total_share += row.share_pct
        if not total_share:
            return 0.0
        return round(weighted_sum / total_share, 2)

    # ---- bids and KPIs ----

    def _bid_strategy(self, brief: AdvertiserBrief, price: float, weighted_cpm: float) -> BidStrategy:
        target_cpa = self.economics.target_cpa(price, brief.purchase_model)
        price_was_stated = brief.estimated_price_point_usd is not None

        # a CPA target only makes sense when the price it is derived from is real
        if target_cpa is not None and price_was_stated:
            model = BidModel.CPA
            cpa_share = self.economics.target_cpa_share[brief.purchase_model]
            rationale = (f"Optimise to a ${target_cpa:.0f} CPA ({cpa_share:.0%} of the ${price:.0f} "
                         f"price point); start CPM bids at ${weighted_cpm:.0f} so the learning phase "
                         f"buys enough impressions to find converters.")
        else:
            model = BidModel.CPM
            rationale = (f"No stated price point, so bid a flat CPM of ${weighted_cpm:.0f} "
                         f"(fit-weighted average of the publisher bands) and review after the "
                         f"pilot; switch to a CPA target once conversions are measurable.")

        return BidStrategy(model=model, starting_cpm_usd=weighted_cpm, target_cpa_usd=target_cpa,
                           max_cpc_usd=self.economics.max_cpc(target_cpa), rationale=rationale)

    def _kpis(self, bid: BidStrategy, conversions: int) -> Kpis:
        targets = {"ctr": self.economics.assumed_ctr}
        if bid.target_cpa_usd is not None:
            targets["cpa_usd"] = bid.target_cpa_usd
        if conversions:
            targets["conversions"] = float(conversions)

        # the primary KPI follows the bid model; everything else measured is secondary
        if bid.model is BidModel.CPA:
            primary = "cpa_usd"
        else:
            primary = "ctr"
        secondary = []
        for name in ("ctr", "cpa_usd", "conversions"):
            if name in targets and name != primary:
                secondary.append(name)
        return Kpis(primary=primary, secondary=secondary, targets=targets)

    # ---- the honesty fields ----

    def _assumptions(self, brief: AdvertiserBrief, confidence: float, pilot_total: float,
                     flight_days: int) -> list[str]:
        notes = list(brief.assumptions)
        if brief.estimated_price_point_usd is None:
            default_price = self.economics.price_by_tier[brief.price_tier]
            notes.append(f"No price stated; the {brief.price_tier} tier default of ${default_price:.0f} "
                         f"drives the AOV and CPA maths.")
        notes.append(f"Confidence {confidence:.2f} -> pilot of ${pilot_total:,.0f} over {flight_days} days.")
        notes.extend(self.economics.describe())
        return notes

    @staticmethod
    def _open_questions(brief: AdvertiserBrief, status: ConfigStatus, decision: RouteDecision,
                        creatives: list[CreativeVariant]) -> list[str]:
        questions = list(brief.clarifying_questions)
        if status is ConfigStatus.NOT_RECOMMENDED:
            questions.append(NOT_RECOMMENDED_QUESTION)
        if RouteFlag.OFF_CATALOG in decision.flags and status is ConfigStatus.DRAFT:
            questions.append(WEAK_FIT_NOTE)
        if brief.estimated_price_point_usd is None:
            questions.append(PRICE_QUESTION)

        failed_personas = []
        for creative in creatives:
            if not creative.lint.passed:
                failed_personas.append(creative.persona_name)
        if failed_personas:
            questions.append(f"Creative for {', '.join(failed_personas)} did not pass the length "
                             f"check; edit before launch.")
        return questions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def recommended_only(assessments: list[PublisherAssessment]) -> list[PublisherAssessment]:
    recommended = []
    for assessment in assessments:
        if assessment.verdict is Verdict.RECOMMEND:
            recommended.append(assessment)
    return recommended


def unique(items: list) -> list:
    """Drop duplicates, keeping the first occurrence in order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
