"""Deterministic stand-ins for the four agents, built from the same signals the LLM sees.

Used when no API key is configured and by the test suite, so the whole pipeline - routing,
guards, lint, allocation, streaming - runs end to end without a network. It is honest about
being a template engine: every reason it writes points at a number in the data pack, and the
trace labels the run as heuristic."""

import re
import time
from dataclasses import dataclass

from app.agents.runner import StageRun
from app.domain.catalog import CatalogRepository
from app.domain.lint import BODY_MAX, CTA_MAX, HEADLINE_MAX
from app.domain.signals import SignalCalculator, age_overlap_pct
from app.domain.taxonomy import terms_for
from app.enums import (
    BrandAttribute,
    ExecutionMode,
    GenderSkew,
    IncomeTier,
    InputQuality,
    PriceTier,
    ProductCategory,
    PurchaseModel,
    Stage,
    Verdict,
)
from app.schemas import (
    AdvertiserBrief,
    CampaignSummary,
    CreativeDraft,
    FitSignals,
    Interpretation,
    MatchOutput,
    PersonaPickDraft,
    PersonaSelectionDraft,
    PublisherAssessment,
    PublisherAssessmentDraft,
    RejectedPersonaDraft,
    ShopperPersona,
    StageMeta,
    Subscores,
    TargetCustomer,
)

# ------------------------------------------------------------------ intake vocabulary

# ordered: the first pattern that matches is the primary category, later ones are secondary
_CATEGORY_PATTERNS: tuple[tuple[ProductCategory, re.Pattern[str]], ...] = (
    (ProductCategory.B2B_SOFTWARE, re.compile(r"\b(b2b|saas|software|enterprise|practices|crm)\b", re.I)),
    (ProductCategory.OUTDOOR_GEAR, re.compile(r"\b(ski|skiers?|backcountry|outerwear|shells?|hiking|climbing)\b", re.I)),
    (ProductCategory.ACTIVEWEAR, re.compile(r"\b(activewear|leggings|sports bras?|workout (?:clothes|gear|apparel)|athleisure)\b", re.I)),
    (ProductCategory.SNACKS_PROTEIN, re.compile(r"\b(protein bars?|snacks?|granola|jerky)\b", re.I)),
    (ProductCategory.SUPPLEMENTS_VITAMINS, re.compile(r"\b(supplements?|creatine|pre-?workout|vitamins?|protein powder|probiotics?)\b", re.I)),
    (ProductCategory.ALCOHOL, re.compile(r"\b(wine|beer|whisk(?:e)?y|spirits|vodka|gin)\b", re.I)),
    (ProductCategory.FUNCTIONAL_BEVERAGES, re.compile(r"\b(drinks?|beverages?|sparkling|soda|seltzer|adaptogens?|kombucha|tonic)\b", re.I)),
    (ProductCategory.HOME_DECOR_CANDLES, re.compile(r"\b(candles?|home decor|vases?|diffusers?)\b", re.I)),
    (ProductCategory.HOUSEHOLD_CLEANING, re.compile(r"\b(cleaning|cleaners?|detergent|refillable|laundry)\b", re.I)),
    (ProductCategory.LUXURY_ACCESSORIES, re.compile(r"\b(handbags?|leather goods|jewel+ery|watches)\b", re.I)),
    (ProductCategory.HOME_TEXTILES_BEDDING, re.compile(r"\b(bedding|linens?|sheets|duvets?|pillows?|towels?)\b", re.I)),
    (ProductCategory.KITCHEN_COOKWARE, re.compile(r"\b(cookware|pans?|skillets?|knives|kitchen)\b", re.I)),
    (ProductCategory.BEAUTY_SKINCARE, re.compile(r"\b(skincare|serums?|makeup|moisturi[sz]er|sunscreen)\b", re.I)),
    (ProductCategory.HAIRCARE, re.compile(r"\b(shampoo|haircare|conditioner)\b", re.I)),
    (ProductCategory.MEAL_KITS, re.compile(r"\bmeal kits?\b", re.I)),
    (ProductCategory.GROCERIES_PANTRY, re.compile(r"\b(grocer(?:y|ies)|pantry|coffee|tea|olive oil)\b", re.I)),
    (ProductCategory.FOOTWEAR, re.compile(r"\b(shoes?|sneakers|boots|footwear)\b", re.I)),
    (ProductCategory.BASICS_SOCKS_UNDERWEAR, re.compile(r"\b(socks|underwear|basics|intimates)\b", re.I)),
    (ProductCategory.MENS_APPAREL, re.compile(r"\bmen'?s\b[^.]*\b(apparel|clothing|wear)\b", re.I)),
    (ProductCategory.WOMENS_APPAREL, re.compile(r"\b(dress(?:es)?|apparel|clothing|fashion|workwear)\b", re.I)),
    (ProductCategory.FITNESS_SERVICES, re.compile(r"\b(gym|fitness classes|yoga studio|personal training)\b", re.I)),
    (ProductCategory.WELLNESS_SERVICES, re.compile(r"\b(spa|massage|therapy|meditation|wellness|feel better|well-?being)\b", re.I)),
    (ProductCategory.KIDS_BABY, re.compile(r"\b(kids?|baby|babies|toddlers?|moms?|mothers?|parents?)\b", re.I)),
    (ProductCategory.GIFTS, re.compile(r"\bgifts?\b", re.I)),
)
_ATTRIBUTE_PATTERNS: tuple[tuple[BrandAttribute, re.Pattern[str]], ...] = (
    (BrandAttribute.SUSTAINABLE, re.compile(r"sustainab|recycled|refillable|ocean plastic|single-use", re.I)),
    (BrandAttribute.PREMIUM, re.compile(r"\bpremium\b|small-batch|vet-formulated|technical|grown-up", re.I)),
    (BrandAttribute.LUXURY, re.compile(r"luxury|handcrafted|italian-made|custom-fit|bespoke", re.I)),
    (BrandAttribute.VALUE, re.compile(r"compete on price|half the cost|affordable|cheap|budget", re.I)),
    (BrandAttribute.SCIENCE_BACKED, re.compile(r"vet-formulated|clinical|science|formulation", re.I)),
    (BrandAttribute.SUBSCRIPTION, re.compile(r"subscription", re.I)),
    (BrandAttribute.GIFTING, re.compile(r"\bgifts?\b", re.I)),
    (BrandAttribute.PLAYFUL, re.compile(r"\bfun\b|playful|don't taste like cardboard", re.I)),
    (BrandAttribute.HERITAGE, re.compile(r"heritage|handcrafted|italian|florence|vermont|portugal", re.I)),
    (BrandAttribute.CONVENIENCE, re.compile(r"convenien|delivered|instant", re.I)),
    (BrandAttribute.NATURAL_CLEAN, re.compile(r"natural|no synthetic|non-toxic|grain-free|clean", re.I)),
    (BrandAttribute.PERSONALIZED, re.compile(r"personali[sz]ed|custom", re.I)),
    (BrandAttribute.PERFORMANCE, re.compile(r"performance|technical|workout|athlet", re.I)),
)
_PRICE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_LUXURY_WORDS = re.compile(r"\b(luxury|handcrafted|italian-made|bespoke|custom-fit|florence)\b", re.I)
_PREMIUM_WORDS = re.compile(r"\b(premium|small-batch|hand-poured|vet-formulated|technical|made in|grown-up|sustainable)\b", re.I)
_BUDGET_WORDS = re.compile(r"\b(compete on price|half the cost|cheap|affordable|budget)\b", re.I)
_BUSINESS_WORDS = re.compile(r"\b(we|our|sell|make|help|offer|brand|company|product|service)\b", re.I)
_AMBIGUOUS_WORDS = re.compile(r"\b(new kind of thing|something new for|a thing for)\b", re.I)
# pet categories: the animal and the item can sit in different sentences
_PET_ANIMAL = re.compile(r"\b(cat|dog|pet|puppy|kitten)s?\b", re.I)
_PET_SUPPLY_ITEM = re.compile(r"\b(box|toys?|supplies|litter|leash|crate)\b", re.I)
_PET_FOOD_ITEM = re.compile(r"\b(food|kibble|treats?|meals?)\b", re.I)
_VAGUE_MAX_WORDS = 8
_FEMALE = re.compile(r"\b(women|woman|female|moms?|mothers?|her)\b", re.I)
_MALE = re.compile(r"\b(men|man|male|dads?|fathers?)\b", re.I)
_AUDIENCE = re.compile(r"\b((?:targeting|for) [a-z][^.;]{6,70})", re.I)


@dataclass(frozen=True)
class _Reading:
    primary: ProductCategory | None
    secondary: list[ProductCategory]
    price: float | None
    tier: PriceTier
    purchase_model: PurchaseModel
    attributes: list[BrandAttribute]


class HeuristicStageExecutor:
    mode = ExecutionMode.HEURISTIC

    def __init__(self, catalog: CatalogRepository, signals: SignalCalculator) -> None:
        self._catalog = catalog
        self._signals = signals

    # ------------------------------------------------------------------ stage 1
    async def intake(self, description: str) -> StageRun[AdvertiserBrief]:
        started = time.perf_counter()
        text = description.strip()
        reading = self._read(text)
        quality = self._quality(text, reading)
        gender = (GenderSkew.FEMALE if _FEMALE.search(text) and not _MALE.search(text)
                  else GenderSkew.MALE if _MALE.search(text) and not _FEMALE.search(text)
                  else GenderSkew.UNKNOWN)
        income = {PriceTier.LUXURY: IncomeTier.HIGH, PriceTier.PREMIUM: IncomeTier.MID_HIGH,
                  PriceTier.BUDGET: IncomeTier.MID}.get(reading.tier)
        signals_found = [m.group(1).strip() for m in _AUDIENCE.finditer(text)][:2]
        category = reading.primary or ProductCategory.OTHER
        assumptions = [f"Read the product as '{category}' from the wording (heuristic mode)."]
        if reading.price is None:
            assumptions.append(f"No price stated; assumed the {reading.tier} tier from the wording.")
        assumptions.append(f"Purchase model read as {reading.purchase_model}.")
        confidence = self._confidence(quality, reading, signals_found)
        brief = AdvertiserBrief(
            business_summary=_summary(text),
            product_category=category,
            secondary_categories=reading.secondary[:3],
            price_tier=reading.tier,
            estimated_price_point_usd=reading.price,
            purchase_model=reading.purchase_model,
            brand_attributes=reading.attributes,
            target_customer=TargetCustomer(age_range=None, gender_skew=gender, income_tier=income,
                                           life_stage=_life_stage(text)),
            audience_signals=signals_found,
            is_consumer_commerce=category not in (ProductCategory.B2B_SOFTWARE,),
            input_quality=quality,
            confidence=confidence,
            assumptions=assumptions if quality is not InputQuality.INSUFFICIENT else [],
            clarifying_questions=self._questions(quality, reading),
            interpretations=self._interpretations(quality, text),
        )
        return StageRun(brief, self._meta(Stage.INTAKE, started))

    @staticmethod
    def _read(text: str) -> _Reading:
        matched = [category for category, pattern in _CATEGORY_PATTERNS if pattern.search(text)]
        if _PET_ANIMAL.search(text):
            pet = [c for c, item in ((ProductCategory.PET_SUPPLIES, _PET_SUPPLY_ITEM),
                                     (ProductCategory.PET_FOOD, _PET_FOOD_ITEM)) if item.search(text)]
            matched = (pet or [ProductCategory.PET_SUPPLIES]) + [c for c in matched if c not in pet]
        primary = matched[0] if matched else None
        price_match = _PRICE.search(text)
        price = float(price_match.group(1).replace(",", "")) if price_match else None
        if _BUDGET_WORDS.search(text):
            tier = PriceTier.BUDGET
        elif (price is not None and price >= 400) or _LUXURY_WORDS.search(text):
            tier = PriceTier.LUXURY
        elif (price is not None and price >= 80) or _PREMIUM_WORDS.search(text):
            tier = PriceTier.PREMIUM
        else:
            tier = PriceTier.MID
        if primary is ProductCategory.B2B_SOFTWARE:
            purchase = PurchaseModel.B2B
        elif re.search(r"\bsubscription\b", text, re.I):
            purchase = PurchaseModel.SUBSCRIPTION
        elif re.search(r"\bgifts?\b", text, re.I):
            purchase = PurchaseModel.GIFTING
        else:
            purchase = PurchaseModel.ONE_TIME
        attributes = [attr for attr, pattern in _ATTRIBUTE_PATTERNS if pattern.search(text)]
        return _Reading(primary=primary, secondary=matched[1:], price=price, tier=tier,
                        purchase_model=purchase, attributes=attributes)

    @staticmethod
    def _quality(text: str, reading: _Reading) -> InputQuality:
        words = len(text.split())
        if reading.primary is None:
            return (InputQuality.VAGUE if _BUSINESS_WORDS.search(text) and words >= 4
                    else InputQuality.INSUFFICIENT)
        if _AMBIGUOUS_WORDS.search(text):
            return InputQuality.AMBIGUOUS
        if reading.primary in (ProductCategory.B2B_SOFTWARE, ProductCategory.OUTDOOR_GEAR):
            return InputQuality.OFF_CATALOG
        if words < _VAGUE_MAX_WORDS and reading.price is None:
            return InputQuality.VAGUE  # a category but nothing else to plan with
        return InputQuality.CLEAR

    @staticmethod
    def _confidence(quality: InputQuality, reading: _Reading, signals_found: list[str]) -> float:
        if quality is InputQuality.INSUFFICIENT:
            return 0.1
        if quality in (InputQuality.VAGUE, InputQuality.AMBIGUOUS):
            return 0.35
        score = 0.55 + (0.15 if reading.price is not None else 0) + (0.1 if signals_found else 0) \
            + (0.1 if reading.purchase_model is not PurchaseModel.ONE_TIME else 0)
        return round(min(score, 0.9) if quality is InputQuality.CLEAR else 0.5, 2)

    @staticmethod
    def _questions(quality: InputQuality, reading: _Reading) -> list[str]:
        questions = []
        if quality in (InputQuality.VAGUE, InputQuality.AMBIGUOUS, InputQuality.INSUFFICIENT):
            questions.append("What exactly do you sell, and to whom?")
        if reading.price is None and quality is not InputQuality.INSUFFICIENT:
            questions.append("What does a typical order cost?")
        if quality is InputQuality.OFF_CATALOG:
            questions.append("Which consumer shoppers, if any, buy this product?")
        return questions[:3]

    @staticmethod
    def _interpretations(quality: InputQuality, text: str) -> list[Interpretation]:
        if quality is not InputQuality.AMBIGUOUS:
            return []
        subject = "this audience" if not _FEMALE.search(text) else "mothers"
        return [
            Interpretation(label=f"A baby or kids product for {subject}",
                           brief_patch="It is a physical product for babies and young children."),
            Interpretation(label=f"A wellness or self-care product for {subject}",
                           brief_patch="It is a wellness product the parent buys for themselves."),
            Interpretation(label=f"Apparel or accessories for {subject}",
                           brief_patch="It is clothing or an accessory."),
        ]

    # ------------------------------------------------------------------ stage 2
    async def match(self, brief: AdvertiserBrief, signals: list[FitSignals]) -> StageRun[MatchOutput]:
        started = time.perf_counter()
        price = self._signals.price_point(brief)
        drafts = [self._assess(brief, s, price) for s in signals]
        return StageRun(MatchOutput(assessments=drafts), self._meta(Stage.MATCH, started))

    def _assess(self, brief: AdvertiserBrief, s: FitSignals, price: float) -> PublisherAssessmentDraft:
        publisher = self._catalog.publisher(s.publisher_id)
        score = s.prior + min(10, 4 * len(s.notes_keyword_hits))
        if brief.price_tier is PriceTier.LUXURY and publisher.audience.income_tier is IncomeTier.HIGH:
            score += 10  # affluent shoppers can afford a price far above their usual order
        if s.category_overlap == 0:
            score = min(score, 45)
        if brief.is_off_catalog:
            score = min(score, 40)
        score = float(max(0, min(100, round(score))))
        verdict = (Verdict.RECOMMEND if score >= 70 else Verdict.CONSIDER if score >= 50
                   else Verdict.EXCLUDE)
        terms = self._matched_terms(brief, publisher)
        audience = publisher.audience
        reasons = []
        if s.category_overlap >= 1:
            reasons.append(f"Same shelf: {', '.join(terms)} shoppers just bought in this category.")
        elif s.category_overlap > 0:
            reasons.append(f"Adjacent shelf: {', '.join(terms)} sits next to this product.")
        reasons.append(f"Audience {audience.age_skew}, {audience.gender_split.female:.0%} female, "
                       f"{audience.income_tier} income: {s.age_overlap_pct:.0%} overlap with the "
                       f"target age range{' (unspecified)' if not brief.target_customer.age_range else ''}.")
        reasons.append(f"AOV ${publisher.avg_order_value_usd:.0f} vs ~${price:.0f} price point "
                       f"({s.aov_ratio:.1f}x).")
        if s.notes_keyword_hits:
            reasons.append(f"Notes mention {', '.join(s.notes_keyword_hits)}: \"{publisher.notes[:70]}\"")
        concerns = []
        if s.aov_fit < 0.6:
            concerns.append("Price sits far from what these shoppers spend per order.")
        if s.age_overlap_pct < 0.3 and brief.target_customer.age_range:
            concerns.append("Little age overlap with the target customer.")
        if s.gender_alignment < 0.4:
            concerns.append("Gender skew works against the target customer.")
        exclusion = None if verdict is Verdict.RECOMMEND else self._exclusion(brief, publisher, s)
        return PublisherAssessmentDraft(
            publisher_id=publisher.id, verdict=verdict, score=score,
            subscores=Subscores(
                audience_fit=round(100 * (0.6 * s.age_overlap_pct + 0.4 * s.gender_alignment)),
                category_fit=round(100 * s.category_overlap),
                price_fit=round(100 * (0.5 * s.income_price_alignment + 0.5 * s.aov_fit)),
                context_fit=float(min(100, 50 + 15 * len(s.notes_keyword_hits))),
            ),
            reasons=reasons[:3], concerns=concerns[:2], exclusion_reason=exclusion,
        )

    @staticmethod
    def _matched_terms(brief: AdvertiserBrief, publisher) -> list[str]:
        publisher_terms = {publisher.category, *publisher.subcategories}
        terms = terms_for(brief.product_category)
        hits = [t for t in (*terms.direct, *terms.adjacent) if t in publisher_terms]
        return hits or [publisher.category]

    @staticmethod
    def _exclusion(brief: AdvertiserBrief, publisher, s: FitSignals) -> str:
        if brief.is_off_catalog:
            return "The advertiser is outside this consumer catalog; no shelf carries it."
        if s.category_overlap == 0:
            return (f"No shelf overlap: {publisher.category} ({', '.join(publisher.subcategories)}) "
                    f"vs {brief.product_category}; audience {publisher.audience.age_skew}, "
                    f"{publisher.audience.income_tier} income.")
        if s.aov_fit < 0.6:
            return (f"Price mismatch: ~${publisher.avg_order_value_usd:.0f} orders vs a "
                    f"{s.aov_ratio:.1f}x price point.")
        return (f"Weaker fit than the recommended set: {s.age_overlap_pct:.0%} age overlap, "
                f"prior {s.prior:.0f}.")

    # ------------------------------------------------------------------ stage 3
    async def select_personas(self, brief: AdvertiserBrief, recommended: list[PublisherAssessment],
                              persona_cap: int) -> StageRun[PersonaSelectionDraft]:
        started = time.perf_counter()
        terms = set(terms_for(brief.product_category).persona)
        for category in brief.secondary_categories:
            terms.update(terms_for(category).persona)
        scored = sorted(((self._persona_score(brief, p, terms), p) for p in self._catalog.personas),
                        key=lambda pair: -pair[0])
        cap = max(1, persona_cap)
        chosen = [pair for pair in scored if pair[0] >= 45][:cap] or scored[: min(3, cap)]
        chosen_ids = {p.id for _, p in chosen}
        selected = [self._pick(brief, p, score, terms, recommended) for score, p in chosen]
        rejected = [RejectedPersonaDraft(persona_id=p.id, why_not=self._why_not(brief, p, terms))
                    for _, p in scored if p.id not in chosen_ids]
        return StageRun(PersonaSelectionDraft(selected=selected, rejected=rejected),
                        self._meta(Stage.PERSONAS, started))

    @staticmethod
    def _persona_score(brief: AdvertiserBrief, persona: ShopperPersona, terms: set[str]) -> float:
        overlap = len(set(persona.category_affinities) & terms)
        score = (45 + min(30, 15 * (overlap - 1))) if overlap else 5
        sensitivity = persona.price_sensitivity
        if brief.price_tier in (PriceTier.PREMIUM, PriceTier.LUXURY):
            score += {"low": 25, "low-medium": 15, "medium": 5, "medium-high": -5, "high": -20}.get(sensitivity, 0)
        elif brief.price_tier is PriceTier.BUDGET:
            score += {"high": 25, "medium-high": 15, "medium": 5, "low-medium": 0, "low": -10}.get(sensitivity, 0)
        else:
            score += 15 if sensitivity == "medium" else 5
        skew = brief.target_customer.gender_skew
        if skew is GenderSkew.FEMALE and "female" in persona.gender_skew:
            score += 10
        elif skew is GenderSkew.MALE and persona.gender_skew == "male":
            score += 10
        elif skew in (GenderSkew.FEMALE, GenderSkew.MALE) and persona.gender_skew == "balanced":
            score += 5
        attrs = set(brief.brand_attributes)
        gifting = brief.purchase_model is PurchaseModel.GIFTING or BrandAttribute.GIFTING in attrs
        if persona.id == "persona_010":  # the Gifter is a mode: gifts yes, subscriptions no
            score += 25 if gifting else 0
            score -= 30 if brief.purchase_model is PurchaseModel.SUBSCRIPTION else 0
        if persona.id == "persona_006":
            score += 20 if BrandAttribute.SUSTAINABLE in attrs else -10
        if persona.id == "persona_008":
            score += 20 if BrandAttribute.VALUE in attrs else 0
            score -= 20 if brief.price_tier is PriceTier.LUXURY else 0
        if persona.id == "persona_001" and BrandAttribute.SCIENCE_BACKED in attrs:
            score += 15
        if persona.id == "persona_009" and BrandAttribute.PERFORMANCE in attrs:
            score += 15
        if persona.id in ("persona_002", "persona_007") and brief.purchase_model is PurchaseModel.SUBSCRIPTION:
            score += 10
        if persona.id == "persona_005" and BrandAttribute.HERITAGE in attrs:
            score += 15
        return float(max(0, min(100, score)))

    def _pick(self, brief: AdvertiserBrief, persona: ShopperPersona, score: float, terms: set[str],
              recommended: list[PublisherAssessment]) -> PersonaPickDraft:
        overlap = sorted(set(persona.category_affinities) & terms)
        preference = persona.messaging_preferences[0]
        why = (f"Affinities {', '.join(overlap)} match the {brief.product_category} shelf; price "
               f"sensitivity '{persona.price_sensitivity}' fits the {brief.price_tier} tier."
               if overlap else
               f"Tentative: no direct affinity, but '{persona.price_sensitivity}' price sensitivity "
               f"and the persona's description make a {brief.price_tier} {brief.product_category} "
               f"purchase plausible.")
        watchouts = list(persona.disinterested_in[:2])
        if persona.id == "persona_010" and brief.purchase_model is PurchaseModel.SUBSCRIPTION:
            watchouts.append("A subscription offer conflicts with gifting; frame a first box as the gift.")
        best = [a.publisher_id for a in recommended
                if age_overlap_pct(persona.age_range,
                                   self._catalog.publisher(a.publisher_id).audience.age_skew) >= 0.4][:3]
        return PersonaPickDraft(
            persona_id=persona.id, fit_score=score, why_plausible=why,
            angle=_truncate(f"Lead with {preference}: {brief.business_summary}", 140),
            watchouts=watchouts, best_publishers=best,
        )

    @staticmethod
    def _why_not(brief: AdvertiserBrief, persona: ShopperPersona, terms: set[str]) -> str:
        overlap = set(persona.category_affinities) & terms
        if not overlap:
            return (f"No affinity for {brief.product_category} (buys {', '.join(persona.category_affinities[:3])}); "
                    f"'{persona.disinterested_in[0]}' is close to this pitch.")
        return (f"Weaker fit than the selected personas: '{persona.price_sensitivity}' price "
                f"sensitivity vs the {brief.price_tier} tier.")

    # ------------------------------------------------------------------ stage 4
    async def write_creative(self, brief: AdvertiserBrief, pick: PersonaPickDraft,
                             persona: ShopperPersona, target_publishers: list[str],
                             feedback: list[str]) -> StageRun[CreativeDraft]:
        started = time.perf_counter()
        preference = persona.messaging_preferences[0]
        product = _product_phrase(brief)
        hook = _HOOKS.get(preference, "Made for how you shop")
        headline = _truncate(f"{product}: {hook}", HEADLINE_MAX)
        body = _truncate(f"{brief.business_summary} {_CLAUSES.get(preference, 'Straightforward, no fuss.')}",
                         BODY_MAX)
        if feedback:  # the retry keeps the angle but strips anything the review flagged
            headline = _truncate(f"{product}, made for you", HEADLINE_MAX)
            body = _truncate(brief.business_summary, BODY_MAX)
        draft = CreativeDraft(
            headline=headline, body=body,
            cta=_truncate(_CTAS.get(preference, "See the details"), CTA_MAX),
            alt_headline=_truncate(f"{hook} - {product}", HEADLINE_MAX),
            persona_reasoning=(f"Heuristic template: led with '{preference}' and avoided "
                               f"'{persona.disinterested_in[0]}'"
                               + (f"; rewritten after review: {'; '.join(feedback)}" if feedback else "")
                               + "."),
        )
        return StageRun(draft, self._meta(Stage.CREATIVE, started))

    # ------------------------------------------------------------------ stage 5b
    async def summarize(self, plan: dict) -> StageRun[CampaignSummary]:
        started = time.perf_counter()
        names = [a["publisher_name"] for a in plan.get("allocation", [])]
        personas = [p["persona_name"] for p in plan.get("personas", [])]
        budget = plan.get("budget", {})
        summary = (f"The pilot puts ${budget.get('total_usd', 0):,.0f} over {budget.get('flight_days', 0)} days "
                   f"on {', '.join(names) or 'no publisher'}, weighted by fit with reach as the tie-breaker. "
                   f"Copy speaks to {', '.join(personas) or 'no persona'}, each with its own angle. "
                   f"The pilot must prove the {plan.get('kpi', 'CTR')} target before the budget scales.")
        risks = list(plan.get("open_questions", []))[:3]
        risks.append("Heuristic mode: reasons are template-generated from the data pack; run with "
                     "an OpenAI key for model judgement.")
        return StageRun(CampaignSummary(strategy_summary=summary, risks=risks),
                        self._meta(Stage.SUMMARY, started))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _meta(stage: Stage, started: float) -> StageMeta:
        return StageMeta(stage=stage, ms=round((time.perf_counter() - started) * 1000),
                         mode=ExecutionMode.HEURISTIC)


_HOOKS = {
    "science-backed claims": "the evidence is on the label",
    "ingredient transparency": "every ingredient, named",
    "outcome-focused": "built for results",
    "time-saving": "one less thing to think about",
    "family-friendly": "made for busy households",
    "aesthetic-forward": "looks as good as it feels",
    "vet-recommended": "the choice vets get behind",
    "craftsmanship": "made properly, made to last",
    "specific sustainability claims": "less waste, measured",
    "speed": "here fast, done faster",
    "discount-forward": "the same quality for less",
    "performance claims": "built to perform",
    "giftable": "the gift they will actually use",
}
_CLAUSES = {
    "science-backed claims": "Formulated with care and nothing to hide.",
    "ingredient transparency": "Read every ingredient before you buy.",
    "time-saving": "Delivered so you never have to remember.",
    "aesthetic-forward": "Designed to be seen.",
    "vet-recommended": "Made for the years that matter most.",
    "craftsmanship": "Quality you keep for years.",
    "specific sustainability claims": "Refill, reuse, and skip the single-use plastic.",
    "speed": "Ordered now, at your door soon.",
    "discount-forward": "Same results, a fairer price.",
    "performance claims": "Made for training days, not rest days.",
    "giftable": "Arrives ready to give.",
}
_CTAS = {
    "science-backed claims": "See the formula", "ingredient transparency": "Read the label",
    "time-saving": "Set it up once", "aesthetic-forward": "Take a look",
    "vet-recommended": "Meet the recipe", "craftsmanship": "See the craft",
    "specific sustainability claims": "See the impact", "speed": "Order now",
    "discount-forward": "Compare prices", "performance claims": "Train with it",
    "giftable": "Send as a gift",
}


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return f"{cut}…" if cut else text[:limit]


def _summary(text: str) -> str:
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].strip()
    return _truncate(first, 160)


def _product_phrase(brief: AdvertiserBrief) -> str:
    return brief.product_category.replace("_", " ").title()


def _life_stage(text: str) -> str | None:
    if re.search(r"\bnew (cat|dog|pet) owners\b", text, re.I):
        return "new pet owners"
    if re.search(r"\b(moms?|mothers?|parents?)\b", text, re.I):
        return "parents"
    if re.search(r"\bretire", text, re.I):
        return "retirees"
    return None
