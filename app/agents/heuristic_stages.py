"""Deterministic stand-ins for the agents, built from the same signals the model sees.

Used without an API key and in tests, so the whole pipeline runs end to end offline. It is a
template engine driven by small tables: every reason it writes points at a number in the data
pack, and the trace labels the run as heuristic."""

import re
import time
from dataclasses import dataclass

from app.agents.openai_agent import StageRun
from app.domain.catalog import CatalogRepository
from app.domain.categories import terms_for
from app.domain.creative_checks import BODY_MAX, CTA_MAX, HEADLINE_MAX
from app.domain.fit_signals import SignalCalculator, age_overlap_pct
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
    PublisherAssessmentDraft,
    RejectedPersonaDraft,
    ShopperPersona,
    StageMeta,
    Subscores,
    TargetCustomer,
)

# ---------------------------------------------------------------- intake vocabulary

# ordered: the first pattern that matches is the primary category, later ones are secondary
CATEGORY_PATTERNS: tuple[tuple[ProductCategory, str], ...] = (
    (ProductCategory.B2B_SOFTWARE, r"\b(b2b|saas|software|enterprise|practices|crm)\b"),
    (ProductCategory.OUTDOOR_GEAR, r"\b(ski|skiers?|backcountry|outerwear|shells?|hiking|climbing)\b"),
    (ProductCategory.ACTIVEWEAR, r"\b(activewear|leggings|sports bras?|workout (?:clothes|gear|apparel)|athleisure)\b"),
    (ProductCategory.SNACKS_PROTEIN, r"\b(protein bars?|snacks?|granola|jerky)\b"),
    (ProductCategory.SUPPLEMENTS_VITAMINS, r"\b(supplements?|creatine|pre-?workout|vitamins?|protein powder|probiotics?)\b"),
    (ProductCategory.ALCOHOL, r"\b(wine|beer|whisk(?:e)?y|spirits|vodka|gin)\b"),
    (ProductCategory.FUNCTIONAL_BEVERAGES, r"\b(drinks?|beverages?|sparkling|soda|seltzer|adaptogens?|kombucha|tonic)\b"),
    (ProductCategory.HOME_DECOR_CANDLES, r"\b(candles?|home decor|vases?|diffusers?)\b"),
    (ProductCategory.HOUSEHOLD_CLEANING, r"\b(cleaning|cleaners?|detergent|refillable|laundry)\b"),
    (ProductCategory.LUXURY_ACCESSORIES, r"\b(handbags?|leather goods|jewel+ery|watches)\b"),
    (ProductCategory.HOME_TEXTILES_BEDDING, r"\b(bedding|linens?|sheets|duvets?|pillows?|towels?)\b"),
    (ProductCategory.KITCHEN_COOKWARE, r"\b(cookware|pans?|skillets?|knives|kitchen)\b"),
    (ProductCategory.BEAUTY_SKINCARE, r"\b(skincare|serums?|makeup|moisturi[sz]er|sunscreen)\b"),
    (ProductCategory.HAIRCARE, r"\b(shampoo|haircare|conditioner)\b"),
    (ProductCategory.MEAL_KITS, r"\bmeal kits?\b"),
    (ProductCategory.GROCERIES_PANTRY, r"\b(grocer(?:y|ies)|pantry|coffee|tea|olive oil)\b"),
    (ProductCategory.FOOTWEAR, r"\b(shoes?|sneakers|boots|footwear)\b"),
    (ProductCategory.BASICS_SOCKS_UNDERWEAR, r"\b(socks|underwear|basics|intimates)\b"),
    (ProductCategory.MENS_APPAREL, r"\bmen'?s\b[^.]*\b(apparel|clothing|wear)\b"),
    (ProductCategory.WOMENS_APPAREL, r"\b(dress(?:es)?|apparel|clothing|fashion|workwear)\b"),
    (ProductCategory.FITNESS_SERVICES, r"\b(gym|fitness classes|yoga studio|personal training)\b"),
    (ProductCategory.WELLNESS_SERVICES, r"\b(spa|massage|therapy|meditation|wellness|feel better|well-?being)\b"),
    (ProductCategory.KIDS_BABY, r"\b(kids?|baby|babies|toddlers?|moms?|mothers?|parents?)\b"),
    (ProductCategory.GIFTS, r"\bgifts?\b"),
)
ATTRIBUTE_PATTERNS: tuple[tuple[BrandAttribute, str], ...] = (
    (BrandAttribute.SUSTAINABLE, r"sustainab|recycled|refillable|ocean plastic|single-use"),
    (BrandAttribute.PREMIUM, r"\bpremium\b|small-batch|vet-formulated|technical|grown-up"),
    (BrandAttribute.LUXURY, r"luxury|handcrafted|italian-made|custom-fit|bespoke"),
    (BrandAttribute.VALUE, r"compete on price|half the cost|affordable|cheap|budget"),
    (BrandAttribute.SCIENCE_BACKED, r"vet-formulated|clinical|science|formulation"),
    (BrandAttribute.SUBSCRIPTION, r"subscription"),
    (BrandAttribute.GIFTING, r"\bgifts?\b"),
    (BrandAttribute.PLAYFUL, r"\bfun\b|playful|don't taste like cardboard"),
    (BrandAttribute.HERITAGE, r"heritage|handcrafted|italian|florence|vermont|portugal"),
    (BrandAttribute.CONVENIENCE, r"convenien|delivered|instant"),
    (BrandAttribute.NATURAL_CLEAN, r"natural|no synthetic|non-toxic|grain-free|clean"),
    (BrandAttribute.PERSONALIZED, r"personali[sz]ed|custom"),
    (BrandAttribute.PERFORMANCE, r"performance|technical|workout|athlet"),
)
# pet categories: the animal and the item can sit in different sentences
PET_ANIMAL = re.compile(r"\b(cat|dog|pet|puppy|kitten)s?\b", re.I)
PET_SUPPLY_ITEM = re.compile(r"\b(box|toys?|supplies|litter|leash|crate)\b", re.I)
PET_FOOD_ITEM = re.compile(r"\b(food|kibble|treats?|meals?)\b", re.I)

PRICE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
LUXURY_WORDS = re.compile(r"\b(luxury|handcrafted|italian-made|bespoke|custom-fit|florence)\b", re.I)
PREMIUM_WORDS = re.compile(r"\b(premium|small-batch|hand-poured|vet-formulated|technical|made in|grown-up|sustainable)\b", re.I)
BUDGET_WORDS = re.compile(r"\b(compete on price|half the cost|cheap|affordable|budget)\b", re.I)
BUSINESS_WORDS = re.compile(r"\b(we|our|sell|make|help|offer|brand|company|product|service)\b", re.I)
AMBIGUOUS_WORDS = re.compile(r"\b(new kind of thing|something new for|a thing for)\b", re.I)
FEMALE_WORDS = re.compile(r"\b(women|woman|female|moms?|mothers?|her)\b", re.I)
MALE_WORDS = re.compile(r"\b(men|man|male|dads?|fathers?)\b", re.I)
AUDIENCE_PHRASE = re.compile(r"\b((?:targeting|for) [a-z][^.;]{6,70})", re.I)
VAGUE_MAX_WORDS = 8

# ---------------------------------------------------------------- persona and copy tables

# persona price_sensitivity -> bonus by advertiser tier group
PRICE_FIT_BONUS: dict[str, dict[str, int]] = {
    "low": {"premium": 25, "mid": 5, "budget": -10},
    "low-medium": {"premium": 15, "mid": 5, "budget": 0},
    "medium": {"premium": 5, "mid": 15, "budget": 5},
    "medium-high": {"premium": -5, "mid": 5, "budget": 15},
    "high": {"premium": -20, "mid": 5, "budget": 25},
}
# a brand attribute a persona rewards
PERSONA_ATTRIBUTE_BONUS: dict[str, tuple[BrandAttribute, int]] = {
    "persona_001": (BrandAttribute.SCIENCE_BACKED, 15),   # Wellness Optimizer
    "persona_002": (BrandAttribute.SUBSCRIPTION, 10),     # Busy Parent
    "persona_005": (BrandAttribute.HERITAGE, 15),         # Affluent Classic
    "persona_006": (BrandAttribute.SUSTAINABLE, 20),      # Sustainability Buyer
    "persona_007": (BrandAttribute.SUBSCRIPTION, 10),     # Convenience-First Millennial
    "persona_008": (BrandAttribute.VALUE, 20),            # Value-Conscious Shopper
    "persona_009": (BrandAttribute.PERFORMANCE, 15),      # Fitness Enthusiast
    "persona_010": (BrandAttribute.GIFTING, 25),          # The Gifter
}
# messaging preference -> (headline hook, body clause, cta)
COPY_BY_PREFERENCE: dict[str, tuple[str, str, str]] = {
    "science-backed claims": ("the evidence is on the label", "Formulated with care and nothing to hide.", "See the formula"),
    "ingredient transparency": ("every ingredient, named", "Read every ingredient before you buy.", "Read the label"),
    "time-saving": ("one less thing to think about", "Delivered so you never have to remember.", "Set it up once"),
    "aesthetic-forward": ("looks as good as it feels", "Designed to be seen.", "Take a look"),
    "vet-recommended": ("the choice vets get behind", "Made for the years that matter most.", "Meet the recipe"),
    "craftsmanship": ("made properly, made to last", "Quality you keep for years.", "See the craft"),
    "specific sustainability claims": ("less waste, measured", "Refill, reuse, and skip the single-use plastic.", "See the impact"),
    "speed": ("here fast, done faster", "Ordered now, at your door soon.", "Order now"),
    "discount-forward": ("the same quality for less", "Same results, a fairer price.", "Compare prices"),
    "performance claims": ("built to perform", "Made for training days, not rest days.", "Train with it"),
    "giftable": ("the gift they will actually use", "Arrives ready to give.", "Send as a gift"),
}
DEFAULT_COPY = ("made for how you shop", "Straightforward, no fuss.", "See the details")


@dataclass(frozen=True)
class Reading:
    """What the keyword tables found in the description."""

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

    # ---------------------------------------------------------------- stage 1: intake
    async def intake(self, description: str) -> StageRun[AdvertiserBrief]:
        started = time.perf_counter()
        text = description.strip()
        reading = read(text)
        quality = classify(text, reading)
        category = reading.primary or ProductCategory.OTHER
        assumptions = [f"Read the product as '{category}' from the wording (heuristic mode).",
                       f"Purchase model read as {reading.purchase_model}."]
        if reading.price is None:
            assumptions.append(f"No price stated; assumed the {reading.tier} tier from the wording.")
        brief = AdvertiserBrief(
            business_summary=first_sentence(text),
            product_category=category,
            secondary_categories=reading.secondary[:3],
            price_tier=reading.tier,
            estimated_price_point_usd=reading.price,
            purchase_model=reading.purchase_model,
            brand_attributes=reading.attributes,
            target_customer=TargetCustomer(age_range=None, gender_skew=gender_of(text),
                                           income_tier=INCOME_BY_TIER.get(reading.tier),
                                           life_stage=life_stage_of(text)),
            audience_signals=[m.group(1).strip() for m in AUDIENCE_PHRASE.finditer(text)][:2],
            is_consumer_commerce=category is not ProductCategory.B2B_SOFTWARE,
            input_quality=quality,
            confidence=confidence_of(quality, reading),
            assumptions=assumptions if quality is not InputQuality.INSUFFICIENT else [],
            clarifying_questions=questions_for(quality, reading),
            interpretations=interpretations_for(quality, text),
        )
        return StageRun(brief, self._meta(Stage.INTAKE, started))

    # ---------------------------------------------------------------- stage 2: match
    async def match(self, brief: AdvertiserBrief, signals: list[FitSignals]) -> StageRun[MatchOutput]:
        started = time.perf_counter()
        price = self._signals.price_point(brief)
        drafts = [self._assess(brief, s, price) for s in signals]
        return StageRun(MatchOutput(assessments=drafts), self._meta(Stage.MATCH, started))

    def _assess(self, brief: AdvertiserBrief, s: FitSignals, price: float) -> PublisherAssessmentDraft:
        publisher = self._catalog.publisher(s.publisher_id)
        audience = publisher.audience
        score = s.prior + min(10, 4 * len(s.notes_keyword_hits))
        if brief.price_tier is PriceTier.LUXURY and audience.income_tier is IncomeTier.HIGH:
            score += 10  # affluent shoppers can afford a price far above their usual order
        if s.category_overlap == 0:
            score = min(score, 45)
        if brief.is_off_catalog:
            score = min(score, 40)
        score = float(max(0, min(100, round(score))))
        verdict = (Verdict.RECOMMEND if score >= 70 else Verdict.CONSIDER if score >= 50
                   else Verdict.EXCLUDE)

        shelf = ", ".join(self._shelf_terms(brief, publisher))
        reasons = []
        if s.category_overlap >= 1:
            reasons.append(f"Same shelf: {shelf} shoppers just bought in this category.")
        elif s.category_overlap > 0:
            reasons.append(f"Adjacent shelf: {shelf} sits next to this product.")
        reasons.append(f"Audience {audience.age_skew}, {audience.gender_split.female:.0%} female, "
                       f"{audience.income_tier} income: {s.age_overlap_pct:.0%} overlap with the target age range.")
        reasons.append(f"AOV ${publisher.avg_order_value_usd:.0f} vs ~${price:.0f} price point ({s.aov_ratio:.1f}x).")
        if s.notes_keyword_hits:
            reasons.append(f"Notes mention {', '.join(s.notes_keyword_hits)}: \"{publisher.notes[:70]}\"")
        concerns = []
        if s.aov_fit < 0.6:
            concerns.append("Price sits far from what these shoppers spend per order.")
        if s.gender_alignment < 0.4:
            concerns.append("Gender skew works against the target customer.")

        exclusion = None
        if verdict is not Verdict.RECOMMEND:
            if brief.is_off_catalog:
                exclusion = "The advertiser is outside this consumer catalog; no shelf carries it."
            elif s.category_overlap == 0:
                exclusion = (f"No shelf overlap: {publisher.category} ({', '.join(publisher.subcategories)}) "
                             f"vs {brief.product_category}; audience {audience.age_skew}, {audience.income_tier} income.")
            elif s.aov_fit < 0.6:
                exclusion = f"Price mismatch: ~${publisher.avg_order_value_usd:.0f} orders vs a {s.aov_ratio:.1f}x price point."
            else:
                exclusion = f"Weaker fit than the recommended set: {s.age_overlap_pct:.0%} age overlap, prior {s.prior:.0f}."

        return PublisherAssessmentDraft(
            publisher_id=publisher.id, verdict=verdict, score=score,
            subscores=Subscores(
                audience_fit=round(100 * (0.6 * s.age_overlap_pct + 0.4 * s.gender_alignment)),
                category_fit=round(100 * s.category_overlap),
                price_fit=round(100 * (0.5 * s.income_price_alignment + 0.5 * s.aov_fit)),
                context_fit=float(min(100, 50 + 15 * len(s.notes_keyword_hits)))),
            reasons=reasons[:3], concerns=concerns[:2], exclusion_reason=exclusion)

    @staticmethod
    def _shelf_terms(brief: AdvertiserBrief, publisher) -> list[str]:
        publisher_terms = {publisher.category, *publisher.subcategories}
        terms = terms_for(brief.product_category)
        return [t for t in (*terms.direct, *terms.adjacent) if t in publisher_terms] or [publisher.category]

    # ---------------------------------------------------------------- stage 3: personas
    async def select_personas(self, brief: AdvertiserBrief, recommended, persona_cap: int) -> StageRun[PersonaSelectionDraft]:
        started = time.perf_counter()
        terms = set(terms_for(brief.product_category).persona)
        for category in brief.secondary_categories:
            terms.update(terms_for(category).persona)
        scored = sorted(((persona_score(brief, p, terms), p) for p in self._catalog.personas),
                        key=lambda pair: -pair[0])
        cap = max(1, persona_cap)
        chosen = [pair for pair in scored if pair[0] >= 45][:cap] or scored[: min(3, cap)]
        chosen_ids = {p.id for _, p in chosen}
        selected = [self._pick(brief, p, score, terms, recommended) for score, p in chosen]
        rejected = [RejectedPersonaDraft(persona_id=p.id, why_not=why_not(brief, p, terms))
                    for _, p in scored if p.id not in chosen_ids]
        return StageRun(PersonaSelectionDraft(selected=selected, rejected=rejected),
                        self._meta(Stage.PERSONAS, started))

    def _pick(self, brief: AdvertiserBrief, persona: ShopperPersona, score: float, terms: set[str],
              recommended) -> PersonaPickDraft:
        overlap = sorted(set(persona.category_affinities) & terms)
        why = (f"Affinities {', '.join(overlap)} match the {brief.product_category} shelf; price "
               f"sensitivity '{persona.price_sensitivity}' fits the {brief.price_tier} tier."
               if overlap else
               f"Tentative: no direct affinity, but '{persona.price_sensitivity}' price sensitivity makes "
               f"a {brief.price_tier} {brief.product_category} purchase plausible.")
        watchouts = list(persona.disinterested_in[:2])
        if persona.id == "persona_010" and brief.purchase_model is PurchaseModel.SUBSCRIPTION:
            watchouts.append("A subscription offer conflicts with gifting; frame a first box as the gift.")
        best = [a.publisher_id for a in recommended
                if age_overlap_pct(persona.age_range, self._catalog.publisher(a.publisher_id).audience.age_skew) >= 0.4][:3]
        return PersonaPickDraft(
            persona_id=persona.id, fit_score=score, why_plausible=why,
            angle=truncate(f"Lead with {persona.messaging_preferences[0]}: {brief.business_summary}", 140),
            watchouts=watchouts, best_publishers=best)

    # ---------------------------------------------------------------- stage 4: creative
    async def write_creative(self, brief: AdvertiserBrief, pick: PersonaPickDraft, persona: ShopperPersona,
                             target_publishers: list[str], feedback: list[str]) -> StageRun[CreativeDraft]:
        started = time.perf_counter()
        preference = persona.messaging_preferences[0]
        hook, clause, cta = COPY_BY_PREFERENCE.get(preference, DEFAULT_COPY)
        product = brief.product_category.replace("_", " ").title()
        if feedback:  # the retry keeps the angle but strips anything the review flagged
            headline, body = f"{product}, made for you", brief.business_summary
        else:
            headline, body = f"{product}: {hook}", f"{brief.business_summary} {clause}"
        reasoning = f"Heuristic template: led with '{preference}' and avoided '{persona.disinterested_in[0]}'"
        if feedback:
            reasoning += f"; rewritten after review: {'; '.join(feedback)}"
        draft = CreativeDraft(
            headline=truncate(headline, HEADLINE_MAX), body=truncate(body, BODY_MAX),
            cta=truncate(cta, CTA_MAX), alt_headline=truncate(f"{hook} - {product}", HEADLINE_MAX),
            persona_reasoning=f"{reasoning}.")
        return StageRun(draft, self._meta(Stage.CREATIVE, started))

    # ---------------------------------------------------------------- stage 5: summary
    async def summarize(self, plan: dict) -> StageRun[CampaignSummary]:
        started = time.perf_counter()
        names = [a["publisher_name"] for a in plan.get("allocation", [])]
        personas = [p["persona_name"] for p in plan.get("personas", [])]
        budget = plan.get("budget", {})
        summary = (f"The pilot puts ${budget.get('total_usd', 0):,.0f} over {budget.get('flight_days', 0)} days "
                   f"on {', '.join(names) or 'no publisher'}, weighted by fit with reach as the tie-breaker. "
                   f"Copy speaks to {', '.join(personas) or 'no persona'}, each with its own angle. "
                   f"The pilot must prove the {plan.get('kpi', 'CTR')} target before the budget scales.")
        risks = [*plan.get("open_questions", [])[:3],
                 "Heuristic mode: reasons are template-generated from the data pack; run with an OpenAI key "
                 "for model judgement."]
        return StageRun(CampaignSummary(strategy_summary=summary, risks=risks), self._meta(Stage.SUMMARY, started))

    @staticmethod
    def _meta(stage: Stage, started: float) -> StageMeta:
        return StageMeta(stage=stage, ms=round((time.perf_counter() - started) * 1000),
                         mode=ExecutionMode.HEURISTIC)


# -------------------------------------------------------------------- intake helpers

INCOME_BY_TIER = {PriceTier.LUXURY: IncomeTier.HIGH, PriceTier.PREMIUM: IncomeTier.MID_HIGH,
                  PriceTier.BUDGET: IncomeTier.MID}


def read(text: str) -> Reading:
    matched = [c for c, pattern in CATEGORY_PATTERNS if re.search(pattern, text, re.I)]
    if PET_ANIMAL.search(text):
        pet = [c for c, item in ((ProductCategory.PET_SUPPLIES, PET_SUPPLY_ITEM),
                                 (ProductCategory.PET_FOOD, PET_FOOD_ITEM)) if item.search(text)]
        matched = (pet or [ProductCategory.PET_SUPPLIES]) + [c for c in matched if c not in pet]
    price_match = PRICE.search(text)
    price = float(price_match.group(1).replace(",", "")) if price_match else None
    if BUDGET_WORDS.search(text):
        tier = PriceTier.BUDGET
    elif (price is not None and price >= 400) or LUXURY_WORDS.search(text):
        tier = PriceTier.LUXURY
    elif (price is not None and price >= 80) or PREMIUM_WORDS.search(text):
        tier = PriceTier.PREMIUM
    else:
        tier = PriceTier.MID
    primary = matched[0] if matched else None
    if primary is ProductCategory.B2B_SOFTWARE:
        purchase = PurchaseModel.B2B
    elif re.search(r"\bsubscription\b", text, re.I):
        purchase = PurchaseModel.SUBSCRIPTION
    elif re.search(r"\bgifts?\b", text, re.I):
        purchase = PurchaseModel.GIFTING
    else:
        purchase = PurchaseModel.ONE_TIME
    attributes = [a for a, pattern in ATTRIBUTE_PATTERNS if re.search(pattern, text, re.I)]
    return Reading(primary, matched[1:], price, tier, purchase, attributes)


def classify(text: str, reading: Reading) -> InputQuality:
    words = len(text.split())
    if reading.primary is None:
        return (InputQuality.VAGUE if BUSINESS_WORDS.search(text) and words >= 4
                else InputQuality.INSUFFICIENT)
    if AMBIGUOUS_WORDS.search(text):
        return InputQuality.AMBIGUOUS
    if reading.primary in (ProductCategory.B2B_SOFTWARE, ProductCategory.OUTDOOR_GEAR):
        return InputQuality.OFF_CATALOG
    if words < VAGUE_MAX_WORDS and reading.price is None:
        return InputQuality.VAGUE  # a category but nothing else to plan with
    return InputQuality.CLEAR


def confidence_of(quality: InputQuality, reading: Reading) -> float:
    if quality is InputQuality.INSUFFICIENT:
        return 0.1
    if quality in (InputQuality.VAGUE, InputQuality.AMBIGUOUS):
        return 0.35
    if quality is InputQuality.OFF_CATALOG:
        return 0.5
    stated = 0.55 + (0.15 if reading.price is not None else 0) \
        + (0.1 if reading.purchase_model is not PurchaseModel.ONE_TIME else 0) + 0.1
    return round(min(stated, 0.9), 2)


def questions_for(quality: InputQuality, reading: Reading) -> list[str]:
    questions = []
    if quality in (InputQuality.VAGUE, InputQuality.AMBIGUOUS, InputQuality.INSUFFICIENT):
        questions.append("What exactly do you sell, and to whom?")
    if reading.price is None and quality is not InputQuality.INSUFFICIENT:
        questions.append("What does a typical order cost?")
    if quality is InputQuality.OFF_CATALOG:
        questions.append("Which consumer shoppers, if any, buy this product?")
    return questions[:3]


def interpretations_for(quality: InputQuality, text: str) -> list[Interpretation]:
    if quality is not InputQuality.AMBIGUOUS:
        return []
    who = "mothers" if FEMALE_WORDS.search(text) else "this audience"
    return [
        Interpretation(label=f"A baby or kids product for {who}",
                       brief_patch="It is a physical product for babies and young children."),
        Interpretation(label=f"A wellness or self-care product for {who}",
                       brief_patch="It is a wellness product the parent buys for themselves."),
        Interpretation(label=f"Apparel or accessories for {who}", brief_patch="It is clothing or an accessory."),
    ]


def gender_of(text: str) -> GenderSkew:
    female, male = bool(FEMALE_WORDS.search(text)), bool(MALE_WORDS.search(text))
    if female and not male:
        return GenderSkew.FEMALE
    if male and not female:
        return GenderSkew.MALE
    return GenderSkew.UNKNOWN


def life_stage_of(text: str) -> str | None:
    if re.search(r"\bnew (cat|dog|pet) owners\b", text, re.I):
        return "new pet owners"
    if re.search(r"\b(moms?|mothers?|parents?)\b", text, re.I):
        return "parents"
    return None


def first_sentence(text: str) -> str:
    return truncate(re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0], 160)


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return f"{cut}…" if cut else text[:limit]


# -------------------------------------------------------------------- persona helpers

def persona_score(brief: AdvertiserBrief, persona: ShopperPersona, terms: set[str]) -> float:
    overlap = len(set(persona.category_affinities) & terms)
    score = (45 + min(30, 15 * (overlap - 1))) if overlap else 5

    tier_group = ("premium" if brief.price_tier in (PriceTier.PREMIUM, PriceTier.LUXURY)
                  else "budget" if brief.price_tier is PriceTier.BUDGET else "mid")
    score += PRICE_FIT_BONUS.get(persona.price_sensitivity, {}).get(tier_group, 0)

    skew = brief.target_customer.gender_skew
    if skew is GenderSkew.FEMALE and "female" in persona.gender_skew:
        score += 10
    elif skew is GenderSkew.MALE and persona.gender_skew == "male":
        score += 10
    elif skew in (GenderSkew.FEMALE, GenderSkew.MALE) and persona.gender_skew == "balanced":
        score += 5

    attributes = set(brief.brand_attributes)
    if brief.purchase_model is PurchaseModel.GIFTING:
        attributes.add(BrandAttribute.GIFTING)
    bonus = PERSONA_ATTRIBUTE_BONUS.get(persona.id)
    if bonus and bonus[0] in attributes:
        score += bonus[1]
    # the three mismatches the data pack calls out explicitly
    if persona.id == "persona_010" and brief.purchase_model is PurchaseModel.SUBSCRIPTION:
        score -= 30  # the Gifter dislikes subscription-only
    if persona.id == "persona_006" and BrandAttribute.SUSTAINABLE not in attributes:
        score -= 10  # the Sustainability Buyer needs a real claim
    if persona.id == "persona_008" and brief.price_tier is PriceTier.LUXURY:
        score -= 20  # the Value-Conscious Shopper dislikes luxury-only messaging
    return float(max(0, min(100, score)))


def why_not(brief: AdvertiserBrief, persona: ShopperPersona, terms: set[str]) -> str:
    if not set(persona.category_affinities) & terms:
        return (f"No affinity for {brief.product_category} (buys {', '.join(persona.category_affinities[:3])}); "
                f"'{persona.disinterested_in[0]}' is close to this pitch.")
    return (f"Weaker fit than the selected personas: '{persona.price_sensitivity}' price sensitivity "
            f"vs the {brief.price_tier} tier.")
