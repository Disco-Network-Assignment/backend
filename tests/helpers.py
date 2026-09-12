"""Sample-data builders. They construct real pydantic models, so the tests also exercise the
contracts; override any field through kwargs."""

from app.enums import (
    BrandAttribute,
    GenderSkew,
    IncomeTier,
    InputQuality,
    PriceTier,
    ProductCategory,
    PurchaseModel,
    Verdict,
)
from app.schemas import (
    AdvertiserBrief,
    CreativeDraft,
    MatchOutput,
    PublisherAssessmentDraft,
    Subscores,
    TargetCustomer,
)

SENIOR_DOG_FOOD = ("We sell premium dog food for senior dogs, targeting owners who care about "
                   "joint health and longevity. Grain-free, vet-formulated, subscription-based.")


def create_sample_brief(**overrides) -> AdvertiserBrief:
    fields = dict(
        business_summary="Premium grain-free dog food subscription for senior dogs.",
        product_category=ProductCategory.PET_FOOD,
        secondary_categories=[ProductCategory.PET_HEALTH],
        price_tier=PriceTier.PREMIUM,
        estimated_price_point_usd=70.0,
        purchase_model=PurchaseModel.SUBSCRIPTION,
        brand_attributes=[BrandAttribute.PREMIUM, BrandAttribute.SUBSCRIPTION, BrandAttribute.SCIENCE_BACKED],
        target_customer=TargetCustomer(age_range="30-60", gender_skew=GenderSkew.BALANCED,
                                       income_tier=IncomeTier.MID_HIGH, life_stage=None),
        audience_signals=["owners who care about joint health and longevity"],
        is_consumer_commerce=True,
        input_quality=InputQuality.CLEAR,
        confidence=0.85,
        assumptions=[],
        clarifying_questions=[],
        interpretations=[],
    )
    fields.update(overrides)
    return AdvertiserBrief(**fields)


def create_sample_assessment(publisher_id: str, score: float, verdict: Verdict | None = None,
                             **overrides) -> PublisherAssessmentDraft:
    verdict = verdict or (Verdict.RECOMMEND if score >= 70 else Verdict.CONSIDER if score >= 50
                          else Verdict.EXCLUDE)
    fields = dict(
        publisher_id=publisher_id, verdict=verdict, score=score,
        subscores=Subscores(audience_fit=score, category_fit=score, price_fit=score, context_fit=score),
        reasons=["sample reason"], concerns=[],
        exclusion_reason=None if verdict is Verdict.RECOMMEND else "sample exclusion",
    )
    fields.update(overrides)
    return PublisherAssessmentDraft(**fields)


def create_match_output(scores: dict[str, float], **overrides) -> MatchOutput:
    return MatchOutput(assessments=[create_sample_assessment(pid, s, **overrides) for pid, s in scores.items()])


def create_sample_creative(**overrides) -> CreativeDraft:
    fields = dict(
        headline="Joint support they will actually eat",
        body="Vet-formulated, grain-free senior recipes delivered monthly. Read every ingredient.",
        cta="See the formula",
        alt_headline="Senior dog food, formulated with care",
        persona_reasoning="Used ingredient transparency; avoided generic pet-brand tone.",
    )
    fields.update(overrides)
    return CreativeDraft(**fields)
