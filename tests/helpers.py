"""Sample-data builders and the scripted StageExecutor the pipeline tests run against. The
builders construct real pydantic models, so the tests also exercise the contracts; override any
field through kwargs."""

from app.agents.context import RunContext
from app.agents.openai_agent import StageRun
from app.domain.catalog import CatalogRepository
from app.domain.fit_signals import SignalCalculator
from app.enums import (
    BrandAttribute,
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
    ClarificationRequest,
    CreativeDraft,
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


class ScriptedExecutor:
    """Stands in for the agents. Returns the brief (or clarification) a test hands it, scores
    publishers by their computed prior so guards and allocation see realistic numbers, picks the
    personas it is told to, and writes one clean creative per persona."""

    def __init__(self, catalog: CatalogRepository, brief: AdvertiserBrief | None = None,
                 clarification: ClarificationRequest | None = None,
                 picks: tuple[str, ...] = ("persona_004", "persona_001", "persona_002")) -> None:
        self.catalog = catalog
        self.signals = SignalCalculator(catalog)
        self.brief = brief or create_sample_brief()
        self.clarification = clarification
        self.picks = picks
        self.calls: list[Stage] = []

    def new_context(self, description: str) -> RunContext:
        return RunContext(catalog=self.catalog, signals=self.signals, description=description)

    async def intake(self, ctx: RunContext, session_id: str | None) -> StageRun:
        self.session_id = session_id
        if self.clarification:
            return self._run(Stage.INTAKE, self.clarification, agent="clarifier", handoffs=1)
        return self._run(Stage.INTAKE, self.brief, agent="brief_writer", handoffs=1)

    async def match(self, ctx: RunContext) -> StageRun:
        scores = {pid: round(s.prior) for pid, s in ctx.fit_signals.items()}
        return self._run(Stage.MATCH, create_match_output(scores), agent="publisher_matcher", tool_calls=3)

    async def select_personas(self, ctx: RunContext, persona_cap: int) -> StageRun:
        best = [r.publisher_id for r in ctx.recommended[:2]]
        selected = [PersonaPickDraft(persona_id=pid, fit_score=90 - 10 * i, why_plausible="scripted",
                                     angle="scripted angle", watchouts=[], best_publishers=best)
                    for i, pid in enumerate(self.picks[:persona_cap])]
        rejected = [RejectedPersonaDraft(persona_id=p.id, why_not="scripted")
                    for p in self.catalog.personas if p.id not in self.picks[:persona_cap]]
        return self._run(Stage.PERSONAS, PersonaSelectionDraft(selected=selected, rejected=rejected),
                         agent="persona_strategist", tool_calls=1)

    async def write_creative(self, ctx: RunContext, pick: PersonaPickDraft, persona: ShopperPersona,
                             target_publishers: list[str]) -> StageRun:
        draft = create_sample_creative(body="Senior recipes delivered monthly. Read every ingredient on the label.",
                                       persona_reasoning=f"Scripted for {persona.name}.")
        return self._run(Stage.CREATIVE, draft, agent=f"copywriter_{persona.id}", tool_calls=1)

    async def summarize(self, ctx: RunContext, plan: dict) -> StageRun:
        return self._run(Stage.SUMMARY, CampaignSummary(strategy_summary="Scripted summary.", risks=["none"]),
                         agent="strategy_summariser")

    def _run(self, stage: Stage, output, agent: str, tool_calls: int = 0, handoffs: int = 0) -> StageRun:
        self.calls.append(stage)
        return StageRun(output, StageMeta(stage=stage, ms=1, agent=agent, model="scripted", prompt_version="test",
                                          input_tokens=10, output_tokens=5, tool_calls=tool_calls, handoffs=handoffs))
