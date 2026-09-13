"""Pydantic contracts: the catalog rows, the typed hand-offs between pipeline stages, and the
API wire shapes.

Models with a `Draft` suffix are exactly what an agent returns (they are the structured-output
schema the OpenAI Agents SDK enforces); the un-suffixed model is the same object after code has
enriched it (signals, guardrails, rank, lint...). Keeping the two apart means the model never
sees fields it is not supposed to fill, and code never trusts fields the model produced."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.enums import (
    BidModel,
    BrandAttribute,
    ConfigStatus,
    EventStatus,
    ExecutionMode,
    FailureKind,
    GenderSkew,
    GuardrailTag,
    IncomeTier,
    InputQuality,
    LintSeverity,
    Objective,
    PriceTier,
    ProductCategory,
    PurchaseModel,
    Stage,
    Verdict,
)

# ---------------------------------------------------------------------------- catalog


class GenderSplit(BaseModel):
    female: float
    male: float
    other: float


class AudienceProfile(BaseModel):
    age_skew: str
    gender_split: GenderSplit
    top_geos: list[str]
    income_tier: IncomeTier


class Publisher(BaseModel):
    """One row of publishers.json - a checkout / order-confirmation surface ads can run on."""

    id: str
    name: str
    category: str
    subcategories: list[str]
    monthly_impressions: int
    avg_order_value_usd: float
    audience: AudienceProfile
    notes: str


class ShopperPersona(BaseModel):
    """One row of shopper_personas.json."""

    id: str
    name: str
    age_range: str
    gender_skew: str
    description: str
    category_affinities: list[str]
    price_sensitivity: str
    messaging_preferences: list[str]
    disinterested_in: list[str]
    typical_aov_usd: float


class ExampleAdvertiser(BaseModel):
    """One numbered line of example_advertisers.txt."""

    id: str
    number: int
    description: str


# ---------------------------------------------------------------------------- stage 1: intake


class TargetCustomer(BaseModel):
    age_range: str | None = Field(description="Like '30-55'; null when the text gives no hint.")
    gender_skew: GenderSkew
    income_tier: IncomeTier | None = Field(description="null when unknown.")
    life_stage: str | None = Field(description="e.g. 'new parents', 'retirees'; null when unknown.")


class Interpretation(BaseModel):
    label: str = Field(description="Short chip text, e.g. 'Supplements for new mothers'.")
    brief_patch: str = Field(
        description="One sentence to append to the description if this reading is chosen."
    )


class AdvertiserBrief(BaseModel):
    """Structured reading of the advertiser's free text (the intake agent's contract)."""

    business_summary: str = Field(description="One neutral sentence: what is sold, to whom.")
    product_category: ProductCategory = Field(description="Closest controlled-vocabulary category.")
    secondary_categories: list[ProductCategory] = Field(
        description="Other categories that plausibly apply (0-3)."
    )
    price_tier: PriceTier
    estimated_price_point_usd: float | None = Field(
        description="Typical single-order price in USD when stated or safely inferable, else null."
    )
    purchase_model: PurchaseModel
    brand_attributes: list[BrandAttribute]
    target_customer: TargetCustomer
    audience_signals: list[str] = Field(
        description="Short quotes from the description that reveal the audience or intent."
    )
    is_consumer_commerce: bool = Field(
        description="False for B2B, services without a consumer purchase, or anything this "
        "consumer catalog cannot serve."
    )
    input_quality: InputQuality
    confidence: float = Field(description="0..1: share of the brief that is stated, not assumed.")
    assumptions: list[str] = Field(description="Every inferred fact, phrased so it can be corrected.")
    clarifying_questions: list[str] = Field(
        description="The 0-3 questions whose answers would most change the plan."
    )
    interpretations: list[Interpretation] = Field(
        description="2-3 readings when the input is ambiguous, otherwise empty."
    )

    @property
    def is_off_catalog(self) -> bool:
        return not self.is_consumer_commerce or self.input_quality is InputQuality.OFF_CATALOG


# ---------------------------------------------------------------------------- stage 2: match


class FitSignals(BaseModel):
    """Deterministic fit evidence per publisher, computed by code (see domain/signals.py)."""

    publisher_id: str
    category_overlap: float
    age_overlap_pct: float
    gender_alignment: float
    income_price_alignment: float
    aov_ratio: float
    aov_fit: float
    reach_index: float
    notes_keyword_hits: list[str]
    prior: float = Field(description="Weighted blend of the above on a 0-100 scale.")


class Subscores(BaseModel):
    audience_fit: float
    category_fit: float
    price_fit: float
    context_fit: float


class PublisherAssessmentDraft(BaseModel):
    """The matcher agent's verdict on one publisher."""

    publisher_id: str
    verdict: Verdict
    score: float = Field(
        description="0-100 overall judgement, not an average: 90+ an obvious home for the "
        "brand, 70 solid, 50 a stretch, under 40 no."
    )
    subscores: Subscores
    reasons: list[str] = Field(description="1-3 concrete reasons citing a number or a note.")
    concerns: list[str] = Field(description="0-2 risks or caveats.")
    exclusion_reason: str | None = Field(
        description="Required whenever the verdict is not 'recommend'; null otherwise."
    )


class MatchOutput(BaseModel):
    assessments: list[PublisherAssessmentDraft]


class PublisherAssessment(PublisherAssessmentDraft):
    """The draft after code enriched it: catalog name, signals, guardrails, final rank."""

    publisher_name: str
    signals: FitSignals
    guardrails_applied: list[GuardrailTag]
    rank: int


# ---------------------------------------------------------------------------- stage 3: personas


class PersonaPickDraft(BaseModel):
    persona_id: str
    fit_score: float = Field(description="0-100.")
    why_plausible: str = Field(
        description="Why this persona buys this product, citing the persona's description or affinities."
    )
    angle: str = Field(description="One-line messaging hook in the persona's own vocabulary.")
    watchouts: list[str] = Field(description="What to avoid, drawn from disinterested_in and the brief.")
    best_publishers: list[str] = Field(
        description="Publisher ids from the recommended set where this persona is most present."
    )


class RejectedPersonaDraft(BaseModel):
    persona_id: str
    why_not: str = Field(description="One sentence naming the mismatch.")


class PersonaSelectionDraft(BaseModel):
    """The persona strategist's contract: 3-5 picks (fewer when the cap is lower) + rejections."""

    selected: list[PersonaPickDraft] = Field(description="Ordered by fit, best first.")
    rejected: list[RejectedPersonaDraft]


class PersonaPick(PersonaPickDraft):
    persona_name: str


class RejectedPersona(RejectedPersonaDraft):
    persona_name: str


class PersonaSelection(BaseModel):
    selected: list[PersonaPick]
    rejected: list[RejectedPersona]


# ---------------------------------------------------------------------------- stage 4: creative


class CreativeDraft(BaseModel):
    """The copywriter's contract for one persona."""

    headline: str = Field(description="At most 60 characters.")
    body: str = Field(description="At most 160 characters, one or two sentences.")
    cta: str = Field(description="At most 20 characters, imperative.")
    alt_headline: str = Field(description="A second headline with a different hook.")
    persona_reasoning: str = Field(
        description="Which messaging preferences were used and which disinterests were avoided."
    )


class LintIssue(BaseModel):
    severity: LintSeverity
    rule: str
    message: str


class LintReport(BaseModel):
    passed: bool
    issues: list[LintIssue]
    retried: bool

    @property
    def hard_issues(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity is LintSeverity.HARD]


class CreativeVariant(CreativeDraft):
    id: str
    persona_id: str
    persona_name: str
    target_publishers: list[str]
    lint: LintReport


# ---------------------------------------------------------------------------- stage 5: config


class Demographics(BaseModel):
    age_range: str | None
    gender_skew: GenderSkew
    income_tiers: list[IncomeTier]


class Targeting(BaseModel):
    demographics: Demographics
    interests: list[str] = Field(description="Persona category affinities.")
    contextual: list[str] = Field(description="Publisher categories / subcategories to run in.")
    geo: list[str]
    persona_ids: list[str]
    exclusions: list[str] = Field(description="Messaging to avoid and publishers not to run on.")


class PublisherAllocation(BaseModel):
    publisher_id: str
    publisher_name: str
    share_pct: float
    budget_usd: float
    suggested_cpm_range_usd: tuple[float, float]
    est_impressions: int
    rationale: str


class BidStrategy(BaseModel):
    model: BidModel
    starting_cpm_usd: float
    target_cpa_usd: float | None
    max_cpc_usd: float | None
    rationale: str


class Budget(BaseModel):
    total_usd: float
    daily_cap_usd: float
    flight_days: int
    pacing: Literal["even"] = "even"


class CreativeRotation(BaseModel):
    mode: Literal["even_then_optimize"] = "even_then_optimize"
    optimize_after_impressions: int


class FrequencyCap(BaseModel):
    impressions: int
    per_days: int


class Kpis(BaseModel):
    primary: str
    secondary: list[str]
    targets: dict[str, float]


class Forecast(BaseModel):
    impressions: int
    clicks: int
    conversions: int
    cpa_usd: float | None


class CampaignConfig(BaseModel):
    """A reviewable draft: its uncertainty (confidence, assumptions, open questions) is data."""

    status: ConfigStatus
    objective: Objective
    confidence: float
    targeting: Targeting
    publisher_allocation: list[PublisherAllocation]
    bid_strategy: BidStrategy
    budget: Budget
    creative_rotation: CreativeRotation
    frequency_cap: FrequencyCap
    kpis: Kpis
    forecast: Forecast
    assumptions: list[str]
    open_questions: list[str]


class CampaignSummary(BaseModel):
    """Optional narrative for the reviewer (stage 5b)."""

    strategy_summary: str = Field(description="3-5 plain sentences interpreting the plan.")
    risks: list[str] = Field(description="Short, specific risks; 2-4 items.")


# ---------------------------------------------------------------------------- plan / trace


class StageMeta(BaseModel):
    stage: Stage
    ms: int
    mode: ExecutionMode
    model: str | None = None
    reasoning_effort: str | None = None
    prompt_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    retried: bool = False


class CampaignPlan(BaseModel):
    run_id: str
    description: str
    mode: ExecutionMode
    brief: AdvertiserBrief
    publishers: list[PublisherAssessment] = Field(description="All 20, ranked; excluded ones included.")
    personas: PersonaSelection | None
    creatives: list[CreativeVariant]
    config: CampaignConfig
    summary: CampaignSummary | None
    trace: list[StageMeta]

    @property
    def recommended(self) -> list[PublisherAssessment]:
        return [p for p in self.publishers if p.verdict is Verdict.RECOMMEND]


class StopResult(BaseModel):
    """What the pipeline returns when the input is too thin to plan anything."""

    run_id: str
    description: str
    mode: ExecutionMode
    brief: AdvertiserBrief
    reason: str
    clarifying_questions: list[str]
    examples: list[str]
    trace: list[StageMeta]


# ---------------------------------------------------------------------------- API


class PlanOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    force_exploratory: bool = Field(
        default=False,
        description="Run personas and creatives even when no publisher was recommended.",
    )
    mode: ExecutionMode | None = Field(
        default=None, description="Override the server's execution mode for this run."
    )


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=2000)
    options: PlanOptions = Field(default_factory=PlanOptions)


class PipelineEvent(BaseModel):
    """One NDJSON line. `data` carries the stage output (already a plain dict) on `completed`."""

    stage: Stage
    status: EventStatus
    ms: int | None = None
    data: Any | None = None
    message: str | None = None
    kind: FailureKind | None = None
    completed: int | None = None
    total: int | None = None

    @model_validator(mode="after")
    def _failed_needs_message(self) -> PipelineEvent:
        if self.status is EventStatus.FAILED and not self.message:
            raise ValueError("a failed event needs a message")
        return self


class PlanResponse(BaseModel):
    """Non-streamed variant of the same run (used by the evals and the tests)."""

    status: Literal["done", "stopped"]
    plan: CampaignPlan | None = None
    stopped: StopResult | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    mode: ExecutionMode
    version: str
