"""Function tools the agents can call. Each one wraps deterministic domain code, so the model
asks for numbers instead of guessing them: fit evidence for a publisher, audience overlap for a
persona, and a review of a draft ad. The docstrings become the tool descriptions the model sees."""

from agents import RunContextWrapper, function_tool

from app.agents.context import RunContext
from app.domain.creative_checks import check_lengths
from app.domain.fit_signals import age_overlap_pct
from app.schemas import CreativeDraft


@function_tool
def fit_signals(ctx: RunContextWrapper[RunContext], publisher_id: str) -> str:
    """Deterministic fit evidence for one publisher against the current advertiser: category
    overlap (1 same shelf, 0.5 adjacent, 0 none), age overlap, gender alignment, income tier vs
    price tier, price vs the publisher's average order value (aov_ratio, aov_fit), reach index,
    and a blended prior on a 0-100 scale. Call it for every publisher you are about to score;
    the publisher's free-text notes are yours to read and weigh."""
    run = ctx.context
    if run.brief is None:
        return "error: the advertiser brief is not available yet"
    if not run.catalog.has_publisher(publisher_id):
        return f"error: unknown publisher id '{publisher_id}'"
    if publisher_id not in run.fit_signals:
        run.fit_signals[publisher_id] = run.signals.compute(run.brief, run.catalog.publisher(publisher_id))
    return run.fit_signals[publisher_id].model_dump_json()


@function_tool
def audience_overlap(ctx: RunContextWrapper[RunContext], persona_id: str) -> str:
    """Age-range overlap (0-1) between one shopper persona and each recommended publisher's
    audience, so best_publishers is grounded in the catalog rather than guessed."""
    run = ctx.context
    if not run.catalog.has_persona(persona_id):
        return f"error: unknown persona id '{persona_id}'"
    persona = run.catalog.persona(persona_id)
    rows = []
    for assessment in run.recommended:
        publisher = run.catalog.publisher(assessment.publisher_id)
        overlap = age_overlap_pct(persona.age_range, publisher.audience.age_skew)
        rows.append(f"{publisher.id} ({publisher.name}): age overlap {overlap:.0%}, "
                    f"{publisher.audience.gender_split.female:.0%} female, {publisher.audience.income_tier} income")
    return "\n".join(rows) or "no recommended publishers"


@function_tool
def check_creative(ctx: RunContextWrapper[RunContext], headline: str, body: str, cta: str,
                   alt_headline: str) -> str:
    """Check a draft ad against the unit's length limits (headline 60, body 160, CTA 20
    characters). Returns 'ok' or the list of problems to fix. Call it before you finalise, and
    again after fixing anything it reported. Claims and persona fit are your own judgement."""
    run = ctx.context
    run.creative_checks += 1
    draft = CreativeDraft(headline=headline, body=body, cta=cta, alt_headline=alt_headline,
                          persona_reasoning="")
    issues = check_lengths(draft)
    return "ok" if not issues else "\n".join(f"- {issue.message}" for issue in issues)
