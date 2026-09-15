"""Function tools the agents can call. Each one wraps deterministic domain code, so the model
asks for numbers instead of guessing them: fit evidence for a publisher, audience overlap for a
persona, and a length check of a draft ad.

The description the model sees for each tool is a prompt, so it lives in prompts/tools/*.md
like every other prompt; `build_tools` reads them and wraps the functions below."""

from dataclasses import dataclass

from agents import FunctionTool, RunContextWrapper, function_tool

from app.agents.context import RunContext
from app.domain.creative_checks import check_lengths
from app.domain.fit_signals import age_overlap_pct
from app.prompts.loader import PromptLoader
from app.schemas import CreativeDraft


def fit_signals(ctx: RunContextWrapper[RunContext], publisher_id: str) -> str:
    """Fit evidence for one publisher (computed once, then cached in the run context)."""
    run = ctx.context
    if run.brief is None:
        return "error: the advertiser brief is not available yet"
    if not run.catalog.has_publisher(publisher_id):
        return f"error: unknown publisher id '{publisher_id}'"
    if publisher_id not in run.fit_signals:
        run.fit_signals[publisher_id] = run.signals.compute(run.brief, run.catalog.publisher(publisher_id))
    return run.fit_signals[publisher_id].model_dump_json()


def audience_overlap(ctx: RunContextWrapper[RunContext], persona_id: str) -> str:
    """Age overlap between one persona and each recommended publisher."""
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
    if not rows:
        return "no recommended publishers"
    return "\n".join(rows)


def check_creative(ctx: RunContextWrapper[RunContext], headline: str, body: str, cta: str,
                   alt_headline: str) -> str:
    """The ad unit's length limits, as a tool the copywriter calls on its own draft."""
    run = ctx.context
    run.creative_checks += 1
    draft = CreativeDraft(headline=headline, body=body, cta=cta, alt_headline=alt_headline,
                          persona_reasoning="")
    issues = check_lengths(draft)
    if not issues:
        return "ok"
    return "\n".join(f"- {issue.message}" for issue in issues)


@dataclass(frozen=True)
class AgentTools:
    fit_signals: FunctionTool
    audience_overlap: FunctionTool
    check_creative: FunctionTool


def build_tools(prompts: PromptLoader) -> AgentTools:
    """Wrap the functions as SDK tools, with their descriptions read from prompts/tools/."""
    return AgentTools(
        fit_signals=function_tool(fit_signals, description_override=prompts.render("tool_fit_signals").input),
        audience_overlap=function_tool(audience_overlap,
                                       description_override=prompts.render("tool_audience_overlap").input),
        check_creative=function_tool(check_creative,
                                     description_override=prompts.render("tool_check_creative").input),
    )
