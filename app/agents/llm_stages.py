"""The contract the pipeline programs against, and its OpenAI Agents SDK implementation.

A StageExecutor answers the judgement questions: what is this advertiser, which publishers fit,
which personas, what should the copy say, plus the optional narrative summary. The pipeline
never touches the SDK directly, which is what lets the tests drive it with a scripted executor.

How the SDK is used per stage:
- intake: a triage agent that HANDS OFF to a brief-writer agent or a clarify agent, with
  SESSION memory so a refined description builds on earlier turns
- match: the matcher calls the `fit_signals` TOOL for deterministic evidence per publisher
- personas: the strategist calls `audience_overlap` to ground best_publishers
- creatives: one copywriter run per persona, each calling `check_creative` to review its own
  draft before finalising (the lint rules as a tool, not an outer retry loop)
- summary: optionally given the hosted, sandboxed CodeInterpreterTool for arithmetic
All runs share one local CONTEXT object (agents/context.py)."""

from typing import Protocol

from agents import CodeInterpreterTool, RunContextWrapper, SQLiteSession, handoff
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from app.agents.context import RunContext
from app.agents.openai_agent import AgentFactory, StageRun, StructuredRunner
from app.agents.tools import audience_overlap, check_creative, fit_signals
from app.domain.catalog import CatalogRepository
from app.domain.creative_checks import CreativeLinter
from app.domain.fit_signals import SignalCalculator
from app.domain.guardrails import AssessmentGuard
from app.enums import BrandAttribute, ProductCategory, Stage
from app.prompts.loader import PromptLoader
from app.schemas import (
    AdvertiserBrief,
    CampaignSummary,
    ClarificationRequest,
    CreativeDraft,
    HandoffReason,
    MatchOutput,
    PersonaPickDraft,
    PersonaSelectionDraft,
    ShopperPersona,
)
from app.settings import Settings


class StageExecutor(Protocol):
    def new_context(self, description: str) -> RunContext: ...

    async def intake(self, ctx: RunContext, session_id: str | None) -> StageRun: ...

    async def match(self, ctx: RunContext) -> StageRun: ...

    async def select_personas(self, ctx: RunContext, persona_cap: int) -> StageRun: ...

    async def write_creative(self, ctx: RunContext, pick: PersonaPickDraft, persona: ShopperPersona,
                             target_publishers: list[str]) -> StageRun: ...

    async def summarize(self, ctx: RunContext, plan: dict) -> StageRun: ...


class LlmStageExecutor:
    def __init__(self, settings: Settings, factory: AgentFactory, runner: StructuredRunner,
                 prompts: PromptLoader, catalog: CatalogRepository, guard: AssessmentGuard,
                 signals: SignalCalculator, linter: CreativeLinter) -> None:
        self._settings = settings
        self._factory = factory
        self._runner = runner
        self._prompts = prompts
        self._catalog = catalog
        self._guard = guard
        self._signals = signals
        self._linter = linter

    def new_context(self, description: str) -> RunContext:
        return RunContext(catalog=self._catalog, signals=self._signals, linter=self._linter,
                          description=description)

    # ---------------------------------------------------------------- stage 1: triage + handoffs
    async def intake(self, ctx: RunContext, session_id: str | None) -> StageRun:
        """Triage decides who answers: the brief writer (a real business) or the clarify agent
        (nothing to plan with). The handoff carries a reason into the run context."""
        brief_prompt = self._prompts.render(
            "intake", categories=", ".join(c.value for c in ProductCategory),
            attributes=", ".join(a.value for a in BrandAttribute))
        brief_writer = self._factory.build(Stage.INTAKE, "brief_writer", brief_prompt.instructions or "",
                                           AdvertiserBrief)
        clarifier = self._factory.build(Stage.INTAKE, "clarifier",
                                        self._prompts.render("clarify").instructions or "", ClarificationRequest)

        async def on_handoff(wrapper: RunContextWrapper[RunContext], reason: HandoffReason) -> None:
            wrapper.context.handoff_reason = reason.reason

        triage_prompt = self._prompts.render("triage")
        triage = self._factory.build(
            Stage.INTAKE, "triage",
            prompt_with_handoff_instructions(triage_prompt.instructions or ""),
            handoffs=[
                handoff(brief_writer, tool_name_override="transfer_to_brief_writer",
                        tool_description_override="The text describes a real business; write the brief.",
                        on_handoff=on_handoff, input_type=HandoffReason),
                handoff(clarifier, tool_name_override="transfer_to_clarifier",
                        tool_description_override="There is nothing to plan with; ask what is needed.",
                        on_handoff=on_handoff, input_type=HandoffReason),
            ])
        session = SQLiteSession(session_id, str(self._settings.sessions_db)) if session_id else None
        user_input = self._prompts.render("triage").input.replace("{{description}}", ctx.description)
        return await self._runner.run(Stage.INTAKE, triage, user_input, (AdvertiserBrief, ClarificationRequest),
                                      ctx, prompt_version=triage_prompt.version, session=session, max_turns=4)

    # ---------------------------------------------------------------- stage 2: matcher + tool
    async def match(self, ctx: RunContext) -> StageRun:
        assert ctx.brief is not None
        prompt = self._prompts.render(
            "match_publishers", catalog=self._catalog.publishers_as_dicts(),
            brief=ctx.brief.model_dump(mode="json"), publisher_count=str(len(self._catalog.publishers)))
        matcher = self._factory.build(Stage.MATCH, "publisher_matcher", prompt.instructions or "",
                                      MatchOutput, tools=[fit_signals])
        return await self._runner.run(Stage.MATCH, matcher, prompt.input, (MatchOutput,), ctx,
                                      prompt_version=prompt.version, check=self._guard.validation_errors)

    # ---------------------------------------------------------------- stage 3: personas + tool
    async def select_personas(self, ctx: RunContext, persona_cap: int) -> StageRun:
        assert ctx.brief is not None
        prompt = self._prompts.render(
            "select_personas", personas=self._catalog.personas_as_dicts(), persona_cap=str(persona_cap),
            brief=ctx.brief.model_dump(mode="json"),
            recommended=[{"publisher_id": r.publisher_id, "name": r.publisher_name, "score": r.score,
                          "reasons": r.reasons} for r in ctx.recommended])
        strategist = self._factory.build(Stage.PERSONAS, "persona_strategist", prompt.instructions or "",
                                         PersonaSelectionDraft, tools=[audience_overlap])
        return await self._runner.run(Stage.PERSONAS, strategist, prompt.input, (PersonaSelectionDraft,), ctx,
                                      prompt_version=prompt.version,
                                      check=lambda out: self._persona_errors(out, persona_cap))

    # ---------------------------------------------------------------- stage 4: copywriter + tool
    async def write_creative(self, ctx: RunContext, pick: PersonaPickDraft, persona: ShopperPersona,
                             target_publishers: list[str]) -> StageRun:
        assert ctx.brief is not None
        ctx.persona = persona
        prompt = self._prompts.render(
            "write_creative", brief=ctx.brief.model_dump(mode="json"), persona=persona.model_dump(mode="json"),
            angle=pick.angle, watchouts="; ".join(pick.watchouts) or "none",
            target_publishers=", ".join(target_publishers) or "any recommended publisher")
        copywriter = self._factory.build(Stage.CREATIVE, f"copywriter_{persona.id}", prompt.instructions or "",
                                         CreativeDraft, tools=[check_creative])
        return await self._runner.run(Stage.CREATIVE, copywriter, prompt.input, (CreativeDraft,), ctx,
                                      prompt_version=prompt.version)

    # ---------------------------------------------------------------- stage 5: summary (+ sandboxed python)
    async def summarize(self, ctx: RunContext, plan: dict) -> StageRun:
        prompt = self._prompts.render("campaign_summary", plan=plan)
        tools = [CodeInterpreterTool()] if self._settings.code_interpreter_enabled else []
        summariser = self._factory.build(Stage.SUMMARY, "strategy_summariser", prompt.instructions or "",
                                         CampaignSummary, tools=tools)
        return await self._runner.run(Stage.SUMMARY, summariser, prompt.input, (CampaignSummary,), ctx,
                                      prompt_version=prompt.version)

    def _persona_errors(self, out: PersonaSelectionDraft, cap: int) -> list[str]:
        ids = [p.persona_id for p in out.selected]
        errors = []
        unknown = [i for i in ids if not self._catalog.has_persona(i)]
        if unknown:
            errors.append(f"unknown persona ids: {', '.join(unknown)}")
        if len(set(ids)) != len(ids):
            errors.append("a persona was selected twice")
        if not 1 <= len(ids) <= cap:
            errors.append(f"select between 1 and {cap} personas (got {len(ids)})")
        return errors
