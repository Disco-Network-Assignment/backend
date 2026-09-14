"""The contract the pipeline programs against, and its OpenAI Agents SDK implementation.

A StageExecutor answers the judgement questions: what is this advertiser, which publishers
fit, which personas, what should the copy say, plus the optional narrative summary. The
pipeline never touches the SDK directly, which is what lets the tests drive it with a
scripted executor.

How the SDK is used per stage:

    intake     a triage agent HANDS OFF to a brief-writer agent or a clarify agent, with
               SESSION memory so a refined description builds on earlier turns
    match      the matcher calls the `fit_signals` TOOL for deterministic evidence per publisher
    personas   the strategist calls `audience_overlap` to ground best_publishers
    creatives  one copywriter per persona, each calling `check_creative` on its own draft
    summary    optionally given the hosted, sandboxed CodeInterpreterTool for arithmetic

All runs share one local CONTEXT object (agents/context.py).
"""

from typing import Protocol

from agents import CodeInterpreterTool, RunContextWrapper, SQLiteSession, handoff
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from app.agents.context import RunContext
from app.agents.openai_agent import AgentFactory, StageRun, StructuredRunner
from app.agents.tools import audience_overlap, check_creative, fit_signals
from app.domain.catalog import CatalogRepository
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INTAKE_MAX_TURNS = 4   # triage -> one handoff -> the specialist's answer; more means it is lost

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class StageExecutor(Protocol):
    def new_context(self, description: str) -> RunContext: ...

    async def intake(self, ctx: RunContext, session_id: str | None) -> StageRun: ...

    async def match(self, ctx: RunContext) -> StageRun: ...

    async def select_personas(self, ctx: RunContext, persona_cap: int) -> StageRun: ...

    async def write_creative(self, ctx: RunContext, pick: PersonaPickDraft, persona: ShopperPersona,
                             target_publishers: list[str]) -> StageRun: ...

    async def summarize(self, ctx: RunContext, plan: dict) -> StageRun: ...


# ---------------------------------------------------------------------------
# SDK implementation
# ---------------------------------------------------------------------------


class LlmStageExecutor:
    def __init__(self, settings: Settings, factory: AgentFactory, runner: StructuredRunner,
                 prompts: PromptLoader, catalog: CatalogRepository, guard: AssessmentGuard,
                 signals: SignalCalculator):
        self.settings = settings
        self.factory = factory
        self.runner = runner
        self.prompts = prompts
        self.catalog = catalog
        self.guard = guard
        self.signals = signals

    def new_context(self, description: str) -> RunContext:
        return RunContext(catalog=self.catalog, signals=self.signals, description=description)

    # ---- stage 1: triage + handoffs ----

    async def intake(self, ctx: RunContext, session_id: str | None) -> StageRun:
        """
        Triage decides who answers: the brief writer (a real business) or the clarifier
        (nothing to plan with). Each handoff carries a one-sentence reason into the run context.
        """
        # the two specialists triage can hand off to
        category_names = ", ".join(category.value for category in ProductCategory)
        attribute_names = ", ".join(attribute.value for attribute in BrandAttribute)
        brief_prompt = self.prompts.render("intake", categories=category_names, attributes=attribute_names)
        brief_writer = self.factory.build(Stage.INTAKE, "brief_writer", brief_prompt.instructions,
                                          AdvertiserBrief)

        clarify_prompt = self.prompts.render("clarify")
        clarifier = self.factory.build(Stage.INTAKE, "clarifier", clarify_prompt.instructions,
                                       ClarificationRequest)

        async def on_handoff(wrapper: RunContextWrapper[RunContext], reason: HandoffReason):
            wrapper.context.handoff_reason = reason.reason

        # the triage agent itself: no output type, it must hand off
        triage_prompt = self.prompts.render("triage", description=ctx.description)
        triage = self.factory.build(
            Stage.INTAKE, "triage",
            prompt_with_handoff_instructions(triage_prompt.instructions),
            handoffs=[
                handoff(brief_writer, tool_name_override="transfer_to_brief_writer",
                        tool_description_override="The text describes a real business; write the brief.",
                        on_handoff=on_handoff, input_type=HandoffReason),
                handoff(clarifier, tool_name_override="transfer_to_clarifier",
                        tool_description_override="There is nothing to plan with; ask what is needed.",
                        on_handoff=on_handoff, input_type=HandoffReason),
            ],
        )

        # session memory: a refined description in the same browser session builds on earlier turns
        session = None
        if session_id:
            session = SQLiteSession(session_id, str(self.settings.sessions_db))

        return await self.runner.run(
            Stage.INTAKE, triage, triage_prompt.input, (AdvertiserBrief, ClarificationRequest), ctx,
            prompt_version=triage_prompt.version, session=session, max_turns=INTAKE_MAX_TURNS,
        )

    # ---- stage 2: matcher + fit_signals tool ----

    async def match(self, ctx: RunContext) -> StageRun:
        prompt = self.prompts.render(
            "match_publishers",
            catalog=self.catalog.publishers_as_dicts(),
            brief=ctx.brief.model_dump(mode="json"),
            publisher_count=str(len(self.catalog.publishers)),
        )
        matcher = self.factory.build(Stage.MATCH, "publisher_matcher", prompt.instructions, MatchOutput,
                                     tools=[fit_signals])
        return await self.runner.run(
            Stage.MATCH, matcher, prompt.input, (MatchOutput,), ctx,
            prompt_version=prompt.version, check=self.guard.validation_errors,
        )

    # ---- stage 3: persona strategist + audience_overlap tool ----

    async def select_personas(self, ctx: RunContext, persona_cap: int) -> StageRun:
        recommended = []
        for assessment in ctx.recommended:
            recommended.append({
                "publisher_id": assessment.publisher_id,
                "name": assessment.publisher_name,
                "score": assessment.score,
                "reasons": assessment.reasons,
            })
        prompt = self.prompts.render(
            "select_personas",
            personas=self.catalog.personas_as_dicts(),
            persona_cap=str(persona_cap),
            brief=ctx.brief.model_dump(mode="json"),
            recommended=recommended,
        )
        strategist = self.factory.build(Stage.PERSONAS, "persona_strategist", prompt.instructions,
                                        PersonaSelectionDraft, tools=[audience_overlap])

        def check(output: PersonaSelectionDraft) -> list[str]:
            return self._persona_errors(output, persona_cap)

        return await self.runner.run(
            Stage.PERSONAS, strategist, prompt.input, (PersonaSelectionDraft,), ctx,
            prompt_version=prompt.version, check=check,
        )

    # ---- stage 4: copywriter + check_creative tool ----

    async def write_creative(self, ctx: RunContext, pick: PersonaPickDraft, persona: ShopperPersona,
                             target_publishers: list[str]) -> StageRun:
        watchouts = "; ".join(pick.watchouts)
        if not watchouts:
            watchouts = "none"
        targets = ", ".join(target_publishers)
        if not targets:
            targets = "any recommended publisher"

        prompt = self.prompts.render(
            "write_creative",
            brief=ctx.brief.model_dump(mode="json"),
            persona=persona.model_dump(mode="json"),
            angle=pick.angle,
            watchouts=watchouts,
            target_publishers=targets,
        )
        copywriter = self.factory.build(Stage.CREATIVE, f"copywriter_{persona.id}", prompt.instructions,
                                        CreativeDraft, tools=[check_creative])
        return await self.runner.run(
            Stage.CREATIVE, copywriter, prompt.input, (CreativeDraft,), ctx, prompt_version=prompt.version,
        )

    # ---- stage 5: summariser (+ sandboxed python, opt-in) ----

    async def summarize(self, ctx: RunContext, plan: dict) -> StageRun:
        prompt = self.prompts.render("campaign_summary", plan=plan)
        tools = []
        if self.settings.code_interpreter_enabled:
            tools.append(CodeInterpreterTool())
        summariser = self.factory.build(Stage.SUMMARY, "strategy_summariser", prompt.instructions,
                                        CampaignSummary, tools=tools)
        return await self.runner.run(
            Stage.SUMMARY, summariser, prompt.input, (CampaignSummary,), ctx, prompt_version=prompt.version,
        )

    # ---- business checks the runner quotes back on a retry ----

    def _persona_errors(self, output: PersonaSelectionDraft, cap: int) -> list[str]:
        errors = []
        selected_ids = []
        for pick in output.selected:
            selected_ids.append(pick.persona_id)

        unknown = []
        for persona_id in selected_ids:
            if not self.catalog.has_persona(persona_id):
                unknown.append(persona_id)
        if unknown:
            errors.append("unknown persona ids: " + ", ".join(unknown))

        if len(set(selected_ids)) != len(selected_ids):
            errors.append("a persona was selected twice")

        if not 1 <= len(selected_ids) <= cap:
            errors.append(f"select between 1 and {cap} personas (got {len(selected_ids)})")
        return errors
