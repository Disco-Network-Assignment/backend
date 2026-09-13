"""The contract the pipeline programs against, and its OpenAI Agents SDK implementation.

A StageExecutor answers the judgement questions: what is this advertiser, which publishers fit,
which personas, what should the copy say, plus the optional summary. The pipeline never touches
the SDK directly, which is what lets the same pipeline run without a key (agents/heuristic_stages.py)
and be tested end to end."""

from typing import Protocol

from app.agents.openai_agent import AgentFactory, StageRun, StructuredRunner
from app.domain.catalog import CatalogRepository
from app.domain.guardrails import AssessmentGuard
from app.enums import BrandAttribute, ExecutionMode, ProductCategory, Stage
from app.prompts.loader import PromptLoader
from app.schemas import (
    AdvertiserBrief,
    CampaignSummary,
    CreativeDraft,
    FitSignals,
    MatchOutput,
    PersonaPickDraft,
    PersonaSelectionDraft,
    PublisherAssessment,
    ShopperPersona,
)


class StageExecutor(Protocol):
    @property
    def mode(self) -> ExecutionMode: ...

    async def intake(self, description: str) -> StageRun[AdvertiserBrief]: ...

    async def match(self, brief: AdvertiserBrief,
                    signals: list[FitSignals]) -> StageRun[MatchOutput]: ...

    async def select_personas(self, brief: AdvertiserBrief, recommended: list[PublisherAssessment],
                              persona_cap: int) -> StageRun[PersonaSelectionDraft]: ...

    async def write_creative(self, brief: AdvertiserBrief, pick: PersonaPickDraft,
                             persona: ShopperPersona, target_publishers: list[str],
                             feedback: list[str]) -> StageRun[CreativeDraft]: ...

    async def summarize(self, plan: dict) -> StageRun[CampaignSummary]: ...


class LlmStageExecutor:
    """Each question is one structured-output agent call: render the prompt, build the agent,
    run it through the StructuredRunner (validation + one retry)."""

    mode = ExecutionMode.LLM

    def __init__(self, factory: AgentFactory, runner: StructuredRunner, prompts: PromptLoader,
                 catalog: CatalogRepository, guard: AssessmentGuard) -> None:
        self._factory = factory
        self._runner = runner
        self._prompts = prompts
        self._catalog = catalog
        self._guard = guard

    async def intake(self, description: str) -> StageRun[AdvertiserBrief]:
        prompt = self._prompts.render(
            "intake",
            categories=", ".join(c.value for c in ProductCategory),
            attributes=", ".join(a.value for a in BrandAttribute),
            description=description)
        return await self._run(Stage.INTAKE, prompt, AdvertiserBrief)

    async def match(self, brief: AdvertiserBrief, signals: list[FitSignals]) -> StageRun[MatchOutput]:
        prompt = self._prompts.render(
            "match_publishers",
            catalog=self._catalog.publishers_as_dicts(),
            brief=brief.model_dump(mode="json"),
            signals=[s.model_dump(mode="json") for s in signals],
            publisher_count=str(len(signals)))
        return await self._run(Stage.MATCH, prompt, MatchOutput, check=self._guard.validation_errors)

    async def select_personas(self, brief: AdvertiserBrief, recommended: list[PublisherAssessment],
                              persona_cap: int) -> StageRun[PersonaSelectionDraft]:
        prompt = self._prompts.render(
            "select_personas",
            personas=self._catalog.personas_as_dicts(),
            persona_cap=str(persona_cap),
            brief=brief.model_dump(mode="json"),
            recommended=[{"publisher_id": r.publisher_id, "name": r.publisher_name,
                          "score": r.score, "reasons": r.reasons} for r in recommended])
        return await self._run(Stage.PERSONAS, prompt, PersonaSelectionDraft,
                               check=lambda out: self._persona_errors(out, persona_cap))

    async def write_creative(self, brief: AdvertiserBrief, pick: PersonaPickDraft,
                             persona: ShopperPersona, target_publishers: list[str],
                             feedback: list[str]) -> StageRun[CreativeDraft]:
        feedback_block = ""
        if feedback:
            feedback_block = self._prompts.render_fragment(
                "lint_retry", issues="\n".join(f"- {i}" for i in feedback))
        prompt = self._prompts.render(
            "write_creative",
            brief=brief.model_dump(mode="json"),
            persona=persona.model_dump(mode="json"),
            angle=pick.angle,
            watchouts="; ".join(pick.watchouts) or "none",
            target_publishers=", ".join(target_publishers) or "any recommended publisher",
            feedback=feedback_block)
        return await self._run(Stage.CREATIVE, prompt, CreativeDraft)

    async def summarize(self, plan: dict) -> StageRun[CampaignSummary]:
        prompt = self._prompts.render("campaign_summary", plan=plan)
        return await self._run(Stage.SUMMARY, prompt, CampaignSummary)

    async def _run(self, stage, prompt, output_type, check=None) -> StageRun:
        agent = self._factory.build(stage, prompt.instructions or "", output_type)
        return await self._runner.run(stage, agent, prompt, output_type, check)

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
