"""The contract the pipeline programs against, and its OpenAI Agents SDK implementation.

A StageExecutor answers the four judgement questions (what is this advertiser, which publishers
fit, which personas, what should the copy say) plus the optional narrative summary. The pipeline
never touches the SDK directly, which is what lets the same pipeline run deterministically
without an API key (agents/heuristic.py) and be tested end to end."""

import logging
from typing import Protocol

from app.agents.factory import AgentFactory
from app.agents.runner import StageRun, StructuredRunner
from app.domain.catalog import CatalogRepository
from app.domain.guards import AssessmentGuard
from app.enums import BrandAttribute, ExecutionMode, ProductCategory, Stage
from app.prompts.registry import PromptRegistry
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
from app.services.cache import StageCache

logger = logging.getLogger(__name__)


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
    """Each question becomes one structured-output agent call: render the prompt, build the
    agent, run it through the StructuredRunner (validation + one retry), cache the answer."""

    mode = ExecutionMode.LLM

    def __init__(self, factory: AgentFactory, runner: StructuredRunner, prompts: PromptRegistry,
                 catalog: CatalogRepository, guard: AssessmentGuard, cache: StageCache) -> None:
        self._factory = factory
        self._runner = runner
        self._prompts = prompts
        self._catalog = catalog
        self._guard = guard
        self._cache = cache

    async def intake(self, description: str) -> StageRun[AdvertiserBrief]:
        prompt = self._prompts.render(
            "intake",
            categories=", ".join(c.value for c in ProductCategory),
            attributes=", ".join(a.value for a in BrandAttribute),
            description=description,
        )
        return await self._run(Stage.INTAKE, prompt, AdvertiserBrief, key_payload=description)

    async def match(self, brief: AdvertiserBrief, signals: list[FitSignals]) -> StageRun[MatchOutput]:
        prompt = self._prompts.render(
            "match_publishers",
            catalog=self._catalog.compact_publishers(),
            brief=brief.model_dump(mode="json"),
            signals=[s.model_dump(mode="json") for s in signals],
            publisher_count=str(len(signals)),
        )
        return await self._run(Stage.MATCH, prompt, MatchOutput,
                               check=self._guard.validation_errors, key_payload=brief.model_dump(mode="json"))

    async def select_personas(self, brief: AdvertiserBrief, recommended: list[PublisherAssessment],
                              persona_cap: int) -> StageRun[PersonaSelectionDraft]:
        prompt = self._prompts.render(
            "select_personas",
            personas=self._catalog.compact_personas(),
            persona_cap=str(persona_cap),
            brief=brief.model_dump(mode="json"),
            recommended=[_recommended_view(r) for r in recommended],
        )
        payload = {"brief": brief.model_dump(mode="json"), "cap": persona_cap,
                   "recommended": [r.publisher_id for r in recommended]}
        return await self._run(Stage.PERSONAS, prompt, PersonaSelectionDraft,
                               check=lambda out: self._persona_errors(out, persona_cap),
                               key_payload=payload)

    async def write_creative(self, brief: AdvertiserBrief, pick: PersonaPickDraft,
                             persona: ShopperPersona, target_publishers: list[str],
                             feedback: list[str]) -> StageRun[CreativeDraft]:
        feedback_block = (self._prompts.render_fragment(
            "lint_retry", issues="\n".join(f"- {i}" for i in feedback)) if feedback else "")
        prompt = self._prompts.render(
            "write_creative",
            brief=brief.model_dump(mode="json"),
            persona=persona.model_dump(mode="json"),
            angle=pick.angle,
            watchouts="; ".join(pick.watchouts) or "none",
            target_publishers=", ".join(target_publishers) or "any recommended publisher",
            feedback=feedback_block,
        )
        payload = {"brief": brief.model_dump(mode="json"), "persona": persona.id,
                   "angle": pick.angle, "feedback": feedback}
        return await self._run(Stage.CREATIVE, prompt, CreativeDraft, key_payload=payload)

    async def summarize(self, plan: dict) -> StageRun[CampaignSummary]:
        prompt = self._prompts.render("campaign_summary", plan=plan)
        return await self._run(Stage.SUMMARY, prompt, CampaignSummary, key_payload=plan)

    # ---- plumbing ----
    async def _run(self, stage, prompt, output_type, check=None, key_payload=None) -> StageRun:
        spec = self._factory.spec(stage)
        key = self._cache.key(stage, prompt.version, spec.model, spec.reasoning_effort, key_payload)
        cached = self._cache.get(key, output_type)
        if cached is not None:
            logger.info("[EXECUTOR] stage=%s served from cache", stage)
            return cached
        agent = self._factory.build(stage, prompt.instructions or "", output_type)
        run = await self._runner.run(stage, agent, prompt, output_type, check)
        self._cache.put(key, run)
        return run

    def _persona_errors(self, out: PersonaSelectionDraft, cap: int) -> list[str]:
        errors = []
        ids = [p.persona_id for p in out.selected]
        unknown = [i for i in ids if not self._catalog.has_persona(i)]
        if unknown:
            errors.append(f"unknown persona ids: {', '.join(unknown)}")
        if len(set(ids)) != len(ids):
            errors.append("a persona was selected twice")
        if not 1 <= len(ids) <= cap:
            errors.append(f"select between 1 and {cap} personas (got {len(ids)})")
        return errors


def _recommended_view(assessment: PublisherAssessment) -> dict:
    return {"publisher_id": assessment.publisher_id, "name": assessment.publisher_name,
            "score": assessment.score, "reasons": assessment.reasons}
