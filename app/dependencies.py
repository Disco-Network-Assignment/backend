"""Composition root. Everything with state is built once here and handed to routes through
FastAPI `Depends`, so a test can swap any piece with `app.dependency_overrides`."""

import logging
from functools import lru_cache

from app.agents.executor import LlmStageExecutor, StageExecutor
from app.agents.factory import AgentFactory
from app.agents.heuristic import HeuristicStageExecutor
from app.agents.runner import StructuredRunner
from app.config import Settings, settings
from app.domain.catalog import CatalogRepository, load_catalog
from app.domain.guards import AssessmentGuard
from app.domain.lint import CreativeLinter
from app.domain.planner import CampaignPlanner
from app.domain.router import InputRouter
from app.domain.signals import SignalCalculator
from app.enums import ExecutionMode
from app.pipeline.orchestrator import CampaignPipeline
from app.prompts.registry import PromptRegistry
from app.services.cache import StageCache

logger = logging.getLogger(__name__)


@lru_cache
def get_catalog() -> CatalogRepository:
    return load_catalog(settings().data_dir)


@lru_cache
def get_prompts() -> PromptRegistry:
    return PromptRegistry(settings().prompts_dir)


def create_executor(mode: ExecutionMode, config: Settings, catalog: CatalogRepository,
                    prompts: PromptRegistry) -> StageExecutor:
    signals = SignalCalculator(catalog)
    if mode is ExecutionMode.HEURISTIC:
        return HeuristicStageExecutor(catalog, signals)
    return LlmStageExecutor(
        factory=AgentFactory(config),
        runner=StructuredRunner(config, prompts),
        prompts=prompts,
        catalog=catalog,
        guard=AssessmentGuard(catalog),
        cache=StageCache(enabled=config.stage_cache_enabled),
    )


def create_pipeline(mode: ExecutionMode, config: Settings | None = None,
                    catalog: CatalogRepository | None = None,
                    prompts: PromptRegistry | None = None) -> CampaignPipeline:
    """Factory for non-request contexts (evals, scripts, tests)."""
    config = config or settings()
    catalog = catalog or get_catalog()
    prompts = prompts or get_prompts()
    return CampaignPipeline(
        executor=create_executor(mode, config, catalog, prompts),
        catalog=catalog,
        signal_calculator=SignalCalculator(catalog),
        guard=AssessmentGuard(catalog),
        router=InputRouter(),
        linter=CreativeLinter(),
        planner=CampaignPlanner(catalog),
        summary_enabled=config.summary_stage_enabled,
    )


class PipelineProvider:
    """Hands out one pipeline per execution mode; a request may ask for a mode, but LLM mode
    silently becomes heuristic when no key is configured (and says so in the trace)."""

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._pipelines: dict[ExecutionMode, CampaignPipeline] = {}

    @property
    def default_mode(self) -> ExecutionMode:
        return self._config.effective_mode

    def resolve_mode(self, requested: ExecutionMode | None) -> ExecutionMode:
        mode = requested or self.default_mode
        if mode is ExecutionMode.LLM and not self._config.openai_api_key:
            logger.warning("[PIPELINE] LLM mode requested without OPENAI_API_KEY; using heuristic")
            return ExecutionMode.HEURISTIC
        return mode

    def for_mode(self, requested: ExecutionMode | None) -> CampaignPipeline:
        mode = self.resolve_mode(requested)
        if mode not in self._pipelines:
            self._pipelines[mode] = create_pipeline(mode, self._config)
        return self._pipelines[mode]


@lru_cache
def get_pipeline_provider() -> PipelineProvider:
    return PipelineProvider(settings())
