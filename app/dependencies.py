"""Composition root. Everything with state is built once here and handed to routes through
FastAPI `Depends`, so a test can swap any piece with `app.dependency_overrides`."""

import logging
from functools import lru_cache

from app.agents.heuristic_stages import HeuristicStageExecutor
from app.agents.llm_stages import LlmStageExecutor, StageExecutor
from app.agents.openai_agent import AgentFactory, StructuredRunner
from app.domain.catalog import CatalogRepository, load_catalog
from app.domain.config_builder import ConfigBuilder
from app.domain.creative_checks import CreativeLinter
from app.domain.fit_signals import SignalCalculator
from app.domain.guardrails import AssessmentGuard
from app.domain.input_policy import InputPolicy
from app.enums import ExecutionMode
from app.pipeline import CampaignPipeline
from app.prompts.loader import PromptLoader
from app.settings import Settings, settings

logger = logging.getLogger(__name__)


@lru_cache
def get_catalog() -> CatalogRepository:
    return load_catalog(settings().data_dir)


@lru_cache
def get_prompts() -> PromptLoader:
    return PromptLoader(settings().prompts_dir)


def create_pipeline(mode: ExecutionMode, config: Settings | None = None,
                    catalog: CatalogRepository | None = None,
                    prompts: PromptLoader | None = None) -> CampaignPipeline:
    """Factory for any context (routes, evals, tests)."""
    config = config or settings()
    catalog = catalog or get_catalog()
    prompts = prompts or get_prompts()
    signals = SignalCalculator(catalog)
    guard = AssessmentGuard(catalog)
    executor: StageExecutor
    if mode is ExecutionMode.HEURISTIC:
        executor = HeuristicStageExecutor(catalog, signals)
    else:
        executor = LlmStageExecutor(AgentFactory(config), StructuredRunner(config, prompts),
                                    prompts, catalog, guard)
    return CampaignPipeline(executor, catalog, signals, guard, InputPolicy(), CreativeLinter(),
                            ConfigBuilder(catalog), summary_enabled=config.summary_stage_enabled)


class PipelineProvider:
    """One pipeline per execution mode. A request may ask for LLM mode, but without a key it
    gets the heuristic one (and the trace says so)."""

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._pipelines: dict[ExecutionMode, CampaignPipeline] = {}

    @property
    def default_mode(self) -> ExecutionMode:
        return self._config.effective_mode

    def for_mode(self, requested: ExecutionMode | None) -> CampaignPipeline:
        mode = requested or self.default_mode
        if mode is ExecutionMode.LLM and not self._config.openai_api_key:
            logger.warning("[PIPELINE] LLM mode requested without OPENAI_API_KEY; using heuristic")
            mode = ExecutionMode.HEURISTIC
        if mode not in self._pipelines:
            self._pipelines[mode] = create_pipeline(mode, self._config)
        return self._pipelines[mode]


@lru_cache
def get_pipeline_provider() -> PipelineProvider:
    return PipelineProvider(settings())
