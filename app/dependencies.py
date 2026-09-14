"""Composition root. Everything with state is built once here and handed to routes through
FastAPI `Depends`, so a test can swap any piece with `app.dependency_overrides`."""

from functools import lru_cache

from fastapi import HTTPException

from app.agents.llm_stages import LlmStageExecutor, StageExecutor
from app.agents.openai_agent import AgentFactory, StructuredRunner
from app.domain.catalog import CatalogRepository, load_catalog
from app.domain.config_builder import ConfigBuilder
from app.domain.fit_signals import SignalCalculator
from app.domain.guardrails import AssessmentGuard
from app.domain.input_policy import InputPolicy
from app.pipeline import CampaignPipeline
from app.prompts.loader import PromptLoader
from app.settings import Settings, settings

NO_KEY_MESSAGE = "OPENAI_API_KEY is not set; the agents cannot run. Add it to backend/.env and restart."


@lru_cache
def get_catalog() -> CatalogRepository:
    return load_catalog(settings().data_dir)


@lru_cache
def get_prompts() -> PromptLoader:
    return PromptLoader(settings().prompts_dir)


def create_pipeline(config: Settings | None = None, catalog: CatalogRepository | None = None,
                    prompts: PromptLoader | None = None,
                    executor: StageExecutor | None = None) -> CampaignPipeline:
    """Factory for any context. Routes and evals use the SDK executor; tests pass a scripted one."""
    config = config or settings()
    catalog = catalog or get_catalog()
    prompts = prompts or get_prompts()
    signals = SignalCalculator(catalog)
    guard = AssessmentGuard(catalog)
    executor = executor or LlmStageExecutor(config, AgentFactory(config), StructuredRunner(config, prompts),
                                            prompts, catalog, guard, signals)
    return CampaignPipeline(executor, catalog, signals, guard, InputPolicy(), ConfigBuilder(catalog),
                            summary_enabled=config.summary_stage_enabled)


@lru_cache
def _pipeline() -> CampaignPipeline:
    return create_pipeline()


def get_pipeline() -> CampaignPipeline:
    """Request dependency: a clear 503 instead of a provider error when the key is missing."""
    if not settings().llm_configured:
        raise HTTPException(503, NO_KEY_MESSAGE)
    return _pipeline()
