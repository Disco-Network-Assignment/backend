"""Shared fixtures. Every test is hermetic: the app runs in heuristic mode with a Settings
object that carries no API key, whatever the developer's .env says."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.dependencies import PipelineProvider, create_pipeline, get_pipeline_provider
from app.domain.catalog import CatalogRepository, load_catalog
from app.enums import ExecutionMode
from app.main import app
from app.prompts.loader import PromptLoader
from app.settings import ROOT_DIR, Settings


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return Settings(openai_api_key="", execution_mode=ExecutionMode.HEURISTIC,
                    tracing_enabled=False, _env_file=None)


@pytest.fixture(scope="session")
def catalog() -> CatalogRepository:
    return load_catalog(ROOT_DIR / "data")


@pytest.fixture(scope="session")
def prompts() -> PromptLoader:
    return PromptLoader(ROOT_DIR / "prompts")


@pytest.fixture(scope="session")
def pipeline(test_settings, catalog, prompts):
    return create_pipeline(ExecutionMode.HEURISTIC, test_settings, catalog, prompts)


@pytest.fixture
async def client(test_settings):
    app.dependency_overrides[get_pipeline_provider] = lambda: PipelineProvider(test_settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
    app.dependency_overrides.clear()
