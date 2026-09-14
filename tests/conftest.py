"""Shared fixtures. Every test is hermetic: the pipeline runs with a scripted StageExecutor
(tests/helpers.py) instead of the OpenAI agents, and a Settings object that ignores .env."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.dependencies import create_pipeline, get_pipeline
from app.domain.catalog import CatalogRepository, load_catalog
from app.main import app
from app.prompts.loader import PromptLoader
from app.settings import ROOT_DIR, Settings
from tests.helpers import ScriptedExecutor


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return Settings(openai_api_key="test-key", tracing_enabled=False, _env_file=None)


@pytest.fixture(scope="session")
def catalog() -> CatalogRepository:
    return load_catalog(ROOT_DIR / "data")


@pytest.fixture(scope="session")
def prompts() -> PromptLoader:
    return PromptLoader(ROOT_DIR / "prompts")


@pytest.fixture
def executor(catalog) -> ScriptedExecutor:
    return ScriptedExecutor(catalog)


@pytest.fixture
def pipeline(test_settings, catalog, prompts, executor):
    return create_pipeline(test_settings, catalog, prompts, executor)


@pytest.fixture
async def client(pipeline):
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
    app.dependency_overrides.clear()
