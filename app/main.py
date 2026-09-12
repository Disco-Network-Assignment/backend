import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import settings
from app.dependencies import get_catalog, get_pipeline_provider, get_prompts
from app.routes import catalog, plan
from app.schemas import HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # fail fast on a broken data pack or prompt file, and warm the default pipeline so the
    # first request does not pay for it
    catalog_repo = get_catalog()
    prompts = get_prompts()
    provider = get_pipeline_provider()
    provider.for_mode(None)
    logger.info("[APP] %s · prompts=%s · mode=%s", catalog_repo, prompts.names, provider.default_mode)
    yield
    logger.info("[APP] shutting down")


app = FastAPI(title="disco-backend", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[*settings().frontend_origins, "http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(plan.router)
app.include_router(catalog.router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(mode=get_pipeline_provider().default_mode, version=__version__)
