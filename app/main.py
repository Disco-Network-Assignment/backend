import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.dependencies import NO_KEY_MESSAGE, get_catalog, get_database, get_prompts
from app.routes import examples, plan, runs
from app.schemas import HealthResponse
from app.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # fail fast on a broken data pack or prompt file
    catalog_repo = get_catalog()
    prompts = get_prompts()
    logger.info("[APP] %s · prompts=%s · llm_configured=%s", catalog_repo, prompts.names,
                settings().llm_configured)
    if not settings().llm_configured:
        logger.warning("[APP] %s", NO_KEY_MESSAGE)
    # the run history table comes from the Alembic migrations (`alembic upgrade head`), not from here
    yield
    await get_database().close()
    logger.info("[APP] shutting down")


app = FastAPI(title="disco-backend", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[*settings().frontend_origins, "http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(plan.router)
app.include_router(examples.router)
app.include_router(runs.router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(llm_configured=settings().llm_configured, version=__version__)
