from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_catalog
from app.domain.catalog import CatalogRepository
from app.schemas import CatalogResponse, ExampleAdvertiser

router = APIRouter(prefix="/api", tags=["catalog"])


@router.get("/catalog", response_model=CatalogResponse)
async def catalog(repo: Annotated[CatalogRepository, Depends(get_catalog)]) -> CatalogResponse:
    """The data pack as the pipeline sees it, so the UI can show publisher and persona cards."""
    return CatalogResponse(publishers=repo.publishers, personas=repo.personas)


@router.get("/examples", response_model=list[ExampleAdvertiser])
async def examples(repo: Annotated[CatalogRepository, Depends(get_catalog)]) -> list[ExampleAdvertiser]:
    """The 15 sample advertiser one-liners, for the example chips."""
    return repo.examples
