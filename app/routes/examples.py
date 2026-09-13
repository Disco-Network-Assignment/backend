from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_catalog
from app.domain.catalog import CatalogRepository
from app.schemas import ExampleAdvertiser

router = APIRouter(prefix="/api", tags=["examples"])


@router.get("/examples", response_model=list[ExampleAdvertiser])
async def examples(repo: Annotated[CatalogRepository, Depends(get_catalog)]) -> list[ExampleAdvertiser]:
    """The 15 sample advertiser one-liners, for the example chips."""
    return repo.examples
