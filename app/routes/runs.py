from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_run_store
from app.runs import DEFAULT_LIST_LIMIT, MAX_LIST_LIMIT, RunStore
from app.schemas import RunRecord, RunSummary

router = APIRouter(prefix="/api/runs", tags=["runs"])

Store = Annotated[RunStore, Depends(get_run_store)]


@router.get("", response_model=list[RunSummary])
async def list_runs(store: Store, limit: int = Query(DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT)) -> list[RunSummary]:
    """The most recent runs, newest first, without their plans."""
    return await store.list(limit)


@router.get("/{run_id}", response_model=RunRecord)
async def get_run(run_id: str, store: Store) -> RunRecord:
    """One stored run with its plan, stop result or error."""
    record = await store.get(run_id)
    if record is None:
        raise HTTPException(404, f"no run with id '{run_id}'")
    return record
