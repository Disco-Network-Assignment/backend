from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.dependencies import PipelineProvider, get_pipeline_provider
from app.enums import FailureKind
from app.errors import StageError
from app.pipeline import CampaignPipeline
from app.schemas import PlanRequest, PlanResponse

router = APIRouter(prefix="/api/plan", tags=["plan"])

NDJSON = "application/x-ndjson"
_STATUS_BY_KIND = {FailureKind.RATE_LIMIT: 429, FailureKind.TIMEOUT: 504, FailureKind.API: 502,
                   FailureKind.VALIDATION: 502, FailureKind.REFUSAL: 422, FailureKind.UNKNOWN: 500}


@router.post("", response_class=StreamingResponse)
async def plan_stream(body: PlanRequest,
                      provider: Annotated[PipelineProvider, Depends(get_pipeline_provider)]) -> StreamingResponse:
    """Run the pipeline and stream one JSON object per line as each stage starts, progresses,
    completes or fails. The final line is `{"stage": "done"}` with the whole CampaignPlan,
    `{"stage": "stopped"}` for input too thin to plan, or a `failed` event."""
    pipeline = provider.for_mode(body.options.mode)
    return StreamingResponse(_ndjson(pipeline, body), media_type=NDJSON,
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.post("/run", response_model=PlanResponse)
async def plan_run(body: PlanRequest,
                   provider: Annotated[PipelineProvider, Depends(get_pipeline_provider)]) -> PlanResponse:
    """The same run as one JSON document; used by the evals and anything that cannot stream."""
    pipeline = provider.for_mode(body.options.mode)
    try:
        return await pipeline.run(body)
    except StageError as e:
        raise HTTPException(_STATUS_BY_KIND.get(e.kind, 500), f"{e.stage}: {e.message}") from e


async def _ndjson(pipeline: CampaignPipeline, body: PlanRequest) -> AsyncIterator[bytes]:
    async for event in pipeline.stream(body):
        yield (event.model_dump_json(exclude_none=True) + "\n").encode("utf-8")
