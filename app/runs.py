"""History of campaign runs, stored in Postgres.

Every run the pipeline finishes is saved once, whatever its outcome:

    done      the full CampaignPlan
    stopped   the StopResult (input too thin to plan)
    failed    the stage and message of the failure

The row also carries a few summary numbers so the history list can be shown without loading
every plan. Saving never breaks a run: if the database is unreachable the failure is logged
and the response still goes out.
"""

import logging
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, Text, select
from sqlalchemy.dialects.postgresql import JSONB, insert

from app.db import Database
from app.schemas import RunRecord, RunSummary

log = logging.getLogger(__name__)

metadata = MetaData()

campaign_runs = Table(
    "campaign_runs",
    metadata,
    Column("run_id", String(32), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("session_id", String(64)),
    Column("description", Text, nullable=False),
    Column("status", String(16), nullable=False),     # done | stopped | failed
    Column("input_quality", String(16)),
    Column("recommended", Integer, nullable=False),
    Column("personas", Integer, nullable=False),
    Column("creatives", Integer, nullable=False),
    Column("budget_usd", Float, nullable=False),
    Column("plan", JSONB),
    Column("stopped", JSONB),
    Column("error", JSONB),
)

SUMMARY_COLUMNS = [
    campaign_runs.c.run_id,
    campaign_runs.c.created_at,
    campaign_runs.c.session_id,
    campaign_runs.c.description,
    campaign_runs.c.status,
    campaign_runs.c.input_quality,
    campaign_runs.c.recommended,
    campaign_runs.c.personas,
    campaign_runs.c.creatives,
    campaign_runs.c.budget_usd,
]

DEFAULT_LIST_LIMIT = 20
MAX_LIST_LIMIT = 100


class RunStore(Protocol):
    """What the pipeline and the routes program against; tests use an in-memory version."""

    async def save(self, record: RunRecord) -> None: ...

    async def list(self, limit: int = DEFAULT_LIST_LIMIT) -> list[RunSummary]: ...

    async def get(self, run_id: str) -> RunRecord | None: ...


class PostgresRunStore:
    def __init__(self, database: Database):
        self.database = database
        self._table_ready = False

    async def ensure_table(self):
        """Create the table if it is missing. Called at startup and before the first write."""
        if self._table_ready:
            return
        async with self.database.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        self._table_ready = True

    async def save(self, record: RunRecord) -> None:
        try:
            await self.ensure_table()
            row = record.model_dump(mode="json")
            row["created_at"] = record.created_at
            # the same run id can only exist once; a retry of the save overwrites, not duplicates
            statement = insert(campaign_runs).values(**row)
            statement = statement.on_conflict_do_update(index_elements=["run_id"], set_=row)
            async with self.database.engine.begin() as connection:
                await connection.execute(statement)
        except Exception:  # noqa: BLE001 - history must never break a run
            log.exception("[RUNS] could not save run %s", record.run_id)

    async def list(self, limit: int = DEFAULT_LIST_LIMIT) -> list[RunSummary]:
        limit = max(1, min(limit, MAX_LIST_LIMIT))
        statement = select(*SUMMARY_COLUMNS).order_by(campaign_runs.c.created_at.desc()).limit(limit)
        await self.ensure_table()
        async with self.database.engine.connect() as connection:
            rows = (await connection.execute(statement)).mappings().all()
        summaries = []
        for row in rows:
            summaries.append(RunSummary.model_validate(dict(row)))
        return summaries

    async def get(self, run_id: str) -> RunRecord | None:
        statement = select(campaign_runs).where(campaign_runs.c.run_id == run_id)
        await self.ensure_table()
        async with self.database.engine.connect() as connection:
            row = (await connection.execute(statement)).mappings().first()
        if row is None:
            return None
        return RunRecord.model_validate(dict(row))


def new_record(run_id: str, description: str, session_id: str | None) -> RunRecord:
    """A record with the fields every outcome shares; the pipeline fills in the outcome."""
    return RunRecord(
        run_id=run_id, created_at=datetime.now(UTC), session_id=session_id, description=description,
        status="failed", input_quality=None, recommended=0, personas=0, creatives=0, budget_usd=0.0,
    )
