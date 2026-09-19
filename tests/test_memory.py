"""Session memory and run history against a real Postgres. Skipped unless
DISCO_TEST_DATABASE_URL is set (for example the docker-compose instance:
postgresql+asyncpg://disco:disco@localhost:5433/disco), so the default suite stays hermetic.
The run history test needs the schema: run `alembic upgrade head` against that database first."""

import os
import uuid

import pytest

from app.agents.memory import SessionStore
from app.db import Database, without_password
from app.runs import PostgresRunStore, new_record
from app.schemas import StopResult

DATABASE_URL = os.environ.get("DISCO_TEST_DATABASE_URL")
needs_postgres = pytest.mark.skipif(not DATABASE_URL, reason="DISCO_TEST_DATABASE_URL not set")


def test_password_is_hidden_in_logs():
    assert without_password("postgresql+asyncpg://disco:secret@db:5432/disco") == \
        "postgresql+asyncpg://disco:***@db:5432/disco"
    assert without_password("sqlite://") == "sqlite://"


@needs_postgres
async def test_items_survive_across_runs_of_the_same_session():
    database = Database(DATABASE_URL)
    store = SessionStore(database)
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    try:
        # first turn: the advertiser's original description is stored
        first = store.session(session_id)
        await first.add_items([{"role": "user", "content": "We sell premium dog food."}])

        # second turn: a new session object for the same id sees the earlier turn
        second = store.session(session_id)
        items = await second.get_items()
        assert [item["content"] for item in items] == ["We sell premium dog food."]

        # a different session id sees nothing
        other = store.session(f"test-{uuid.uuid4().hex[:8]}")
        assert await other.get_items() == []

        await second.clear_session()
        assert await first.get_items() == []
    finally:
        await database.close()


@needs_postgres
async def test_run_history_round_trips_through_postgres():
    database = Database(DATABASE_URL)
    store = PostgresRunStore(database)
    run_id = uuid.uuid4().hex[:12]
    try:
        record = new_record(run_id, "idk just try it", session_id="s-test")
        record.status = "stopped"
        record.stopped = StopResult(run_id=run_id, description="idk just try it", reason="too thin",
                                    clarifying_questions=["What do you sell?"], examples=[], trace=[])
        await store.save(record)
        await store.save(record)  # saving twice is an overwrite, not a duplicate

        listing = await store.list(limit=5)
        assert listing[0].run_id == run_id and listing[0].status == "stopped"

        loaded = await store.get(run_id)
        assert loaded is not None and loaded.stopped.reason == "too thin" and loaded.plan is None
        assert await store.get("missing") is None
    finally:
        await database.close()
