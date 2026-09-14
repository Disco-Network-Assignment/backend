"""Session memory against a real Postgres. Skipped unless DISCO_TEST_DATABASE_URL is set
(for example the docker-compose instance: postgresql+asyncpg://disco:disco@localhost:5433/disco),
so the default suite stays hermetic."""

import os
import uuid

import pytest

from app.agents.memory import SessionStore, _without_password

DATABASE_URL = os.environ.get("DISCO_TEST_DATABASE_URL")
needs_postgres = pytest.mark.skipif(not DATABASE_URL, reason="DISCO_TEST_DATABASE_URL not set")


def test_password_is_hidden_in_logs():
    assert _without_password("postgresql+asyncpg://disco:secret@db:5432/disco") == \
        "postgresql+asyncpg://disco:***@db:5432/disco"
    assert _without_password("sqlite://") == "sqlite://"


@needs_postgres
async def test_items_survive_across_runs_of_the_same_session():
    store = SessionStore(DATABASE_URL)
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
        await store.close()
