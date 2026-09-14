"""Conversation memory for the intake agents, stored in Postgres.

The OpenAI Agents SDK replays a session's earlier items into a run, so an advertiser who
refines their description in the same browser session is understood as one conversation.
The SDK's SQLAlchemySession does the storing; this module owns the one database engine the
process shares and hands out a session object per session id.

Tables (`agent_sessions`, `agent_messages`) are created by the SDK on first use.
"""

import logging

from agents.extensions.memory import SQLAlchemySession
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

log = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._engine = None

    def session(self, session_id: str) -> SQLAlchemySession:
        """The memory for one browser session; the SDK reads and appends to it during a run."""
        return SQLAlchemySession(session_id, engine=self._get_engine(), create_tables=True)

    def _get_engine(self) -> AsyncEngine:
        """One engine (and connection pool) per process, opened the first time memory is needed."""
        if self._engine is None:
            log.info("[MEMORY] connecting to %s", _without_password(self.database_url))
            self._engine = create_async_engine(self.database_url, pool_pre_ping=True)
        return self._engine

    async def close(self):
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None


def _without_password(url: str) -> str:
    """'postgresql+asyncpg://user:secret@host/db' -> 'postgresql+asyncpg://user:***@host/db'."""
    scheme, separator, rest = url.partition("://")
    if not separator or "@" not in rest:
        return url
    credentials, _, host_part = rest.rpartition("@")
    user, _, _ = credentials.partition(":")
    return f"{scheme}://{user}:***@{host_part}"
