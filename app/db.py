"""The one Postgres connection pool the process shares.

Two things live in the database: the agents' conversation memory (agents/memory.py, tables
owned by the OpenAI Agents SDK) and the history of campaign runs (runs.py). Both borrow the
engine from here, so there is one pool, one URL, and one place to close it on shutdown.
"""

import logging

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

log = logging.getLogger(__name__)


class Database:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._engine = None

    @property
    def engine(self) -> AsyncEngine:
        """Opened the first time anything needs the database, not at import time."""
        if self._engine is None:
            log.info("[DB] connecting to %s", without_password(self.database_url))
            self._engine = create_async_engine(self.database_url, pool_pre_ping=True)
        return self._engine

    async def close(self):
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None


def without_password(url: str) -> str:
    """'postgresql+asyncpg://user:secret@host/db' -> 'postgresql+asyncpg://user:***@host/db'."""
    scheme, separator, rest = url.partition("://")
    if not separator or "@" not in rest:
        return url
    credentials, _, host_part = rest.rpartition("@")
    user, _, _ = credentials.partition(":")
    return f"{scheme}://{user}:***@{host_part}"
