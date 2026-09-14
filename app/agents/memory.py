"""Conversation memory for the intake agents, stored in Postgres.

The OpenAI Agents SDK replays a session's earlier items into a run, so an advertiser who
refines their description in the same browser session is understood as one conversation.
The SDK's SQLAlchemySession does the storing; this module hands out a session object per
session id on the process's shared database engine.

Tables (`agent_sessions`, `agent_messages`) are created by the SDK on first use.
"""

from agents.extensions.memory import SQLAlchemySession

from app.db import Database


class SessionStore:
    def __init__(self, database: Database):
        self.database = database

    def session(self, session_id: str) -> SQLAlchemySession:
        """The memory for one browser session; the SDK reads and appends to it during a run."""
        return SQLAlchemySession(session_id, engine=self.database.engine, create_tables=True)
