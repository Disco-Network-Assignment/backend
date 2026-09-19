"""Alembic environment: runs the migrations against the app's own database.

The URL comes from the app settings (DATABASE_URL) and the target metadata from app/runs.py, so
`alembic revision --autogenerate` compares the database with the tables the code actually uses.
The driver is asyncpg, so the migrations run through an async engine.

The OpenAI Agents SDK's session tables (`agent_sessions`, `agent_messages`) are owned and
created by the SDK; they are not in this metadata and autogenerate must leave them alone.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from app.runs import metadata
from app.settings import settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = metadata
DATABASE_URL = settings().database_url


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Only manage our own tables: a table that exists in the database but not in the metadata
    (the SDK's session tables) is never proposed for deletion."""
    if type_ == "table" and reflected and compare_to is None:
        return False
    return True


def run_migrations_offline() -> None:
    """`alembic upgrade head --sql`: print the SQL instead of running it."""
    context.configure(url=DATABASE_URL, target_metadata=target_metadata, literal_binds=True,
                      include_object=include_object, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata,
                      include_object=include_object)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(DATABASE_URL, poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
