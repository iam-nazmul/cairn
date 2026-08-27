"""Checkpointer factory. ENV selects the backend; nothing else branches on it."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.config import Settings


@asynccontextmanager
async def checkpointer_scope(settings: Settings) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    if settings.env == "dev":
        yield InMemorySaver()
        return

    if settings.env == "local":
        # The connection closes on exit: build AND invoke inside this scope.
        async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as saver:
            await saver.setup()  # idempotent; creates the checkpoint tables
            yield saver
        return

    if not settings.database_url:
        raise ValueError("DATABASE_URL is required when ENV=prod")

    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
        await saver.setup()  # idempotent; creates the checkpoint tables
        yield saver
