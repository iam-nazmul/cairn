"""Checkpointer factory -- short-term, thread-scoped memory (SPEC §7.1).

`ENV` selects the backend and NOTHING outside this module (and store.py) is
allowed to branch on it: graph code is identical across all three backends, only
the instance handed to `compile()` differs.

Exposed as an async context manager because the durable savers are context
managers whose connection closes on exit -- build the graph inside the scope and
keep it there (FastAPI: the lifespan handler).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.config import Settings


@asynccontextmanager
async def checkpointer_scope(settings: Settings) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    if settings.env == "dev":
        yield InMemorySaver()
        return

    if settings.env == "local":
        # from_conn_string yields inside the context manager and closes the
        # connection on exit -- build AND invoke the graph inside this scope.
        async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as saver:
            await saver.setup()  # idempotent; creates the checkpoint tables
            yield saver
        return

    raise NotImplementedError(
        f"checkpointer backend for ENV={settings.env!r} is not wired yet (Postgres lands in M4)"
    )
