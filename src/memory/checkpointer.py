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

from src.config import Settings


@asynccontextmanager
async def checkpointer_scope(settings: Settings) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    if settings.env == "dev":
        yield InMemorySaver()
        return

    raise NotImplementedError(
        f"checkpointer backend for ENV={settings.env!r} is not wired yet "
        "(SQLite lands in M2, Postgres in M4)"
    )
