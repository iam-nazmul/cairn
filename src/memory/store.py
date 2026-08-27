"""Store factory -- long-term, cross-thread memory scoped by `user_id` (SPEC §7.2).

Compiled in from M1 so the graph shape never changes, but the `load_memory` /
`write_memory` nodes that use it arrive in M3.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from src.config import Settings


@asynccontextmanager
async def store_scope(settings: Settings) -> AsyncIterator[BaseStore]:
    if settings.env in ("dev", "local"):
        yield InMemoryStore()
        return

    raise NotImplementedError(
        f"store backend for ENV={settings.env!r} is not wired yet (Postgres lands in M4)"
    )
