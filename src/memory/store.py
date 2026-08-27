"""Store factory: long-term memory scoped by `user_id` (SPEC §7.2)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore

from src.config import Settings


@asynccontextmanager
async def store_scope(settings: Settings) -> AsyncIterator[BaseStore]:
    if settings.env in ("dev", "local"):
        yield InMemoryStore()
        return

    if not settings.database_url:
        raise ValueError("DATABASE_URL is required when ENV=prod")

    # The SAME database as the checkpointer (SPEC §11): one URL, one deletion path.
    async with AsyncPostgresStore.from_conn_string(settings.database_url) as store:
        await store.setup()  # idempotent; creates the store tables
        yield store
