"""Thread index and user deletion (SPEC §10)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from src.memory.facts import FACTS_NS, PREFERENCES_NS

THREADS_NS = "threads"

# Every namespace user data lands in. Miss one and deletion becomes partial.
USER_NAMESPACES = (FACTS_NS, PREFERENCES_NS, THREADS_NS)

_PAGE = 1000


@dataclass(frozen=True)
class DeletionReport:
    """What was actually removed, so callers can verify rather than trust."""

    user_id: str
    threads_deleted: int
    facts_deleted: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "threads_deleted": self.threads_deleted,
            "facts_deleted": self.facts_deleted,
        }


async def register_thread(store: BaseStore, user_id: str, thread_id: str) -> None:
    """Idempotent: the same thread re-registers onto the same key every turn."""
    await store.aput((user_id, THREADS_NS), thread_id, {"thread_id": thread_id})


async def list_threads(store: BaseStore, user_id: str) -> list[str]:
    items = await store.asearch((user_id, THREADS_NS), limit=_PAGE)
    return sorted(str(item.value["thread_id"]) for item in items)


async def forget_user(
    store: BaseStore, checkpointer: BaseCheckpointSaver[Any], user_id: str
) -> DeletionReport:
    """Delete every checkpoint and Store entry belonging to `user_id`."""
    threads = await list_threads(store, user_id)
    for thread_id in threads:
        await checkpointer.adelete_thread(thread_id)

    facts_deleted = 0
    for namespace in USER_NAMESPACES:
        items = await store.asearch((user_id, namespace), limit=_PAGE)
        for item in items:
            await store.adelete((user_id, namespace), item.key)
            if namespace != THREADS_NS:
                facts_deleted += 1

    return DeletionReport(
        user_id=user_id, threads_deleted=len(threads), facts_deleted=facts_deleted
    )
