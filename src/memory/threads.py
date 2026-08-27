"""Thread index and user deletion (SPEC §10 right-to-be-forgotten).

The checkpointer is keyed by `thread_id` alone -- checkpoint metadata carries no
`user_id`, so there is no way to ask it "which threads belong to this user?".
Deleting a user therefore needs an index, kept in the Store under
`(user_id, "threads")`.

This is NOT conversation history in the Store (which would violate the memory
boundaries in .claude/references/memory-placement.md): it holds thread *ids*, no
message content. It exists so that deletion can be complete and verifiable, which
is the whole point of §10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from src.memory.facts import FACTS_NS, PREFERENCES_NS

THREADS_NS = "threads"

# Every namespace a user's data can land in. Adding a third place user data goes
# means adding it here, or deletion silently becomes partial.
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
    """Delete everything belonging to a user, across BOTH memory systems.

    1. every checkpoint for every thread the user owns, and
    2. every Store namespace scoped to that user.

    Other users' data is untouched: every key is reached through a namespace built
    from this `user_id`.
    """
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
