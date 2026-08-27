"""The critical memory tests (CLAUDE.md Testing).

Four properties in dependency order: continuity, isolation, durability, and that
none of it depends on the client resending history.

Parametrized over backends. In-memory is here for a fast signal only -- it cannot
fail the durability test, so it never counts as passing coverage on its own.
SQLite is the real backend for M2; Postgres joins the parameter list in M4.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from langgraph.graph.state import CompiledStateGraph

from src.config import Context, Settings
from src.graph.build import build_graph
from src.graph.state import ChatState
from src.memory.checkpointer import checkpointer_scope
from src.memory.store import store_scope

Graph = CompiledStateGraph[ChatState, Context, ChatState, ChatState]

CONTEXT = Context(user_id="u-1")

# Thread-scoped probe: a conversational topic belongs to the checkpointer and is
# deliberately NOT extracted as a durable fact, so it cannot cross threads via
# the Store. Using "My name is Alice" here would test the wrong system -- that IS
# a durable fact, so from M3 on it legitimately reaches other threads.
# See .claude/references/memory-placement.md.
STATEMENT = "We were discussing invoice 42."
QUESTION = "What were we discussing?"


def turn(text: str) -> dict[str, object]:
    """The client sends ONLY the new message."""
    return {"messages": [{"role": "user", "content": text}], "question": text}


POSTGRES_URI = os.environ.get("POSTGRES_TEST_URI", "")

requires_postgres = pytest.mark.skipif(
    not POSTGRES_URI, reason="set POSTGRES_TEST_URI to run the Postgres memory tests"
)


def settings_for(backend: str, tmp_path: Path) -> Settings:
    if backend == "memory":
        return Settings(env="dev", llm_provider="fake")
    if backend == "sqlite":
        return Settings(env="local", llm_provider="fake", sqlite_path=str(tmp_path / "cp.db"))
    if backend == "postgres":
        return Settings(env="prod", llm_provider="fake", database_url=POSTGRES_URI)
    raise AssertionError(f"unknown backend {backend!r}")


def thread_id(label: str) -> str:
    """Unique per run: Postgres persists between runs, so fixed ids would collide."""
    return f"t-{label}-{uuid.uuid4().hex[:12]}"


@asynccontextmanager
async def graph_for(settings: Settings) -> AsyncIterator[Graph]:
    """Build the graph inside the saver's scope -- the connection closes on exit."""
    async with checkpointer_scope(settings) as checkpointer, store_scope(settings) as store:
        yield build_graph(checkpointer=checkpointer, store=store, settings=settings)


# In-memory is a fast signal only -- it cannot fail durability, so it never
# counts as passing coverage on its own. CLAUDE.md requires SQLite AND Postgres.
BACKENDS = [
    "memory",
    "sqlite",
    pytest.param("postgres", marks=[pytest.mark.postgres, requires_postgres]),
]


@pytest.mark.parametrize("backend", BACKENDS)
async def test_thread_continuity(backend: str, tmp_path: Path) -> None:
    """Same thread_id: the second turn sees the first. The canonical test."""
    async with graph_for(settings_for(backend, tmp_path)) as graph:
        config = {"configurable": {"thread_id": thread_id("continuity")}}

        await graph.ainvoke(turn(STATEMENT), config, context=CONTEXT)
        out = await graph.ainvoke(turn(QUESTION), config, context=CONTEXT)

    assert "invoice 42" in out["answer"].lower()


@pytest.mark.parametrize("backend", BACKENDS)
async def test_thread_isolation(backend: str, tmp_path: Path) -> None:
    """Different thread_id: conversation state does not leak, even for one user."""
    async with graph_for(settings_for(backend, tmp_path)) as graph:
        await graph.ainvoke(
            turn(STATEMENT), {"configurable": {"thread_id": thread_id("a")}}, context=CONTEXT
        )
        out = await graph.ainvoke(
            turn(QUESTION), {"configurable": {"thread_id": thread_id("b")}}, context=CONTEXT
        )

    assert "invoice 42" not in out["answer"].lower()


@pytest.mark.parametrize("backend", BACKENDS)
async def test_history_accumulates_without_the_client_resending_it(
    backend: str, tmp_path: Path
) -> None:
    async with graph_for(settings_for(backend, tmp_path)) as graph:
        config = {"configurable": {"thread_id": thread_id("accumulate")}}

        await graph.ainvoke(turn(STATEMENT), config, context=CONTEXT)
        await graph.ainvoke(turn(QUESTION), config, context=CONTEXT)
        snapshot = await graph.aget_state(config)

    # 2 turns x (human + ai); each invoke contributed only its own new message.
    assert len(snapshot.values["messages"]) == 4
    assert snapshot.values["messages"][0].content == STATEMENT


DURABLE_BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=[pytest.mark.postgres, requires_postgres]),
]


@pytest.mark.parametrize("backend", DURABLE_BACKENDS)
async def test_durability_across_a_process_restart(backend: str, tmp_path: Path) -> None:
    """A NEW saver instance against the same backing store still sees the thread.

    This is the one in-memory can never pass, which is why it is excluded.
    """
    settings = settings_for(backend, tmp_path)
    config = {"configurable": {"thread_id": thread_id("durable")}}

    async with graph_for(settings) as graph_a:
        await graph_a.ainvoke(turn(STATEMENT), config, context=CONTEXT)

    # Scope closed: connection gone, exactly as after a process restart.
    async with graph_for(settings) as graph_b:
        snapshot = await graph_b.aget_state(config)
        assert snapshot.values["messages"], "checkpoint did not survive the restart"

        out = await graph_b.ainvoke(turn(QUESTION), config, context=CONTEXT)

    assert "invoice 42" in out["answer"].lower()


async def test_sqlite_file_is_actually_written(tmp_path: Path) -> None:
    settings = settings_for("sqlite", tmp_path)
    async with graph_for(settings) as graph:
        await graph.ainvoke(
            turn(STATEMENT), {"configurable": {"thread_id": thread_id("file")}}, context=CONTEXT
        )

    # Blocking stat in a test assertion, not on the request path.
    assert Path(settings.sqlite_path).exists()  # noqa: ASYNC240
