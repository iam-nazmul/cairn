"""Shared fixtures. M1 runs on the in-memory backend; SQLite and Postgres join in M2/M4."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from src.config import Context, Mode, Settings
from src.graph.build import build_graph
from src.graph.state import ChatState


@pytest.fixture
def settings() -> Settings:
    return Settings(env="dev", llm_provider="fake")


@pytest.fixture
def graph(settings: Settings) -> CompiledStateGraph[ChatState, Context, ChatState, ChatState]:
    return build_graph(checkpointer=InMemorySaver(), store=InMemoryStore(), settings=settings)


@pytest.fixture
def context() -> Context:
    return Context(user_id="u-test")


def make_runtime(
    user_id: str = "u-test", store: BaseStore | None = None, mode: Mode = "chat"
) -> Runtime[Context]:
    """A stub Runtime for unit-testing a node or a router in isolation."""
    return Runtime(context=Context(user_id=user_id, mode=mode), store=store)


def make_state(**overrides: Any) -> ChatState:
    """A fabricated ChatState with sane defaults."""
    base: dict[str, Any] = {
        "messages": [],
        "question": "",
        "retrieved": [],
        "long_term_facts": [],
        "answer": "",
        "searches": [],
        "new_hits": 0,
    }
    base.update(overrides)
    return ChatState(**base)  # type: ignore[typeddict-item]
