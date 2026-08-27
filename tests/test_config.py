"""Configuration surface: .env.example completeness and logging wiring."""

from __future__ import annotations

import logging
import pathlib

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from src.config import Context, Settings
from src.graph.build import build_graph
from src.graph.state import ChatState
from src.observability import configure_logging

Graph = CompiledStateGraph[ChatState, Context, ChatState, ChatState]

ENV_EXAMPLE = pathlib.Path(__file__).parent.parent / ".env.example"


def documented_keys() -> set[str]:
    keys: set[str] = set()
    for line in ENV_EXAMPLE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0].strip())
    return keys


def test_env_example_documents_every_setting() -> None:
    """A setting nobody can discover may as well not be configurable."""
    missing = {name.upper() for name in Settings.model_fields} - documented_keys()
    assert not missing, f".env.example is missing: {sorted(missing)}"


def test_env_example_has_no_stale_keys() -> None:
    """Guards the other direction: a removed setting left behind as documentation."""
    known = {name.upper() for name in Settings.model_fields} | {
        "POSTGRES_TEST_URI",
        "CAIRN_TEST_OLLAMA",
        "CAIRN_TEST_OLLAMA_MODEL",
    }
    assert not documented_keys() - known


def test_configure_logging_enables_the_cairn_logger() -> None:
    cairn = logging.getLogger("cairn")
    original = cairn.level
    try:
        configure_logging(Settings(env="dev", log_level="INFO"))
        assert logging.getLogger("cairn.graph").isEnabledFor(logging.INFO)
    finally:
        cairn.setLevel(original)


def test_log_level_setting_is_honoured() -> None:
    cairn = logging.getLogger("cairn")
    original = cairn.level
    try:
        configure_logging(Settings(env="dev", log_level="WARNING"))
        assert not logging.getLogger("cairn.graph").isEnabledFor(logging.INFO)
    finally:
        cairn.setLevel(original)


async def test_node_logs_emit_without_a_test_harness_forcing_the_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The regression this guards: instrumentation that only works under caplog.

    No caplog.set_level here on purpose -- configure_logging alone must be enough,
    which is the production path through the API's lifespan handler.
    """
    cairn = logging.getLogger("cairn")
    original = cairn.level
    try:
        configure_logging(Settings(env="dev", log_level="INFO"))
        graph = build_graph(
            checkpointer=InMemorySaver(),
            store=InMemoryStore(),
            settings=Settings(env="dev", llm_provider="fake"),
        )
        question = "How long do I have to submit an expense report?"
        await graph.ainvoke(
            {"messages": [{"role": "user", "content": question}], "question": question},
            {"configurable": {"thread_id": "t-log-wiring"}},
            context=Context(user_id="u-1"),
        )
    finally:
        cairn.setLevel(original)

    assert any("node=retrieve" in r.getMessage() for r in caplog.records)
