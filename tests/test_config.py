"""Configuration surface: .env.example completeness and logging wiring."""

from __future__ import annotations

import logging
import pathlib
import re

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


# A bare assignment, commented out or not. Anchored so prose that merely mentions
# FOO=bar mid-sentence does not count as documenting a setting.
_ASSIGNMENT_RE = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=")


def documented_keys() -> set[str]:
    """Keys `.env.example` documents -- a commented-out assignment counts.

    One setting must stay unassigned: `OLLAMA_BASE_URL` has no value that is
    right everywhere, because `localhost` means this machine to uvicorn and the
    container to a container. Shipping either one misconfigures somebody, so the
    file explains it and assigns nothing. Documented is not the same as set.
    """
    keys: set[str] = set()
    for line in ENV_EXAMPLE.read_text().splitlines():
        match = _ASSIGNMENT_RE.match(line.strip())
        if match:
            keys.add(match.group(1))
    return keys


def test_env_example_documents_every_setting() -> None:
    """A setting nobody can discover may as well not be configurable."""
    missing = {name.upper() for name in Settings.model_fields} - documented_keys()
    assert not missing, f".env.example is missing: {sorted(missing)}"


def test_env_example_does_not_assign_ollama_base_url() -> None:
    """No value is right everywhere: `localhost` is this machine to uvicorn and
    the container to a container. An assignment here reads as though it also
    configures Docker, where Compose sets the address and this is ignored."""
    for line in ENV_EXAMPLE.read_text().splitlines():
        assert not line.strip().startswith("OLLAMA_BASE_URL="), line


def test_env_example_names_the_docker_override() -> None:
    """Whoever wants to repoint a container looks here first; the knob they need
    is spelled differently, so the file has to say so."""
    assert "DOCKER_OLLAMA_BASE_URL" in ENV_EXAMPLE.read_text()


# Documented in .env.example but never read by Settings: the file has to carry
# them because that is where the tool that reads them looks.
NOT_APP_SETTINGS = {
    # Consumed by the test suite.
    "POSTGRES_TEST_URI",
    "CAIRN_TEST_OLLAMA",
    "CAIRN_TEST_OLLAMA_MODEL",
    # Read by Docker Compose itself, which loads .env before the app exists.
    # DOCKER_OLLAMA_BASE_URL becomes the container's OLLAMA_BASE_URL; it is named
    # differently so a host value here cannot leak into the container.
    "COMPOSE_FILE",
    "DOCKER_OLLAMA_BASE_URL",
}


def test_env_example_has_no_stale_keys() -> None:
    """Guards the other direction: a removed setting left behind as documentation."""
    known = {name.upper() for name in Settings.model_fields} | NOT_APP_SETTINGS

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
