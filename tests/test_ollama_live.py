"""Live integration check against a local Ollama server.

Opt-in: these are slow (seconds per call) and depend on a running server and a
pulled model, so they are skipped unless CAIRN_TEST_OLLAMA=1. The quality gates
must stay offline and deterministic, which is what LLM_PROVIDER=fake is for.

    CAIRN_TEST_OLLAMA=1 uv run pytest tests/test_ollama_live.py -m ollama
"""

from __future__ import annotations

import os

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src.config import Context, Settings
from src.graph.build import build_graph
from src.rag.prompts import cited_chunks

pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(
        os.environ.get("CAIRN_TEST_OLLAMA") != "1",
        reason="live Ollama test; set CAIRN_TEST_OLLAMA=1 to run",
    ),
]

MODEL = os.environ.get("CAIRN_TEST_OLLAMA_MODEL", "llama3.1")


@pytest.fixture
def ollama_settings() -> Settings:
    return Settings(env="dev", llm_provider="ollama", llm_model=MODEL)


async def test_real_model_answers_with_a_citation(ollama_settings: Settings) -> None:
    """The citation contract has to hold for a real model, not just the stub."""
    graph = build_graph(
        checkpointer=InMemorySaver(), store=InMemoryStore(), settings=ollama_settings
    )
    question = "How long do I have to submit an expense report?"

    out = await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}], "question": question},
        {"configurable": {"thread_id": "t-ollama-cite"}},
        context=Context(user_id="u-1"),
    )

    assert out["retrieved"]
    assert cited_chunks(out["answer"], out["retrieved"]), (
        f"real model returned an uncited answer: {out['answer']!r}"
    )


async def test_real_model_refuses_off_corpus_questions(ollama_settings: Settings) -> None:
    graph = build_graph(
        checkpointer=InMemorySaver(), store=InMemoryStore(), settings=ollama_settings
    )
    question = "Explain quantum chromodynamics."

    out = await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}], "question": question},
        {"configurable": {"thread_id": "t-ollama-clarify"}},
        context=Context(user_id="u-1"),
    )

    assert out["retrieved"] == []
    assert "[S" not in out["answer"]
