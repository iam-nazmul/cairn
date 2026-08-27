"""Agent mode: multi-step retrieval that never loosens the grounding contract."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src.config import Context, Settings
from src.graph.build import build_graph
from src.graph.nodes import make_route_after_retrieve
from src.graph.state import RetrievedChunk
from src.rag.llm import DeterministicChatModel
from src.rag.retrieve import InMemoryVectorStore
from tests.conftest import make_runtime, make_state

QUESTION = "How long do I have to submit an expense report?"


class CountingVectorStore:
    """Wraps the real store and records every query it is asked for."""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self._inner = InMemoryVectorStore()

    async def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """See `VectorStore.search`."""
        self.queries.append(query)
        return await self._inner.search(query, top_k=top_k)


class BarrenVectorStore:
    """Finds nothing, whatever it is asked. Forces the no-answer path."""

    async def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """See `VectorStore.search`."""
        return []


def weak(score: float) -> RetrievedChunk:
    return RetrievedChunk(text="...", source="doc://kb/x", score=score)


async def run(
    vector_store: Any, settings: Settings, mode: str, question: str = QUESTION
) -> dict[str, Any]:
    graph = build_graph(
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        settings=settings,
        vector_store=vector_store,
        chat_model=DeterministicChatModel(),
    )
    return dict(
        await graph.ainvoke(
            {"messages": [{"role": "user", "content": question}], "question": question},
            {"configurable": {"thread_id": f"t-{mode}"}},
            context=Context(user_id="u-agent", mode=mode),  # type: ignore[arg-type]
        )
    )


# --- the loop ----------------------------------------------------------------


async def test_chat_mode_searches_exactly_once(settings: Settings) -> None:
    store = CountingVectorStore()

    result = await run(store, settings, "chat")

    assert len(store.queries) == 1
    assert result["searches"] == store.queries


async def test_agent_mode_searches_again_when_retrieval_is_mediocre(
    settings: Settings,
) -> None:
    """The whole point: a first search that is usable but not good gets another."""
    store = CountingVectorStore()
    # Above the answerable floor, below "stop looking".
    tuned = settings.model_copy(update={"agent_good_score": 0.99, "agent_max_searches": 3})

    result = await run(store, tuned, "agent")

    assert len(store.queries) > 1, "agent mode should not stop at the first search"
    assert result["searches"] == store.queries


async def test_agent_mode_respects_its_search_budget(settings: Settings) -> None:
    """Each extra search costs a model call, so the ceiling has to hold."""
    store = CountingVectorStore()
    tuned = settings.model_copy(update={"agent_good_score": 0.99, "agent_max_searches": 2})

    await run(store, tuned, "agent")

    assert len(store.queries) == 2


async def test_a_budget_of_one_makes_agent_mode_behave_like_chat(settings: Settings) -> None:
    store = CountingVectorStore()
    tuned = settings.model_copy(update={"agent_good_score": 0.99, "agent_max_searches": 1})

    await run(store, tuned, "agent")

    assert len(store.queries) == 1


async def test_agent_mode_stops_early_when_retrieval_is_already_good(
    settings: Settings,
) -> None:
    store = CountingVectorStore()
    tuned = settings.model_copy(update={"agent_good_score": 0.01, "agent_max_searches": 5})

    await run(store, tuned, "agent")

    assert len(store.queries) == 1, "no point rewriting a query that already worked"


# --- grounding is not negotiable ---------------------------------------------


async def test_agent_mode_still_refuses_to_answer_without_sources(
    settings: Settings,
) -> None:
    """Searching harder must never become licence to answer from model priors."""
    tuned = settings.model_copy(update={"agent_max_searches": 3})

    result = await run(BarrenVectorStore(), tuned, "agent", "Explain quantum chromodynamics.")

    assert result["retrieved"] == []
    assert "[S" not in result["answer"]


async def test_agent_mode_answers_with_citations(settings: Settings) -> None:
    result = await run(InMemoryVectorStore(), settings, "agent")

    assert "[S" in result["answer"]


# --- merging across searches -------------------------------------------------


async def test_repeated_sources_are_merged_not_duplicated(settings: Settings) -> None:
    """One document reached by two queries must stay one [S] block, or the answer
    cites two different numbers for the same source."""
    tuned = settings.model_copy(update={"agent_good_score": 0.99, "agent_max_searches": 3})

    result = await run(InMemoryVectorStore(), tuned, "agent")

    sources = [chunk["source"] for chunk in result["retrieved"]]
    assert len(sources) == len(set(sources))


async def test_retrieval_does_not_leak_between_turns(settings: Settings) -> None:
    """`retrieved` is per-turn but lives in the checkpoint, so a second turn must
    not merge into the chunks found for the first question."""
    tuned = settings.model_copy(update={"agent_good_score": 0.99})
    graph = build_graph(
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        settings=tuned,
        vector_store=InMemoryVectorStore(),
        chat_model=DeterministicChatModel(),
    )
    config = {"configurable": {"thread_id": "t-leak"}}
    context = Context(user_id="u-agent", mode="agent")

    await graph.ainvoke(
        {"messages": [{"role": "user", "content": QUESTION}], "question": QUESTION},
        config,
        context=context,
    )
    second = "How do I connect to the corporate VPN?"
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": second}], "question": second},
        config,
        context=context,
    )

    assert result["searches"][0] == second, "each turn starts its own search log"
    assert all("expense" not in chunk["text"].lower() for chunk in result["retrieved"])


# --- the router --------------------------------------------------------------


def test_router_ignores_the_agent_budget_in_chat_mode(settings: Settings) -> None:
    route = make_route_after_retrieve(settings)
    state = make_state(retrieved=[weak(settings.retrieval_min_score * 2)], searches=["one"])

    assert route(state, make_runtime(mode="chat")) == "generate"


def test_router_keeps_searching_in_agent_mode(settings: Settings) -> None:
    route = make_route_after_retrieve(settings)
    state = make_state(
        retrieved=[weak(settings.retrieval_min_score * 2)], searches=["one"], new_hits=1
    )

    assert route(state, make_runtime(mode="agent")) == "research"


def test_router_stops_when_a_rewrite_finds_nothing_new(settings: Settings) -> None:
    """Otherwise the budget is spent paying a model call to re-find what is here."""
    route = make_route_after_retrieve(settings)
    state = make_state(
        retrieved=[weak(settings.retrieval_min_score * 2)],
        searches=["one", "two"],
        new_hits=0,
    )

    assert route(state, make_runtime(mode="agent")) == "generate"


def test_router_sends_a_fruitless_agent_turn_to_clarify(settings: Settings) -> None:
    route = make_route_after_retrieve(settings)
    state = make_state(retrieved=[], searches=["one", "two", "three"])

    assert route(state, make_runtime(mode="agent")) == "clarify"


@pytest.mark.parametrize("mode", ["chat", "agent"])
def test_the_grounding_floor_is_the_same_in_both_modes(settings: Settings, mode: str) -> None:
    route = make_route_after_retrieve(settings)
    below = make_state(
        retrieved=[weak(settings.retrieval_min_score / 2)],
        searches=["one", "two", "three"],
    )

    assert route(below, make_runtime(mode=mode)) == "clarify"  # type: ignore[arg-type]
