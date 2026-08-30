"""Research mode: a researcher that gathers and a writer that composes (SPEC §13.4).

The split is only worth having if the hand-off cannot lose provenance, so most
of this file is about where a citation comes from.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from langchain.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src.config import Context, Settings
from src.graph.nodes import make_route_after_retrieve, make_supervise
from src.graph.state import Evidence, RetrievedChunk
from src.graph.subagents import build_writer
from src.memory.facts import FACTS_NS, PREFERENCES_NS
from src.rag.llm import DeterministicChatModel
from src.rag.prompts import cited_evidence
from src.rag.retrieve import InMemoryVectorStore
from tests.conftest import make_runtime, make_state

QUESTION = "How long do I have to submit an expense report?"
USER = "u_research"


class CountingVectorStore:
    """Records every query, so 'the writer cannot search' is checkable."""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self._inner = InMemoryVectorStore()

    async def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """See `VectorStore.search`."""
        self.queries.append(query)
        return await self._inner.search(query, top_k=top_k)


class BarrenVectorStore:
    async def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """See `VectorStore.search`."""
        return []


class UncitingWriter:
    """Answers, never cites. The writer half of a loop that will not converge."""

    def __init__(self) -> None:
        self.composed = 0

    async def ainvoke(self, input: Any, /) -> AIMessage:
        """See `ChatModel.ainvoke`."""
        messages: list[AnyMessage] = list(input)
        system = str(messages[0].content)
        if "two-agent system" in system:
            self.composed += 1
            return AIMessage(content="I need more to go on.")
        return AIMessage(content="a different search")


def ev(marker: str, score: float) -> Evidence:
    return Evidence(id=marker, text="...", source=f"doc://kb/{marker}", score=score)


async def run(
    vector_store: Any,
    settings: Settings,
    chat_model: Any | None = None,
    store: Any | None = None,
    thread_id: str = "t-research",
) -> dict[str, Any]:
    graph = build_graph_for(vector_store, settings, chat_model, store)
    return dict(
        await graph.ainvoke(
            {"messages": [{"role": "user", "content": QUESTION}], "question": QUESTION},
            {"configurable": {"thread_id": thread_id}},
            context=Context(user_id=USER, mode="research"),
        )
    )


def build_graph_for(
    vector_store: Any, settings: Settings, chat_model: Any | None = None, store: Any | None = None
) -> Any:
    from src.graph.build import build_graph

    return build_graph(
        checkpointer=InMemorySaver(),
        store=store if store is not None else InMemoryStore(),
        settings=settings,
        vector_store=vector_store,
        chat_model=chat_model or DeterministicChatModel(),
    )


# --- the hand-off ------------------------------------------------------------


async def test_the_writer_answers_from_what_the_researcher_gathered(
    settings: Settings,
) -> None:
    result = await run(InMemoryVectorStore(), settings)

    assert result["answer"]
    assert result["evidence"], "the researcher must hand something over"
    assert result["last_agent"] == "writer"


async def test_every_citation_maps_to_evidence_from_this_turn(settings: Settings) -> None:
    result = await run(InMemoryVectorStore(), settings)

    cited = cited_evidence(result["answer"], result["evidence"])
    assert cited, "an answer with no traceable citation must not ship"
    minted = {e["id"] for e in result["evidence"]}
    assert all(c["id"] in minted for c in cited)


async def test_ids_are_minted_by_the_researcher_and_survive_the_handoff(
    settings: Settings,
) -> None:
    result = await run(InMemoryVectorStore(), settings)

    evidence = result["evidence"]
    assert [e["id"] for e in evidence] == [f"S{i}" for i in range(1, len(evidence) + 1)]
    # Best score first: the id is assigned to the chunk, not to a position the
    # writer might re-sort.
    assert evidence == sorted(evidence, key=lambda e: -e["score"])


def test_the_writer_is_compiled_without_a_vector_store() -> None:
    """Structural, not a prompt: it has nothing to search with."""
    assert "vector_store" not in inspect.signature(build_writer).parameters


async def test_only_the_researcher_searches(settings: Settings) -> None:
    store = CountingVectorStore()

    result = await run(store, settings)

    # Every query the corpus saw is one the researcher logged; the writer adds none.
    assert store.queries == result["searches"]


async def test_a_research_turn_appends_exactly_one_assistant_message(
    settings: Settings,
) -> None:
    result = await run(InMemoryVectorStore(), settings)

    assistant = [m for m in result["messages"] if m.type == "ai"]
    assert len(assistant) == 1, "subagent chatter must not reach the checkpoint"


# --- ceilings ----------------------------------------------------------------


async def test_a_writer_that_never_cites_is_stopped_by_the_handoff_ceiling(
    settings: Settings,
) -> None:
    writer = UncitingWriter()

    result = await run(
        InMemoryVectorStore(), Settings(env="dev", llm_provider="fake"), chat_model=writer
    )

    # The ceiling counts attempts: the writer tries, hands back, is sent out
    # again once, and the second failure ends the turn instead of a third.
    assert result["handoffs"] == settings.supervisor_max_handoffs
    assert writer.composed == settings.supervisor_max_handoffs
    assert result["answer"], "the turn still ends, on the no-answer path"


def test_the_supervisor_sends_work_back_only_within_budget(settings: Settings) -> None:
    supervise = make_supervise(settings)
    handed_back = make_state(evidence=[ev("S1", 0.4)], last_agent="writer", handoffs=1)
    spent = make_state(
        evidence=[ev("S1", 0.4)],
        last_agent="writer",
        handoffs=settings.supervisor_max_handoffs,
    )

    assert supervise(handed_back, make_runtime(mode="research")) == "researcher"
    assert supervise(spent, make_runtime(mode="research")) == "clarify"


def test_an_answer_ends_the_turn(settings: Settings) -> None:
    supervise = make_supervise(settings)
    done = make_state(answer="Yes [S1].", evidence=[ev("S1", 0.4)], last_agent="writer")

    assert supervise(done, make_runtime(mode="research")) == "write_memory"


# --- grounding is identical in every mode (SPEC §13, §13.5) ------------------


async def test_research_mode_refuses_to_answer_without_sources(settings: Settings) -> None:
    result = await run(BarrenVectorStore(), settings)

    assert result["evidence"] == []
    assert result["answer"], "it says what it could not find"
    assert "[S" not in result["answer"]


@pytest.mark.parametrize("mode", ["chat", "agent", "research"])
def test_no_mode_answers_from_below_the_floor(settings: Settings, mode: str) -> None:
    """The same floor, whichever decider a mode routes through."""
    below = settings.retrieval_min_score / 2

    if mode == "research":
        supervise = make_supervise(settings)
        state = make_state(evidence=[ev("S1", below)], last_agent="researcher")
        assert supervise(state, make_runtime(mode="research")) == "clarify"  # type: ignore[arg-type]
    else:
        route = make_route_after_retrieve(settings)
        state = make_state(
            retrieved=[RetrievedChunk(text="...", source="doc://kb/x", score=below)],
            searches=["one", "two", "three"],
        )
        assert route(state, make_runtime(mode=mode)) == "clarify"  # type: ignore[arg-type]


def test_the_floor_is_read_from_one_place(settings: Settings) -> None:
    """Raise the floor and every mode moves together, or this drifts."""
    strict = Settings(env="dev", llm_provider="fake", retrieval_min_score=0.9)
    chunk = RetrievedChunk(text="...", source="doc://kb/x", score=0.5)

    assert (
        make_route_after_retrieve(strict)(
            make_state(retrieved=[chunk], searches=["a", "b", "c"]), make_runtime(mode="agent")
        )
        == "clarify"
    )
    assert (
        make_supervise(strict)(
            make_state(evidence=[ev("S1", 0.5)], last_agent="researcher"),
            make_runtime(mode="research"),
        )
        == "clarify"
    )


# --- who gets which memory ---------------------------------------------------


async def test_preferences_go_to_the_writer_and_facts_to_the_researcher(
    settings: Settings,
) -> None:
    store = InMemoryStore()
    await store.aput((USER, PREFERENCES_NS), "lang", {"text": "User prefers answers in Bengali."})
    await store.aput((USER, FACTS_NS), "team", {"text": "User works in accounts payable."})
    corpus = CountingVectorStore()

    result = await run(corpus, settings, store=store)

    assert result["preferences"] == ["User prefers answers in Bengali."]
    # The fact widens the first query; the preference never touches retrieval.
    assert "accounts payable" in corpus.queries[0]
    assert "Bengali" not in corpus.queries[0]


# --- over HTTP ---------------------------------------------------------------


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    from fastapi.testclient import TestClient

    from src.api import routes

    test_settings = Settings(
        env="local", llm_provider="fake", sqlite_path=str(tmp_path / "research.db")
    )
    monkeypatch.setattr(routes, "get_settings", lambda: test_settings)
    with TestClient(routes.app) as test_client:
        yield test_client


def post(client: Any, thread_id: str, mode: str = "research") -> dict[str, Any]:
    response = client.post(
        "/chat",
        json={"user_id": USER, "thread_id": thread_id, "message": QUESTION, "mode": mode},
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_research_mode_answers_with_citations(client: Any) -> None:
    body = post(client, "t-api-research")

    assert body["answer"]
    assert body["citations"], "a research answer is grounded like any other"


def test_an_unknown_mode_is_rejected_rather_than_treated_as_chat(client: Any) -> None:
    response = client.post(
        "/chat",
        json={"user_id": USER, "thread_id": "t-api-bad", "message": QUESTION, "mode": "freestyle"},
    )

    assert response.status_code == 422


def test_the_stream_and_the_blocking_endpoint_agree_in_research_mode(client: Any) -> None:
    """Two transports, one turn -- the third mode does not get to drift either."""
    import json as _json

    blocking = post(client, "t-api-res-a")
    body = {
        "user_id": USER,
        "thread_id": "t-api-res-b",
        "message": QUESTION,
        "mode": "research",
    }
    with client.stream("POST", "/chat/stream", json=body) as response:
        raw = "".join(response.iter_text())
    events = [
        _json.loads(block.removeprefix("data:").strip())
        for block in raw.split("\n\n")
        if block.strip()
    ]

    assert events[-1]["answer"] == blocking["answer"]
    assert events[-1]["citations"] == blocking["citations"]
    assert [e for e in events if e["type"] == "search"], "the researcher's work is shown"
