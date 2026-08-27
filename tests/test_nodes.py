"""Each node unit-tested in isolation with a fabricated ChatState (CLAUDE.md Testing)."""

from __future__ import annotations

from langchain.messages import AIMessage, HumanMessage

from src.config import Settings
from src.graph.nodes import (
    make_clarify,
    make_generate,
    make_retrieve,
    make_route_after_retrieve,
    route_after_generate,
)
from src.graph.state import RetrievedChunk
from src.rag.llm import ChatModel, DeterministicChatModel
from src.rag.retrieve import InMemoryVectorStore
from tests.conftest import make_runtime, make_state

CHUNK = RetrievedChunk(
    text="Receipts are required over 25 dollars.", source="doc://kb/x", score=0.7
)


class _NoCitationModel:
    """Stands in for a model that ignores the citation instruction."""

    async def ainvoke(self, input: object, /) -> AIMessage:
        return AIMessage(content="Receipts are required over 25 dollars.")


async def test_retrieve_returns_only_the_keys_it_changes(settings: Settings) -> None:
    node = make_retrieve(InMemoryVectorStore(), settings)
    state = make_state(question="expense report receipts")

    result = await node(state, runtime=make_runtime())

    assert set(result) == {"retrieved", "searches", "new_hits"}
    assert result["retrieved"]
    assert result["searches"] == ["expense report receipts"], "the query it ran is the log"
    assert state["retrieved"] == [], "node must not mutate the state it was handed"


async def test_generate_appends_one_message_and_cites(settings: Settings) -> None:
    node = make_generate(DeterministicChatModel(), settings)
    state = make_state(
        question="What is the receipt threshold?",
        retrieved=[CHUNK],
        messages=[HumanMessage(content="What is the receipt threshold?")],
    )

    result = await node(state, runtime=make_runtime())

    assert set(result) == {"answer", "messages"}
    # The add-messages reducer appends: return the NEW message only, never the list.
    assert len(result["messages"]) == 1
    assert "[S1]" in result["answer"]


async def test_generate_drops_an_uncited_answer(settings: Settings) -> None:
    """An uncited answer must never ship as grounded (CLAUDE.md Retrieval).

    It is dropped rather than raised on: the usual cause is the model correctly
    declining because the retrieved context did not support the question.
    """
    node = make_generate(_NoCitationModel(), settings)
    state = make_state(question="What is the receipt threshold?", retrieved=[CHUNK])

    result = await node(state, runtime=make_runtime())

    assert result["answer"] == ""
    assert "messages" not in result, "the discarded reply must not enter history"
    assert route_after_generate(make_state(**result)) == "clarify"


def test_route_after_generate_ends_on_a_cited_answer() -> None:
    assert route_after_generate(make_state(answer="Receipts over 25 dollars. [S1]")) == "generate"


async def test_clarify_answers_without_citing_anything(settings: Settings) -> None:
    node = make_clarify(DeterministicChatModel(), settings)
    state = make_state(question="Tell me about lattice gauge theory.", retrieved=[])

    result = await node(state, runtime=make_runtime())

    assert set(result) == {"answer", "messages"}
    assert "[S" not in result["answer"], "nothing was retrieved, so nothing is citable"


async def test_clarify_still_uses_conversation_memory(settings: Settings) -> None:
    """Answering from checkpointed history is memory, not a model prior."""
    node = make_clarify(DeterministicChatModel(), settings)
    state = make_state(
        question="What's my name?",
        messages=[
            HumanMessage(content="My name is Alice."),
            AIMessage(content="Noted."),
            HumanMessage(content="What's my name?"),
        ],
    )

    result = await node(state, runtime=make_runtime())

    assert "alice" in str(result["answer"]).lower()


def test_route_sends_empty_retrieval_to_clarify(settings: Settings) -> None:
    route = make_route_after_retrieve(settings)
    assert route(make_state(retrieved=[]), make_runtime()) == "clarify"


def test_route_sends_low_confidence_retrieval_to_clarify(settings: Settings) -> None:
    route = make_route_after_retrieve(settings)
    weak = RetrievedChunk(text="...", source="doc://kb/x", score=settings.retrieval_min_score / 2)
    assert route(make_state(retrieved=[weak]), make_runtime()) == "clarify"


def test_route_sends_confident_retrieval_to_generate(settings: Settings) -> None:
    route = make_route_after_retrieve(settings)
    assert route(make_state(retrieved=[CHUNK]), make_runtime()) == "generate"


def test_deterministic_model_satisfies_the_chat_model_protocol() -> None:
    model: ChatModel = DeterministicChatModel()
    assert model is not None
