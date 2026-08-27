"""Long-term memory: cross-thread facts, cross-user isolation, memory boundaries.

The Store is user-scoped and outlives any single conversation. These tests assert
the two systems stay separate in BOTH directions: durable facts must cross
threads, and conversation state must not.
"""

from __future__ import annotations

import pytest
from langchain.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from src.config import Context, Settings
from src.graph.build import build_graph
from src.graph.state import ChatState
from src.memory.facts import FACTS_NS, PREFERENCES_NS

Graph = CompiledStateGraph[ChatState, Context, ChatState, ChatState]

PREFERENCE = "My preferred language is Bengali."
ASK_PREFERENCE = "What language do I prefer?"


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def graph(settings: Settings, store: InMemoryStore) -> Graph:
    return build_graph(checkpointer=InMemorySaver(), store=store, settings=settings)


def turn(text: str) -> dict[str, object]:
    return {"messages": [{"role": "user", "content": text}], "question": text}


async def facts_for(store: BaseStore, user_id: str) -> list[str]:
    out: list[str] = []
    for namespace in (FACTS_NS, PREFERENCES_NS):
        items = await store.asearch((user_id, namespace), limit=50)
        out.extend(str(item.value["text"]) for item in items)
    return sorted(out)


async def test_a_fact_learned_in_one_thread_is_known_in_another(graph: Graph) -> None:
    """The whole point of the Store: it survives leaving the conversation."""
    context = Context(user_id="u-1")

    await graph.ainvoke(turn(PREFERENCE), {"configurable": {"thread_id": "t-a"}}, context=context)
    out = await graph.ainvoke(
        turn(ASK_PREFERENCE), {"configurable": {"thread_id": "t-brand-new"}}, context=context
    )

    assert "bengali" in out["answer"].lower()


async def test_facts_do_not_leak_across_users(graph: Graph) -> None:
    """SPEC §10 isolation. Namespaces come from the authenticated user_id only."""
    await graph.ainvoke(
        turn(PREFERENCE), {"configurable": {"thread_id": "t-u1"}}, context=Context(user_id="u-1")
    )
    out = await graph.ainvoke(
        turn(ASK_PREFERENCE),
        {"configurable": {"thread_id": "t-u2"}},
        context=Context(user_id="u-2"),
    )

    assert "bengali" not in out["answer"].lower()


async def test_the_fact_is_written_under_the_right_user(graph: Graph, store: InMemoryStore) -> None:
    await graph.ainvoke(
        turn(PREFERENCE), {"configurable": {"thread_id": "t-1"}}, context=Context(user_id="u-1")
    )

    assert await facts_for(store, "u-1") == ["preferred language is Bengali"]
    assert await facts_for(store, "u-2") == []


async def test_restating_a_fact_upserts_instead_of_duplicating(
    graph: Graph, store: InMemoryStore
) -> None:
    """SPEC §7.2 idempotency -- the reason keys are derived, not generated."""
    context = Context(user_id="u-1")
    config = {"configurable": {"thread_id": "t-upsert"}}

    await graph.ainvoke(turn(PREFERENCE), config, context=context)
    await graph.ainvoke(turn("My preferred language is English."), config, context=context)

    facts = await facts_for(store, "u-1")
    assert facts == ["preferred language is English"], "restating must update, not accumulate"


async def test_conversation_state_never_reaches_the_store(
    graph: Graph, store: InMemoryStore
) -> None:
    """A conversational turn is checkpointer territory and must write no facts."""
    await graph.ainvoke(
        turn("We were discussing invoice 42."),
        {"configurable": {"thread_id": "t-conv"}},
        context=Context(user_id="u-1"),
    )

    assert await facts_for(store, "u-1") == []


async def test_questions_write_nothing(graph: Graph, store: InMemoryStore) -> None:
    await graph.ainvoke(
        turn("How long do I have to submit an expense report?"),
        {"configurable": {"thread_id": "t-q"}},
        context=Context(user_id="u-1"),
    )

    assert await facts_for(store, "u-1") == []


async def test_facts_are_loaded_into_state_not_into_messages(
    graph: Graph, store: InMemoryStore
) -> None:
    """Durable facts must never be checkpointed as conversation (CLAUDE.md)."""
    context = Context(user_id="u-1")
    await graph.ainvoke(turn(PREFERENCE), {"configurable": {"thread_id": "t-x"}}, context=context)

    config = {"configurable": {"thread_id": "t-y"}}
    out = await graph.ainvoke(turn(ASK_PREFERENCE), config, context=context)
    snapshot = await graph.aget_state(config)

    # The fact reached the model through state, loaded from the Store this turn.
    assert "preferred language is Bengali" in out["long_term_facts"]

    # The brand-new thread's history holds exactly this turn's exchange: the fact
    # was not injected into `messages` as an extra turn, and the user's own message
    # was not rewritten to carry it. (It legitimately appears in the assistant's
    # answer -- that is the model USING the fact, which is the point.)
    messages = snapshot.values["messages"]
    assert [m.type for m in messages] == ["human", "ai"]
    assert str(messages[0].content) == ASK_PREFERENCE


async def test_extraction_can_be_turned_off(store: InMemoryStore) -> None:
    settings = Settings(env="dev", llm_provider="fake", memory_extraction="off")
    graph = build_graph(checkpointer=InMemorySaver(), store=store, settings=settings)

    await graph.ainvoke(
        turn(PREFERENCE), {"configurable": {"thread_id": "t-off"}}, context=Context(user_id="u-1")
    )

    assert await facts_for(store, "u-1") == []


class _ExtractorModel:
    """Stands in for a model asked to extract facts, in the documented format."""

    async def ainvoke(self, input: object, /) -> AIMessage:
        return AIMessage(content="role: accounts payable clerk\nNONE")


async def test_llm_extraction_path(store: InMemoryStore) -> None:
    """MEMORY_EXTRACTION=llm is a real option, not a stub (SPEC §11)."""
    settings = Settings(env="dev", llm_provider="fake", memory_extraction="llm")
    graph = build_graph(
        checkpointer=InMemorySaver(),
        store=store,
        settings=settings,
        chat_model=_ExtractorModel(),
    )

    await graph.ainvoke(
        turn("I just moved to the AP team."),
        {"configurable": {"thread_id": "t-llm"}},
        context=Context(user_id="u-1"),
    )

    assert await facts_for(store, "u-1") == ["role is accounts payable clerk"]


async def test_load_memory_is_a_no_op_for_a_user_with_no_facts(graph: Graph) -> None:
    out = await graph.ainvoke(
        turn("How long do I have to submit an expense report?"),
        {"configurable": {"thread_id": "t-empty"}},
        context=Context(user_id="u-new"),
    )

    assert out["long_term_facts"] == []
