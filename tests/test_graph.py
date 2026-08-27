"""M1 graph behaviour: single-turn grounded answers, routing, and the state contract."""

from __future__ import annotations

import pytest
from langgraph.graph.state import CompiledStateGraph

from src.config import Context
from src.graph.state import ChatState

Graph = CompiledStateGraph[ChatState, Context, ChatState, ChatState]

GROUNDED_Q = "How long do I have to submit an expense report?"
OFF_CORPUS_Q = "Explain quantum chromodynamics."


def turn(text: str) -> dict[str, object]:
    """The client sends ONLY the new message -- prior turns come from the checkpoint."""
    return {"messages": [{"role": "user", "content": text}], "question": text}


async def test_single_turn_answers_with_a_citation(graph: Graph, context: Context) -> None:
    out = await graph.ainvoke(
        turn(GROUNDED_Q), {"configurable": {"thread_id": "t-1"}}, context=context
    )

    assert out["answer"]
    assert out["retrieved"]
    assert "[S1]" in out["answer"]


async def test_off_corpus_question_takes_the_clarify_path(graph: Graph, context: Context) -> None:
    """Empty retrieval must not fall through to model priors (CLAUDE.md Grounding)."""
    out = await graph.ainvoke(
        turn(OFF_CORPUS_Q), {"configurable": {"thread_id": "t-2"}}, context=context
    )

    assert out["retrieved"] == []
    assert "[S" not in out["answer"]
    assert "quark" not in out["answer"].lower(), "must not answer from prior knowledge"


async def test_messages_are_appended_not_overwritten(graph: Graph, context: Context) -> None:
    config = {"configurable": {"thread_id": "t-3"}}

    first = await graph.ainvoke(turn(GROUNDED_Q), config, context=context)
    second = await graph.ainvoke(turn("And what about receipts?"), config, context=context)

    assert len(first["messages"]) == 2  # human + ai
    assert len(second["messages"]) == 4  # reducer appended, did not replace
    assert second["messages"][0].content == GROUNDED_Q


async def test_invoking_without_a_thread_id_fails(graph: Graph, context: Context) -> None:
    """A missing thread_id must be loud, not a silently unpersisted turn."""
    with pytest.raises(Exception, match=r"(?i)thread_id|checkpointer"):
        await graph.ainvoke(turn(GROUNDED_Q), {"configurable": {}}, context=context)


async def test_checkpoint_is_written_for_the_thread(graph: Graph, context: Context) -> None:
    config = {"configurable": {"thread_id": "t-4"}}
    await graph.ainvoke(turn(GROUNDED_Q), config, context=context)

    # The checkpoint is the source of truth, not the response.
    snapshot = await graph.aget_state(config)
    assert snapshot.values["messages"]
    assert snapshot.values["answer"]
