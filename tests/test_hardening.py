"""M4 hardening: context budget (SPEC §10/§11) and observability (SPEC §10)."""

from __future__ import annotations

import logging

import pytest
from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from src.config import Context, Settings
from src.graph.build import build_graph
from src.graph.state import ChatState
from src.rag.prompts import assemble_messages
from tests.conftest import make_state

Graph = CompiledStateGraph[ChatState, Context, ChatState, ChatState]


def long_history(turns: int) -> list[object]:
    messages: list[object] = []
    for i in range(turns):
        messages.append(HumanMessage(content=f"Question {i} about expense policy. " * 20))
        messages.append(AIMessage(content=f"Answer {i} from the handbook. " * 20))
    messages.append(HumanMessage(content="What is the receipt threshold?"))
    return messages


def test_history_is_trimmed_to_the_budget() -> None:
    state = make_state(question="What is the receipt threshold?", messages=long_history(30))

    untrimmed = assemble_messages(state, "system")
    trimmed = assemble_messages(state, "system", max_history_tokens=300)

    assert len(trimmed) < len(untrimmed)


def test_trimming_keeps_the_current_question_last() -> None:
    state = make_state(question="What is the receipt threshold?", messages=long_history(30))

    trimmed = assemble_messages(state, "system", max_history_tokens=300)

    assert str(trimmed[-1].content) == "What is the receipt threshold?"
    assert trimmed[0].type == "system"


def test_trimming_starts_on_a_human_turn() -> None:
    """A dangling assistant turn at the front confuses chat models."""
    state = make_state(question="What is the receipt threshold?", messages=long_history(30))

    trimmed = assemble_messages(state, "system", max_history_tokens=300)

    assert trimmed[1].type == "human"


def test_a_single_oversized_turn_still_survives() -> None:
    """The budget must never leave the model with nothing to answer."""
    state = make_state(question="x", messages=[HumanMessage(content="y " * 5000)])

    trimmed = assemble_messages(state, "system", max_history_tokens=10)

    assert len(trimmed) >= 2


def test_short_history_is_untouched() -> None:
    state = make_state(question="hi", messages=[HumanMessage(content="hi")])

    assert len(assemble_messages(state, "system", max_history_tokens=1500)) == 2


async def test_trimming_does_not_delete_from_the_checkpoint(settings: Settings) -> None:
    """Trimming is a prompt-time budget, not a memory deletion."""
    graph = build_graph(
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        settings=Settings(env="dev", llm_provider="fake", max_history_tokens=50),
    )
    config = {"configurable": {"thread_id": "t-trim"}}
    context = Context(user_id="u-1")

    for i in range(4):
        message = f"Question {i} about the expense report policy and receipts."
        await graph.ainvoke(
            {"messages": [{"role": "user", "content": message}], "question": message},
            config,
            context=context,
        )

    snapshot = await graph.aget_state(config)
    assert len(snapshot.values["messages"]) == 8, "the full thread stays checkpointed"


async def test_every_node_logs_timing(
    graph: Graph, context: Context, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="cairn.graph")
    question = "How long do I have to submit an expense report?"

    await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}], "question": question},
        {"configurable": {"thread_id": "t-obs"}},
        context=context,
    )

    logged = "\n".join(record.getMessage() for record in caplog.records)
    for node in ("load_memory", "retrieve", "generate", "write_memory"):
        assert f"node={node}" in logged
    assert "ms=" in logged
    assert "thread=t-obs" in logged


async def test_retrieval_hits_and_scores_are_logged(
    graph: Graph, context: Context, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="cairn.graph")
    question = "How long do I have to submit an expense report?"

    await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}], "question": question},
        {"configurable": {"thread_id": "t-obs-hits"}},
        context=context,
    )

    retrieve_line = next(
        r.getMessage() for r in caplog.records if "node=retrieve" in r.getMessage()
    )
    assert "hits=" in retrieve_line
    assert "doc://kb/" in retrieve_line


async def test_answer_token_cost_is_logged(
    graph: Graph, context: Context, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="cairn.graph")
    question = "How do I connect to the corporate VPN?"

    await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}], "question": question},
        {"configurable": {"thread_id": "t-obs-tokens"}},
        context=context,
    )

    assert "answer_tokens~" in "\n".join(r.getMessage() for r in caplog.records)
