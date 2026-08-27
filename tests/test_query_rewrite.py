"""History-aware retrieval (SPEC §6.2, "optionally history-rewritten query").

A follow-up turn shares no vocabulary with the corpus -- the terms that make it
findable were established earlier in the thread.
"""

from __future__ import annotations

from langchain.messages import AIMessage, HumanMessage

from src.config import Settings
from src.graph.nodes import make_retrieve
from src.rag.retrieve import InMemoryVectorStore, augment_query_with_history
from tests.conftest import make_runtime, make_state

FIRST = "How long do I have to submit an expense report?"
FOLLOW_UP = "And when do I actually get the money back?"


def test_augment_is_a_no_op_without_prior_turns() -> None:
    messages = [HumanMessage(content=FIRST)]
    assert augment_query_with_history(FIRST, messages) == FIRST


def test_augment_folds_in_the_previous_user_turn() -> None:
    messages = [
        HumanMessage(content=FIRST),
        AIMessage(content="Within 30 days. [S1]"),
        HumanMessage(content=FOLLOW_UP),
    ]
    augmented = augment_query_with_history(FOLLOW_UP, messages)

    assert FIRST in augmented
    assert FOLLOW_UP in augmented


def test_augment_ignores_assistant_turns() -> None:
    """Reusing the assistant's words would anchor retrieval to its last answer."""
    messages = [
        HumanMessage(content=FIRST),
        AIMessage(content="Priority one tickets are answered within one hour."),
        HumanMessage(content=FOLLOW_UP),
    ]
    assert "priority one" not in augment_query_with_history(FOLLOW_UP, messages).lower()


def test_augment_caps_the_number_of_turns_folded_in() -> None:
    messages = [
        HumanMessage(content="first question"),
        HumanMessage(content="second question"),
        HumanMessage(content="third question"),
        HumanMessage(content=FOLLOW_UP),
    ]
    augmented = augment_query_with_history(FOLLOW_UP, messages, max_turns=2)

    assert "first question" not in augmented
    assert "second question" in augmented
    assert "third question" in augmented


async def test_bare_follow_up_retrieves_nothing_on_its_own(settings: Settings) -> None:
    """The gap this feature closes: the follow-up alone is unfindable."""
    assert await InMemoryVectorStore().search(FOLLOW_UP, top_k=4) == []


async def test_follow_up_retrieves_once_history_is_folded_in(settings: Settings) -> None:
    node = make_retrieve(InMemoryVectorStore(), settings)
    state = make_state(
        question=FOLLOW_UP,
        messages=[
            HumanMessage(content=FIRST),
            AIMessage(content="Within 30 days. [S1]"),
            HumanMessage(content=FOLLOW_UP),
        ],
    )

    result = await node(state, runtime=make_runtime())

    assert result["retrieved"], "history-rewritten query should find the expense docs"
    assert any("expenses" in c["source"] for c in result["retrieved"])


async def test_self_contained_question_is_unaffected(settings: Settings) -> None:
    """The fallback only fires on weak retrieval, so normal turns are unchanged."""
    node = make_retrieve(InMemoryVectorStore(), settings)
    direct = await node(make_state(question=FIRST), runtime=make_runtime())

    with_history = await node(
        make_state(
            question=FIRST,
            messages=[
                HumanMessage(content="How do I connect to the VPN?"),
                AIMessage(content="Install the client. [S1]"),
                HumanMessage(content=FIRST),
            ],
        ),
        runtime=make_runtime(),
    )

    assert direct["retrieved"] == with_history["retrieved"]
