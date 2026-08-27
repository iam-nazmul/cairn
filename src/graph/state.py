"""ChatState -- the graph's typed state object and its reducers (SPEC §6.1)."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages


class RetrievedChunk(TypedDict):
    """One retrieved chunk. `source` and `score` are what make citation possible.

    CLAUDE.md: retrieve MUST return {text, source, score} per chunk so that
    generate can cite. An answer without citations is a bug.
    """

    text: str
    source: str
    score: float


class ChatState(TypedDict):
    """Graph state.

    `messages` uses the add-messages reducer: nodes return the NEW messages only
    and LangGraph appends them. Returning the whole list overwrites history and
    breaks the reducer -- the bug CLAUDE.md calls out explicitly.

    Every other field is per-turn and overwrites.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    retrieved: list[RetrievedChunk]
    long_term_facts: list[str]
    answer: str
