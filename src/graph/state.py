"""ChatState -- the graph's typed state object and its reducers (SPEC §6.1)."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages


class RetrievedChunk(TypedDict):
    """One retrieved chunk. `source` and `score` are what make citation possible."""

    text: str
    source: str
    score: float


class ChatState(TypedDict):
    """Graph state. `messages` appends via its reducer; every other field overwrites."""

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    retrieved: list[RetrievedChunk]
    long_term_facts: list[str]
    answer: str
