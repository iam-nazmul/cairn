"""ChatState -- the graph's typed state object and its reducers (SPEC §6.1)."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

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
    # Agent mode (SPEC §13.2). Every query tried this turn, oldest first;
    # its length is the search budget spent.
    searches: list[str]
    # Sources the most recent search added that earlier ones had not. Zero means
    # refining stopped paying for itself, which is a reason to stop.
    new_hits: int

    # Tools (SPEC §13.3). The raw `TOOL ...` directive `generate` emitted, which
    # `plan` -- the only node that may propose an effect -- resolves and validates.
    tool_request: str
    # The resolved call awaiting a decision, or None. Thread-scoped: it lives in
    # the checkpoint, never the Store.
    pending_action: dict[str, Any] | None
    # Every call decided this thread, done or rejected. `act` reads it to refuse
    # a second send of a call_id it has already performed.
    tool_calls: list[dict[str, Any]]
