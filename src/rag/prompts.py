"""Prompt assembly: system instructions + long-term facts + retrieved context.

Also owns the citation contract. `generate` emits `[S1]`-style markers; those
markers are mapped back onto the retrieved chunks they refer to, which is how
/chat builds its `citations` array (SPEC §9) without adding a field to ChatState.
"""

from __future__ import annotations

import re
from typing import cast

from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from src.graph.state import ChatState, RetrievedChunk

CITATION_RE = re.compile(r"\[S(\d+)\]")

GROUNDED_SYSTEM = """You are a knowledge-base assistant.

Answer using ONLY the retrieved context below, the conversation so far, and the \
known facts about the user. Never answer from your own prior knowledge.

Every claim drawn from the retrieved context MUST carry a citation marker naming \
the block it came from, e.g. [S1]. An answer with no citation marker is invalid.

If the retrieved context does not support an answer, say so instead of guessing."""

CLARIFY_SYSTEM = """You are a knowledge-base assistant.

Retrieval found no relevant documents for this question, so you have NO sources \
to cite and MUST NOT answer from your own prior knowledge.

You may still answer from the conversation so far and from the known facts about \
the user -- that is remembered context, not a guess. Otherwise, ask the user one \
short clarifying question."""


def format_facts(facts: list[str]) -> str:
    if not facts:
        return "Known facts about the user: (none on file)"
    lines = "\n".join(f"- {f}" for f in facts)
    return f"Known facts about the user:\n{lines}"


def format_context(chunks: list[RetrievedChunk], max_chars: int) -> str:
    """Render chunks as numbered, citable blocks, capped at `max_chars`.

    The cap is the retrieved-context half of the SPEC §10 cost control: chunks are
    dropped whole, lowest-ranked first, so a citation marker never points at a
    block that was truncated away.
    """
    if not chunks:
        return "Retrieved context: (nothing retrieved)"

    blocks: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks, start=1):
        block = f"[S{i}] source={chunk['source']} score={chunk['score']}\n{chunk['text']}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "Retrieved context:\n" + "\n\n".join(blocks)


def build_system_prompt(
    *,
    chunks: list[RetrievedChunk],
    facts: list[str],
    max_chars: int,
    grounded: bool = True,
) -> str:
    header = GROUNDED_SYSTEM if grounded else CLARIFY_SYSTEM
    parts = [header, format_facts(facts)]
    if grounded:
        parts.append(format_context(chunks, max_chars))
    return "\n\n".join(parts)


def assemble_messages(
    state: ChatState, system: str, max_history_tokens: int | None = None
) -> list[AnyMessage]:
    """System prompt + checkpointed history (which already ends with this turn).

    History comes from the checkpointer via `state["messages"]` -- never from the
    client and never from a hand-rolled table (CLAUDE.md memory boundaries).

    Long threads are trimmed to `max_history_tokens` before the model call: the
    oldest turns are dropped, the newest kept. Nothing is deleted from the
    checkpoint -- the full thread is still there, and /threads/{id}/history still
    returns all of it. This is the SPEC §11 context-overflow mitigation.
    """
    history: list[AnyMessage] = list(state.get("messages") or [])
    if not history:
        # Defensive: a node unit-tested with a bare state still gets the question.
        history = [HumanMessage(content=state["question"])]

    if max_history_tokens is not None:
        trimmed = trim_messages(
            history,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=max_history_tokens,
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
        # trim_messages can return nothing if a single turn blows the budget;
        # the current question must always survive.
        history = cast(list[AnyMessage], list(trimmed)) or history[-1:]

    return [SystemMessage(content=system), *history]


def cited_chunks(answer: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """The chunks an answer actually cites, in first-mention order."""
    seen: set[int] = set()
    out: list[RetrievedChunk] = []
    for match in CITATION_RE.finditer(answer):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(chunks) and idx not in seen:
            seen.add(idx)
            out.append(chunks[idx])
    return out
