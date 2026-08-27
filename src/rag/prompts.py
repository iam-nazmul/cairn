"""Prompt assembly and the citation-marker contract."""

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
    """Render chunks as numbered, citable blocks, capped at `max_chars`."""
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
    """System prompt + checkpointed history, trimmed to `max_history_tokens`."""
    history: list[AnyMessage] = list(state.get("messages") or [])
    if not history:
        # A node unit-tested with a bare state still gets the question.
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
        # trim_messages can return nothing if one turn blows the budget.
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
