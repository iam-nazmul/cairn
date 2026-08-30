"""Prompt assembly and the citation-marker contract."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, cast

from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from src.graph.state import ChatState, RetrievedChunk
from src.tools.registry import Tool

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


REFINE_SYSTEM = """You are refining a search query against a document index.

You will see the user's question, the queries already tried, and what those \
returned. Write ONE new search query that is likely to surface what is still \
missing -- different wording, a narrower aspect, or a term the documents would \
use rather than the user's phrasing.

Output the query alone: no quotes, no explanation, no prefix."""

TOOLS_SYSTEM = """You can also perform actions by calling a tool.

To call one, reply with a single line and nothing else:

TOOL <name> {{"arg": "value"}}

The arguments must be one JSON object. Do not explain the call, do not answer in \
the same reply, and never claim an action is done unless a tool:// block below \
shows it already ran -- such a block means that call has happened, so answer \
from it instead of requesting it again.

Available tools:
{tools}"""

DECLINED_TEMPLATE = (
    "The user was asked to approve {preview} and declined it, so the action did "
    "NOT happen. Say plainly what was not done. Do not offer to do it anyway."
)

# `TOOL <name> {json}` -- the whole reply, because a directive is not an answer.
TOOL_RE = re.compile(r"\ATOOL\s+([A-Za-z_][A-Za-z0-9_]*)\s+(\{.*\})\s*\Z", re.S)

_MAX_QUERY_CHARS = 200


def build_refine_prompt(question: str, tried: list[str], chunks: list[RetrievedChunk]) -> str:
    found = (
        "\n".join(f"- {c['source']} (score {c['score']}): {c['text'][:200]}" for c in chunks)
        or "- nothing yet"
    )
    return f"Question: {question}\n\nQueries tried:\n" + (
        "\n".join(f"- {t}" for t in tried) + f"\n\nFound so far:\n{found}"
    )


def parse_refined_query(reply: str, question: str, tried: list[str]) -> str:
    """First usable line of the model's answer, or the question as a fallback.

    A model that ignores the format is common; an unusable query is not a failure
    worth aborting the turn for, because a repeated search simply adds no new
    sources and the router stops on its own.
    """
    seen = {t.strip().lower() for t in tried}
    for line in reply.splitlines():
        candidate = line.strip().strip("\"'`").removeprefix("Query:").strip()
        if candidate and len(candidate) <= _MAX_QUERY_CHARS and candidate.lower() not in seen:
            return candidate
    return question


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


def format_tools(tools: Iterable[Tool]) -> str:
    return "\n".join(f"- {t.signature()}: {t.description}" for t in tools)


def parse_tool_request(reply: str) -> tuple[str, dict[str, Any]] | None:
    """Read a `TOOL name {json}` directive, or None if the reply is an answer.

    Strict on purpose: the directive must be the entire reply, so an answer that
    merely mentions a tool cannot be mistaken for a request to run one.
    """
    match = TOOL_RE.match(reply.strip())
    if match is None:
        return None
    try:
        args = json.loads(match.group(2))
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict) or any(not isinstance(k, str) for k in args):
        return None
    return match.group(1), cast(dict[str, Any], args)


def build_system_prompt(
    *,
    chunks: list[RetrievedChunk],
    facts: list[str],
    max_chars: int,
    grounded: bool = True,
    tools: Iterable[Tool] = (),
    declined: str = "",
) -> str:
    header = GROUNDED_SYSTEM if grounded else CLARIFY_SYSTEM
    parts = [header, format_facts(facts)]
    if grounded:
        parts.append(format_context(chunks, max_chars))
    offered = list(tools)
    if offered:
        parts.append(TOOLS_SYSTEM.format(tools=format_tools(offered)))
    if declined:
        # The model cannot know a human said no; it is not in the transcript.
        parts.append(DECLINED_TEMPLATE.format(preview=declined))
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
