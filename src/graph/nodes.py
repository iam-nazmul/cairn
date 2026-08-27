"""Graph nodes (SPEC §6.2).

Nodes are pure: they read `state`, return a dict of ONLY the keys they change,
and never mutate `state` in place. Nodes doing I/O are async.

Each node is produced by a small factory so its dependencies (vector store, chat
model, settings) are injected explicitly at wiring time in build.py rather than
reached for through module globals -- which is what makes them unit-testable with
a fabricated ChatState. The node itself keeps the documented
`(state, runtime) -> dict` signature.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from langgraph.runtime import Runtime

from src.config import Context, Settings
from src.graph.state import ChatState
from src.rag.llm import ChatModel
from src.rag.prompts import assemble_messages, build_system_prompt
from src.rag.retrieve import VectorStore


class Node(Protocol):
    """A graph node.

    LangGraph declares `runtime` as KEYWORD-ONLY in its node protocol
    (langgraph/graph/_node.py::_NodeWithRuntime), so nodes are written
    `(state, *, runtime)`. The skill example in .claude/skills/add-graph-node
    shows it positionally -- that runs, but does not type-check.
    """

    def __call__(
        self, state: ChatState, *, runtime: Runtime[Context]
    ) -> Awaitable[dict[str, Any]]: ...


class UngroundedAnswerError(RuntimeError):
    """Raised when the model answers from retrieved context without citing it.

    CLAUDE.md: "Answers without citations are a bug." Rather than shipping an
    uncited claim, the turn fails loudly.
    """


def make_retrieve(vector_store: VectorStore, settings: Settings) -> Node:
    async def retrieve(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        chunks = await vector_store.search(state["question"], top_k=settings.retrieval_top_k)
        return {"retrieved": chunks}

    return retrieve


def make_generate(chat_model: ChatModel, settings: Settings) -> Node:
    async def generate(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        chunks = list(state.get("retrieved") or [])
        system = build_system_prompt(
            chunks=chunks,
            facts=list(state.get("long_term_facts") or []),
            max_chars=settings.max_context_chars,
            grounded=True,
        )
        reply = await chat_model.ainvoke(assemble_messages(state, system))
        answer = reply.text if isinstance(reply.text, str) else str(reply.content)

        from src.rag.prompts import cited_chunks

        if chunks and not cited_chunks(answer, chunks):
            raise UngroundedAnswerError(
                "generate produced an answer with no citation marker despite "
                f"{len(chunks)} retrieved chunk(s)"
            )

        # `messages` gets ONLY the new message -- the add-messages reducer appends.
        return {"answer": answer, "messages": [reply]}

    return generate


def make_clarify(chat_model: ChatModel, settings: Settings) -> Node:
    """The no-answer path taken when retrieval comes back empty or weak.

    It may still answer from conversation history and long-term facts (that is
    remembered context, not model priors) but has no sources and cites nothing.
    """

    async def clarify(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        system = build_system_prompt(
            chunks=[],
            facts=list(state.get("long_term_facts") or []),
            max_chars=settings.max_context_chars,
            grounded=False,
        )
        reply = await chat_model.ainvoke(assemble_messages(state, system))
        answer = reply.text if isinstance(reply.text, str) else str(reply.content)
        return {"answer": answer, "messages": [reply]}

    return clarify


def make_route_after_retrieve(settings: Settings) -> Callable[[ChatState], str]:
    """Empty or low-confidence retrieval must NOT fall through to model priors."""

    def route_after_retrieve(state: ChatState) -> str:
        chunks = state.get("retrieved") or []
        if not chunks:
            return "clarify"
        if max(c["score"] for c in chunks) < settings.retrieval_min_score:
            return "clarify"
        return "generate"

    return route_after_retrieve
