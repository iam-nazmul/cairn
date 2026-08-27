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

from langchain.messages import HumanMessage, SystemMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from src.config import Context, Settings
from src.graph.state import ChatState, RetrievedChunk
from src.memory.facts import (
    FACTS_NS,
    LLM_EXTRACTION_PROMPT,
    PREFERENCES_NS,
    extract_facts,
    parse_llm_facts,
)
from src.memory.threads import register_thread
from src.rag.llm import ChatModel
from src.rag.prompts import assemble_messages, build_system_prompt, cited_chunks
from src.rag.retrieve import VectorStore, augment_query_with_history


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


def make_load_memory(settings: Settings) -> Node:
    """Read this user's durable facts from the Store (SPEC §6.2). No-op if none.

    Facts are listed rather than semantically searched: the set is small and
    bounded, and listing is deterministic. Semantic lookup (SPEC §7.2's SHOULD)
    arrives in M4 with the pgvector-backed Postgres store.
    """

    async def load_memory(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        store = runtime.store
        if store is None:
            return {"long_term_facts": []}

        user_id = runtime.context.user_id
        facts: list[str] = []
        for namespace in (FACTS_NS, PREFERENCES_NS):
            # Namespaced by the authenticated user_id ONLY -- never a request field.
            items = await store.asearch((user_id, namespace), limit=settings.max_long_term_facts)
            facts.extend(str(item.value.get("text", "")) for item in items)

        return {"long_term_facts": sorted(f for f in facts if f)}

    return load_memory


def make_write_memory(settings: Settings, chat_model: ChatModel) -> Node:
    """Upsert durable facts from this turn into the Store (SPEC §6.2, §7.2).

    Only the user's own turn is scanned. Nothing conversational goes here: that
    belongs to the checkpointer.
    """

    async def write_memory(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        store = runtime.store
        if store is None or settings.memory_extraction == "off":
            return {}

        user_id = runtime.context.user_id

        # Index the thread against the user. The checkpointer cannot answer
        # "which threads belong to this user?", and SPEC §10 deletion needs it.
        thread_id = get_config().get("configurable", {}).get("thread_id")
        if thread_id:
            await register_thread(store, user_id, str(thread_id))

        question = state["question"]
        if settings.memory_extraction == "llm":
            reply = await chat_model.ainvoke(
                [
                    SystemMessage(content=LLM_EXTRACTION_PROMPT),
                    HumanMessage(content=question),
                ]
            )
            text = reply.text if isinstance(reply.text, str) else str(reply.content)
            facts = parse_llm_facts(text)
        else:
            facts = extract_facts(question)

        for fact in facts:
            # Upsert on a stable key: a fresh uuid per turn would duplicate
            # facts instead of updating them (SPEC §7.2).
            await store.aput((user_id, fact.namespace), fact.key, {"text": fact.text})

        # Writes nothing to state: durable facts must never enter `messages`.
        return {}

    return write_memory


def make_retrieve(vector_store: VectorStore, settings: Settings) -> Node:
    def too_weak(chunks: list[RetrievedChunk]) -> bool:
        return not chunks or max(c["score"] for c in chunks) < settings.retrieval_min_score

    async def retrieve(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        question = state["question"]
        chunks = await vector_store.search(question, top_k=settings.retrieval_top_k)

        # Only fall back to a history-rewritten query when the direct search came
        # back empty or weak, so self-contained questions behave exactly as before.
        if too_weak(chunks):
            augmented = augment_query_with_history(question, state.get("messages") or [])
            if augmented != question:
                retried = await vector_store.search(augmented, top_k=settings.retrieval_top_k)
                if not too_weak(retried):
                    chunks = retried

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
        reply = await chat_model.ainvoke(
            assemble_messages(state, system, settings.max_history_tokens)
        )
        answer = reply.text if isinstance(reply.text, str) else str(reply.content)

        if chunks and not cited_chunks(answer, chunks):
            # The model answered without citing anything -- usually because the
            # retrieved context did not actually support the question, which the
            # prompt tells it to say rather than guess. An uncited answer must
            # never ship as grounded, so drop it (no message appended, empty
            # answer) and let the router hand the turn to clarify.
            return {"answer": ""}

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
        reply = await chat_model.ainvoke(
            assemble_messages(state, system, settings.max_history_tokens)
        )
        answer = reply.text if isinstance(reply.text, str) else str(reply.content)
        return {"answer": answer, "messages": [reply]}

    return clarify


def route_after_generate(state: ChatState) -> str:
    """An uncited answer is not shippable -- fall through to the no-answer path."""
    return "generate" if state.get("answer") else "clarify"


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
