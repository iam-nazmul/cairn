"""Graph nodes. Pure: return only changed keys, never mutate state."""

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
    """A graph node. `runtime` is keyword-only, per LangGraph's node protocol."""

    def __call__(
        self, state: ChatState, *, runtime: Runtime[Context]
    ) -> Awaitable[dict[str, Any]]: ...


def make_load_memory(settings: Settings) -> Node:
    """Read this user's durable facts from the Store into state."""

    async def load_memory(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        store = runtime.store
        if store is None:
            return {"long_term_facts": []}

        user_id = runtime.context.user_id
        facts: list[str] = []
        for namespace in (FACTS_NS, PREFERENCES_NS):
            items = await store.asearch((user_id, namespace), limit=settings.max_long_term_facts)
            facts.extend(str(item.value.get("text", "")) for item in items)

        return {"long_term_facts": sorted(f for f in facts if f)}

    return load_memory


def make_write_memory(settings: Settings, chat_model: ChatModel) -> Node:
    """Upsert durable facts from this turn into the Store."""

    async def write_memory(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        store = runtime.store
        if store is None or settings.memory_extraction == "off":
            return {}

        user_id = runtime.context.user_id

        # SPEC §10 deletion needs this: the checkpointer cannot map user -> threads.
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
            # Stable key: a fresh uuid per turn would duplicate, not update (SPEC §7.2).
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

        # Fall back only on weak retrieval, so self-contained questions are unaffected.
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
            # Uncited: drop it rather than ship it as grounded; the router sends
            # the turn to clarify.
            return {"answer": ""}

        # ONLY the new message -- the reducer appends.
        return {"answer": answer, "messages": [reply]}

    return generate


def make_clarify(chat_model: ChatModel, settings: Settings) -> Node:
    """No-answer path: retrieval was empty/weak, or generate could not cite."""

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
