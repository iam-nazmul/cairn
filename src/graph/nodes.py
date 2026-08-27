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
    LLM_EXTRACTION_PROMPT,
    extract_facts,
    load_user_facts,
    parse_llm_facts,
)
from src.memory.threads import register_thread
from src.rag.llm import ChatModel
from src.rag.prompts import (
    REFINE_SYSTEM,
    assemble_messages,
    build_refine_prompt,
    build_system_prompt,
    cited_chunks,
    parse_refined_query,
)
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
            return {"long_term_facts": [], **_reset_retrieval()}

        facts = await load_user_facts(store, runtime.context.user_id, settings.max_long_term_facts)
        return {"long_term_facts": facts, **_reset_retrieval()}

    return load_memory


def _reset_retrieval() -> dict[str, Any]:
    """Clear last turn's retrieval. It is per-turn, but the checkpoint keeps it,
    so without this agent mode would merge into chunks found for another
    question. `messages` is the only field meant to accumulate across turns."""
    return {"retrieved": [], "searches": [], "new_hits": 0}


def _merge_chunks(
    existing: list[RetrievedChunk], found: list[RetrievedChunk]
) -> tuple[list[RetrievedChunk], int]:
    """Combine searches, best score per source wins. Returns the new-source count.

    Deduplicating by source matters for citations: the same document arriving
    from two queries must stay one [S] block, or the answer cites two numbers for
    one source.
    """
    by_source = {chunk["source"]: chunk for chunk in existing}
    new_sources = 0
    for chunk in found:
        current = by_source.get(chunk["source"])
        if current is None:
            new_sources += 1
            by_source[chunk["source"]] = chunk
        elif chunk["score"] > current["score"]:
            by_source[chunk["source"]] = chunk

    merged = sorted(by_source.values(), key=lambda c: (-c["score"], c["source"]))
    return merged, new_sources


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
        query = question

        # Fall back only on weak retrieval, so self-contained questions are unaffected.
        if too_weak(chunks):
            augmented = augment_query_with_history(question, state.get("messages") or [])
            if augmented != question:
                retried = await vector_store.search(augmented, top_k=settings.retrieval_top_k)
                if not too_weak(retried):
                    chunks = retried
                    query = augmented

        # First search of the turn, so there is nothing to merge with.
        return {"retrieved": chunks, "searches": [query], "new_hits": len(chunks)}

    return retrieve


def make_research(vector_store: VectorStore, chat_model: ChatModel, settings: Settings) -> Node:
    """Agent mode: rewrite the query from what is missing, search again, merge.

    Chat mode never reaches this node -- one retrieval is the whole of it.
    """

    async def research(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        tried = list(state.get("searches") or [])
        found = list(state.get("retrieved") or [])

        reply = await chat_model.ainvoke(
            [
                SystemMessage(content=REFINE_SYSTEM),
                HumanMessage(content=build_refine_prompt(state["question"], tried, found)),
            ]
        )
        text = reply.text if isinstance(reply.text, str) else str(reply.content)
        query = parse_refined_query(text, state["question"], tried)

        chunks = await vector_store.search(query, top_k=settings.retrieval_top_k)
        merged, new_sources = _merge_chunks(found, chunks)

        return {"retrieved": merged, "searches": [*tried, query], "new_hits": new_sources}

    return research


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


def make_route_after_retrieve(
    settings: Settings,
) -> Callable[[ChatState, Runtime[Context]], str]:
    """Empty or low-confidence retrieval must NOT fall through to model priors.

    In agent mode this also decides whether another search is worth its model
    call. The grounding verdict at the end is identical either way -- searching
    more never lowers the bar for what may be answered.
    """

    def answer_or_clarify(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "clarify"
        if max(c["score"] for c in chunks) < settings.retrieval_min_score:
            return "clarify"
        return "generate"

    def route_after_retrieve(state: ChatState, runtime: Runtime[Context]) -> str:
        chunks = list(state.get("retrieved") or [])
        if runtime.context.mode != "agent":
            return answer_or_clarify(chunks)

        best = max((c["score"] for c in chunks), default=0.0)
        searched = len(state.get("searches") or [])

        if best >= settings.agent_good_score:
            return answer_or_clarify(chunks)  # already good enough
        if searched >= settings.agent_max_searches:
            return answer_or_clarify(chunks)  # budget spent
        if searched > 1 and not state.get("new_hits"):
            # The last rewrite surfaced nothing new, so further ones are paying a
            # model call to re-find what is already here.
            return answer_or_clarify(chunks)
        return "research"

    return route_after_retrieve
