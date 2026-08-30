"""Researcher and writer subagents (SPEC §13.4).

Two compiled graphs with their own state. The researcher owns retrieval and
decides when it has enough; the writer owns composition and citation and has no
way to search -- it is compiled without a vector store, so it cannot cite a
source the researcher never found.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import Settings
from src.graph.state import Evidence, RetrievedChunk
from src.rag.llm import ChatModel
from src.rag.prompts import (
    REFINE_SYSTEM,
    build_refine_prompt,
    build_writer_prompt,
    cited_evidence,
    parse_refined_query,
)
from src.rag.retrieve import VectorStore, merge_chunks


class ResearchState(TypedDict):
    """The researcher's own state. Only `evidence` and the search log leave it."""

    question: str
    hints: list[str]
    searches: list[str]
    retrieved: list[RetrievedChunk]
    new_hits: int
    evidence: list[Evidence]


class WriterState(TypedDict):
    """What the writer is allowed to see. No query, no vector store, no chunks
    the researcher rejected."""

    question: str
    evidence: list[Evidence]
    preferences: list[str]
    messages: list[AnyMessage]
    answer: str


def build_researcher(
    vector_store: VectorStore, chat_model: ChatModel, settings: Settings
) -> CompiledStateGraph[ResearchState, Any, ResearchState, ResearchState]:
    """Search, judge, search again -- §13.2's loop, extracted behind its own state."""
    from src.graph.nodes import keep_searching  # circular at module scope

    async def gather(state: ResearchState) -> dict[str, Any]:
        tried = list(state.get("searches") or [])
        found = list(state.get("retrieved") or [])

        if not tried:
            # Durable facts are retrieval hints here, never instructions to the
            # writer: they widen the first query and nothing else.
            query = " ".join([state["question"], *state.get("hints", [])[:2]])
        else:
            reply = await chat_model.ainvoke(
                [
                    SystemMessage(content=REFINE_SYSTEM),
                    HumanMessage(content=build_refine_prompt(state["question"], tried, found)),
                ]
            )
            text = reply.text if isinstance(reply.text, str) else str(reply.content)
            query = parse_refined_query(text, state["question"], tried)

        chunks = await vector_store.search(query, top_k=settings.retrieval_top_k)
        merged, new_sources = merge_chunks(found, chunks)
        return {"retrieved": merged, "searches": [*tried, query], "new_hits": new_sources}

    async def commit(state: ResearchState) -> dict[str, Any]:
        """Assign the ids the writer will cite by. They are minted here, once,
        and travel with the evidence: a citation that survives the hand-off
        cannot be re-derived from a position in someone else's list."""
        kept = [
            chunk
            for chunk in state.get("retrieved") or []
            if chunk["score"] >= settings.retrieval_min_score
        ]
        return {
            "evidence": [
                Evidence(id=f"S{i}", text=c["text"], source=c["source"], score=c["score"])
                for i, c in enumerate(kept, start=1)
            ]
        }

    def enough(state: ResearchState) -> str:
        if keep_searching(
            list(state.get("retrieved") or []),
            list(state.get("searches") or []),
            state.get("new_hits", 0),
            settings,
        ):
            return "gather"
        return "commit"

    builder: StateGraph[ResearchState, Any, ResearchState, ResearchState] = StateGraph(
        ResearchState
    )
    builder.add_node("gather", gather)
    builder.add_node("commit", commit)
    builder.add_edge(START, "gather")
    builder.add_conditional_edges("gather", enough, {"gather": "gather", "commit": "commit"})
    builder.add_edge("commit", END)
    return builder.compile()


def build_writer(
    chat_model: ChatModel, settings: Settings
) -> CompiledStateGraph[WriterState, Any, WriterState, WriterState]:
    """Compose from the evidence, or return nothing. Compiled WITHOUT a vector
    store: the writer's inability to search is structural, not a prompt."""

    async def compose(state: WriterState) -> dict[str, Any]:
        evidence = list(state.get("evidence") or [])
        system = build_writer_prompt(
            evidence=evidence,
            preferences=list(state.get("preferences") or []),
            max_chars=settings.max_context_chars,
        )
        history: list[AnyMessage] = list(state.get("messages") or []) or [
            HumanMessage(content=state["question"])
        ]
        reply = await chat_model.ainvoke([SystemMessage(content=system), *history])
        answer = reply.text if isinstance(reply.text, str) else str(reply.content)

        # Same rule as `generate`: an answer nobody can trace is not an answer.
        # Here it also means the writer is asking for more evidence.
        if not cited_evidence(answer, evidence):
            return {"answer": ""}
        return {"answer": answer}

    builder: StateGraph[WriterState, Any, WriterState, WriterState] = StateGraph(WriterState)
    builder.add_node("compose", compose)
    builder.add_edge(START, "compose")
    builder.add_edge("compose", END)
    return builder.compile()
