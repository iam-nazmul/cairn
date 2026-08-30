"""Graph nodes. Pure: return only changed keys, never mutate state."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from src.config import Context, Settings
from src.graph.state import ChatState, RetrievedChunk
from src.memory.facts import (
    LLM_EXTRACTION_PROMPT,
    PREFERENCES_NS,
    extract_facts,
    load_namespace,
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
    parse_tool_request,
)
from src.rag.retrieve import VectorStore, augment_query_with_history, merge_chunks
from src.tools.registry import Tool, ToolRegistry


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
            return {"long_term_facts": [], "preferences": [], **_reset_retrieval()}

        # Registered here, not in write_memory: a turn that pauses for approval
        # (SPEC §13.3) must be owned before anyone can resume it, and a turn that
        # never finishes must still be reachable by deletion (SPEC §10).
        thread_id = get_config().get("configurable", {}).get("thread_id")
        if thread_id:
            await register_thread(store, runtime.context.user_id, str(thread_id))

        user_id = runtime.context.user_id
        facts = await load_user_facts(store, user_id, settings.max_long_term_facts)
        preferences = await load_namespace(
            store, user_id, PREFERENCES_NS, settings.max_long_term_facts
        )
        return {"long_term_facts": facts, "preferences": preferences, **_reset_retrieval()}

    return load_memory


def _reset_retrieval() -> dict[str, Any]:
    """Clear last turn's retrieval. It is per-turn, but the checkpoint keeps it,
    so without this agent mode would merge into chunks found for another
    question. `messages` is the only field meant to accumulate across turns.

    Tool state resets with it, which is what makes `TOOL_MAX_CALLS` a per-turn
    budget rather than a lifetime one for the thread."""
    return {
        "retrieved": [],
        "searches": [],
        "new_hits": 0,
        "tool_request": "",
        "pending_action": None,
        "tool_calls": [],
        "evidence": [],
        "handoffs": 0,
        "last_agent": "",
    }


def make_write_memory(settings: Settings, chat_model: ChatModel) -> Node:
    """Upsert durable facts from this turn into the Store."""

    async def write_memory(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        store = runtime.store
        if store is None or settings.memory_extraction == "off":
            return {}

        user_id = runtime.context.user_id
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
        merged, new_sources = merge_chunks(found, chunks)

        return {"retrieved": merged, "searches": [*tried, query], "new_hits": new_sources}

    return research


def make_generate(
    chat_model: ChatModel, settings: Settings, tools: ToolRegistry | None = None
) -> Node:
    async def generate(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        chunks = list(state.get("retrieved") or [])
        system = build_system_prompt(
            chunks=chunks,
            facts=list(state.get("long_term_facts") or []),
            max_chars=settings.max_context_chars,
            grounded=True,
            tools=_offered_tools(state, settings, tools),
        )
        reply = await chat_model.ainvoke(
            assemble_messages(state, system, settings.max_history_tokens)
        )
        answer = reply.text if isinstance(reply.text, str) else str(reply.content)

        if parse_tool_request(answer) is not None:
            # A directive is a request to act, not an answer: it must not reach
            # `messages`, and `plan` -- not this node -- resolves it.
            return {"answer": "", "tool_request": answer.strip()}

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
            declined=_declined_preview(state),
        )
        reply = await chat_model.ainvoke(
            assemble_messages(state, system, settings.max_history_tokens)
        )
        answer = reply.text if isinstance(reply.text, str) else str(reply.content)
        return {"answer": answer, "messages": [reply]}

    return clarify


def _offered_tools(state: ChatState, settings: Settings, tools: ToolRegistry | None) -> list[Tool]:
    """The tools `generate` may name -- none once the turn's budget is spent,
    or the model would keep asking for a call `plan` is bound to refuse."""
    if not settings.tools_enabled or tools is None:
        return []
    if len(_completed(state)) >= settings.tool_max_calls:
        return []
    return list(tools)


def _completed(state: ChatState) -> list[dict[str, Any]]:
    return [c for c in (state.get("tool_calls") or []) if c.get("status") == "done"]


def _declined_preview(state: ChatState) -> str:
    for call in reversed(state.get("tool_calls") or []):
        if call.get("status") == "rejected":
            return str(call.get("preview") or call.get("tool") or "")
    return ""


def make_plan(tools: ToolRegistry, settings: Settings) -> Node:
    """Resolve `generate`'s directive into a concrete call. Proposes; never acts.

    Returning `pending_action: None` is the refusal: the router sends the turn to
    the no-answer path rather than inventing a call the registry does not have.
    """

    async def plan(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        refused = {"tool_request": "", "pending_action": None}
        if len(_completed(state)) >= settings.tool_max_calls:
            return refused

        parsed = parse_tool_request(state.get("tool_request") or "")
        if parsed is None:
            return refused
        name, args = parsed
        tool = tools.get(name)
        if tool is None or not tool.validate(args):
            return refused

        return {
            "tool_request": "",
            "pending_action": {
                "call_id": f"c_{uuid.uuid4().hex[:12]}",
                "tool": tool.name,
                "args": args,
                "effect": tool.effect,
                "editable": sorted(tool.editable_fields),
                "preview": tool.preview(args),
                "requested_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        }

    return plan


async def approve(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
    """Suspend until a human decides. NOTHING observable may happen before the
    `interrupt()` call: a resume re-runs this node from its first line."""
    pending = dict(state.get("pending_action") or {})
    decision = interrupt(pending)

    if not _approved(decision):
        rejected = {**pending, "status": "rejected", "decided_at": _now()}
        return {"pending_action": None, "tool_calls": [*(state.get("tool_calls") or []), rejected]}

    edits = decision.get("edits") or {} if isinstance(decision, dict) else {}
    editable = set(pending.get("editable") or [])
    # Proposed args are kept beside the final ones: an audit that cannot show
    # what the human changed is not an audit.
    final = {**pending["args"], **{k: v for k, v in edits.items() if k in editable}}
    return {"pending_action": {**pending, "args": final, "proposed_args": pending["args"]}}


def _approved(decision: Any) -> bool:
    """Anything but an explicit approval is a refusal -- including a malformed one."""
    if decision is True:
        return True
    return isinstance(decision, dict) and decision.get("decision") == "approve"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def make_act(tools: ToolRegistry) -> Node:
    """Perform the approved call, once. Contains no `interrupt()`, so a resume
    never replays it -- and the `call_id` guard covers the case where something
    else does."""

    async def act(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        pending = dict(state.get("pending_action") or {})
        calls = list(state.get("tool_calls") or [])
        call_id = str(pending.get("call_id", ""))

        if any(c.get("call_id") == call_id and c.get("status") == "done" for c in calls):
            return {"pending_action": None}

        tool = tools.get(str(pending.get("tool", "")))
        if tool is None:
            return {"pending_action": None}

        result = await tool.run(**pending["args"])
        # Tool output is evidence like any other, so the citation gate applies to
        # it unchanged (SPEC §13.3). A second grounding path would be free to drift.
        merged, _ = merge_chunks(
            list(state.get("retrieved") or []),
            [RetrievedChunk(text=result, source=f"tool://{tool.name}/{call_id}", score=1.0)],
        )
        calls.append({**pending, "status": "done", "result": result, "decided_at": _now()})
        return {"retrieved": merged, "tool_calls": calls, "pending_action": None}

    return act


def route_after_generate(state: ChatState) -> str:
    """An uncited answer is not shippable -- fall through to the no-answer path."""
    if state.get("tool_request"):
        return "plan"
    return "generate" if state.get("answer") else "clarify"


def route_after_plan(state: ChatState) -> str:
    """`read` runs straight away; `write` cannot happen before a human says so."""
    pending = state.get("pending_action")
    if not pending:
        return "clarify"
    return "act" if pending.get("effect") == "read" else "approve"


def route_after_approval(state: ChatState) -> str:
    """`approve` clears `pending_action` when the human declined."""
    return "act" if state.get("pending_action") else "clarify"


def grounded_enough(scored: Iterable[Mapping[str, Any]], min_score: float) -> bool:
    """The grounding floor, in one place. Every mode asks this same question --
    a second copy is a second thing to drift (SPEC §13.5)."""
    best = max((float(item["score"]) for item in scored), default=0.0)
    return best >= min_score and best > 0.0


def keep_searching(
    chunks: list[RetrievedChunk], searches: list[str], new_hits: int, settings: Settings
) -> bool:
    """Is another search worth its model call? Shared by agent mode's router and
    the researcher subagent, so the budget cannot mean two different things."""
    best = max((c["score"] for c in chunks), default=0.0)
    if best >= settings.agent_good_score:
        return False  # already good enough; another rewrite re-finds it
    if len(searches) >= settings.agent_max_searches:
        return False  # budget spent
    # A rewrite that surfaced nothing new means further ones are paying a model
    # call to re-find what is already here.
    return not (len(searches) > 1 and not new_hits)


def make_route_after_retrieve(
    settings: Settings,
) -> Callable[[ChatState, Runtime[Context]], str]:
    """Empty or low-confidence retrieval must NOT fall through to model priors.

    In agent mode this also decides whether another search is worth its model
    call. The grounding verdict at the end is identical either way -- searching
    more never lowers the bar for what may be answered.
    """

    def answer_or_clarify(chunks: list[RetrievedChunk]) -> str:
        return "generate" if grounded_enough(chunks, settings.retrieval_min_score) else "clarify"

    def route_after_retrieve(state: ChatState, runtime: Runtime[Context]) -> str:
        chunks = list(state.get("retrieved") or [])
        if runtime.context.mode != "agent":
            return answer_or_clarify(chunks)

        if keep_searching(
            chunks, list(state.get("searches") or []), state.get("new_hits", 0), settings
        ):
            return "research"
        return answer_or_clarify(chunks)

    return route_after_retrieve


def make_researcher(researcher: Any) -> Node:
    """Run the researcher subagent. Only the evidence and the search log cross
    back: chunks it looked at and rejected are its own business (SPEC §13.4)."""

    async def researcher_node(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        facts = list(state.get("long_term_facts") or [])
        preferences = set(state.get("preferences") or [])
        result = await researcher.ainvoke(
            {
                "question": state["question"],
                # Facts are retrieval hints; preferences are the writer's business.
                "hints": [f for f in facts if f not in preferences],
                "searches": list(state.get("searches") or []),
                # Evidence it already produced is where a second pass resumes.
                "retrieved": [
                    RetrievedChunk(text=e["text"], source=e["source"], score=e["score"])
                    for e in state.get("evidence") or []
                ],
                "new_hits": 0,
                "evidence": [],
            }
        )
        return {
            "evidence": result["evidence"],
            "searches": result["searches"],
            "new_hits": result["new_hits"],
            "last_agent": "researcher",
        }

    return researcher_node


def make_writer(writer: Any) -> Node:
    """Run the writer subagent. It sees the evidence, the user's preferences and
    the conversation -- never the query, the corpus, or a way to reach either."""

    async def writer_node(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        result = await writer.ainvoke(
            {
                "question": state["question"],
                "evidence": list(state.get("evidence") or []),
                "preferences": list(state.get("preferences") or []),
                "messages": list(state.get("messages") or []),
                "answer": "",
            }
        )
        answer = str(result.get("answer") or "")
        if not answer:
            # Could not cite what it was given: hand the work back rather than
            # ship an answer nobody can trace.
            return {"handoffs": state.get("handoffs", 0) + 1, "last_agent": "writer"}

        # ONLY the writer's final message reaches `messages`. Neither subagent's
        # internal chatter may be checkpointed into every later turn.
        return {"answer": answer, "messages": [AIMessage(content=answer)], "last_agent": "writer"}

    return writer_node


def make_supervise(settings: Settings) -> Callable[[ChatState, Runtime[Context]], str]:
    """Who runs next, decided from state alone.

    A router, not a model call: "the researcher came back empty, send it out
    again" is a condition, and paying a third model call per turn to phrase it
    would buy nothing (SPEC §13.4).
    """

    def supervise(state: ChatState, runtime: Runtime[Context]) -> str:
        if state.get("answer"):
            return "write_memory"

        evidence = list(state.get("evidence") or [])
        if not grounded_enough(evidence, settings.retrieval_min_score):
            # Same floor as every other mode. Searching differently never lowers it.
            return "clarify"
        if state.get("last_agent") != "writer":
            return "writer"
        # The writer handed it back. Gather more -- but not forever.
        if state.get("handoffs", 0) >= settings.supervisor_max_handoffs:
            return "clarify"
        return "researcher"

    return supervise


def route_by_mode(state: ChatState, runtime: Runtime[Context]) -> str:
    """`research` mode gathers with a subagent instead of retrieving inline."""
    return "researcher" if runtime.context.mode == "research" else "retrieve"
