"""StateGraph wiring. Flow diagram in CLAUDE.md."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from src.config import Context, Settings, get_settings
from src.graph.nodes import (
    approve,
    make_act,
    make_clarify,
    make_generate,
    make_load_memory,
    make_plan,
    make_research,
    make_researcher,
    make_retrieve,
    make_route_after_retrieve,
    make_supervise,
    make_write_memory,
    make_writer,
    route_after_approval,
    route_after_generate,
    route_after_plan,
    route_by_mode,
)
from src.graph.state import ChatState
from src.graph.subagents import build_researcher, build_writer
from src.observability import instrument
from src.rag.llm import ChatModel, get_chat_model
from src.rag.retrieve import VectorStore, get_vector_store
from src.tools.registry import ToolRegistry, build_registry


def build_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    store: BaseStore | None = None,
    settings: Settings | None = None,
    vector_store: VectorStore | None = None,
    chat_model: ChatModel | None = None,
    tools: ToolRegistry | None = None,
) -> CompiledStateGraph[ChatState, Context, ChatState, ChatState]:
    """Compile the graph. `checkpointer` is required -- without it there is no memory."""
    settings = settings or get_settings()
    vector_store = vector_store or get_vector_store()
    chat_model = chat_model or get_chat_model(settings)
    # Built either way: TOOLS_ENABLED gates what `generate` may offer, so the
    # tool nodes exist but stay unreachable when it is off.
    tools = tools if tools is not None else build_registry()

    builder = StateGraph(ChatState, context_schema=Context)

    builder.add_node("load_memory", instrument("load_memory", make_load_memory(settings)))
    builder.add_node("retrieve", instrument("retrieve", make_retrieve(vector_store, settings)))
    builder.add_node(
        "research", instrument("research", make_research(vector_store, chat_model, settings))
    )
    builder.add_node("generate", instrument("generate", make_generate(chat_model, settings, tools)))
    builder.add_node("plan", instrument("plan", make_plan(tools, settings)))
    builder.add_node("approve", instrument("approve", approve))
    builder.add_node("act", instrument("act", make_act(tools)))
    # Research mode (SPEC §13.4): two subagents behind their own state, wrapped
    # as nodes because their schemas are not the parent's.
    builder.add_node(
        "researcher",
        instrument(
            "researcher", make_researcher(build_researcher(vector_store, chat_model, settings))
        ),
    )
    builder.add_node(
        "writer", instrument("writer", make_writer(build_writer(chat_model, settings)))
    )
    builder.add_node("clarify", instrument("clarify", make_clarify(chat_model, settings)))
    builder.add_node(
        "write_memory", instrument("write_memory", make_write_memory(settings, chat_model))
    )

    builder.add_edge(START, "load_memory")
    builder.add_conditional_edges(
        "load_memory", route_by_mode, {"retrieve": "retrieve", "researcher": "researcher"}
    )
    # One router on both nodes: `research` loops back through the same decision,
    # so the budget and the grounding verdict cannot drift apart.
    route_after_retrieve = make_route_after_retrieve(settings)
    # Annotated because dict is invariant in its key type: a bare dict[str, str]
    # is not the dict[Hashable, str] add_conditional_edges declares.
    destinations: dict[Hashable, str] = {
        "generate": "generate",
        "clarify": "clarify",
        "research": "research",
    }
    builder.add_conditional_edges("retrieve", route_after_retrieve, destinations)
    builder.add_conditional_edges("research", route_after_retrieve, destinations)
    # generate -> clarify when the model returned an answer it could not cite;
    # generate -> plan when it asked for a tool instead of answering (SPEC §13.3).
    builder.add_conditional_edges(
        "generate",
        route_after_generate,
        {"generate": "write_memory", "clarify": "clarify", "plan": "plan"},
    )
    # An effect never happens before the human decision: `plan` only proposes,
    # `approve` only suspends, `act` alone performs -- and only downstream of both.
    builder.add_conditional_edges(
        "plan", route_after_plan, {"approve": "approve", "act": "act", "clarify": "clarify"}
    )
    builder.add_conditional_edges(
        "approve", route_after_approval, {"act": "act", "clarify": "clarify"}
    )
    builder.add_edge("act", "generate")
    # One supervisor on both subagents, for the same reason one router governs
    # the research loop: two would be free to disagree about when to stop.
    supervise = make_supervise(settings)
    handoffs: dict[Hashable, str] = {
        "researcher": "researcher",
        "writer": "writer",
        "clarify": "clarify",
        "write_memory": "write_memory",
    }
    builder.add_conditional_edges("researcher", supervise, handoffs)
    builder.add_conditional_edges("writer", supervise, handoffs)
    builder.add_edge("clarify", "write_memory")
    builder.add_edge("write_memory", END)

    return builder.compile(checkpointer=checkpointer, store=store)
