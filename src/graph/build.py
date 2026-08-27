"""StateGraph wiring and compilation (SPEC §6.3, §6.4).

M1 flow:  START -> retrieve -> (generate | clarify) -> END
The load_memory / write_memory nodes join in M3, per SPEC §12.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from src.config import Context, Settings, get_settings
from src.graph.nodes import make_clarify, make_generate, make_retrieve, make_route_after_retrieve
from src.graph.state import ChatState
from src.rag.llm import ChatModel, get_chat_model
from src.rag.retrieve import VectorStore, get_vector_store


def build_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    store: BaseStore | None = None,
    settings: Settings | None = None,
    vector_store: VectorStore | None = None,
    chat_model: ChatModel | None = None,
) -> CompiledStateGraph[ChatState, Context, ChatState, ChatState]:
    """Compile the graph. `checkpointer` is required -- without it there is no memory."""
    settings = settings or get_settings()
    vector_store = vector_store or get_vector_store()
    chat_model = chat_model or get_chat_model(settings)

    # context_schema declares Context so nodes can read runtime.context.user_id.
    builder = StateGraph(ChatState, context_schema=Context)

    builder.add_node("retrieve", make_retrieve(vector_store, settings))
    builder.add_node("generate", make_generate(chat_model, settings))
    builder.add_node("clarify", make_clarify(chat_model, settings))

    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        make_route_after_retrieve(settings),
        {"generate": "generate", "clarify": "clarify"},
    )
    builder.add_edge("generate", END)
    builder.add_edge("clarify", END)

    return builder.compile(checkpointer=checkpointer, store=store)
