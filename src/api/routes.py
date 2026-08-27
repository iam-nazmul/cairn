"""HTTP layer (SPEC §9)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.api import web
from src.config import Context, get_settings
from src.graph.build import build_graph
from src.memory.checkpointer import checkpointer_scope
from src.memory.facts import load_user_facts
from src.memory.store import store_scope
from src.memory.threads import forget_user, list_threads
from src.observability import configure_logging
from src.rag.prompts import cited_chunks


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class Citation(BaseModel):
    source: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    thread_id: str


class NewThreadResponse(BaseModel):
    thread_id: str


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]


class DeletionResponse(BaseModel):
    """What was actually removed, so the caller can verify rather than trust."""

    user_id: str
    threads_deleted: int
    facts_deleted: int


class ThreadListResponse(BaseModel):
    user_id: str
    threads: list[str]


class FactListResponse(BaseModel):
    user_id: str
    facts: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    env: str
    llm_provider: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    async with checkpointer_scope(settings) as checkpointer, store_scope(settings) as store:
        app.state.settings = settings
        app.state.checkpointer = checkpointer
        app.state.store = store
        app.state.graph = build_graph(checkpointer=checkpointer, store=store, settings=settings)
        yield


app = FastAPI(title="cairn", lifespan=lifespan)
web.mount(app)


@app.get("/health")
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(status="ok", env=settings.env, llm_provider=settings.llm_provider)


@app.post("/threads")
async def create_thread() -> NewThreadResponse:
    """Mint a thread id. Nothing is written until the first turn is checkpointed."""
    return NewThreadResponse(thread_id=f"t_{uuid.uuid4().hex[:16]}")


@app.post("/chat")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    graph = request.app.state.graph

    # thread_id in config (where the checkpointer reads it); user_id in context.
    config = {"configurable": {"thread_id": body.thread_id}}

    # ONLY the new message: prior turns are restored from the checkpoint.
    result: dict[str, Any] = await graph.ainvoke(
        {"messages": [{"role": "user", "content": body.message}], "question": body.message},
        config,
        context=Context(user_id=body.user_id),
    )

    answer = result["answer"]
    citations = [
        Citation(source=chunk["source"], score=chunk["score"])
        for chunk in cited_chunks(answer, result.get("retrieved") or [])
    ]
    return ChatResponse(answer=answer, citations=citations, thread_id=body.thread_id)


@app.get("/users/{user_id}/threads")
async def user_threads(request: Request, user_id: str) -> ThreadListResponse:
    """The user's conversations, from the same index deletion walks."""
    threads = await list_threads(request.app.state.store, user_id)
    return ThreadListResponse(user_id=user_id, threads=threads)


@app.get("/users/{user_id}/facts")
async def user_facts(request: Request, user_id: str) -> FactListResponse:
    """What the Store holds on this user -- the same list `load_memory` injects."""
    settings = request.app.state.settings
    facts = await load_user_facts(request.app.state.store, user_id, settings.max_long_term_facts)
    return FactListResponse(user_id=user_id, facts=facts)


@app.delete("/users/{user_id}")
async def forget(request: Request, user_id: str) -> DeletionResponse:
    """Right to be forgotten (SPEC §10): every thread and every stored fact."""
    report = await forget_user(request.app.state.store, request.app.state.checkpointer, user_id)
    return DeletionResponse(**report.as_dict())


@app.get("/threads/{thread_id}/history")
async def thread_history(request: Request, thread_id: str) -> HistoryResponse:
    """Read history straight from the checkpoint, never a hand-rolled table."""
    graph = request.app.state.graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})

    messages = snapshot.values.get("messages") if snapshot.values else None
    if not messages:
        raise HTTPException(status_code=404, detail=f"no history for thread {thread_id!r}")

    return HistoryResponse(
        thread_id=thread_id,
        messages=[HistoryMessage(role=m.type, content=str(m.content)) for m in messages],
    )
