"""HTTP layer (SPEC §9)."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api import web
from src.config import Context, get_settings
from src.graph.build import build_graph
from src.memory.checkpointer import checkpointer_scope
from src.memory.facts import load_user_facts
from src.memory.store import store_scope
from src.memory.threads import forget_user, list_threads
from src.observability import configure_logging
from src.rag.llm import explain, probe, unreachable_hint
from src.rag.prompts import cited_chunks

logger = logging.getLogger("cairn.api")


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
    # Liveness of THIS service is `status`; the provider is a separate machine.
    # `None` means there was nothing to probe. Never fails the response -- Docker
    # health-checks this endpoint, and a down model must not restart the API.
    llm_reachable: bool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    # Say it once at boot rather than as a stack trace on someone's first message.
    if await probe(settings) is False:
        logger.warning("%s", unreachable_hint(settings))

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
    return HealthResponse(
        status="ok",
        env=settings.env,
        llm_provider=settings.llm_provider,
        llm_reachable=await probe(settings),
    )


@app.post("/threads")
async def create_thread() -> NewThreadResponse:
    """Mint a thread id. Nothing is written until the first turn is checkpointed."""
    return NewThreadResponse(thread_id=f"t_{uuid.uuid4().hex[:16]}")


def _turn(body: ChatRequest) -> tuple[dict[str, Any], dict[str, Any], Context]:
    """The three invoke arguments. ONLY the new message: the rest is checkpointed."""
    return (
        {"messages": [{"role": "user", "content": body.message}], "question": body.message},
        # thread_id in config (where the checkpointer reads it); user_id in context.
        {"configurable": {"thread_id": body.thread_id}},
        Context(user_id=body.user_id),
    )


def _citations(result: dict[str, Any]) -> list[Citation]:
    return [
        Citation(source=chunk["source"], score=chunk["score"])
        for chunk in cited_chunks(result.get("answer") or "", result.get("retrieved") or [])
    ]


@app.post("/chat")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    graph = request.app.state.graph
    payload, config, context = _turn(body)

    try:
        result: dict[str, Any] = await graph.ainvoke(payload, config, context=context)
    except Exception as exc:
        logger.exception("chat thread=%s FAILED", body.thread_id)
        # 503, not 500: the model is a dependency that is down, not a bug here.
        raise HTTPException(503, detail=explain(exc, request.app.state.settings)) from exc

    return ChatResponse(
        answer=result["answer"], citations=_citations(result), thread_id=body.thread_id
    )


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


def _chunk_text(chunk: Any) -> str:
    text = getattr(chunk, "text", None)
    return text if isinstance(text, str) else str(getattr(chunk, "content", "") or "")


@app.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    """The same turn as `POST /chat`, emitted as it is produced.

    Events are SSE JSON: `token` (a fragment of the answer), `restart` (discard
    what was drawn -- see below), `final` (authoritative answer + citations), and
    `error`. A provider that does not stream degrades to one `token` carrying the
    whole answer, so the browser needs no separate path for it.
    """
    graph = request.app.state.graph
    payload, config, context = _turn(body)

    async def events() -> AsyncIterator[str]:
        result: dict[str, Any] = {}
        streaming_node = ""
        try:
            async for part in graph.astream(
                payload,
                config,
                context=context,
                stream_mode=["messages", "values"],
                version="v2",
            ):
                if part["type"] == "values":
                    result = part["data"]
                    continue

                chunk, metadata = part["data"]
                # write_memory also calls a model under MEMORY_EXTRACTION=llm.
                # Only the two answering nodes may reach the browser.
                node = str(metadata.get("langgraph_node", ""))
                if node not in ("generate", "clarify"):
                    continue
                text = _chunk_text(chunk)
                if not text:
                    continue

                if streaming_node and node != streaming_node:
                    # generate -> clarify: the draft cited nothing, so it must not
                    # ship as grounded. Tell the browser to drop what it drew.
                    yield _sse({"type": "restart"})
                streaming_node = node
                yield _sse({"type": "token", "text": text})

        except Exception as exc:
            logger.exception("chat_stream thread=%s FAILED", body.thread_id)
            # The status line is long gone by now, so the diagnosis has to travel
            # in the event body.
            yield _sse({"type": "error", "detail": explain(exc, request.app.state.settings)})
            return

        yield _sse(
            {
                "type": "final",
                "answer": result.get("answer") or "",
                "citations": [c.model_dump() for c in _citations(result)],
                "thread_id": body.thread_id,
            }
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Proxies buffer by default, which turns a stream back into one blob.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
