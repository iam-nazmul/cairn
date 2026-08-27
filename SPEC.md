# SPEC: Memory-Enabled RAG Chatbot (LangGraph + Checkpointer)

**Status:** Approved
**Owner:** _Md. Nazmul Hossain_
**Last updated:** 2026-08-27

---

## 1. Summary

Build a Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in a private knowledge base and **remembers the conversation across turns and sessions**. Memory is provided by LangGraph's persistence layer: a **checkpointer** for short-term, thread-scoped conversation state, and a **store** for long-term, cross-thread facts. The chatbot is orchestrated as a LangGraph state graph so that retrieval, generation, and memory read/write are explicit, inspectable, and resumable.

## 2. Goals

- Answer user questions using retrieved documents rather than model priors alone, with source citations.
- Maintain conversation continuity within a session without re-sending full history from the client — state is restored from the checkpointer by `thread_id`.
- Persist conversations across process restarts and days/weeks later (durable checkpointer).
- Support long-term, cross-conversation memory (e.g. user preferences, stable facts) via a store.
- Be resumable and fault-tolerant: a crashed or interrupted run can continue from the last checkpoint.

## 3. Non-Goals

- Model fine-tuning or training of custom LLMs.
- Multi-agent orchestration beyond a single retrieval-and-answer graph (may be a later phase).
- Building the ingestion/ETL pipeline for the corpus (assumed to exist or covered by a separate spec; see §11 open questions).
- A production-grade auth system — this spec assumes an authenticated `user_id` and `thread_id` are provided by the calling layer.

## 4. Definitions

- **Thread** — one conversation. Identified by `thread_id`. All checkpoints for a conversation share this key.
- **Checkpoint** — a serialized snapshot of the graph state saved after each super-step. History is retained (enables time travel and resume).
- **Checkpointer** — the storage backend that writes/reads checkpoints. Provides **short-term, thread-scoped memory**.
- **Store** — a separate key-value/semantic store for **long-term, cross-thread memory** (survives across different `thread_id`s, typically scoped by `user_id`).
- **RAG** — retrieve relevant chunks from a vector index, then condition generation on them.

## 5. Architecture Overview

```
                 ┌─────────────────────────────────────────────┐
   user turn ──▶ │                 LangGraph app                │
 (thread_id,     │                                             │
  user_id)       │   START                                     │
                 │     │                                       │
                 │     ▼                                       │
                 │  [load_memory] ◀──── Store (long-term)      │
                 │     │                                       │
                 │     ▼                                       │
                 │  [retrieve] ◀─────── Vector store / index   │
                 │     │                                       │
                 │     ▼                                       │
                 │  [generate] ◀─────── LLM                    │
                 │     │                                       │
                 │     ▼                                       │
                 │  [write_memory] ────▶ Store (long-term)     │
                 │     │                                       │
                 │    END                                      │
                 │                                             │
                 │  state saved every step ─▶ Checkpointer     │
                 └─────────────────────────────────────────────┘
```

Two memory systems, deliberately separated:

| Concern | Mechanism | Scope | Example |
|---|---|---|---|
| Conversation history / continuity | **Checkpointer** | Per `thread_id` (short-term) | "What did I just ask?" |
| Durable user facts & preferences | **Store** | Per `user_id`, across threads (long-term) | "User prefers answers in Bengali." |

## 6. Graph Design

### 6.1 State

The graph state is a typed object. Message history uses an append (add-messages) reducer so each node contributes without overwriting.

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]   # full turn history (checkpointed)
    question: str                             # current user query
    retrieved: list[dict]                     # [{text, source, score}, ...]
    long_term_facts: list[str]                # loaded from the Store
    answer: str                               # final grounded answer
```

### 6.2 Nodes

1. **`load_memory`** — read long-term facts for `user_id` from the Store and place them in state. No-op if none exist.
2. **`retrieve`** — embed the (optionally history-rewritten) query, run similarity search against the vector store, return top-`k` chunks with source metadata and scores.
3. **`generate`** — call the LLM with a prompt assembled from: system instructions, long-term facts, retrieved context, and the checkpointed `messages` history. Produce a grounded answer with citations.
4. **`write_memory`** — optionally extract durable facts/preferences from the turn and upsert them to the Store.

### 6.3 Edges

`START → load_memory → retrieve → generate → write_memory → END`

(Phase 2 may add a conditional edge after `retrieve`: if retrieval confidence is below a threshold, route to a clarify/no-answer node instead of `generate`.)

### 6.4 Compilation & invocation

```python
graph = builder.compile(checkpointer=checkpointer, store=store)

config = {"configurable": {"thread_id": thread_id}}

# Client sends ONLY the new message; prior turns are restored from the checkpoint.
result = await graph.ainvoke({"messages": [{"role": "user", "content": text}],
                              "question": text},
                             config=config,
                             context=Context(user_id=user_id))
```

> **Amended at implementation time (M1).** Current LangGraph injects a `Runtime`
> into nodes rather than a raw `config`: `thread_id` still travels in
> `config["configurable"]` (where the checkpointer reads it), but `user_id` now
> travels in a separate `context=` argument, declared to the graph via
> `StateGraph(ChatState, context_schema=Context)` and read inside a node as
> `runtime.context.user_id`. Node signature is `async def node(state, *, runtime)`
> — `runtime` is keyword-only. The core memory claim is unchanged.

The `thread_id` is the key: same `thread_id` ⇒ same restored conversation state; different `thread_id` ⇒ isolated conversation.

## 7. Memory Requirements

### 7.1 Short-term (checkpointer)

- MUST persist state after every node execution, keyed by `thread_id`.
- MUST restore the latest checkpoint automatically on the next `invoke` for that thread.
- MUST retain checkpoint history to support resume and time-travel/replay.
- Backend selection by environment:
  - **Dev:** in-memory saver — no durability, resets on restart.
  - **Local / staging:** SQLite saver — durable across process restarts, single node.
  - **Production:** Postgres saver — durable, concurrent, horizontally scalable, crash recovery.
- The graph code MUST be identical across backends; only the checkpointer instance changes.

### 7.2 Long-term (store)

- MUST persist facts scoped by `user_id`, readable from any thread.
- SHOULD support semantic lookup (embeddings) so relevant facts can be retrieved, not just exact keys.
- Writes SHOULD be idempotent/upsert to avoid duplicate facts.
- Namespacing convention: `(user_id, "facts")` and `(user_id, "preferences")`.

## 8. Tech Stack

- **Orchestration:** LangGraph (`StateGraph`), compiled with a checkpointer and store.
- **Checkpointer packages:** `langgraph-checkpoint` (base) + `langgraph-checkpoint-sqlite` and/or `langgraph-checkpoint-postgres`.
- **LLM:** pluggable via LangChain chat-model interface (provider configurable).
- **Embeddings + vector store:** configurable (e.g. pgvector / a managed vector DB); MUST return source metadata for citations.
- **API layer:** async HTTP service (e.g. FastAPI) exposing the endpoints in §9.
- **Runtime:** Python 3.11+.

> Pin exact package versions at implementation time and record them in the repo — the checkpointer/store APIs evolve between LangGraph releases.

## 9. API Surface

`POST /chat`

Request:
```json
{ "user_id": "u_123", "thread_id": "t_abc", "message": "..." }
```

Response:
```json
{
  "answer": "...",
  "citations": [{ "source": "doc://kb/42", "score": 0.83 }],
  "thread_id": "t_abc"
}
```

Supporting endpoints:
- `GET /threads/{thread_id}/history` — return checkpointed message history for a thread.
- `POST /threads` — create a new `thread_id`.
- `GET /health` — liveness/readiness.

## 10. Non-Functional Requirements

- **Latency:** p95 end-to-end response ≤ target (define at implementation; retrieval + one LLM call).
- **Isolation:** no state leakage between `thread_id`s or `user_id`s.
- **Durability:** production checkpointer survives process/host restarts with no lost committed turns.
- **Observability:** log per-node timing, retrieval hits/scores, and token usage; expose graph state for a given checkpoint for debugging.
- **Security/Privacy:** conversation state and long-term facts are user-scoped; support deletion of a user's threads and stored facts (right-to-be-forgotten).
- **Cost control:** cap retrieved-context size and history length passed to the LLM (context-window management / trimming or summarization for long threads).

## 11. Risks & Open Questions

- **Long threads overflow the context window.** Mitigation: trim old messages or summarize the thread into a running summary node (Phase 2).
- ~~**What extracts long-term facts** in `write_memory`~~ — **RESOLVED (M3): deterministic rules by default, LLM extraction behind a flag.**
  Explicit `remember that ...` commands plus a tight set of first-person patterns
  (`my <attribute> is <value>`, `I prefer ...`). Rationale: a bad fact is not wrong
  once — it is injected into every future prompt on every future thread for that
  user, so precision beats recall. Rules are also deterministic (testable in the
  gates without a model) and yield stable upsert keys derived from the normalized
  attribute, so restating a fact updates it instead of duplicating it (§7.2). LLM
  extraction has the opposite property: the same fact phrased two ways produces two
  keys and two rows. Available as `MEMORY_EXTRACTION=llm`; not the default, because
  it adds a model call to the write path of every turn, including turns containing
  no facts. Accepted cost: rules miss paraphrases they were not written for.
- **Corpus ingestion & re-indexing** is out of scope here — confirm the vector store and metadata schema are owned elsewhere.
- **Citation faithfulness** — need an eval to verify answers are actually grounded in retrieved chunks.
- ~~**Store backend**~~ — **RESOLVED (M3): reuse the checkpointer's Postgres with pgvector. One `DATABASE_URL`, not two.**
  Right-to-be-forgotten (§10) spans both memory systems, and M4 must make "delete
  this user's threads *and* facts" verifiable; one database makes that one
  transaction against one credential with one backup story. A separate vector DB
  adds a second failure domain and a second deletion path — precisely where a
  partial delete would hide — for no benefit at this scale. Note this concerns the
  **Store**, not the corpus index: the corpus vector store remains the pluggable
  seam that §3 assigns elsewhere.

## 12. Milestones

1. **M1 — Skeleton:** State graph with `retrieve → generate`, in-memory checkpointer, single-turn answers with citations.
2. **M2 — Short-term memory:** SQLite checkpointer, multi-turn continuity by `thread_id`, `/chat` + history endpoints.
3. **M3 — Long-term memory:** Store integration, `load_memory` / `write_memory` nodes, cross-thread facts.
4. **M4 — Production hardening:** Postgres checkpointer + store, observability, context trimming/summarization, deletion APIs, evals.