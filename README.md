# cairn

A memory-enabled **RAG chatbot** orchestrated as a **LangGraph** state graph.

**[`SPEC.md`](SPEC.md) is the design document** — architecture, memory model, API
surface and milestones. **[`CLAUDE.md`](CLAUDE.md)** is the working guide for
making changes. Read SPEC.md first.

## The one idea

Two memory systems, deliberately kept separate:

| Concern | Mechanism | Scope |
|---|---|---|
| Conversation history | **Checkpointer** | per `thread_id` |
| Durable user facts | **Store** | per `user_id`, across threads |

The client sends only the new message each turn; prior turns are restored from
the checkpoint by `thread_id`.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

The default `.env` points at a local **Ollama** server:

```bash
ollama serve            # if it is not already running
ollama pull llama3.1    # LLM_MODEL
```

Set `LLM_MODEL` to any model you have pulled (`ollama ls`). Note that thinking
models such as `qwen3` emit their reasoning inline unless configured otherwise,
which shows up in the answer text.

Tests never touch Ollama: they set `LLM_PROVIDER=fake` explicitly, so the gates
stay offline and deterministic. To exercise a real model:

```bash
CAIRN_TEST_OLLAMA=1 uv run pytest -m ollama
```

## Running it

```bash
ENV=local uv run uvicorn src.api.routes:app --reload
```

`ENV` selects the backend: `dev` → in-memory (resets on restart), `local` →
SQLite at `SQLITE_PATH`, `prod` → Postgres at `DATABASE_URL`. The graph code is
identical across all three.

For `prod`, one Postgres backs **both** the checkpointer and the Store (SPEC §11):

```bash
docker compose up -d        # pgvector/pgvector:pg16 on :5433
ENV=prod uv run uvicorn src.api.routes:app
```

| Endpoint | Purpose |
|---|---|
| `POST /chat` | One turn. Send **only** the new message — prior turns come from the checkpoint. |
| `POST /threads` | Mint a new `thread_id`. |
| `GET /threads/{id}/history` | Checkpointed history for a thread. |
| `DELETE /users/{user_id}` | Right to be forgotten: every thread **and** every stored fact. |
| `GET /health` | Liveness, plus the active backend and provider. |

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"user_id":"u_1","thread_id":"t_1","message":"How long do I have to submit an expense report?"}'
```

## Quality gates

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src
```

`pytest` alone skips the two opt-in suites. To run everything:

```bash
docker compose up -d
POSTGRES_TEST_URI=postgresql://cairn:cairn@localhost:5433/cairn?sslmode=disable \
  CAIRN_TEST_OLLAMA=1 uv run pytest
```

## Deleting a user (SPEC §10)

```bash
curl -X DELETE localhost:8000/users/u_123
# {"user_id":"u_123","threads_deleted":2,"facts_deleted":1}
```

Deletion spans **both** memory systems: every checkpoint for every thread the user
owns, and every Store namespace scoped to them. The checkpointer is keyed by
`thread_id` alone and its metadata carries no `user_id`, so a thread index is kept
in the Store under `(user_id, "threads")` — thread ids only, no message content.
Adding a third place user data lands means adding it to `USER_NAMESPACES`, or
deletion silently becomes partial. A test asserts that list stays complete.

## Observability

Every node logs its duration, retrieval hits and scores, facts loaded, and
approximate answer tokens, tagged with `thread_id`:

```
node=retrieve thread=t_abc ms=0.4 hits=2 doc://kb/expenses-2=0.4 doc://kb/expenses-1=0.2
node=generate thread=t_abc ms=1412.7 answer_tokens~24
```

## Status

| Milestone | State |
|---|---|
| M1 — skeleton: `retrieve → generate`, in-memory checkpointer, cited answers | done |
| M2 — short-term memory: SQLite checkpointer, `/chat` + threads endpoints | done |
| M3 — long-term memory: Store, `load_memory` / `write_memory` | done |
| M4 — hardening: Postgres, observability, trimming, deletion APIs | done |

## How a turn is routed

```
START → load_memory → retrieve → generate ──────→ write_memory → END
                          │          │ (uncited)       ▲
                          └────────→ clarify ──────────┘
```

`retrieve` searches the question directly; if that comes back empty or weak it
retries with recent user turns folded in, so a follow-up like *"and when do I get
the money back?"* still finds the right documents (SPEC §6.2's "history-rewritten
query").

Two routes lead to `clarify`, the no-answer path: retrieval found nothing usable,
or `generate` produced an answer it could not cite. An uncited answer is never
returned as grounded — it is dropped and the turn is re-answered without sources,
rather than presenting model priors as knowledge.

## Long-term memory

`load_memory` reads this user's durable facts from the Store; `write_memory`
upserts new ones after the answer. Namespaces are `(user_id, "facts")` and
`(user_id, "preferences")`, built from the authenticated `user_id` only.

Extraction is **deterministic by default** (`MEMORY_EXTRACTION=rules`): explicit
`remember that …` commands plus first-person patterns (`my <attribute> is <value>`,
`I prefer …`). A bad fact is not wrong once — it is injected into every future
prompt on every future thread for that user, so this errs towards precision. Keys
are derived from the normalized attribute, which makes writes idempotent upserts:
restating "my preferred language is Bengali" updates the row rather than adding a
second one (SPEC §7.2). `MEMORY_EXTRACTION=llm` trades that for recall at the cost
of a model call on every turn's write path; `off` disables writes.

Semantic lookup over facts (SPEC §7.2's SHOULD) lands in M4 with the pgvector
store; today the fact set is small and bounded, so `load_memory` lists it.

### Which memory does this belong to?

| Data | Home | Crosses threads? |
|---|---|---|
| "We were discussing invoice 42" | Checkpointer | no |
| "My preferred language is Bengali" | Store | yes, same user only |

The tests assert both directions — a conversational probe must *not* cross
threads, and a durable fact *must*. Using a durable fact to test thread isolation
tests the wrong system; see `.claude/references/memory-placement.md`.

## Seams (deliberately stubbed)

Two dependencies are stubbed behind interfaces so the graph and tests run
end-to-end without external services. Both are marked `>>> SEAM <<<` in source.

- **Vector store** — `VectorStore` in [`src/rag/retrieve.py`](src/rag/retrieve.py),
  seeded from [`src/rag/fixtures.py`](src/rag/fixtures.py). Corpus ingestion is out
  of scope per SPEC §3; a real index implements the same protocol and must return
  `source` and `score` so answers stay citable.
- **Chat model** — `ChatModel` in [`src/rag/llm.py`](src/rag/llm.py). `LLM_PROVIDER=ollama`
  talks to a local Ollama server; `fake` selects a deterministic scripted stand-in
  so the gates run offline; any other value goes through LangChain's
  `init_chat_model`.
