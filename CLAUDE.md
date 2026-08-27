# CLAUDE.md

Working rules for this repository. `SPEC.md` is the design; this file is what you
must do. Module-level guidance lives in each module's `README.md`.

## Architecture

```
 POST /chat                ┌─ LangGraph app ─────────────────────────────────┐
 {user_id,                 │                                                 │
  thread_id,  ──────────▶  │   START                                         │
  message}                 │     │                                           │
                           │     ▼                                           │
                           │  load_memory ◀────────┐                         │
                           │     │                 │                         │
                           │     ▼                 │                         │
   VectorStore (seam) ───▶ │  retrieve             │                         │
                           │     │ └─ empty/weak ──┼──────┐                  │
                           │     ▼                 │      ▼                  │
   ChatModel (seam) ─────▶ │  generate             │   clarify               │
                           │     │ └─ uncited ─────┼──────┘  │               │
                           │     ▼                 │         │               │
                           │  write_memory ────────┘◀────────┘               │
                           │     │                                           │
                           │    END                                          │
                           └─────┼───────────────────────┼───────────────────┘
                                 │                       │
                    after every node                  read/write
                                 ▼                       ▼
                    ┌────────────────────┐   ┌──────────────────────────┐
                    │   CHECKPOINTER     │   │         STORE            │
                    │   short-term       │   │         long-term        │
                    │   key: thread_id   │   │   key: user_id           │
                    │   conversation     │   │   (user_id,"facts")      │
                    │   history          │   │   (user_id,"preferences")│
                    │                    │   │   (user_id,"threads")    │
                    └────────────────────┘   └──────────────────────────┘
                       ENV=dev    in-memory        in-memory
                       ENV=local  SQLite           in-memory
                       ENV=prod   Postgres ◀── same DATABASE_URL ──▶ Postgres
```

Two memory systems, deliberately separate. Conversation history is restored by
`thread_id`; durable user facts are scoped to `user_id` and cross threads.

## Commands

```bash
uv sync --extra dev
cp .env.example .env

uv run uvicorn src.api.routes:app --reload     # ENV picks the backend
docker compose up --build                      # API + Postgres

uv run pytest                                  # gates -- all four must pass
uv run ruff check .
uv run ruff format .
uv run mypy src
```

Full suite, including the opt-in Postgres and Ollama tests:

```bash
docker compose up -d
POSTGRES_TEST_URI=postgresql://cairn:cairn@localhost:5433/cairn?sslmode=disable \
  CAIRN_TEST_OLLAMA=1 uv run pytest
```

## Rules

**Invocation**
- Always pass `thread_id` in `config["configurable"]`. Never invoke without one —
  memory silently fails to persist.
- Pass `user_id` in `context=`, not in `configurable`. Read it as
  `runtime.context.user_id`.
- Send only the new message. Never resend history from the client.

**State**
- Return only the keys your node changes. Never mutate `state` in place.
- Return `{"messages": [new_msg]}` to append. Returning the whole list overwrites
  and breaks the reducer.

**Memory boundaries**
- Conversation history → checkpointer. Durable user facts → Store. Never cross them.
- Build Store namespaces from `runtime.context.user_id` only, never a request field.
- Add a new place user data lands → add it to `USER_NAMESPACES` in
  `src/memory/threads.py`, or deletion silently becomes partial.

**Retrieval and grounding**
- `retrieve` must return `{text, source, score}` per chunk.
- An answer that cites nothing must never ship as grounded. Empty or weak
  retrieval routes to `clarify`, never to model priors.

**Dependencies**
- Do not bump `langgraph` or any `langgraph-checkpoint-*` package without
  re-running `tests/test_memory.py` against SQLite **and** Postgres. The
  checkpointer APIs change between releases.

**Writing code here**
- Keep comments minimal. One line for a non-obvious *why*; nothing that restates
  the code. Long rationale belongs in the module README.
- In docstrings on an implementation, point at the parent/protocol method
  (`See VectorStore.search.`) instead of restating its contract.
- Nodes that do I/O are `async`. Keep blocking calls off the request path.

## Where things live

| Path | What | Read before changing |
|---|---|---|
| `src/graph/` | State, nodes, wiring | [src/graph/README.md](src/graph/README.md) |
| `src/memory/` | Checkpointer, Store, facts, deletion | [src/memory/README.md](src/memory/README.md) |
| `src/rag/` | Retrieval, prompts, LLM seam | [src/rag/README.md](src/rag/README.md) |
| `src/api/` | HTTP layer | [src/api/README.md](src/api/README.md) |
| `.claude/references/` | Verified LangGraph API, memory placement | — |

## Gotchas

- `ruff format .` formats Python inside Markdown. `*.md` is excluded in
  `pyproject.toml`; do not remove that exclusion.
- LangGraph declares `runtime` **keyword-only**: `async def node(state, *, runtime)`.
- Ollama binds to `127.0.0.1`, so containers cannot reach it. See
  [docker-compose.yml](docker-compose.yml).
- `caplog.set_level` makes logging tests pass even when nothing configures
  logging. Assert against the real path (`configure_logging`).
