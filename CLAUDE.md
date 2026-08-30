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
                           │     │ └─ research ─▶ researcher ⇄ writer ⇄ «sup» │
                           │     ▼                 │                         │
   VectorStore (seam) ───▶ │  retrieve ◀──┐        │                         │
                           │     │        │        │                         │
                           │     ├─ agent ┴ research (rewrite query, search   │
                           │     │          again, merge) -- chat never loops │
                           │     │ └─ empty/weak ──┼──────┐                  │
                           │     ▼                 │      ▼                  │
   ChatModel (seam) ─────▶ │  generate             │   clarify               │
                           │     │ ├─ uncited ─────┼──────┘  │               │
                           │     │ └─ TOOL ──▶ plan ──▶ approve (interrupt)  │
   ToolRegistry (seam) ──▶ │     │              └──▶ act ──▶ generate        │
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
- Streaming does not exempt anything from that: `generate` streams before the
  citation gate runs, so a `generate → clarify` switch must emit `restart` and
  the client must discard the draft. Filter streamed tokens to the `generate` and
  `clarify` nodes — `write_memory` calls a model too.
- Keep `POST /chat/stream` and `POST /chat` interchangeable. A test asserts they
  return the same answer and citations; do not let one grow behaviour.
- Agent mode changes how evidence is gathered, never what may be answered. The
  `RETRIEVAL_MIN_SCORE` floor and the citation requirement are identical in both
  modes; a test pins that. Cap extra searches with `AGENT_MAX_SEARCHES` — each one
  is a model call.
- `mode` goes in `context=` beside `user_id`, never in `configurable`. It is per
  turn: one thread may mix all three.
- `research` mode splits the turn between two subagents. The writer must never be
  given a vector store: an agent that can both search and compose is how a system
  starts citing sources that never reached the answer.
- Evidence ids are minted by the researcher and travel with the chunk. Never
  re-derive a citation from a position in the writer's list.
- Only the writer's final message may append to `messages`. Subagent chatter is
  checkpointed forever if it lands there.
- One supervisor governs both subagents, exactly as one router governs the
  research loop.

**Tools and approval**
- A `Tool` is `effect="write"` unless it says otherwise. Never widen that default;
  never add a tool without a test that it does not run before a resume.
- Nothing observable may happen before `interrupt()` in a node — a resume re-runs
  the node from its first line. Effects belong in `act`, which has no `interrupt()`.
- `act` must stay idempotent per `call_id`. A double-clicked approval sends once.
- Tool output enters `retrieved` as a `tool://` chunk so the existing citation
  gate covers it. Do not add a second grounding path.
- `TOOLS_ENABLED` requires `ENV=local` or `ENV=prod`; `Settings` refuses `dev`.
- Resuming must verify the `user_id` owns the thread, exactly as deletion does.

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
| `src/api/` | HTTP layer, streaming, browser UI | [src/api/README.md](src/api/README.md) |
| `src/tools/` | Tool registry, effect classes, transports | [src/tools/README.md](src/tools/README.md) |
| `.claude/references/` | Verified LangGraph API, memory placement | — |

## Gotchas

- `ruff format .` formats Python inside Markdown. `*.md` is excluded in
  `pyproject.toml`; do not remove that exclusion.
- LangGraph declares `runtime` **keyword-only**: `async def node(state, *, runtime)`.
- Ollama binds to `127.0.0.1`, so containers cannot reach it. Either rebind it
  (`OLLAMA_HOST=0.0.0.0:11434`, needs root) or overlay
  [docker-compose.host-net.yml](docker-compose.host-net.yml) on Linux.
- `OLLAMA_BASE_URL` in `.env` is for host runs only. Compose reads `.env` for
  `${VAR}`, so it takes its override from `DOCKER_OLLAMA_BASE_URL` instead —
  otherwise the host value aims the container at itself.
- `caplog.set_level` makes logging tests pass even when nothing configures
  logging. Assert against the real path (`configure_logging`).
- `/health` must stay 200 when `llm_reachable` is false. Compose health-checks it;
  failing it when the model is down restart-loops a healthy API.
