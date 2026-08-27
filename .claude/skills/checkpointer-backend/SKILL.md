---
name: checkpointer-backend
description: Add, switch, or debug a checkpointer/store backend (in-memory, SQLite, Postgres) in src/memory/. Use when wiring ENV → backend selection, when persistence works in dev but not local/prod, or when migrating from M1/M2 to M3/M4. Covers setup(), connection lifecycle, and the identical-graph-code rule.
---

# Checkpointer & store backends

The graph code **must be identical across backends** — only the instance handed to `compile()` changes. If a node needs to know which backend it is on, that is a design bug.

## ENV → backend

| `ENV` | Checkpointer | Store | Durable? |
|---|---|---|---|
| `dev` | `InMemorySaver` | `InMemoryStore` | no — resets on restart |
| `local` | SQLite saver | `InMemoryStore` or SQLite | yes, single node |
| `prod` | `PostgresSaver` | `PostgresStore` (pgvector) | yes, concurrent |

Selection belongs in `src/memory/checkpointer.py` and `src/memory/store.py` as factories. Nothing else in the codebase branches on `ENV`.

## In-memory (M1)

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = InMemorySaver()
store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)
```

## Postgres (M4)

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

async with (
    AsyncPostgresStore.from_conn_string(DB_URI) as store,
    AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    await store.setup()          # first run only — creates tables
    await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer, store=store)
```

### The two failure modes that cost the most time

1. **`setup()` never ran.** Tables don't exist; the first invoke fails with a missing-relation error. It is idempotent — safe to call on boot, or run once as a migration step.

2. **The context manager closed before you invoked.** `from_conn_string` yields inside `async with`. Building the graph in that block and then invoking *outside* it gives you a closed connection. In FastAPI, open it in the **lifespan handler** and keep the graph on `app.state` — do not construct per request:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as cp:
        await cp.setup()
        app.state.graph = build_graph(checkpointer=cp, store=...)
        yield
```

## Semantic store (SPEC §7.2)

Long-term facts should be findable by meaning, not just by key:

```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(index={"embed": init_embeddings(...), "dims": 1536})
await store.aput(("u-1", "facts"), "k1", {"text": "prefers answers in Bengali"})
hits = await store.asearch(("u-1", "facts"), query="what language?", limit=3)
```

Namespaces are `(user_id, "facts")` and `(user_id, "preferences")`. Writes are upserts keyed by a stable id — generating a fresh `uuid4()` per turn duplicates facts instead of updating them, which is the idempotency requirement in SPEC §7.2.

## Before you switch or upgrade

- Run `/memory-test` against the **new** backend, not just the old one.
- Do not bump LangGraph without re-running those tests — the checkpointer/store APIs change between releases (`CLAUDE.md`).
- Open question from SPEC §11: whether the store reuses the checkpointer's Postgres with pgvector or a separate vector DB. Confirm before building M3; it determines whether one `DB_URI` or two.
