# LangGraph API — verified surface

Verified 2026-08-27 against the official docs (docs.langchain.com/oss/python/langgraph) via Context7.
`CLAUDE.md` warns these APIs change between releases — **re-verify before trusting this file**, and re-run the memory tests after any LangGraph bump.

---

## ✅ Drift from CLAUDE.md / SPEC.md — RESOLVED in M1

`CLAUDE.md` and `SPEC.md` §6.4 were updated to the current API. Kept below for the record.

`CLAUDE.md` (Conventions) and `SPEC.md` §6.4 document this node signature:

```python
def node(state, config) -> dict          # user_id via config["configurable"]["user_id"]
```

Current LangGraph injects a **`Runtime`** instead:

```python
# `runtime` is KEYWORD-ONLY (langgraph/graph/_node.py::_NodeWithRuntime).
async def node(state: ChatState, *, runtime: Runtime[Context]) -> dict:
    user_id = runtime.context.user_id
    store   = runtime.store
```

What changes, precisely:

| | CLAUDE.md / SPEC.md | Current API |
|---|---|---|
| `thread_id` | `config["configurable"]["thread_id"]` | **unchanged** — still correct |
| `user_id` | `config["configurable"]["user_id"]` | `runtime.context.user_id`, declared via `context_schema` |
| Store access | implicit | `runtime.store` |
| Passed at invoke | inside `config` | separate `context=` kwarg |

`thread_id` staying in `config` is the important half — the SPEC's core memory claim is intact. Only the `user_id` and store plumbing moved. Resolve this in the docs before M3, when `load_memory` / `write_memory` start depending on it.

---

## State

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    retrieved: list[dict]
    long_term_facts: list[str]
    answer: str
```

`add_messages` appends. Returning `{"messages": [new_msg]}` adds one; returning the whole list overwrites and breaks the reducer. Use `RemoveMessage` to delete — it requires a key with the `add_messages` reducer.

Other reducers: `Annotated[list[str], operator.add]` to accumulate, no annotation to overwrite.

## Context schema (for `user_id`)

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str

builder = StateGraph(ChatState, context_schema=Context)
```

## Compile and invoke

```python
graph = builder.compile(checkpointer=checkpointer, store=store)

await graph.ainvoke(
    {"messages": [{"role": "user", "content": text}], "question": text},
    {"configurable": {"thread_id": thread_id}},
    context=Context(user_id=user_id),
)
```

Read a thread's checkpoint directly — the source of truth when debugging:

```python
state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
```

## Checkpointers

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()      # idempotent; creates tables
```

`from_conn_string` is a context manager — the connection closes on exit. Build and hold the graph inside it (FastAPI: the lifespan handler).

## Store

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langchain.embeddings import init_embeddings

store = InMemoryStore(index={"embed": init_embeddings(...), "dims": 1536})

await store.aput(("u-1", "facts"), "stable-key", {"text": "prefers Bengali"})
hits = await store.asearch(("u-1", "facts"), query="what language?", limit=3)
```

Namespace is a tuple. Upsert by **stable key** — a fresh `uuid4()` per turn duplicates facts rather than updating them (SPEC §7.2 idempotency).

## Context-window management

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

messages = trim_messages(
    state["messages"],
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=...,
    start_on="human",
    end_on=("human", "tool"),
)
```

For running summaries there is `langmem.short_term.SummarizationNode` (`RunningSummary` held in a `context` state key) — an extra dependency, relevant to the SPEC §11 long-thread risk in Phase 2.

## Sources

- https://docs.langchain.com/oss/python/langgraph/persistence
- https://docs.langchain.com/oss/python/langgraph/checkpointers
- https://docs.langchain.com/oss/python/langgraph/add-memory
- https://docs.langchain.com/oss/python/langgraph/stores
- https://docs.langchain.com/oss/python/langgraph/graph-api
