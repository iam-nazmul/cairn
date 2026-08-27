---
name: add-graph-node
description: Add a new node to the LangGraph state graph in src/graph/. Use when adding a step to the START → load_memory → retrieve → generate → write_memory → END flow, or when adding a conditional branch such as the low-confidence clarify path. Covers the node signature, state contract, and wiring.
---

# Adding a graph node

## 1. Write the node in `src/graph/nodes.py`

Nodes are **pure**: they read `state`, return a dict containing **only the keys they change**, and never mutate `state` in place.

```python
from langgraph.runtime import Runtime
from src.graph.state import ChatState
from src.config import Context

async def my_node(state: ChatState, runtime: Runtime[Context]) -> dict:
    user_id = runtime.context.user_id      # NOT config["configurable"]["user_id"]
    store = runtime.store                  # injected when compiled with store=
    ...
    return {"answer": text}                # only changed keys
```

**Signature note:** current LangGraph injects a `Runtime`, not a raw `config`. `CLAUDE.md` still documents the older `def node(state, config)` form — see `.claude/references/langgraph-current-api.md`. `thread_id` is still read from `config["configurable"]`; `user_id` comes from `runtime.context`.

Nodes doing I/O (retrieval, LLM, store) are `async`. Keep blocking calls out of the request path.

## 2. Respect the state contract

`messages` uses the `add_messages` reducer — returning `{"messages": [msg]}` **appends**. Never return the full list; that is the overwrite bug `CLAUDE.md` calls out.

Every other field (`question`, `retrieved`, `long_term_facts`, `answer`) is set per turn and overwrites. If your node needs accumulate-not-overwrite semantics, add an explicit reducer in `src/graph/state.py`:

```python
from typing import Annotated
from operator import add

class ChatState(TypedDict):
    retrieved: Annotated[list[dict], add]   # accumulates across nodes
```

## 3. Wire it in `src/graph/build.py`

```python
builder.add_node("my_node", my_node)
builder.add_edge("retrieve", "my_node")
builder.add_edge("my_node", "generate")
```

For a conditional branch (e.g. retrieval confidence below threshold → clarify):

```python
def route(state: ChatState) -> str:
    if not state["retrieved"] or max(c["score"] for c in state["retrieved"]) < THRESHOLD:
        return "clarify"
    return "generate"

builder.add_conditional_edges("retrieve", route, {"clarify": "clarify", "generate": "generate"})
```

Remember to remove the now-superseded unconditional edge — leaving both makes the graph run `generate` regardless.

## 4. Test it

Unit-test the node in isolation with a fabricated `ChatState` and a stub `Runtime`. Then add it to the graph-level memory tests (`/memory-test`), because a new node changes what gets checkpointed after every super-step.

## Checklist

- [ ] Returns only changed keys; no in-place mutation of `state`
- [ ] `messages` returned as a list to append, never the whole history
- [ ] `async` if it does I/O
- [ ] Reads `user_id` from `runtime.context`, not `config`
- [ ] Wired in `build.py`, superseded edges removed
- [ ] Unit test with fabricated state + memory tests still pass
