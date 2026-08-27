# src/graph — state, nodes, wiring

For developers and agents changing the graph. Rules that apply everywhere are in
[CLAUDE.md](../../CLAUDE.md); the flow diagram is there too.

| File | Contents |
|---|---|
| `state.py` | `ChatState`, `RetrievedChunk`, the `add_messages` reducer |
| `nodes.py` | Node factories and the two routers |
| `build.py` | `StateGraph` wiring, instrumentation, `compile()` |

## Adding a node

Write a factory that closes over its dependencies and returns the node:

```python
def make_my_node(dep: Thing, settings: Settings) -> Node:
    async def my_node(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        return {"answer": ...}          # only the keys it changes
    return my_node
```

Factories exist so dependencies are injected at wiring time rather than reached
for through module globals — that is what lets a node be unit-tested with a
fabricated `ChatState` and a stub `Runtime`. `runtime` is keyword-only because
LangGraph's `_NodeWithRuntime` protocol declares it that way; a positional
`runtime` runs but fails `mypy --strict`.

Wire it in `build.py`, remove any edge it supersedes, and add a unit test with a
fabricated state (`tests/conftest.py::make_state`, `make_runtime`).

## Why two routers

`route_after_retrieve` sends empty or low-scoring retrieval to `clarify` instead
of letting the model answer from priors.

`route_after_generate` catches the subtler case: the model answered but cited
nothing. That is usually the model correctly declining, because the grounding
prompt tells it to say so rather than guess. An earlier version raised an
exception here, which turned a legitimate question into a 502 — the model was
obeying its instructions. It now drops the reply (no message appended, empty
`answer`) and the router hands the turn to `clarify`. The invariant is unchanged:
an uncited answer never ships as grounded.

## State contract

`messages` appends through its reducer. Every other field is per-turn and
overwrites. Long-term facts arrive in `long_term_facts` each turn from the Store
— they are never written into `messages`, because `messages` is checkpointed per
thread and would strand them there.

## Testing

Unit-test nodes in isolation, then check `tests/test_memory.py` still passes: a
new node changes what is checkpointed after every super-step.
