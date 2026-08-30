# src/graph — state, nodes, wiring

For developers and agents changing the graph. Rules that apply everywhere are in
[CLAUDE.md](../../CLAUDE.md); the flow diagram is there too.

| File | Contents |
|---|---|
| `state.py` | `ChatState`, `RetrievedChunk`, the `add_messages` reducer |
| `nodes.py` | Node factories and the routers |
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

## Agent mode: the research loop

`runtime.context.mode` is `"chat"` or `"agent"`. It is per **turn**, not per
thread — one conversation can mix both, and the checkpoint records which produced
each answer.

Chat mode is unchanged: `retrieve` once, then answer. Agent mode may loop
`research → retrieve's router → research`, where `research` asks the model for a
better query, searches again, and merges the results.

`route_after_retrieve` is attached to **both** `retrieve` and `research`, so the
loop re-enters the same decision. That is deliberate: a second router for the
loop could drift from the first, and the thing it would drift on is the grounding
verdict. Agent mode stops and answers when any of these holds:

| Condition | Why |
|---|---|
| best score ≥ `AGENT_GOOD_SCORE` | retrieval already worked |
| searches ≥ `AGENT_MAX_SEARCHES` | budget spent — each extra search is a model call |
| `new_hits == 0` after a rewrite | refining stopped surfacing anything new |

Then the same floor applies as in chat mode: below `RETRIEVAL_MIN_SCORE` it goes
to `clarify`. **Searching harder never lowers the bar for what may be answered** —
a test asserts the floor is identical in both modes.

Two things are easy to get wrong here:

`_merge_chunks` deduplicates by `source`, keeping the best score. Without it the
same document reached by two queries becomes two `[S]` blocks, and the answer
cites two numbers for one source.

`load_memory` resets `retrieved`, `searches` and `new_hits`. They are per-turn,
but the checkpoint holds last turn's values, so agent mode would otherwise merge
into chunks found for a different question. It runs first on every turn, which is
why the reset lives there rather than in the request the API builds.

## Tools and approval

Off unless `TOOLS_ENABLED=1`, and refused outright under `ENV=dev` — an
in-memory checkpointer drops a pending approval on restart, which abandons the
turn silently.

```
generate ── tool_request ──▶ plan ── effect=read  ──▶ act ──▶ generate
                              │  └─ effect=write ──▶ approve ──▶ act
                              └─ unknown/over budget ──▶ clarify
```

Four things here are load-bearing:

**`generate` proposes nothing.** It emits a `TOOL name {json}` directive as its
whole reply; `plan` is the only node that resolves one into a call. A directive
is not an answer, so it never reaches `messages` and never ships as one.

**`approve` does nothing before `interrupt()`.** A resume re-runs the node from
its first line, so anything above that call happens twice. The effect lives in
`act`, one edge later, which contains no `interrupt()` at all.

**`act` refuses a `call_id` it already performed.** Belt and braces for the same
replay hazard: a double-clicked approval must not send twice.

**Tool output is evidence.** `act` merges the result into `retrieved` as a
`tool://<name>/<call_id>` chunk with score `1.0`, so the ordinary citation gate
applies to a tool-derived answer unchanged. A second grounding path would be free
to drift from the first — the same argument as the single router above.

`tool_request`, `pending_action` and `tool_calls` reset in `load_memory` with the
rest of the per-turn state, which is what makes `TOOL_MAX_CALLS` a per-turn
budget rather than a lifetime one for the thread.

Known limitation: tools sit behind `generate`, so a turn whose retrieval was too
weak goes to `clarify` and never gets the chance to ask for one. Actions that
follow from retrieved content work; a bare "send this email" on an off-corpus
thread does not.

## State contract

`messages` appends through its reducer. Every other field is per-turn and
overwrites. Long-term facts arrive in `long_term_facts` each turn from the Store
— they are never written into `messages`, because `messages` is checkpointed per
thread and would strand them there.

## Testing

Unit-test nodes in isolation, then check `tests/test_memory.py` still passes: a
new node changes what is checkpointed after every super-step.
