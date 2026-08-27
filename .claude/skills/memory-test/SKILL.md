---
name: memory-test
description: Write or run the memory tests for the LangGraph chatbot — thread continuity, cross-thread isolation, cross-user store isolation, and durability across restarts. Use when touching src/graph/, src/memory/, checkpointer or store code, or when a conversation "forgets" between turns. CLAUDE.md calls these the critical tests.
---

# Memory tests

Four properties, in dependency order. A failure at level N makes levels above it meaningless.

## 1. Continuity — same `thread_id` remembers

The canonical test. Two invocations, one thread, the second must see the first.

```python
async def test_thread_continuity(graph):
    cfg = {"configurable": {"thread_id": "t-continuity"}}
    ctx = Context(user_id="u-1")

    await graph.ainvoke({"messages": [{"role": "user", "content": "My name is Alice."}],
                         "question": "My name is Alice."}, cfg, context=ctx)
    out = await graph.ainvoke({"messages": [{"role": "user", "content": "What's my name?"}],
                               "question": "What's my name?"}, cfg, context=ctx)

    assert "alice" in out["answer"].lower()
```

Note the client sends **only the new message**. If the test has to resend history to pass, the checkpointer is not wired up — that is the bug, not the assertion.

## 2. Isolation — different `thread_id` does not

```python
async def test_thread_isolation(graph):
    ctx = Context(user_id="u-1")
    await graph.ainvoke({"messages": [{"role": "user", "content": "My name is Alice."}],
                         "question": "My name is Alice."},
                        {"configurable": {"thread_id": "t-a"}}, context=ctx)
    out = await graph.ainvoke({"messages": [{"role": "user", "content": "What's my name?"}],
                               "question": "What's my name?"},
                              {"configurable": {"thread_id": "t-b"}}, context=ctx)

    assert "alice" not in out["answer"].lower()
```

## 3. Cross-user store isolation

Long-term facts are namespaced by `user_id`. A second user must never see the first's facts, **even on a fresh thread**.

```python
async def test_store_user_isolation(store):
    await store.aput(("u-1", "facts"), "k1", {"text": "prefers Bengali"})
    assert await store.asearch(("u-2", "facts"), query="language") == []
```

This is the right-to-be-forgotten surface too: assert that deleting a user's namespace leaves the other user's facts intact.

## 4. Durability — survives process restart

In-memory passes 1–3 and still loses everything on restart. Build a **new** checkpointer instance against the same backing store and re-read the thread:

```python
async def test_durability(tmp_path):
    graph_a = build_graph(checkpointer=sqlite_saver(tmp_path / "cp.db"))
    await graph_a.ainvoke(..., {"configurable": {"thread_id": "t-durable"}}, context=ctx)

    graph_b = build_graph(checkpointer=sqlite_saver(tmp_path / "cp.db"))   # fresh instance
    state = await graph_b.aget_state({"configurable": {"thread_id": "t-durable"}})
    assert state.values["messages"]
```

## Run against every backend

`CLAUDE.md` requires SQLite **and** Postgres. Parametrize rather than duplicating:

```python
@pytest.fixture(params=["sqlite", "postgres"])
def graph(request): ...
```

In-memory is fine for fast unit runs but **never** counts as passing coverage — it cannot fail test 4.

```bash
uv run pytest tests/test_memory.py -q
uv run pytest tests/test_memory.py -q -k postgres
```

## Debugging a failure

| Symptom | First thing to check |
|---|---|
| Second turn forgets | `thread_id` missing from config, or graph compiled without `checkpointer=` |
| History grows but answers ignore it | `generate` not reading `state["messages"]`, or trimming too aggressively |
| History *replaced* each turn | node returned the full `messages` list instead of just the new message |
| Facts leak across users | namespace built from something other than `runtime.context.user_id` |
| Passes in-memory, fails on SQLite/Postgres | `setup()` never called, or the saver's context manager closed before invoke |

Inspect a thread directly with `await graph.aget_state(cfg)` — the checkpoint is the source of truth, not the response.
