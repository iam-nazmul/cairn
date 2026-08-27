# src/api — HTTP layer

For developers and agents changing the endpoints. Request/response shapes are
SPEC §9.

| Endpoint | Notes |
|---|---|
| `POST /chat` | One turn. Body carries `user_id`, `thread_id`, `message`. |
| `POST /threads` | Mints an id; nothing persists until the first turn. |
| `GET /threads/{id}/history` | Straight from the checkpoint. 404 if unknown. |
| `DELETE /users/{user_id}` | Both memory systems. Idempotent. |
| `GET /health` | Liveness, plus active backend and provider. |

## The graph is built once, in the lifespan handler

`checkpointer_scope` and `store_scope` are context managers whose connection
closes on exit. Building the graph per request would hand each request a closed
connection — the single most expensive mistake to debug here. The graph,
checkpointer and store live on `app.state`.

`configure_logging` also runs there; without it the per-node instrumentation
never emits.

## Invocation

`thread_id` goes in `config["configurable"]`, `user_id` in `context=`. The
request body carries both, but only the new message:

```python
await graph.ainvoke(
    {"messages": [{"role": "user", "content": body.message}], "question": body.message},
    {"configurable": {"thread_id": body.thread_id}},
    context=Context(user_id=body.user_id),
)
```

History is never resent by the client, and `/threads/{id}/history` reads
`aget_state` rather than any table of our own.

## Citations

`citations` is derived per response from the markers the answer actually used, so
an uncited answer returns an empty list rather than the full retrieved set.

## Testing

`TestClient` as a context manager runs the lifespan handler. Tests monkeypatch
`routes.get_settings` to point at a temp SQLite file, so they exercise a real
durable backend without touching a developer's database.
