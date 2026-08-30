# src/api — HTTP layer

For developers and agents changing the endpoints. Request/response shapes are
SPEC §9.

| Endpoint | Notes |
|---|---|
| `POST /chat` | One turn. Body carries `user_id`, `thread_id`, `message`, `mode`. |
| `POST /chat/stream` | The same turn as SSE. See [Streaming](#streaming). |
| `POST /threads` | Mints an id; nothing persists until the first turn. |
| `GET /threads/{id}/history` | Straight from the checkpoint. 404 if unknown. |
| `GET /users/{user_id}/threads` | The user's conversations, from the thread index. |
| `GET /users/{user_id}/facts` | What the Store holds on the user. |
| `DELETE /users/{user_id}/threads/{id}` | One conversation. Durable facts kept. |
| `DELETE /users/{user_id}` | Both memory systems. Idempotent. |
| `GET /threads/{id}/pending?user_id=` | The approval this thread is parked on, or 404. |
| `POST /threads/{id}/resume` | Decide it and finish the turn. See [Approval](#approval). |
| `GET /health` | Liveness, backend, provider, provider reachability. |
| `GET /` | The browser UI. See [The UI](#the-ui). |

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

## Streaming

`POST /chat/stream` runs the same turn as `POST /chat` and must stay
interchangeable with it — a test asserts both produce the same answer and the
same citations. It drives `graph.astream(..., stream_mode=["messages", "values"],
version="v2")`: `messages` parts carry `(chunk, metadata)`, `values` parts carry
the whole state, and the last one is authoritative.

SSE events, one JSON object per `data:` line:

| Event | Meaning |
|---|---|
| `search` | Agent mode only: a query it ran, and the source count so far. |
| `token` | A fragment of the answer. Append it. |
| `restart` | Discard everything drawn so far and start the bubble again. |
| `final` | The authoritative `answer` + `citations`. Always sent last. |
| `error` | The turn failed; `detail` carries the message. |

Three things about it are easy to get wrong:

**`restart` is the grounding contract, not a glitch.** `generate` streams tokens
before anything has checked whether the answer cites a source. When it cannot,
`route_after_generate` sends the turn to `clarify`, which answers again from
scratch. The tokens already on screen are a draft that failed the citation gate,
so the browser must drop them — shipping them would put an uncited answer in
front of the user as though it were grounded.

**Filter by `langgraph_node`.** Under `MEMORY_EXTRACTION=llm`, `write_memory`
also calls the model. Only `generate` and `clarify` may reach the browser;
without the filter the fact extractor's output streams into the answer.

**A non-streaming provider still emits one message event.** `messages` mode emits
messages a node writes to state, not just per-token callbacks, so the scripted
`fake` model arrives as a single `token` carrying the whole answer. Degraded, not
broken — the browser needs no separate path, and neither does `test_web.py`.

`final` is always sent even when tokens streamed, because `citations` can only be
computed once the answer is whole.

`search` events are emitted **only** in agent mode, so a chat turn's stream is
exactly what it was before the mode existed. They are derived from `searches`
growing in the `values` payloads — no extra stream mode. Agent mode can spend
several seconds and two model calls before the first token, and an idle spinner
for that long reads as a hang. They are live-turn only: searches are not
checkpointed as messages, so reopening a thread from history does not replay them.

`mode` defaults to `"chat"`, so callers written before it are unaffected. An
unknown mode is a 422 from the `Mode` literal rather than a silent fallback.
See [src/graph/README.md](../graph/README.md) for the loop and its stop rules.

## When the provider is down

The commonest failure here is not a bug: Ollama binds to `127.0.0.1`, so the
container cannot reach it and every turn dies with `httpx.ConnectError: All
connection attempts failed`. That message tells the reader nothing, and it only
appears at the bottom of a forty-line traceback in the container log.

So a provider failure is translated once, in `llm.py::explain`, and surfaced in
three places: a warning at boot, the SSE `error` event, and a 503 from `/chat`
(503, not 500 — a dependency is down, which is not a defect in this service).
`explain` walks the `__cause__`/`__context__` chain, because providers re-raise
transport errors wrapped and matching only the outermost type quietly falls back
to the useless string.

`unreachable_hint` branches on where the process is running, because the right
advice is the *opposite* in each case and a wrong hint sends the reader off to
fix something that was never broken:

| Running | `OLLAMA_BASE_URL` | Actual cause |
|---|---|---|
| Host | anything | Ollama is not running, or the URL is wrong |
| Container | loopback | The URL names the container itself — see below |
| Container | a host address | Ollama is bound to `127.0.0.1` on the host |

The middle row is the one that misleads, because Ollama's own location is
irrelevant to it: what matters is that the **API** is containerised, so its
`localhost` is the container and can never reach the host.

It used to happen by default. Compose substitutes `${VAR}` from the project's
`.env`, where `OLLAMA_BASE_URL=http://localhost:11434` is correct — for a host
run — and it silently beat the compose default. `docker-compose.yml` now reads
`${DOCKER_OLLAMA_BASE_URL}` instead: a different name so the host setting cannot
collide with the container one, pinned by a test. Leaving the key commented out
in `.env.example` was tried first and lasted about a day; a name that cannot
collide needs no discipline to keep working.

`docker compose config` shows what actually reaches the container.

`GET /health` carries `llm_reachable`, and **must keep returning 200 when it is
false**. Compose health-checks this endpoint; failing it when the model is down
would restart-loop a perfectly healthy API. Liveness of this service and liveness
of the model are two different questions — the UI badge turns amber rather than
green, instead of claiming everything is fine while every turn fails.

## The UI

`web.py` mounts `/static` and serves `templates/index.html` at `/`. Both
directories resolve from `__file__`, because uvicorn and the container start from
different working directories. Nothing but the page lives here — the browser
talks to the same JSON API as any other client.

Tailwind arrives through the **Play CDN**, which compiles in the browser: no node
toolchain in a Python repo, at the cost of needing network on page load. It is
explicitly not meant for production traffic. To swap in a compiled sheet:

```bash
npx @tailwindcss/cli -i src/api/static/app.css -o src/api/static/tailwind.css --minify
```

…then replace the CDN `<script>` in `base.html` with a `<link>` to it. Every
Tailwind class lives in `templates/` — `app.js` only ever clones `<template>`
elements — so that build needs to scan the templates directory alone.

`test_web.py` asserts every `tpl-*` id `app.js` clones exists in the markup. A
renamed template is otherwise a null dereference nothing catches until someone
opens the page.

### Rendering answers

Models answer in Markdown — llama3.1 returns numbered lists and `**bold**`
unprompted — so `static/markdown.js` renders it: headings, nested lists, fenced
and inline code, emphasis, links, quotes, rules. Not tables; they fall through to
paragraphs and the knowledge base has none.

It **builds DOM nodes and never assigns HTML from a string**, which is the whole
reason it exists rather than a CDN parser. Answer text is model output rendered
into the page; with `innerHTML` that is an injection point, and the alternative
is pulling in marked *and* DOMPurify — two more network dependencies on the one
path where a mistake is an XSS hole. A test greps the file for HTML sinks. Links
are scheme-checked too, so `[x](javascript:...)` renders as literal text.

Two behaviours look like bugs and are not: headings shift down a level (`#` → 
`<h2>`, since the page owns the `<h1>`), and an unclosed fence renders as a code
block to the end of the text — which is what a half-streamed one should look
like.

Styling lives in the `.md` rules in `base.html`, not in the script: the renderer
emits bare semantic elements, so every Tailwind class stays in `templates/` and
the compiled-stylesheet swap above still only needs to scan that directory.

### Copy and download

Each code block gets a copy and a download button; each answer gets a copy
button. `app.js` wraps every rendered `<pre>` in the `tpl-code` toolbar after
render, so `markdown.js` stays a plain renderer and the classes stay in
`templates/`. The toolbar is a **sibling** of the `<pre>`, not inside it —
otherwise the button labels would land in the copied snippet.

One delegated listener on the message list handles all three, because answers are
re-rendered on every streamed frame and per-button listeners would be attached
and discarded dozens of times per turn. Buttons are found by data attribute, and
a test pins those names in both files.

Copy puts the **Markdown source** on the clipboard, not the rendered text, so
fences, list markers and `[S1]` citations survive a paste. `navigator.clipboard`
needs a secure context, which this is not when the app is reached on a LAN
address rather than `localhost`, so there is a `execCommand("copy")` fallback.

`[S1]` inside a code block deliberately stays literal rather than becoming a
chip: rewriting the contents of a code block would corrupt what the copy and
download buttons hand back.

### Streaming and repaints

Streamed tokens repaint on `requestAnimationFrame` rather than per token, since
each repaint reparses the whole answer. The frame reads the accumulated text when
it runs, and `final` cancels any frame still queued so it cannot repaint stale
text over the authoritative answer.

## Testing

`TestClient` as a context manager runs the lifespan handler. Tests monkeypatch
`routes.get_settings` to point at a temp SQLite file, so they exercise a real
durable backend without touching a developer's database.

## Approval

A turn can end without an answer: `status: "awaiting_approval"` and a `pending`
payload naming the call. Nothing was done — the run is parked on a checkpoint
until `POST /threads/{id}/resume` decides it.

```json
{ "user_id": "u_123", "call_id": "c_7f", "decision": "approve",
  "edits": {"subject": "Rewritten"} }
```

The status codes carry the failure modes: **403** the thread is not this user's,
**404** nothing is pending, **409** the pending call is a different `call_id` —
the approver is acting on a screen that has moved on — **410** the approval aged
past `APPROVAL_TTL_SECONDS`, in which case it is resumed **as a rejection** so
the checkpoint is not left parked forever.

403 is leak-free here: the check is "is this thread in *your* index", so the
answer is identical for a thread that does not exist and one belonging to
somebody else.

On the stream the turn ends with an `interrupt` event instead of `final`,
preceded by `restart` if anything was drawn — the directive streams like an
answer and is not one. An interrupt does not surface as a stream part, so the
handler asks the checkpoint rather than trusting the stream.

The browser UI does not implement the approval prompt; `TOOLS_ENABLED` is an API
feature for now.
