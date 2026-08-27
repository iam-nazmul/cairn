# CLAUDE.md

Working context for this repository. Read `SPEC.md` for the full design; this file is the practical guide for making changes.

## What this is

A memory-enabled **RAG chatbot** orchestrated as a **LangGraph** state graph. Two memory systems, kept separate on purpose:

- **Checkpointer** → short-term, thread-scoped conversation state. Saved after every node, restored automatically by `thread_id`. The client sends only the new message each turn.
- **Store** → long-term, cross-thread facts/preferences, scoped by `user_id`.

Graph flow: `START → load_memory → retrieve → generate → write_memory → END`.

## Stack

- Python 3.11+
- LangGraph (`StateGraph`) — orchestration
- `langgraph-checkpoint` + `langgraph-checkpoint-sqlite` / `langgraph-checkpoint-postgres`
- LangChain chat-model + embeddings interfaces (provider configurable)
- Vector store with source metadata (pgvector or managed DB)
- FastAPI (async) for the HTTP layer

> Package versions are pinned in `pyproject.toml`. The checkpointer/store APIs change between LangGraph releases — **do not upgrade LangGraph without re-running the memory tests.**

## Layout

```
src/
  graph/
    state.py        # ChatState TypedDict + reducers
    nodes.py        # load_memory, retrieve, generate, write_memory
    build.py        # StateGraph wiring + compile()
  memory/
    checkpointer.py # backend factory (memory | sqlite | postgres)
    store.py        # long-term store factory
  rag/
    retrieve.py     # embedding + similarity search
    prompts.py      # prompt assembly
  api/
    routes.py       # /chat, /threads, /health
  config.py         # env-driven settings
tests/
```

## Commands

```bash
# setup
uv sync                       # or: pip install -e ".[dev]"
cp .env.example .env          # set LLM/embeddings keys, DB URLs, ENV

# run
uv run uvicorn src.api.routes:app --reload

# quality gates (run before committing)
uv run pytest                 # tests
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src               # types
```

`ENV` selects the checkpointer backend: `dev` → in-memory, `local` → SQLite, `prod` → Postgres. Graph code is identical across all three; only the backend instance differs.

## How the graph works

- **State** lives in `graph/state.py`. `messages` uses the **add-messages reducer** — nodes *append*, they never overwrite the list. Other fields (`question`, `retrieved`, `long_term_facts`, `answer`) are set per turn.
- **Invocation always passes `thread_id` in config and `user_id` in context:**
  ```python
  config = {"configurable": {"thread_id": thread_id}}
  await graph.ainvoke({"messages": [{"role": "user", "content": text}], "question": text},
                      config, context=Context(user_id=user_id))
  ```
  `user_id` moved out of `configurable` into `context=` in current LangGraph; see
  `.claude/references/langgraph-current-api.md`.
  Same `thread_id` = same restored conversation. Different `thread_id` = isolated. **Never** invoke without a `thread_id` — memory silently won't persist.
- **Compile** with both systems: `builder.compile(checkpointer=checkpointer, store=store)`.

## Conventions

- **Adding a node:** write a pure `async def node(state, *, runtime) -> dict` (`runtime` is keyword-only; `user_id` comes from `runtime.context`) returning only the keys it changes; wire it in `graph/build.py`; never mutate `state` in place.
- **Memory boundaries:** conversation history comes from the checkpointer — do **not** hand-roll a history table or make the client resend past turns. Durable user facts go through `memory/store.py`, never into the checkpointed `messages`.
- **Retrieval:** `retrieve` must return `{text, source, score}` per chunk so `generate` can cite. Answers without citations are a bug.
- **Grounding:** `generate` answers from retrieved context + history + long-term facts. Do not let it fall back to model priors when retrieval is empty — route to a clarify/no-answer path instead (see SPEC §6.3).
- **Context budget:** cap retrieved-context size and trim/summarize long threads before the LLM call. Long threads overflowing the window is a known risk (SPEC §11).
- **Async:** API and I/O-bound nodes are async. Keep blocking calls out of the request path.

## Testing

- Unit-test each node in isolation with a fabricated `ChatState`.
- **Memory tests are the critical ones:** invoke twice on the same `thread_id` and assert the second turn sees the first (e.g. "My name is Alice" → "What's my name?"). Assert two different `thread_id`s stay isolated. Run these against SQLite *and* Postgres backends.
- Add a grounding/citation eval: verify answers reference retrieved chunks, not priors.

## Don't

- Don't remove `thread_id` from invocation config.
- Don't overwrite `messages` (respect the reducer).
- Don't store long-term facts in the checkpointed conversation state, or conversation history in the Store.
- Don't bump the LangGraph version casually — pin and test.
- Don't leak state across users/threads; deletion of a user's threads and facts must stay supported (right-to-be-forgotten).

## Open decisions

Tracked in `SPEC.md` §11 — how long-term facts get extracted in `write_memory`, and whether the Store reuses the checkpointer's Postgres (pgvector) or a separate vector DB. Confirm these before building M3.