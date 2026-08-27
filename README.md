# cairn

A memory-enabled **RAG chatbot** orchestrated as a **LangGraph** state graph.

**[`SPEC.md`](SPEC.md) is the design document** — architecture, memory model, API
surface and milestones. **[`CLAUDE.md`](CLAUDE.md)** is the working guide for
making changes. Read SPEC.md first.

## The one idea

Two memory systems, deliberately kept separate:

| Concern | Mechanism | Scope |
|---|---|---|
| Conversation history | **Checkpointer** | per `thread_id` |
| Durable user facts | **Store** | per `user_id`, across threads |

The client sends only the new message each turn; prior turns are restored from
the checkpoint by `thread_id`.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

The default `.env` points at a local **Ollama** server:

```bash
ollama serve            # if it is not already running
ollama pull llama3.1    # LLM_MODEL
```

Set `LLM_MODEL` to any model you have pulled (`ollama ls`). Note that thinking
models such as `qwen3` emit their reasoning inline unless configured otherwise,
which shows up in the answer text.

Tests never touch Ollama: they set `LLM_PROVIDER=fake` explicitly, so the gates
stay offline and deterministic. To exercise a real model:

```bash
CAIRN_TEST_OLLAMA=1 uv run pytest -m ollama
```

## Running it

```bash
ENV=local uv run uvicorn src.api.routes:app --reload
```

`ENV` selects the checkpointer: `dev` → in-memory (resets on restart), `local` →
SQLite at `SQLITE_PATH`, `prod` → Postgres (M4). The graph code is identical
across all three.

| Endpoint | Purpose |
|---|---|
| `POST /chat` | One turn. Send **only** the new message — prior turns come from the checkpoint. |
| `POST /threads` | Mint a new `thread_id`. |
| `GET /threads/{id}/history` | Checkpointed history for a thread. |
| `GET /health` | Liveness, plus the active backend and provider. |

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"user_id":"u_1","thread_id":"t_1","message":"How long do I have to submit an expense report?"}'
```

## Quality gates

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src
```

## Status

| Milestone | State |
|---|---|
| M1 — skeleton: `retrieve → generate`, in-memory checkpointer, cited answers | done |
| M2 — short-term memory: SQLite checkpointer, `/chat` + threads endpoints | done |
| M3 — long-term memory: Store, `load_memory` / `write_memory` | not started |
| M4 — hardening: Postgres, observability, trimming, deletion APIs | not started |

## How a turn is routed

```
START → retrieve → generate → END
             ↘         ↘
              clarify ←┘ → END
```

`retrieve` searches the question directly; if that comes back empty or weak it
retries with recent user turns folded in, so a follow-up like *"and when do I get
the money back?"* still finds the right documents (SPEC §6.2's "history-rewritten
query").

Two routes lead to `clarify`, the no-answer path: retrieval found nothing usable,
or `generate` produced an answer it could not cite. An uncited answer is never
returned as grounded — it is dropped and the turn is re-answered without sources,
rather than presenting model priors as knowledge.

## Seams (deliberately stubbed)

Two dependencies are stubbed behind interfaces so the graph and tests run
end-to-end without external services. Both are marked `>>> SEAM <<<` in source.

- **Vector store** — `VectorStore` in [`src/rag/retrieve.py`](src/rag/retrieve.py),
  seeded from [`src/rag/fixtures.py`](src/rag/fixtures.py). Corpus ingestion is out
  of scope per SPEC §3; a real index implements the same protocol and must return
  `source` and `score` so answers stay citable.
- **Chat model** — `ChatModel` in [`src/rag/llm.py`](src/rag/llm.py). `LLM_PROVIDER=ollama`
  talks to a local Ollama server; `fake` selects a deterministic scripted stand-in
  so the gates run offline; any other value goes through LangChain's
  `init_chat_model`.
