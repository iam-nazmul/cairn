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
| M2 — short-term memory: SQLite checkpointer, `/chat` + threads endpoints | not started |
| M3 — long-term memory: Store, `load_memory` / `write_memory` | not started |
| M4 — hardening: Postgres, observability, trimming, deletion APIs | not started |

## Seams (deliberately stubbed)

Two dependencies are stubbed behind interfaces so the graph and tests run
end-to-end without external services. Both are marked `>>> SEAM <<<` in source.

- **Vector store** — `VectorStore` in [`src/rag/retrieve.py`](src/rag/retrieve.py),
  seeded from [`src/rag/fixtures.py`](src/rag/fixtures.py). Corpus ingestion is out
  of scope per SPEC §3; a real index implements the same protocol and must return
  `source` and `score` so answers stay citable.
- **Chat model** — `ChatModel` in [`src/rag/llm.py`](src/rag/llm.py). `LLM_PROVIDER=fake`
  selects a deterministic scripted stand-in so the gates run offline; any other
  value goes to a real provider via LangChain.
