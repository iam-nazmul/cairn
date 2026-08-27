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

### Known gap (for M2)

`retrieve` searches the raw question, so a follow-up that depends on the previous
turn ("and when do I get the money back?") retrieves nothing and falls to the
clarify path. SPEC §6.2 already allows for this — "embed the *(optionally
history-rewritten)* query" — and multi-turn continuity is M2's job.
| M3 — long-term memory: Store, `load_memory` / `write_memory` | not started |
| M4 — hardening: Postgres, observability, trimming, deletion APIs | not started |

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
