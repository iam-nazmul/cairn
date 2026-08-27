# src/rag — retrieval, prompts, LLM seam

For developers and agents changing how answers are grounded. Two dependencies are
stubbed behind protocols here so the graph and the gates run without external
services.

| File | Contents |
|---|---|
| `retrieve.py` | `VectorStore` protocol, seeded stub, query augmentation |
| `fixtures.py` | Seed corpus |
| `prompts.py` | Prompt assembly, trimming, citation markers |
| `llm.py` | `ChatModel` protocol, scripted stand-in, provider resolution |

## Seam 1 — the vector store

Corpus ingestion is out of scope (SPEC §3). `InMemoryVectorStore` scores by token
overlap, not embeddings, so results are stable and need no API key.

To point at a real index, implement `VectorStore` and return `source` and `score`
alongside `text` — citation depends on both. Nothing else needs to change.

## Seam 2 — the chat model

`LLM_PROVIDER=fake` selects `DeterministicChatModel`, a scripted stand-in whose
only job is to make the plumbing observable: it reads the history, facts and
context it was handed and reflects them back. It is not a model and does not try
to be one. It exists because the memory tests assert on answer *content*
("What's my name?" → "Alice") and the citation eval asserts on markers, so the
gates cannot depend on a network call.

Anything else goes to a real provider: `ollama` builds `ChatOllama` directly so
`base_url` and `num_ctx` can be passed; other values go through
`init_chat_model`.

## Citations

`generate` emits `[S1]`-style markers. `cited_chunks` maps them back to the
chunks they refer to, which is how `/chat` builds its `citations` array without
adding a field to `ChatState`.

`format_context` drops whole chunks, lowest-ranked first, when capping at
`max_chars` — so a marker never points at a block that was truncated away.

## History-aware retrieval

`retrieve` searches the raw question first. Only if that comes back empty or weak
does it retry with recent turns folded in, so self-contained questions behave
exactly as before.

Only prior **user** turns are folded in. Reusing the assistant's words would
anchor retrieval to whatever it last said, which drags an unrelated follow-up
back to the previous topic.

This is a deterministic stand-in for the "history-rewritten query" SPEC §6.2
allows. It cannot distinguish a follow-up from a topic change — an off-topic
question after an on-topic one may retrieve stale chunks. `route_after_generate`
catches the consequence: the model will not cite them, and the turn falls to
`clarify`. A real standalone-question rewriter would fix the cause.

## Trimming

`assemble_messages` trims history to `max_history_tokens` before the model call:
newest kept, oldest dropped, always starting on a human turn, never returning
nothing. This is a **prompt** budget. The checkpoint keeps every turn and
`/threads/{id}/history` still returns all of them.
