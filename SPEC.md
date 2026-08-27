# SPEC: Memory-Enabled RAG Chatbot (LangGraph + Checkpointer)

**Status:** Approved
**Owner:** _Md. Nazmul Hossain_
**Last updated:** 2026-08-27

> **§13 (Agent Mode) added 2026-08-27.** §13.2 is implemented (M5). §13.3 and
> §13.4 are **design only, not built** — no code in the repository implements
> them. Sections 1–12 keep their original numbering because code comments and
> `CLAUDE.md` cite them by number.

---

## 1. Summary

Build a Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in a private knowledge base and **remembers the conversation across turns and sessions**. Memory is provided by LangGraph's persistence layer: a **checkpointer** for short-term, thread-scoped conversation state, and a **store** for long-term, cross-thread facts. The chatbot is orchestrated as a LangGraph state graph so that retrieval, generation, and memory read/write are explicit, inspectable, and resumable.

The graph runs in one of two **modes**, chosen per turn (§13): `chat` retrieves once and answers; `agent` may search repeatedly, rewriting the query from what it found. The mode changes how evidence is gathered — never what may be answered without it.

## 2. Goals

- Answer user questions using retrieved documents rather than model priors alone, with source citations.
- Maintain conversation continuity within a session without re-sending full history from the client — state is restored from the checkpointer by `thread_id`.
- Persist conversations across process restarts and days/weeks later (durable checkpointer).
- Support long-term, cross-conversation memory (e.g. user preferences, stable facts) via a store.
- Be resumable and fault-tolerant: a crashed or interrupted run can continue from the last checkpoint.
- Offer a **multi-step agent mode** that decides for itself whether the evidence it has is enough, and searches again when it is not — under an explicit cost ceiling, and without relaxing the grounding rule (§13.2).

## 3. Non-Goals

- Model fine-tuning or training of custom LLMs.
- ~~Multi-agent orchestration beyond a single retrieval-and-answer graph (may be a later phase).~~ — **AMENDED (§13).** The later phase is now specified. Multi-step routing within one graph is built (§13.2); human-in-the-loop approval (§13.3) and a researcher/writer split (§13.4) are designed and scheduled as M6 and M7. Still out of scope: agents that other systems can call as services, and any topology beyond the ones §13.4 names.
- Building the ingestion/ETL pipeline for the corpus (assumed to exist or covered by a separate spec; see §11 open questions).
- A production-grade auth system — this spec assumes an authenticated `user_id` and `thread_id` are provided by the calling layer. **This bounds §13.3:** an approval is only as trustworthy as the identity of whoever gave it, so the approver is whoever the calling layer already authenticated.

## 4. Definitions

- **Thread** — one conversation. Identified by `thread_id`. All checkpoints for a conversation share this key.
- **Checkpoint** — a serialized snapshot of the graph state saved after each super-step. History is retained (enables time travel and resume).
- **Checkpointer** — the storage backend that writes/reads checkpoints. Provides **short-term, thread-scoped memory**.
- **Store** — a separate key-value/semantic store for **long-term, cross-thread memory** (survives across different `thread_id`s, typically scoped by `user_id`).
- **RAG** — retrieve relevant chunks from a vector index, then condition generation on them.
- **Mode** — `chat` or `agent`, chosen per turn. Travels in `context=`, not in `configurable`, because it is not a property of the thread: one conversation may mix both.
- **Router** — a function on a conditional edge that reads state and names the next node. Routers decide; nodes do work. A router must stay free of side effects, because the same state can be routed more than once.
- **Interrupt** — a pause raised by `interrupt()` inside a node, which suspends the run and persists it to the checkpointer. Resumed by invoking the graph again on the **same `thread_id`** with `Command(resume=...)`. See §13.3.
- **Subagent** — a compiled graph used as a node inside another graph. The unit of composition in §13.4.

## 5. Architecture Overview

```
                 ┌─────────────────────────────────────────────┐
   user turn ──▶ │                 LangGraph app                │
 (thread_id,     │                                             │
  user_id)       │   START                                     │
                 │     │                                       │
                 │     ▼                                       │
                 │  [load_memory] ◀──── Store (long-term)      │
                 │     │                                       │
                 │     ▼                                       │
                 │  [retrieve] ◀─────── Vector store / index   │
                 │     │                                       │
                 │     ▼                                       │
                 │  «route» ─── agent, not enough yet ──┐      │
                 │     │                                ▼      │
                 │     │                       [research] ◀─ LLM
                 │     │  (rewrite query, search again, merge) │
                 │     │                                │      │
                 │     │◀───────────────────────────────┘      │
                 │     │  chat never loops                     │
                 │     ▼                                       │
                 │  [generate] ◀─────── LLM                    │
                 │     │                                       │
                 │     ▼                                       │
                 │  [write_memory] ────▶ Store (long-term)     │
                 │     │                                       │
                 │    END                                      │
                 │                                             │
                 │  state saved every step ─▶ Checkpointer     │
                 └─────────────────────────────────────────────┘
```

Empty or low-confidence retrieval routes to a `clarify` node instead of `generate`, in both modes. §13.2 gives the full routing table.

Two memory systems, deliberately separated:

| Concern | Mechanism | Scope | Example |
|---|---|---|---|
| Conversation history / continuity | **Checkpointer** | Per `thread_id` (short-term) | "What did I just ask?" |
| Durable user facts & preferences | **Store** | Per `user_id`, across threads (long-term) | "User prefers answers in Bengali." |

## 6. Graph Design

### 6.1 State

The graph state is a typed object. Message history uses an append (add-messages) reducer so each node contributes without overwriting.

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]   # full turn history (checkpointed)
    question: str                             # current user query
    retrieved: list[dict]                     # [{text, source, score}, ...]
    long_term_facts: list[str]                # loaded from the Store
    answer: str                               # final grounded answer
    searches: list[str]                       # agent mode: queries tried this turn
    new_hits: int                             # agent mode: sources the last search added
```

> **Amended (M5).** `searches` and `new_hits` were added for agent mode (§13.2).
> Both are **per turn**, but the checkpointer keeps last turn's values, so
> `load_memory` clears them (along with `retrieved`) at the start of every turn.
> Without that reset, a second question merges into chunks found for the first.

### 6.2 Nodes

1. **`load_memory`** — read long-term facts for `user_id` from the Store and place them in state. No-op if none exist.
2. **`retrieve`** — embed the (optionally history-rewritten) query, run similarity search against the vector store, return top-`k` chunks with source metadata and scores.
3. **`generate`** — call the LLM with a prompt assembled from: system instructions, long-term facts, retrieved context, and the checkpointed `messages` history. Produce a grounded answer with citations.
4. **`write_memory`** — optionally extract durable facts/preferences from the turn and upsert them to the Store.
5. **`research`** *(agent mode only, M5)* — ask the model for a better query given what is still missing, search again, and merge the results into `retrieved`. Never reached in chat mode. See §13.2.
6. **`clarify`** *(M1)* — the no-answer path: say what could not be found instead of answering from priors.

### 6.3 Edges

`START → load_memory → retrieve → generate → write_memory → END`

~~(Phase 2 may add a conditional edge after `retrieve`: if retrieval confidence is below a threshold, route to a clarify/no-answer node instead of `generate`.)~~ — **DONE (M1, extended M5).** Two conditional edges now exist:

- after `retrieve` **and** after `research` — the same router on both, deciding `generate` / `clarify` / `research` (§13.2);
- after `generate` — an answer that cites nothing falls through to `clarify` rather than shipping as grounded.

### 6.4 Compilation & invocation

```python
graph = builder.compile(checkpointer=checkpointer, store=store)

config = {"configurable": {"thread_id": thread_id}}

# Client sends ONLY the new message; prior turns are restored from the checkpoint.
result = await graph.ainvoke({"messages": [{"role": "user", "content": text}],
                              "question": text},
                             config=config,
                             context=Context(user_id=user_id))
```

> **Amended at implementation time (M1).** Current LangGraph injects a `Runtime`
> into nodes rather than a raw `config`: `thread_id` still travels in
> `config["configurable"]` (where the checkpointer reads it), but `user_id` now
> travels in a separate `context=` argument, declared to the graph via
> `StateGraph(ChatState, context_schema=Context)` and read inside a node as
> `runtime.context.user_id`. Node signature is `async def node(state, *, runtime)`
> — `runtime` is keyword-only. The core memory claim is unchanged.

> **Amended at implementation time (M5).** `Context` also carries `mode`
> (`"chat" | "agent"`), so the call is
> `context=Context(user_id=user_id, mode=mode)`. Routers that need it take the
> runtime as a **positional** second parameter — `def route(state, runtime)` —
> unlike nodes, where it is keyword-only.

The `thread_id` is the key: same `thread_id` ⇒ same restored conversation state; different `thread_id` ⇒ isolated conversation.

## 7. Memory Requirements

### 7.1 Short-term (checkpointer)

- MUST persist state after every node execution, keyed by `thread_id`.
- MUST restore the latest checkpoint automatically on the next `invoke` for that thread.
- MUST retain checkpoint history to support resume and time-travel/replay.
- Backend selection by environment:
  - **Dev:** in-memory saver — no durability, resets on restart.
  - **Local / staging:** SQLite saver — durable across process restarts, single node.
  - **Production:** Postgres saver — durable, concurrent, horizontally scalable, crash recovery.
- The graph code MUST be identical across backends; only the checkpointer instance changes.

### 7.2 Long-term (store)

- MUST persist facts scoped by `user_id`, readable from any thread.
- SHOULD support semantic lookup (embeddings) so relevant facts can be retrieved, not just exact keys.
- Writes SHOULD be idempotent/upsert to avoid duplicate facts.
- Namespacing convention: `(user_id, "facts")` and `(user_id, "preferences")`.

## 8. Tech Stack

- **Orchestration:** LangGraph (`StateGraph`), compiled with a checkpointer and store.
- **Checkpointer packages:** `langgraph-checkpoint` (base) + `langgraph-checkpoint-sqlite` and/or `langgraph-checkpoint-postgres`.
- **LLM:** pluggable via LangChain chat-model interface (provider configurable).
- **Embeddings + vector store:** configurable (e.g. pgvector / a managed vector DB); MUST return source metadata for citations.
- **API layer:** async HTTP service (e.g. FastAPI) exposing the endpoints in §9.
- **Runtime:** Python 3.11+.

> Pin exact package versions at implementation time and record them in the repo — the checkpointer/store APIs evolve between LangGraph releases.

## 9. API Surface

`POST /chat`

Request:
```json
{ "user_id": "u_123", "thread_id": "t_abc", "message": "...", "mode": "chat" }
```

`mode` is `"chat"` (default) or `"agent"`. It MUST default to `chat`, so callers written before §13 keep their behaviour, and an unrecognised value MUST be rejected rather than silently treated as `chat`.

Response:
```json
{
  "answer": "...",
  "citations": [{ "source": "doc://kb/42", "score": 0.83 }],
  "thread_id": "t_abc"
}
```

Supporting endpoints:
- `POST /chat/stream` — the same turn as Server-Sent Events. In agent mode it also emits a `search` event per query run, because an agent turn can spend several seconds and two model calls before its first token.
- `GET /threads/{thread_id}/history` — return checkpointed message history for a thread.
- `POST /threads` — create a new `thread_id`.
- `GET /health` — liveness/readiness.
- `DELETE /users/{user_id}` — right-to-be-forgotten (§10).

§13.3 adds two more when it is built (`GET`/`POST` on a thread's pending approval); they are specified there, not here, because nothing implements them yet.

## 10. Non-Functional Requirements

- **Latency:** p95 end-to-end response ≤ target (define at implementation; retrieval + one LLM call).
- **Isolation:** no state leakage between `thread_id`s or `user_id`s.
- **Durability:** production checkpointer survives process/host restarts with no lost committed turns.
- **Observability:** log per-node timing, retrieval hits/scores, and token usage; expose graph state for a given checkpoint for debugging.
- **Security/Privacy:** conversation state and long-term facts are user-scoped; support deletion of a user's threads and stored facts (right-to-be-forgotten).
- **Cost control:** cap retrieved-context size and history length passed to the LLM (context-window management / trimming or summarization for long threads).

## 11. Risks & Open Questions

- ~~**Long threads overflow the context window.**~~ — **MITIGATED (M4):** history is trimmed to `MAX_HISTORY_TOKENS` before the model call (`assemble_messages`), newest turns kept, oldest dropped. Trimming is prompt-time only: the full thread stays in the checkpoint and `/threads/{id}/history` still returns all of it. Running-summary compaction remains a Phase 2 option if trimming proves lossy.
- ~~**What extracts long-term facts** in `write_memory`~~ — **RESOLVED (M3): deterministic rules by default, LLM extraction behind a flag.**
  Explicit `remember that ...` commands plus a tight set of first-person patterns
  (`my <attribute> is <value>`, `I prefer ...`). Rationale: a bad fact is not wrong
  once — it is injected into every future prompt on every future thread for that
  user, so precision beats recall. Rules are also deterministic (testable in the
  gates without a model) and yield stable upsert keys derived from the normalized
  attribute, so restating a fact updates it instead of duplicating it (§7.2). LLM
  extraction has the opposite property: the same fact phrased two ways produces two
  keys and two rows. Available as `MEMORY_EXTRACTION=llm`; not the default, because
  it adds a model call to the write path of every turn, including turns containing
  no facts. Accepted cost: rules miss paraphrases they were not written for.
- **Corpus ingestion & re-indexing** is out of scope here — confirm the vector store and metadata schema are owned elsewhere.
- ~~**Citation faithfulness**~~ — **ADDRESSED (M1, extended M4):** `tests/test_citations.py` asserts every answer cites a chunk that retrieval actually returned, and that the cited claim appears verbatim in the chunk it points at. An answer the model cannot cite is dropped and re-answered by the no-answer path rather than shipped as grounded.
- ~~**Store backend**~~ — **RESOLVED (M3): reuse the checkpointer's Postgres with pgvector. One `DATABASE_URL`, not two.**
  Right-to-be-forgotten (§10) spans both memory systems, and M4 must make "delete
  this user's threads *and* facts" verifiable; one database makes that one
  transaction against one credential with one backup story. A separate vector DB
  adds a second failure domain and a second deletion path — precisely where a
  partial delete would hide — for no benefit at this scale. Note this concerns the
  **Store**, not the corpus index: the corpus vector store remains the pluggable
  seam that §3 assigns elsewhere.

## 12. Milestones

1. **M1 — Skeleton:** State graph with `retrieve → generate`, in-memory checkpointer, single-turn answers with citations.
2. **M2 — Short-term memory:** SQLite checkpointer, multi-turn continuity by `thread_id`, `/chat` + history endpoints.
3. **M3 — Long-term memory:** Store integration, `load_memory` / `write_memory` nodes, cross-thread facts.
4. **M4 — Production hardening:** Postgres checkpointer + store, observability, context trimming/summarization, deletion APIs, evals.
5. **M5 — Agent mode (multi-step routing):** ✅ **done.** `chat` / `agent` modes, the `research` node, one router governing the loop, search budget, streamed search events, browser UI. §13.2.
6. **M6 — Tools and human-in-the-loop:** side-effecting tools plus `interrupt()`-based approval before any of them run. §13.3. **Depends on M5.**
7. **M7 — Researcher + writer (multi-agent):** split retrieval from composition into two subagents under a supervisor. §13.4. **Depends on M6** — a researcher that can act needs the approval gate first.
---

## 13. Agent Mode

One principle governs all three phases below, and nothing here may weaken it:

> **An agent may look harder for evidence. It may never answer with less.**

Agent mode changes *how* evidence is gathered — how many searches, by which
subagent, with which approvals. It does not change the `RETRIEVAL_MIN_SCORE`
floor, the requirement that every claim carry a citation marker, or the rule that
an uncited answer is dropped rather than shipped as grounded (§2, §11). A test
asserts the grounding floor is byte-identical across modes; any future phase MUST
extend that test rather than exempt itself from it.

### 13.1 Modes

| Mode | Behaviour | Cost per turn |
|---|---|---|
| `chat` | Retrieve once, then answer. | 1 model call |
| `agent` | Retrieve, judge, optionally rewrite the query and search again, then answer from everything gathered. | 1 + N model calls, N ≤ `AGENT_MAX_SEARCHES` − 1 |

Mode is chosen **per turn**, not per thread. A conversation may mix both, and the
checkpoint records which mode produced each answer. It travels in `context=`
(§6.4) rather than `configurable`, because `configurable` is where thread
identity lives and mode is not a property of the thread.

### 13.2 Multi-step routing — *implemented (M5)*

A router reads state and names the next node; the loop is an edge back to a node
that does work. This is the shape the user story asks for: *a node decides, then
execution moves to a different node.*

```
retrieve ──▶ «route_after_retrieve» ──▶ generate    (evidence is good enough)
                    ▲         └────────▶ clarify     (nothing worth answering from)
                    │         └────────▶ research    (agent mode: try again)
                    │                        │
                    └────────────────────────┘
```

**The same router is bound to both `retrieve` and `research`.** This is a
requirement, not an implementation detail: a second router for the loop would be
free to drift from the first, and the thing it would drift on is the grounding
verdict. One router means the budget and the answer/refuse decision cannot
disagree.

`research` MUST be the only node that rewrites a query, and it MUST NOT call the
model that produces the answer with the answer prompt — it asks for a search
query and nothing else.

**Stop conditions.** Agent mode stops searching when any holds:

| Condition | Rationale |
|---|---|
| best score ≥ `AGENT_GOOD_SCORE` | Retrieval already worked; another rewrite re-finds what is there. |
| `len(searches)` ≥ `AGENT_MAX_SEARCHES` | The cost ceiling for a turn (§10). Counts the first search, so `1` makes agent mode behave exactly like chat. |
| `new_hits == 0` | The last rewrite surfaced no new source, so refining has stopped paying for itself. |

Then the ordinary verdict applies: below `RETRIEVAL_MIN_SCORE`, `clarify`.

**Merging.** Chunks from several searches MUST be deduplicated by `source`,
keeping the best score. This is a correctness requirement, not tidiness: the same
document reached by two queries would otherwise occupy two `[S]` blocks, and the
answer would cite two different numbers for one source.

**Observability.** Each search is streamed to the client as it runs (§9). An
agent turn can spend several seconds and two model calls before its first token,
and silence for that long is indistinguishable from a hang.

**Known limitation.** On a small corpus the first search usually finds everything
there is, and the extra model call buys nothing. Agent mode is worth its cost on
vague or many-sided questions over a large corpus; it is not a better default.
Hence per-turn selection rather than a global setting.

### 13.3 Human-in-the-loop approval — *designed, NOT built (M6)*

> **Nothing in the repository implements this.** Agent mode today is read-only:
> it searches a vector index and writes durable facts, and neither warrants an
> approval prompt. **This phase is meaningless until side-effecting tools exist**,
> which is why M6 is scoped as *tools **and** approval*, not approval alone. A
> gate with nothing behind it is theatre.

**Requirement.** Before executing any tool marked as having external effects, the
graph MUST suspend, surface what it intends to do, and wait for an explicit human
decision. It MUST NOT perform the effect and then ask.

**Mechanism.** LangGraph's `interrupt()` (`langgraph.types`), which suspends the
run and persists it to the checkpointer. The run is resumed by invoking the graph
on the **same `thread_id`** with `Command(resume=<decision>)`; the resume value
becomes the return value of `interrupt()` inside the node.

```python
from langgraph.types import Command, interrupt

decision = interrupt({
    "action": "send_email",
    "to": to, "subject": subject, "body": body,
})
if decision.get("approve") is not True:
    return "cancelled by the user"
# the effect happens only past this line
```

**Three constraints that are easy to get wrong:**

1. **Resuming re-runs the node from its start**, not from the `interrupt()` call.
   Any work before `interrupt()` executes a second time. Therefore a node
   containing `interrupt()` MUST do nothing observable before it — no writes, no
   sends, no counters. Put the effect strictly *after* the call.
2. **A durable checkpointer is mandatory.** Under `ENV=dev` the in-memory saver
   loses a pending approval on restart, silently abandoning the turn. §7.1's
   backend table therefore gains a rule: **approval flows require `ENV=local` or
   `ENV=prod`.**
3. **An approval is scoped to the thread it was raised on.** Resuming MUST verify
   the `user_id` owns that thread, exactly as `DELETE /users/{id}/threads/{id}`
   does — otherwise anyone who guesses a `thread_id` can approve someone else's
   pending action. §3's authentication boundary defines who the approver is.

**API surface** (added to §9 when built):

| Endpoint | Purpose |
|---|---|
| `GET /threads/{thread_id}/pending` | The interrupt payload awaiting a decision, or 404. |
| `POST /threads/{thread_id}/resume` | Body carries the decision; resumes the run. |

`POST /chat` and `POST /chat/stream` gain a terminal state: a turn may end
`awaiting_approval` instead of returning an answer. Clients MUST handle it; the
stream gains an `interrupt` event carrying the payload.

**Open questions.**
- What marks a tool as risky — a decorator, a registry, or a per-tool flag? A default of "risky unless declared safe" fails closed and is the safer default.
- Does an approval expire? A checkpoint pending for a week is a stale intention that may no longer be wanted.
- Is an edited approval allowed (approve *but change the recipient*), or only yes/no? Editing is more useful and materially harder to audit.

### 13.4 Researcher + writer — *designed, NOT built (M7)*

> **Nothing in the repository implements this.** Scheduled after M6 because a
> researcher that can act needs the approval gate to exist first.

**Shape.** Two subagents — compiled graphs used as nodes — under a supervisor
that decides which runs next and when the work is done.

```
        ┌── supervisor ──┐
        │                │
        ▼                ▼
   researcher   ────▶  writer  ────▶ generate/END
   (search, gather,     (compose the answer from
    judge sufficiency)   what the researcher gathered)
```

**Division of labour.** The researcher owns retrieval and decides when the
evidence is sufficient — essentially §13.2's loop, extracted. The writer owns
composition and citation, and **must not retrieve**: giving both agents the
ability to search is how a system starts citing sources that never reached the
answer.

**State boundary.** The two MUST NOT share one flat state. The researcher's
intermediate queries and rejected chunks are its own business; what crosses to
the writer is the merged, deduplicated evidence set and nothing else. Shared keys
written by a subgraph require a reducer declared in the parent (LangGraph rule),
and `messages` in particular must not accumulate both agents' internal chatter —
that is checkpointed per thread and would be replayed into every later turn.

**Citation provenance is the hard part.** With one agent, `[S1]` indexes the list
`generate` was handed. With two, the writer cites evidence the researcher chose,
and the mapping must survive the handoff intact. The existing citation eval
(§11) MUST be extended to assert that a cited chunk was actually retrieved by the
researcher in that turn — not merely that a marker exists.

**Cost.** A supervisor turn is at minimum three model calls (supervise, research,
write) against agent mode's two and chat's one. `AGENT_MAX_SEARCHES` bounds the
researcher; the supervisor needs its own hand-off ceiling, or two agents can pass
work back and forth indefinitely.

**Open questions.**
- Supervisor, or a fixed `researcher → writer` sequence? A fixed sequence is cheaper and fully deterministic; a supervisor is only worth it if the writer can genuinely send work back.
- Does the writer inherit long-term facts (§7.2), or only the researcher? Facts shape tone and language, which is the writer's concern — but they are also retrieval hints, which is the researcher's.
- Is this mode a third value of `mode`, or does it replace `agent`? A third value keeps the migration reversible.
