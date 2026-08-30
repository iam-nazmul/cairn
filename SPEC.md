# SPEC: Memory-Enabled RAG Chatbot (LangGraph + Checkpointer)

**Status:** Approved
**Owner:** _Md. Nazmul Hossain_
**Last updated:** 2026-08-27

> **§13 (Agent Mode) added 2026-08-27.** §13.2 (multi-step routing) and §13.3
> (tools and approval) are implemented, as M5 and M6. §13.4 (researcher + writer)
> is **design only, not built** — no code in the repository implements it; its
> open questions are closed, so it is buildable as written and carries a
> definition of done. Fields and endpoints it introduces are marked *(M7)*
> wherever they appear in §§4–12. Sections 1–12 keep their original numbering
> because code comments and `CLAUDE.md` cite them by number.

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
- ~~Multi-agent orchestration beyond a single retrieval-and-answer graph (may be a later phase).~~ — **AMENDED (§13).** The later phase is now specified. Multi-step routing (§13.2) and human-in-the-loop approval over side-effecting tools (§13.3) are built; the researcher/writer split (§13.4) is designed and scheduled as M7. Still out of scope: agents that other systems can call as services, and any topology beyond the ones §13.4 names.
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
- **Tool** — a named function the model may ask to run, declared in a registry with an `effect` of `read` or `write`. `write` means the effect leaves the process and needs approval; it is the default (§13.3).
- **Approval** — a human decision on one proposed `write` tool call, identified by its `call_id` and scoped to the thread it was raised on.
- **Subagent** — a compiled graph used as a node inside another graph. The unit of composition in §13.4.
- **Evidence** — the deduplicated `{id, text, source, score}` set a researcher hands to a writer. The `id` is what makes a citation traceable across the handoff (§13.4). *(M7)*

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

    tool_request: str                         # the raw TOOL directive generate emitted
    pending_action: dict | None               # tool call awaiting approval
    tool_calls: list[dict]                    # calls decided this turn; makes act idempotent
    evidence: list[Evidence]                  # (M7) the researcher -> writer handoff
```

> **Amended (M5).** `searches` and `new_hits` were added for agent mode (§13.2).
> Both are **per turn**, but the checkpointer keeps last turn's values, so
> `load_memory` clears them (along with `retrieved`) at the start of every turn.
> Without that reset, a second question merges into chunks found for the first.

> **Amended (M6).** `tool_request` was added at implementation time: `generate`
> signals that it wants a tool, and `plan` — the only node allowed to propose an
> effect — resolves the directive into a call. All three tool fields reset in
> `load_memory` with the rest of the per-turn state, which is what makes
> `TOOL_MAX_CALLS` a per-turn budget rather than a lifetime one for the thread.
> They live in the checkpoint, never the Store (§7).

> **Planned (M7).** `evidence` does not exist yet. It is the *only* key a
> researcher subgraph may write into parent state (§13.4).

### 6.2 Nodes

1. **`load_memory`** — read long-term facts for `user_id` from the Store and place them in state. No-op if none exist.
2. **`retrieve`** — embed the (optionally history-rewritten) query, run similarity search against the vector store, return top-`k` chunks with source metadata and scores.
3. **`generate`** — call the LLM with a prompt assembled from: system instructions, long-term facts, retrieved context, and the checkpointed `messages` history. Produce a grounded answer with citations.
4. **`write_memory`** — optionally extract durable facts/preferences from the turn and upsert them to the Store.
5. **`research`** *(agent mode only, M5)* — ask the model for a better query given what is still missing, search again, and merge the results into `retrieved`. Never reached in chat mode. See §13.2.
6. **`clarify`** *(M1)* — the no-answer path: say what could not be found instead of answering from priors.
7. **`plan` / `approve` / `act`** *(M6)* — propose a tool call, suspend for a human decision on a `write` effect, then perform it exactly once. `act` merges the result into `retrieved` and hands the turn back to `generate`. §13.3.
8. **`researcher` / `writer`** *(M7, not built)* — compiled subgraphs used as nodes, sequenced by a supervisor router. §13.4.

### 6.3 Edges

`START → load_memory → retrieve → generate → write_memory → END`

~~(Phase 2 may add a conditional edge after `retrieve`: if retrieval confidence is below a threshold, route to a clarify/no-answer node instead of `generate`.)~~ — **DONE (M1, extended M5).** Two conditional edges now exist:

- after `retrieve` **and** after `research` — the same router on both, deciding `generate` / `clarify` / `research` (§13.2);
- after `generate` — an answer that cites nothing falls through to `clarify` rather than shipping as grounded.

**DONE (M6).** The second of these gained a `plan` destination rather than a third router being added, and two routers now govern the tool path: `route_after_plan` (`approve` for a `write` effect, `act` for a `read` one, `clarify` for a call the registry does not have) and `route_after_approval` (`act` when the human approved, `clarify` when they did not). `act → generate` closes the loop, bounded by `TOOL_MAX_CALLS`.

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
- **Approval flows (§13.3) require `ENV=local` or `ENV=prod`.** The in-memory saver loses a pending approval on restart, silently abandoning the turn; `Settings` refuses `TOOLS_ENABLED` under `ENV=dev` rather than discovering this at the first interrupt.

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

`mode` is `"chat"` (default) or `"agent"` — and `"research"` once §13.4 is built. It MUST default to `chat`, so callers written before §13 keep their behaviour, and an unrecognised value MUST be rejected rather than silently treated as `chat`. That rejection is what lets §13.4 add a third mode without breaking older clients.

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

- `GET /threads/{thread_id}/pending?user_id=…` — the approval this thread is parked on, or 404 (§13.3).
- `POST /threads/{thread_id}/resume` — decide it and finish the turn (§13.3).

`POST /chat` has a second terminal state: `{"status": "awaiting_approval", "pending": {…}}` instead of an answer. `status` defaults to `"answered"`, so a caller that ignores it is unaffected until tools are switched on.

## 10. Non-Functional Requirements

- **Latency:** p95 end-to-end response ≤ target (define at implementation; retrieval + one LLM call).
- **Isolation:** no state leakage between `thread_id`s or `user_id`s.
- **Durability:** production checkpointer survives process/host restarts with no lost committed turns.
- **Observability:** log per-node timing, retrieval hits/scores, and token usage; expose graph state for a given checkpoint for debugging.
- **Security/Privacy:** conversation state and long-term facts are user-scoped; support deletion of a user's threads and stored facts (right-to-be-forgotten).
- **Cost control:** cap retrieved-context size and history length passed to the LLM (context-window management / trimming or summarization for long threads). Every loop the graph can enter MUST carry a ceiling: `AGENT_MAX_SEARCHES` (§13.2), `TOOL_MAX_CALLS` (§13.3), `SUPERVISOR_MAX_HANDOFFS` (§13.4).

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
6. **M6 — Tools and human-in-the-loop:** ✅ **done.** Tool registry with effect classes, `plan` / `approve` / `act`, `interrupt()`-based approval, per-turn tool budget, pending/resume endpoints, `interrupt` stream event. §13.3.
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
| `new_hits == 0` **after at least one `research` pass** (`len(searches) > 1`) | The last rewrite surfaced no new source, so refining has stopped paying for itself. The guard matters: an empty *first* search is not a stop condition — it is precisely the case agent mode exists for, so it routes to `research`. |

Then the ordinary verdict applies: below `RETRIEVAL_MIN_SCORE`, `clarify`.

**Knobs** (`src/config.py`), listed because §13.3 and §13.4 add ceilings alongside them:

| Key | Default | Meaning |
|---|---|---|
| `RETRIEVAL_MIN_SCORE` | `0.05` | Below this, no answer ships. Identical in every mode. |
| `RETRIEVAL_TOP_K` | `4` | Chunks per search. |
| `AGENT_MAX_SEARCHES` | `3` | Searches per turn, counting the first. `1` makes agent mode behave like chat. |
| `AGENT_GOOD_SCORE` | `0.5` | Stop refining: retrieval already worked. |

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

### 13.3 Human-in-the-loop approval — *implemented (M6)*

> **Built as designed**, with the amendments recorded at the end of this section.
> M6 was scoped as *tools **and** approval*, not approval alone, because a gate
> with nothing behind it is theatre: before it, the graph could only read.
> `TOOLS_ENABLED` is off by default — a graph that can act is a different risk
> profile from one that can only search.

**Requirement.** Before executing any tool marked as having external effects, the
graph MUST suspend, surface what it intends to do, and wait for an explicit human
decision. It MUST NOT perform the effect and then ask.

#### The tool registry

```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str                       # shown to the model
    effect: Literal["read", "write"] = "write"
    editable_fields: frozenset[str] = frozenset()
    run: Callable[..., Awaitable[str]]
```

`effect` defaults to `"write"`, so a tool needs approval **unless it declares
itself safe**. This resolves the first open question in favour of failing closed:
the realistic mistake is adding a tool and forgetting to classify it, and the two
failures are not symmetric — a misfiled `read` costs one unnecessary prompt, a
misfiled `write` sends an email nobody approved.

#### Graph shape

```
generate ──▶ «route_after_generate» ──▶ write_memory   (answered, no tool wanted)
                        ├────────────▶ clarify        (uncited -- unchanged)
                        └────────────▶ plan           (model requested a tool)

plan ──▶ «route_after_plan» ──▶ act        (effect="read": just run it)
                   └─────────▶ approve     (effect="write": interrupt() here)

approve ──▶ «route_after_approval» ──▶ act       (approved, possibly edited)
                        └───────────▶ clarify   (rejected: say what was not done)

act ──▶ generate      (bounded by TOOL_MAX_CALLS, then straight to write_memory)
```

`plan` selects the tool and arguments and does nothing else — it is the only node
that may propose an effect. `approve` contains the `interrupt()` and nothing
else. `act` performs the effect and has no `interrupt()`, so it is never replayed
by a resume.

**Tool output is evidence, not an exemption.** `act` merges its result into
`retrieved` as an ordinary chunk with `source: "tool://<name>/<call_id>"` and
`score: 1.0`, so the existing citation gate applies to a tool-derived answer
byte for byte — a claim about what the tool returned must cite the call that
returned it. The alternative, a separate `tool_results` key with its own
grounding check, is rejected for exactly the reason §13.2 forbids a second
router: a duplicated grounding gate is a gate that will drift. Cost of the choice:
`score: 1.0` asserts "the system produced this on request", which is not a
similarity score and will look odd in retrieval telemetry; it MUST be excluded
from retrieval-quality metrics by its `tool://` scheme.

#### Interrupt mechanics

LangGraph's `interrupt()` (`langgraph.types`) suspends the run and persists it to
the checkpointer. The run resumes by invoking the graph on the **same
`thread_id`** with `Command(resume=<decision>)`; the resume value becomes the
return value of `interrupt()` inside the node.

```python
from langgraph.types import Command, interrupt

decision = interrupt({
    "call_id": call_id,
    "action": "send_email",
    "to": to, "subject": subject, "body": body,
})
if decision.get("approve") is not True:
    return {"pending_action": None, "answer": "cancelled by the user"}
# the effect happens only past this line -- in `act`, one edge later
```

**Four constraints that are easy to get wrong:**

1. **Resuming re-runs the node from its start**, not from the `interrupt()` call.
   Any work before `interrupt()` executes a second time. Therefore a node
   containing `interrupt()` MUST do nothing observable before it — no writes, no
   sends, no counters. Put the effect strictly *after* the call.
2. **An effect must be idempotent across resumes.** `plan` mints a `call_id`;
   `act` appends it to `tool_calls` before returning, and re-entering `act` with
   a `call_id` already recorded is a no-op returning the stored result. Without
   this, a client that posts the same approval twice — a double-click, a retry
   after a timeout — sends two emails, and the checkpoint cannot tell them apart.
3. **A durable checkpointer is mandatory.** Under `ENV=dev` the in-memory saver
   loses a pending approval on restart, silently abandoning the turn. §7.1's
   backend table therefore gains a rule: **approval flows require `ENV=local` or
   `ENV=prod`**, and the API MUST refuse to enable tools under `ENV=dev` rather
   than discover this at the first interrupt.
4. **An approval is scoped to the thread it was raised on.** Resuming MUST verify
   the `user_id` owns that thread, exactly as `DELETE /users/{id}` does —
   otherwise anyone who guesses a `thread_id` can approve someone else's pending
   action. §3's authentication boundary defines who the approver is.

#### API surface (added to §9 when built)

| Endpoint | Purpose |
|---|---|
| `GET /threads/{thread_id}/pending` | The interrupt payload awaiting a decision, or 404. |
| `POST /threads/{thread_id}/resume` | Body carries the decision; resumes the run. |

```json
// GET .../pending -> 200
{ "call_id": "c_7f", "tool": "send_email", "args": {"to": "...", "subject": "...", "body": "..."},
  "editable": ["subject", "body"], "requested_at": "2026-08-27T10:04:11Z" }

// POST .../resume
{ "user_id": "u_123", "call_id": "c_7f", "decision": "approve", "edits": {"subject": "..."} }
```

`resume` returns the same body as `POST /chat`. Status codes carry the failure
modes that matter: **403** when `user_id` does not own the thread, **404** when
nothing is pending, **409** when `call_id` is not the pending call (the approver
is acting on a screen that has since moved on), **410** when the approval has
expired.

`POST /chat` and `POST /chat/stream` gain a terminal state: a turn may end
`{"status": "awaiting_approval", "pending": {...}}` instead of returning an
answer. Clients MUST handle it; the stream gains an `interrupt` event carrying
the same payload, and — like the `restart` event (§13.2, streaming rule in
`CLAUDE.md`) — anything already streamed for that turn is not an answer.

#### Resolved open questions

- **What marks a tool as risky?** A registry field defaulting to `"write"`. Fails closed; see above.
- **Does an approval expire?** Yes — `APPROVAL_TTL_SECONDS`, default 86400. A resume past the TTL returns 410 and resumes the run with a rejection, so a week-old intention ends the turn cleanly instead of leaving a checkpoint pending forever.
- **Is an edited approval allowed?** Yes, but only for fields the tool lists in `editable_fields` — never the recipient of an effect, only its content. The resume record stores proposed *and* final arguments, so the audit trail shows what the human changed. Editing recipients is where approval stops being an approval.

#### Configuration

| Key | Default | Meaning |
|---|---|---|
| `TOOLS_ENABLED` | `false` | Master switch. `false` keeps `plan`/`approve`/`act` unreachable. |
| `TOOL_MAX_CALLS` | `2` | Tool calls per turn. Each one costs a `generate` pass. |
| `APPROVAL_TTL_SECONDS` | `86400` | Age past which a pending approval is refused. |

#### Definition of done (M6) — met

`tests/test_tools_approval.py` asserts each of: a `write` tool performs **zero**
effects before a resume; `decision: "reject"` performs none at all and the answer
says what was not done; two resumes of one `call_id` produce **one** effect (and
`act` alone refuses a `call_id` it already performed); a resume from a different
`user_id` returns 403 and performs none; an expired approval is declined rather
than performed; edits cannot reach a field the tool did not declare editable; and
the grounding parity test extended to tool turns — a tool-derived answer that
cites nothing still routes to `clarify`.

#### Amended at implementation time (M6)

- **`generate` signals, `plan` proposes.** The model's reply is a whole-reply
  `TOOL <name> {json}` directive, carried in a new state field `tool_request`
  (§6.1). Keeping the parse in `plan` costs nothing and keeps "only `plan` may
  propose an effect" true rather than nearly true. A directive is not an answer,
  so it never reaches `messages`.
- **`tool_calls` is per turn, not per thread.** It resets in `load_memory` with
  the rest of the per-turn state, which is what makes `TOOL_MAX_CALLS` a turn
  budget. Idempotency only ever needs to span one run.
- **Thread registration moved to `load_memory`.** Ownership is checked against
  the user's thread index, and a turn that parks on an approval has not reached
  `write_memory` yet — the owner could not resume their own thread. It also
  closes a latent §10 gap: a thread whose turn crashed, or one written under
  `MEMORY_EXTRACTION=off`, was invisible to deletion.
- **`GET /threads/{id}/pending` takes `?user_id=`.** The ownership rule needs an
  identity and a GET has no body.
- **The stream asks the checkpoint.** An interrupt does not surface as a part of
  `astream(stream_mode=["messages", "values"])`, so the handler reads the state
  after the stream ends. `restart` precedes `interrupt` only if tokens were
  actually drawn.
- **Known limitation: tools sit behind `generate`.** A turn whose retrieval was
  too weak routes to `clarify` and never reaches the node that could ask for a
  tool. Actions that follow from retrieved content work; a bare "send this email"
  on an off-corpus thread does not. Moving the tool path in front of retrieval
  would let an action run with no evidence gathered at all, which is the trade
  §13 exists to refuse.
- **No browser UI.** The approval prompt is an API-level feature in M6; the UI
  ships `chat` and `agent` only.

### 13.4 Researcher + writer — *designed, NOT built (M7)*

> **Nothing in the repository implements this.** Scheduled after M6 because a
> researcher that can act needs the approval gate to exist first.

**Shape.** Two subagents — compiled graphs used as nodes — plus a supervisor
that decides which runs next and when the work is done.

```
        ┌── supervisor ──┐
        │                │
        ▼                ▼
   researcher   ────▶  writer  ────▶ write_memory/END
   (search, gather,     (compose the answer from
    judge sufficiency)   what the researcher gathered)
```

**The supervisor is a router, not a model call.** It reads state — did the
researcher return evidence, did the writer hand work back, is the handoff budget
spent — and names the next subagent. This resolves the first open question in
favour of the cheap deterministic option: a model call to decide "the researcher
returned nothing, ask again" buys nothing a condition on state cannot decide, and
it costs a third call on every turn. A model-driven supervisor stays available
behind `SUPERVISOR_MODEL=1` for the case the evals actually show: a writer that
sends work back with a reason the router cannot interpret.

**Division of labour.** The researcher owns retrieval and decides when the
evidence is sufficient — essentially §13.2's loop, extracted. The writer owns
composition and citation, and **must not retrieve**: giving both agents the
ability to search is how a system starts citing sources that never reached the
answer.

**State boundary.** The two MUST NOT share one flat state. The researcher's
intermediate queries, rejected chunks and `new_hits` counter stay inside its own
subgraph state. Exactly one key crosses to the writer:

```python
class Evidence(TypedDict):
    id: str      # "S1", "S2", ... assigned by the researcher, stable for the turn
    text: str
    source: str
    score: float
```

Shared keys written by a subgraph require a reducer declared in the parent
(LangGraph rule), and `messages` in particular must not accumulate both agents'
internal chatter — that is checkpointed per thread and would be replayed into
every later turn. **Only the writer's final message appends to `messages`.**

**Citation provenance is the hard part.** With one agent, `[S1]` indexes the list
`generate` was handed. With two, the writer cites evidence the researcher chose,
and the mapping must survive the handoff intact — which is why the id lives on
`Evidence` rather than being re-derived from list position on the far side. The
existing citation eval (§11) MUST be extended to assert that every `[S<n>]` the
writer emits resolves to an `Evidence.id` the researcher produced **in that
turn**, not merely that a marker exists.

**Long-term facts split by namespace.** §7.2 already separates them, and the
split matches the division of labour: `(user_id, "preferences")` — tone,
language, format — goes to the writer; `(user_id, "facts")` goes to the
researcher as retrieval hints. Neither agent gets both, and the second open
question is resolved by that boundary rather than by duplicating the facts into
both prompts.

**A third mode, not a replacement.** `mode: "research"` joins `chat` and `agent`
(§9, §13.1). `agent` keeps its exact current behaviour, which keeps the migration
reversible and keeps the §13.2 tests meaningful; §9's rule that an unrecognised
mode is rejected already protects older clients.

**Cost.** A `research`-mode turn is at minimum two model calls (research, write)
against agent mode's two and chat's one — and up to
`AGENT_MAX_SEARCHES + SUPERVISOR_MAX_HANDOFFS + 1`. `AGENT_MAX_SEARCHES` bounds
the researcher; `SUPERVISOR_MAX_HANDOFFS` (default `2`) bounds the ping-pong,
because two agents with no ceiling can pass work back and forth indefinitely.

#### Definition of done (M7)

Tests asserting: every citation the writer emits maps to evidence the researcher
returned this turn; the writer has no path to the vector store (its subgraph is
compiled without one); `messages` after a `research` turn contains exactly one
new assistant message; `SUPERVISOR_MAX_HANDOFFS` terminates a researcher/writer
loop that refuses to converge; and the §13 grounding parity test extended to the
third mode.

### 13.5 What every phase must preserve

| Invariant | Enforced by |
|---|---|
| `RETRIEVAL_MIN_SCORE` and the citation requirement are identical in all modes | one grounding parity test, extended per phase — never exempted |
| One router owns the answer/refuse verdict | §13.2; a second router may not re-decide it |
| An uncited answer routes to `clarify` | `route_after_generate`, unchanged by M6/M7 |
| Durable facts never enter `messages`; conversation never enters the Store | §7, `USER_NAMESPACES` (`src/memory/threads.py`) |
| Effects happen after approval, exactly once | §13.3 constraints 1–2 |
