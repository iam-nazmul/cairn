# Where does this piece of data go?

A decision procedure for the two memory systems. `SPEC.md` §5 and §7 define them; this file is for the moment you are holding a value and don't know where to put it.

## The question to ask

> **Should this survive the conversation it was learned in?**

- **No** → checkpointer. It is conversation state, keyed by `thread_id`.
- **Yes** → Store. It is a durable user fact, keyed by `user_id`.
- **Neither** → it is per-turn scratch. Leave it in state and let it be overwritten next turn.

The trap is data that feels durable but isn't. "The user is asking about invoicing" is *this conversation's* topic — checkpointer. "The user works in accounts payable" is a standing fact — Store.

## Table

| Data | Home | Key | Why |
|---|---|---|---|
| Message history | Checkpointer | `thread_id` | Restored automatically; client sends only the new turn |
| Retrieved chunks this turn | State (per-turn) | — | Recomputed every turn; no reason to persist |
| Current question | State (per-turn) | — | Overwritten next turn |
| "Prefers answers in Bengali" | Store | `(user_id, "preferences")` | Must hold on a brand-new thread |
| "Works in accounts payable" | Store | `(user_id, "facts")` | Standing fact about the person |
| "We were discussing invoice #42" | Checkpointer | `thread_id` | Scoped to this conversation |
| Draft/partial state mid-graph | Checkpointer | `thread_id` | Enables resume after a crash |
| Corpus documents | Vector store | — | Neither memory system; owned elsewhere (SPEC §3) |

## Rules that fall out of this

**Never hand-roll history.** If you find yourself writing a messages table, or making the client resend prior turns, the checkpointer is not doing its job — fix that instead. This is the single most common way to end up with two sources of truth that drift.

**Never put durable facts in `messages`.** They get checkpointed per thread, so they vanish the moment the user starts a new conversation — the exact failure the Store exists to prevent.

**Never put history in the Store.** It is `user_id`-scoped, so thread A's history bleeds into thread B.

**Namespace from the authenticated `user_id` only.** Never from a request field or a thread id. Cross-user leakage is the SPEC §10 isolation failure.

## Deletion (SPEC §10)

Right-to-be-forgotten spans **both** systems. Deleting a user means:

1. every checkpoint for every `thread_id` that user owns, and
2. every Store namespace scoped to that `user_id` — `facts` and `preferences` both.

Adding a third place user data lands means extending deletion too. That is what `memory-boundary-auditor` checks for.

## Open questions (SPEC §11)

- **What extracts facts in `write_memory`** — heuristic, LLM extraction, or explicit user command? Undecided. It determines how noisy the Store gets and whether upserts collide.
- **Store backend** — reuse the checkpointer's Postgres with pgvector, or a separate vector DB? Determines whether there is one `DB_URI` or two.

Both need deciding before M3.
