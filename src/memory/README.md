# src/memory — checkpointer, store, facts, deletion

For developers and agents touching persistence. The memory boundaries are the
part that breaks quietly, so read [CLAUDE.md](../../CLAUDE.md) first.

| File | Contents |
|---|---|
| `checkpointer.py` | `checkpointer_scope` — ENV → backend |
| `store.py` | `store_scope` — ENV → backend |
| `facts.py` | Durable-fact extraction |
| `threads.py` | Thread index and `forget_user` |

## Which memory does this data belong to?

Ask: **should this survive the conversation it was learned in?**

| Data | Home | Crosses threads? |
|---|---|---|
| "We were discussing invoice 42" | Checkpointer | no |
| "My preferred language is Bengali" | Store | yes, same user |
| Retrieved chunks, current question | Per-turn state | no |

The trap is data that feels durable but is not: a conversation's *topic* is
thread-scoped, while a standing fact about the person is not. This matters for
tests too — probing thread isolation with "My name is Alice" tests the wrong
system, because a name is a durable fact and legitimately reaches other threads.

## Backends

Both factories are async context managers because the durable savers close their
connection on scope exit. Build **and** invoke the graph inside the scope; in
FastAPI that means the lifespan handler, never per request.

`setup()` is idempotent and must run before first use, or the first invoke fails
with a missing-relation error.

`ENV=prod` points the checkpointer and the Store at the **same** `DATABASE_URL`
(SPEC §11). One database means deleting a user is one path against one
credential, rather than two systems that can fall out of sync.

Nothing outside this package may branch on `ENV`. Graph code is identical across
backends; only the instance handed to `compile()` differs.

## Fact extraction

Deterministic rules by default (`MEMORY_EXTRACTION=rules`): `remember that ...`,
`my <attribute> is <value>`, `I prefer ...`.

The Store's failure mode is noise, and noise is expensive here — a bad fact is
not wrong once, it is injected into every future prompt on every future thread
for that user. So this errs towards precision. Keys are derived from the
normalized attribute, which makes writes idempotent upserts: restating a fact
updates the row instead of adding another. `MEMORY_EXTRACTION=llm` inverts that
trade — better recall, but two phrasings of one fact produce two rows, plus a
model call on every turn's write path.

Accepted cost: rules miss paraphrases nobody wrote a pattern for.

## Deletion

The checkpointer is keyed by `thread_id` alone and its metadata carries no
`user_id`, so nothing can answer "which threads belong to this user?". Hence the
thread index in `(user_id, "threads")` — thread ids only, no message content.

`forget_user` deletes every checkpoint for every thread the user owns, then every
Store namespace in `USER_NAMESPACES`. **Add a new namespace there whenever user
data lands somewhere new**, or deletion silently becomes partial — the failure
where you believe a user was erased and they were not. A test asserts that list
stays complete.

## Testing

`tests/test_memory.py` is the critical suite: continuity, isolation, durability
across a simulated restart. It is parametrized over in-memory, SQLite and
Postgres. In-memory cannot fail the durability test, so it never counts as
coverage on its own. Postgres thread ids are uuid-suffixed because the database
persists between runs and fixed ids would quietly invalidate the assertions.
