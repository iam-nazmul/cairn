# cairn

A chatbot that answers questions from your own documents, **cites where each
answer came from**, and remembers you between conversations.

- Ask a question, get an answer grounded in the knowledge base with sources
  attached. If nothing relevant is found, it says so instead of guessing.
- Pick up a conversation where you left off — days later, after a restart.
- Tell it something durable ("my preferred language is Bengali") and it still
  knows in a brand-new conversation.
- Ask it to forget you, and it deletes everything.

Runs entirely on your own machine against a local LLM, or against a hosted
provider if you prefer.

## Quick start

```bash
docker compose up --build
curl localhost:8000/health
```

That starts the API on port 8000 and its database. The language model comes from
[Ollama](https://ollama.com) running on your machine:

```bash
ollama pull llama3.1
```

Ollama listens only on `127.0.0.1` by default, which the container cannot reach.
Allow it once:

```bash
sudo systemctl edit ollama     # [Service]
                               # Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

Prefer not to change that? `docker compose --profile ollama up --build` runs the
model inside Docker instead (it re-downloads the models), or
`LLM_PROVIDER=fake docker compose up --build` starts everything with a canned
stand-in model, which is enough to try the endpoints.

Running without Docker:

```bash
uv sync --extra dev
cp .env.example .env
uv run uvicorn src.api.routes:app --reload
```

## Talking to it

Start a conversation, then send messages to it. You only ever send the newest
message — earlier turns are remembered for you.

```bash
THREAD=$(curl -s -X POST localhost:8000/threads | jq -r .thread_id)

curl -s localhost:8000/chat -H 'content-type: application/json' -d "{
  \"user_id\": \"u_123\",
  \"thread_id\": \"$THREAD\",
  \"message\": \"How long do I have to submit an expense report?\"
}" | jq
```

```json
{
  "answer": "According to [S2], you have up to 30 days from the purchase date...",
  "citations": [{ "source": "doc://kb/expenses-1", "score": 0.2 }],
  "thread_id": "t_c1c068f2c495"
}
```

`[S2]` in the answer refers to the second retrieved source. `citations` lists
only the sources the answer actually used, so an empty list means the reply was
not drawn from your documents.

| Endpoint | What it does |
|---|---|
| `POST /threads` | Start a conversation and get its id |
| `POST /chat` | Send one message and get an answer with citations |
| `GET /threads/{id}/history` | Everything said in a conversation |
| `DELETE /users/{user_id}` | Erase a user: every conversation and every remembered fact |
| `GET /health` | Check it is running |

## What it remembers

Two different things, on purpose:

- **A conversation** is remembered by its `thread_id`. Different conversation,
  clean slate.
- **Durable facts about you** are remembered by `user_id` and follow you into
  every new conversation. It picks these up when you say things like *"my
  preferred language is Bengali"* or *"remember that I work in accounts payable"*.

Deleting a user removes both:

```bash
curl -X DELETE localhost:8000/users/u_123
# {"user_id":"u_123","threads_deleted":2,"facts_deleted":1}
```

## Your documents

This build ships with a handful of sample documents so it works out of the box.
Connecting it to a real document collection is an integration step — see
[src/rag/README.md](src/rag/README.md).

## Configuring

Everything is set through environment variables, documented with defaults in
[`.env.example`](.env.example). The ones most worth knowing:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `llama3.1` | Any model you have pulled (`ollama ls`) |
| `ENV` | `dev` | `dev` forgets on restart; `local` and `prod` do not |
| `MEMORY_EXTRACTION` | `rules` | `off` to stop remembering durable facts |
| `LOG_LEVEL` | `INFO` | Per-request timings and retrieval scores |

## For developers

[SPEC.md](SPEC.md) is the design, [CLAUDE.md](CLAUDE.md) the working rules, and
each module has its own guide: [graph](src/graph/README.md),
[memory](src/memory/README.md), [rag](src/rag/README.md), [api](src/api/README.md).
