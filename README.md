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
```

Then open **<http://localhost:8000>** and ask it something.

That starts the API on port 8000 and its database. The language model comes from
[Ollama](https://ollama.com) running on your machine:

```bash
ollama pull llama3.1
```

Ollama listens only on `127.0.0.1` by default, which the container cannot reach —
if you skip this, every answer fails and the status dot in the sidebar turns
amber. Allow it once:

```bash
sudo systemctl edit ollama     # [Service]
                               # Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

Prefer not to touch your Ollama setup? On **Linux**, run the API in the host's
network namespace instead — then its `localhost` is your `localhost`, and the
default `127.0.0.1` bind is reached with no root and no edit:

```bash
docker compose -f docker-compose.yml -f docker-compose.host-net.yml up --build
```

(Linux only: on Docker Desktop the "host" is a VM, so it buys nothing. The
container also stops being isolated from your network.)

Failing that, `docker compose --profile ollama up --build` runs the model inside
Docker instead (it re-downloads the models), or `LLM_PROVIDER=fake docker compose
up --build` starts everything with a canned stand-in model, which is enough to
try the endpoints.

`OLLAMA_BASE_URL` in `.env` applies only when you run the app directly on your
machine. Docker needs a different address — `localhost` inside a container means
the container — so Compose passes its own and takes overrides from
`DOCKER_OLLAMA_BASE_URL`. `docker compose config` shows what the container gets.

Running without Docker:

```bash
uv sync --extra dev
cp .env.example .env
uv run uvicorn src.api.routes:app --reload
```

## In the browser

<http://localhost:8000> is a chat window over the same API. Answers stream in as
the model writes them — formatted, so lists, bold and code blocks read as such
rather than as raw asterisks — with the sources each one used listed underneath.

- **Conversations** in the sidebar. Pick one up where you left it — the history
  is restored from the checkpoint, not from anything the browser kept. The bin
  icon deletes one: its messages go, what cairn remembers about you stays.
- **What cairn remembers** lists the durable facts on file for you. Say *"my
  preferred language is Bengali"* and watch it appear, then start a brand-new
  conversation and see it still there.
- **Signed in as** is the `user_id`. Change it and you get a different person's
  conversations and a different set of remembered facts.
- **Forget me** erases both, and tells you how much it deleted.

An answer with no sources under it was not drawn from your documents — the UI
says so rather than letting it pass as grounded.

Every answer has a **Copy** button, and every code block a **Copy** and a
**Download**. Copying gives you the Markdown behind the answer rather than the
formatted text, so it pastes into a document or an editor unchanged.

The stylesheet is compiled in the browser from a CDN, so the page needs internet
on first load even though everything else runs locally. Swapping that for a
prebuilt file is two commands: see [src/api/README.md](src/api/README.md).

## Talking to it directly

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
| `POST /chat/stream` | The same, streamed back as the answer is written |
| `GET /threads/{id}/history` | Everything said in a conversation |
| `GET /users/{user_id}/threads` | A user's conversations |
| `GET /users/{user_id}/facts` | What is remembered about a user |
| `DELETE /users/{user_id}/threads/{id}` | Erase one conversation, keeping remembered facts |
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
