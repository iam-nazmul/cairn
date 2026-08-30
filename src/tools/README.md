# src/tools — what the graph may do, and what it must ask first

For developers and agents adding a tool. The graph shape and the approval flow
are SPEC §13.3; this file is the part you need to add one safely.

## The default is `write`

```python
Tool(name="poke", description="Poke something.", run=poke)  # effect="write"
```

A tool nobody classified needs approval. This is not caution for its own sake:
the realistic mistake is adding a tool and forgetting to think about its effect,
and the two failures are not symmetric. A misfiled `read` costs one unnecessary
prompt. A misfiled `write` sends an email nobody approved.

Declare `effect="read"` only when the call changes nothing outside this process.
`draft_email` composes and returns text — read. `send_email` hands it to a
transport — write.

## `editable_fields` is content, never target

A human approving a call may edit the fields listed there, and nothing else.
`send_email` allows `subject` and `body`; `to` is deliberately absent. Editing
the recipient is the point at which an approval stops being an approval — the
human sees one action and authorises a different one. A test pins it.

## Adding a tool

1. Write an `async def` returning a string. Its parameters are the arguments the
   model must supply; `Tool.validate` binds them against this signature, so a
   call with the wrong shape is refused before it is ever proposed.
2. Register it in `build_registry`.
3. Add a test that it does not run before a resume. `tests/test_tools_approval.py`
   has the pattern: a recording transport whose list must stay empty.

Nothing else changes. `plan` resolves any registered name, `route_after_plan`
reads `effect`, and `act` performs it — none of them know your tool exists.

## Transports

`Transport` is a seam like `VectorStore` and `ChatModel`. `LoggingTransport` is
the default and sends nothing: real egress belongs to a deployment, not to this
repository, and a default that reached the network would make every test run a
live send. Pass a real one to `build_registry` at wiring time.
