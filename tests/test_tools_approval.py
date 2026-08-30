"""Tools and human-in-the-loop approval (SPEC §13.3).

The claim under test is narrow and absolute: a tool with an external effect does
not run until a human says so, and it runs at most once when they do.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from src.api import routes
from src.config import Context, Settings
from src.graph.build import build_graph
from src.graph.nodes import make_act
from src.rag.prompts import cited_chunks, parse_tool_request
from src.rag.retrieve import InMemoryVectorStore
from src.tools.registry import Tool, ToolRegistry, build_registry
from tests.conftest import make_runtime, make_state

USER = "u_tools"
# Retrieval has to succeed for `generate` to be reached at all, so the request
# rides on a question the corpus answers.
ASK = "Email the expense report deadline to alice@example.com"
SEND = 'TOOL send_email {"to": "alice@example.com", "subject": "Deadline", "body": "See policy."}'
DRAFT = 'TOOL draft_email {"to": "alice@example.com", "subject": "Deadline", "body": "See policy."}'

UNCITED = "The action completed."


class RecordingTransport:
    """Counts sends. The whole suite turns on this list staying empty."""

    def __init__(self) -> None:
        self.sent: list[dict[str, str]] = []

    async def send(self, *, to: str, subject: str, body: str) -> str:
        """See `Transport.send`."""
        self.sent.append({"to": to, "subject": subject, "body": body})
        return f"Email sent to {to}."


class ToolRequestingModel:
    """Asks for one tool call, then answers from what the call returned."""

    def __init__(self, directive: str = SEND, cite: bool = True) -> None:
        self.directive = directive
        self.cite = cite
        self.prompts: list[str] = []

    async def ainvoke(self, input: Any, /) -> AIMessage:
        """See `ChatModel.ainvoke`."""
        messages: list[AnyMessage] = list(input)
        system = str(messages[0].content)
        self.prompts.append(system)

        if "declined it" in system:
            return AIMessage(content="I did not send the email -- you declined it.")
        if "source=tool://" in system:
            return AIMessage(content="Sent, as you approved [S1]." if self.cite else UNCITED)
        if "Available tools:" in system:
            return AIMessage(content=self.directive)
        return AIMessage(content="Nothing to do here.")


def tool_settings(**overrides: Any) -> Settings:
    return Settings(env="local", llm_provider="fake", tools_enabled=True, **overrides)


def build(
    transport: RecordingTransport,
    settings: Settings | None = None,
    model: ToolRequestingModel | None = None,
    tools: ToolRegistry | None = None,
) -> Any:
    return build_graph(
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        settings=settings or tool_settings(),
        vector_store=InMemoryVectorStore(),
        chat_model=model or ToolRequestingModel(),
        tools=tools if tools is not None else build_registry(transport),
    )


async def turn(graph: Any, thread_id: str, message: str = ASK) -> dict[str, Any]:
    return dict(
        await graph.ainvoke(
            {"messages": [{"role": "user", "content": message}], "question": message},
            {"configurable": {"thread_id": thread_id}},
            context=Context(user_id=USER),
        )
    )


async def resume(graph: Any, thread_id: str, decision: str, **extra: Any) -> dict[str, Any]:
    return dict(
        await graph.ainvoke(
            Command(resume={"decision": decision, **extra}),
            {"configurable": {"thread_id": thread_id}},
            context=Context(user_id=USER),
        )
    )


# --- the gate itself ---------------------------------------------------------


async def test_a_write_tool_sends_nothing_before_a_decision() -> None:
    transport = RecordingTransport()

    result = await turn(build(transport), "t-hold")

    assert result["__interrupt__"], "the run must park on the approval"
    assert transport.sent == [], "the effect happened before anyone approved it"
    assert result["__interrupt__"][0].value["tool"] == "send_email"


async def test_approval_lets_it_through_exactly_once() -> None:
    transport = RecordingTransport()
    graph = build(transport)

    await turn(graph, "t-yes")
    result = await resume(graph, "t-yes", "approve")

    assert len(transport.sent) == 1
    assert transport.sent[0]["to"] == "alice@example.com"
    assert result["answer"]
    assert not result.get("__interrupt__")


async def test_a_second_resume_does_not_send_again() -> None:
    transport = RecordingTransport()
    graph = build(transport)

    await turn(graph, "t-twice")
    await resume(graph, "t-twice", "approve")
    await resume(graph, "t-twice", "approve")

    assert len(transport.sent) == 1


async def test_act_refuses_a_call_id_it_already_performed() -> None:
    """The guard behind the double-resume case: a replay of `act` sends nothing."""
    transport = RecordingTransport()
    act = make_act(build_registry(transport))
    pending = {
        "call_id": "c_1",
        "tool": "send_email",
        "args": {"to": "a@b.c", "subject": "s", "body": "b"},
    }
    state = make_state(
        pending_action=pending,
        tool_calls=[{**pending, "status": "done"}],
    )

    result = await act(state, runtime=make_runtime(user_id=USER))

    assert transport.sent == []
    assert result == {"pending_action": None}


async def test_rejection_performs_nothing_and_says_so() -> None:
    transport = RecordingTransport()
    graph = build(transport)

    await turn(graph, "t-no")
    result = await resume(graph, "t-no", "reject")

    assert transport.sent == []
    assert "did not send" in result["answer"]
    assert [c["status"] for c in result["tool_calls"]] == ["rejected"]


async def test_a_malformed_decision_is_not_an_approval() -> None:
    transport = RecordingTransport()
    graph = build(transport)

    await turn(graph, "t-junk")
    await resume(graph, "t-junk", "yes please")

    assert transport.sent == []


# --- effect classes and budgets ----------------------------------------------


async def test_a_read_tool_needs_no_approval() -> None:
    transport = RecordingTransport()
    graph = build(transport, model=ToolRequestingModel(directive=DRAFT))

    result = await turn(graph, "t-read")

    assert not result.get("__interrupt__"), "a read-effect tool must not stop the run"
    assert result["answer"]
    assert result["tool_calls"][0]["tool"] == "draft_email"


def test_a_directive_may_be_followed_by_commentary() -> None:
    """Models add a line explaining themselves; the call still stands."""
    parsed = parse_tool_request(SEND + "\nI will send this once you approve.")

    assert parsed is not None
    assert parsed[0] == "send_email"


def test_a_reply_that_merely_mentions_a_tool_is_not_a_request() -> None:
    """Anchored at the start for exactly this: talking about an action is not
    asking to take one."""
    assert parse_tool_request(f"I could run this for you:\n{SEND}") is None
    assert parse_tool_request("You can use TOOL send_email to do that.") is None


def test_a_directive_with_unparseable_arguments_is_refused() -> None:
    assert parse_tool_request('TOOL send_email {"to": "a@b.c", oops}') is None


async def test_an_unclassified_tool_needs_approval() -> None:
    """`effect` defaults to write, so forgetting to classify fails closed."""

    async def run(to: str) -> str:
        return f"poked {to}"

    tool = Tool(name="poke", description="Poke something.", run=run)

    assert tool.effect == "write"
    assert tool.needs_approval


async def test_an_unknown_tool_is_refused_rather_than_invented() -> None:
    transport = RecordingTransport()
    graph = build(transport, model=ToolRequestingModel(directive='TOOL wire_money {"amount": 1}'))

    result = await turn(graph, "t-unknown")

    assert not result.get("__interrupt__")
    assert transport.sent == []
    assert result["tool_calls"] == []


async def test_the_turn_budget_stops_a_tool_loop() -> None:
    transport = RecordingTransport()
    # The model asks for the same read tool forever; TOOL_MAX_CALLS is what ends it.
    graph = build(
        transport,
        settings=tool_settings(tool_max_calls=1),
        model=ToolRequestingModel(directive=DRAFT),
    )

    result = await turn(graph, "t-budget")

    assert len(result["tool_calls"]) == 1


async def test_edits_are_confined_to_the_fields_the_tool_declared() -> None:
    transport = RecordingTransport()
    graph = build(transport)

    await turn(graph, "t-edit")
    await resume(
        graph, "t-edit", "approve", edits={"subject": "Edited", "to": "mallory@example.com"}
    )

    sent = transport.sent[0]
    assert sent["subject"] == "Edited"
    # The recipient is the effect's target, not its content: never editable.
    assert sent["to"] == "alice@example.com"


# --- grounding is identical on a tool turn (SPEC §13, §13.5) -----------------


async def test_a_tool_answer_that_cites_nothing_does_not_ship() -> None:
    transport = RecordingTransport()
    graph = build(transport, model=ToolRequestingModel(cite=False))

    await turn(graph, "t-uncited")
    result = await resume(graph, "t-uncited", "approve")

    assert len(transport.sent) == 1, "the approved effect still happens"
    # The draft is dropped and the turn ends on the no-answer path instead.
    assert result["answer"] != UNCITED, "an uncited answer must not ship as grounded"
    assert cited_chunks(result["answer"], result["retrieved"]) == []


async def test_tools_are_unreachable_when_the_switch_is_off() -> None:
    transport = RecordingTransport()
    graph = build(transport, settings=Settings(env="local", llm_provider="fake"))

    result = await turn(graph, "t-off")

    assert not result.get("__interrupt__")
    assert transport.sent == []
    assert result["tool_calls"] == []


def test_tools_require_a_durable_checkpointer() -> None:
    """ENV=dev loses a pending approval on restart, so it is refused up front."""
    with pytest.raises(ValueError, match="ENV=local or ENV=prod"):
        Settings(env="dev", tools_enabled=True)


# --- the HTTP surface --------------------------------------------------------


@pytest.fixture
def tool_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[TestClient, RecordingTransport]]:
    transport = RecordingTransport()
    settings = tool_settings(sqlite_path=str(tmp_path / "tools.db"))
    monkeypatch.setattr(routes, "get_settings", lambda: settings)

    real_build = routes.build_graph

    def build_with_tools(**kwargs: Any) -> Any:
        kwargs["chat_model"] = ToolRequestingModel()
        kwargs["tools"] = build_registry(transport)
        return real_build(**kwargs)

    monkeypatch.setattr(routes, "build_graph", build_with_tools)
    with TestClient(routes.app) as client:
        yield client, transport


def ask(client: TestClient, thread_id: str, user_id: str = USER) -> dict[str, Any]:
    response = client.post(
        "/chat", json={"user_id": user_id, "thread_id": thread_id, "message": ASK}
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_chat_reports_awaiting_approval_instead_of_an_answer(
    tool_client: tuple[TestClient, RecordingTransport],
) -> None:
    client, transport = tool_client

    body = ask(client, "t-api-hold")

    assert body["status"] == "awaiting_approval"
    assert body["answer"] == ""
    assert body["pending"]["tool"] == "send_email"
    assert transport.sent == []


def test_pending_then_resume_completes_the_turn(
    tool_client: tuple[TestClient, RecordingTransport],
) -> None:
    client, transport = tool_client
    ask(client, "t-api-yes")

    pending = client.get("/threads/t-api-yes/pending", params={"user_id": USER})
    assert pending.status_code == 200, pending.text
    call_id = pending.json()["call_id"]
    assert pending.json()["editable"] == ["body", "subject"]

    done = client.post(
        "/threads/t-api-yes/resume",
        json={"user_id": USER, "call_id": call_id, "decision": "approve"},
    )
    assert done.status_code == 200, done.text
    assert done.json()["status"] == "answered"
    assert done.json()["answer"]
    assert len(transport.sent) == 1


def test_another_user_cannot_approve_your_action(
    tool_client: tuple[TestClient, RecordingTransport],
) -> None:
    client, transport = tool_client
    ask(client, "t-api-mine")
    call_id = client.get("/threads/t-api-mine/pending", params={"user_id": USER}).json()["call_id"]

    stolen = client.post(
        "/threads/t-api-mine/resume",
        json={"user_id": "u_someone_else", "call_id": call_id, "decision": "approve"},
    )

    assert stolen.status_code == 403
    assert transport.sent == []
    assert (
        client.get("/threads/t-api-mine/pending", params={"user_id": "u_someone_else"}).status_code
        == 403
    )


def test_resuming_a_call_id_that_is_not_the_pending_one_is_a_conflict(
    tool_client: tuple[TestClient, RecordingTransport],
) -> None:
    client, transport = tool_client
    ask(client, "t-api-stale")

    stale = client.post(
        "/threads/t-api-stale/resume",
        json={"user_id": USER, "call_id": "c_gone", "decision": "approve"},
    )

    assert stale.status_code == 409
    assert transport.sent == []


def test_resuming_twice_over_http_sends_once(
    tool_client: tuple[TestClient, RecordingTransport],
) -> None:
    client, transport = tool_client
    ask(client, "t-api-twice")
    call_id = client.get("/threads/t-api-twice/pending", params={"user_id": USER}).json()["call_id"]
    body = {"user_id": USER, "call_id": call_id, "decision": "approve"}

    first = client.post("/threads/t-api-twice/resume", json=body)
    second = client.post("/threads/t-api-twice/resume", json=body)

    assert first.status_code == 200
    assert second.status_code == 404, "nothing is pending any more"
    assert len(transport.sent) == 1


def test_an_expired_approval_is_declined_not_performed(
    tool_client: tuple[TestClient, RecordingTransport], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, transport = tool_client
    ask(client, "t-api-stale-ttl")
    call_id = client.get("/threads/t-api-stale-ttl/pending", params={"user_id": USER}).json()[
        "call_id"
    ]
    monkeypatch.setattr(routes, "_expired", lambda pending, settings: True)

    late = client.post(
        "/threads/t-api-stale-ttl/resume",
        json={"user_id": USER, "call_id": call_id, "decision": "approve"},
    )

    assert late.status_code == 410
    assert transport.sent == [], "an expired approval must not send"
    # Resumed as a rejection, so the checkpoint is not left parked forever.
    assert (
        client.get("/threads/t-api-stale-ttl/pending", params={"user_id": USER}).status_code == 404
    )


def test_the_stream_reports_the_interrupt_and_discards_the_draft(
    tool_client: tuple[TestClient, RecordingTransport],
) -> None:
    client, transport = tool_client

    with client.stream(
        "POST", "/chat/stream", json={"user_id": USER, "thread_id": "t-api-stream", "message": ASK}
    ) as response:
        events = [line for line in response.iter_lines() if line.startswith("data: ")]

    types = [json.loads(line.removeprefix("data: "))["type"] for line in events]
    assert "interrupt" in types
    assert "final" not in types, "a parked turn has no answer to be final about"
    if "token" in types:
        # Whatever was drawn was the directive, not an answer.
        assert types.index("restart") < types.index("interrupt")
    assert transport.sent == []
