"""API surface (SPEC §9), driven against the real SQLite backend."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import routes
from src.config import Settings

USER = "u_api"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    test_settings = Settings(env="local", llm_provider="fake", sqlite_path=str(tmp_path / "api.db"))
    monkeypatch.setattr(routes, "get_settings", lambda: test_settings)
    # The context manager runs the lifespan handler, which builds the graph.
    with TestClient(routes.app) as test_client:
        yield test_client


def chat(client: TestClient, thread_id: str, message: str) -> dict[str, object]:
    response = client.post(
        "/chat", json={"user_id": USER, "thread_id": thread_id, "message": message}
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["env"] == "local"


def test_create_thread_returns_an_id(client: TestClient) -> None:
    response = client.post("/threads")
    assert response.status_code == 200
    assert response.json()["thread_id"].startswith("t_")


def test_two_threads_get_different_ids(client: TestClient) -> None:
    first = client.post("/threads").json()["thread_id"]
    second = client.post("/threads").json()["thread_id"]
    assert first != second


def test_chat_answers_with_citations(client: TestClient) -> None:
    body = chat(client, "t-api-cite", "How long do I have to submit an expense report?")

    assert body["answer"]
    assert body["thread_id"] == "t-api-cite"
    citations = body["citations"]
    assert citations, "a grounded answer must come back with citations"
    for citation in citations:
        assert citation["source"].startswith("doc://")
        assert 0.0 <= citation["score"] <= 1.0


def test_off_corpus_chat_returns_no_citations(client: TestClient) -> None:
    body = chat(client, "t-api-clarify", "Explain quantum chromodynamics.")
    assert body["citations"] == []


def test_chat_remembers_across_turns_on_one_thread(client: TestClient) -> None:
    """The client sends only the new message; history comes from the checkpoint."""
    chat(client, "t-api-memory", "My name is Alice.")
    body = chat(client, "t-api-memory", "What's my name?")

    assert "alice" in str(body["answer"]).lower()


def test_chat_does_not_leak_across_threads(client: TestClient) -> None:
    chat(client, "t-api-x", "My name is Alice.")
    body = chat(client, "t-api-y", "What's my name?")

    assert "alice" not in str(body["answer"]).lower()


def test_history_returns_the_checkpointed_turns(client: TestClient) -> None:
    chat(client, "t-api-history", "My name is Alice.")
    chat(client, "t-api-history", "What's my name?")

    response = client.get("/threads/t-api-history/history")
    assert response.status_code == 200
    messages = response.json()["messages"]

    assert len(messages) == 4  # 2 turns x (human + ai)
    assert [m["role"] for m in messages] == ["human", "ai", "human", "ai"]
    assert messages[0]["content"] == "My name is Alice."


def test_history_for_an_unknown_thread_is_404(client: TestClient) -> None:
    assert client.get("/threads/t-nope/history").status_code == 404


def test_chat_requires_a_thread_id(client: TestClient) -> None:
    response = client.post("/chat", json={"user_id": USER, "message": "hello"})
    assert response.status_code == 422


def test_chat_rejects_an_empty_thread_id(client: TestClient) -> None:
    response = client.post("/chat", json={"user_id": USER, "thread_id": "", "message": "hello"})
    assert response.status_code == 422
