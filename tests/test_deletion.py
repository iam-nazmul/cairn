"""Right to be forgotten (SPEC §10), across BOTH memory systems."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import routes
from src.config import Settings

ALICE = "u_alice"
BOB = "u_bob"
FACT = "My preferred language is Bengali."
ASK = "What language do I prefer?"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    test_settings = Settings(env="local", llm_provider="fake", sqlite_path=str(tmp_path / "del.db"))
    monkeypatch.setattr(routes, "get_settings", lambda: test_settings)
    with TestClient(routes.app) as test_client:
        yield test_client


def chat(client: TestClient, user: str, thread: str, message: str) -> dict[str, object]:
    response = client.post("/chat", json={"user_id": user, "thread_id": thread, "message": message})
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_deleting_a_user_removes_their_facts(client: TestClient) -> None:
    chat(client, ALICE, "t-del-1", FACT)
    assert "bengali" in str(chat(client, ALICE, "t-del-2", ASK)["answer"]).lower()

    response = client.delete(f"/users/{ALICE}")
    assert response.status_code == 200
    assert response.json()["facts_deleted"] >= 1

    # A brand-new thread must no longer know the fact.
    assert "bengali" not in str(chat(client, ALICE, "t-del-3", ASK)["answer"]).lower()


def test_deleting_a_user_removes_their_threads(client: TestClient) -> None:
    chat(client, ALICE, "t-del-hist", "We were discussing invoice 42.")
    assert client.get("/threads/t-del-hist/history").status_code == 200

    report = client.delete(f"/users/{ALICE}").json()
    assert report["threads_deleted"] >= 1

    # The checkpoints are gone, so there is no history left to read.
    assert client.get("/threads/t-del-hist/history").status_code == 404


def test_deletion_does_not_touch_other_users(client: TestClient) -> None:
    """The isolation guarantee that makes deletion safe to expose."""
    chat(client, ALICE, "t-del-a", FACT)
    chat(client, BOB, "t-del-b", FACT)

    client.delete(f"/users/{ALICE}")

    assert "bengali" in str(chat(client, BOB, "t-del-b2", ASK)["answer"]).lower()
    assert client.get("/threads/t-del-b/history").status_code == 200


def test_deletion_is_idempotent(client: TestClient) -> None:
    chat(client, ALICE, "t-del-idem", FACT)

    first = client.delete(f"/users/{ALICE}").json()
    second = client.delete(f"/users/{ALICE}").json()

    assert first["threads_deleted"] >= 1
    assert second == {"user_id": ALICE, "threads_deleted": 0, "facts_deleted": 0}


def test_deleting_an_unknown_user_reports_zeroes(client: TestClient) -> None:
    response = client.delete("/users/u_nobody")
    assert response.status_code == 200
    assert response.json()["threads_deleted"] == 0


def test_every_user_namespace_is_covered_by_deletion() -> None:
    """A third place for user data means extending USER_NAMESPACES, or deletion
    silently becomes partial (memory-placement.md)."""
    from src.memory.facts import FACTS_NS, PREFERENCES_NS
    from src.memory.threads import THREADS_NS, USER_NAMESPACES

    assert set(USER_NAMESPACES) == {FACTS_NS, PREFERENCES_NS, THREADS_NS}
