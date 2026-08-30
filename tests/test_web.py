"""The browser UI: the page itself, its endpoints, and streamed turns."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from src.api import routes, web
from src.config import Settings
from src.rag.llm import explain, unreachable_hint

USER = "u_web"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    test_settings = Settings(env="local", llm_provider="fake", sqlite_path=str(tmp_path / "web.db"))
    monkeypatch.setattr(routes, "get_settings", lambda: test_settings)
    with TestClient(routes.app) as test_client:
        yield test_client


@pytest.fixture
def streaming_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """The scripted stand-in is not a LangChain model, so it emits no token events.

    A real provider does. This swaps in one that streams, which is the only way to
    cover the token path without a live Ollama.
    """
    test_settings = Settings(env="dev", llm_provider="fake")
    monkeypatch.setattr(routes, "get_settings", lambda: test_settings)
    monkeypatch.setattr(
        "src.graph.build.get_chat_model",
        lambda settings: GenericFakeChatModel(
            messages=iter(["Reports are due within 30 days. [S1]"] * 50)
        ),
    )
    with TestClient(routes.app) as test_client:
        yield test_client


def chat(client: TestClient, thread_id: str, message: str) -> dict[str, object]:
    response = client.post(
        "/chat", json={"user_id": USER, "thread_id": thread_id, "message": message}
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def read_events(client: TestClient, message: str, thread_id: str) -> list[dict[str, object]]:
    """Drain POST /chat/stream into decoded SSE events."""
    body = {"user_id": USER, "thread_id": thread_id, "message": message}
    with client.stream("POST", "/chat/stream", json=body) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())

    return [
        json.loads(block.removeprefix("data:").strip())
        for block in raw.split("\n\n")
        if block.strip()
    ]


# --- the page ----------------------------------------------------------------


def test_index_serves_the_chat_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="composer"' in response.text


def test_index_references_its_static_assets(client: TestClient) -> None:
    """A template typo here is a 404 the test suite should catch, not the browser."""
    response = client.get("/")
    assert "/static/app.js" in response.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/favicon.svg").status_code == 200
    # app.js imports this by relative path; a 404 breaks the whole module.
    assert client.get("/static/markdown.js").status_code == 200


def test_markdown_renderer_never_builds_html_from_strings() -> None:
    """Answers are model output. The renderer is safe because it creates nodes and
    sets textContent -- innerHTML anywhere would make it an injection point, and
    there is no sanitizer here to catch it."""
    code = "\n".join(
        line
        for line in (web.STATIC_DIR / "markdown.js").read_text().splitlines()
        if not line.strip().startswith(("//", "*", "/*"))  # the comments say "innerHTML"
    )

    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in code, f"{sink} in the markdown renderer"


def test_static_directory_is_the_one_next_to_the_module() -> None:
    """Resolved from __file__, so uvicorn and the container agree on it."""
    assert (web.STATIC_DIR / "app.js").is_file()
    assert (web.TEMPLATES_DIR / "index.html").is_file()


def test_every_template_id_app_js_binds_to_exists() -> None:
    """app.js clones by id; a renamed template is otherwise a runtime null."""
    markup = (web.TEMPLATES_DIR / "index.html").read_text()
    script = (web.STATIC_DIR / "app.js").read_text()

    for template_id in (
        "tpl-user",
        "tpl-assistant",
        "tpl-source",
        "tpl-ungrounded",
        "tpl-marker",
        "tpl-typing",
        "tpl-thread",
        "tpl-fact",
        "tpl-error",
        "tpl-code",
        "tpl-search",
        "tpl-toast",
    ):
        assert f'id="{template_id}"' in markup
        assert f'"{template_id}"' in script


def test_every_data_hook_app_js_binds_to_exists() -> None:
    """The toolbar buttons are wired by data attribute through one delegated
    listener, so a rename in either file is a button that silently does nothing."""
    markup = (web.TEMPLATES_DIR / "index.html").read_text()
    script = (web.STATIC_DIR / "app.js").read_text()

    for hook in (
        "data-answer",
        "data-sources",
        "data-detail",
        "data-code",
        "data-lang",
        "data-label",
        "data-copy-code",
        "data-download-code",
        "data-copy-answer",
        "data-open-thread",
        "data-delete-thread",
        "data-mode",
        "data-query",
        "data-approve",
        "data-decline",
        "data-edits",
    ):
        assert hook in markup, f"{hook} missing from index.html"
        assert hook in script, f"{hook} missing from app.js"


# --- endpoints behind the sidebar --------------------------------------------


def test_threads_endpoint_lists_conversations_after_a_turn(client: TestClient) -> None:
    assert client.get(f"/users/{USER}/threads").json()["threads"] == []

    client.post("/chat", json={"user_id": USER, "thread_id": "t-web-1", "message": "hello"})

    assert client.get(f"/users/{USER}/threads").json()["threads"] == ["t-web-1"]


def test_threads_endpoint_does_not_leak_across_users(client: TestClient) -> None:
    client.post("/chat", json={"user_id": USER, "thread_id": "t-web-mine", "message": "hello"})

    assert client.get("/users/u_someone_else/threads").json()["threads"] == []


def test_facts_endpoint_returns_what_the_store_holds(client: TestClient) -> None:
    client.post(
        "/chat",
        json={"user_id": USER, "thread_id": "t-web-2", "message": "My name is Alice."},
    )

    facts = client.get(f"/users/{USER}/facts").json()["facts"]
    assert any("alice" in fact.lower() for fact in facts)


def test_facts_endpoint_does_not_leak_across_users(client: TestClient) -> None:
    client.post(
        "/chat",
        json={"user_id": USER, "thread_id": "t-web-3", "message": "My name is Alice."},
    )

    assert client.get("/users/u_nobody/facts").json()["facts"] == []


# --- deleting one conversation -----------------------------------------------


def test_deleting_a_thread_erases_only_that_conversation(client: TestClient) -> None:
    chat(client, "t-del-keep", "We were discussing invoice 42.")
    chat(client, "t-del-drop", "We were discussing invoice 99.")

    response = client.delete(f"/users/{USER}/threads/t-del-drop")
    assert response.status_code == 200
    assert response.json() == {"user_id": USER, "thread_id": "t-del-drop"}

    assert client.get("/threads/t-del-drop/history").status_code == 404
    assert client.get("/threads/t-del-keep/history").status_code == 200
    assert client.get(f"/users/{USER}/threads").json()["threads"] == ["t-del-keep"]


def test_deleting_a_thread_keeps_durable_facts(client: TestClient) -> None:
    """Facts belong to the user, not to the conversation they were learned in."""
    chat(client, "t-del-facts", "My name is Alice.")

    client.delete(f"/users/{USER}/threads/t-del-facts")

    facts = client.get(f"/users/{USER}/facts").json()["facts"]
    assert any("alice" in fact.lower() for fact in facts)


def test_cannot_delete_another_users_thread(client: TestClient) -> None:
    """Checkpoints carry no owner, so without the index check any caller could
    delete any conversation by guessing its id."""
    chat(client, "t-del-mine", "We were discussing invoice 42.")

    response = client.delete("/users/u_intruder/threads/t-del-mine")

    assert response.status_code == 404
    assert client.get("/threads/t-del-mine/history").status_code == 200
    assert client.get(f"/users/{USER}/threads").json()["threads"] == ["t-del-mine"]


def test_deleting_an_unknown_thread_is_404(client: TestClient) -> None:
    assert client.delete(f"/users/{USER}/threads/t-never-existed").status_code == 404


def test_deleting_a_thread_leaves_the_index_consistent(client: TestClient) -> None:
    """A stale index entry would make forget_user report a deletion it never made."""
    chat(client, "t-del-a", "one")
    chat(client, "t-del-b", "two")
    client.delete(f"/users/{USER}/threads/t-del-a")

    report = client.delete(f"/users/{USER}").json()

    assert report["threads_deleted"] == 1


def test_forget_clears_what_the_sidebar_shows(client: TestClient) -> None:
    client.post(
        "/chat",
        json={"user_id": USER, "thread_id": "t-web-4", "message": "My name is Alice."},
    )
    client.delete(f"/users/{USER}")

    assert client.get(f"/users/{USER}/threads").json()["threads"] == []
    assert client.get(f"/users/{USER}/facts").json()["facts"] == []


# --- streaming ---------------------------------------------------------------


def test_stream_ends_with_a_final_event_carrying_citations(client: TestClient) -> None:
    events = read_events(client, "How long do I have to submit an expense report?", "t-stream-1")

    final = events[-1]
    assert final["type"] == "final"
    assert final["thread_id"] == "t-stream-1"
    assert final["answer"]
    assert final["citations"], "a grounded answer must stream back with citations"


def test_stream_matches_what_the_blocking_endpoint_returns(client: TestClient) -> None:
    """Two transports, one turn -- they must not drift apart."""
    question = "How do I connect to the corporate VPN?"

    blocking = client.post(
        "/chat", json={"user_id": USER, "thread_id": "t-stream-a", "message": question}
    ).json()
    streamed = read_events(client, question, "t-stream-b")[-1]

    assert streamed["answer"] == blocking["answer"]
    assert streamed["citations"] == blocking["citations"]


def test_stream_persists_the_turn_like_any_other(client: TestClient) -> None:
    read_events(client, "We were discussing invoice 42.", "t-stream-2")
    events = read_events(client, "What were we discussing?", "t-stream-2")

    assert "invoice 42" in str(events[-1]["answer"]).lower()
    history = client.get("/threads/t-stream-2/history").json()["messages"]
    assert len(history) == 4


def test_off_corpus_stream_finals_with_no_citations(client: TestClient) -> None:
    events = read_events(client, "Explain quantum chromodynamics.", "t-stream-3")

    assert events[-1]["citations"] == []


def test_stream_emits_tokens_from_a_streaming_provider(streaming_client: TestClient) -> None:
    events = read_events(streaming_client, "When are expense reports due?", "t-stream-4")

    tokens = [e for e in events if e["type"] == "token"]
    assert len(tokens) > 1, "a streaming provider should arrive in pieces, not one blob"
    assert "".join(str(e["text"]) for e in tokens) == events[-1]["answer"]


def test_an_uncited_draft_is_retracted_mid_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`generate` streams before anything checks it can cite a source. When it
    cannot, the turn reroutes to `clarify` and the draft on screen must be
    withdrawn -- otherwise an uncited answer stands as though it were grounded."""
    monkeypatch.setattr(routes, "get_settings", lambda: Settings(env="dev", llm_provider="fake"))
    monkeypatch.setattr(
        "src.graph.build.get_chat_model",
        # No [S] marker anywhere, so `generate` cannot cite what it retrieved.
        lambda settings: GenericFakeChatModel(messages=iter(["Reports are due soon."] * 50)),
    )

    with TestClient(routes.app) as client:
        events = read_events(client, "When are expense reports due?", "t-stream-retract")

    kinds = [e["type"] for e in events]
    assert "restart" in kinds, "an uncited draft must be withdrawn, not left on screen"
    assert kinds.index("restart") > kinds.index("token"), "withdrawn only after it was drawn"
    assert kinds[-1] == "final"
    assert events[-1]["citations"] == []


# --- an unreachable provider -------------------------------------------------


class _RefusingChatModel:
    """A provider that is configured but not answering."""

    async def ainvoke(self, input: object, /) -> object:
        raise httpx.ConnectError("All connection attempts failed")


@pytest.fixture
def offline_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    settings = Settings(env="dev", llm_provider="ollama", llm_model="llama3.1")
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr("src.graph.build.get_chat_model", lambda settings: _RefusingChatModel())
    monkeypatch.setattr(routes, "probe", _never_reachable)
    with TestClient(routes.app) as test_client:
        yield test_client


async def _never_reachable(settings: Settings) -> bool:
    return False


def test_health_reports_an_unreachable_provider_without_failing(
    offline_client: TestClient,
) -> None:
    """Docker health-checks this endpoint. A down model must not restart the API."""
    response = offline_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["llm_reachable"] is False


def test_health_reports_nothing_to_probe_as_none(client: TestClient) -> None:
    assert client.get("/health").json()["llm_reachable"] is None


def test_stream_explains_an_unreachable_provider(offline_client: TestClient) -> None:
    """'All connection attempts failed' is not something a reader can act on."""
    events = read_events(offline_client, "When are expense reports due?", "t-offline-1")

    error = events[-1]
    assert error["type"] == "error"
    assert "ollama ls" in str(error["detail"])


def test_chat_returns_503_for_an_unreachable_provider(offline_client: TestClient) -> None:
    """503, not 500: a dependency is down, which is not a bug in this service."""
    response = offline_client.post(
        "/chat", json={"user_id": USER, "thread_id": "t-offline-2", "message": "hello"}
    )

    assert response.status_code == 503
    assert "ollama ls" in response.json()["detail"]


# --- the advice has to match where the process is running --------------------


def hint_for(url: str, *, container: bool, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr("src.rag.llm.in_container", lambda: container)
    return unreachable_hint(
        Settings(env="dev", llm_provider="ollama", llm_model="llama3.1", ollama_base_url=url)
    )


def test_on_the_host_the_hint_does_not_blame_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sending someone to edit docker-compose.yml when they run uvicorn directly
    is worse than no message -- they go and fix something that was never broken."""
    hint = hint_for("http://localhost:11434", container=False, monkeypatch=monkeypatch)

    assert "ollama ls" in hint
    assert "docker" not in hint.lower()


def test_in_a_container_a_loopback_url_names_the_real_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`cp .env.example .env` used to put OLLAMA_BASE_URL=http://localhost:11434
    into Compose's ${...} substitution, aiming the container at itself."""
    hint = hint_for("http://localhost:11434", container=True, monkeypatch=monkeypatch)

    assert ".env" in hint
    assert "host.docker.internal" in hint
    assert "OLLAMA_HOST=0.0.0.0" not in hint, "the bind address is not the problem here"


def test_in_a_container_a_host_url_blames_the_bind_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hint = hint_for("http://host.docker.internal:11434", container=True, monkeypatch=monkeypatch)

    assert "OLLAMA_HOST=0.0.0.0:11434" in hint


def test_compose_does_not_substitute_ollama_base_url_from_dotenv() -> None:
    """Compose reads `.env` for ${VAR}. OLLAMA_BASE_URL is set there for HOST runs
    (http://localhost:11434), which inside a container names the container itself
    -- substituting it here silently breaks every containerised turn. The compose
    override is DOCKER_OLLAMA_BASE_URL precisely so the two cannot collide."""
    # Comments discuss ${OLLAMA_BASE_URL} by name; only live lines substitute.
    active = [
        line
        for line in Path("docker-compose.yml").read_text().splitlines()
        if not line.strip().startswith("#")
    ]

    assert not any("${OLLAMA_BASE_URL" in line for line in active)
    default = "${DOCKER_OLLAMA_BASE_URL:-http://host.docker.internal:11434}"
    assert any(default in line for line in active)


def test_host_net_overlay_points_everything_at_localhost() -> None:
    """In the host network namespace the container's localhost IS the host's, so
    both the bridge-network addresses from docker-compose.yml must be overridden
    -- half an override reaches Ollama but loses Postgres."""
    overlay = Path("docker-compose.host-net.yml").read_text()

    assert "network_mode: host" in overlay
    assert "http://localhost:11434" in overlay
    assert "@localhost:5433" in overlay


def test_the_host_net_overlay_documents_how_to_make_it_stick() -> None:
    """Passing both -f files every time is the trap: one plain `docker compose up`
    recreates the container on bridge networking and every turn starts failing.
    COMPOSE_FILE in .env is the fix, so the guidance has to be discoverable."""
    setting = "COMPOSE_FILE=docker-compose.yml:docker-compose.host-net.yml"

    assert setting in Path(".env.example").read_text()
    assert setting in Path("docker-compose.host-net.yml").read_text()


def test_explain_passes_through_a_failure_that_is_not_a_connection_problem() -> None:
    settings = Settings(env="dev", llm_provider="ollama", llm_model="llama3.1")

    assert explain(ValueError("model 'llama3.1' not found"), settings) == (
        "model 'llama3.1' not found"
    )


def test_explain_sees_through_a_wrapped_connection_error() -> None:
    """Providers re-raise transport errors wrapped, so matching the top type alone
    would silently fall back to the unhelpful message."""
    settings = Settings(env="dev", llm_provider="ollama", llm_model="llama3.1")
    try:
        try:
            raise httpx.ConnectError("All connection attempts failed")
        except httpx.ConnectError as cause:
            raise RuntimeError("during task with name 'clarify'") from cause
    except RuntimeError as wrapped:
        assert explain(wrapped, settings) == unreachable_hint(settings)


# --- chat mode vs agent mode -------------------------------------------------


def test_chat_is_the_default_mode(client: TestClient) -> None:
    """Callers that predate the setting keep the single-retrieval behaviour."""
    events = read_events(client, "When are expense reports due?", "t-mode-default")

    assert [e for e in events if e["type"] == "search"] == []


def test_agent_mode_streams_the_searches_it_ran(client: TestClient) -> None:
    body = {
        "user_id": USER,
        "thread_id": "t-mode-agent",
        "message": "When are expense reports due?",
        "mode": "agent",
    }
    with client.stream("POST", "/chat/stream", json=body) as response:
        assert response.status_code == 200, response.read()
        raw = "".join(response.iter_text())

    events = [
        json.loads(block.removeprefix("data:").strip())
        for block in raw.split("\n\n")
        if block.strip()
    ]
    searches = [e for e in events if e["type"] == "search"]

    assert searches, "agent mode should report the searches it ran"
    assert all(e["query"] for e in searches)
    assert events[-1]["type"] == "final"


def test_agent_mode_still_returns_citations(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={
            "user_id": USER,
            "thread_id": "t-mode-cite",
            "message": "How long do I have to submit an expense report?",
            "mode": "agent",
        },
    )

    assert response.status_code == 200
    assert response.json()["citations"], "agent mode does not relax grounding"


def test_an_unknown_mode_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"user_id": USER, "thread_id": "t-mode-bad", "message": "hi", "mode": "wizard"},
    )

    assert response.status_code == 422


def test_a_thread_can_mix_both_modes(client: TestClient) -> None:
    """Mode is per turn, not per conversation, so history has to survive a switch."""
    client.post(
        "/chat",
        json={
            "user_id": USER,
            "thread_id": "t-mode-mixed",
            "message": "We were discussing invoice 42.",
            "mode": "agent",
        },
    )
    body = chat(client, "t-mode-mixed", "What were we discussing?")

    assert "invoice 42" in str(body["answer"]).lower()


def test_a_provider_that_does_not_stream_arrives_in_one_piece(client: TestClient) -> None:
    """The scripted stand-in returns a whole message, so `messages` mode emits it
    as a single event. Degraded, not broken: the browser renders it the same way."""
    events = read_events(client, "When are expense reports due?", "t-stream-5")

    tokens = [e for e in events if e["type"] == "token"]
    assert len(tokens) == 1
    assert tokens[0]["text"] == events[-1]["answer"]
