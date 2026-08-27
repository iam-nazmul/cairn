"""Chat-model seam: `fake` is a scripted stand-in, anything else a real provider."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse

import httpx
from langchain.messages import AIMessage, AnyMessage

from src.config import Settings
from src.rag.retrieve import tokenize

_BLOCK_RE = re.compile(r"\[S(\d+)\] source=(\S+) score=([0-9.]+)\n(.+?)(?=\n\n\[S\d+\]|\Z)", re.S)
_FACTS_RE = re.compile(r"^- (.+)$", re.M)
# "My name is Alice", "my preferred language is Bengali"
_RECALL_RE = re.compile(r"\bmy ([a-z][a-z ]{0,24}?) is ([^.,!?\n]+)", re.I)
# Thread-scoped by design: deliberately not a durable fact.
_TOPIC_RE = re.compile(r"\bwe (?:were|are) discussing\s+([^.,!?\n]+)", re.I)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class ChatModel(Protocol):
    """The slice of LangChain's chat-model interface the graph uses."""

    async def ainvoke(self, input: Sequence[AnyMessage], /) -> AIMessage: ...


def _text(message: AnyMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def _best_sentence(text: str, question_tokens: set[str]) -> str:
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        return text.strip()
    return max(sentences, key=lambda s: len(tokenize(s) & question_tokens))


class DeterministicChatModel:
    """Scripted stand-in for a chat model. See `ChatModel.ainvoke`."""

    async def ainvoke(self, input: Sequence[AnyMessage], /) -> AIMessage:
        """See `ChatModel.ainvoke`. Reflects back the context it was handed."""
        messages = list(input)
        system = _text(messages[0]) if messages else ""
        turns = messages[1:]

        question = _text(turns[-1]) if turns else ""
        prior = [_text(m) for m in turns[:-1] if m.type == "human"]

        blocks = _BLOCK_RE.findall(system)
        facts = _FACTS_RE.findall(system)
        question_tokens = tokenize(question)

        parts: list[str] = []

        # Memory, not model priors -- allowed even on the clarify path.
        recalled = self._recall(prior, question)
        topic = self._recall_topic(prior, question)
        if recalled is not None:
            attribute, value = recalled
            parts.append(f"Your {attribute} is {value}.")
        elif topic is not None:
            parts.append(f"We were discussing {topic}.")
        else:
            fact = self._matching_fact(facts, question_tokens)
            if fact is not None:
                parts.append(f"From what I know about you: {fact}")

        if blocks:
            index, _source, _score, text = blocks[0]
            parts.append(f"{_best_sentence(text, question_tokens)} [S{index}]")
        elif not parts:
            parts.append(
                "I could not find anything in the knowledge base about that, and "
                "nothing we have discussed covers it. Could you rephrase or add "
                "more detail?"
            )

        return AIMessage(content=" ".join(parts))

    @staticmethod
    def _recall(prior_human_turns: Sequence[str], question: str) -> tuple[str, str] | None:
        """Find a 'my X is Y' statement from earlier in the thread that answers X."""
        memo: dict[str, str] = {}
        for turn in prior_human_turns:
            for match in _RECALL_RE.finditer(turn):
                memo[match.group(1).strip().lower()] = match.group(2).strip()

        lowered = question.lower()
        for attribute, value in memo.items():
            if attribute in lowered:
                return attribute, value
        return None

    @staticmethod
    def _recall_topic(prior_human_turns: Sequence[str], question: str) -> str | None:
        """Recall the thread's topic -- checkpointer territory, never the Store."""
        if "discussing" not in question.lower():
            return None
        for turn in reversed(prior_human_turns):
            match = _TOPIC_RE.search(turn)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _matching_fact(facts: Sequence[str], question_tokens: set[str]) -> str | None:
        best: tuple[int, str] | None = None
        for fact in facts:
            overlap = len(tokenize(fact) & question_tokens)
            if overlap and (best is None or overlap > best[0]):
                best = (overlap, fact)
        return None if best is None else best[1]


def _is_connection_failure(exc: BaseException) -> bool:
    """Walk the cause chain for a refused/failed connection.

    Every provider tunnels these through httpx, but each wraps them differently,
    so match the chain rather than one library's exception type.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ConnectionError | httpx.TransportError):
            return True
        current = current.__cause__ or current.__context__
    return False


def in_container() -> bool:
    """Docker creates this marker file; Compose runs are the case that matters."""
    return Path("/.dockerenv").exists()


def unreachable_hint(settings: Settings) -> str:
    """What to actually do about an unreachable provider.

    The right advice is the opposite depending on where this process runs, so one
    message for both is worse than useless -- it sends the reader off to fix
    something that was never broken.
    """
    if settings.llm_provider != "ollama":
        return f"Cannot reach the {settings.llm_provider!r} provider."

    url = settings.ollama_base_url
    preamble = f"Cannot reach Ollama at {url}."
    loopback = (urlparse(url).hostname or "") in ("localhost", "127.0.0.1", "::1")

    if not in_container():
        return (
            f"{preamble} Is it running? `ollama ls` should answer, and "
            "OLLAMA_BASE_URL must point at it."
        )
    if loopback:
        # Overwhelmingly this is OLLAMA_BASE_URL leaking out of .env into
        # Compose's ${...} substitution.
        return (
            f"{preamble} Inside a container that address is the container itself, "
            "not your machine. Compose substitutes OLLAMA_BASE_URL from the "
            "project's .env file, so a value meant for a host run overrides the "
            "compose default: unset it in .env (see .env.example), or set it to "
            "http://host.docker.internal:11434. `docker compose config` shows "
            "what actually reached the container."
        )
    return (
        f"{preamble} Ollama binds to 127.0.0.1 by default, which a container "
        "cannot reach: set OLLAMA_HOST=0.0.0.0:11434 on the host (see "
        "docker-compose.yml), run `docker compose --profile ollama up`, or start "
        "with LLM_PROVIDER=fake."
    )


def explain(exc: BaseException, settings: Settings) -> str:
    """A turn's failure, phrased for whoever has to fix it."""
    return unreachable_hint(settings) if _is_connection_failure(exc) else str(exc)


async def probe(settings: Settings) -> bool | None:
    """Is the provider answering? `None` when there is nothing to probe."""
    if settings.llm_provider != "ollama":
        # `fake` is always up; hosted providers have no free liveness endpoint.
        return None
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
        return response.status_code == 200
    except Exception:
        return False


def get_chat_model(settings: Settings) -> ChatModel:
    """Resolve the configured provider."""
    if settings.llm_provider == "fake":
        return DeterministicChatModel()

    if not settings.llm_model:
        raise ValueError(
            f"LLM_MODEL is required when LLM_PROVIDER={settings.llm_provider!r} "
            "(e.g. LLM_MODEL=llama3.1 for Ollama)"
        )

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return cast(
            ChatModel,
            ChatOllama(
                model=settings.llm_model,
                base_url=settings.ollama_base_url,
                temperature=settings.llm_temperature,
                num_ctx=settings.ollama_num_ctx,
            ),
        )

    from langchain.chat_models import init_chat_model

    model = init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=settings.llm_temperature,
    )
    return cast(ChatModel, model)
