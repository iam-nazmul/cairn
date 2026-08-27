"""Vector store interface and the seeded stub implementation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from langchain.messages import AnyMessage

from src.graph.state import RetrievedChunk
from src.rag.fixtures import SEED_DOCS, SeedDoc

_WORD_RE = re.compile(r"[a-z0-9]+")

# The lexical scorer is a stand-in for embeddings, not a search engine.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)


def tokenize(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


@runtime_checkable
class VectorStore(Protocol):
    """Similarity search. Implementations MUST return `source` and `score`."""

    async def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]: ...


class InMemoryVectorStore:
    """Deterministic, dependency-free `VectorStore` for local runs and tests."""

    def __init__(self, docs: tuple[SeedDoc, ...] = SEED_DOCS) -> None:
        self._docs = docs
        self._index: list[tuple[SeedDoc, set[str]]] = [(d, tokenize(d.text)) for d in docs]

    async def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """See `VectorStore.search`. Scores by token overlap, not embeddings."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scored: list[RetrievedChunk] = []
        for doc, doc_tokens in self._index:
            overlap = len(query_tokens & doc_tokens)
            if not overlap:
                continue
            score = round(overlap / len(query_tokens), 4)
            scored.append(RetrievedChunk(text=doc.text, source=doc.source, score=score))

        scored.sort(key=lambda c: (-c["score"], c["source"]))
        return scored[:top_k]


def augment_query_with_history(
    question: str, messages: Sequence[AnyMessage], max_turns: int = 2
) -> str:
    """Fold recent user turns into the search query (SPEC §6.2)."""
    prior = [str(m.content) for m in messages if m.type == "human"]
    if prior and prior[-1].strip() == question.strip():
        prior = prior[:-1]  # drop the current turn
    recent = prior[-max_turns:]
    return " ".join([*recent, question]) if recent else question


def get_vector_store() -> VectorStore:
    """Default vector store. Replace the body to point at a real index."""
    return InMemoryVectorStore()
