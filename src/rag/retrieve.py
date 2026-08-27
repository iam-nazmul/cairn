"""Vector store interface and the stubbed, seeded implementation (SPEC §6.2).

The `VectorStore` protocol is the seam described in src/rag/fixtures.py: swap in
a pgvector- or managed-DB-backed implementation without touching the graph.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from src.graph.state import RetrievedChunk
from src.rag.fixtures import SEED_DOCS, SeedDoc

_WORD_RE = re.compile(r"[a-z0-9]+")

# Words carrying no retrieval signal. Kept tiny and explicit: the lexical scorer
# below is a stand-in for embeddings, not a search engine.
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
    """Similarity search over the corpus.

    Implementations MUST return `source` and `score` alongside `text`: the
    citation requirement in CLAUDE.md depends on it.
    """

    async def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]: ...


class InMemoryVectorStore:
    """Deterministic, dependency-free `VectorStore` used for local runs and tests.

    Scores by token-overlap coefficient rather than embedding cosine similarity so
    that results are stable and require no API key. Scores are in [0.0, 1.0] and
    comparable to each other, which is all the graph's routing threshold needs.
    """

    def __init__(self, docs: tuple[SeedDoc, ...] = SEED_DOCS) -> None:
        self._docs = docs
        self._index: list[tuple[SeedDoc, set[str]]] = [(d, tokenize(d.text)) for d in docs]

    async def search(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
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

        # Sort by score desc, then source for a stable tie-break.
        scored.sort(key=lambda c: (-c["score"], c["source"]))
        return scored[:top_k]


def get_vector_store() -> VectorStore:
    """Default vector store. Replace the body to point at a real index."""
    return InMemoryVectorStore()
