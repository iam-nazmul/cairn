"""The retrieval contract: {text, source, score} per chunk (CLAUDE.md)."""

from __future__ import annotations

from src.rag.fixtures import SEED_DOCS
from src.rag.retrieve import InMemoryVectorStore, VectorStore


async def test_chunks_carry_text_source_and_score() -> None:
    store = InMemoryVectorStore()
    chunks = await store.search("expense report receipts", top_k=4)

    assert chunks, "seeded corpus should match an obviously on-topic query"
    for chunk in chunks:
        assert set(chunk) == {"text", "source", "score"}
        assert chunk["text"] and chunk["source"]
        assert 0.0 <= chunk["score"] <= 1.0


async def test_results_are_ranked_by_descending_score() -> None:
    store = InMemoryVectorStore()
    chunks = await store.search("how do I connect to the VPN", top_k=4)
    scores = [c["score"] for c in chunks]
    assert scores == sorted(scores, reverse=True)


async def test_top_k_is_respected() -> None:
    store = InMemoryVectorStore()
    assert len(await store.search("expense invoice vpn onboarding support", top_k=2)) <= 2


async def test_off_corpus_query_retrieves_nothing() -> None:
    """The empty-retrieval case that must route to clarify, not to model priors."""
    store = InMemoryVectorStore()
    assert await store.search("quantum chromodynamics lattice gauge", top_k=4) == []


async def test_sources_are_citable_identifiers() -> None:
    store = InMemoryVectorStore()
    chunks = await store.search("invoice purchase order", top_k=4)
    known = {d.source for d in SEED_DOCS}
    assert all(c["source"] in known for c in chunks)


def test_in_memory_store_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryVectorStore(), VectorStore)
