"""Grounding / citation eval (SPEC §11 "citation faithfulness").

Asserts answers are traceable to chunks that retrieval actually returned, rather
than to the model's priors.
"""

from __future__ import annotations

import pytest
from langgraph.graph.state import CompiledStateGraph

from src.config import Context
from src.graph.state import ChatState
from src.rag.prompts import CITATION_RE, cited_chunks

Graph = CompiledStateGraph[ChatState, Context, ChatState, ChatState]

EVAL_QUESTIONS = [
    "How long do I have to submit an expense report?",
    "How do I connect to the corporate VPN?",
    "What happens during onboarding week?",
    "What does accounts payable need on a supplier invoice?",
    "How fast are priority one support tickets answered?",
]


@pytest.mark.parametrize("question", EVAL_QUESTIONS)
async def test_answer_cites_a_chunk_that_was_actually_retrieved(
    graph: Graph, context: Context, question: str
) -> None:
    config = {"configurable": {"thread_id": f"t-eval-{hash(question)}"}}
    out = await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}], "question": question},
        config,
        context=context,
    )

    retrieved = out["retrieved"]
    assert retrieved, f"eval question should hit the seeded corpus: {question!r}"

    cited = cited_chunks(out["answer"], retrieved)
    assert cited, f"answer carries no citation marker: {out['answer']!r}"

    retrieved_sources = {c["source"] for c in retrieved}
    for chunk in cited:
        assert chunk["source"] in retrieved_sources


@pytest.mark.parametrize("question", EVAL_QUESTIONS)
async def test_answer_text_is_traceable_to_the_cited_chunk(
    graph: Graph, context: Context, question: str
) -> None:
    """The cited claim must appear in the chunk it points at -- not merely be plausible."""
    config = {"configurable": {"thread_id": f"t-trace-{hash(question)}"}}
    out = await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}], "question": question},
        config,
        context=context,
    )

    cited = cited_chunks(out["answer"], out["retrieved"])
    claim = CITATION_RE.split(out["answer"])[0].strip()
    assert any(claim in chunk["text"] for chunk in cited), (
        f"claim {claim!r} does not appear verbatim in any cited chunk"
    )


async def test_citation_markers_never_point_past_the_retrieved_set(
    graph: Graph, context: Context
) -> None:
    question = "How do I connect to the corporate VPN?"
    config = {"configurable": {"thread_id": "t-marker-range"}}
    out = await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}], "question": question},
        config,
        context=context,
    )

    markers = [int(m) for m in CITATION_RE.findall(out["answer"])]
    assert markers
    assert all(1 <= m <= len(out["retrieved"]) for m in markers)


def test_cited_chunks_ignores_dangling_markers() -> None:
    chunks: list[dict[str, object]] = [{"text": "t", "source": "doc://kb/1", "score": 0.5}]
    assert cited_chunks("claim [S9]", chunks) == []  # type: ignore[arg-type]
