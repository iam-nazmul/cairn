"""Per-node instrumentation (SPEC §10 observability).

Wraps each node so every super-step logs its duration, plus the signals worth
having when a turn goes wrong: which chunks were retrieved and at what scores,
how many long-term facts were loaded, and the approximate token cost of the
answer. Node code stays clean -- instrumentation is applied at wiring time.

`thread_id` is included on every line so a single conversation can be followed
through the logs.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING, Any

from langchain_core.messages.utils import count_tokens_approximately
from langgraph.config import get_config
from langgraph.runtime import Runtime

from src.config import Context
from src.graph.state import ChatState

if TYPE_CHECKING:
    from src.graph.nodes import Node

logger = logging.getLogger("cairn.graph")


def _thread_id() -> str:
    try:
        value = get_config().get("configurable", {}).get("thread_id")
    except Exception:  # pragma: no cover - outside a graph run
        return "-"
    return str(value) if value else "-"


def _summarize(result: dict[str, Any]) -> str:
    """The per-node detail worth logging, if the node produced any."""
    parts: list[str] = []

    if "retrieved" in result:
        chunks = result["retrieved"] or []
        if chunks:
            hits = " ".join(f"{c['source']}={c['score']}" for c in chunks)
            parts.append(f"hits={len(chunks)} {hits}")
        else:
            parts.append("hits=0")

    if "long_term_facts" in result:
        parts.append(f"facts={len(result['long_term_facts'] or [])}")

    if "answer" in result:
        answer = result["answer"] or ""
        parts.append(f"answer_tokens~{count_tokens_approximately([('ai', answer)])}")
        if not answer:
            parts.append("uncited=dropped")

    return " ".join(parts)


def instrument(name: str, node: Node) -> Node:
    """Wrap a node so it logs its timing and outputs."""

    async def instrumented(state: ChatState, *, runtime: Runtime[Context]) -> dict[str, Any]:
        started = perf_counter()
        try:
            result = await node(state, runtime=runtime)
        except Exception:
            elapsed_ms = (perf_counter() - started) * 1000
            logger.exception("node=%s thread=%s ms=%.1f FAILED", name, _thread_id(), elapsed_ms)
            raise

        elapsed_ms = (perf_counter() - started) * 1000
        logger.info(
            "node=%s thread=%s ms=%.1f %s", name, _thread_id(), elapsed_ms, _summarize(result)
        )
        return result

    return instrumented
