"""Durable-fact extraction for `write_memory` (SPEC §11)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from langgraph.store.base import BaseStore

FACTS_NS = "facts"
PREFERENCES_NS = "preferences"

# "remember that X", "remember: X", "remember X"
_REMEMBER_RE = re.compile(r"\bremember\b(?:\s+that\b|\s*:)?\s+(.+)", re.I)
# "my name is Alice", "my preferred language is Bengali"
_ATTRIBUTE_RE = re.compile(r"\bmy ([a-z][a-z ]{0,24}?) is\s+([^.,!?\n]+)", re.I)
# "I prefer answers in Bengali"
_PREFER_RE = re.compile(r"\bi (?:prefer|really like)\s+([^.,!?\n]+)", re.I)

_PREFERENCE_WORDS = frozenset(
    {"preference", "preferences", "preferred", "language", "tone", "style", "format", "timezone"}
)

_MAX_VALUE_CHARS = 120


@dataclass(frozen=True)
class Fact:
    """One durable fact bound for the Store."""

    key: str  # stable: the same attribute always lands on the same key
    text: str
    namespace: str  # FACTS_NS | PREFERENCES_NS


async def load_namespace(store: BaseStore, user_id: str, namespace: str, limit: int) -> list[str]:
    """One namespace's facts. SPEC §13.4 splits them: preferences shape how an
    answer is written, facts are hints for what to search for."""
    items = await store.asearch((user_id, namespace), limit=limit)
    return sorted(text for item in items if (text := str(item.value.get("text", ""))))


async def load_user_facts(store: BaseStore, user_id: str, limit: int) -> list[str]:
    """Every durable fact on file for `user_id`, across both fact namespaces."""
    facts: list[str] = []
    for namespace in (FACTS_NS, PREFERENCES_NS):
        facts.extend(await load_namespace(store, user_id, namespace, limit))
    return sorted(f for f in facts if f)


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "note"


def _namespace_for(attribute: str) -> str:
    words = set(re.findall(r"[a-z]+", attribute.lower()))
    return PREFERENCES_NS if words & _PREFERENCE_WORDS else FACTS_NS


def _clean(value: str) -> str:
    return value.strip().rstrip(".").strip()[:_MAX_VALUE_CHARS]


def _from_attribute(text: str) -> list[Fact]:
    facts: list[Fact] = []
    for match in _ATTRIBUTE_RE.finditer(text):
        attribute = _clean(match.group(1)).lower()
        value = _clean(match.group(2))
        if not attribute or not value:
            continue
        facts.append(
            Fact(
                key=slugify(attribute),
                text=f"{attribute} is {value}",
                namespace=_namespace_for(attribute),
            )
        )
    return facts


def extract_facts(text: str) -> list[Fact]:
    """Pull durable facts out of a single user turn."""
    if not text or not text.strip():
        return []

    facts: list[Fact] = []
    seen: set[str] = set()

    def add(new: Sequence[Fact]) -> None:
        for fact in new:
            identity = f"{fact.namespace}/{fact.key}"
            if identity not in seen:
                seen.add(identity)
                facts.append(fact)

    # Explicit instruction wins.
    remembered = _REMEMBER_RE.search(text)
    if remembered:
        payload = _clean(remembered.group(1))
        structured = _from_attribute(payload)
        if structured:
            add(structured)
        elif payload:
            add([Fact(key=f"note-{slugify(payload)}", text=payload, namespace=FACTS_NS)])
        return facts

    add(_from_attribute(text))

    # Keyed on the value, so distinct preferences coexist and restatements upsert.
    for match in _PREFER_RE.finditer(text):
        value = _clean(match.group(1))
        if value:
            add(
                [
                    Fact(
                        key=f"prefers-{slugify(value)}",
                        text=f"prefers {value}",
                        namespace=PREFERENCES_NS,
                    )
                ]
            )

    return facts


LLM_EXTRACTION_PROMPT = """Extract durable facts about the USER from their message.

A durable fact is something still true in a different conversation next month: \
their name, role, or a standing preference. NOT the topic they are asking about.

Output one fact per line as `attribute: value`, lower-case attribute. \
Output exactly NONE if there are no durable facts."""


def parse_llm_facts(response: str) -> list[Fact]:
    """Parse the LLM extractor's output into the same keyed shape as the rules."""
    facts: list[Fact] = []
    seen: set[str] = set()
    for line in response.splitlines():
        line = line.strip().lstrip("-*").strip()
        if not line or line.upper().startswith("NONE"):
            continue
        attribute, separator, value = line.partition(":")
        if not separator:
            continue
        attribute = _clean(attribute).lower()
        value = _clean(value)
        if not attribute or not value:
            continue
        key = slugify(attribute)
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            Fact(key=key, text=f"{attribute} is {value}", namespace=_namespace_for(attribute))
        )
    return facts
