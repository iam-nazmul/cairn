"""Fact extraction rules (SPEC §11, resolved in M3)."""

from __future__ import annotations

from src.memory.facts import (
    FACTS_NS,
    PREFERENCES_NS,
    extract_facts,
    parse_llm_facts,
    slugify,
)


def test_extracts_a_self_description() -> None:
    (fact,) = extract_facts("My name is Alice.")
    assert fact.key == "name"
    assert fact.text == "name is Alice"
    assert fact.namespace == FACTS_NS


def test_preferences_land_in_the_preferences_namespace() -> None:
    (fact,) = extract_facts("My preferred language is Bengali.")
    assert fact.namespace == PREFERENCES_NS
    assert fact.key == "preferred-language"


def test_the_same_attribute_always_yields_the_same_key() -> None:
    """The idempotency requirement in SPEC §7.2: restating updates, never duplicates."""
    first = extract_facts("My name is Alice.")[0]
    second = extract_facts("my name is Alicia")[0]
    assert first.key == second.key
    assert first.text != second.text


def test_explicit_remember_command() -> None:
    (fact,) = extract_facts("Remember that I work in accounts payable.")
    assert fact.namespace == FACTS_NS
    assert "accounts payable" in fact.text


def test_remember_with_a_structured_payload_uses_the_attribute_key() -> None:
    (fact,) = extract_facts("Remember that my role is auditor.")
    assert fact.key == "role"
    assert fact.text == "role is auditor"


def test_i_prefer_keys_on_the_value_so_preferences_coexist() -> None:
    facts = extract_facts("I prefer short answers.") + extract_facts("I prefer dark mode.")
    assert len({f.key for f in facts}) == 2
    assert all(f.namespace == PREFERENCES_NS for f in facts)


def test_questions_are_not_facts() -> None:
    assert extract_facts("What's my name?") == []
    assert extract_facts("Is my name Alice?") == []


def test_conversational_topics_are_not_facts() -> None:
    """Thread-scoped context belongs to the checkpointer, never the Store."""
    assert extract_facts("We were discussing invoice 42.") == []
    assert extract_facts("How long do I have to submit an expense report?") == []


def test_empty_input() -> None:
    assert extract_facts("") == []
    assert extract_facts("   ") == []


def test_long_values_are_truncated() -> None:
    (fact,) = extract_facts("My note is " + "x" * 500)
    assert len(fact.text) < 200


def test_slugify() -> None:
    assert slugify("Preferred Language") == "preferred-language"
    assert slugify("!!!") == "note"


def test_parse_llm_facts_reads_attribute_value_lines() -> None:
    facts = parse_llm_facts("name: Alice\nrole: auditor\n")
    assert [f.key for f in facts] == ["name", "role"]
    assert facts[0].text == "name is Alice"


def test_parse_llm_facts_handles_none_and_junk() -> None:
    assert parse_llm_facts("NONE") == []
    assert parse_llm_facts("no colon here") == []
    assert parse_llm_facts("") == []


def test_parse_llm_facts_uses_the_same_stable_keys_as_the_rules() -> None:
    assert parse_llm_facts("name: Alice")[0].key == extract_facts("My name is Alice.")[0].key
