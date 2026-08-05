"""Tests for the mem0 prompt ports + fallback parsers (isotope_zero.extraction).

Covers the four ``str.format()``-style template constants ported from
``mem0/configs/prompts.py`` (``FACT_EXTRACTION_PROMPT``,
``MEMORY_ANSWER_PROMPT``, ``UPDATE_CLASSIFICATION_PROMPT``,
``ENTITY_RELATIONSHIP_PROMPT``) and the three zero-dependency fallback
parsers (``parse_extraction_json``, ``parse_triplets``,
``parse_update_action``) that mirror mem0's tolerant JSON handling in
``mem0.memory.utils``.

Constraint 3 (local-first sub-ms): no LLM client is on the search hot path —
these are pure string templates + regex/JSON parsers.
"""
from __future__ import annotations

import json

from isotope_zero.extraction.prompts import (
    ENTITY_RELATIONSHIP_PROMPT,
    FACT_EXTRACTION_PROMPT,
    MEMORY_ANSWER_PROMPT,
    UPDATE_CLASSIFICATION_PROMPT,
    parse_extraction_json,
    parse_triplets,
    parse_update_action,
)


# ---------------------------------------------------------------------------
# Prompt template constants
# ---------------------------------------------------------------------------


def test_fact_extraction_prompt_format_fills_date_and_keeps_facts_key():
    """format() substitutes {date} and the body still references the facts key."""
    filled = FACT_EXTRACTION_PROMPT.format(date="2026-08-04")
    # Placeholder consumed.
    assert "{date}" not in filled
    # Date anchor present (mem0 uses Today's date is <date>).
    assert "2026-08-04" in filled
    # The "facts" json key referenced by the few-shot examples survives.
    assert '"facts"' in filled


def test_fact_extraction_prompt_format_with_messages_concat():
    """The filled prompt is a plain string a caller can append messages to."""
    base = FACT_EXTRACTION_PROMPT.format(date="2026-08-04")
    full = base + "\nuser: I like tea\nassistant: Noted."
    assert "I like tea" in full
    # Re-formatting the same template is deterministic.
    assert FACT_EXTRACTION_PROMPT.format(date="2026-08-04") == base


def test_memory_answer_prompt_is_static_template():
    """MEMORY_ANSWER_PROMPT has no placeholders and reads as a task preamble."""
    assert "{date}" not in MEMORY_ANSWER_PROMPT
    assert "answering questions" in MEMORY_ANSWER_PROMPT.lower()
    # str.format on a brace-free string is a no-op (proves it's format-safe).
    assert MEMORY_ANSWER_PROMPT.format() == MEMORY_ANSWER_PROMPT


def test_update_classification_prompt_lists_four_events():
    """UPDATE_CLASSIFICATION_PROMPT enumerates ADD/UPDATE/DELETE/NONE."""
    assert "{date}" not in UPDATE_CLASSIFICATION_PROMPT
    assert "ADD" in UPDATE_CLASSIFICATION_PROMPT
    assert "UPDATE" in UPDATE_CLASSIFICATION_PROMPT
    assert "DELETE" in UPDATE_CLASSIFICATION_PROMPT
    assert "NONE" in UPDATE_CLASSIFICATION_PROMPT
    # format()-safe: the JSON example braces are doubled.
    UPDATE_CLASSIFICATION_PROMPT.format()


def test_entity_relationship_prompt_describes_triplet_schema():
    """ENTITY_RELATIONSHIP_PROMPT names the source/relationship/destination keys."""
    assert "{date}" not in ENTITY_RELATIONSHIP_PROMPT
    for key in ("source", "relationship", "destination"):
        assert key in ENTITY_RELATIONSHIP_PROMPT
    # format()-safe: example braces are doubled.
    ENTITY_RELATIONSHIP_PROMPT.format()


# ---------------------------------------------------------------------------
# parse_extraction_json
# ---------------------------------------------------------------------------


def test_parse_extraction_json_fenced_facts_list():
    """Markdown-fenced JSON with a facts list parses to a dict."""
    raw = '```json\n{"facts": ["Loves tea", "Name is Sam"]}\n```'
    out = parse_extraction_json(raw)
    assert isinstance(out, dict)
    assert out["facts"] == ["Loves tea", "Name is Sam"]


def test_parse_extraction_json_trailing_comma():
    """Trailing commas before ]/} are tolerated by strict JSON loader."""
    raw = '{"facts": ["a", "b",]}'
    assert parse_extraction_json(raw) == {"facts": ["a", "b"]}


def test_parse_extraction_json_garbage_returns_empty_dict_no_exception():
    """Malformed input never raises; returns {} when nothing is usable."""
    for bad in ("total garbage {", "{{{{", "not json at all", "", None, 12345):
        assert parse_extraction_json(bad) == {}  # type: ignore[arg-type]


def test_parse_extraction_json_bare_list_wraps_as_facts():
    """A bare JSON list of strings is wrapped as {'facts': [...]}."""
    assert parse_extraction_json('["alpha", "beta"]') == {
        "facts": ["alpha", "beta"]
    }


def test_parse_extraction_json_loose_facts_from_text():
    """When strict JSON fails, quoted strings are regex-extracted as facts."""
    raw = 'Here are the facts: ["prefers rust", "lives in Berlin"].'
    out = parse_extraction_json(raw)
    assert out.get("facts") == ["prefers rust", "lives in Berlin"]


# ---------------------------------------------------------------------------
# parse_triplets
# ---------------------------------------------------------------------------


def test_parse_triplets_json_list_of_objects():
    """A JSON list of {source,relationship,destination} dicts -> tuples."""
    raw = json.dumps(
        [
            {"source": "Marcus", "relationship": "works_at", "destination": "Shopify"},
            {"source": "Marcus", "relationship": "spouse", "destination": "Elena"},
        ]
    )
    out = parse_triplets(raw)
    assert out == [
        ("Marcus", "works_at", "Shopify"),
        ("Marcus", "spouse", "Elena"),
    ]


def test_parse_triplets_fenced():
    """Fenced JSON triplet lists parse correctly."""
    raw = '```json\n[{"source": "A", "relationship": "r", "destination": "B"}]\n```'
    assert parse_triplets(raw) == [("A", "r", "B")]


def test_parse_triplets_garbage_returns_empty_no_exception():
    """Garbage input never raises; returns []."""
    for bad in ("noise", "{{{", "", None, "no triplets here"):
        assert parse_triplets(bad) == []  # type: ignore[arg-type]


def test_parse_triplets_arrow_form():
    """'source -- relationship -- destination' lines parse tolerantly."""
    raw = "Marcus -- works_at -- Shopify\nElena -- celebrates_at -- Osteria"
    assert parse_triplets(raw) == [
        ("Marcus", "works_at", "Shopify"),
        ("Elena", "celebrates_at", "Osteria"),
    ]


# ---------------------------------------------------------------------------
# parse_update_action
# ---------------------------------------------------------------------------


def test_parse_update_action_delete_with_id():
    """DELETE event with an id field -> ('DELETE', 'id123')."""
    raw = '{"event": "DELETE", "id": "id123"}'
    assert parse_update_action(raw) == ("DELETE", "id123")


def test_parse_update_action_bare_keyword_and_id():
    """A bare 'DELETE id123' string -> ('DELETE', 'id123')."""
    assert parse_update_action("DELETE id123") == ("DELETE", "id123")


def test_parse_update_action_add_in_memory_list():
    """A {'memory': [{'event': 'ADD', 'id': '7'}]} payload -> ('ADD', '7')."""
    raw = '{"memory": [{"id": "7", "text": "...", "event": "ADD"}]}'
    assert parse_update_action(raw) == ("ADD", "7")


def test_parse_update_action_update_keeps_case():
    """UPDATE classification preserves the existing id verbatim."""
    raw = '{"memory": [{"id": "abc-1", "event": "UPDATE", "old_memory": "x"}]}'
    assert parse_update_action(raw) == ("UPDATE", "abc-1")


def test_parse_update_action_none_event():
    """NONE event yields ('NONE', None) — the default no-op."""
    assert parse_update_action('{"event": "NONE", "id": "0"}') == ("NONE", "0")


def test_parse_update_action_garbage_returns_none_no_exception():
    """Garbage input never raises; returns ('NONE', None)."""
    for bad in ("???", "{{{", "", None, 42):
        assert parse_update_action(bad) == ("NONE", None)  # type: ignore[arg-type]


def test_parse_update_action_lowercase_event_normalized():
    """Lowercase event labels are uppercased to the canonical action."""
    assert parse_update_action('{"event": "delete", "id": "x9"}') == (
        "DELETE",
        "x9",
    )
