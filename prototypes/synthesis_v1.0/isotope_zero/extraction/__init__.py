"""Prompt templates and zero-dependency fallback parsers.

Ports mem0's prompt TEMPLATE STRINGS into local-first, sub-ms isotope_zero:
the prompts are pure ``str.format()``-style templates (no ``datetime.now()``
interpolation at import time, no LLM client import) so they sit OFF the
vector search hot path (constraint 3). The fallback parsers never raise on
malformed input, mirroring mem0's tolerant JSON handling in
``mem0.memory.utils.normalize_facts`` and ``mem0.memory.utils.normalize_entities``.
"""
from __future__ import annotations

from isotope_zero.extraction.prompts import (
    ENTITY_RELATIONSHIP_PROMPT,
    FACT_EXTRACTION_PROMPT,
    MEMORY_ANSWER_PROMPT,
    UPDATE_CLASSIFICATION_PROMPT,
    parse_extraction_json,
    parse_triplets,
    parse_update_action,
)

__all__ = [
    "ENTITY_RELATIONSHIP_PROMPT",
    "FACT_EXTRACTION_PROMPT",
    "MEMORY_ANSWER_PROMPT",
    "UPDATE_CLASSIFICATION_PROMPT",
    "parse_extraction_json",
    "parse_triplets",
    "parse_update_action",
]
