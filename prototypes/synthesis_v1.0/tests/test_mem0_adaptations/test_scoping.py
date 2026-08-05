"""Tests for the multi-tier scoping port (isotope_zero.core.scoping + store filter).

Covers two layers:

1. ``isotope_zero.core.scoping`` — pure functions ``parse_scope`` /
   ``build_scope`` / ``match_scope`` porting mem0's ``_build_session_scope``
   (``mem0/memory/main.py:407``) and the subset-filter visibility rule.

2. ``MemoryStore.sql_lookup(scope=...)`` — the on-disk enforcement of the
   same backward-compatible visibility rule via a ``scope = ? OR scope =
   'default' OR scope IS NULL`` clause. A global (``default``) card is
   visible to every scoped query; a scoped query never leaks across
   tenants; ``scope=None`` (the default) disables filtering entirely.

``vector_search``/``hybrid_search`` already carry a ``scope`` param (exact
   mask + ``None``=global); we assert their existing behavior too so the
   whole scoping surface is pinned in one place.

Conventions match the existing suite: ``:memory:`` store, plain ``test_*``
functions, real ``MemoryStore`` wiring.
"""
from __future__ import annotations

from isotope_zero.core.scoping import (
    GLOBAL_SCOPE,
    build_scope,
    match_scope,
    parse_scope,
)
from isotope_zero.core.store import MemoryStore
from isotope_zero.types import MemoryCard, now_ts


def _card(id: str, fact: str = "A fact.", embedding: list[float] | None = None) -> MemoryCard:
    return MemoryCard(
        id=id,
        fact=fact,
        evidence="evidence",
        timestamp=now_ts(),
        tags=[],
        embedding=embedding,
        source_tokens=5,
    )


# --------------------------------------------------------------------------- #
# Pure-function helpers (isotope_zero.core.scoping)
# --------------------------------------------------------------------------- #
def test_build_scope_canonical_sorts_keys_and_joins_with_ampersand():
    """build_scope emits sorted key=value pairs joined by '&' (mem0:407)."""
    # Input in non-sorted order; output MUST be agent < run < user.
    s = build_scope({"user_id": "U1", "agent_id": "A1", "run_id": "R1"})
    assert s == "agent_id=A1&run_id=R1&user_id=U1"


def test_parse_scope_round_trips_build_scope():
    """parse_scope(build_scope(d)) == d for a canonical dict."""
    d = {"agent_id": "A1", "run_id": "R1", "user_id": "U1"}
    s = build_scope(d)
    assert parse_scope(s) == d


def test_build_scope_empty_variants_yield_global():
    """None / {} / all-empty-values dicts all produce the GLOBAL_SCOPE sentinel."""
    assert build_scope(None) == GLOBAL_SCOPE
    assert build_scope({}) == GLOBAL_SCOPE
    assert build_scope({"user_id": "", "agent_id": None}) == GLOBAL_SCOPE


def test_parse_scope_global_variants_and_tolerances():
    """None / '' / 'default' parse to {}; trailing '&' and whitespace tolerated."""
    assert parse_scope(None) == {}
    assert parse_scope("") == {}
    assert parse_scope("default") == {}
    # Trailing ampersand does not inject an empty pair.
    assert parse_scope("agent_id=A1&") == {"agent_id": "A1"}
    # Surrounding whitespace is stripped.
    assert parse_scope("  user_id=U1  ") == {"user_id": "U1"}
    # A value containing '=' survives (partition on FIRST '=').
    assert parse_scope("user_id=U=1") == {"user_id": "U=1"}
    # A bare token with no '=' is dropped, not guessed.
    assert parse_scope("baretoken&user_id=U1") == {"user_id": "U1"}


def test_match_scope_global_query_sees_every_card():
    """A None/''/default query matches any card scope."""
    assert match_scope("user_id=U1", None) is True
    assert match_scope("user_id=U1", "") is True
    assert match_scope("user_id=U1", "default") is True
    assert match_scope("agent_id=A1&run_id=R1", "default") is True


def test_match_scope_global_card_visible_to_scoped_query():
    """A default/''/None card is visible to every scoped query (backward compat)."""
    assert match_scope("default", "user_id=U1") is True
    assert match_scope("", "user_id=U1") is True
    assert match_scope(None, "user_id=U1") is True


def test_match_scope_subset_filter_and_mismatches():
    """Query is a subset filter: every query key must match the card's value."""
    # Exact scoped match.
    assert match_scope("user_id=U1", "user_id=U1") is True
    # Query narrows card identity (card has MORE keys than the query).
    assert match_scope("agent_id=A1&run_id=R1&user_id=U1", "user_id=U1") is True
    # Query key absent on card -> no match.
    assert match_scope("agent_id=A1", "user_id=U1") is False
    # Query key present but value differs -> no match.
    assert match_scope("user_id=U1", "user_id=U2") is False
    # Two-key query against a one-key card missing the other key.
    assert match_scope("user_id=U1", "agent_id=A1&user_id=U1") is False


# --------------------------------------------------------------------------- #
# MemoryStore.sql_lookup scope filter (on-disk enforcement)
# --------------------------------------------------------------------------- #
def test_sql_lookup_scope_none_finds_all_tenants():
    """scope=None (default) disables filtering — every live card surfaces."""
    store = MemoryStore(":memory:")
    store.add(_card("g", "global fact"), scope="default")
    store.add(_card("u1", "user one fact"), scope="user_id=U1")
    store.add(_card("u2", "user two fact"), scope="user_id=U2")

    hits = store.sql_lookup("fact", "fact", scope=None)
    ids = {h.id for h in hits}
    assert ids == {"g", "u1", "u2"}


def test_sql_lookup_scoped_query_finds_own_tenant_plus_global():
    """A scoped query returns its own tenant's cards AND global (default) cards."""
    store = MemoryStore(":memory:")
    store.add(_card("g", "shared fact"), scope="default")
    store.add(_card("u1", "shared fact"), scope="user_id=U1")
    store.add(_card("u2", "shared fact"), scope="user_id=U2")

    hits = store.sql_lookup("fact", "shared fact", scope="user_id=U1")
    ids = {h.id for h in hits}
    # u1 (exact scope) + g (global default, backward compat) — NOT u2.
    assert ids == {"u1", "g"}
    assert "u2" not in ids


def test_sql_lookup_scoped_query_excludes_other_tenants():
    """A user_id=U2 query must NOT surface a user_id=U1 card."""
    store = MemoryStore(":memory:")
    store.add(_card("u1", "tenant fact"), scope="user_id=U1")

    hits = store.sql_lookup("fact", "tenant fact", scope="user_id=U2")
    assert hits == []


def test_sql_lookup_global_card_found_by_any_query_including_none():
    """A default-scope card is visible to None, to 'default', and to scoped queries."""
    store = MemoryStore(":memory:")
    store.add(_card("g", "global only"), scope="default")

    assert {h.id for h in store.sql_lookup("fact", "global only", scope=None)} == {"g"}
    assert {h.id for h in store.sql_lookup("fact", "global only", scope="default")} == {"g"}
    assert {h.id for h in store.sql_lookup("fact", "global only", scope="user_id=U1")} == {"g"}


def test_sql_lookup_scope_filter_substring_path_too():
    """The scope filter applies on the substring (LIKE) path, not just exact."""
    store = MemoryStore(":memory:")
    store.add(_card("u1", "the quick brown fox"), scope="user_id=U1")
    store.add(_card("u2", "the quick brown fox"), scope="user_id=U2")

    # Substring match (no wildcards in value, but value != fact so exact path
    # misses and we fall through to the substring LIKE path).
    u1_hits = store.sql_lookup("fact", "quick brown", scope="user_id=U1")
    assert {h.id for h in u1_hits} == {"u1"}
    u2_hits = store.sql_lookup("fact", "quick brown", scope="user_id=U2")
    assert {h.id for h in u2_hits} == {"u2"}
    none_hits = store.sql_lookup("fact", "quick brown", scope=None)
    assert {h.id for h in none_hits} == {"u1", "u2"}


def test_sql_lookup_scope_filter_tags_path():
    """The scope filter applies on the tags (Python-side membership) path too."""
    store = MemoryStore(":memory:")
    c1 = _card("u1", "fact", embedding=None)
    c1.tags = ["secret"]
    store.add(c1, scope="user_id=U1")
    c2 = _card("u2", "fact", embedding=None)
    c2.tags = ["secret"]
    store.add(c2, scope="user_id=U2")

    u1_hits = store.sql_lookup("tags", "secret", scope="user_id=U1")
    assert {h.id for h in u1_hits} == {"u1"}
    u2_hits = store.sql_lookup("tags", "secret", scope="user_id=U2")
    assert {h.id for h in u2_hits} == {"u2"}
    none_hits = store.sql_lookup("tags", "secret", scope=None)
    assert {h.id for h in none_hits} == {"u1", "u2"}


# --------------------------------------------------------------------------- #
# MemoryStore.vector_search scope (existing behavior — pinned for completeness)
# --------------------------------------------------------------------------- #
def test_vector_search_scope_none_finds_all_tenants():
    """scope=None disables masking — every tenant's card is a candidate."""
    store = MemoryStore(":memory:")
    store.add(_card("u1", "same", embedding=[1.0, 0.0]), scope="user_id=U1")
    store.add(_card("u2", "same", embedding=[1.0, 0.0]), scope="user_id=U2")

    hits = store.vector_search([1.0, 0.0], k=5, scope=None)
    ids = {h.id for h, _ in hits}
    assert ids == {"u1", "u2"}


def test_vector_search_exact_scope_isolates_tenant():
    """vector_search's mask is exact-string: scope='user_id=U1' sees only U1."""
    store = MemoryStore(":memory:")
    store.add(_card("u1", "same", embedding=[1.0, 0.0]), scope="user_id=U1")
    store.add(_card("u2", "same", embedding=[1.0, 0.0]), scope="user_id=U2")

    u1_hits = store.vector_search([1.0, 0.0], k=5, scope="user_id=U1")
    assert {h.id for h, _ in u1_hits} == {"u1"}
    u2_hits = store.vector_search([1.0, 0.0], k=5, scope="user_id=U2")
    assert {h.id for h, _ in u2_hits} == {"u2"}


def test_vector_search_default_scope_sees_default_cards():
    """The default scope string surfaces default-scope (and only default) cards."""
    store = MemoryStore(":memory:")
    store.add(_card("g", "same", embedding=[1.0, 0.0]), scope="default")
    store.add(_card("u1", "same", embedding=[1.0, 0.0]), scope="user_id=U1")

    hits = store.vector_search([1.0, 0.0], k=5, scope="default")
    assert {h.id for h, _ in hits} == {"g"}
