"""Unit tests for the SQLite storage backend (isotope_zero.core.store)."""
from __future__ import annotations

import math
import os
import tempfile

import pytest

from isotope_zero.core.store import MemoryStore
from isotope_zero.types import MemoryCard, now_ts


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def _card(
    id: str,
    fact: str = "A fact.",
    evidence: str = "evidence",
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
    source_tokens: int = 5,
) -> MemoryCard:
    return MemoryCard(
        id=id,
        fact=fact,
        evidence=evidence,
        timestamp=now_ts(),
        tags=tags or [],
        embedding=embedding,
        source_tokens=source_tokens,
    )


def test_add_and_get_roundtrip():
    store = MemoryStore(":memory:")
    c = _card("c1", fact="The user's name is Alice.", embedding=_norm([1.0, 0.0, 0.0]))
    store.add(c)
    got = store.get("c1")
    assert got is not None
    assert got.id == "c1"
    assert got.fact == "The user's name is Alice."
    # Embedding round-trips through float32 blob.
    assert got.embedding is not None
    assert len(got.embedding) == 3
    assert got.embedding[0] == pytest.approx(1.0, abs=1e-5)


def test_get_missing_returns_none():
    store = MemoryStore(":memory:")
    assert store.get("does-not-exist") is None


def test_add_duplicate_id_raises():
    store = MemoryStore(":memory:")
    store.add(_card("dup", fact="first"))
    with pytest.raises(Exception):
        store.add(_card("dup", fact="second"))


def test_update_upserts_existing():
    store = MemoryStore(":memory:")
    store.add(_card("u1", fact="first fact", tags=["t1"]))
    store.update(_card("u1", fact="second fact", tags=["t2"]))
    got = store.get("u1")
    assert got is not None
    assert got.fact == "second fact"
    assert got.tags == ["t2"]


def test_update_inserts_if_absent():
    store = MemoryStore(":memory:")
    store.update(_card("new", fact="inserted via update"))
    assert store.count() == 1
    assert store.get("new").fact == "inserted via update"


def test_delete_returns_true_then_false():
    store = MemoryStore(":memory:")
    store.add(_card("d1"))
    assert store.delete("d1") is True
    assert store.delete("d1") is False  # already gone
    assert store.count() == 0


def test_all_orders_by_timestamp():
    store = MemoryStore(":memory:")
    store.add(MemoryCard(id="a", fact="A", evidence="", timestamp=100.0, tags=[], source_tokens=1))
    store.add(MemoryCard(id="b", fact="B", evidence="", timestamp=50.0, tags=[], source_tokens=1))
    store.add(MemoryCard(id="c", fact="C", evidence="", timestamp=200.0, tags=[], source_tokens=1))
    ordered = [c.id for c in store.all()]
    assert ordered == ["b", "a", "c"]  # timestamp ascending


def test_sql_lookup_fact_case_insensitive_substring():
    store = MemoryStore(":memory:")
    store.add(_card("1", fact="The user prefers Rust."))
    store.add(_card("2", fact="The project is Mercury."))
    hits = store.sql_lookup("fact", "rust")
    assert len(hits) == 1
    assert hits[0].id == "1"
    # Case-insensitive.
    assert len(store.sql_lookup("fact", "MERCURY")) == 1


def test_sql_lookup_tags_membership():
    store = MemoryStore(":memory:")
    store.add(_card("1", fact="A", tags=["preference", "ui"]))
    store.add(_card("2", fact="B", tags=["project"]))
    hits = store.sql_lookup("tags", "ui")
    assert len(hits) == 1
    assert hits[0].id == "1"


def test_sql_lookup_evidence():
    store = MemoryStore(":memory:")
    store.add(_card("1", fact="x", evidence="user said: 'I love tea'"))
    assert len(store.sql_lookup("evidence", "tea")) == 1


def test_sql_lookup_invalid_field_raises():
    store = MemoryStore(":memory:")
    with pytest.raises(ValueError):
        store.sql_lookup("bogus", "x")


def test_vector_search_topk_and_skip_null():
    store = MemoryStore(":memory:")
    store.add(_card("1", fact="aligns with query", embedding=_norm([1.0, 0.0, 0.0, 0.0])))
    store.add(_card("2", fact="orthogonal", embedding=_norm([0.0, 1.0, 0.0, 0.0])))
    store.add(_card("3", fact="no embedding", embedding=None))  # must be skipped
    hits = store.vector_search(_norm([1.0, 0.0, 0.0, 0.0]), k=5)
    assert len(hits) == 2  # null-embedding card skipped
    assert hits[0][0].id == "1"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    # Top-k bound respected.
    assert len(store.vector_search(_norm([1.0, 0.0, 0.0, 0.0]), k=1)) == 1


def test_vector_search_degenerate_query_returns_empty():
    store = MemoryStore(":memory:")
    store.add(_card("1", embedding=_norm([1.0, 0.0])))
    assert store.vector_search([], k=5) == []
    assert store.vector_search([0.0, 0.0], k=5) == []


def test_count_and_db_size():
    store = MemoryStore(":memory:")
    assert store.count() == 0
    store.add(_card("1"))
    store.add(_card("2"))
    assert store.count() == 2
    assert store.db_size_bytes() == 0  # in-memory


def test_file_backed_persistence_and_size():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.sqlite")
        s1 = MemoryStore(path)
        s1.add(_card("persist", fact="persisted fact"))
        size_after = s1.db_size_bytes()
        assert size_after > 0
        s1.close()
        # Reopen — data survives.
        s2 = MemoryStore(path)
        got = s2.get("persist")
        assert got is not None
        assert got.fact == "persisted fact"
        s2.close()


def test_embedding_none_roundtrips_none():
    store = MemoryStore(":memory:")
    store.add(_card("n1", embedding=None))
    got = store.get("n1")
    assert got.embedding is None
