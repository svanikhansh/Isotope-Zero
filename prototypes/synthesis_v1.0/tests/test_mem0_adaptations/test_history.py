"""Tests for the revision-history tracker (isotope_zero.core.history).

Mem0 port: every UPDATE/DELETE on a memory is recorded as an append-only
audit row (mem0/memory/main.py:2064 for UPDATE, :2096 for DELETE), and a
revert is itself a logged mutation. These tests pin that contract against
isotope_zero's SQLite backbone.
"""
from __future__ import annotations

from isotope_zero.core.history import MemoryHistoryTracker, init_history
from isotope_zero.core.store import MemoryStore
from isotope_zero.types import MemoryCard, now_ts


def _card(id: str, fact: str = "A fact.") -> MemoryCard:
    return MemoryCard(
        id=id,
        fact=fact,
        evidence="evidence",
        timestamp=now_ts(),
        tags=[],
        embedding=None,
        source_tokens=5,
    )


def test_record_two_updates_then_get_history():
    """record() appends one row per call; get_history returns oldest-first."""
    store = MemoryStore(":memory:")
    store.add(_card("c1", fact="v0"))
    tracker = MemoryHistoryTracker(store._conn)

    # Two sequential updates, each recorded.
    h1 = tracker.record("c1", "default", "v0", "v1", "UPDATE")
    store.update(_card("c1", fact="v1"))
    h2 = tracker.record("c1", "default", "v1", "v2", "UPDATE")
    store.update(_card("c1", fact="v2"))

    hist = tracker.get_history("c1")
    assert len(hist) == 2
    # Oldest-first ordering.
    assert hist[0].history_id == h1
    assert hist[0].old_fact == "v0"
    assert hist[0].new_fact == "v1"
    assert hist[0].event_type == "UPDATE"
    assert hist[0].rolled_back == 0
    assert hist[1].history_id == h2
    assert hist[1].old_fact == "v1"
    assert hist[1].new_fact == "v2"


def test_rollback_reverts_fact_and_appends_rollback_row():
    """rollback() writes old_fact back onto the live card and logs the revert."""
    store = MemoryStore(":memory:")
    store.add(_card("c1", fact="v0"))
    tracker = MemoryHistoryTracker(store._conn)

    tracker.record("c1", "default", "v0", "v1", "UPDATE")
    store.update(_card("c1", fact="v1"))
    h2 = tracker.record("c1", "default", "v1", "v2", "UPDATE")
    store.update(_card("c1", fact="v2"))

    # Revert the v1->v2 mutation: fact should return to v1 via store.get.
    ok = tracker.rollback(h2)
    assert ok is True
    got = store.get("c1")
    assert got is not None
    assert got.fact == "v1"

    # The trail now holds 3 rows: 2 UPDATEs + 1 rollback marker.
    hist = tracker.get_history("c1")
    assert len(hist) == 3
    assert hist[-1].event_type == "rollback"
    # The reverted row is now flagged rolled_back.
    rolled = [r for r in hist if r.history_id == h2]
    assert rolled and rolled[0].rolled_back == 1


def test_rollback_unknown_id_returns_false():
    """rollback() on an absent history id is a clean False, no side effects."""
    store = MemoryStore(":memory:")
    store.add(_card("c1", fact="v0"))
    tracker = MemoryHistoryTracker(store._conn)
    assert tracker.rollback("does-not-exist") is False
    # No history row was created.
    assert tracker.get_history("c1") == []


def test_rollback_already_rolled_returns_false():
    """A second rollback() on the same history id is refused (idempotent)."""
    store = MemoryStore(":memory:")
    store.add(_card("c1", fact="v0"))
    tracker = MemoryHistoryTracker(store._conn)

    tracker.record("c1", "default", "v0", "v1", "UPDATE")
    store.update(_card("c1", fact="v1"))
    h = tracker.record("c1", "default", "v1", "v2", "UPDATE")
    store.update(_card("c1", fact="v2"))

    assert tracker.rollback(h) is True
    # Second attempt: same id, already rolled back -> False.
    assert tracker.rollback(h) is False
    # Trail unchanged by the refused second attempt (still 3 rows).
    assert len(tracker.get_history("c1")) == 3


def test_init_history_is_idempotent():
    """Calling init_history twice does not duplicate or error."""
    store = MemoryStore(":memory:")
    # _init_schema already called init_history once during construction.
    init_history(store._conn)  # second time must be a no-op.
    tracker = MemoryHistoryTracker(store._conn)
    tracker.record("c1", "default", "a", "b", "UPDATE")
    assert len(tracker.get_history("c1")) == 1
