"""Tests for the TTL (time-to-live) port — isotope_zero's mem0 TTL adaptation.

Covers the time-bounded-memory surface added alongside the content-aware dedup
port. The contract (see ``store.MemoryCard`` + ``store.MemoryStore``):

- A card with ``ttl_seconds`` set gets ``expiration_timestamp = timestamp +
  ttl_seconds`` computed at ``add()`` time. Retrieval paths (``get``/``all``/
  ``count``/``vector_search``) EXCLUDE a card once ``now > expiration_timestamp``
  via a ``WHERE (expiration_timestamp IS NULL OR expiration_timestamp >
  unixepoch())`` clause shared across every read SELECT.

- A card with ``ttl_seconds=None`` (the default) gets a NULL
  ``expiration_timestamp`` and NEVER expires — the pre-TTL behaviour, preserving
  backward compatibility for every caller that doesn't pass it.

- ``get(id)`` is the ONE exception: it retrieves an expired card by id anyway,
  because it is the audit-trail accessor (a supervisor re-reading a folded
  memory should still find it). Only the RETRIEVAL paths enforce expiry.

- ``purge_expired()`` hard-deletes rows whose TTL has elapsed (``expiration <=
  unixepoch()``), fires the FTS5 ``aad`` trigger per row so full-text stays in
  sync, and invalidates the vector cache (``_mark_vec_dirty``) so the next
  search rebuilds without the purged rows. Idempotent: returns 0 when nothing
  is expired.

The expiry threshold is wall-clock ``now``. The store's read-path SQL uses
``expiration_timestamp <= unixepoch()``, and ``unixepoch()`` returns **integer**
seconds — so a card whose ``expiration_timestamp = 1785893487.506 + 1 = .506``
is NOT yet expired at ``unixepoch() = 1785893488`` (the trailing ``.506`` keeps
it ahead of the integer-second boundary). We therefore use a 1s TTL but sleep
1.6s (past the next integer boundary, not just past the nominal 1s), which is
the smallest sleep that deterministically forges expiry on any wall clock. The
fuzzing suite's ``hypothesis`` dependence is NOT required here; these are plain
``test_*`` functions against a real ``:memory:`` store.
"""
from __future__ import annotations

import time

import pytest

from isotope_zero.core.store import MemoryStore
from isotope_zero.types import MemoryCard, now_ts

# The store's read-path SQL filters expired rows with
# ``expiration_timestamp <= unixepoch()``. ``unixepoch()`` returns INTEGER
# seconds, so a card stamped at ``t.506`` with a 1s TTL expires at
# ``(t+1).506`` — which stays AHEAD of the integer boundary ``t+1`` and is
# therefore NOT yet expired there. To forge expiry deterministically without a
# fragile per-test wall-clock we use a 1s TTL + a 1.6s sleep (clears the next
# integer boundary on any clock) — the minimum robust pair.
_TTL_S = 1


def _expire_it(store: MemoryStore | None = None) -> None:
    """Sleep past the next ``unixepoch()`` integer boundary so a 1s-TTL card
    is deterministically expired.

    ``unixepoch()`` returns INTEGER seconds, so a card stamped at ``t.506``
    with a 1s TTL expires at ``(t+1).506`` — which stays AHEAD of the integer
    boundary ``t+1`` until the clock reaches ``t+2``. When a ``store`` is
    passed we poll its OWN ``unixepoch()`` and only return once we've observed
    it advance a full second past the call time (guaranteeing any 1s-TTL card
    stamped before this call is now expired). Without a store we sleep a fixed
    2s — enough on any clock for a 1s-TTL card stamped within the last second.
    """
    if store is None:
        time.sleep(2.0)
        return
    # Poll the store's own SQLite clock until it advances a full second past
    # the moment we entered. A 1s-TTL card stamped any time before this call
    # is then guaranteed expired (its expiration is at most now+1, and the
    # clock is now >= now+2 > expiration).
    start = store._conn.execute("select unixepoch()").fetchone()[0]
    while True:
        now = store._conn.execute("select unixepoch()").fetchone()[0]
        if now - start >= 2:
            return
        time.sleep(0.2)


# --------------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------------- #
def _card(
    id: str,
    fact: str = "A fact.",
    ttl_seconds: int | None = None,
    embedding: list[float] | None = None,
    scope: str = "default",
) -> MemoryCard:
    return MemoryCard(
        id=id,
        fact=fact,
        evidence="evidence",
        timestamp=now_ts(),
        tags=[],
        embedding=embedding,
        source_tokens=4,
        scope=scope,
        ttl_seconds=ttl_seconds,
    )


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


# --------------------------------------------------------------------------- #
# 1. NULL-TTL survival: a card with no TTL never expires and is always visible.
# --------------------------------------------------------------------------- #
def test_null_ttl_card_survives_in_all_and_count(store: MemoryStore):
    """A card with ``ttl_seconds=None`` (the default) NEVER expires — the
    pre-TTL behaviour every existing caller relies on."""
    store.add(_card("immortal", fact="never expires", ttl_seconds=None))
    assert store.count() == 1
    assert [c.id for c in store.all()] == ["immortal"]


def test_null_ttl_card_not_purged_by_purge_expired(store: MemoryStore):
    """``purge_expired()`` must NEVER touch a NULL-TTL row (backward compat)."""
    store.add(_card("immortal", fact="never expires", ttl_seconds=None))
    deleted = store.purge_expired()
    assert deleted == 0, f"expected 0 purged, got {deleted}"
    assert store.count() == 1, "NULL-TTL card must survive purge_expired()"


# --------------------------------------------------------------------------- #
# 2. Expiry exclusion: an expired card is hidden from all retrieval paths.
# --------------------------------------------------------------------------- #
def test_expired_card_excluded_from_all(store: MemoryStore):
    """``all()`` excludes a card once ``now > expiration_timestamp``."""
    store.add(_card("ephemeral", fact="expires soon", ttl_seconds=_TTL_S))
    _expire_it(store)  # forge expiry deterministically
    assert [c.id for c in store.all()] == [], "expired card must NOT surface in all()"


def test_expired_card_excluded_from_count(store: MemoryStore):
    """``count()`` excludes expired cards (live-card count)."""
    store.add(_card("ephemeral", fact="expires soon", ttl_seconds=_TTL_S))
    assert store.count() == 1  # still live immediately after add
    _expire_it(store)
    assert store.count() == 0, "expired card must NOT count as live"


def test_expired_card_excluded_from_vector_search(store: MemoryStore):
    """``vector_search`` must not return an expired card, even on an exact
    embedding match."""
    dim = 8
    needle = [1.0] + [0.0] * (dim - 1)
    store.add(_card("ephemeral", fact="expires soon", ttl_seconds=_TTL_S, embedding=needle))
    _expire_it(store)  # expire it
    hits = store.vector_search(needle, k=5)
    assert hits == [], f"expired card must NOT surface in vector_search, got {hits}"


# --------------------------------------------------------------------------- #
# 3. Audit-trail access: get(id) retrieves an expired card anyway.
# --------------------------------------------------------------------------- #
def test_get_retrieves_expired_card_for_audit_trail(store: MemoryStore):
    """``get(id)`` is the audit-trail accessor: it retrieves an EXPIRED card
    by id even though the retrieval paths hide it. A supervisor re-reading a
    folded memory should still find it."""
    store.add(_card("ephemeral", fact="expires soon", ttl_seconds=_TTL_S))
    _expire_it(store)
    # The retrieval paths all hide it now...
    assert store.count() == 0
    assert [c.id for c in store.all()] == []
    # ...but get(id) still returns it (audit trail).
    c = store.get("ephemeral")
    assert c is not None, "get() must retrieve an expired card for the audit trail"
    assert c.id == "ephemeral"
    assert c.ttl_seconds == 1
    assert c.expiration_timestamp is not None


# --------------------------------------------------------------------------- #
# 4. purge_expired(): hard-deletes elapsed-TTL rows.
# --------------------------------------------------------------------------- #
def test_purge_expired_hard_deletes_expired_rows(store: MemoryStore):
    """``purge_expired()`` hard-DELETEs rows whose TTL elapsed, so even the
    audit-trail ``get(id)`` can no longer find them."""
    store.add(_card("ephemeral", fact="expires soon", ttl_seconds=_TTL_S))
    _expire_it(store)
    deleted = store.purge_expired()
    assert deleted == 1, f"expected 1 purged, got {deleted}"
    # After hard-delete, even get() can't find it.
    assert store.get("ephemeral") is None
    assert store.count() == 0


def test_purge_expired_only_touchs_expired_not_live(store: MemoryStore):
    """``purge_expired()`` deletes ONLY expired rows; live (NULL or
    unexpired) cards survive untouched."""
    store.add(_card("immortal", fact="never", ttl_seconds=None))
    store.add(_card("live", fact="not yet", ttl_seconds=3600))  # 1h, still live
    store.add(_card("dead", fact="expired", ttl_seconds=_TTL_S))
    _expire_it(store)  # only "dead" is expired
    deleted = store.purge_expired()
    assert deleted == 1
    assert store.count() == 2  # immortal + live survive
    assert {c.id for c in store.all()} == {"immortal", "live"}
    assert store.get("dead") is None


# --------------------------------------------------------------------------- #
# 5. Idempotency: purge_expired() is a no-op (returns 0) when nothing's expired.
# --------------------------------------------------------------------------- #
def test_purge_expired_is_idempotent_noop(store: MemoryStore):
    """``purge_expired()`` is idempotent: a second call right after a purge
    that deleted nothing (or already ran) returns 0 and changes nothing."""
    store.add(_card("immortal", fact="never", ttl_seconds=None))
    assert store.purge_expired() == 0  # nothing expired
    assert store.purge_expired() == 0  # idempotent re-run
    assert store.count() == 1


def test_purge_expired_double_call_returns_zero_second_time(store: MemoryStore):
    """After a purge deletes the expired set, a second purge returns 0
    (they're already gone — no double-count, no error)."""
    store.add(_card("ephemeral", fact="expires soon", ttl_seconds=_TTL_S))
    _expire_it(store)
    assert store.purge_expired() == 1  # first call: deletes it
    assert store.purge_expired() == 0  # second call: nothing left to delete


# --------------------------------------------------------------------------- #
# 6. Vector cache invalidation on purge: a purged card does not re-surface.
# --------------------------------------------------------------------------- #
def test_vector_cache_invalidated_on_purge(store: MemoryStore):
    """After ``purge_expired()`` hard-deletes a row, the vector cache is
    invalidated so a subsequent search rebuilds WITHOUT the purged card
    (the cached matrix must not keep a stale entry pointing at a deleted row).

    We force the cache to materialise BEFORE the purge (a search primes it),
    then purge, then search again — the second search must come back empty."""
    dim = 8
    needle = [1.0] + [0.0] * (dim - 1)
    store.add(_card("ephemeral", fact="expires soon", ttl_seconds=_TTL_S, embedding=needle))
    # Prime the vector cache while the card is still live (but not yet expired).
    hits = store.vector_search(needle, k=5)
    assert len(hits) == 1, f"pre-purge search should find the live card, got {hits}"
    _expire_it(store)  # now expired
    # The cache still holds the (now-expired) row. Purge must invalidate it.
    deleted = store.purge_expired()
    assert deleted == 1
    # Cache was invalidated by purge; this search rebuilds without the purged row.
    # An expired-and-purged card must NOT surface even via the (rebuilt) cache.
    hits2 = store.vector_search(needle, k=5)
    assert hits2 == [], (
        f"purged card must not re-surface after cache invalidation, got {hits2}"
    )
