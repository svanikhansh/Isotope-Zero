"""Tests for the content-aware dedup port (isotope_zero.core.dedup + store).

Covers the ``enable_dedup`` flag on ``MemoryStore`` and the scope-aware
duplicate detection in ``add()``:

1. **Off-by-default (additive).** With the default ``enable_dedup=False``, two
   cards with identical fact+tags in the same scope BOTH insert — the
   pre-port ``add()`` semantics. This is the contract the scope-isolation
   tests rely on (they deliberately insert same-fact cards across tenants).

2. **Opt-in dedup.** With ``enable_dedup=True``, a second write of the same
   fact+tags in the SAME scope is folded into the existing row (touch, not a
   second INSERT) — mirroring mem0's ``seen_hashes`` skip at
   ``mem0/memory/main.py:1011``.

3. **Scope-aware.** Dedup is scoped: two cards with identical fact+tags but
   DIFFERENT scopes both persist. mem0 builds its ``existing_hashes`` set
   from scope-filtered ``existing_results`` (main.py:993-1010), so a duplicate
   is only a duplicate within the same tenant boundary.

4. **Tag-aware.** Two cards with the same fact but different tags have
   distinct content fingerprints (``fact + "||" + sorted(tags)``) and both
   persist — they are semantically distinct memories.

Conventions match the existing suite: ``:memory:`` store, plain ``test_*``
functions, real ``MemoryStore`` wiring.
"""
from __future__ import annotations

from isotope_zero.core.store import MemoryStore
from isotope_zero.types import MemoryCard, now_ts


def _card(
    id: str,
    fact: str = "User prefers dark mode.",
    tags: list[str] | None = None,
    scope: str = "default",
    embedding: list[float] | None = None,
) -> MemoryCard:
    return MemoryCard(
        id=id,
        fact=fact,
        evidence="evidence",
        timestamp=now_ts(),
        tags=tags or [],
        embedding=embedding,
        source_tokens=5,
        scope=scope,
    )


# --------------------------------------------------------------------------- #
# Off-by-default: enable_dedup=False keeps same-fact/same-scope cards
# --------------------------------------------------------------------------- #
def test_dedup_default_off_keeps_same_fact_same_scope():
    """Default store (enable_dedup=False) inserts both same-fact/same-scope cards."""
    store = MemoryStore(":memory:")
    store.add(_card("a", fact="likes tea", tags=["pref"]))
    store.add(_card("b", fact="likes tea", tags=["pref"]))  # identical content
    assert store.count() == 2, f"expected 2 rows, got {store.count()}"
    assert {c.id for c in store.all()} == {"a", "b"}


def test_dedup_default_off_no_fingerprint_stored():
    """With dedup off, content_fingerprint is never computed/stored (NULL)."""
    store = MemoryStore(":memory:")
    store.add(_card("a", fact="likes tea", tags=["pref"]))
    c = store.get("a")
    assert c.content_fingerprint is None, (
        f"expected None fingerprint with dedup off, got {c.content_fingerprint!r}"
    )


# --------------------------------------------------------------------------- #
# Opt-in: enable_dedup=True drops same-scope/same-fact twins
# --------------------------------------------------------------------------- #
def test_dedup_on_drops_same_fact_same_scope():
    """enable_dedup=True folds a same-scope/same-fact twin into the existing row."""
    store = MemoryStore(":memory:", enable_dedup=True)
    store.add(_card("a", fact="likes tea", tags=["pref"]))
    store.add(_card("b", fact="likes tea", tags=["pref"]))  # dup -> touch a, skip
    assert store.count() == 1, f"expected 1 row (twin folded), got {store.count()}"
    assert store.get("a") is not None  # original survives
    assert store.get("b") is None  # twin never inserted


def test_dedup_on_stores_fingerprint():
    """With dedup on, the survivor's content_fingerprint is populated (non-None)."""
    store = MemoryStore(":memory:", enable_dedup=True)
    store.add(_card("a", fact="likes tea", tags=["pref"]))
    c = store.get("a")
    assert c.content_fingerprint is not None, "expected a populated fingerprint"
    assert len(c.content_fingerprint) == 64  # sha256 hex digest length


def test_dedup_on_touch_increments_access_count():
    """A dup write touches the survivor: its access_count bumps (mem0 idiom)."""
    store = MemoryStore(":memory:", enable_dedup=True)
    store.add(_card("a", fact="likes tea", tags=["pref"]))
    assert store.get("a").access_count == 0
    store.add(_card("b", fact="likes tea", tags=["pref"]))  # dup -> touch a
    survivor = store.get("a")
    assert survivor.access_count >= 1, (
        f"expected touched survivor access_count>=1, got {survivor.access_count}"
    )


# --------------------------------------------------------------------------- #
# Scope-aware: same fact, different scope -> both persist (even with dedup on)
# --------------------------------------------------------------------------- #
def test_dedup_on_keeps_same_fact_different_scope():
    """Dedup is scoped: same-fact cards in DIFFERENT scopes both persist.

    This mirrors mem0, which builds its existing_hashes set from
    scope-filtered existing_results (main.py:993-1010). Two users storing the
    same preference are two distinct memories.
    """
    store = MemoryStore(":memory:", enable_dedup=True)
    store.add(_card("u1", fact="likes tea", tags=["pref"], scope="user_id=1"))
    store.add(_card("u2", fact="likes tea", tags=["pref"], scope="user_id=2"))
    assert store.count() == 2, f"expected 2 rows (cross-scope), got {store.count()}"
    assert {c.id for c in store.all()} == {"u1", "u2"}


def test_dedup_on_drops_same_fact_same_scope_explicit():
    """Within ONE explicit scope, a same-fact twin is folded (dedup fires)."""
    store = MemoryStore(":memory:", enable_dedup=True)
    store.add(_card("u1a", fact="likes tea", tags=["pref"], scope="user_id=1"))
    store.add(_card("u1b", fact="likes tea", tags=["pref"], scope="user_id=1"))
    assert store.count() == 1, f"expected 1 row (same-scope twin folded), got {store.count()}"


# --------------------------------------------------------------------------- #
# Tag-aware: same fact, different tags -> both persist (distinct memories)
# --------------------------------------------------------------------------- #
def test_dedup_on_keeps_same_fact_different_tags():
    """Same fact but different tags => distinct fingerprint => both persist.

    content_aware_fingerprint hashes ``fact + "||" + sorted(tags)``, so a
    ``["pref"]``-tagged and a ``["todo"]``-tagged card with the same fact are
    semantically distinct memories (mem0 hashes text alone; we fold tags in).
    """
    store = MemoryStore(":memory:", enable_dedup=True)
    store.add(_card("a", fact="the meeting", tags=["pref"]))
    store.add(_card("b", fact="the meeting", tags=["todo"]))  # different tags
    assert store.count() == 2, f"expected 2 rows (distinct tags), got {store.count()}"
    assert {c.id for c in store.all()} == {"a", "b"}


def test_dedup_on_tag_order_independent():
    """Tag order doesn't matter: ``["b","a"]`` and ``["a","b"]`` are the same memory."""
    store = MemoryStore(":memory:", enable_dedup=True)
    store.add(_card("a", fact="the meeting", tags=["b", "a"]))
    store.add(_card("b", fact="the meeting", tags=["a", "b"]))  # sorted-equal
    assert store.count() == 1, (
        f"expected 1 row (tag-order-independent fingerprint), got {store.count()}"
    )
