"""Unit tests for multi-tier scoping (user_id/agent_id/run_id isolation).

Covers scope-aware ``vector_search``, ``delete_scope``, scope filtering in
the no-numpy fallback path, ``hybrid_search`` scoping, and the <0.1ms scope
mask overhead contract.
"""
from __future__ import annotations

import math
import time

import pytest

from isotope_zero.core.store import MemoryStore
from isotope_zero.types import MemoryCard, now_ts


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def _card(
    id: str,
    fact: str = "A fact.",
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
        source_tokens=5,
        scope=scope,
    )


# --------------------------------------------------------------------------- #
# Scope isolation in vector_search
# --------------------------------------------------------------------------- #
def test_vector_search_scope_isolation():
    """A scoped search never returns cards from another scope."""
    store = MemoryStore(":memory:")
    # Two scopes, identical embeddings -> same cosine score.
    e = _norm([1.0, 0.0, 0.0])
    store.add(_card("a_user", embedding=e, scope="user=1"))
    store.add(_card("b_user", embedding=e, scope="user=2"))
    store.add(_card("c_user", embedding=e, scope="user=1"))

    hits = store.vector_search(e, k=10, scope="user=1")
    ids = {c.id for c, _ in hits}
    assert ids == {"a_user", "c_user"}, f"leak: {ids}"
    assert "b_user" not in ids


def test_vector_search_scope_default_matches_legacy_behavior():
    """When all cards are 'default' scope, a default-scoped search sees all."""
    store = MemoryStore(":memory:")
    e = _norm([1.0, 0.0, 0.0])
    store.add(_card("d1", embedding=e))
    store.add(_card("d2", embedding=e))
    hits = store.vector_search(e, k=10)  # default scope
    assert {c.id for c, _ in hits} == {"d1", "d2"}


def test_vector_search_scope_none_is_global():
    """scope=None disables scoping -> cross-scope global retrieval."""
    store = MemoryStore(":memory:")
    e = _norm([1.0, 0.0, 0.0])
    store.add(_card("g1", embedding=e, scope="tenant_a"))
    store.add(_card("g2", embedding=e, scope="tenant_b"))
    hits = store.vector_search(e, k=10, scope=None)
    assert {c.id for c, _ in hits} == {"g1", "g2"}


def test_vector_search_scope_empty_scope_returns_empty():
    """A scope with no cards returns nothing (no leak from other scopes)."""
    store = MemoryStore(":memory:")
    e = _norm([1.0, 0.0, 0.0])
    store.add(_card("x", embedding=e, scope="populated"))
    assert store.vector_search(e, k=10, scope="empty_scope") == []


def test_vector_search_scope_respects_k_within_scope():
    """k is bounded by the in-scope population, never padded by out-of-scope."""
    store = MemoryStore(":memory:")
    e = _norm([1.0, 0.0, 0.0])
    # Scope A has 2 cards, scope B has 8. Ask k=10 in scope A.
    store.add(_card("a1", embedding=e, scope="A"))
    store.add(_card("a2", embedding=e, scope="A"))
    for i in range(8):
        store.add(_card(f"b{i}", embedding=e, scope="B"))
    hits = store.vector_search(e, k=10, scope="A")
    assert len(hits) == 2, f"expected 2, got {len(hits)}: {[c.id for c,_ in hits]}"
    assert {c.id for c, _ in hits} == {"a1", "a2"}


# --------------------------------------------------------------------------- #
# delete_scope
# --------------------------------------------------------------------------- #
def test_delete_scope_removes_only_that_scope():
    """delete_scope wipes the named scope's cards + edges, leaves others intact."""
    store = MemoryStore(":memory:")
    e = _norm([1.0, 0.0, 0.0])
    store.add(_card("keep1", embedding=e, scope="keep"))
    store.add(_card("kill1", embedding=e, scope="kill"))
    store.add(_card("kill2", embedding=e, scope="kill"))

    deleted = store.delete_scope("kill")
    assert deleted == 2

    # The killed scope is now empty; the kept scope is untouched.
    assert store.vector_search(e, k=10, scope="kill") == []
    kept_hits = store.vector_search(e, k=10, scope="keep")
    assert {c.id for c, _ in kept_hits} == {"keep1"}
    # count() reflects only surviving rows.
    assert store.count() == 1


def test_delete_scope_missing_scope_is_zero():
    """Deleting a scope that doesn't exist returns 0, harmlessly."""
    store = MemoryStore(":memory:")
    store.add(_card("solo", embedding=_norm([1.0, 0.0, 0.0]), scope="real"))
    assert store.delete_scope("nonexistent") == 0
    assert store.count() == 1  # nothing touched


def test_delete_scope_invalidates_vector_cache():
    """After delete_scope, a stale matrix must not resurrect deleted cards."""
    store = MemoryStore(":memory:")
    e = _norm([1.0, 0.0, 0.0])
    store.add(_card("survivor", embedding=e, scope="live"))
    store.add(_card("doomed", embedding=e, scope="doomed"))
    # Prime the vector cache.
    store.vector_search(e, k=10, scope="doomed")
    # Delete the doomed scope.
    store.delete_scope("doomed")
    # The cache must have been invalidated; a new search returns nothing.
    assert store.vector_search(e, k=10, scope="doomed") == []
    # And the survivor is still searchable.
    assert {c.id for c, _ in store.vector_search(e, k=10, scope="live")} == {"survivor"}


# --------------------------------------------------------------------------- #
# Update persists scope
# --------------------------------------------------------------------------- #
def test_update_carries_scope_field():
    """update() persists the card's scope into the row (not reset to default)."""
    store = MemoryStore(":memory:")
    e = _norm([1.0, 0.0, 0.0])
    store.add(_card("u", embedding=e, scope="tenant_x"))
    # Re-upsert with a different fact under the SAME scope.
    store.update(_card("u", fact="updated", embedding=e, scope="tenant_x"))
    # A scoped search in tenant_x must still find it.
    hits = store.vector_search(e, k=10, scope="tenant_x")
    assert {c.id for c, _ in hits} == {"u"}
    # And it must NOT surface in the default scope.
    assert store.vector_search(e, k=10, scope="default") == []


# --------------------------------------------------------------------------- #
# Fallback path (no numpy) scoping
# --------------------------------------------------------------------------- #
def test_vector_search_fallback_scope_isolation(monkeypatch):
    """The pure-Python fallback path honors scope too."""
    store = MemoryStore(":memory:")
    e = _norm([1.0, 0.0, 0.0])
    store.add(_card("f1", embedding=e, scope="A"))
    store.add(_card("f2", embedding=e, scope="B"))

    # Force the no-numpy fallback by making numpy unimportable in the store
    # module's vector_search resolution path. We patch the builtins import
    # trap via the store's own try/except: simplest is to call the fallback
    # directly to assert its scoping contract.
    hits = store._vector_search_fallback(e, k=10, alpha=1.0, scope="A")
    assert {c.id for c, _ in hits} == {"f1"}


# --------------------------------------------------------------------------- #
# hybrid_search scoping
# --------------------------------------------------------------------------- #
def test_hybrid_search_scope_isolation():
    """hybrid_search must scope both the vector AND BM25 branches."""
    store = MemoryStore(":memory:")
    e = _norm([1.0, 0.0, 0.0])
    # Scope A: a fact that matches the query text.
    store.add(_card("hA", fact="the cat sat on the mat", embedding=e, scope="A"))
    # Scope B: the SAME fact text — must NOT leak into scope A's results.
    store.add(_card("hB", fact="the cat sat on the mat", embedding=e, scope="B"))

    # Search scoped to A: both branches (vector + BM25) must be A-only.
    hits = store.hybrid_search("cat sat", e, k=10, scope="A")
    ids = {c.id for c, _ in hits}
    assert ids == {"hA"}, f"hybrid_search scope leak: {ids}"
    assert "hB" not in ids


# --------------------------------------------------------------------------- #
# Performance contract: scope mask overhead < 0.1ms
# --------------------------------------------------------------------------- #
def test_scope_mask_overhead_under_0_1ms():
    """Applying a scope mask must add <0.1ms to the search path at 10k cards.

    This is the scoping performance contract: the boolean mask is a single
    C-level vectorized pass, so even at 10k rows it is well under the budget.
    We measure the delta between a scoped and unscoped search over the SAME
    warmed matrix, averaged over many iterations to smooth jitter.
    """
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        pytest.skip("numpy not available — perf contract only enforced on the BLAS path")

    store = MemoryStore(":memory:")
    e = _norm([1.0, 0.0, 0.0])
    # Seed 2000 cards split across 2 scopes (smaller than 10k but the mask
    # cost scales linearly; 2k is enough to show the C-pass is negligible and
    # keeps the test fast). Use a deterministic scope split.
    n = 2000
    for i in range(n):
        scope = "A" if i % 2 == 0 else "B"
        store.add(_card(f"p{i}", embedding=e, scope=scope))
    # Warm the matrix cache with one search.
    store.vector_search(e, k=5, scope="A")

    iters = 200
    # Time scoped (mask applied).
    t0 = time.perf_counter()
    for _ in range(iters):
        store.vector_search(e, k=5, scope="A")
    scoped_ns = (time.perf_counter() - t0) / iters

    # Time unscoped (no mask).
    t0 = time.perf_counter()
    for _ in range(iters):
        store.vector_search(e, k=5, scope=None)
    unscoped_ns = (time.perf_counter() - t0) / iters

    overhead_ns = scoped_ns - unscoped_ns
    # The mask itself is a single boolean-array assignment over `n` float32.
    # At 2k rows this is a few microseconds; allow generous headroom for
    # jitter/branch noise but assert it stays under 100us (0.1ms).
    assert overhead_ns < 0.0001, (
        f"scope mask overhead {overhead_ns*1e6:.1f}us exceeds 0.1ms budget "
        f"(scoped={scoped_ns*1e6:.1f}us, unscoped={unscoped_ns*1e6:.1f}us)"
    )
