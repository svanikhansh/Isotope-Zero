"""Tests for ``isotope_zero.retrieval.hybrid_search.HybridSearcher``.

Ports mem0's multi-signal RRF retrieval (``mem0/memory/main.py:1369`` search
fusing vector + BM25 + entity boost at ``main.py:1763``) onto isotope_zero's
existing store/FTS5/graph/decay primitives. These tests pin:

  * ``search`` returns k ``(id, score)`` tuples for a text-only query.
  * RRF math: a card strong in BOTH branches outranks one strong in one.
  * Entity-graph boost: an entity-linked card ranks above a pure-vector tie.
  * Ebbinghaus decay: a fresh card outranks a stale twin all else equal.
  * Graceful degradation: vector-only and BM25-only legs both work; a missing
    query signal raises ``ValueError``.

Conventions match the existing suite: ``:memory:`` store, plain ``test_*``
functions, explicit unit vectors so cosine is a precise 1.0/0.0 signal.
"""
from __future__ import annotations

import math
import time

import pytest

from isotope_zero.core.store import MemoryStore
from isotope_zero.graph.relation_graph import RelationGraph
from isotope_zero.retrieval.hybrid_search import HybridSearcher
from isotope_zero.types import MemoryCard, now_ts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _unit_vec(i: int, dim: int = 16) -> list[float]:
    """One-hot unit vector at position i%dim — exact, L2-normalized by
    construction. Vector_search cosine is then a precise 1.0/0.0 signal."""
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def _blend_query(slots: list[int], dim: int = 16) -> list[float]:
    """L2-normalized even blend of the given one-hot slots.

    Two cards on different slots are ORTHOGONAL to each other (cosine 0.0),
    so the store's auto-linker (``core/graph.py:121`` ``auto_link_cards``,
    threshold 0.75) will NEVER link them — but a query blending their slots
    is equally similar to both (a perfect vector tie). Used by the boost and
    decay tests where two "twin" cards must tie on the query vector yet stay
    graph-disconnected so a manual edge to only one of them discriminates.
    """
    v = [0.0] * dim
    w = 1.0 / math.sqrt(len(slots))
    for s in slots:
        v[s % dim] += w
    return v


def _card(
    id: str,
    fact: str,
    *,
    embedding: list[float] | None = None,
    timestamp: float | None = None,
    last_access: float = 0.0,
    stability: float = 1.0,
    scope: str = "default",
    tags: list[str] | None = None,
) -> MemoryCard:
    # ``tags`` default to ``[id]`` (a per-card unique tag) so two cards that
    # share an identical ``fact`` text (used by the boost/decay tests to forge
    # a true BM25 tie) do NOT collide in the content-aware dedup fingerprint
    # (``core/dedup.py`` hashes ``fact + "||" + sorted(tags)``). Distinct tags
    # -> distinct fingerprint -> both rows persist -> the tie the test forges
    # is over the LIVE rows, not a dedup-touched single row. Callers that pass
    # explicit ``tags`` override this default.
    if tags is None:
        tags = [id]
    return MemoryCard(
        id=id,
        fact=fact,
        evidence="e",
        timestamp=timestamp if timestamp is not None else now_ts(),
        tags=tags,
        embedding=embedding,
        source_tokens=5,
        last_access=last_access,
        stability=stability,
        scope=scope,
    )


class _StubEmbedder:
    """Deterministic embedder returning a list[float] unit vector keyed by the
    first alphabetic token. Used so the ``query_text``-only path (which calls
    ``store.embedder.embed_text``) is exercised without an ONNX dependency.

    Mirrors the contract ``store.embedder`` documents (``core/store.py:311``):
    ``.embed_text(text) -> list[float]``, L2-normalized. Empty/degenerate input
    returns a zero vector so the searcher skips the vector leg.
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def embed_text(self, text: str) -> list[float]:
        tokens = [t for t in text.lower().split() if t.isalpha()]
        if not tokens:
            return [0.0] * self._dim
        # Stable mapping: token -> (sum of ord codes) % dim. Deterministic and
        # spread across the dim slots so different tokens get different vectors.
        slot = sum(ord(c) for c in tokens[0]) % self._dim
        return _unit_vec(slot, self._dim)


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


@pytest.fixture
def stub_embedder() -> _StubEmbedder:
    return _StubEmbedder()


# --------------------------------------------------------------------------- #
# 1. search returns k (id, score) tuples for a text query
# --------------------------------------------------------------------------- #
class TestSearchReturnsK:
    def test_text_only_returns_k_tuples(self, store, stub_embedder):
        store.embedder = stub_embedder
        # 20 cards across 16 vector slots; facts share query tokens with c1.
        for i in range(20):
            store.add(
                _card(
                    f"c{i}",
                    fact=f"memory number {i} about python scripting" if i < 5
                    else f"unrelated note {i} on rust systems",
                    embedding=_unit_vec(i % 16),
                )
            )
        searcher = HybridSearcher(store)
        results = searcher.search(query_text="python", k=5)
        assert len(results) == 5
        assert all(isinstance(t, tuple) and len(t) == 2 for t in results)
        assert all(isinstance(t[0], str) and isinstance(t[1], float) for t in results)
        # Scores are desc-ordered (RRF fused score).
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_vec_only_returns_k_tuples(self, store):
        for i in range(20):
            store.add(_card(f"c{i}", fact=f"note {i}", embedding=_unit_vec(i % 16)))
        searcher = HybridSearcher(store)
        results = searcher.search(query_vec=_unit_vec(0), k=5)
        assert len(results) == 5
        # The card whose embedding matches the query vector should rank first.
        # _unit_vec(0) matches c0 and c16 (both slot 0); either is correct.
        assert results[0][0] in {"c0", "c16"}

    def test_both_branches_returns_k(self, store, stub_embedder):
        store.embedder = stub_embedder
        for i in range(20):
            store.add(
                _card(
                    f"c{i}",
                    fact="python python python" if i == 0 else f"note {i}",
                    embedding=_unit_vec(i % 16),
                )
            )
        searcher = HybridSearcher(store)
        results = searcher.search(
            query_vec=_unit_vec(0), query_text="python", k=5
        )
        assert len(results) == 5
        # c0 is rank-1 in BOTH branches (vector slot 0 + fact matches "python")
        # so it must surface first under RRF (a card strong in both wins).
        assert results[0][0] == "c0"


# --------------------------------------------------------------------------- #
# 2. RRF: a card strong in BOTH branches outranks one strong in one
# --------------------------------------------------------------------------- #
class TestRRFFusion:
    def test_card_in_both_branches_outranks_single_branch_top(self, store, stub_embedder):
        """A card that is rank-1 in vector AND rank-1 in BM25 must outrank a
        card that is rank-1 in only one branch — the whole point of late
        fusion (RRF rewards concurrence)."""
        store.embedder = stub_embedder
        # c0: vector slot 0 + fact matches "python" -> rank 1 in BOTH.
        store.add(_card("c0", fact="python python python", embedding=_unit_vec(0)))
        # c1: vector slot 1 (no overlap with query slot 0) but fact matches
        # "python" -> rank 1 in BM25 only. c0 should still win overall because
        # it gets BOTH rank terms.
        store.add(_card("c1", fact="python rules", embedding=_unit_vec(1)))
        # filler so vector_search has depth
        for i in range(2, 10):
            store.add(_card(f"c{i}", fact=f"noise {i}", embedding=_unit_vec(i % 16)))

        searcher = HybridSearcher(store, alpha=0.5)
        # Query vector hits slot 0 (c0). Query text "python" hits c0 + c1.
        results = searcher.search(query_vec=_unit_vec(0), query_text="python", k=10)
        ids = [cid for cid, _ in results]
        assert ids[0] == "c0", f"expected c0 (both branches) first, got {ids[0]}"

    def test_alpha_weights_vector_branch_more(self, store, stub_embedder):
        """With alpha=1.0 the BM25 branch contributes nothing; the result is
        pure vector ranking (modulo boosts). With alpha=0.0, vector
        contributes nothing and BM25 dominates."""
        store.embedder = stub_embedder
        store.add(_card("c0", fact="python", embedding=_unit_vec(0)))
        store.add(_card("c1", fact="python python", embedding=_unit_vec(1)))
        for i in range(2, 8):
            store.add(_card(f"c{i}", fact=f"noise {i}", embedding=_unit_vec(i % 16)))

        # alpha=1.0: pure vector. c0 (slot 0) ranks first.
        s_vec = HybridSearcher(store, alpha=1.0)
        r_vec = s_vec.search(query_vec=_unit_vec(0), query_text="python", k=5)
        assert r_vec[0][0] == "c0"

        # alpha=0.0: pure BM25. Both c0 and c1 match "python"; c1's fact has
        # "python" twice so it ranks higher under BM25 (term frequency).
        s_bm25 = HybridSearcher(store, alpha=0.0)
        r_bm25 = s_bm25.search(query_vec=_unit_vec(0), query_text="python", k=5)
        ids_bm25 = [cid for cid, _ in r_bm25]
        assert "c1" in ids_bm25 and "c0" in ids_bm25
        assert ids_bm25[0] == "c1", f"BM25 should rank c1 first (tf), got {ids_bm25[0]}"


# --------------------------------------------------------------------------- #
# 3. Entity-graph boost: entity-linked card ranks above a pure-vector tie
# --------------------------------------------------------------------------- #
class TestEntityBoost:
    def test_entity_linked_card_outranks_vector_tie(self, store, stub_embedder):
        """Construct a deterministic vector tie where the two tied cards are
        graph-ISOLATED from each other so a manual edge discriminates them.

        The twins live on DIFFERENT vector slots (0 and 1) — so they are
        ORTHOGONAL to each other (cosine 0.0) and the store's auto-linker
        (``core/graph.py:121`` ``auto_link_cards``, threshold 0.75) will NOT
        create a ``semantic`` edge between them. But a query vector that
        BLENDS slots 0 and 1 (``_blend_query([0, 1])``) is equally similar to
        BOTH twins (cosine = 1/√2 ≈ 0.707) — a perfect vector tie. Neither
        fact mentions the query entity "alpha", so neither is a BM25 hit nor
        a boost seed; only a manual ``related_to`` edge from an
        entity-mentioning seed to ONE twin introduces that twin into the
        boost neighborhood.

        Without the boost the two twins share an identical RRF score (same
        vector rank-share; no BM25 contribution) and would be ordered by id
        asc. The boost (0.1 * retention ≈ 0.1) outweighs the rank-position
        gap (rank 2 vs 1 is ~0.0115 - 0.0111 ≈ 0.0004 of RRF weight), so the
        boost flips the order in favor of the linked twin."""
        store.embedder = stub_embedder
        # Linked twin. Sorts AFTER c_aa by id-asc, so WITHOUT the boost it
        # would rank BELOW c_aa (the deterministic id-asc tie-break). Its slot
        # (0) is orthogonal to c_aa's slot (1) — no auto-link fires.
        store.add(_card("c_zz", fact="bravo charlie", embedding=_unit_vec(0)))
        # Unlinked twin. Sorts FIRST by id-asc; the boost is what overcomes
        # its rank-1 vector position advantage.
        store.add(_card("c_aa", fact="bravo charlie", embedding=_unit_vec(1)))
        # Seed mentions the query entity "alpha"; embedding on slot 2 (orthogonal
        # to both twins) so auto-link never connects it to them either.
        store.add(_card("c_seed", fact="alpha delta echo", embedding=_unit_vec(2)))
        rg = RelationGraph(store._conn)
        # Manual edge to ONLY c_zz -> only c_zz is boost-eligible.
        rg.add_edge("c_seed", "c_zz", relation_type="related_to", weight=1.0)

        searcher = HybridSearcher(store)
        # Blend query ties the two twins on the vector leg; "alpha" hits the
        # seed on the BM25 leg.
        results = searcher.search(
            query_vec=_blend_query([0, 1]), query_text="alpha", k=5
        )
        ids = [cid for cid, _ in results]
        assert "c_zz" in ids and "c_aa" in ids
        # c_zz is graph-linked to c_seed (which mentions "alpha"); it gets
        # the entity boost and must rank above the unlinked twin.
        idx_zz = ids.index("c_zz")
        idx_aa = ids.index("c_aa")
        assert idx_zz < idx_aa, (
            f"entity-linked c_zz ({idx_zz}) should rank above "
            f"unlinked c_aa ({idx_aa})"
        )


# --------------------------------------------------------------------------- #
# 4. Ebbinghaus decay: a fresh card outranks a stale twin all else equal
# --------------------------------------------------------------------------- #
class TestDecayReRank:
    def test_recent_card_outranks_stale_twin(self, store, stub_embedder):
        """Two cards with identical embeddings and facts (a vector+BM25 tie).
        One was just touched (last_access = now); the other is 30 days stale.
        The fresh card's retention (~1.0) weights its entity boost more than
        the stale card's (~exp(-30*24/24) ≈ 0.0). To make the boost
        discriminate, BOTH must be entity-linked (else the boost term is 0 for
        both and decay has nothing to weight)."""
        store.embedder = stub_embedder
        now = now_ts()
        # Seed a graph neighbor that mentions the query entity so both twins
        # are in the entity neighborhood (boost-eligible).
        store.add(_card("c_seed", fact="zulu query", embedding=_unit_vec(5)))
        rg = RelationGraph(store._conn)
        # Fresh twin: last_access = now (retention ~1.0).
        store.add(
            _card(
                "c_fresh",
                fact="zulu twin",
                embedding=_unit_vec(0),
                last_access=now,
                stability=1.0,
            )
        )
        # Stale twin: last_access = 30 days ago (retention ~exp(-720/24) ≈ 0).
        stale_ts = now - (30 * 24 * 3600)
        store.add(
            _card(
                "c_stale",
                fact="zulu twin",
                embedding=_unit_vec(0),
                last_access=stale_ts,
                stability=1.0,
            )
        )
        rg.add_edge("c_seed", "c_fresh", relation_type="related_to", weight=1.0)
        rg.add_edge("c_seed", "c_stale", relation_type="related_to", weight=1.0)

        searcher = HybridSearcher(store)
        results = searcher.search(query_vec=_unit_vec(0), query_text="zulu", k=5)
        ids = [cid for cid, _ in results]
        assert "c_fresh" in ids and "c_stale" in ids
        idx_fresh = ids.index("c_fresh")
        idx_stale = ids.index("c_stale")
        assert idx_fresh < idx_stale, (
            f"fresh c_fresh ({idx_fresh}) should rank above stale "
            f"c_stale ({idx_stale}) under decay-weighted boost"
        )

    def test_fresh_card_unboosted_ties_stale_unboosted(self, store, stub_embedder):
        """When NEITHER card is entity-linked, the boost term is 0 for both,
        so decay has nothing to weight and the two ties remain ordered by id
        (the deterministic tie-break). This guards against the boost being
        silently applied to un-neighborhood cards."""
        store.embedder = stub_embedder
        now = now_ts()
        store.add(
            _card("c_fresh", fact="zulu twin", embedding=_unit_vec(0), last_access=now)
        )
        store.add(
            _card(
                "c_stale",
                fact="zulu twin",
                embedding=_unit_vec(0),
                last_access=now - (30 * 24 * 3600),
            )
        )
        searcher = HybridSearcher(store)
        results = searcher.search(query_vec=_unit_vec(0), query_text="zulu", k=5)
        # No graph edges -> no boost -> RRF scores tie -> id-asc tie-break.
        ids = [cid for cid, _ in results]
        assert ids.index("c_fresh") < ids.index("c_stale")  # id-asc: f < s


# --------------------------------------------------------------------------- #
# 5. Degradation + error handling
# --------------------------------------------------------------------------- #
class TestDegradation:
    def test_raises_when_no_signal(self, store):
        searcher = HybridSearcher(store)
        with pytest.raises(ValueError):
            searcher.search(query_vec=None, query_text=None, k=5)

    def test_raises_when_empty_text_and_no_vec(self, store):
        searcher = HybridSearcher(store)
        with pytest.raises(ValueError):
            searcher.search(query_vec=None, query_text="", k=5)

    def test_bm25_only_when_no_embedder(self, store):
        """store.embedder is None by default; query_text-only must still work
        via the BM25 leg (vector leg skipped because embedding fails)."""
        assert store.embedder is None
        store.add(_card("c0", fact="python python python", embedding=_unit_vec(0)))
        store.add(_card("c1", fact="rust rust rust", embedding=_unit_vec(1)))
        searcher = HybridSearcher(store)
        results = searcher.search(query_text="python", k=5)
        assert len(results) >= 1
        assert results[0][0] == "c0"

    def test_vector_only_when_no_query_text(self, store):
        store.add(_card("c0", fact="python", embedding=_unit_vec(0)))
        store.add(_card("c1", fact="rust", embedding=_unit_vec(1)))
        searcher = HybridSearcher(store)
        results = searcher.search(query_vec=_unit_vec(0), k=5)
        assert len(results) >= 1
        assert results[0][0] == "c0"

    def test_no_match_returns_empty(self, store, stub_embedder):
        store.embedder = stub_embedder
        store.add(_card("c0", fact="python", embedding=_unit_vec(0)))
        searcher = HybridSearcher(store)
        # A query vector that matches no slot and a query text with no matching
        # fact token -> both branches empty -> [].
        results = searcher.search(
            query_vec=[0.0] * 16, query_text="nomatchtoken", k=5
        )
        assert results == []

    def test_scope_isolates_results(self, store, stub_embedder):
        """Cards in a different scope never surface (the BM25 JOIN and the
        vector mask both filter on scope)."""
        store.embedder = stub_embedder
        store.add(
            _card("c_in", fact="python", embedding=_unit_vec(0), scope="user=u1")
        )
        store.add(
            _card("c_out", fact="python", embedding=_unit_vec(0), scope="user=u2")
        )
        searcher = HybridSearcher(store)
        results = searcher.search(
            query_vec=_unit_vec(0), query_text="python", k=5, scope="user=u1"
        )
        ids = [cid for cid, _ in results]
        assert "c_in" in ids
        assert "c_out" not in ids


# --------------------------------------------------------------------------- #
# 6. Rank plumbing (unit-level)
# --------------------------------------------------------------------------- #
class TestRankPlumbing:
    def test_vector_ranks_are_1_indexed_desc(self, store):
        for i in range(5):
            store.add(_card(f"c{i}", fact=f"note {i}", embedding=_unit_vec(i)))
        searcher = HybridSearcher(store)
        ranks = searcher.vector_ranks(_unit_vec(0), k=5)
        assert ranks[0] == ("c0", 1)  # top match is rank 1
        assert all(r[1] >= 1 for r in ranks)
        # ranks ascending (1, 2, 3, ...)
        assert [r[1] for r in ranks] == list(range(1, len(ranks) + 1))

    def test_bm25_search_are_1_indexed_desc(self, store):
        store.add(_card("c0", fact="python python", embedding=_unit_vec(0)))
        store.add(_card("c1", fact="python", embedding=_unit_vec(1)))
        searcher = HybridSearcher(store)
        ranks = searcher.bm25_search("python", k=5)
        assert ranks[0][1] == 1  # top BM25 match is rank 1
        assert all(r[1] >= 1 for r in ranks)
        assert "c0" in {r[0] for r in ranks}
