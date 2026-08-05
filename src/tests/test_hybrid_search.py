"""Tests for Late Fusion hybrid search: FTS5 inverted index + RRF + entity boost.

Covers:
  * FTS5 sync on add/update/delete AND direct-to-connection writes (the path
    the eval harness's bulk seeder takes, bypassing the Python methods).
  * `_rrf_fusion` math: rank-only fusion, a card strong in BOTH branches wins,
    alpha weighting, k-truncation, the `60` smoothing denominator.
  * Entity-graph boost: the Mem0 decay formula 0.5/(1+0.001*(N-1)^2), and that
    a card linked to query-entity-matching cards surfaces even with no lexical
    or vector match of its own.
  * End-to-end `hybrid_search` ranking vs pure `vector_search` recall.
  * p99 latency < 5.0 ms at 10,000 cards (the perf target).
"""
from __future__ import annotations

import math
import statistics
import time

import pytest

from isotope_zero.graph import relation_graph as graph
from isotope_zero.core.store import (
    MemoryStore,
    _extract_entities,
    _fts5_escape,
    _fts5_query,
    _rrf_fusion,
)
from isotope_zero.types import MemoryCard, now_ts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _unit_vec(i: int, dim: int = 16) -> list[float]:
    """One-hot unit vector at position i%dim — exact, L2-normalized by construction.
    Used so vector_search cosine is a precise 1.0/0.0 signal and the latency
    benchmark measures search machinery, not embedding generation."""
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def _card(
    id: str,
    fact: str,
    *,
    evidence: str = "e",
    timestamp: float | None = None,
    embedding: list[float] | None = None,
    tags: list[str] | None = None,
) -> MemoryCard:
    return MemoryCard(
        id=id,
        fact=fact,
        evidence=evidence,
        timestamp=timestamp if timestamp is not None else now_ts(),
        tags=tags or [],
        embedding=embedding,
        source_tokens=5,
    )


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


# --------------------------------------------------------------------------- #
# 1. FTS5 inverted index + sync triggers
# --------------------------------------------------------------------------- #
class TestFTS5Sync:
    def test_fts_table_exists_and_empty_at_init(self, store):
        n = store._conn.execute("SELECT count(*) FROM memories_fts").fetchone()[0]
        assert n == 0

    def test_insert_trigger_populates_fts(self, store):
        store.add(_card("c1", "the user prefers python"))
        n = store._conn.execute("SELECT count(*) FROM memories_fts").fetchone()[0]
        assert n == 1

    def test_delete_trigger_evicts_fts(self, store):
        store.add(_card("c1", "the user prefers python"))
        store.add(_card("c2", "rust is fast"))
        assert store.delete("c1") is True
        n = store._conn.execute("SELECT count(*) FROM memories_fts").fetchone()[0]
        assert n == 1  # only c2 remains

    def test_update_trigger_resyncs_fact(self, store):
        store.add(_card("c1", "old fact about vector search"))
        # Update to a NEW fact text; FTS must reflect the new text, not the old.
        store.update(_card("c1", "new fact about keyword bm25"))
        old_hit = store._conn.execute(
            "SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'vector*'"
        ).fetchone()[0]
        new_hit = store._conn.execute(
            "SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'bm25*'"
        ).fetchone()[0]
        assert old_hit == 0, "old fact text must be evicted from FTS on update"
        assert new_hit == 1, "new fact text must be indexed on update"

    def test_direct_write_triggers_sync(self, store):
        """The eval harness's bulk seeder writes rows directly to the connection,
        bypassing add(). The triggers must still keep FTS in sync."""
        with store._lock:
            cur = store._conn.cursor()
            cur.execute(
                "INSERT INTO memories(id, fact, evidence, timestamp, source_tokens) "
                "VALUES (?, ?, ?, ?, ?)",
                ("direct1", "a fact written raw via sqlite", "e", now_ts(), 5),
            )
            cur.close()
        n = store._conn.execute(
            "SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'raw*'"
        ).fetchone()[0]
        assert n == 1, "direct INSERT must trigger FTS sync"

    def test_rebuild_fts_heals_partial_index(self, store):
        """rebuild_fts must reconstruct the index after it's been partially
        corrupted (some FTS rows missing).

        FTS5 external-content tables are populated on CREATE from the content
        table, so the realistic corruption is a PARTIAL index — evict a row
        via the valid 'delete' command, then rebuild and confirm it returns."""
        store.add(_card("c1", "alpha beta"))
        store.add(_card("c2", "gamma delta"))
        assert store._conn.execute(
            "SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'alpha*'"
        ).fetchone()[0] == 1
        # Corrupt: evict c1's FTS row via the valid per-rowid 'delete' command.
        c1_rowid = store._conn.execute(
            "SELECT rowid FROM memories WHERE id = 'c1'"
        ).fetchone()[0]
        store._conn.execute(
            "INSERT INTO memories_fts(memories_fts, rowid, fact) "
            "VALUES ('delete', ?, 'alpha beta')",
            (c1_rowid,),
        )
        assert store._conn.execute(
            "SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'alpha*'"
        ).fetchone()[0] == 0, "precondition: c1 evicted from FTS"
        # Heal.
        store.rebuild_fts()
        assert store._conn.execute(
            "SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'alpha*'"
        ).fetchone()[0] == 1, "rebuild must re-index the evicted row"

    def test_rebuild_fts_prunes_archived_rows(self, store):
        """'rebuild' pulls ALL content-table rows including archived; rebuild_fts
        must evict archived cards so they stay out of search."""
        store.add(_card("c1", "live alpha", embedding=_unit_vec(0)))
        store.add(_card("c2", "live beta", embedding=_unit_vec(1)))
        store.archive_card("c1")
        store.rebuild_fts()
        # archived c1 must NOT match even though 'rebuild' would pull it.
        archived_hit = store._conn.execute(
            "SELECT count(*) FROM memories_fts f JOIN memories m ON m.rowid=f.rowid "
            "WHERE memories_fts MATCH 'alpha*' AND m.archived = 0"
        ).fetchone()[0]
        assert archived_hit == 0, "archived card must be pruned from FTS by rebuild"

    def test_archived_cards_excluded_from_search(self, store):
        store.add(_card("c1", "the user likes sqlite", embedding=_unit_vec(0)))
        store.add(_card("c2", "the user likes postgres", embedding=_unit_vec(1)))
        store.archive_card("c1")  # soft-delete: archived=ts
        # FTS MATCH must not return the archived card in hybrid_search.
        hits = store.hybrid_search("sqlite user", _unit_vec(0), k=5)
        ids = [c.id for c, _ in hits]
        assert "c1" not in ids, "archived card must not surface in hybrid search"


# --------------------------------------------------------------------------- #
# 2. RRF fusion math (pure function, no store needed)
# --------------------------------------------------------------------------- #
class TestRRFFusion:
    def test_card_strong_in_both_branches_wins(self):
        # 'a' is rank-1 in BOTH semantic and bm25; 'b'/'c' are rank-1 in one.
        sem = [("a", 0.99), ("b", 0.80), ("d", 0.50)]
        bm = [("a", 12.0), ("c", 8.0), ("d", 1.0)]
        fused = _rrf_fusion(sem, bm, {}, alpha=0.7, k=5)
        assert fused[0][0] == "a", "card strong in both branches must rank #1"
        # 'a' score = 0.7/61 + 0.3/61 = 1.0/61 = 0.016393...
        assert fused[0][1] == pytest.approx(1.0 / 61.0, abs=1e-6)

    def test_alpha_weights_branches(self):
        # alpha=1.0 => pure semantic rank (bm25 contributes nothing).
        sem = [("a", 0.9), ("b", 0.8)]
        bm = [("b", 100.0), ("c", 1.0)]
        fused_pure_sem = _rrf_fusion(sem, bm, {}, alpha=1.0, k=5)
        # 'a' rank-1 semantic => 1/61; 'b' rank-2 semantic => 1/62 + 0 bm25.
        assert fused_pure_sem[0][0] == "a"
        assert fused_pure_sem[0][1] == pytest.approx(1.0 / 61.0, abs=1e-6)
        # alpha=0.0 => pure bm25.
        fused_pure_bm = _rrf_fusion(sem, bm, {}, alpha=0.0, k=5)
        assert fused_pure_bm[0][0] == "b", "alpha=0 => bm25 rank-1 wins"
        assert fused_pure_bm[0][1] == pytest.approx(1.0 / 61.0, abs=1e-6)

    def test_absent_from_branch_contributes_zero(self):
        # 'a' only in semantic, 'b' only in bm25 — each gets only its branch term.
        fused = _rrf_fusion([("a", 1.0)], [("b", 1.0)], {}, alpha=0.5, k=5)
        by_id = dict(fused)
        assert by_id["a"] == pytest.approx(0.5 / 61.0, abs=1e-6)
        assert by_id["b"] == pytest.approx(0.5 / 61.0, abs=1e-6)
        # Tie => id-asc tiebreak.
        assert fused[0][0] == "a"

    def test_k_truncates(self):
        sem = [(f"c{i}", 1.0) for i in range(10)]
        bm: list = []
        fused = _rrf_fusion(sem, bm, {}, alpha=1.0, k=3)
        assert len(fused) == 3
        assert [c for c, _ in fused] == ["c0", "c1", "c2"]

    def test_entity_boost_adds_on_top(self):
        sem = [("a", 1.0), ("b", 1.0)]
        bm = [("a", 1.0)]
        # boost 'b' enough to overcome its rank disadvantage vs 'a'.
        fused = _rrf_fusion(sem, bm, {"b": 0.5}, alpha=0.7, k=5)
        assert fused[0][0] == "b", "entity boost must lift 'b' above rank-1 'a'"

    def test_raw_scores_discarded(self):
        # Swap the RAW scores (0.1 vs 9999) but keep ranks identical => same fused score.
        sem_lo = [("a", 0.1)]
        sem_hi = [("a", 9999.0)]
        bm = []
        f_lo = _rrf_fusion(sem_lo, bm, {}, alpha=1.0)
        f_hi = _rrf_fusion(sem_hi, bm, {}, alpha=1.0)
        assert f_lo[0][1] == f_hi[0][1], "RRF must fuse by RANK not raw score"


# --------------------------------------------------------------------------- #
# 3. Entity-graph boost (Mem0 decay formula)
# --------------------------------------------------------------------------- #
class TestEntityBoost:
    def test_decay_formula_single_witness(self):
        # N_linked=1 => 0.5 / (1 + 0) = 0.5 (max boost, single source).
        s = MemoryStore(":memory:")
        s.add(_card("m1", "the user codes in python", embedding=_unit_vec(0)))
        s.add(_card("target", "unrelated fact about weather", embedding=_unit_vec(5)))
        graph.insert_edge(s._conn, "m1", "target", "semantic", 0.9)
        s._conn.commit()
        boosts = s._entity_boosts("python", top_n_per_branch=10)
        assert boosts.get("target") == pytest.approx(0.5, abs=1e-6)

    def test_decay_formula_diminishing_returns(self):
        # N_linked=2 => 0.5/(1+0.001*1) = 0.5/1.001 ~ 0.4995 (barely less).
        # N_linked=100 => 0.5/(1+0.001*9801) ~ 0.5/10.8 ~ 0.0463 (much less).
        s = MemoryStore(":memory:")
        s.add(_card("target", "unrelated", embedding=_unit_vec(9)))
        for i in range(2):
            cid = f"src{i}"
            s.add(_card(cid, "python entity", embedding=_unit_vec(i)))
            graph.insert_edge(s._conn, cid, "target", "semantic", 0.5)
        s._conn.commit()
        b2 = s._entity_boosts("python", top_n_per_branch=10).get("target", 0.0)
        assert b2 == pytest.approx(0.5 / (1 + 0.001 * 1), abs=1e-5)

    def test_boost_surfaces_card_without_own_match(self):
        s = MemoryStore(":memory:")
        # 'target' has NO 'python' in its fact and a NON-matching vector.
        s.add(_card("m1", "the user codes in python", embedding=_unit_vec(0)))
        s.add(
            _card(
                "target",
                "completely different text about the weather today",
                embedding=_unit_vec(7),
            )
        )
        graph.insert_edge(s._conn, "m1", "target", "semantic", 0.9)
        s._conn.commit()
        hits = s.hybrid_search("python", _unit_vec(0), k=5)
        ids = [c.id for c, _ in hits]
        assert "target" in ids, "entity-boosted card must surface despite no own match"
        # And it should be near the top (boost ~0.5 >> RRF terms ~0.016).
        assert ids[0] == "target"

    def test_no_entities_no_boost(self):
        assert _extract_entities("the a an is it of") == []
        s = MemoryStore(":memory:")
        assert s._entity_boosts("", top_n_per_branch=10) == {}

    def test_boost_does_not_introduce_non_branch_cards(self):
        # Regression: an entity boost for a card that is in NEITHER branch's
        # hits (e.g. an archived/hard-deleted graph neighbor, or a card with
        # no embedding and no lexical match) must NOT surface it. The boost is
        # a re-ranking signal for cards the branches already deemed
        # searchable; it must never inject a fresh candidate (which, at ~0.5,
        # would dominate RRF scores ~0.017 and hijack rank 1).
        sem = [("good", 0.9)]
        bm = [("good", 5.0)]
        # 'zombie' is boosted but absent from both branches.
        boosts = {"zombie": 0.5, "good": 0.1}
        res = _rrf_fusion(sem, bm, boosts, alpha=0.7, k=5)
        ids = [cid for cid, _ in res]
        assert "zombie" not in ids, (
            "boosted card not in any branch must not surface (entity boost is "
            "re-ranking only, never candidate-introduction)"
        )
        assert ids[0] == "good"


# --------------------------------------------------------------------------- #
# 4. End-to-end hybrid_search vs pure vector_search recall
# --------------------------------------------------------------------------- #
class TestHybridRanking:
    def test_hybrid_surfaces_lexical_miss_by_vector(self, store):
        # The vector branch ranks 'a' high (it has the query vector) but 'a'
        # does NOT contain the query keyword; 'b' contains the keyword but
        # the vector branch ranks it lower. Hybrid must surface BOTH, and the
        # one matching BOTH modalities ranks highest — proving the BM25 branch
        # contributes recall the vector branch alone would miss.
        store.add(_card("a", "rust programming language", embedding=_unit_vec(1)))
        store.add(_card("b", "python programming language", embedding=_unit_vec(2)))
        vec_only = store.vector_search(_unit_vec(2), k=5, alpha=1.0)
        vec_ids = [c.id for c, _ in vec_only]
        assert "a" in vec_ids and "b" in vec_ids  # sanity: both reachable by vector
        # Query 'python': 'b' matches the keyword AND has the query vector.
        hits = store.hybrid_search("python", _unit_vec(2), k=5)
        assert hits[0][0].id == "b", "card matching BOTH modalities must rank #1"

    def test_vector_only_when_query_text_empty(self, store):
        store.add(_card("a", "some fact", embedding=_unit_vec(2)))
        store.add(_card("b", "other fact", embedding=_unit_vec(3)))
        # Empty query text => no FTS branch; pure vector ranking.
        hits = store.hybrid_search("", _unit_vec(2), k=2)
        assert hits and hits[0][0].id == "a"

    def test_bm25_only_when_query_vec_zero(self, store):
        store.add(_card("a", "the user likes sqlite", embedding=_unit_vec(0)))
        store.add(_card("b", "the user likes rust", embedding=_unit_vec(1)))
        # Zero query vector => no vector branch; pure BM25.
        hits = store.hybrid_search("sqlite", [0.0] * 16, k=2)
        assert hits and hits[0][0].id == "a"

    def test_degenerate_returns_empty(self, store):
        store.add(_card("a", "x", embedding=_unit_vec(0)))
        assert store.hybrid_search("", [0.0] * 16, k=5) == []

    def test_k_respected(self, store):
        for i in range(8):
            store.add(_card(f"c{i}", f"fact number {i}", embedding=_unit_vec(i)))
        hits = store.hybrid_search("fact number", _unit_vec(0), k=3)
        assert len(hits) <= 3

    def test_k_maintained_under_mid_hydration_delete(self, store):
        # Regression: a card in the fused top-k that is hard-deleted between
        # fusion and the batch_get hydration must not shrink the returned list
        # below k — the over-fetch buffer promotes the next-best candidate.
        for i in range(8):
            store.add(_card(f"c{i}", f"search item {i}", embedding=_unit_vec(i)))
        store.add(_card("best", "vector search engine", embedding=_unit_vec(0)))
        # Simulate a TOCTOU delete: wrap batch_get to drop 'best' on hydration.
        orig_batch_get = store.batch_get

        def drop_best(ids):
            return orig_batch_get([i for i in ids if i != "best"])

        store.batch_get = drop_best
        hits = store.hybrid_search("vector search", _unit_vec(0), k=5)
        assert len(hits) == 5, (
            f"mid-hydration delete shrank results to {len(hits)}; over-fetch "
            "buffer should promote the next-best candidate"
        )
        assert "best" not in [c.id for c, _ in hits]


# --------------------------------------------------------------------------- #
# 5. Performance: p99 latency < 5.0 ms at 10,000 cards
# --------------------------------------------------------------------------- #
def _seed_n_cards(store: MemoryStore, n: int, dim: int = 32) -> None:
    """Bulk-seed n cards with deterministic one-hot embeddings + varied facts.
    Writes directly to the connection (the eval-harness path) so the FTS
    triggers fire once per row, then rebuilds the FTS index in one pass."""
    cur = store._conn.cursor()
    cur.execute("BEGIN")
    facts = [
        "the user likes python and sqlite for vector search",
        "rust provides fast keyword bm25 retrieval",
        "the embedding model maps text to dense vectors",
        "hybrid fusion combines semantic and lexical signals",
    ]
    for i in range(n):
        v = [0.0] * dim
        v[i % dim] = 1.0
        cur.execute(
            "INSERT INTO memories(id, fact, evidence, timestamp, source_tokens, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"card-{i}",
                facts[i % len(facts)],
                "e",
                float(i),
                5,
                __import__("array").array("f", v).tobytes(),
            ),
        )
    cur.execute("COMMIT")
    cur.close()
    # FTS triggers fired per-row; rebuild once for a clean index (also exercises
    # the rebuild path under load).
    store.rebuild_fts()
    store._mark_vec_dirty()


class TestHybridPerf:
    @pytest.mark.perf
    def test_p99_under_5ms_at_10k_cards(self):
        """The spec target: hybrid retrieval p99 < 5.0 ms at 10,000 cards.

        Measures full hybrid_search calls (vector BLAS + FTS5 BM25 + RRF +
        entity boost + hydration) after a warm-up. One-hot embeddings make
        the matmul exact so the latency reflects the search machinery, not
        embedding quality. This is a SLA gate, not a micro-benchmark: a
        single p99 sample over the limit fails the build.
        """
        store = MemoryStore(":memory:")
        _seed_n_cards(store, 10_000, dim=32)
        query_vec = _unit_vec(0, dim=32)
        query = "vector embedding hybrid search"

        # Warm up the vector cache + SQLite page cache (production has hot
        # caches; a cold first call would dominate p99 unfairly).
        for _ in range(50):
            store.hybrid_search(query, query_vec, k=10, alpha=0.7)

        # Measure. 200 samples => p99 = the 2nd-worst (index 198 of 200 sorted).
        N = 200
        samples: list[float] = []
        for _ in range(N):
            t0 = time.perf_counter()
            store.hybrid_search(query, query_vec, k=10, alpha=0.7)
            samples.append((time.perf_counter() - t0) * 1000.0)  # ms

        samples.sort()
        p99 = samples[int(math.ceil(N * 0.99)) - 1]
        median = statistics.median(samples)
        print(f"\n[p99] n={N} median={median:.3f}ms p99={p99:.3f}ms (target <5.0ms)")
        assert p99 < 5.0, f"hybrid_search p99={p99:.3f}ms exceeds 5.0ms SLA at 10k cards"
