"""Reciprocal Rank Fusion retrieval over the existing store + FTS5 + graph.

Ports Mem0's multi-signal retrieval semantics onto isotope_zero's already-wired
primitives, with NO per-query LLM and NO network on the search hot path
(constraint 3). Mem0 inlines its retrieval into one file: ``mem0/memory/main.py``
fuses a vector_store.search (``main.py:1632`` ``_search_vector_store``) with an
entity-neighborhood boost (``main.py:1763`` ``_search_entity`` →
``memory_boosts`` keyed by ``linked_memory_ids``, weight
``similarity * ENTITY_BOOST_WEIGHT * memory_count_weight`` at ``main.py:1793``).
This module reproduces that *fusion shape* — two ranked branches merged by rank
position plus a graph-derived additive boost — but reads the signals straight
from isotope_zero's SQLite + BLAS + ``card_edges`` instead of a separate vector
store and entity store.

Fusion math (per Cormack et al., "Reciprocal Rank Fusion", SIGIR 2009, the same
formula mem0's ``core/graph.py`` neighborhood approximates):

    score(d) = alpha/(K + r_vec(d)) + (1-alpha)/(K + r_bm25(d))
             + entity_boost(d) * retention(d)

where ``r_*`` are 1-indexed ranks WITHIN each branch (a card absent from a
branch contributes 0 from that term), ``K`` is the RRF smoothing constant
(``60`` — so ``60 + rank >= 61`` and the division is zero-safe), and
``entity_boost(d)`` is a fixed ``0.1`` for cards in the graph-neighborhood of any
query entity (reusing ``isotope_zero.graph.relation_graph.RelationGraph.multi_hop_expand``
which BFS-es the existing ``card_edges`` table — no new table). ``retention(d)``
is the Ebbinghaus ``exp(-dt/S)`` from ``isotope_zero.core.decay.calculate_retention``,
so a recently-touched card outranks a stale twin all else equal.

Cross-module imports (graph, decay) are guarded with ``try/except ImportError``
and fall back to ``None`` so this module imports even when a sibling is absent
(constraint 5) — the fusion then runs vector+BM25-only, which is still correct.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from typing import Any

# RRF smoothing constant (Cormack et al. 2009). 60 + (1-indexed rank) >= 61,
# so the division can never hit zero — no guard needed on the denominator.
_RRF_K: int = 60

# Branch depth: how many candidates each leg contributes to the fusion pool.
# RRF's 1/(60+r) weighting makes a branch's contribution at rank 60 ~0.00017
# (alpha-weighted), well below the cosine score noise floor — deeper pools add
# latency, not recall. Matches the store's own ``top_n_per_branch`` default
# shape (``core/store.py`` ``hybrid_search``).
_BRANCH_K: int = 60

# Fixed additive boost for a card in the query-entity graph neighborhood.
# Mem0 scales boost by similarity * ENTITY_BOOST_WEIGHT (``main.py:1793``); here
# the entity match is binary (the card IS a graph neighbor of a query entity),
# so a constant 0.1 re-ranks linked cards above pure-vector ties without
# dominating RRF scores (~1/60 ≈ 0.017). This is a RE-RANKING signal only — a
# card must already be surfaced by a branch to receive it (see ``_fuse``).
_ENTITY_BOOST: float = 0.1

# Entity boost is layered on a card's retention, which lives in [0, 1], so the
# effective boost is at most 0.1 * 1.0 = 0.1 — enough to break a tie, never
# enough to override a rank-1 hit in either branch.

# Minimal English stopword set (mirrors ``core/store.py:_STOPWORDS``) so this
# module's entity extraction is self-contained when the store helpers aren't
# importable. Kept in sync deliberately; if one changes, change both.
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that
    the to was were will with this these those i you he she they we us our
    your his her their my me him them not no but if then than so such too very
    can do does did done doing about above after again all also am any each
    few more most other some only own same s t don now into out up down off
    over under again further once here there when where why how what which who
    whom whose
    """.split()
)

# Lazily-imported sibling modules. ``None`` when unavailable (constraint 5):
# the fusion degrades to vector+BM25-only, which is still correct, just
# un-boosted and un-decayed. Resolved at call time, not import time, so a
# missing sibling never raises at import.
try:  # pragma: no cover - exercised via the guarded path in tests
    from isotope_zero.graph.relation_graph import (
        RelationGraph as _RelationGraph,
    )
except ImportError:  # pragma: no cover
    _RelationGraph = None  # type: ignore[assignment,misc]

try:  # pragma: no cover
    from isotope_zero.core import decay as _decay
except ImportError:  # pragma: no cover
    _decay = None  # type: ignore[assignment,misc]

# Reuse the store's FTS5 query builder if importable (single source of truth
# for the MATCH expression shape); otherwise fall back to the local builder so
# this module works standalone. Both produce identical OR-joined prefix terms.
try:  # pragma: no cover
    from isotope_zero.core.store import _fts5_query as _store_fts5_query
except ImportError:  # pragma: no cover
    _store_fts5_query = None  # type: ignore[assignment,misc]


def _local_fts5_escape(token: str) -> str:
    """Escape a bare token for an FTS5 double-quoted phrase (mirrors
    ``core/store.py:_fts5_escape``). Doubles embedded quotes."""
    return token.replace('"', '""')


def _local_extract_entities(query: str) -> list[str]:
    """Zero-dependency entity extraction (mirrors ``core/store.py:_extract_entities``).

    Tokenize, lowercase, stopword-strip; return content-word tokens in
    first-seen order. This is a *recall* signal for the graph boost, not an NER.
    """
    if not query:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{1,}", query.lower())
    seen: list[str] = []
    seen_set: set[str] = set()
    for tok in tokens:
        if tok in _STOPWORDS or tok in seen_set:
            continue
        seen_set.add(tok)
        seen.append(tok)
    return seen


def _local_fts5_query(query: str) -> str:
    """Build an FTS5 MATCH expression: OR-joined double-quoted prefix terms.

    Each content-word entity becomes ``"tok"*`` so BM25 recalls any card whose
    fact mentions the token (prefix match: ``embed`` → ``embedding``). Returns
    ``""`` (no MATCH) when there are no entities.
    """
    entities = _local_extract_entities(query)
    if not entities:
        return ""
    return " OR ".join('"%s"*' % _local_fts5_escape(e) for e in entities)


def _build_fts5_query(query: str) -> str:
    """Use the store's canonical FTS5 query builder when importable, else local.

    Keeping one builder authoritative prevents the two from drifting in escape
    semantics; the local copy is the standalone fallback (constraint 5).
    """
    if _store_fts5_query is not None:
        try:
            return _store_fts5_query(query)
        except Exception:  # noqa: BLE001 - never let a helper crash the search
            pass
    return _local_fts5_query(query)


class HybridSearcher:
    """Reciprocal Rank Fusion of vector + FTS5 BM25, with graph + decay boosts.

    Wraps a ``MemoryStore`` (the same connection ``core/graph.init_graph`` and
    ``core/store.py:_init_schema`` already created ``card_edges`` and
    ``memories_fts`` on). All reads go through the store's existing public
    methods (``vector_search``) or the live FTS5 index — NO new schema, NO new
    tables, NO per-query LLM/network (constraints 1, 3).

    The fusion is *late*: each branch ranks independently by its native score
    (cosine for the vector leg, BM25 relevance for the keyword leg), then RRF
    merges them by RANK POSITION — this is what makes the two score scales
    commensurable (a cosine of 0.99 and a BM25 of 12.0 are not directly
    comparable, but "rank 1 in each branch" is). The ``60`` denominator damps
    top-rank dominance so a noisy branch can't swamp a precise one.

    Args:
        store: a ``MemoryStore`` with ``vector_search``, ``batch_get``,
            ``_conn``, and an optional ``embedder`` (with ``embed_text``).
        alpha: vector-leg weight in ``[0, 1]`` (default ``0.7``). ``alpha=1.0``
            is vector-only-with-boost; ``alpha=0.0`` is BM25-only-with-boost.
        branch_k: candidates each leg contributes to the fusion pool
            (default ``60``). Deeper pools add latency, not recall, under RRF's
            ``1/(60+r)`` weighting.
    """

    def __init__(
        self,
        store: Any,
        alpha: float = 0.7,
        branch_k: int = _BRANCH_K,
    ) -> None:
        self._store = store
        self._alpha = float(alpha)
        self._branch_k = int(branch_k)
        # Lazily-built RelationGraph over the store's connection. Reused across
        # calls so the BFS adjacency is rebuilt only when the graph changes;
        # RelationGraph holds no mutable state beyond the connection.
        self._graph: Any = None
        self._graph_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Graph access (lazy, guarded)
    # ------------------------------------------------------------------ #
    def _get_graph(self) -> Any:
        """Return a RelationGraph over the store's connection, or ``None``.

        Built once and cached; the underlying ``card_edges`` table is live in
        the shared connection, so a cached graph reflects edges added after
        construction. ``None`` when ``relation_graph`` is unavailable (the
        fusion then runs without the entity boost — still correct).
        """
        if _RelationGraph is None:
            return None
        if self._graph is None:
            with self._graph_lock:
                if self._graph is None:
                    try:
                        self._graph = _RelationGraph(self._store._conn)
                    except Exception:  # noqa: BLE001 - degrade gracefully
                        return None
        return self._graph

    # ------------------------------------------------------------------ #
    # Branch 1: vector ranks (delegates to store.vector_search)
    # ------------------------------------------------------------------ #
    def vector_ranks(
        self,
        query_vec: list[float],
        k: int = _BRANCH_K,
        scope: str | None = "default",
    ) -> list[tuple[str, int]]:
        """Vector cosine ranks as ``(card_id, 1-indexed_rank)``.

        Delegates to ``store.vector_search(query_vec, k, scope=scope)`` which
        returns ``list[(MemoryCard, score)]`` ordered by score desc (the
        store's native ranking). Rank 1 = top match. Empty list when the query
        vector is degenerate or no cards are embeddable.
        """
        if not query_vec or all(v == 0.0 for v in query_vec):
            return []
        try:
            hits = self._store.vector_search(query_vec, k=k, scope=scope)
        except TypeError:
            # Older store signature without the ``scope`` kwarg — fall back to
            # the original (query_vec, k, alpha) form. Backward compat.
            try:
                hits = self._store.vector_search(query_vec, k)
            except Exception:  # noqa: BLE001
                return []
        except Exception:  # noqa: BLE001 - never let a branch crash fusion
            return []
        return [(card.id, rank) for rank, (card, _score) in enumerate(hits, start=1)]

    # ------------------------------------------------------------------ #
    # Branch 2: BM25 ranks (queries the live memories_fts index)
    # ------------------------------------------------------------------ #
    def bm25_search(
        self,
        query_text: str,
        k: int = _BRANCH_K,
        scope: str | None = "default",
    ) -> list[tuple[str, int]]:
        """FTS5 BM25 ranks as ``(card_id, 1-indexed_rank)``.

        Queries ``memories_fts`` (the external-content FTS5 table at
        ``core/store.py:513``) directly: ``SELECT rowid FROM memories_fts WHERE
        memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT ?``. Each FTS
        rowid maps to a base-table rowid, thence to the card id (``memories.id``
        is an UNINDEXED FTS column, readable straight from the index). Rank 1 =
        top BM25 match. Empty list when the query has no content-word entities
        or FTS5 is unavailable (degrades to vector-only — still correct).
        """
        fts_query = _build_fts5_query(query_text)
        if not fts_query:
            return []
        conn = self._store._conn
        cur = conn.cursor()
        try:
            # ``id`` is UNINDEXED in the FTS schema, so it is stored in the
            # index row and read without a JOIN to ``memories``. We still JOIN
            # to the base table ONLY for the live-row + scope filter (the FTS
            # index is not curated for archived/superseded/scope, so the base
            # table is the source of truth — see ``core/store.py`` BM25 branch).
            if scope is not None:
                rows = cur.execute(
                    "SELECT f.id "
                    "FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
                    "WHERE memories_fts MATCH ? "
                    "AND m.superseded_by IS NULL AND m.archived = 0 "
                    "AND m.scope = ? "
                    "ORDER BY bm25(memories_fts) LIMIT ?",
                    (fts_query, scope, k),
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT f.id "
                    "FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
                    "WHERE memories_fts MATCH ? "
                    "AND m.superseded_by IS NULL AND m.archived = 0 "
                    "ORDER BY bm25(memories_fts) LIMIT ?",
                    (fts_query, k),
                ).fetchall()
        except sqlite3.OperationalError:
            # FTS5 missing OR query parse error -> vector-only fallback.
            rows = []
        finally:
            cur.close()
        return [(str(r[0]), rank) for rank, r in enumerate(rows, start=1)]

    # ------------------------------------------------------------------ #
    # Entity-graph boost (reuses card_edges neighborhood)
    # ------------------------------------------------------------------ #
    def _entity_neighborhood(self, query_entities: list[str]) -> set[str]:
        """Card ids in the graph-neighborhood of any query entity.

        For each query entity, find cards whose ``fact`` mentions it (the BM25
        branch's matched set, reused to avoid a second FTS scan), then BFS the
        ``card_edges`` table from each via ``RelationGraph.multi_hop_expand``.
        The union is the boost-eligible set. Empty when the graph module is
        unavailable or no query entities were given.
        """
        if not query_entities:
            return set()
        graph = self._get_graph()
        if graph is None:
            return set()
        conn = self._store._conn
        # Seed: cards whose fact mentions a query entity (case-insensitive).
        # A single LIKE OR-chain over the content-word entities; for prototype
        # scales this is cheaper than a second FTS5 MATCH and avoids coupling
        # the boost to the BM25 branch's MATCH expression.
        seeds: list[str] = []
        cur = conn.cursor()
        try:
            like_clauses = " OR ".join(
                "LOWER(fact) LIKE ?" for _ in query_entities
            )
            params = tuple(f"%{e}%" for e in query_entities)
            rows = cur.execute(
                f"SELECT id FROM memories WHERE ({like_clauses}) "
                "AND superseded_by IS NULL AND archived = 0",
                params,
            ).fetchall()
            seeds = [str(r[0]) for r in rows]
        except sqlite3.OperationalError:
            seeds = []
        finally:
            cur.close()

        neighborhood: set[str] = set()
        for seed_id in seeds:
            try:
                expanded = graph.multi_hop_expand(seed_id, max_hops=2)
            except Exception:  # noqa: BLE001 - degrade gracefully
                expanded = []
            neighborhood.update(expanded)
            # A seed card is also boost-eligible (it mentions the entity).
            neighborhood.add(seed_id)
        return neighborhood

    # ------------------------------------------------------------------ #
    # Retention (Ebbinghaus decay, guarded)
    # ------------------------------------------------------------------ #
    def _retention(self, card: Any, current_ts: float | None) -> float:
        """Ebbinghaus retention for *card* via ``core.decay.calculate_retention``.

        Passes ``last_access`` + ``stability``; when ``last_access <= 0`` the
        decay engine treats it as freshly encoded (retention 1.0) — see
        ``core/decay.py:51``. Returns ``1.0`` when the decay module is
        unavailable (no temporal penalty — still correct, just un-decayed).
        """
        if _decay is None:
            return 1.0
        last_access = float(getattr(card, "last_access", 0.0) or 0.0)
        stability = float(getattr(card, "stability", 1.0) or 1.0)
        try:
            return float(
                _decay.calculate_retention(last_access, stability, current_ts)
            )
        except Exception:  # noqa: BLE001 - degrade gracefully
            return 1.0

    # ------------------------------------------------------------------ #
    # Fusion
    # ------------------------------------------------------------------ #
    def _fuse(
        self,
        vec_ranks: list[tuple[str, int]],
        bm25_ranks: list[tuple[str, int]],
        entity_neighborhood: set[str],
        cards_by_id: dict[str, Any],
        current_ts: float | None,
        k: int,
    ) -> list[tuple[str, float]]:
        """RRF + entity-boost * retention, returning top-k ``(id, fused_score)``.

        Score(d) = alpha/(K + r_vec(d)) + (1-alpha)/(K + r_bm25(d))
                 + entity_boost(d) * retention(d)

        ``r_*`` are 1-indexed ranks within each branch; a card absent from a
        branch contributes 0 from that branch's term. The entity boost is a
        RE-RANKING signal only: applied to cards already surfaced by a branch
        (present in ``scores``), so an archived/out-of-scope graph neighbor
        can never hijack the top of the results. Sorted by fused score desc,
        ties broken by id asc for determinism.
        """
        scores: dict[str, float] = {}
        alpha = self._alpha
        for cid, rank in vec_ranks:
            scores[cid] = scores.get(cid, 0.0) + alpha / (_RRF_K + rank)
        for cid, rank in bm25_ranks:
            scores[cid] = scores.get(cid, 0.0) + (1.0 - alpha) / (_RRF_K + rank)

        # Entity boost layered on retention — only for already-surfaced cards.
        for cid in list(scores.keys()):
            if cid in entity_neighborhood:
                card = cards_by_id.get(cid)
                retention = self._retention(card, current_ts) if card else 1.0
                scores[cid] = scores[cid] + _ENTITY_BOOST * retention

        ranked = sorted(scores.items(), key=lambda e: (-e[1], e[0]))
        return ranked[:k] if k > 0 else ranked

    # ------------------------------------------------------------------ #
    # Public search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_vec: list[float] | None = None,
        query_text: str | None = None,
        query_entities: list[str] | None = None,
        k: int = 5,
        scope: str | None = "default",
    ) -> list[tuple[str, float]]:
        """Multi-signal RRF search returning top-k ``(card_id, fused_score)``.

        Gathers vector ranks (via ``store.vector_search``) and BM25 ranks (via
        the live ``memories_fts`` index), fuses them with Reciprocal Rank
        Fusion, and re-ranks by an entity-graph boost (cards in the
        ``card_edges`` neighborhood of query entities) weighted by Ebbinghaus
        retention. No per-query LLM or network (constraint 3).

        Args:
            query_vec: L2-normalized query embedding. When ``None`` and
                ``query_text`` is given, the store's embedder is used
                (``store.embedder.embed_text``); when the embedder is ``None``
                or its output is degenerate, the vector leg is skipped
                (BM25-only). When given, takes precedence over ``query_text``
                for the vector leg.
            query_text: free-text query for the BM25 leg (and for embedding
                when ``query_vec`` is None). When ``None`` and ``query_vec``
                is given, the BM25 leg is skipped (vector-only).
            query_entities: explicit entity list for the graph boost. When
                ``None``, entities are extracted from ``query_text`` (zero-dep
                tokenizer + stopword strip). Empty list disables the boost.
            k: number of results (default ``5``).
            scope: scope string filter (default ``"default"``). Pass ``None``
                for cross-scope global retrieval.

        Returns:
            Top-k ``(card_id, fused_score)`` tuples ordered by fused score
            desc, ties broken by id asc. Scores are RRF-scaled (live in
            ~[0, 1/61] + boost), NOT clamped to [0, 1] — callers comparing to
            cosine should use ``vector_search`` directly.

        Raises:
            ValueError: when both ``query_vec`` and ``query_text`` are None
                (no signal to fuse).
        """
        if query_vec is None and not query_text:
            raise ValueError(
                "HybridSearcher.search requires at least one of "
                "query_vec or query_text"
            )

        import time as _time

        current_ts = _time.time()

        # Resolve the vector query: explicit vec, else embed the text.
        vec: list[float] | None = None
        if query_vec is not None:
            vec = query_vec
        elif query_text:
            embedder = getattr(self._store, "embedder", None)
            if embedder is not None:
                try:
                    candidate = embedder.embed_text(query_text)
                    # A degenerate (all-zero) embedding means the embedder is
                    # in fallback or the text is empty — skip the vector leg.
                    if candidate and not all(v == 0.0 for v in candidate):
                        vec = candidate
                except Exception:  # noqa: BLE001 - never let embed fail search
                    vec = None

        # Branches: vector-only, BM25-only, or both (the late-fusion case).
        vec_ranks: list[tuple[str, int]] = []
        if vec is not None:
            vec_ranks = self.vector_ranks(vec, k=self._branch_k, scope=scope)

        bm25_ranks: list[tuple[str, int]] = []
        if query_text:
            bm25_ranks = self.bm25_search(query_text, k=self._branch_k, scope=scope)

        # Degenerate: both legs empty (no signal matched anything).
        if not vec_ranks and not bm25_ranks:
            return []

        # Entities for the graph boost: explicit list, else extract from text.
        if query_entities is not None:
            entities = list(query_entities)
        elif query_text:
            entities = _local_extract_entities(query_text)
        else:
            entities = []

        entity_neighborhood = self._entity_neighborhood(entities)

        # Hydrate only the fused candidate ids (one batched SELECT) so the
        # retention lookup has the card's last_access/stability. Reuses the
        # store's batch_get (single round-trip, column order matches
        # _row_to_card).
        candidate_ids = list({cid for cid, _ in vec_ranks} | {cid for cid, _ in bm25_ranks})
        cards_by_id: dict[str, Any] = {}
        if candidate_ids:
            try:
                cards = self._store.batch_get(candidate_ids)
            except Exception:  # noqa: BLE001 - hydrate best-effort
                cards = []
            for c in cards:
                cards_by_id[c.id] = c

        fused = self._fuse(
            vec_ranks,
            bm25_ranks,
            entity_neighborhood,
            cards_by_id,
            current_ts,
            k,
        )
        return fused


def _smoke() -> None:
    """Inline smoke test: build a store, seed, search, print results.

    Run via ``python -m isotope_zero.retrieval.hybrid_search`` to confirm the
    module wires up against a real store end-to-end (constraint 4).
    """
    from isotope_zero.core.store import MemoryStore
    from isotope_zero.types import MemoryCard, now_ts

    store = MemoryStore(":memory:")
    # Two unit vectors so the vector leg is a precise signal.
    dim = 8
    v_python = [0.0] * dim
    v_python[0] = 1.0
    v_rust = [0.0] * dim
    v_rust[1] = 1.0

    store.add(
        MemoryCard(
            id="c1",
            fact="the user prefers python for scripting",
            evidence="",
            timestamp=now_ts(),
            tags=[],
            embedding=v_python,
            source_tokens=5,
        )
    )
    store.add(
        MemoryCard(
            id="c2",
            fact="the user writes rust for systems work",
            evidence="",
            timestamp=now_ts(),
            tags=[],
            embedding=v_rust,
            source_tokens=5,
        )
    )

    searcher = HybridSearcher(store)
    results = searcher.search(query_vec=v_python, query_text="python", k=5)
    print("smoke results (id, score):")
    for cid, score in results:
        print(f"  {cid}: {score:.6f}")
    assert results, "smoke: expected at least one result"
    assert results[0][0] == "c1", f"smoke: expected c1 first, got {results[0][0]}"
    print("smoke OK")


if __name__ == "__main__":
    _smoke()
