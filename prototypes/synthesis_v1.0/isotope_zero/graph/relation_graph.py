"""Entity-relation graph traversal over the existing ``card_edges`` table.

Mem0 models entities as first-class rows in a *separate* vector store
(``mem0/memory/main.py:554`` ``entity_store``) and links each entity to the
memories that mention it via a ``linked_memory_ids`` payload
(``mem0/memory/main.py:633``). Entities are extracted from text by
``mem0/utils/entity_extraction.py`` (spaCy NER + noun-compound heuristics) and
two entities that co-occur in a memory's source text are implicitly related.

This module ports that *traversal* semantics onto isotope_zero's already-wired
``core.graph`` ``card_edges`` table (source_id, target_id, relation_type,
weight, created_at — created by ``core.graph.init_graph`` at
``core/graph.py:54`` and initialized by the store at ``core/store.py:488``).
We do NOT create a parallel table, do NOT call any LLM, and do NOT touch the
network on the search hot path. Edge persistence delegates to the existing
upsert at ``core/graph.py:66`` (``insert_edge``, max-weight-on-conflict) so the
strongest observed relationship per (source, target, relation_type) wins.

Public API:
    - :meth:`RelationGraph.add_edge` — delegate to ``core.graph.insert_edge``.
    - :meth:`RelationGraph.multi_hop_expand` — undirected BFS over
      ``card_edges``, bounded by ``max_hops``, cycle-safe via a visited set.
    - :meth:`RelationGraph.graph_density` — mean outgoing-edge weight.
    - :meth:`RelationGraph.extract_triplets_from_text` — emit
      ``(e_i, "co_occurs_with", e_j)`` for every unordered entity pair that
      co-occurs (case-insensitive substring) in *text*. No LLM, pure Python.
"""
from __future__ import annotations

import collections
import difflib
import sqlite3
from typing import Any

# Cross-module import is guarded: a sibling porter's module may not ship in
# the same checkout, and the full test suite must stay green independently.
try:  # pragma: no cover - exercised when core.graph is present (always, here)
    from isotope_zero.core import graph as _core_graph
except ImportError:  # pragma: no cover
    _core_graph = None  # type: ignore[assignment]


class RelationGraph:
    """Entity-relation traversal over the shared ``card_edges`` table.

    Wraps a ``sqlite3.Connection`` (the store's connection — the same one
    ``core.graph.init_graph`` already created ``card_edges`` on). All edge
    writes delegate to ``core.graph.insert_edge`` so there is a single
    upsert authority and no schema drift.

    The graph is treated as **undirected** for traversal
    (``multi_hop_expand`` follows both ``source -> target`` and
    ``target -> source``), mirroring how Mem0's co-occurrence relation is
    symmetric: if entity A co-occurs with B, B co-occurs with A.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind to *conn* and ensure the ``card_edges`` table exists.

        ``init_graph`` is idempotent (``CREATE TABLE IF NOT EXISTS`` at
        ``core/graph.py:29``), so calling it here is safe whether the store
        already initialized the table or not — and keeps this class usable on
        a bare ``:memory:`` connection in tests.
        """
        self._conn = conn
        if _core_graph is not None:
            _core_graph.init_graph(conn)

    # ------------------------------------------------------------------ #
    # Edge persistence
    # ------------------------------------------------------------------ #
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "related_to",
        weight: float = 0.5,
    ) -> None:
        """Insert or upsert an edge, delegating to ``core.graph.insert_edge``.

        Reuses the existing max-weight-on-conflict upsert at
        ``core/graph.py:66`` (``ON CONFLICT DO UPDATE SET weight =
        MAX(weight, excluded.weight)``) rather than reimplementing it, so the
        strongest observed relationship per unique triple always wins and there
        is one persistence authority for ``card_edges``.

        Args:
            source_id: Card id at the edge's tail.
            target_id: Card id at the edge's head.
            relation_type: Free-form label (default ``"related_to"``); the
                existing table defaults to ``"semantic"`` but the default
                here matches the task's co-occurrence semantics.
            weight: Edge weight in ``[0, 1]`` (clamped by ``insert_edge`` at
                ``core/graph.py:79``); defaults to ``0.5``.
        """
        if _core_graph is None:  # pragma: no cover - core.graph always present
            raise RuntimeError("isotope_zero.core.graph is unavailable")
        _core_graph.insert_edge(
            self._conn,
            source_id,
            target_id,
            relation_type=relation_type,
            weight=weight,
        )

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #
    def multi_hop_expand(
        self,
        start_card_id: str,
        max_hops: int = 2,
        min_weight: float = 0.0,
    ) -> list[str]:
        """BFS over ``card_edges`` treating edges as UNDIRECTED.

        Starting at *start_card_id*, follow every edge whose ``weight`` is
        ``>= min_weight`` in *both* directions (``source -> target`` and
        ``target -> source``), up to *max_hops* levels deep. A ``visited`` set
        guarantees termination on cyclic graphs (a node is enqueued at most
        once). The start node itself is excluded from the result.

        This mirrors Mem0's neighborhood expansion — the set of memories
        reachable from a seed entity via the co-occurrence graph — but reads
        the edges Mem0 would have written into a vector-store payload straight
        from the SQLite ``card_edges`` table, with no per-query LLM or network.

        Args:
            start_card_id: Seed card id (excluded from the returned list).
            max_hops: Maximum BFS depth (``0`` returns ``[]``).
            min_weight: Only follow edges with ``weight >= min_weight``.

        Returns:
            Sorted list of reachable card ids (excluding the start). Sorted
            for deterministic test assertions; the traversal order itself is
            BFS.
        """
        if max_hops <= 0:
            return []

        # Build the undirected adjacency for edges passing the weight filter.
        # One query (UNION of both directions) instead of N round-trips.
        cur = self._conn.cursor()
        try:
            rows = cur.execute(
                "SELECT source_id, target_id FROM card_edges WHERE weight >= ?",
                (min_weight,),
            ).fetchall()
        finally:
            cur.close()

        adjacency: dict[str, set[str]] = {}
        for src, tgt in rows:
            adjacency.setdefault(src, set()).add(tgt)
            adjacency.setdefault(tgt, set()).add(src)

        visited: set[str] = {start_card_id}
        # (node, depth): depth is the hop count at which we reached the node.
        queue: collections.deque[tuple[str, int]] = collections.deque(
            [(start_card_id, 0)]
        )
        reachable: list[str] = []

        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                # Don't expand past max_hops. Neighbors enqueued here would
                # be at depth+1 > max_hops.
                continue
            for neighbor in adjacency.get(node, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    reachable.append(neighbor)
                    queue.append((neighbor, depth + 1))

        return sorted(reachable)

    # ------------------------------------------------------------------ #
    # Density
    # ------------------------------------------------------------------ #
    def graph_density(self, card_id: str) -> float:
        """Mean weight of *card_id*'s outgoing edges.

        Sum of outgoing edge weights divided by the count of outgoing edges
        (``max(1, count)`` guards the no-edges case). This is the local
        connectivity strength of a card — a hub with many strong edges has high
        density. Only OUTGOING edges (``source_id = card_id``) are counted,
        matching ``core.graph.compound_weight`` at ``core/graph.py:229``;
        incoming edges are a separate centrality signal.

        Args:
            card_id: Card id to measure.

        Returns:
            Mean outgoing weight, or ``0.0`` if the card has no outgoing
            edges.
        """
        cur = self._conn.cursor()
        try:
            row = cur.execute(
                "SELECT COALESCE(SUM(weight), 0.0), COUNT(*) "
                "FROM card_edges WHERE source_id = ?",
                (card_id,),
            ).fetchone()
        finally:
            cur.close()
        total = float(row[0]) if row else 0.0
        count = int(row[1]) if row else 0
        return total / count if count > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Text -> triplets (no LLM)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_entity(
        entity: str, known: list[str], threshold: float = 0.88
    ) -> str:
        """Canonicalize *entity* onto the closest already-seen entity string.

        Port of the mem0 v0.x entity-merge safety net (which matched entities
        by embedding cosine >= 0.7 + top-1) onto a stdlib-only, offline
        foundation: ``difflib.get_close_matches`` uses ``SequenceMatcher``
        ratio, so "OpenAI" -> "Open AI" (~0.9) and
        "Artificial Intelligence" -> "artificial intelligence" (1.0 after
        lowercasing) collapse onto one canonical form, while near-synonyms
        like "AI" vs "Artificial Intelligence" (~0.13) correctly stay separate
        — the LLM extraction step is responsible for THAT normalization, this
        is only the long-tail edit-distance safety net. Deliberately strict
        (0.88) so unrelated multi-word entities never falsely merge.

        Args:
            entity: The candidate string.
            known: Canonical forms seen so far in this batch.
            threshold: Min ``SequenceMatcher.ratio()`` to adopt a known form.

        Returns:
            The canonical form to use for *entity* (either its closest known
            match above threshold, or the original entity unchanged).
        """
        if not known:
            return entity
        # Case-insensitive match first: exact modulo case collapses for free.
        lower = entity.lower()
        for k in known:
            if k.lower() == lower:
                return k
        close = difflib.get_close_matches(
            entity, known, n=1, cutoff=threshold
        )
        return close[0] if close else entity

    def extract_triplets_from_text(
        self,
        text: str,
        entities: list[str],
    ) -> list[tuple[str, str, str]]:
        """Emit co-occurrence triplets for entity pairs found in *text*.

        For every UNORDERED pair of distinct entities where BOTH appear in
        *text* (case-insensitive substring search), emit
        ``(e_i, "co_occurs_with", e_j)``. This is the deterministic,
        no-LLM analog of Mem0's entity-relation extraction: Mem0 runs spaCy NER
        (``mem0/utils/entity_extraction.py``) then links entities that share a
        memory; here the caller supplies the entity list and we link the ones
        that co-occur in the given text.

        Entities within edit-distance threshold of an earlier-seen form first
        collapse onto that canonical form (see :meth:`_resolve_entity`), so
        "OpenAI" and "Open AI" yield one entity rather than a near-self pair.

        Args:
            text: Source text to scan.
            entities: Candidate entity strings. Empty list -> ``[]``.

        Returns:
            List of ``(subject, "co_occurs_with", object)`` tuples. Each
            unordered pair appears at most once; ordering within a pair is
            by sorted position so ``(A, _, B)`` and ``(B, _, A)`` are not
            both emitted.
        """
        if not entities or not text:
            return []

        lower_text = text.lower()
        # Which entities are present (case-insensitive substring)?
        present: list[str] = [
            e for e in entities if e and e.lower() in lower_text
        ]
        # Canonicalize onto near-duplicate forms seen earlier in this batch
        # (mem0 entity-merge safety net, see ``_resolve_entity``), then
        # deduplicate by normalized form so duplicate entities don't yield
        # duplicate triplets or self-pairs and spellings like "OpenAI" /
        # "Open AI" collapse onto one canonical id before pairing.
        seen: set[str] = set()
        known: list[str] = []
        unique_present: list[str] = []
        for e in present:
            canonical = self._resolve_entity(e, known)
            key = canonical.lower()
            if key not in seen:
                seen.add(key)
                known.append(canonical)
                unique_present.append(canonical)

        triplets: list[tuple[str, str, str]] = []
        n = len(unique_present)
        for i in range(n):
            for j in range(i + 1, n):
                e_i = unique_present[i]
                e_j = unique_present[j]
                triplets.append((e_i, "co_occurs_with", e_j))
        return triplets

    # ------------------------------------------------------------------ #
    # Smoke test
    # ------------------------------------------------------------------ #
    def _smoke(self) -> None:  # pragma: no cover
        """Inline integration test on an in-memory DB."""
        conn = sqlite3.connect(":memory:")
        rg = RelationGraph(conn)

        rg.add_edge("a", "b", "related_to", 0.5)
        rg.add_edge("b", "c", "related_to", 0.5)
        print("multi_hop_expand(a, 2):", rg.multi_hop_expand("a", max_hops=2))
        print("graph_density(a):", rg.graph_density("a"))
        trips = rg.extract_triplets_from_text(
            "alice and bob hike", ["alice", "bob", "carol"]
        )
        print("triplets:", trips)
        conn.close()


if __name__ == "__main__":
    RelationGraph(sqlite3.connect(":memory:"))._smoke()  # type: ignore[func-returns-value]
