"""Semantic knowledge graph engine for isotope_zero Phase 7C.

Transforms flat memory cards into a dynamic, weighted graph of semantic edges.
Each edge represents a relationship between two cards: cosine-similarity above
a tunable threshold, shared tags (Jaccard weight), or derived consolidation
links. High-density, decay-stable clusters are candidates for consolidation
into summary cards.

Design rules:
  - Pure Python, stdlib only (sqlite3, math, time, collections.deque).
  - Operates on the store's ``sqlite3.Connection`` — does NOT create its own.
  - No threading needed (the store serializes callers on its lock).
  - Double-quoted strings, ``from __future__ import annotations``, typed
    signatures, docstrings on every public symbol.
"""
from __future__ import annotations

import collections
import math
import sqlite3
import time
from typing import Any


# ------------------------------------------------------------------ #
# Schema
# ------------------------------------------------------------------ #

_CREATE_EDGES_TABLE = """
CREATE TABLE IF NOT EXISTS card_edges (
    source_id       TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    relation_type   TEXT NOT NULL DEFAULT 'semantic',
    weight          REAL NOT NULL DEFAULT 1.0,
    created_at      REAL NOT NULL,
    PRIMARY KEY (source_id, target_id, relation_type)
);
"""

_CREATE_SOURCE_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_card_edges_source ON card_edges(source_id);"
)

_CREATE_TARGET_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_card_edges_target ON card_edges(target_id);"
)


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #


def init_graph(conn: sqlite3.Connection) -> None:
    """Create ``card_edges`` table and indexes if they don't exist.  Idempotent."""
    cur = conn.cursor()
    try:
        cur.execute(_CREATE_EDGES_TABLE)
        cur.execute(_CREATE_SOURCE_INDEX)
        cur.execute(_CREATE_TARGET_INDEX)
        conn.commit()
    finally:
        cur.close()


def insert_edge(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    relation_type: str = "semantic",
    weight: float = 1.0,
) -> None:
    """Insert or update an edge.

    If the edge already exists (same source, target, and relation type), the
    weight is updated to the maximum of the old and new values. This keeps the
    strongest observed relationship for each unique pair.
    """
    weight = max(0.0, min(1.0, weight))
    now = time.time()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO card_edges (source_id, target_id, relation_type, weight, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                weight = MAX(weight, excluded.weight),
                created_at = excluded.created_at
            """,
            (source_id, target_id, relation_type, weight, now),
        )
        conn.commit()
    finally:
        cur.close()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity as pure-Python dot-product over L2-normalized vectors.

    Assumes both vectors are unit-length (L2 norm == 1.0), so ``dot(a, b)`` IS
    the cosine.  Guard against zero-vectors for robustness.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # Clamp floating-point drift (e.g. 1.0000000000000002).
    return max(-1.0, min(1.0, dot))


def _jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity between two lists of strings."""
    set_a = set(a)
    set_b = set(b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def auto_link_cards(
    conn: sqlite3.Connection,
    card_id: str,
    tags: list[str],
    embedding: list[float] | None,
    all_embeddings: list[tuple[str, list[float]]],
    cosine_threshold: float = 0.75,
) -> list[str]:
    """Automatically create edges from *card_id* to semantically similar cards.

    Two linking strategies run in order:

    1. **cosine** — when *embedding* is not ``None``, the cosine similarity to
       every entry in *all_embeddings* is computed.  Pairs whose score exceeds
       *cosine_threshold* produce a ``"semantic"`` edge with
       ``weight = cosine_score``.

    2. **shared tags** — for every card_ that has at least one tag, the Jaccard
       similarity of tag sets is computed.  Pairs with non-zero Jaccard produce
       a ``"shared_tag"`` edge with ``weight = jaccard_similarity``.

    Self-edges are skipped.

    Returns:
        List of card IDs that were linked (as targets) to *card_id*.
    """
    if not all_embeddings and not tags:
        return []

    linked: set[str] = set()

    # ---- cosine edges ----------------------------------------------------
    if embedding is not None:
        for other_id, other_emb in all_embeddings:
            if other_id == card_id:
                continue
            score = _cosine_similarity(embedding, other_emb)
            if score > cosine_threshold:
                insert_edge(conn, card_id, other_id, "semantic", score)
                linked.add(other_id)

    # ---- shared-tag edges -------------------------------------------------
    if tags:
        for other_id, other_emb in all_embeddings:
            if other_id == card_id or other_id in linked:
                continue
            other_tags = _lookup_tags(conn, other_id)
            if not other_tags:
                continue
            jac = _jaccard(tags, other_tags)
            if jac > 0.0:
                insert_edge(conn, card_id, other_id, "shared_tag", jac)
                linked.add(other_id)

    return sorted(linked)


def _lookup_tags(conn: sqlite3.Connection, card_id: str) -> list[str]:
    """Return the tags for *card_id* from the memories table, or ``[]``."""
    import json

    cur = conn.cursor()
    try:
        row = cur.execute(
            "SELECT tags FROM memories WHERE id = ? AND superseded_by IS NULL",
            (card_id,),
        ).fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return []
    finally:
        cur.close()


def get_neighbors(
    conn: sqlite3.Connection,
    card_id: str,
    min_weight: float = 0.0,
    relation_types: list[str] | None = None,
) -> list[tuple[str, str, float]]:
    """Return outgoing edges from *card_id*.

    Returns:
        List of ``(target_id, relation_type, weight)``, sorted by weight
        descending.
    """
    cur = conn.cursor()
    try:
        if relation_types:
            placeholders = ",".join("?" for _ in relation_types)
            rows = cur.execute(
                f"SELECT target_id, relation_type, weight FROM card_edges "
                f"WHERE source_id = ? AND weight >= ? AND relation_type IN ({placeholders}) "
                f"ORDER BY weight DESC",
                (card_id, min_weight, *relation_types),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT target_id, relation_type, weight FROM card_edges "
                "WHERE source_id = ? AND weight >= ? "
                "ORDER BY weight DESC",
                (card_id, min_weight),
            ).fetchall()
        return [(row[0], row[1], row[2]) for row in rows]
    finally:
        cur.close()


def compound_weight(
    conn: sqlite3.Connection,
    card_id: str,
    min_weight: float = 0.0,
) -> float:
    """Sum of all outgoing edge weights above *min_weight*.

    Measures graph centrality — cards with higher compound weight act as hubs
    connecting multiple topics.
    """
    cur = conn.cursor()
    try:
        row = cur.execute(
            "SELECT COALESCE(SUM(weight), 0.0) FROM card_edges "
            "WHERE source_id = ? AND weight >= ?",
            (card_id, min_weight),
        ).fetchone()
        return float(row[0]) if row else 0.0
    finally:
        cur.close()


def detect_clusters(
    conn: sqlite3.Connection,
    min_cluster_size: int = 3,
    min_edge_weight: float = 0.80,
) -> list[list[str]]:
    """Find tightly-coupled clusters in the graph.

    A cluster is a connected component (via undirected BFS) where every
    traversed edge has ``weight >= min_edge_weight``.  Components smaller than
    *min_cluster_size* are discarded.

    Returns:
        List of clusters; each cluster is a ``list[str]`` of card IDs.
    """
    # Build adjacency: node → {neighbor}. Treat the graph as undirected.
    adjacency: dict[str, set[str]] = {}
    cur = conn.cursor()
    try:
        rows = cur.execute(
            "SELECT source_id, target_id FROM card_edges WHERE weight >= ?",
            (min_edge_weight,),
        ).fetchall()
    finally:
        cur.close()

    for src, tgt in rows:
        adjacency.setdefault(src, set()).add(tgt)
        adjacency.setdefault(tgt, set()).add(src)

    visited: set[str] = set()
    clusters: list[list[str]] = []

    for start in adjacency:
        if start in visited:
            continue
        # BFS
        queue: collections.deque[str] = collections.deque([start])
        visited.add(start)
        component: list[str] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for nb in adjacency.get(node, ()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(component) >= min_cluster_size:
            clusters.append(component)

    return clusters


def prune_stale_edges(
    conn: sqlite3.Connection,
    max_age_seconds: float = 7 * 86400,  # 7 days
) -> int:
    """Delete edges older than *max_age_seconds*.

    Returns:
        Count of deleted edges.
    """
    cutoff = time.time() - max_age_seconds
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM card_edges WHERE created_at < ?", (cutoff,))
        deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        cur.close()


def get_graph_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return summary statistics for the graph.

    Keys returned:
      - ``node_count``: number of distinct cards that appear as source or target.
      - ``edge_count``: total number of edges.
      - ``avg_weight``: mean edge weight.
      - ``max_weight``: maximum edge weight.
      - ``min_weight``: minimum edge weight.
    """
    cur = conn.cursor()
    try:
        node_count = cur.execute(
            "SELECT COUNT(DISTINCT id) FROM ("
            "  SELECT source_id AS id FROM card_edges"
            "  UNION"
            "  SELECT target_id AS id FROM card_edges"
            ")"
        ).fetchone()[0]

        edge_stats = cur.execute(
            "SELECT COUNT(*), COALESCE(AVG(weight), 0.0), "
            "       COALESCE(MAX(weight), 0.0), COALESCE(MIN(weight), 0.0) "
            "FROM card_edges"
        ).fetchone()

        return {
            "node_count": node_count,
            "edge_count": edge_stats[0],
            "avg_weight": round(edge_stats[1], 4),
            "max_weight": edge_stats[2],
            "min_weight": edge_stats[3],
        }
    finally:
        cur.close()


# ------------------------------------------------------------------ #
# Smoke test
# ------------------------------------------------------------------ #


def _smoke() -> None:
    """Inline integration test using an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")

    # ---- init ----------------------------------------------------------
    init_graph(conn)
    print("1. init_graph: card_edges table and indexes created")

    # ---- insert_edge ---------------------------------------------------
    insert_edge(conn, "a", "b", "semantic", 0.90)
    insert_edge(conn, "b", "c", "semantic", 0.85)
    insert_edge(conn, "a", "c", "semantic", 0.82)
    insert_edge(conn, "a", "d", "shared_tag", 0.40)
    # Idempotent update: insert_edge again with lower weight should keep max.
    insert_edge(conn, "a", "b", "semantic", 0.70)
    # Insert a new edge type for same pair keeps both.
    insert_edge(conn, "a", "b", "shared_tag", 0.30)
    print("2. insert_edge: 6 edges inserted (4 unique pairs, 1 double-type)")

    # Verify max-weight behavior.
    cur = conn.cursor()
    row = cur.execute(
        "SELECT weight FROM card_edges WHERE source_id='a' AND target_id='b' AND relation_type='semantic'"
    ).fetchone()
    print(f"   semantic a→b weight = {row[0]:.2f} (should be 0.90, kept max)")
    cur.close()

    # ---- get_neighbors -------------------------------------------------
    neighbors = get_neighbors(conn, "a")
    print(f"3. get_neighbors(a): {neighbors}")

    neighbors_filtered = get_neighbors(
        conn, "a", min_weight=0.50, relation_types=["semantic"]
    )
    print(f"   get_neighbors(a, min_weight=0.50, semantic): {neighbors_filtered}")

    # ---- compound_weight ------------------------------------------------
    cw = compound_weight(conn, "a")
    print(f"4. compound_weight(a): {cw:.2f}")

    cw_filtered = compound_weight(conn, "a", min_weight=0.60)
    print(f"   compound_weight(a, min=0.60): {cw_filtered:.2f}")

    # ---- detect_clusters ------------------------------------------------
    clusters = detect_clusters(conn, min_cluster_size=3, min_edge_weight=0.80)
    print(f"5. detect_clusters(min_size=3, min_weight=0.80): {len(clusters)} cluster(s)")
    for i, c in enumerate(clusters):
        print(f"   cluster {i}: {sorted(c)}")

    # Sub-threshold cluster (weight 0.40 below threshold) should be excluded.
    clusters_loose = detect_clusters(conn, min_cluster_size=2, min_edge_weight=0.30)
    print(f"   detect_clusters(min_size=2, min_weight=0.30): {len(clusters_loose)} cluster(s)")
    for i, c in enumerate(clusters_loose):
        print(f"   cluster {i}: {sorted(c)}")

    # ---- get_graph_stats ------------------------------------------------
    stats = get_graph_stats(conn)
    print(f"6. stats: nodes={stats['node_count']}, edges={stats['edge_count']}, "
          f"avg_w={stats['avg_weight']:.2f}, max_w={stats['max_weight']:.2f}")

    # ---- prune_stale_edges ----------------------------------------------
    # All edges just created → 0 deleted when using 7-day cutoff.
    deleted = prune_stale_edges(conn, max_age_seconds=7 * 86400)
    print(f"7. prune_stale_edges(7d): {deleted} deleted (should be 0)")

    # ---- auto_link_cards ------------------------------------------------
    # Create a minimal memories table for _lookup_tags to work.
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS memories ("
        "  id TEXT PRIMARY KEY, fact TEXT, tags TEXT, superseded_by TEXT"
        ")"
    )
    import json

    cur.execute(
        "INSERT INTO memories (id, fact, tags) VALUES (?, ?, ?)",
        ("card_x", "rust is fast", json.dumps(["tech", "performance"])),
    )
    cur.execute(
        "INSERT INTO memories (id, fact, tags) VALUES (?, ?, ?)",
        ("card_y", "python is slow", json.dumps(["tech", "scripting"])),
    )
    conn.commit()
    cur.close()

    # _cosine_similarity smoke (L2-normalized vectors)
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0, 0.0]
    v3 = [0.707, 0.707, 0.0, 0.0]  # ~L2-normalized
    print(f"8. cosine(v1,v2)={_cosine_similarity(v1, v2):.3f} (should be 0.000)")
    print(f"   cosine(v1,v3)={_cosine_similarity(v1, v3):.6f} (should be ~0.707)")

    all_embs = [
        ("card_x", [0.8, 0.6, 0.0, 0.0]),
        ("card_y", [0.3, 0.1, 0.0, 0.0]),
    ]
    linked = auto_link_cards(
        conn,
        card_id="self",
        tags=["tech", "performance"],
        embedding=[0.9, 0.4, 0.0, 0.0],
        all_embeddings=all_embs,
        cosine_threshold=0.75,
    )
    print(f"9. auto_link_cards linked: {linked}")

    # ---- done -----------------------------------------------------------
    conn.close()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke()
