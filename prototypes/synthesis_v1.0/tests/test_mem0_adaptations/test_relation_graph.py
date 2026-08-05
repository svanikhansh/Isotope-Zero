"""Tests for the entity-relation graph traversal ported from Mem0.

Covers ``isotope_zero.graph.RelationGraph``: edge insertion (delegates to the
existing ``core.graph`` upsert), undirected multi-hop BFS expansion, local
graph density, and no-LLM co-occurrence triplet extraction.

Conventions match the existing suite: ``:memory:`` store, plain ``test_*``
functions, real ``MemoryStore`` + ``core.graph`` wiring.
"""
from __future__ import annotations

import sqlite3

from isotope_zero.core.store import MemoryStore
from isotope_zero.graph import RelationGraph
from isotope_zero.types import MemoryCard, now_ts


def _card(id: str, fact: str = "A fact.") -> MemoryCard:
    return MemoryCard(
        id=id,
        fact=fact,
        evidence="evidence",
        timestamp=now_ts(),
        tags=[],
        source_tokens=5,
    )


def test_add_edge_delegates_to_core_graph_upsert():
    """add_edge persists a row in card_edges via the shared insert_edge."""
    store = MemoryStore(":memory:")
    rg = RelationGraph(store._conn)
    rg.add_edge("a", "b", "related_to", 0.5)

    cur = store._conn.cursor()
    try:
        row = cur.execute(
            "SELECT source_id, target_id, relation_type, weight "
            "FROM card_edges WHERE source_id = ? AND target_id = ?",
            ("a", "b"),
        ).fetchone()
    finally:
        cur.close()
    assert row is not None
    assert row[0] == "a"
    assert row[1] == "b"
    assert row[2] == "related_to"
    assert row[3] == 0.5


def test_multi_hop_expand_two_hops_returns_b_and_c():
    """A->B->C: expand(A, max_hops=2) reaches both B and C, not A."""
    store = MemoryStore(":memory:")
    # Three live cards so the ids are real graph nodes.
    store.add(_card("a", "fact a"))
    store.add(_card("b", "fact b"))
    store.add(_card("c", "fact c"))

    rg = RelationGraph(store._conn)
    rg.add_edge("a", "b")
    rg.add_edge("b", "c")

    reachable = rg.multi_hop_expand("a", max_hops=2)
    assert reachable == ["b", "c"], f"expected [b, c], got {reachable}"
    assert "a" not in reachable


def test_multi_hop_expand_undirected_follows_reverse_direction():
    """Edge B->A is followed in reverse: expand(B) reaches A even though the
    edge was stored as source=B, target=A (undirected)."""
    store = MemoryStore(":memory:")
    store.add(_card("a"))
    store.add(_card("b"))

    rg = RelationGraph(store._conn)
    rg.add_edge("b", "a")  # stored direction is b -> a

    reachable = rg.multi_hop_expand("a", max_hops=1)
    assert reachable == ["b"], f"undirected follow failed: {reachable}"


def test_multi_hop_expand_terminates_on_cycle():
    """A<->B<->C with a back-edge C->A must terminate (visited set)."""
    store = MemoryStore(":memory:")
    for cid in ("a", "b", "c"):
        store.add(_card(cid))

    rg = RelationGraph(store._conn)
    rg.add_edge("a", "b")
    rg.add_edge("b", "c")
    rg.add_edge("c", "a")  # creates a cycle a -> b -> c -> a

    reachable = rg.multi_hop_expand("a", max_hops=10)
    # Must terminate and contain the cycle's other nodes exactly once.
    assert sorted(reachable) == ["b", "c"]


def test_multi_hop_expand_max_hops_zero_returns_empty():
    store = MemoryStore(":memory:")
    store.add(_card("a"))
    store.add(_card("b"))
    rg = RelationGraph(store._conn)
    rg.add_edge("a", "b")
    assert rg.multi_hop_expand("a", max_hops=0) == []


def test_multi_hop_expand_respects_min_weight():
    """Edges below min_weight are not followed."""
    store = MemoryStore(":memory:")
    for cid in ("a", "b", "c"):
        store.add(_card(cid))
    rg = RelationGraph(store._conn)
    rg.add_edge("a", "b", weight=0.9)
    rg.add_edge("b", "c", weight=0.1)  # below the 0.5 cutoff
    reachable = rg.multi_hop_expand("a", max_hops=2, min_weight=0.5)
    assert reachable == ["b"], f"low-weight edge was followed: {reachable}"


def test_graph_density_positive_for_outgoing_edges():
    """graph_density(A) > 0 when A has outgoing edges."""
    store = MemoryStore(":memory:")
    store.add(_card("a"))
    store.add(_card("b"))
    rg = RelationGraph(store._conn)
    rg.add_edge("a", "b", weight=0.4)
    rg.add_edge("a", "c", weight=0.6)  # c need not be a stored card for the edge
    density = rg.graph_density("a")
    assert density > 0.0
    # mean of 0.4 and 0.6
    assert abs(density - 0.5) < 1e-9


def test_graph_density_zero_for_no_outgoing_edges():
    store = MemoryStore(":memory:")
    store.add(_card("a"))
    rg = RelationGraph(store._conn)
    assert rg.graph_density("a") == 0.0


def test_extract_triplets_co_occurrence_includes_alice_bob_not_carol():
    """Text mentioning alice+bob yields an alice/bob triplet; carol absent."""
    rg = RelationGraph(sqlite3.connect(":memory:"))
    triplets = rg.extract_triplets_from_text(
        "alice and bob hike",
        ["alice", "bob", "carol"],
    )
    assert len(triplets) == 1
    subj, rel, obj = triplets[0]
    assert rel == "co_occurs_with"
    assert {subj, obj} == {"alice", "bob"}
    # carol does not appear in the text, so no triplet touches it.
    assert "carol" not in subj and "carol" not in obj


def test_extract_triplets_case_insensitive():
    rg = RelationGraph(sqlite3.connect(":memory:"))
    trips = rg.extract_triplets_from_text(
        "Alice and Bob hiked together", ["alice", "bob"]
    )
    assert len(trips) == 1
    assert {trips[0][0], trips[0][2]} == {"alice", "bob"}


def test_extract_triplets_empty_entities_returns_empty():
    rg = RelationGraph(sqlite3.connect(":memory:"))
    assert rg.extract_triplets_from_text("some text", []) == []


def test_extract_triplets_empty_text_returns_empty():
    rg = RelationGraph(sqlite3.connect(":memory:"))
    assert rg.extract_triplets_from_text("", ["alice", "bob"]) == []


def test_extract_triplets_single_entity_no_pair():
    rg = RelationGraph(sqlite3.connect(":memory:"))
    assert rg.extract_triplets_from_text("alice alone", ["alice"]) == []


def test_extract_triplets_all_present_three_entities_yields_three_pairs():
    rg = RelationGraph(sqlite3.connect(":memory:"))
    trips = rg.extract_triplets_from_text(
        "alice bob carol together", ["alice", "bob", "carol"]
    )
    # 3 choose 2 = 3 unordered pairs
    assert len(trips) == 3


def test_extract_triplets_canonicalizes_near_duplicate_spellings():
    """Near-edit-distance spellings ("OpenAI" / "Open AI") collapse onto one
    canonical entity instead of emitting a near-self pair."""
    rg = RelationGraph(sqlite3.connect(":memory:"))
    trips = rg.extract_triplets_from_text(
        "OpenAI and Open AI are names for the same org",
        ["OpenAI", "Open AI", "Microsoft"],
    )
    # Microsoft absent; OpenAI/Open AI collapse to a single canonical entity,
    # so there is no pair at all.
    assert trips == []


def test_extract_triplets_keeps_distinct_entities_separate():
    """Entities sharing a substring but with low edit ratio ("AI" vs
    "Artificial Intelligence") must stay separate — never falsely merge."""
    rg = RelationGraph(sqlite3.connect(":memory:"))
    trips = rg.extract_triplets_from_text(
        "AI and Artificial Intelligence are discussed here",
        ["AI", "Artificial Intelligence"],
    )
    assert len(trips) == 1
    subj, rel, obj = trips[0]
    assert rel == "co_occurs_with"
    assert {subj, obj} == {"AI", "Artificial Intelligence"}
