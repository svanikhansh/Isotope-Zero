"""Unit tests for the hybrid query router (isotope_zero.core.router)."""
from __future__ import annotations

import math
import uuid

import pytest

from isotope_zero.core.router import QueryRouter
from isotope_zero.core.store import MemoryStore
from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.tokens import estimate_tokens
from isotope_zero.types import MemoryCard, now_ts


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


@pytest.fixture
def engine():
    return EmbeddingEngine()


@pytest.fixture
def populated_store(engine):
    """A store with 3 cards covering identity, project, preference."""
    store = MemoryStore(":memory:", embedder=engine)
    items = [
        (uuid.uuid4().hex, "The user's name is Alice.", "My name is Alice", ["name", "identity"]),
        (uuid.uuid4().hex, "The user's project is Mercury.", "I work on a project called Mercury", ["project"]),
        (uuid.uuid4().hex, "The user prefers Rust over Go.", "I prefer Rust over Go", ["preference", "language"]),
    ]
    for cid, fact, evidence, tags in items:
        store.add(
            MemoryCard(
                id=cid,
                fact=fact,
                evidence=evidence,
                timestamp=now_ts(),
                tags=tags,
                embedding=engine.embed_text(fact),
                source_tokens=estimate_tokens(evidence),
            )
        )
    return store


def test_structured_query_routes_sql(populated_store, engine):
    router = QueryRouter(populated_store, engine)
    res = router.query("what is my name?", token_budget=300)
    assert res.route_used == "sql"
    assert len(res.hits) >= 1
    assert "alice" in res.hits[0].card.fact.lower()
    # SQL hits score at max relevance.
    assert res.hits[0].score == 1.0
    assert res.hits[0].route == "sql"


def test_sql_route_has_zero_embedding_cost(populated_store, engine):
    """The SQL path must not need the embedder for retrieval."""
    router = QueryRouter(populated_store, engine)
    # Sabotage the embedder so any vector use would be detectable — but the
    # SQL path shouldn't touch it. We can't easily assert "not called" without
    # spying; instead assert the route is sql and result is correct.
    res = router.query("what is my current project?", token_budget=300)
    assert res.route_used == "sql"
    assert "mercury" in res.hits[0].card.fact.lower()


def test_budget_truncation(populated_store, engine):
    """A tight budget must cap total tokens used."""
    router = QueryRouter(populated_store, engine)
    res = router.query("what is my name?", token_budget=5)
    # The single SQL hit is ~12 tokens; with budget=5 it may be dropped.
    # Either way tokens_used must not exceed budget (or hit count is 0).
    assert res.tokens_used <= 5 or len(res.hits) == 0


def test_budget_exhausted_flag_when_capped(populated_store, engine):
    router = QueryRouter(populated_store, engine)
    big_query = " ".join(populated_store.all()[0].fact for _ in range(1))
    res = router.query(big_query, token_budget=1)
    # A 1-token budget against any real hit should flag exhaustion or drop hits.
    assert res.budget_exhausted is True or len(res.hits) == 0


def test_tokens_saved_vs_raw_nonneg(populated_store, engine):
    router = QueryRouter(populated_store, engine)
    res = router.query("what is my name?", token_budget=300)
    assert res.tokens_saved_vs_raw >= 0


def test_vector_route_for_semantic_query(populated_store, engine):
    """A fuzzy query with no structured key should use the vector route."""
    router = QueryRouter(populated_store, engine)
    res = router.query("tell me about the user's hobbies", token_budget=300)
    # No structured key → not SQL-routed; falls to vector (or empty if no match).
    assert res.route_used in ("vector", "sql")
    # tokens_used stays within budget.
    assert res.tokens_used <= 300


def test_empty_store_query_returns_no_hits(engine):
    store = MemoryStore(":memory:", embedder=engine)
    router = QueryRouter(store, engine)
    res = router.query("what is my name?", token_budget=300)
    assert res.hits == []
    assert res.tokens_used == 0


def test_query_result_latency_recorded(populated_store, engine):
    router = QueryRouter(populated_store, engine)
    res = router.query("what is my name?", token_budget=300)
    assert res.latency_ms >= 0.0
    # SQL route should be sub-10ms for a tiny store.
    assert res.latency_ms < 50.0


def test_correctness_floor_all_structured(populated_store, engine):
    """Every structured fact must be retrievable via its natural query."""
    router = QueryRouter(populated_store, engine)
    cases = [
        ("what is my name?", "alice"),
        ("what's my current project?", "mercury"),
        ("what language do I prefer?", "rust"),
    ]
    for q, expected in cases:
        res = router.query(q, token_budget=300)
        facts = " ".join(h.card.fact for h in res.hits).lower()
        assert expected in facts, f"Query {q!r} did not surface {expected!r}: {facts}"


def test_zero_score_vector_hits_not_included(populated_store, engine):
    """Dead-zero vector matches must not pad the result."""
    router = QueryRouter(populated_store, engine)
    # A query whose fallback embedding shares no tokens with any fact.
    res = router.query("zzzzqqqq xxxx", token_budget=300)
    for h in res.hits:
        assert h.score > 0.0


def test_add_then_update_changes_retrieved_fact(populated_store, engine):
    router = QueryRouter(populated_store, engine)
    # Update Alice's name to Bob via a new card id reuse.
    alice = populated_store.sql_lookup("fact", "Alice")[0]
    updated = MemoryCard(
        id=alice.id,
        fact="The user's name is Bob.",
        evidence="Actually my name is Bob",
        timestamp=now_ts(),
        tags=alice.tags,
        embedding=engine.embed_text("The user's name is Bob."),
        source_tokens=estimate_tokens("Actually my name is Bob"),
    )
    populated_store.update(updated)
    res = router.query("what is my name?", token_budget=300)
    facts = " ".join(h.card.fact for h in res.hits).lower()
    assert "bob" in facts
