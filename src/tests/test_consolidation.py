"""Tests for Phase 3 async consolidation (isotope_zero.core.consolidation).

Covers: deduplication merging, temporal decay/pruning, and concurrency
safety (background consolidation runs alongside reads/writes without lock
contention / corrupting SQLite).
"""
from __future__ import annotations

import asyncio
import math
import threading
import time

import pytest

from isotope_zero.core.consolidation import Consolidator
from isotope_zero.retrieval.hybrid_search import QueryRouter
from isotope_zero.core.store import MemoryStore
from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.types import MemoryCard, now_ts


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def _card(
    id: str,
    fact: str = "A fact.",
    evidence: str = "e",
    timestamp: float = 100.0,
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
    source_tokens: int = 5,
    access_count: int = 0,
    last_access: float = 0.0,
) -> MemoryCard:
    return MemoryCard(
        id=id,
        fact=fact,
        evidence=evidence,
        timestamp=timestamp,
        tags=tags or [],
        embedding=embedding,
        source_tokens=source_tokens,
        access_count=access_count,
        last_access=last_access,
    )


@pytest.fixture
def engine():
    return EmbeddingEngine()


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #

def test_exact_fact_duplicate_merges_into_one(engine):
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    emb = engine.embed_text("The user's name is Alice")
    store.add(_card("a", fact="The user's name is Alice.", evidence="My name is Alice", embedding=emb, timestamp=now - 10, last_access=now - 10, tags=["name"]))
    store.add(_card("b", fact="The user's name is Alice.", evidence="I am Alice", embedding=emb, timestamp=now - 5, last_access=now - 5, tags=["identity"]))

    report = Consolidator(store, embedder=engine).run()

    assert report.merged_cards == 1
    assert report.decayed_cards == 0  # fresh cards not pruned
    assert store.count() == 1
    survivor = store.all()[0]
    # Survivor is the EARLIEST-timestamp member.
    assert survivor.id == "a"
    # Evidence is merged (both fragments present), not lost.
    assert "Alice" in survivor.evidence
    assert "I am Alice" in survivor.evidence
    # Tags unioned.
    assert set(survivor.tags) == {"name", "identity"}


def test_near_duplicate_cosine_merges(engine):
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    # Two facts with very similar (but not identical) embeddings — real ONNX
    # embeds these near-identically; fallback also yields high cosine.
    store.add(_card("a", fact="I prefer dark mode in my editor", evidence="dark mode", embedding=engine.embed_text("I prefer dark mode in my editor"), timestamp=now - 10, last_access=now - 10))
    store.add(_card("b", fact="I like dark mode for my editor", evidence="dark theme", embedding=engine.embed_text("I like dark mode for my editor"), timestamp=now - 5, last_access=now - 5))
    store.add(_card("c", fact="My favorite database is PostgreSQL", evidence="postgres", embedding=engine.embed_text("My favorite database is PostgreSQL"), timestamp=now, last_access=now))

    report = Consolidator(store, embedder=engine).run()

    # a+b merged, c survives alone => 1 merged, 2 survivors.
    assert report.merged_cards == 1
    assert report.survivors == 2
    assert store.count() == 2


def test_distinct_cards_not_merged(engine):
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    store.add(_card("a", fact="The user's name is Alice", embedding=engine.embed_text("The user's name is Alice"), timestamp=now - 10, last_access=now - 10))
    store.add(_card("b", fact="The user's favorite database is PostgreSQL", embedding=engine.embed_text("The user's favorite database is PostgreSQL"), timestamp=now, last_access=now))

    report = Consolidator(store, embedder=engine).run()

    assert report.merged_cards == 0
    assert report.survivors == 2


def test_merge_preserves_access_counts(engine):
    """A merged survivor inherits the SUM of the cluster's recall pressure."""
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    emb = engine.embed_text("The user prefers Rust")
    store.add(_card("a", fact="The user prefers Rust", embedding=emb, timestamp=now - 20, access_count=3, last_access=now - 5))
    store.add(_card("b", fact="The user prefers Rust", embedding=emb, timestamp=now - 10, access_count=2, last_access=now - 2))

    Consolidator(store, embedder=engine).run()

    survivor = store.all()[0]
    assert survivor.access_count == 5  # 3 + 2
    assert survivor.last_access == now - 2  # max


def test_merge_earliest_timestamp_survives(engine):
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    emb = engine.embed_text("dup fact")
    store.add(_card("late", fact="dup fact", embedding=emb, timestamp=now - 5, last_access=now - 5))
    store.add(_card("early", fact="dup fact", embedding=emb, timestamp=now - 20, last_access=now - 20))

    Consolidator(store, embedder=engine).run()

    assert store.count() == 1
    assert store.all()[0].id == "early"


# --------------------------------------------------------------------------- #
# Temporal decay & pruning
# --------------------------------------------------------------------------- #

def test_vitality_recency_dominates_for_fresh_card(engine):
    store = MemoryStore(":memory:", embedder=engine)
    now = now_ts()
    fresh = _card("f", fact="fresh", timestamp=now, last_access=now, access_count=0)
    old = _card("o", fact="old", timestamp=now - 60 * 86400, last_access=now - 60 * 86400, access_count=0)
    cons = Consolidator(store, embedder=engine)
    assert cons.vitality(fresh, now=now) > cons.vitality(old, now=now)


def test_vitality_access_count_boosts_score(engine):
    """Two equally-old cards: the recalled one scores higher."""
    store = MemoryStore(":memory:", embedder=engine)
    now = now_ts()
    base_ts = now - 10 * 86400
    recalled = _card("r", fact="r", timestamp=base_ts, last_access=now - 100, access_count=5)
    never = _card("n", fact="n", timestamp=base_ts, last_access=base_ts, access_count=0)
    cons = Consolidator(store, embedder=engine)
    assert cons.vitality(recalled, now=now) > cons.vitality(never, now=now)


def test_decay_prunes_old_never_recalled_card(engine):
    store = MemoryStore(":memory:", embedder=engine)
    now = now_ts()
    # Old (120 days), never recalled, last_access = timestamp. Candidate.
    store.add(_card("stale", fact="an old stale note", embedding=engine.embed_text("an old stale note"), timestamp=now - 120 * 86400, last_access=now - 120 * 86400, access_count=0))
    # Fresh, recalled — must survive.
    store.add(_card("live", fact="a current fact", embedding=engine.embed_text("a current fact"), timestamp=now - 10, last_access=now - 5, access_count=3))

    report = Consolidator(store, embedder=engine, min_age_seconds=0).run()

    assert report.decayed_cards == 1
    assert store.count() == 1
    assert store.get("live") is not None
    assert store.get("stale") is None
    assert report.pruned_mean_vitality > 0.0


def test_decay_never_prunes_recalled_card(engine):
    """A card that has been recalled (access_count > 0) is never pruned,
    regardless of age."""
    store = MemoryStore(":memory:", embedder=engine)
    now = now_ts()
    store.add(_card("old_recalled", fact="old but recalled", embedding=engine.embed_text("old but recalled"), timestamp=now - 365 * 86400, last_access=now - 365 * 86400, access_count=1))

    report = Consolidator(store, embedder=engine, min_age_seconds=0).run()

    assert report.decayed_cards == 0
    assert store.get("old_recalled") is not None


def test_decay_respects_grace_period(engine):
    """A fresh never-recalled card is NOT pruned during the grace period."""
    store = MemoryStore(":memory:", embedder=engine)
    now = now_ts()
    # 1 minute old, never recalled. With a 1-hour grace it must survive.
    store.add(_card("fresh", fact="a fresh never-recalled note", embedding=engine.embed_text("a fresh never-recalled note"), timestamp=now - 60, last_access=now - 60, access_count=0))

    report = Consolidator(store, embedder=engine, min_age_seconds=3600).run()

    assert report.decayed_cards == 0
    assert store.get("fresh") is not None


# --------------------------------------------------------------------------- #
# Concurrency / no-lock-contention
# --------------------------------------------------------------------------- #

def test_consolidation_safe_alongside_concurrent_reads_writes(engine):
    """Background consolidation sweeps run concurrently with a mix of read
    and write traffic on other threads. Must complete without raising and
    leave the DB usable (no 'database is locked', no corruption)."""
    store = MemoryStore(":memory:", embedder=engine)
    router = QueryRouter(store, engine)

    # Seed some cards.
    for i in range(20):
        store.add(_card(f"seed-{i}", fact=f"fact number {i}", embedding=engine.embed_text(f"fact number {i}"), timestamp=float(i), access_count=i % 3))

    cons = Consolidator(store, embedder=engine, min_age_seconds=0)
    stop_flag = threading.Event()
    errors: list[Exception] = []

    def reader():
        while not stop_flag.is_set():
            try:
                router.query("what is my fact?", token_budget=100)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
                return

    def writer():
        j = 0
        while not stop_flag.is_set():
            try:
                store.add(_card(f"w-{threading.get_ident()}-{j}", fact=f"concurrent fact {j}", embedding=engine.embed_text(f"concurrent fact {j}"), timestamp=float(j)))
                j += 1
            except Exception as e:  # noqa: BLE001
                errors.append(e)
                return

    threads = [threading.Thread(target=reader) for _ in range(2)] + [threading.Thread(target=writer) for _ in range(2)]
    for t in threads:
        t.start()

    # Run several consolidation sweeps while reads/writes hammer the store.
    for _ in range(3):
        cons.run()

    stop_flag.set()
    for t in threads:
        t.join(timeout=5)

    assert errors == [], f"concurrency errors: {errors}"
    # Store still queryable post-sweep.
    res = router.query("what is my fact?", token_budget=100)
    assert res.latency_ms >= 0.0


def test_background_loop_runs_and_stops_cleanly(engine):
    """The daemon background loop executes sweeps and stops on demand."""
    store = MemoryStore(":memory:", embedder=engine)
    for i in range(10):
        store.add(_card(f"d-{i}", fact=f"fact {i}", embedding=engine.embed_text(f"fact {i}"), timestamp=float(i)))
    cons = Consolidator(store, embedder=engine, min_age_seconds=0)

    cons.start_background_loop(interval_seconds=0.1)
    # Let at least one sweep fire.
    time.sleep(0.4)
    cons.stop(timeout=2)

    # The background thread is no longer alive.
    assert cons._thread is None or not cons._thread.is_alive()


def test_consolidation_report_fields(engine):
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    store.add(_card("a", fact="dup", embedding=engine.embed_text("dup"), timestamp=now - 10, last_access=now - 10))
    store.add(_card("b", fact="dup", embedding=engine.embed_text("dup"), timestamp=now, last_access=now))

    report = Consolidator(store, embedder=engine).run()

    assert report.merged_cards == 1
    assert report.survivors == 1
    assert report.tokens_before > report.tokens_after
    assert report.tokens_reclaimed == report.tokens_before - report.tokens_after
    assert report.tokens_reclaimed > 0
    assert report.latency_ms >= 0.0


def test_run_async_returns_report(engine):
    """The async wrapper returns the same ConsolidationReport type as run()."""
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    store.add(_card("a", fact="dup", embedding=engine.embed_text("dup"), timestamp=now - 10, last_access=now - 10))
    store.add(_card("b", fact="dup", embedding=engine.embed_text("dup"), timestamp=now, last_access=now))
    cons = Consolidator(store, embedder=engine)

    report = asyncio.run(cons.run_async())

    assert report.merged_cards == 1
    assert report.survivors == 1
