"""Semantic edge-case tests for isotope_zero consolidation.

Covers the negation guard (which stops a fact and its negation from being
silently merged even when a real embedder places them near-identically) and
the temporal-sequence merge rules (earliest survivor, evidence union, summed
access pressure, decay vs. recall survival).
"""
from __future__ import annotations

import pytest

from isotope_zero.core.consolidation import Consolidator, _are_negations
from isotope_zero.core.store import MemoryStore
from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.types import MemoryCard, now_ts


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _card(
    id: str,
    fact: str,
    *,
    evidence: str = "e",
    timestamp: float | None = None,
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
    source_tokens: int = 5,
    access_count: int = 0,
    last_access: float = 0.0,
) -> MemoryCard:
    """Build a MemoryCard with now-relative defaults (never epoch 0)."""
    ts = timestamp if timestamp is not None else now_ts()
    return MemoryCard(
        id=id,
        fact=fact,
        evidence=evidence,
        timestamp=ts,
        tags=tags or [],
        embedding=embedding,
        source_tokens=source_tokens,
        access_count=access_count,
        last_access=last_access,
    )


@pytest.fixture
def engine() -> EmbeddingEngine:
    return EmbeddingEngine()


# --------------------------------------------------------------------------- #
# 1. Negation handling
# --------------------------------------------------------------------------- #
def test_negation_not_merged_despite_vector_proximity(engine):
    """The negation guard overrides cosine: a fact and its negation survive.

    Real embedders place 'User uses Mac' and 'User does not use Mac' very
    close together (cosine well above the dedup threshold), but they assert
    opposite polarities and must NEVER be folded. In fallback mode cosine is
    meaningless, but the negation guard still blocks the merge via the
    exact-fact / token-overlap paths being overridden. count==2 in both modes.
    """
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    fact_a = "User uses Mac"
    fact_b = "User does not use Mac"
    store.add(_card("a", fact_a, evidence="ev-a", timestamp=now - 10,
                    last_access=now - 10, embedding=engine.embed_text(fact_a)))
    store.add(_card("b", fact_b, evidence="ev-b", timestamp=now - 5,
                    last_access=now - 5, embedding=engine.embed_text(fact_b)))

    report = Consolidator(store, embedder=engine).run()

    assert store.count() == 2
    assert report.merged_cards == 0
    # Both originals survive unmerged.
    assert store.get("a") is not None
    assert store.get("b") is not None
    assert store.get("a").fact == fact_a
    assert store.get("b").fact == fact_b


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("User uses Mac", "User does not use Mac", True),
        ("The user prefers dark mode", "The user does not prefer dark mode", True),
        ("I like tea", "I do not like tea", True),
        ("The user works at Acme", "The user no longer works at Acme", True),
        ("User uses Mac", "User uses Windows", False),
        ("User uses Mac", "User uses Mac", False),
        ("I like tea", "I like coffee", False),
    ],
)
def test_are_negations_parametrized(a, b, expected):
    """Direct unit tests of the module-level negation guard."""
    assert _are_negations(a, b) is expected


def test_duplicate_still_merged_when_not_negation(engine):
    """Identical facts (no negation) must still merge — the guard only blocks
    opposite-polarity pairs, not legitimate duplicates."""
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    fact = "User uses Mac"
    emb = engine.embed_text(fact)
    store.add(_card("a", fact, evidence="ev-a", timestamp=now - 10,
                    last_access=now - 10, embedding=emb))
    store.add(_card("b", fact, evidence="ev-b", timestamp=now - 5,
                    last_access=now - 5, embedding=emb))

    report = Consolidator(store, embedder=engine).run()

    assert store.count() == 1
    assert report.merged_cards == 1


@pytest.mark.skipif(
    not EmbeddingEngine().is_real,
    reason="needs semantic embeddings",
)
def test_distinct_facts_still_merged_when_paraphrase(engine):
    """Two paraphrased facts (no negation) merge via the semantic cosine path.

    Real-mode only: the fallback embedder is not semantic, so cosine on
    paraphrases is not reliable. In real mode 'I prefer dark mode in my
    editor' and 'I like dark mode for my editor' sit close enough to merge.
    """
    assert engine.is_real
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    fact_a = "I prefer dark mode in my editor"
    fact_b = "I like dark mode for my editor"
    store.add(_card("a", fact_a, evidence="ev-a", timestamp=now - 10,
                    last_access=now - 10, embedding=engine.embed_text(fact_a)))
    store.add(_card("b", fact_b, evidence="ev-b", timestamp=now - 5,
                    last_access=now - 5, embedding=engine.embed_text(fact_b)))

    report = Consolidator(store, embedder=engine).run()

    assert store.count() == 1
    assert report.merged_cards == 1


def test_are_duplicates_method_directly(engine):
    """Call Consolidator._are_duplicates directly: negation pair -> False,
    identical pair -> True."""
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    cons = Consolidator(store, embedder=engine)

    fact_pos = "User uses Mac"
    fact_neg = "User does not use Mac"
    card_pos = _card("a", fact_pos, timestamp=now,
                     embedding=engine.embed_text(fact_pos))
    card_neg = _card("b", fact_neg, timestamp=now,
                     embedding=engine.embed_text(fact_neg))
    card_dup = _card("c", fact_pos, timestamp=now,
                     embedding=engine.embed_text(fact_pos))

    assert cons._are_duplicates(card_pos, card_neg) is False
    assert cons._are_duplicates(card_pos, card_dup) is True


# --------------------------------------------------------------------------- #
# 2. Temporal sequence — newer facts override older, preserving evidence
# --------------------------------------------------------------------------- #
def test_merge_survivor_is_earliest_timestamp(engine):
    """The survivor of a duplicate cluster is the EARLIEST-timestamp member
    (canonical 'first seen')."""
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    emb = engine.embed_text("dup fact")
    early_id = "early"
    store.add(_card(early_id, "dup fact", evidence="e1",
                    timestamp=now - 20, last_access=now - 20, embedding=emb))
    store.add(_card("late", "dup fact", evidence="e2",
                    timestamp=now - 10, last_access=now - 10, embedding=emb))

    Consolidator(store, embedder=engine).run()

    survivors = store.all()
    assert len(survivors) == 1
    assert survivors[0].id == early_id


def test_merge_preserves_all_evidence_fragments(engine):
    """A 3-card duplicate cluster unions its distinct evidence fragments — the
    temporal sequence never LOSES evidence; older fragments survive in the
    merge."""
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    emb = engine.embed_text("dup fact")
    store.add(_card("c1", "dup fact", evidence="ev1",
                    timestamp=now - 20, last_access=now - 20, embedding=emb))
    store.add(_card("c2", "dup fact", evidence="ev2",
                    timestamp=now - 10, last_access=now - 10, embedding=emb))
    store.add(_card("c3", "dup fact", evidence="ev3",
                    timestamp=now - 5, last_access=now - 5, embedding=emb))

    Consolidator(store, embedder=engine).run()

    survivor = store.all()[0]
    assert "ev1" in survivor.evidence
    assert "ev2" in survivor.evidence
    assert "ev3" in survivor.evidence


def test_merge_access_counts_summed(engine):
    """A merged survivor inherits the SUM of the cluster's recall pressure,
    and the most-recent last_access."""
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    emb = engine.embed_text("dup fact")
    store.add(_card("c1", "dup fact", evidence="e1",
                    timestamp=now - 20, access_count=3, last_access=now - 5,
                    embedding=emb))
    store.add(_card("c2", "dup fact", evidence="e2",
                    timestamp=now - 10, access_count=2, last_access=now - 2,
                    embedding=emb))

    Consolidator(store, embedder=engine).run()

    survivor = store.all()[0]
    assert survivor.access_count == 5
    assert survivor.last_access == now - 2


def test_newer_fact_does_not_silently_overwrite_older_content(engine):
    """The survivor's fact is the EARLIEST card's fact, kept as-is — the merge
    is conservative and never invents a new fact."""
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    emb = engine.embed_text("dup fact")
    early_card = _card("early", "dup fact", evidence="e1",
                       timestamp=now - 20, last_access=now - 20, embedding=emb)
    store.add(early_card)
    store.add(_card("late", "dup fact", evidence="e2",
                    timestamp=now - 10, last_access=now - 10, embedding=emb))

    Consolidator(store, embedder=engine).run()

    survivor = store.all()[0]
    assert survivor.fact == early_card.fact


def test_temporal_decay_keeps_recalled_prunes_cold(engine):
    """Recency + recall determines survival: an old never-recalled card is
    pruned, while an equally-old recalled card survives."""
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    old = now - 120 * 86400  # 120 days old
    store.add(_card("cold", "a cold old note", evidence="e",
                    timestamp=old, access_count=0, last_access=old,
                    embedding=engine.embed_text("a cold old note")))
    store.add(_card("recalled", "a recalled note", evidence="e",
                    timestamp=old, access_count=5, last_access=now - 10,
                    embedding=engine.embed_text("a recalled note")))

    report = Consolidator(store, embedder=engine, min_age_seconds=0).run()

    assert store.count() == 1
    assert report.decayed_cards == 1
    assert store.all()[0].id == "recalled"
    assert store.get("cold") is None


def test_fresh_card_survives_grace_period(engine):
    """A 1-minute-old never-recalled card is NOT pruned during the 1-hour
    grace period."""
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    store.add(_card("fresh", "a fresh never-recalled note", evidence="e",
                    timestamp=now - 60, access_count=0, last_access=now - 60,
                    embedding=engine.embed_text("a fresh never-recalled note")))

    report = Consolidator(store, embedder=engine, min_age_seconds=3600).run()

    assert report.decayed_cards == 0
    assert store.count() == 1
    assert store.get("fresh") is not None


# --------------------------------------------------------------------------- #
# 3. Near-duplicate CORRECTIONS — the newest fact must win (Section C regressions)
# --------------------------------------------------------------------------- #
# The adversarial report's Section C failure mode: two facts differing by a
# single substantive token (a correction, e.g. inline -> remote) have cosine
# > 0.88, so they merge — and the OLD earliest-wins rule kept the STALE fact
# while deleting the NEWER correct one (a silent lost update). These 35
# regression cases assert the correction (rev, newer) becomes the live
# survivor and the stale original (orig, older) is superseded, not kept.

_TEMPLATE_SUBJECTS = [
    "the api gateway", "the order service", "the search index",
    "the billing store", "the deploy pipeline", "the metrics collector",
    "the auth cache", "the report job", "the notification hub",
    "the sync worker",
]


def _subject_slug(subject: str) -> str:
    return subject.removeprefix("the ").replace(" ", "_")


# (param_id, orig_fact, rev_fact) — orig is OLDER, rev is NEWER and must win.
_CORRECTION_REGRESSION_PAIRS: list[tuple[str, str, str]] = []
# tpl2 (K8s -> Nomad) — 3 subjects.
for _s in ["the auth cache", "the notification hub", "the sync worker"]:
    _CORRECTION_REGRESSION_PAIRS.append(
        (f"tpl2_subject_{_subject_slug(_s)}",
         f"{_s} runs on K8s.",
         f"{_s} no longer runs on K8s; it runs on Nomad.")
    )
# tpl4 (inline -> remote) — all 10 subjects.
for _s in _TEMPLATE_SUBJECTS:
    _CORRECTION_REGRESSION_PAIRS.append(
        (f"tpl4_subject_{_subject_slug(_s)}",
         f"{_s} is configured with inline.",
         f"{_s} is configured with remote.")
    )
# tpl6 (HTML -> JSON) — all 10 subjects.
for _s in _TEMPLATE_SUBJECTS:
    _CORRECTION_REGRESSION_PAIRS.append(
        (f"tpl6_subject_{_subject_slug(_s)}",
         f"{_s} produces HTML output.",
         f"{_s} produces JSON output.")
    )
# tpl7 (shard-1 -> shard-2) — all 10 subjects; subjects already begin with
# "the ", so orig/rev literally render as 'The the api gateway connects to
# shard-1.' — kept verbatim.
for _s in _TEMPLATE_SUBJECTS:
    _CORRECTION_REGRESSION_PAIRS.append(
        (f"tpl7_subject_{_subject_slug(_s)}",
         f"The {_s} connects to shard-1.",
         f"The {_s} connects to shard-2.")
    )
# tpl8 (follow -> manual) — 2 subjects.
for _s in ["the auth cache", "the sync worker"]:
    _CORRECTION_REGRESSION_PAIRS.append(
        (f"tpl8_subject_{_subject_slug(_s)}",
         f"{_s} default is follow.",
         f"{_s} default changed to manual.")
    )


@pytest.mark.skipif(
    not EmbeddingEngine().is_real,
    reason="regression suite requires real semantic embeddings",
)
@pytest.mark.parametrize(
    "case_id, orig, rev",
    _CORRECTION_REGRESSION_PAIRS,
    ids=[p[0] for p in _CORRECTION_REGRESSION_PAIRS],
)
def test_near_duplicate_correction_keeps_newest(case_id, orig, rev, engine):
    """Regression: a near-duplicate correction (single-token substitution, no
    negation) must keep the NEWEST fact as the live survivor — the stale
    original is superseded, never the other way around."""
    assert engine.is_real
    now = now_ts()
    store = MemoryStore(":memory:", embedder=engine)
    store.add(_card(f"{case_id}-a", orig, evidence=orig, timestamp=now - 10,
                    last_access=now - 10, embedding=engine.embed_text(orig)))
    store.add(_card(f"{case_id}-b", rev, evidence=rev, timestamp=now - 5,
                    last_access=now - 5, embedding=engine.embed_text(rev)))

    Consolidator(store, embedder=engine).run()

    live_facts = {c.fact.strip() for c in store.all()}
    assert rev.strip() in live_facts, (
        f"{case_id}: correction {rev!r} is not a live survivor; live={live_facts}"
    )
    assert orig.strip() not in live_facts, (
        f"{case_id}: stale original {orig!r} still live — lost update!"
    )
