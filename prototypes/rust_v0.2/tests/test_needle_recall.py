"""Permanent needle-in-a-haystack recall suite (Section B regression guard).

This is the always-on (NOT gated behind IZERO_STRESS) regression suite for the
adversarial report's Section B: a single critical fact hidden among semantically
near-identical distractors that differ only by a decision-critical numeric
token ("port is 2204" vs "... port is 2203/22 ...").

The router's lexical exact-match boost (isotope_zero.core.router._vector_path)
is what keeps the needle inside the retrieved set at scale. This suite pins the
requirement so a future change that regresses the boost (or the embedding)
fails loudly.

Assertions
----------
- HARD floor: recall >= 90.0% at 500 distractors / 100 queries.
- Sweep: recall >= 90.0% at [50, 100, 200, 300, 500] distractors (fewer reps
  at small counts to stay fast). The printed table makes the scaling curve
  (log-linear vs worse) visible in CI logs.
"""
from __future__ import annotations

import pytest

from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.eval.adversarial import (
    NEEDLE_FACT,
    NEEDLE_QUERIES,
    _make_distractors,
    run_needle,
)

# (n_distractors, n_queries): fewer reps at small counts keeps the suite fast
# while the 500-distractor / 100-query point carries the hard 90% floor.
SWEEP: list[tuple[int, int]] = [
    (50, 20),
    (100, 20),
    (200, 40),
    (300, 60),
    (500, 100),
]

# The recall floor the spec requires (hard assertion).
MIN_RECALL_PCT = 90.0


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """One shared ONNX embedding engine for the whole module."""
    return EmbeddingEngine()


@pytest.fixture(scope="module")
def recall_log():
    """Collect (n_distractors, n_queries, recall_pct) rows and print the table
    after the module's tests run, so the numbers land in CI stdout."""
    rows: list[tuple[int, int, float]] = []
    yield rows
    print("\n" + "=" * 62)
    print("Needle-in-a-haystack recall  (lexical-boost router, Section B)")
    print(f"  needle fact: {NEEDLE_FACT!r}")
    print(f"  query sample: {NEEDLE_QUERIES[0]!r}")
    print("-" * 62)
    print(f"{'distractors':>12} {'queries':>8} {'recall %':>10}")
    print("-" * 62)
    for n_dist, n_q, pct in rows:
        marker = "" if pct >= MIN_RECALL_PCT else "   <-- FAIL"
        print(f"{n_dist:>12} {n_q:>8} {pct:>9.1f}%{marker}")
    print("=" * 62)


def _run_recall(engine: EmbeddingEngine, n_distractors: int, n_queries: int) -> float:
    """Build the needle + distractor store and run the needle queries.

    Reuses the exact needle construction from the adversarial harness
    (`run_needle` / `_make_distractors` / `NEEDLE_FACT` / `NEEDLE_QUERIES`).
    """
    result = run_needle(n_distractors, engine, reps=n_queries)
    return result.recall_pct


def test_needle_recall_floor_at_500(engine, recall_log) -> None:
    """HARD assertion: >=90% recall at 500 distractors / 100 queries."""
    n_dist, n_q = 500, 100
    recall = _run_recall(engine, n_dist, n_q)
    recall_log.append((n_dist, n_q, recall))
    assert recall >= MIN_RECALL_PCT, (
        f"needle recall {recall:.1f}% < {MIN_RECALL_PCT:.1f}% floor "
        f"at {n_dist} distractors / {n_q} queries — the lexical boost "
        f"no longer separates the exact needle from near-miss distractors."
    )


@pytest.mark.parametrize("n_distractors,n_queries", SWEEP)
def test_needle_recall_sweep(engine, recall_log, n_distractors, n_queries) -> None:
    """Recall must hold across the distractor-count sweep (scaling check)."""
    recall = _run_recall(engine, n_distractors, n_queries)
    recall_log.append((n_distractors, n_queries, recall))
    assert recall >= MIN_RECALL_PCT, (
        f"needle recall {recall:.1f}% < {MIN_RECALL_PCT:.1f}% at "
        f"{n_distractors} distractors / {n_queries} queries."
    )
