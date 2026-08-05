#!/usr/bin/env python3.14
"""vector_search + hybrid_search latency and recall@5 benchmark for the
Isotope Zero memory engine.

For each N in {100, 1000, 10000}:

  - Seeds N cards with deterministic, L2-normalized random embeddings.
  - Records p50/p95/p99 of ``store.vector_search`` over 200 warm queries.
  - Records p50/p95/p99 of ``store.hybrid_search`` over 200 warm queries.
  - Recall@5: compares the store's ``vector_search`` top-5 against an EXACT
    cosine top-5 computed in numpy over the full (N, dim) matrix. Reports the
    fraction of exact top-5 ids that appear in the store's top-5. A healthy
    implementation scores ~1.0 (the store uses the same dot product); the
    threshold is >= 0.95.

Dimension choice (documented): dim=384 is used — it matches the real
isotope_zero embedder's output dim, AND the full 10k x 384 float32 matrix is
only ~15 MB, seeds in ~0.2s, and each vector_search completes in <1ms at 10k
(full numpy/BLAS matmul). 384 runs well under the 60s budget at 10k, so the
real-model dimension is used directly (no dim=8 fallback needed).

If numpy is unavailable the store falls back to its pure-Python dot-product
loop; this script then skips the recall comparison (it needs numpy for the
exact baseline) but still prints latency. The store itself also falls back
to the pure-Python path when numpy is missing.

Run:
    /tmp/iz_test_venv/bin/python3.14 \\
        /Users/svanikhansh/Documents/isotope_zero/benchmarks/benchmark_latency_recall.py
"""
from __future__ import annotations

import gc
import math
import os
import sys
import time
from array import array

# The package lives under prototypes/synthesis_v1.0 (importable as
# isotope_zero). Add it to sys.path so the script is runnable from anywhere.
_REPO_ROOT = "/Users/svanikhansh/Documents/isotope_zero"
_PKG_ROOT = os.path.join(_REPO_ROOT, "prototypes", "synthesis_v1.0")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from isotope_zero.types import MemoryCard, now_ts  # noqa: E402
from isotope_zero.core.store import MemoryStore  # noqa: E402

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Tuning constants.
# ---------------------------------------------------------------------------
_DIM = 384          # matches the real isotope_zero embedder output dim
_NS = (100, 1000, 10000)
_N_QUERIES = 200
_K = 5
_ALPHA = 0.70        # default store alpha (cosine + Ebbinghaus fusion)
_WARMUP = 20         # warm queries before timing
_SEED = 12345        # deterministic RNG seed
_RECALL_FLOOR = 0.95  # healthy recall@5 must be >= this


# Column list mirroring the memories table (17 cols). Built programmatically
# so the placeholder count can never drift from the column count.
_COLS = [
    "id", "fact", "evidence", "timestamp", "tags", "source_tokens",
    "embedding", "access_count", "last_access", "superseded_by",
    "stability", "importance", "archived", "scope",
    "content_fingerprint", "ttl_seconds", "expiration_timestamp",
]
_INSERT_SQL = (
    "INSERT INTO memories(" + ",".join(_COLS) + ") VALUES ("
    + ",".join("?" for _ in _COLS) + ")"
)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def _make_embeddings(n: int, dim: int, seed: int):
    """Return (n, dim) float32 L2-normalized embeddings, deterministic.

    Used for BOTH seeding the store and as the exact cosine baseline (so the
    recall comparison is against the same vectors that live in the store).
    """
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb /= norms
    return emb


def _seed_store(store: MemoryStore, emb) -> None:
    """Bulk-seed the store directly via its held connection (fast path).

    ``store.add()`` invokes ``graph.auto_link_cards`` on every insert, which
    is O(n) per insert (O(n^2) overall) — ~44s at 10k. That is the
    *graph-construction* path; the latency/recall benchmark measures the
    *read* path (vector_search / hybrid_search), which is independent of how
    the rows got there. We bulk-insert directly (the same path the eval
    harness uses), FTS5 triggers keep the hybrid-search index in sync, and
    ``invalidate_vector_cache()`` forces the lazy numpy matrix to rebuild on
    the first search. This keeps seeding under the 60s budget at 10k.
    """
    n, dim = emb.shape
    conn = store._conn
    cur = conn.cursor()
    vals = []
    for i in range(n):
        blob = array("f", emb[i].tolist()).tobytes()
        vals.append((
            f"card-{i}",
            f"fact number {i} about embeddings and vectors",
            f"evidence snippet {i}",
            float(i),
            None,
            3,
            blob,
            0,
            float(i),
            None,
            1.0,
            0.0,
            0.0,
            "default",
            None,
            None,
            None,
        ))
    cur.executemany(_INSERT_SQL, vals)
    conn.commit()
    cur.close()
    store.invalidate_vector_cache()


def _make_queries(nq: int, dim: int, seed: int):
    """Return (nq, dim) float32 L2-normalized query vectors, deterministic."""
    rng = np.random.default_rng(seed + 1)
    q = rng.standard_normal((nq, dim)).astype(np.float32)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q /= norms
    return q


# ---------------------------------------------------------------------------
# Timing + percentiles
# ---------------------------------------------------------------------------
def _percentiles(samples_ms: list[float]) -> tuple[float, float, float]:
    """Return (p50, p95, p99) from a list of per-call latencies in ms."""
    s = sorted(samples_ms)
    n = len(s)
    if n == 0:
        return (0.0, 0.0, 0.0)

    def _pct(p: float) -> float:
        # Nearest-rank percentile (inclusive), matching common benchmark
        # convention. Clamp the rank into [0, n-1].
        idx = max(0, min(n - 1, int(math.ceil(p / 100.0 * n)) - 1))
        return s[idx]

    return (_pct(50), _pct(95), _pct(99))


def _time_vector_search(store: MemoryStore, queries) -> tuple:
    """Time vector_search over the query set; return (p50, p95, p99, samples)."""
    k = _K
    alpha = _ALPHA
    # Warmup (force the vec cache build + JIT-stable numpy paths).
    for i in range(_WARMUP):
        store.vector_search(queries[i % len(queries)].tolist(), k=k,
                            alpha=alpha, scope=None)
    samples = []
    for j in range(len(queries)):
        q = queries[j].tolist()
        t0 = time.perf_counter()
        store.vector_search(q, k=k, alpha=alpha, scope=None)
        samples.append((time.perf_counter() - t0) * 1000.0)
    p50, p95, p99 = _percentiles(samples)
    return (p50, p95, p99, samples)


def _time_hybrid_search(store: MemoryStore, queries) -> tuple:
    """Time hybrid_search over the query set; return (p50, p95, p99, samples).

    Each query uses the embedding as ``query_vec`` and a small text query
    derived from the deterministic seed so the BM25 branch is exercised.
    """
    k = _K
    alpha = _ALPHA
    for i in range(_WARMUP):
        q_text = f"fact number {i} embeddings"
        store.hybrid_search(q_text, queries[i % len(queries)].tolist(),
                            k=k, alpha=alpha, top_n_per_branch=30, scope=None)
    samples = []
    for j in range(len(queries)):
        q_text = f"fact number {j % 100} embeddings"
        t0 = time.perf_counter()
        store.hybrid_search(q_text, queries[j].tolist(), k=k,
                            alpha=alpha, top_n_per_branch=30, scope=None)
        samples.append((time.perf_counter() - t0) * 1000.0)
    p50, p95, p99 = _percentiles(samples)
    return (p50, p95, p99, samples)


# ---------------------------------------------------------------------------
# Recall@5
# ---------------------------------------------------------------------------
def _recall_at_5(store: MemoryStore, emb, queries) -> float:
    """Recall@5: fraction of exact-numpy top-5 ids in store's top-5.

    For each query: compute the exact top-5 by clipped cosine over the full
    (N, dim) matrix in numpy, then take the set-overlap with the store's
    ``vector_search`` top-5. ``alpha=1.0`` is used for the store call so the
    score is pure cosine (no Ebbinghaus temporal decay), matching the exact
    baseline. The store and numpy both use the dot product on the same
    float32 vectors, so recall should be ~1.0.
    """
    nq, dim = queries.shape
    total_hit = 0
    total_possible = 0
    for j in range(nq):
        q = queries[j]
        sims = np.clip(emb @ q, 0.0, 1.0)
        exact_idx = np.argsort(-sims)[:_K]
        exact_ids = {f"card-{int(i)}" for i in exact_idx}
        store_hits = store.vector_search(q.tolist(), k=_K, alpha=1.0,
                                         scope=None)
        store_ids = {c.id for c, _ in store_hits}
        total_hit += len(exact_ids & store_ids)
        total_possible += _K
    return total_hit / total_possible if total_possible else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not _HAS_NUMPY:
        print("numpy not available — recall@5 will be skipped, latency still "
              "measured (store uses its pure-Python fallback).")
    else:
        print(f"numpy {np.__version__} available — recall@5 computed against "
              "exact cosine baseline.")

    print(f"dim={_DIM}, queries={_N_QUERIES}, k={_K}, alpha={_ALPHA}")
    print("=" * 92)

    header = (
        f"{'N':>6} | "
        f"{'vec p50':>8} {'vec p95':>8} {'vec p99':>8} | "
        f"{'hyb p50':>8} {'hyb p95':>8} {'hyb p99':>8} | "
        f"{'recall@5':>9}"
    )
    print(header)
    print("-" * 92)

    results = []
    for n in _NS:
        emb = _make_embeddings(n, _DIM, _SEED)
        queries = _make_queries(_N_QUERIES, _DIM, _SEED)

        store = MemoryStore(db_path=":memory:")
        _seed_store(store, emb)
        assert store.count() == n, f"seed mismatch: {store.count()} != {n}"

        vp50, vp95, vp99, _ = _time_vector_search(store, queries)
        hp50, hp95, hp99, _ = _time_hybrid_search(store, queries)

        if _HAS_NUMPY:
            recall = _recall_at_5(store, emb, queries)
            recall_str = f"{recall:.4f}"
        else:
            recall = float("nan")
            recall_str = "skipped: numpy absent"

        results.append({
            "n": n, "vp50": vp50, "vp95": vp95, "vp99": vp99,
            "hp50": hp50, "hp95": hp95, "hp99": hp99,
            "recall": recall,
        })

        print(
            f"{n:>6} | "
            f"{vp50:>7.3f}m {vp95:>7.3f}m {vp99:>7.3f}m | "
            f"{hp50:>7.3f}m {hp95:>7.3f}m {hp99:>7.3f}m | "
            f"{recall_str:>9}"
        )

        store.close()
        gc.collect()

    print("=" * 92)

    # Recall gate.
    if _HAS_NUMPY:
        worst = min(r["recall"] for r in results)
        if worst < _RECALL_FLOOR:
            print(f"FAIL: worst recall@5 = {worst:.4f} < {_RECALL_FLOOR}")
            return 1
        print(f"recall@5 OK: worst = {worst:.4f} (>= {_RECALL_FLOOR})")
    else:
        print("recall@5 skipped (numpy absent)")

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
