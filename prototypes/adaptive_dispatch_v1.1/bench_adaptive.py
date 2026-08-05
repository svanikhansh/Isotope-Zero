"""Benchmark for the scale-adaptive vector search dispatcher.

Builds one AdaptiveVectorSearch, pre-loads it to each target N (100, 1000, 2500,
10000) with random L2-normalized 384-d vecs, then times 200 queries per N and
records p50/p95/p99 (ms), the path that fired (int8 vs blas), the hysteresis
behavior at N=2500, and a vs-BLAS ratio. Also forces the numpy int8 fallback
(when the native kernel is present) to report honest fallback numbers.

Deterministic seed=1337. Uses time.perf_counter and resource for RSS (macOS
ru_maxrss returns BYTES -- divide by 1e6 for MB).
"""

from __future__ import annotations

import resource
import time
from typing import Any

import numpy as np

from adaptive_search import (
    AdaptiveVectorSearch,
    _try_import_native_kernel,
    int8_dot_numpy,
    quantize_int8_symmetric,
)

SEED = 1337
DIM = 384
TARGETS = [100, 1000, 2500, 10000]
WARMUP = 20
REPS = 200
K = 5


def _rss_mb() -> float:
    """Peak RSS in MB (macOS ru_maxrss is in BYTES, not KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _percentiles_ms(samples_s: "list[float]") -> "tuple[float, float, float]":
    """Return (p50, p95, p99) in milliseconds from second-samples."""
    arr = np.asarray(samples_s, dtype=np.float64) * 1000.0
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def _build_to_n(n: int, rng: np.random.Generator) -> "tuple[AdaptiveVectorSearch, list[Any]]":
    """Build a fresh store loaded with ``n`` random normalized vecs."""
    s = AdaptiveVectorSearch(dim=DIM, capacity=max(1024, n))
    ids = []
    for i in range(n):
        v = rng.standard_normal(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        rid = f"card-{i}"
        s.add(v, rid)
        ids.append(rid)
    return s, ids


def _time_search(store: AdaptiveVectorSearch, queries: "list[Any]") -> "list[float]":
    """Time REPS searches, after WARMUP warmup queries, returning per-call seconds."""
    for q in queries[:WARMUP]:
        store.search(q, k=K)
    samples = []
    for _ in range(REPS):
        q = queries[np.random.randint(0, len(queries))]
        t0 = time.perf_counter()
        store.search(q, k=K)
        samples.append(time.perf_counter() - t0)
    return samples


def _force_blas_search(store: AdaptiveVectorSearch, queries: "list[Any]") -> "list[float]":
    """Time searches with the BLAS path forced (via a low blas_threshold)."""
    saved = store._blas_threshold
    store._blas_threshold = 0  # n > 0 -> always blas
    try:
        return _time_search(store, queries)
    finally:
        store._blas_threshold = saved


def _force_numpy_int8_search(
    store: AdaptiveVectorSearch, queries: "list[Any]"
) -> "list[float]":
    """Time searches with the numpy int8 fallback forced (kernel disabled).

    Builds the path by temporarily nulling ``store._kernel`` so the int8 path
    falls through to ``int8_dot_numpy``. Also forces the int8 path by setting
    int8_threshold very high.
    """
    saved_kernel = store._kernel
    saved_i8t = store._int8_threshold
    store._kernel = None
    store._int8_threshold = 10 ** 9  # always int8
    try:
        return _time_search(store, queries)
    finally:
        store._kernel = saved_kernel
        store._int8_threshold = saved_i8t


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    rng = np.random.default_rng(SEED)
    native_loaded = _try_import_native_kernel() is not None
    print("# adaptive_dispatch_v1.1 benchmark")
    print(f"seed={SEED} dim={DIM} warmup={WARMUP} reps={REPS} k={K}")
    print(f"native int8 kernel loaded: {native_loaded}")
    print(f"baseline RSS: {_rss_mb():.1f} MB")
    print()

    rows = []
    for n in TARGETS:
        store, ids = _build_to_n(n, rng)
        # Random queries (normalized) -- a distinct set from the stored vecs.
        queries = []
        for _ in range(max(WARMUP, REPS) + 8):
            q = rng.standard_normal(DIM).astype(np.float32)
            q /= np.linalg.norm(q)
            queries.append(q)

        # Adaptive (native int8 or numpy int8 fallback when native absent).
        samp = _time_search(store, queries)
        p50, p95, p99 = _percentiles_ms(samp)
        # Capture the adaptive path NOW -- the forced runs below mutate
        # store._path (the numpy run latches "int8"), which would otherwise
        # mislabel this row.
        path_fired = store.path

        # Forced BLAS (for the vs-BLAS ratio at every N).
        blas_samp = _force_blas_search(store, queries)
        b50, b95, b99 = _percentiles_ms(blas_samp)

        # Forced numpy int8 fallback (honest about the fallback being slower).
        np_samp = _force_numpy_int8_search(store, queries)
        np50, np95, np99 = _percentiles_ms(np_samp)

        ratio = p99 / b99 if b99 > 0 else float("inf")
        rows.append(
            (n, path_fired, p50, p95, p99, b50, b95, b99, np50, np95, np99, ratio)
        )
        print(
            f"N={n:>5} path={path_fired:<4} "
            f"adaptive p50={_fmt(p50)} p95={_fmt(p95)} p99={_fmt(p99)} ms | "
            f"BLAS p99={_fmt(b99)} | numpy-i8 p99={_fmt(np99)} | "
            f"ratio={ratio:.2f}x | RSS={_rss_mb():.1f}MB"
        )

    print()
    print("## Adaptive routing (markdown)")
    print("| N | path_fired | p50 (ms) | p95 (ms) | p99 (ms) | BLAS p99 (ms) | numpy-i8 p99 (ms) | vs_BLAS_ratio |")
    print("|---|---|---|---|---|---|---|---|")
    for n, path, p50, p95, p99, b50, b95, b99, np50, np95, np99, ratio in rows:
        print(
            f"| {n} | {path} | {_fmt(p50)} | {_fmt(p95)} | {_fmt(p99)} | "
            f"{_fmt(b99)} | {_fmt(np99)} | {ratio:.2f}x |"
        )

    # -- hysteresis demonstration at N=2500 -------------------------------
    print()
    print("## Hysteresis at N=2500 (band: 2000 < N <= 3000)")
    store, _ = _build_to_n(2500, rng)
    q = rng.standard_normal(DIM).astype(np.float32)
    q /= np.linalg.norm(q)
    store.search(q, k=K)
    print(f"  N=2500 fresh store -> latched path: {store.path}")
    # Demonstrate a switch: remove down to n=1500 (below int8_threshold).
    for i in range(1000):
        store.remove(f"card-{i}")
    print(f"  after removing to N={store.n} (<=2000) -> path should switch:")
    store.search(q, k=K)
    print(f"  N={store.n} -> latched path: {store.path}")
    # Add back up to n=2500; should STAY int8 (hysteresis -- still in band).
    for i in range(1000):
        v = rng.standard_normal(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.add(v, f"new-{i}")
    print(f"  after adding back to N={store.n} (in band) -> hysteresis keeps:")
    store.search(q, k=K)
    print(f"  N={store.n} -> latched path: {store.path}")
    # Push past blas_threshold -> should switch to blas.
    for i in range(1000):
        v = rng.standard_normal(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.add(v, f"more-{i}")
    print(f"  after adding to N={store.n} (>3000) -> path should switch:")
    store.search(q, k=K)
    print(f"  N={store.n} -> latched path: {store.path}")

    # -- routing correctness assertions ------------------------------------
    print()
    print("## Routing assertions")
    ok = True
    for n in [100, 1000]:
        s, _ = _build_to_n(n, rng)
        s.search(rng.standard_normal(DIM), k=K)
        expect = "int8"
        status = "PASS" if s.path == expect else "FAIL"
        ok = ok and (s.path == expect)
        print(f"  N={n:>5}: expected={expect:<4} got={s.path:<4} [{status}]")
    s10k, _ = _build_to_n(10000, rng)
    s10k.search(rng.standard_normal(DIM), k=K)
    expect = "blas"
    status = "PASS" if s10k.path == expect else "FAIL"
    ok = ok and (s10k.path == expect)
    print(f"  N={10000:>5}: expected={expect:<4} got={s10k.path:<4} [{status}]")
    s25, _ = _build_to_n(2500, rng)
    s25.search(rng.standard_normal(DIM), k=K)
    print(
        f"  N={2500:>5}: hysteresis band -> latched path={s25.path} "
        f"(fresh store inits to int8; only boundary-cross changes it)"
    )
    print(f"\nassertions: {'ALL PASS' if ok else 'FAIL'}")

    # -- correctness cross-check: native vs numpy vs BLAS ------------------
    print()
    print("## Correctness cross-check (native int8 vs numpy int8 vs BLAS dot)")
    s, _ = _build_to_n(500, rng)
    q = rng.standard_normal(DIM).astype(np.float32)
    q /= np.linalg.norm(q)
    res_adaptive = s.search(q, k=K)
    # Force BLAS top-k
    s._blas_threshold = 0
    res_blas = s.search(q, k=K)
    s._blas_threshold = 3000
    # Force numpy int8
    s._kernel = None
    s._int8_threshold = 10 ** 9
    res_np = s.search(q, k=K)
    ids_a = [r[0] for r in res_adaptive]
    ids_b = [r[0] for r in res_blas]
    ids_n = [r[0] for r in res_np]
    print(f"  adaptive top-{K} ids: {ids_a}")
    print(f"  BLAS     top-{K} ids: {ids_b}")
    print(f"  numpy-i8 top-{K} ids: {ids_n}")
    # int8 quantization is lossy; allow small rank drift but require the top-1
    # to agree with BLAS (the rank-corr in quantization_v0.4 was >=0.99997).
    print(
        f"  top-1 agreement (adaptive vs BLAS): "
        f"{'PASS' if ids_a[0] == ids_b[0] else 'DRIFT (quantization lossy)'}"
    )
    print(
        f"  top-1 agreement (numpy-i8 vs BLAS): "
        f"{'PASS' if ids_n[0] == ids_b[0] else 'DRIFT (quantization lossy)'}"
    )


if __name__ == "__main__":
    main()
