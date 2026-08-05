#!/usr/bin/env python3.14
"""RSS (resident set size) floor + tracemalloc zero-leak benchmark for the
Isotope Zero memory engine.

Measures two memory-health properties of ``isotope_zero.core.store.MemoryStore``:

  1. **Cold-start RSS floor.** Importing the package and opening a fresh
     ``:memory:`` store should land at a modest RSS. The original design
     target documented in ``store.py`` is ~15 MB RSS for a cold store. We
     report the *actual* figure here — the floor is dominated by the Python
     3.14 interpreter + numpy (numpy is a lazy import of the store but the
     benchmark itself imports it for the vec-cache build, and psutil is
     loaded for RSS measurement), so the observed floor is higher than the
     bare-store 15 MB. We then seed N=10,000 cards (small dim=8 embeddings)
     and report RSS at 10k.

  2. **Zero-leak check.** Run 100,000 add-then-delete cycles on a single
     store and assert that ``tracemalloc``'s current + peak allocated bytes
     return to baseline (no monotonic growth). A bounded transient delta is
     tolerated (Python's pymalloc arenas are not returned to the OS on every
     free; the threshold catches genuine unbounded growth, not allocator
     noise).

Exit code 0 on success, non-zero on leak or RSS regression.

Stdlib + psutil only. psutil is pip-installed into the test venv on first
run if missing.

Run:
    /tmp/iz_test_venv/bin/python3.14 \\
        /Users/svanikhansh/Documents/isotope_zero/benchmarks/profile_rss_and_allocs.py
"""
from __future__ import annotations

import gc
import math
import os
import sys
import tracemalloc
from array import array

# ---------------------------------------------------------------------------
# Ensure psutil is available (stdlib + psutil is the full dependency set).
# ---------------------------------------------------------------------------
try:
    import psutil  # type: ignore[import-not-found]
except ImportError:
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "psutil"],
    )
    import psutil  # type: ignore[import-not-found]

# The package lives under prototypes/synthesis_v1.0 (importable as
# isotope_zero). Add it to sys.path so the script is runnable from anywhere.
_REPO_ROOT = "/Users/svanikhansh/Documents/isotope_zero"
_PKG_ROOT = os.path.join(_REPO_ROOT, "prototypes", "synthesis_v1.0")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from isotope_zero.types import MemoryCard, now_ts  # noqa: E402
from isotope_zero.core.store import MemoryStore  # noqa: E402

# ---------------------------------------------------------------------------
# Tuning constants.
# ---------------------------------------------------------------------------
_SEED_N = 10_000          # cards to seed for the RSS-at-10k probe
_DIM = 8                  # small embeddings (the RSS probe is about row count)
_LEAK_CYCLES = 100_000    # add-then-delete cycles for the zero-leak probe
# A real monotonic leak grows without bound across cycles; pymalloc retains
# freed arenas so a few KB of transient delta is expected and harmless. The
# leak is flagged only if current traced memory grows by more than this
# fraction of the peak seen during the run — i.e. a sustained, growing
# footprint, not a one-off high-water mark.
_LEAK_GROWTH_RATIO = 0.10  # cur may be up to 10% of (peak-baseline) over base
# The documented cold-store RSS target (~15 MB) is the *store-only* floor;
# with the interpreter + numpy + psutil resident the observed floor is higher.
# We report the actual number and only FAIL if it is pathologically large.
_RSS_COLD_FAIL_MB = 64.0   # hard fail ceiling (catch a real regression)
_RSS_AT_10K_FAIL_MB = 512.0  # ceiling for the 10k-card footprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rss_mb(proc: "psutil.Process") -> float:
    """Current resident set size in MiB."""
    return proc.memory_info().rss / (1024.0 * 1024.0)


def _norm(v: list[float]) -> list[float]:
    """L2-normalize a vector (so dot == cosine, matching the store contract)."""
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


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


def _seed_cards_direct(store: MemoryStore, n: int, dim: int) -> None:
    """Seed ``n`` cards via a direct bulk INSERT into the store's connection.

    ``store.add()`` invokes ``graph.auto_link_cards`` on every insert, which
    is O(n) per insert (it loads ALL existing embeddings into a Python list
    each call) — O(n^2) overall, ~44s and >200MB RSS at 10k. That path is
    the *graph-construction* path, not the *storage* path; the RSS probe is
    measuring the storage footprint of N cards, so we bulk-insert directly
    (the same path the eval harness uses) and then call
    ``invalidate_vector_cache()`` so the store's lazy caches rebuild on the
    next read. This isolates the storage RSS from the graph-builder's
    transient working set.
    """
    conn = store._conn
    cur = conn.cursor()
    vals = []
    for i in range(n):
        # Deterministic, distinct-enough-for-RSS-probe embeddings at dim=8.
        v = _norm([
            (i % 7) * 0.3 + 0.1,
            (i % 5) * 0.2 + 0.05,
            (i % 3) * 0.4 + 0.1,
            0.1, 0.2, 0.3, 0.15, 0.25,
        ])
        blob = array("f", v).tobytes()
        vals.append((
            f"card-{i}", f"fact {i}", f"evidence {i}", float(i),
            None, 3, blob, 0, float(i), None, 1.0, 0.0, 0.0, "default",
            None, None, None,
        ))
    cur.executemany(_INSERT_SQL, vals)
    conn.commit()
    cur.close()
    store.invalidate_vector_cache()


# ---------------------------------------------------------------------------
# Probe 1: cold-start RSS floor + RSS at 10k
# ---------------------------------------------------------------------------
def probe_rss(proc: "psutil.Process") -> dict:
    """Measure RSS at: baseline, after open, after 10k seed, after vec cache."""
    gc.collect()
    rows: list[tuple[str, float]] = []

    rows.append(("baseline (after import)", _rss_mb(proc)))
    store = MemoryStore(db_path=":memory:")
    rows.append(("after open :memory: store", _rss_mb(proc)))

    _seed_cards_direct(store, _SEED_N, _DIM)
    gc.collect()
    rows.append((f"after seed {_SEED_N} cards (dim={_DIM})", _rss_mb(proc)))

    # Build the vector-search cache (the (n, dim) float32 numpy matrix) so
    # the RSS figure reflects the steady-state read-path memory, not just the
    # SQL row storage.
    q = _norm([0.5, 0.1, 0.3, 0.2, 0.1, 0.4, 0.05, 0.2])
    _ = store.vector_search(q, k=5, alpha=1.0, scope=None)
    gc.collect()
    rows.append(("after vec cache build", _rss_mb(proc)))

    rss_at_10k = rows[-2][1]   # the seed figure (before vec cache)
    rss_after_cache = rows[-1][1]
    rss_cold = rows[1][1]      # after open

    store.close()
    gc.collect()
    return {
        "rows": rows,
        "rss_cold_mb": rss_cold,
        "rss_at_10k_mb": rss_at_10k,
        "rss_after_cache_mb": rss_after_cache,
    }


# ---------------------------------------------------------------------------
# Probe 2: tracemalloc zero-leak over 100k add/delete cycles
# ---------------------------------------------------------------------------
def probe_leak() -> dict:
    """Run 100k add-then-delete cycles; assert tracemalloc returns to baseline.

    Uses ``store.add()`` / ``store.delete()`` (the public write path) on a
    single store so the leak check exercises the real insert+delete code path
    including graph auto-linking, FTS5 trigger firing, and cache invalidation.
    Because each card is deleted before the next is added, the live row count
    stays at 0-1 throughout, so the O(n) graph scan per add is O(1) here and
    100k cycles complete in a few seconds.
    """
    tracemalloc.start()
    store = MemoryStore(db_path=":memory:")
    baseline_cur, baseline_peak = tracemalloc.get_traced_memory()

    emb = _norm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ts0 = now_ts()
    for cyc in range(_LEAK_CYCLES):
        cid = f"leak-{cyc}"
        card = MemoryCard(
            id=cid,
            fact=f"fact {cyc}",
            evidence=f"evidence {cyc}",
            timestamp=ts0 + cyc,
            tags=[],
            embedding=emb,
            source_tokens=3,
        )
        store.add(card)
        store.delete(cid)

    final_cur, final_peak = tracemalloc.get_traced_memory()
    store.close()
    tracemalloc.stop()

    cur_delta = final_cur - baseline_cur
    peak_delta = final_peak - baseline_peak
    # The leak is real only if current memory grew AND stayed grown
    # proportionally to the peak. pymalloc retains arenas, so a flat ~few-KB
    # delta is allocator noise; a monotonic leak shows up as cur_delta that is
    # a large fraction of peak_delta (the footprint never receded).
    # Allow up to _LEAK_GROWTH_RATIO of peak_delta as transient retention.
    leak = False
    if peak_delta > 0:
        ratio = cur_delta / peak_delta
        leak = ratio > _LEAK_GROWTH_RATIO and cur_delta > 1024 * 1024
    else:
        leak = cur_delta > 1024 * 1024  # >1MB absolute growth with no peak = leak

    return {
        "baseline_cur_b": baseline_cur,
        "final_cur_b": final_cur,
        "baseline_peak_b": baseline_peak,
        "final_peak_b": final_peak,
        "cur_delta_b": cur_delta,
        "peak_delta_b": peak_delta,
        "leak": leak,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_table(rows: list[tuple[str, float]]) -> str:
    width = max(len(label) for label, _ in rows)
    lines = []
    for label, val in rows:
        lines.append(f"  {label.ljust(width)}  {val:7.1f} MB")
    return "\n".join(lines)


def main() -> int:
    proc = psutil.Process()
    print("=" * 68)
    print("isotope_zero — RSS + tracemalloc zero-leak benchmark")
    print("=" * 68)

    # Probe 1: RSS
    print("\n[1] Cold-start RSS + RSS at 10k cards (dim=8)\n")
    rss = probe_rss(proc)
    print(_fmt_table(rss["rows"]))
    print(f"\n  cold-start RSS (after open store): {rss['rss_cold_mb']:.1f} MB")
    print(f"  RSS at 10k cards:                   {rss['rss_at_10k_mb']:.1f} MB")
    print(f"  RSS at 10k after vec cache:         {rss['rss_after_cache_mb']:.1f} MB")

    rss_fail = False
    if rss["rss_cold_mb"] > _RSS_COLD_FAIL_MB:
        print(f"  FAIL: cold-start RSS {rss['rss_cold_mb']:.1f} MB > "
              f"{_RSS_COLD_FAIL_MB:.0f} MB ceiling")
        rss_fail = True
    if rss["rss_at_10k_mb"] > _RSS_AT_10K_FAIL_MB:
        print(f"  FAIL: RSS at 10k {rss['rss_at_10k_mb']:.1f} MB > "
              f"{_RSS_AT_10K_FAIL_MB:.0f} MB ceiling")
        rss_fail = True
    # NOTE: the documented ~15 MB target is the bare-store floor. With the
    # Python 3.14 interpreter + numpy + psutil resident, the observed floor is
    # higher. We report the ACTUAL number (per the task instructions: do not
    # fudge). The target is surfaced as information, not a hard gate.

    # Probe 2: leak
    print("\n[2] tracemalloc zero-leak (100k add-then-delete cycles)\n")
    leak = probe_leak()
    print(f"  baseline traced current : {leak['baseline_cur_b']/1024:.1f} KB")
    print(f"  final    traced current : {leak['final_cur_b']/1024:.1f} KB")
    print(f"  baseline traced peak    : {leak['baseline_peak_b']/1024:.1f} KB")
    print(f"  final    traced peak     : {leak['final_peak_b']/1024:.1f} KB")
    print(f"  current delta vs baseline: {leak['cur_delta_b']/1024:.1f} KB")
    print(f"  peak    delta vs baseline: {leak['peak_delta_b']/1024:.1f} KB")
    if leak["leak"]:
        print("  FAIL: monotonic memory growth detected (current did not "
              "return to baseline)")
    else:
        print("  PASS: no monotonic growth (current within tolerance of baseline)")

    # Verdict
    print("\n" + "=" * 68)
    if rss_fail or leak["leak"]:
        print("RESULT: FAIL")
        print("=" * 68)
        return 1
    print("RESULT: PASS")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
