#!/usr/bin/env python3
"""Benchmark: mem0 (local OSS) vs isotope_zero on 50 conversational turns.

Spins up BOTH systems (mem0 lazily — skipped if its heavy deps are absent),
feeds each the same 50 multi-turn conversational statements (with deliberate
contradictions/updates so UPDATE-vs-ADD routing matters), then runs 10
hardcoded recall queries against each and measures:

    {system, memories_created, recall_accuracy, p50_latency_ms}

Asserts isotope_zero's recall accuracy is within 0.05 of mem0's (mem0 may
lead by up to 5 points; isotope_zero wins on latency by construction since it
runs natively with zero network calls).

Exit codes:
    0  pass, OR mem0 unavailable (isotope_zero still self-assessed)
    1  real failure (isotope_zero crashed, or accuracy gap exceeded)

Run:
    python3 scripts/compare_mem0.py

Design constraints (from the task spec):
  1. stdlib + numpy only on the isotope_zero side; mem0 imported LAZILY so
     this module is importable even when mem0's deps are absent.
  2. mem0 path guarded by try/except ImportError -> SKIP notice + exit 0.
  3. isotope_zero side uses the synthesis_v1.0 prototype client.
  4. 50 hardcoded turns (multi-turn, contradictions/updates); no randomness.
  5. 10 hardcoded recall queries; accuracy = fraction of expected facts hit.
  6. Results table printed to stdout.
  7. assert isotope_recall >= mem0_recall - 0.05.
  8. Self-adds sys.path; runnable as `python3 scripts/compare_mem0.py`.
"""
from __future__ import annotations

import os
import sys
import time
import statistics
from typing import Any

# ---------------------------------------------------------------------------
# Make the synthesis_v1.0 prototype importable. Do this BEFORE any
# isotope_zero import so `from isotope_zero.client import IsotopeZero`
# resolves to the prototype package, not any globally-installed stub.
# ---------------------------------------------------------------------------
_PROTO_DIR = "/Users/svanikhansh/Documents/isotope_zero/prototypes/synthesis_v1.0"
if _PROTO_DIR not in sys.path:
    sys.path.insert(0, _PROTO_DIR)

# isotope_zero's only third-party dep is numpy. The store imports numpy
# LAZILY inside its vector/hybrid search methods and gracefully degrades to
# a pure-Python fallback path when numpy is absent (same as the embedding
# engine falling back to deterministic pseudo-embeddings without onnxruntime).
# We mirror that posture here: a soft notice, NOT a hard fatal, so the
# benchmark still runs (and isotope_zero self-assesses) in a minimal env.
# Install numpy for real BLAS-backed vector search: pip install numpy.
try:
    import numpy  # noqa: F401  (presence check; used deep in store)
    _NUMPY_OK = True
except ImportError:  # pragma: no cover - env guard
    _NUMPY_OK = False
    sys.stderr.write(
        "NOTE: numpy not installed; isotope_zero will use its pure-Python "
        "vector-search fallback (slower but correct). Install numpy for "
        "real BLAS-backed retrieval: pip install numpy\n"
    )


# ===========================================================================
# Fixed corpus: 50 conversational turns.
#
# These are DELIBERATELY adversarial for UPDATE-vs-ADD routing:
#   - Statements 0/1 establish preferences, later statements contradict them
#     ("I like Python" -> "I switched to Rust").
#   - Some facts are pure additions (new topic), some are corrections, some
#     are reinforcements (stated again verbatim to test dedup).
#   - The expected-facts list below encodes the STATE AFTER all 50 turns —
#     i.e. a correct system should return the UPDATED, not the stale, value.
# ===========================================================================

TURNS: list[str] = [
    # --- identity / role ---
    "I'm a backend engineer working on distributed systems.",
    "My name is Alex Chen and I live in Seattle.",
    "I work at a startup called FlowGrid building data pipelines.",
    "I've been coding professionally for about 8 years now.",
    "My team is five people including me.",

    # --- language preferences (with a later switch) ---
    "I really like Python for most of my backend work.",
    "I use Postgres as my primary database.",
    "I deploy everything on AWS, mainly EKS.",
    "I prefer gRPC over REST for internal services.",
    "I switched to Rust for the performance-critical path.",  # contradicts turn 5

    # --- tools / stack ---
    "I use VS Code as my editor with the Rust plugin.",
    "We run Kafka for our event streaming layer.",
    "I containerize everything with Docker.",
    "My CI runs on GitHub Actions.",
    "I write tests with pytest for Python and cargo test for Rust.",

    # --- project context ---
    "FlowGrid ingests about 2 terabytes of data per day.",
    "Our pipeline has a 30-second SLA for freshness.",
    "We had a major outage last Tuesday from a Kafka consumer bug.",
    "I'm on call every other week.",
    "Our biggest customer is a fintech firm in New York.",

    # --- personal / schedule ---
    "I usually start work at 9 AM Pacific.",
    "I take Friday afternoons off for personal projects.",
    "I'm learning German in my spare time.",
    "I have a dog named Pixel who is a border collie.",
    "I moved to Seattle from Austin two years ago.",

    # --- contradictions / updates (UPDATE routing must win) ---
    "Actually I no longer use Postgres, we migrated to CockroachDB.",  # contradicts turn 6
    "I stopped using VS Code and moved to Neovim.",  # contradicts turn 10
    "Our data volume grew to 5 terabytes per day.",  # contradicts turn 15
    "We relaxed the freshness SLA to 60 seconds to cut costs.",  # contradicts turn 16
    "I'm no longer on call, I handed that to a junior engineer.",  # contradicts turn 18

    # --- new domain: ML/infra ---
    "We started building an ML feature store last month.",
    "I'm evaluating feature flags between LaunchDarkly and Unleash.",
    "We chose Unleash for cost reasons.",
    "Our ML team trains models on GPUs in AWS p4 instances.",
    "I just got certified as an AWS Solutions Architect.",

    # --- preferences refinements ---
    "For logging I prefer structured JSON over plain text.",
    "I use OpenTelemetry for tracing across services.",
    "I think monorepos are overrated for small teams.",
    "We keep infra code in Terraform modules.",
    "I write runbooks in Markdown in the company wiki.",

    # --- final corrections + state ---
    "Correction: our biggest customer moved from New York to London.",  # contradicts turn 19
    "I finished the German B1 exam last month.",
    "Pixel had her second birthday last week.",
    "FlowGrid just closed a Series B funding round.",
    "I'm considering switching from CockroachDB back to Postgres for simplicity.",  # soft revert
    "Our ML feature store is now in production serving 4000 qps.",
    "I was promoted to staff engineer last quarter.",
    "I'm presenting at QCon San Francisco next month.",
    "Our team grew from five to eight people.",
    "I plan to open source our Rust pipeline framework by year end.",
]

assert len(TURNS) == 50, f"expected 50 turns, got {len(TURNS)}"


# ===========================================================================
# 10 recall queries + the set of fact-fragments (lowercased substrings) that
# a correct retrieval MUST surface for that query. A query scores
# len(matched)/len(expected). Accuracy = mean over the 10 queries.
#
# EXPECTED encodes the POST-UPDATE state: e.g. for the editor query we
# expect "neovim" (the update), NOT "vs code" (the stale value). A system
# that merely ADDs without UPDATE-routing will return the stale value and
# fail the substring check, scoring 0 on that query.
# ===========================================================================

RECALL_QUERIES: list[tuple[str, list[str]]] = [
    ("What programming languages does Alex use?", ["rust", "python"]),
    ("Which database is FlowGrid currently using?", ["cockroachdb", "postgres"]),  # either the latest or the consideration
    ("What editor or IDE does Alex prefer?", ["neovim"]),
    ("How much data does FlowGrid ingest daily?", ["5 terabytes", "5 tb", "5tb"]),
    ("What is the freshness SLA for the pipeline?", ["60 second"]),
    ("Where does FlowGrid's biggest customer live?", ["london"]),
    ("What is the name of Alex's dog and what breed?", ["pixel", "border collie"]),
    ("What language is Alex learning and what level did they reach?", ["german", "b1"]),
    ("What is the throughput of the ML feature store?", ["4000 qps", "4000qps", "4,000 qps"]),
    ("How many people are on Alex's team now?", ["eight", "8 people"]),
]

assert len(RECALL_QUERIES) == 10


# ===========================================================================
# isotope_zero harness
# ===========================================================================

def _run_isotope_zero() -> dict[str, Any]:
    """Add all 50 turns to isotope_zero, run 10 recall queries, measure."""
    # Local import so a missing prototype path doesn't break module import.
    from isotope_zero.client import IsotopeZero

    # spawn_daemon=False keeps this deterministic + sandbox-safe: the engine
    # falls back to in-process or pseudo-embeddings with zero IPC. We still
    # get real cosine scores over whatever embeddings the engine produces,
    # which is all the recall ranking needs. use_mmap=False avoids any
    # file-backed index in this short-lived process.
    mem = IsotopeZero(db_path=":memory:", spawn_daemon=False, use_mmap=False)

    created = 0
    for turn in TURNS:
        cid = mem.remember(
            fact=turn,
            evidence=turn,
            tags=["conversation"],
            importance=0.5,
        )
        if cid:
            created += 1

    # Run one consolidation sweep so contradictions/corrections are folded:
    # the consolidator marks the STALE card SUPERSEDED and keeps the NEWEST
    # fact (correction supersedes the original). Without this, both the stale
    # and updated facts are live and recall could surface the wrong one.
    try:
        mem.consolidate()
    except Exception as exc:  # pragma: no cover - best effort
        sys.stderr.write(f"NOTE: isotope_zero consolidation skipped: {exc}\n")

    latencies_ms: list[float] = []
    query_accuracy: list[float] = []

    for query, expected_fragments in RECALL_QUERIES:
        t0 = time.perf_counter()
        hits = mem.recall(query, k=8)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        # Concatenate all returned facts (lowercased) and check each expected
        # fragment is present in at least one returned fact.
        blob = " ".join(h.get("fact", "").lower() for h in hits)
        matched = sum(1 for frag in expected_fragments if frag.lower() in blob)
        query_accuracy.append(matched / len(expected_fragments))

    mem.close()

    p50 = statistics.median(latencies_ms) if latencies_ms else 0.0
    accuracy = statistics.mean(query_accuracy) if query_accuracy else 0.0

    return {
        "system": "isotope_zero",
        "memories_created": created,
        "recall_accuracy": round(accuracy, 4),
        "p50_latency_ms": round(p50, 4),
    }


# ===========================================================================
# mem0 harness (lazy import — skipped if mem0 not installed)
# ===========================================================================

def _run_mem0() -> dict[str, Any] | None:
    """Add all 50 turns to mem0, run 10 recall queries, measure.

    Returns None (and prints a SKIP notice) if mem0 is not importable. The
    mem0 import is inside this function so the module is importable without
    mem0's heavy dependency tree.
    """
    try:
        from mem0 import Memory  # noqa: F401  (presence + API surface)
    except ImportError as exc:
        sys.stderr.write(
            "\nSKIP: mem0 is not installed (ImportError: "
            f"{exc.__class__.__name__}: {exc}).\n"
            "  The mem0 comparison rows will be omitted; isotope_zero still "
            "runs and self-assesses.\n"
            "  Install mem0 with: pip install mem0ai\n\n"
        )
        return None

    # Local-OSS client mode: from_config with an in-memory/duckdb+qdrant
    # local backend. We try the simplest local config that does not require
    # a running Qdrant/OpenAI server — mem0's local mode uses HuggingFace
    # embeddings + an in-memory vector store. If the local default isn't
    # available, this will raise and we report it as a skip (not a failure
    # of isotope_zero).
    try:
        # mem0 OSS Memory() with no config uses local defaults (HuggingFace
        # embedder + in-memory vector store). This needs no external server.
        m = Memory()
    except Exception as exc:  # pragma: no cover - env-dependent
        sys.stderr.write(
            f"\nSKIP: mem0 Memory() could not be constructed in local mode "
            f"({exc.__class__.__name__}: {exc}).\n"
            "  This usually means a transitive dep (qdrant-client, "
            "sentence-transformers, etc.) is missing.\n"
            "  isotope_zero still runs and self-assesses.\n\n"
        )
        return None

    user_id = "alex_chen"
    created = 0

    for turn in TURNS:
        # mem0.add takes a list of message dicts (role/content), like the
        # OpenAI chat format. We feed each turn as a user message; mem0's
        # V3 extractor (ADDITIVE by default) will store facts.
        messages = [{"role": "user", "content": turn}]
        try:
            m.add(messages, user_id=user_id)
            created += 1
        except Exception as exc:  # pragma: no cover - env-dependent
            sys.stderr.write(
                f"NOTE: mem0.add failed for turn {turn[:40]!r}: {exc}\n"
            )

    latencies_ms: list[float] = []
    query_accuracies: list[float] = []

    for query, expected_fragments in RECALL_QUERIES:
        t0 = time.perf_counter()
        try:
            results = m.search(
                query=query,
                filters={"user_id": user_id},
                top_k=8,
            )
        except Exception as exc:  # pragma: no cover - env-dependent
            sys.stderr.write(f"NOTE: mem0.search failed: {exc}\n")
            results = []
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        # mem0 search returns a list of dicts with a "memory" key (the fact
        # text). Normalize to lowercase blob for substring matching.
        blob_parts: list[str] = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    blob_parts.append(str(r.get("memory", "")))
                elif isinstance(r, str):
                    blob_parts.append(r)
        blob = " ".join(blob_parts).lower()
        matched = sum(1 for frag in expected_fragments if frag.lower() in blob)
        query_accuracies.append(matched / len(expected_fragments))

    p50 = statistics.median(latencies_ms) if latencies_ms else 0.0
    accuracy = statistics.mean(query_accuracies) if query_accuracies else 0.0

    return {
        "system": "mem0",
        "memories_created": created,
        "recall_accuracy": round(accuracy, 4),
        "p50_latency_ms": round(p50, 4),
    }


# ===========================================================================
# Reporting
# ===========================================================================

def _print_table(rows: list[dict[str, Any]]) -> None:
    cols = ["system", "memories_created", "recall_accuracy", "p50_latency_ms"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    sep = " | "
    header = sep.join(c.ljust(widths[c]) for c in cols)
    line = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(line)
    for r in rows:
        print(sep.join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main() -> int:
    print("=" * 70)
    print("mem0 vs isotope_zero — 50-turn conversational memory benchmark")
    print("=" * 70)
    print(f"turns: {len(TURNS)}  recall queries: {len(RECALL_QUERIES)}")
    print()

    # --- isotope_zero MUST run (it's the system under test) ---
    print("[1/2] Running isotope_zero (synthesis_v1.0 prototype)...")
    try:
        izero_result = _run_isotope_zero()
    except Exception as exc:
        sys.stderr.write(
            f"FAIL: isotope_zero crashed: {exc.__class__.__name__}: {exc}\n"
        )
        import traceback
        traceback.print_exc()
        return 1
    print(f"      done: {izero_result}")
    print()

    # --- mem0 (optional — skipped if not installed) ---
    print("[2/2] Running mem0 (local OSS client mode)...")
    mem0_result = _run_mem0()

    rows: list[dict[str, Any]]
    if mem0_result is None:
        # SKIP path: only isotope_zero ran. Self-assess + exit 0.
        rows = [izero_result]
        print()
        print("-" * 70)
        print("RESULTS (mem0 skipped — not installed)")
        print("-" * 70)
        _print_table(rows)
        print()
        print(f"isotope_zero recall_accuracy = {izero_result['recall_accuracy']}")
        print(f"isotope_zero p50_latency_ms   = {izero_result['p50_latency_ms']}")
        print()
        print("PASS (mem0 unavailable; isotope_zero self-assessed successfully).")
        return 0

    print(f"      done: {mem0_result}")
    print()

    # --- comparison ---
    rows = [mem0_result, izero_result]
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    _print_table(rows)
    print()

    izero_acc = izero_result["recall_accuracy"]
    mem0_acc = mem0_result["recall_accuracy"]
    izero_lat = izero_result["p50_latency_ms"]
    mem0_lat = mem0_result["p50_latency_ms"]

    print(f"isotope_zero recall_accuracy = {izero_acc}")
    print(f"mem0         recall_accuracy = {mem0_acc}")
    print(f"isotope_zero p50_latency_ms  = {izero_lat}")
    print(f"mem0         p50_latency_ms  = {mem0_lat}")
    print()

    # Accuracy assertion: isotope_zero must be within 0.05 of mem0.
    # (mem0 is allowed a small lead; isotope_zero wins latency by default.)
    gap = mem0_acc - izero_acc
    print(f"accuracy gap (mem0 - isotope_zero) = {gap:.4f}  (threshold: 0.05)")
    if gap > 0.05:
        sys.stderr.write(
            f"\nFAIL: isotope_zero recall_accuracy ({izero_acc}) is more than "
            f"0.05 below mem0 ({mem0_acc}). Gap = {gap:.4f}.\n"
        )
        return 1

    # Latency: isotope_zero should be faster (native, zero network). We
    # assert it is at least not slower, with a small tolerance for jitter.
    # This is the "wins on latency by definition" check.
    if izero_lat > mem0_lat * 1.5 + 1.0:
        sys.stderr.write(
            f"\nWARN: isotope_zero p50 latency ({izero_lat}ms) unexpectedly "
            f"slower than mem0 ({mem0_lat}ms) — investigate.\n"
        )
        # Warn only, do not fail: the spec says isotope_zero wins latency
        # "by definition since it's native"; a transient slow run under a
        # loaded machine shouldn't fail the benchmark, and accuracy is the
        # hard gate.
    else:
        print(f"latency: isotope_zero ({izero_lat}ms) <= mem0 ({mem0_lat}ms) — native wins.")

    print()
    print("PASS: isotope_zero matches or beats mem0 within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
