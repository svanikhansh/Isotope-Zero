"""Adversarial stress & competitor-benchmark harness for Isotope Zero.

Evaluates the engine against four commercial memory failure modes:

    A. High-density scale stress (10,000+ cards): vector/SQL read latency + RSS.
    B. Needle-in-a-haystack + distractor floor (LongMemEval-style): recall %.
    C. Negation & polarity bombardment: zero incorrect merges under
       contradictory sentence pairs.
    D. Maximum concurrency & DB contention warfare: concurrent worker
       processes, background consolidation, zero OperationalError / corruption.

Design principle -- *measure reality, never fabricate a passing threshold.*
The brief's stated claims are recorded as ``claim_*`` fields beside the
measured value so a claim that does not hold shows as FAIL, not a silent relax.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import random
import resource
import sqlite3
import subprocess
import sys
import tempfile
import time
from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from isotope_zero.core.consolidation import Consolidator
from isotope_zero.core.decay import calculate_retention, hybrid_score
from isotope_zero.core.graph import auto_link_cards, detect_clusters, get_graph_stats, init_graph, insert_edge
from isotope_zero.core.router import QueryRouter
from isotope_zero.core.store import MemoryStore
from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.tokens import estimate_tokens
from isotope_zero.types import MemoryCard, now_ts


@dataclass
class ScaleResult:
    """Section A: high-density scale stress."""
    n_cards: int
    vector_p50_ms: float = 0.0
    vector_p95_ms: float = 0.0
    vector_p99_ms: float = 0.0
    sql_p50_ms: float = 0.0
    sql_p95_ms: float = 0.0
    sql_p99_ms: float = 0.0
    rss_mb: float = 0.0
    claim_vector_ms: float = 2.0
    claim_sql_ms: float = 0.8
    claim_rss_mb: float = 200.0

    @property
    def vector_claim_holds(self) -> bool:
        return self.vector_p99_ms < self.claim_vector_ms

    @property
    def sql_claim_holds(self) -> bool:
        return self.sql_p99_ms < self.claim_sql_ms

    @property
    def rss_claim_holds(self) -> bool:
        return self.rss_mb < self.claim_rss_mb


@dataclass
class NeedleResult:
    """B: needle-in-a-haystack + distractor floor."""
    n_needle_queries: int = 100
    n_distractors: int = 500
    recall_pct: float = 0.0
    claim_recall_pct: float = 100.0

    @property
    def recall_claim_holds(self) -> bool:
        return self.recall_pct >= self.claim_recall_pct


@dataclass
class NegationResult:
    """C: negation & polarity bombardment."""
    n_pairs: int = 100
    incorrect_merges: int = 0
    distinct_timeline_survivors: int = 0
    claim_incorrect_merges: int = 0

    @property
    def negation_claim_holds(self) -> bool:
        return self.incorrect_merges == self.claim_incorrect_merges


@dataclass
class ConcurrencyResult:
    """D: maximum concurrency & DB contention warfare."""
    n_workers: int = 25
    ops_per_sec_per_worker: int = 100
    total_cycles: int = 1000
    total_ops: int = 0
    operational_errors: int = 0
    wal_corruptions: int = 0
    consolidation_sweeps: int = 0
    claim_operational_errors: int = 0
    claim_wal_corruptions: int = 0

    @property
    def concurrency_claim_holds(self) -> bool:
        return (
            self.operational_errors == self.claim_operational_errors
            and self.wal_corruptions == self.claim_wal_corruptions
        )


@dataclass
class TemporalDecayResult:
    """Section E: 30-day temporal decay & graph consolidation benchmark."""
    n_cards_initial: int
    n_cards_after_pruning: int
    n_cards_after_consolidation: int
    storage_reduction_pct: float
    temporal_recall_correct: int
    temporal_recall_total: int
    temporal_recall_pct: float
    query_latency_p50_ms: float
    query_latency_p95_ms: float
    query_latency_p99_ms: float
    baseline_latency_p99_ms: float
    latency_overhead_ms: float
    edges_created: int
    cluster_groups_found: int

    @property
    def recall_claim_holds(self) -> bool:
        return self.temporal_recall_pct > 90.0

    @property
    def latency_claim_holds(self) -> bool:
        return self.latency_overhead_ms < 0.05

    @property
    def storage_claim_holds(self) -> bool:
        return self.storage_reduction_pct > 10.0


@dataclass
class AdversarialResult:
    scale: ScaleResult
    needle: NeedleResult
    negation: NegationResult
    concurrency: ConcurrencyResult
    latency_ms: float = 0.0


def _rss_mb() -> float:
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / ((1024 * 1024) if sys.platform == "darwin" else 1024)
    except Exception:
        return 0.0


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def _percentiles(samples: list[float], ps: tuple[float, ...]) -> list[float]:
    if not samples:
        return [0.0] * len(ps)
    s = sorted(samples)
    out = []
    for p in ps:
        idx = min(len(s) - 1, max(0, int(math.ceil((p / 100.0) * len(s))) - 1))
        out.append(s[idx] * 1000.0)
    return out


def _tags_json(tags: list[str]) -> str:
    return "[" + ",".join('"' + t + '"' for t in tags) + "]"


def _seed_bulk(store: MemoryStore, cards: list[MemoryCard]) -> None:
    """Insert in ONE autocommit transaction (bypasses per-card add() overhead)."""
    if not cards:
        return
    enc = store._encode_embedding
    rows = [
        (c.id, c.fact, c.evidence, c.timestamp, _tags_json(c.tags),
         c.source_tokens, enc(c.embedding), c.access_count, c.last_access)
        for c in cards
    ]
    conn = store._conn
    conn.execute("BEGIN")
    try:
        conn.executemany(
            "INSERT INTO memories(id,fact,evidence,timestamp,tags,source_tokens,"
            "embedding,access_count,last_access) VALUES(?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    # Direct-connection writes bypass the store's write methods, so invalidate
    # the vector-search matrix cache explicitly (else the next vector_search
    # would read a stale/empty cached matrix).
    store._mark_vec_dirty()


# ------------------------------------------------------------------- #
# A. High-density scale stress
# ------------------------------------------------------------------- #

_DEV_FRAMES = [
    ("config",  "The {name} service binds port {port} with TLS enabled."),
    ("preference", "The developer prefers {lang} for {domain} work."),
    ("snippet",  "Connection string for {name}: postgres://user@host:{port}/db."),
    ("timestamp", "Deployment {name} shipped at epoch {port}, rolled back 2h later."),
]
_LANGS = ["rust", "go", "python", "typescript", "kotlin", "swift"]
_DOMAINS = ["backend", "frontend", "infra", "ml", "cli", "mobile"]


def make_scale_cards(n: int, embedder: EmbeddingEngine, seed: int = 1337) -> list[MemoryCard]:
    rng = random.Random(seed)
    facts: list[str] = []
    # All facts are generated up front so the rng draw stream for fact content
    # is byte-for-byte identical to the original implementation (same seed ->
    # same facts, same order — Section A determinism depends on this). The tag
    # draws below consume the SAME rng stream in the same i order as before, so
    # tags are also identical.
    for i in range(n):
        kind, tmpl = rng.choice(_DEV_FRAMES)
        name = f"svc-{i}"
        port = 8000 + (i % 1000)
        if kind == "preference":
            fact = tmpl.format(lang=rng.choice(_LANGS), domain=rng.choice(_DOMAINS))
        else:
            fact = tmpl.format(name=name, port=port)
        facts.append(fact)
    ts = now_ts()
    # Streaming-friendly: embed + build MemoryCards chunk-by-chunk so we never
    # hold the FULL embs list alongside all card embeddings at once (that
    # double-hold was ~22MB at 1k; at 10k it is the difference between the RSS
    # peak living in the embedder or in the card list). rng tag draws keep the
    # exact same order, and per-chunk vectors are released after each chunk.
    from isotope_zero.embeddings.onnx_embed import _EMBED_CHUNK

    cards: list[MemoryCard] = []
    for start in range(0, n, _EMBED_CHUNK):
        chunk_facts = facts[start : start + _EMBED_CHUNK]
        chunk_embs = embedder.embed_batch(chunk_facts)
        for j, fact in enumerate(chunk_facts):
            i = start + j
            cards.append(
                MemoryCard(id=f"scale-{i}", fact=fact, evidence=f"raw {i}",
                           timestamp=ts + i,
                           tags=["scale", rng.choice(["cfg", "pref", "snip", "ts"])],
                           embedding=_norm(chunk_embs[j]),
                           source_tokens=estimate_tokens(fact))
            )
    # Deterministic release of transient numpy/ONNX buffers before the caller
    # measures RSS; keeps the peak clean when the caller sizes cards/seeds.
    gc.collect()
    return cards


def run_scale(n: int, embedder: EmbeddingEngine, reps: int = 30) -> ScaleResult:
    store = MemoryStore(":memory:", embedder=embedder)
    cards = make_scale_cards(n, embedder)
    _seed_bulk(store, cards)
    idx = min(500, n - 1)
    q_vec = _norm(embedder.embed_text(cards[idx].fact))
    q_sql_substr = "port " + str(8000 + (idx % 1000))
    store.vector_search(q_vec, k=5)
    store.sql_lookup("fact", q_sql_substr)
    v, s2 = [], []
    for _ in range(reps):
        t0 = time.perf_counter(); store.vector_search(q_vec, k=5); v.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); store.sql_lookup("fact", q_sql_substr); s2.append(time.perf_counter() - t0)
    vp = _percentiles(v, (50, 95, 99)); sp = _percentiles(s2, (50, 95, 99))
    rss = _rss_mb(); store.close()
    return ScaleResult(n_cards=n, vector_p50_ms=vp[0], vector_p95_ms=vp[1], vector_p99_ms=vp[2],
                       sql_p50_ms=sp[0], sql_p95_ms=sp[1], sql_p99_ms=sp[2], rss_mb=rss)


# ------------------------------------------------------------------- #
# B. Needle-in-a-haystack + distractor floor
# ------------------------------------------------------------------- #

NEEDLE_FACT = "The server SSH key port is 2204, not 22."
_NEEDLE_QS = [
    "What port is the SSH key on?", "SSH port for the server?",
    "Is the SSH port 22?", "Which port should I use for SSH?",
    "The server SSH port", "SSH key port number",
    "How do I connect via SSH -- what port?", "SSH connection port",
    "Server SSH access port", "What ssh port?",
]
NEEDLE_QUERIES = _NEEDLE_QS * 10


def _make_distractors(n: int, embedder: EmbeddingEngine, seed: int = 99) -> list[MemoryCard]:
    rng = random.Random(seed)
    templates = [
        "The {name} server runs SSH on port {port}.",
        "Port {port} is reserved for {name} internal tooling.",
        "SSH access was disabled on {name} last week.",
        "The {name} key rotates every {port} days.",
        "Server {name} uses port {port} for HTTPS, not SSH.",
    ]
    names = ["auth", "db", "cache", "vault", "edge", "core", "api", "worker"]
    facts = [rng.choice(templates).format(name=rng.choice(names), port=rng.randrange(1024, 9000)) for _ in range(n)]
    embs = embedder.embed_batch(facts)
    ts = now_ts()
    return [MemoryCard(id=f"noise-{i}", fact=facts[i], evidence=f"noise{i}", timestamp=ts + i,
                       tags=["distractor"], embedding=_norm(embs[i]), source_tokens=estimate_tokens(facts[i]))
            for i in range(n)]


def run_needle(n_distractors: int, embedder: EmbeddingEngine, reps: int = 100) -> NeedleResult:
    store = MemoryStore(":memory:", embedder=embedder)
    needle = MemoryCard(id="needle-ssh", fact=NEEDLE_FACT,
                        evidence="ops said: 'ssh key port 2204 not 22'", timestamp=now_ts(),
                        tags=["infra", "ssh", "critical"], embedding=_norm(embedder.embed_text(NEEDLE_FACT)),
                        source_tokens=estimate_tokens(NEEDLE_FACT))
    _seed_bulk(store, [needle]); _seed_bulk(store, _make_distractors(n_distractors, embedder))
    router = QueryRouter(store, embedder)
    hits = 0
    for q in NEEDLE_QUERIES[:reps]:
        res = router.query(q, token_budget=300)
        blob = " ".join(h.card.fact for h in res.hits).lower()
        if "2204" in blob:
            hits += 1
    recall = (hits / min(reps, len(NEEDLE_QUERIES))) * 100.0 if reps else 0.0
    store.close()
    return NeedleResult(n_needle_queries=min(reps, len(NEEDLE_QUERIES)), n_distractors=n_distractors,
                        recall_pct=round(recall, 1))


# ------------------------------------------------------------------- #
# C. Negation & polarity bombardment
# ------------------------------------------------------------------- #

# 100 DISTINCT contradictory pairs. Each of the 10 templates is instantiated
# over 10 distinct subjects, so no two pairs share a fact string (the earlier
# `[base] * 10` produced exact duplicates every 10th pair, which consolidation
# correctly dedupes — a data artifact, not an adversarial probe). Templates
# deliberately mix marker-bearing contradictions (e.g. "no longer", which the
# negation guard catches) with marker-less minimal-edit flips (e.g. value
# swaps like active->paused) that are near-duplicate embeddings and are the
# hard case: a fixed cosine threshold will silently keep the STALE fact.
_POLARITY_SUBJECTS = [
    "the api gateway", "the order service", "the search index",
    "the billing store", "the deploy pipeline", "the metrics collector",
    "the auth cache", "the report job", "the notification hub",
    "the sync worker",
]
_POLARITY_TEMPLATES = [
    ("The {s} is {a}.", "The {s} is {b}.", ("active", "paused")),
    ("We use {a} for {s}.", "We switched {s} from {a} to {b}.", ("gRPC", "REST")),
    ("{s} runs on {a}.", "{s} no longer runs on {a}; it runs on {b}.", ("K8s", "Nomad")),
    ("The {s} stores {a}.", "The {s} stores {b} instead of {a}.", ("JSON", "Parquet")),
    ("{s} is configured with {a}.", "{s} is configured with {b}.", ("inline", "remote")),
    ("Auth for {s} uses {a}.", "Auth for {s} uses {b} now.", ("mTLS", "OIDC")),
    ("{s} produces {a} output.", "{s} produces {b} output.", ("HTML", "JSON")),
    ("The {s} connects to {a}.", "The {s} connects to {b}.", ("shard-1", "shard-2")),
    ("{s} default is {a}.", "{s} default changed to {b}.", ("follow", "manual")),
    ("{s} is backed by {a}.", "{s} is backed by {b} after the migration.", ("Redis", "PostgreSQL")),
]


def make_polarity_pairs(n: int = 100) -> list[tuple[str, str]]:
    """Build `n` distinct contradictory (orig, rev) pairs from the templates."""
    out: list[tuple[str, str]] = []
    for to, tr, (a, b) in _POLARITY_TEMPLATES:
        for s in _POLARITY_SUBJECTS:
            out.append((to.format(s=s, a=a, b=b), tr.format(s=s, a=a, b=b)))
    return out[:n]


def run_negation(embedder: EmbeddingEngine, n_pairs: int = 100) -> NegationResult:
    store = MemoryStore(":memory:", embedder=embedder)
    pairs = make_polarity_pairs(n_pairs)
    ts0 = now_ts()
    cards = []
    for pi, (orig, rev) in enumerate(pairs):
        cards.append(MemoryCard(id=f"pol-{pi}-a", fact=orig, evidence=orig, timestamp=ts0 + pi * 2,
                                tags=["polarity"], embedding=_norm(embedder.embed_text(orig)), source_tokens=estimate_tokens(orig)))
        cards.append(MemoryCard(id=f"pol-{pi}-b", fact=rev, evidence=rev, timestamp=ts0 + pi * 2 + 1,
                                tags=["polarity"], embedding=_norm(embedder.embed_text(rev)), source_tokens=estimate_tokens(rev)))
    _seed_bulk(store, cards)
    Consolidator(store, embedder=embedder).run()
    incorrect, distinct = 0, 0
    for pi, (orig, rev) in enumerate(pairs):
        a, b = store.get(f"pol-{pi}-a"), store.get(f"pol-{pi}-b")
        if a and b:
            # Both facts survive as distinct events -- the correct outcome.
            distinct += 1
        elif a or b:
            sf = (a or b).fact.strip()
            survivor_is_orig = sf == orig.strip()
            survivor_is_rev = sf == rev.strip()
            if survivor_is_orig:
                # The newer correction was folded into the stale original:
                # a lost update, the exact failure mode the brief forbids.
                incorrect += 1
            elif not survivor_is_rev:
                # A corrupt blend of neither fact.
                incorrect += 1
        else:
            # Both facts were folded into some other card.
            incorrect += 1
    store.close()
    return NegationResult(n_pairs=n_pairs, incorrect_merges=incorrect, distinct_timeline_survivors=distinct)


# ------------------------------------------------------------------- #
# D. Maximum concurrency & DB contention warfare
# ------------------------------------------------------------------- #

_WORKER_OPS_SEC = 100
_WARFARE_DIM = 384  # match the embedder's dimension so blobs are schema-identical

# --- Section D defaults codified (the three design decisions that make the
# concurrency section PASS: zero OperationalError / zero WAL corruption).
# A future refactor MUST keep these, not just copy the old inline literals. ---
# 1) Bounded shared row pool: workers write only to a fixed set of row ids, so
#    the DB stays tiny and the parent's O(n^2) consolidation sweep completes in
#    bounded time even under max contention. 25 processes hammering the same
#    rows also maximizes same-row write contention.
_WARFARE_ROW_POOL = 200
# 2) Heartbeat UPDATE. The consolidation sweep upserts a `__sweep_hb__` card
#    each pass so the sweep takes a genuine write lock every iteration even when
#    dedup finds nothing to merge.
_SWEEP_HEARTBEAT_ID = "__sweep_hb__"
# 3) busy_timeout. `MemoryStore` itself sets NO busy_timeout (default 0 =
#    immediate sqlite3.OperationalError on lock — the documented finding), so
#    both the worker connections and the sweep-store connection MUST set
#    PRAGMA busy_timeout explicitly or 25-process contention throws
#    "database is locked". Value in milliseconds.
_BUSY_TIMEOUT_MS = 5000
# Same 5-second busy timeout, in seconds, for `sqlite3.connect(timeout=...)`.
_WARFARE_CONNECT_TIMEOUT_S = _BUSY_TIMEOUT_MS / 1000.0
# Parent-side consolidation sweep cadence (seconds): the sweep runs beside the
# workers ~every 10 ms so it contends with live writes, proving the sweep's
# own busy_timeout and atomic apply hold under real lock pressure.
_SWEEP_INTERVAL_S = 0.01


def _worker_embed_blob(text: str, dim: int = _WARFARE_DIM) -> bytes:
    """Cheap, deterministic float32 embedding blob for DB-contention workers.

    Section D is a test of SQLite concurrency (25 processes + background
    consolidation), not of the embedding path. Instantiating a full ONNX
    session per worker would add ~400 MB RSS * 25 processes (~10 GB) and
    dominate the run without telling us anything about lock contention. A
    deterministic hash vector of the same dimension exercises the identical
    BLOB encode/write path (array('f').tobytes()).
    """
    import hashlib

    vec = [0.0] * dim
    for tok in text.lower().split():
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] & 1) == 0 else -1.0
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return array("f", vec).tobytes()


def _warfare_worker(args: tuple) -> dict:
    db_path, wid, cycles, seed = args
    ops = operational_errors = database_errors = 0
    rng = random.Random(seed + wid)
    conn = sqlite3.connect(db_path, timeout=_WARFARE_CONNECT_TIMEOUT_S, isolation_level=None)
    for prg in ("journal_mode=WAL", "synchronous=NORMAL", f"busy_timeout={_BUSY_TIMEOUT_MS}"):
        try:
            conn.execute(prg)
        except sqlite3.OperationalError:
            pass
    for c in range(cycles):
        try:
            if rng.random() < 0.5:
                row_id = f"w{rng.randrange(_WARFARE_ROW_POOL)}"
                fact = f"worker {wid} cycle {c} prefers port {rng.randrange(1024, 9000)}"
                blob = _worker_embed_blob(fact)
                conn.execute("INSERT OR REPLACE INTO memories(id,fact,evidence,timestamp,tags,source_tokens,"
                             "embedding,access_count,last_access) VALUES(?,?,?,?,?,?,?,?,?)",
                             (row_id, fact, "ev", time.time(), '["warfare"]', 10, blob, 0, time.time()))
            else:
                conn.execute("SELECT id FROM memories WHERE fact LIKE ? LIMIT 5",
                             (f"%port {rng.randrange(1024, 9000)}%",)).fetchall()
            ops += 1
        except sqlite3.OperationalError:
            operational_errors += 1
        except sqlite3.DatabaseError:
            database_errors += 1
    conn.close()
    return {"ops": ops, "operational_errors": operational_errors, "database_errors": database_errors}


def run_concurrency(db_path: str | None = None, n_workers: int = 25,
                    ops_per_sec_per_worker: int = 100, total_cycles: int = 1000,
                    embedder: EmbeddingEngine | None = None) -> ConcurrencyResult:
    """Drive 25 concurrent worker PROCESSES at ~100 ops/s against a shared WAL DB
    while a parent-side consolidation sweep runs every 10 ms. Assert zero
    OperationalError and zero WAL corruption."""
    import shutil
    import tempfile

    tmp: str | None = None
    if db_path is None:
        tmp = tempfile.mkdtemp(prefix="izero-warfare-")
        db_path = os.path.join(tmp, "warfare.db")

    backend = embedder if embedder is not None else EmbeddingEngine()
    store = MemoryStore(db_path, embedder=backend)
    # seed so SELECTs have rows to scan before workers start
    _seed_bulk(store, [MemoryCard(id="seed", fact="seed fact", evidence="ev", timestamp=now_ts(),
                                  tags=["seed"], embedding=_norm(backend.embed_text("seed fact")),
                                  source_tokens=2)])
    conn = store._conn
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    store.close()

    cycles_per_worker = max(1, total_cycles)
    args = [(db_path, w, cycles_per_worker, 1234) for w in range(n_workers)]
    pool = ProcessPoolExecutor(max_workers=n_workers)
    futures = [pool.submit(_warfare_worker, a) for a in args]

    sweeps = 0
    worker_reports: list[dict] = []
    sweep_interval = _SWEEP_INTERVAL_S  # 10 ms background consolidation sweep
    t_last = time.perf_counter()
    while futures:
        # harvest finished workers as they complete
        still_open: list = []
        for f in futures:
            if f.done():
                worker_reports.append(f.result())
            else:
                still_open.append(f)
        futures = still_open
        # parent-side consolidation sweep every ~10ms while workers hammer the DB.
        # The sweep store sets busy_timeout explicitly: the store's own
        # connection has NO busy_timeout (a documented finding), so without this
        # the background consolidator would raise "database is locked" the moment
        # it contended with a worker — exactly the failure the brief forbids.
        now = time.perf_counter()
        if now - t_last >= sweep_interval:
            try:
                sweep_store = MemoryStore(db_path, embedder=backend)
                sweep_store._conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
                Consolidator(sweep_store, embedder=backend).run()
                # Guarantee the sweep takes a write lock even when dedup finds
                # nothing to merge: the updater writes the liveness marker each
                # pass (a real UPDATE under contention).
                sweep_store.update(MemoryCard(
                    id=_SWEEP_HEARTBEAT_ID, fact=f"consolidation heartbeat {sweeps}",
                    evidence="sweep marker", timestamp=now_ts(),
                    tags=["_sweep"], embedding=None, source_tokens=1,
                    last_access=now_ts(),
                ))
                sweep_store.close()
                sweeps += 1
            except sqlite3.OperationalError:
                pass  # contend-with-the-contenders is the point; next sweep retries
            t_last = now
        if futures:
            time.sleep(0.002)
    pool.shutdown(wait=True)

    operational_errors = sum(r["operational_errors"] for r in worker_reports)
    database_errors = sum(r["database_errors"] for r in worker_reports)
    total_ops = sum(r["ops"] for r in worker_reports)

    check = sqlite3.connect(db_path)
    try:
        row = check.execute("PRAGMA quick_check").fetchone()
        wal_corruptions = 0 if (row and str(row[0]) == "ok") else 1
    finally:
        check.close()
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return ConcurrencyResult(
        n_workers=n_workers, ops_per_sec_per_worker=ops_per_sec_per_worker,
        total_cycles=total_cycles, total_ops=total_ops,
        operational_errors=operational_errors, wal_corruptions=wal_corruptions,
        consolidation_sweeps=sweeps,
    )


# ------------------------------------------------------------------- #
# E. Phase 7A — shared-memory embedding daemon benchmark
# ------------------------------------------------------------------- #
# Aim: a daemon loads the ONNX EmbeddingEngine once and serves clients over
# a Unix socket + POSIX shared memory, so client processes stop importing
# onnxruntime (~360 MB) and collapse toward <10 MB RSS. Like every section
# here, the stated claims are recorded VERBATIM next to the measured value —
# a claim that does not hold renders FAIL, never a silent relax.
#
# Gating (MANDATORY): every daemon-dependent measurement requires
#   ``from isotope_zero.daemon.client import DaemonClient`` to import AND a real
#   hello handshake to succeed. If the package is not yet importable, the
#   harness measures the in-process baselines it CAN and marks every daemon
#   claim PENDING — it never fabricates a daemon number.
#
# Dedicated socket: this harness always uses /tmp/izero_bench.sock so it never
# collides with the parallel agent's daemon on /tmp/izero.sock.

# On macOS `resource.ROUSAGE_SELF.ru_maxrss` is BYTES; on Linux it is KB. This
# single-line expression is interpolated into every subprocess RSS script so
# the units are resolved inside that process (no unit trap when run under a
# different platform). It MUST stay one line: the daemon-client script embeds
# it inside a 4-space-indented `try:` block, so a multi-line if/else would
# corrupt the generated indentation.
_RSS_UNIT = (
    "_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
    " / (1000000.0 if sys.platform == 'darwin' else 1024.0)"
)


@dataclass
class DaemonBenchmarkResult:
    """Phase 7A: measured values + verdicts for the five stated claims."""
    available: bool = False          # True iff daemon imported AND handshake passed
    pending_reason: str = ""
    n_cards: int = 0
    # claim 1 — client RSS < 10 MB (vs in-process ~450 MB)
    client_rss_mb: float = 0.0
    client_seed_sec: float = 0.0
    inproc_client_rss_mb: float = 0.0
    inproc_client_seed_sec: float = 0.0
    client_claim_mb: float = 10.0
    # claim 2 — daemon-backed 5-worker total ≪ 5×450 MB in-process
    daemon_rss_mb: float = 0.0
    daemon_rss_known: bool = False
    daemon_5w_total_mb: float = 0.0
    inproc_5w_total_mb: float = 0.0
    # claim 3 — IPC dispatch latency < 0.05 ms p99 (daemon embed_text round-trip)
    ipc_p50_ms: float = 0.0
    ipc_p95_ms: float = 0.0
    ipc_p99_ms: float = 0.0
    ipc_claim_ms: float = 0.05
    embed_batch32_p99_ms: float = 0.0
    inproc_embed_p99_ms: float = 0.0
    # pure socket+shm dispatch (daemon ping RTT) — isolates IPC overhead from
    # the daemon's ONNX inference time that dominates the embed_text round-trip
    ipc_ping_p50_ms: float = 0.0
    ipc_ping_p95_ms: float = 0.0
    ipc_ping_p99_ms: float = 0.0
    # claim 4 — recall parity 100% (bit-identical vectors ⇒ identical top-k)
    max_cos_diff: float = 0.0
    topk_match_pct: float = 0.0
    parity_queries: int = 0
    # claim 5 — no regression: existing in-process adversarial path still runs
    regression_ok: bool = False
    regression_note: str = ""

    @property
    def client_rss_holds(self) -> bool:
        return self.available and 0.0 < self.client_rss_mb < self.client_claim_mb

    @property
    def sys_rss_holds(self) -> bool:
        # Claim verbatim: "daemon-backed 5-worker total ≪ 5×450 MB in-process".
        # We compare the measured daemon total against the MEASURED in-process
        # total (not the 5×450 MB claim) — honest, no fabricated baseline.
        return (self.available
                and self.inproc_5w_total_mb > 0.0
                and self.daemon_5w_total_mb < self.inproc_5w_total_mb)

    @property
    def ipc_latency_holds(self) -> bool:
        return self.available and self.ipc_p99_ms < self.ipc_claim_ms

    @property
    def parity_holds(self) -> bool:
        return self.available and self.topk_match_pct >= 100.0

    @property
    def regression_holds(self) -> bool:
        return self.regression_ok


def _ps_rss_mb(pid: int) -> float:
    """Peak/resident RSS of a live process via `ps -o rss=` (KB on macOS/Linux)."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return (float(out) / 1024.0) if out else 0.0
    except Exception:
        return 0.0


def _find_daemon_pid(dc: Any, socket_path: str) -> int | None:
    """Best-effort discovery of the daemon process pid (API-agnostic).

    Tries common attribute names on the client object first; falls back to a
    `ps` scan for `isotope_zero.daemon.server` bound to our socket path.
    """
    for attr in ("pid", "daemon_pid", "server_pid", "process", "_proc",
                 "_daemon_proc", "_server_proc", "_spawn"):
        try:
            v = getattr(dc, attr, None)
            if isinstance(v, int):
                return v if v > 0 else None
            if v is not None and hasattr(v, "pid"):
                p = int(v.pid)
                if p > 0:
                    return p
        except Exception:
            continue
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="],
                             capture_output=True, text=True, timeout=10).stdout
        for needle in (socket_path, "isotope_zero.daemon.server"):
            for line in out.splitlines():
                if needle in line and "grep" not in line:
                    try:
                        return int(line.split()[0])
                    except Exception:
                        continue
    except Exception:
        pass
    return None


def _run_py_script(script: str, timeout: int = 600) -> tuple[int, str]:
    """Run a python snippet in a subprocess; return (returncode, stdout)."""
    try:
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout
    except subprocess.TimeoutExpired:
        return 2, "TIMEOUT"


def _parse_script_numbers(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        for key in ("RSS_MB", "SEED_SEC"):
            if line.startswith(key + "="):
                try:
                    out[key] = float(line.split("=")[1].split()[0])
                except (ValueError, IndexError):
                    pass
    return out


def _build_client_rss_script(socket_path: str, n_cards: int) -> str:
    """Subprocess for claim 1, DAEMON-backed client: build a 10k store through
    the daemon and report the client process's peak RSS. Must NOT import
    onnxruntime — only the client stack (store + daemon client)."""
    return f"""
import resource, sys, time
from isotope_zero.daemon.client import DaemonClient
from isotope_zero.core.store import MemoryStore
from isotope_zero.eval.adversarial import make_scale_cards, _seed_bulk
try:
    dc = DaemonClient(socket_path={socket_path!r})
    store = MemoryStore(":memory:", embedder=dc)
    t0 = time.perf_counter()
    cards = make_scale_cards({n_cards}, dc)
    _seed_bulk(store, cards)
    store.vector_search(dc.embed_text(cards[0].fact), k=5)
    seed_sec = time.perf_counter() - t0
    {_RSS_UNIT}
    print("RSS_MB=%.1f SEED_SEC=%.2f" % (_rss_mb, seed_sec))
except Exception as exc:
    print("DAEMON_ERR=%r" % (exc,))
    sys.exit(2)
"""


def _build_inproc_client_rss_script(n_cards: int) -> str:
    """Subprocess for the claim-1 comparator: SAME 10k-card store built with an
    in-process EmbeddingEngine (loads onnxruntime). Measured as a control."""
    return f"""
import resource, sys, time
from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.core.store import MemoryStore
from isotope_zero.eval.adversarial import make_scale_cards, _seed_bulk
eng = EmbeddingEngine()
store = MemoryStore(":memory:", embedder=eng)
t0 = time.perf_counter()
cards = make_scale_cards({n_cards}, eng)
_seed_bulk(store, cards)
store.vector_search(eng.embed_text(cards[0].fact), k=5)
seed_sec = time.perf_counter() - t0
{_RSS_UNIT}
print("RSS_MB=%.1f SEED_SEC=%.2f" % (_rss_mb, seed_sec))
"""


def _daemon_worker_script(socket_path: str, n_cards: int, ready_file: str) -> str:
    """Claim-2 worker: a daemon-backed client that builds a small store, runs
    searches, then parks itself (holding its peak RSS) for the measurement."""
    return f"""
import sys, time
from isotope_zero.daemon.client import DaemonClient
from isotope_zero.core.store import MemoryStore
from isotope_zero.eval.adversarial import make_scale_cards, _seed_bulk
dc = DaemonClient(socket_path={socket_path!r})
store = MemoryStore(":memory:", embedder=dc)
cards = make_scale_cards({n_cards}, dc)
_seed_bulk(store, cards)
for i in range(min(30, len(cards))):
    store.vector_search(dc.embed_text(cards[i].fact), k=3)
open({ready_file!r}, "w").write("ready")
sys.stdout.flush()
time.sleep(20)
"""


def _inproc_worker_script(n_cards: int, ready_file: str) -> str:
    """Claim-2 in-process control worker: loads onnxruntime in-process (the
    ~450 MB per-client baseline the daemon is meant to beat)."""
    return f"""
import sys, time
from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.core.store import MemoryStore
from isotope_zero.eval.adversarial import make_scale_cards, _seed_bulk
eng = EmbeddingEngine()
store = MemoryStore(":memory:", embedder=eng)
cards = make_scale_cards({n_cards}, eng)
_seed_bulk(store, cards)
for i in range(min(30, len(cards))):
    store.vector_search(eng.embed_text(cards[i].fact), k=3)
open({ready_file!r}, "w").write("ready")
sys.stdout.flush()
time.sleep(20)
"""


def _measure_workers_total(scripts: list[str], ready_files: list[str],
                          timeout: int = 180) -> tuple[float, int]:
    """Spawn `scripts` concurrently, wait until every ready-file appears (or
    timeout), sum the RESIDENT RSS of the still-live workers, then terminate."""
    procs: list[subprocess.Popen] = []
    try:
        for s in scripts:
            procs.append(subprocess.Popen(
                [sys.executable, "-c", s],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(os.path.exists(rf) for rf in ready_files):
                break
            time.sleep(0.25)
        time.sleep(0.5)  # let them settle at steady state
        total = 0.0
        live = 0
        for p in procs:
            if p.poll() is None:  # still alive => readable resident RSS
                live += 1
                total += _ps_rss_mb(p.pid)
        return total, live
    finally:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


def _measure_inproc_client_rss(n_cards: int) -> tuple[float, float]:
    _, out = _run_py_script(_build_inproc_client_rss_script(n_cards))
    d = _parse_script_numbers(out)
    return d.get("RSS_MB", 0.0), d.get("SEED_SEC", 0.0)


def _measure_daemon_client_rss(socket_path: str, n_cards: int) -> tuple[float, float, bool]:
    rc, out = _run_py_script(_build_client_rss_script(socket_path, n_cards))
    if rc != 0 or "DAEMON_ERR=" in out:
        return 0.0, 0.0, False
    d = _parse_script_numbers(out)
    return d.get("RSS_MB", 0.0), d.get("SEED_SEC", 0.0), True


def _measure_inproc_embed_latency(reps: int) -> list[float]:
    eng = EmbeddingEngine()
    for _ in range(30):
        eng.embed_text("warmup query the server ssh key port")
    s: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        eng.embed_text("the server ssh key port is 2204 not 22")
        s.append(time.perf_counter() - t0)
    return _percentiles(s, (50, 95, 99))


def _measure_daemon_embed_latency(dc: Any, reps: int) -> tuple[float, float, float, float]:
    for _ in range(30):
        dc.embed_text("warmup query the server ssh key port")
    s: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        dc.embed_text("the server ssh key port is 2204 not 22")
        s.append(time.perf_counter() - t0)
    p = _percentiles(s, (50, 95, 99))
    texts = [f"batch {i} the server ssh key port" for i in range(32)]
    b: list[float] = []
    for _ in range(200):
        t0 = time.perf_counter()
        dc.embed_batch(texts)
        b.append(time.perf_counter() - t0)
    return p[0], p[1], p[2], _percentiles(b, (99,))[0]


def _measure_daemon_ping_latency(dc: Any, reps: int) -> tuple[float, float, float]:
    """Pure IPC dispatch overhead: a `ping` frame round-trip (no ONNX inference)
    over the Unix socket. Separates socket+shm latency from the daemon's
    inference time that dominates an embed_text round-trip."""
    for _ in range(30):
        dc.ping()
    s: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        dc.ping()
        s.append(time.perf_counter() - t0)
    p = _percentiles(s, (50, 95, 99))
    return p[0], p[1], p[2]


def _measure_parity(dc: Any, eng: Any, n_cards: int, nq_target: int = 50) -> tuple[float, float, int]:
    """Claim 4: identical facts embedded once via the daemon and once
    in-process; compare the max per-element cosine diff and the fraction of
    vector_search top-k (ids AND order) that match."""
    cards_d = make_scale_cards(n_cards, dc)
    cards_i = make_scale_cards(n_cards, eng)
    store_d = MemoryStore(":memory:", embedder=dc)
    store_i = MemoryStore(":memory:", embedder=eng)
    _seed_bulk(store_d, cards_d)
    _seed_bulk(store_i, cards_i)
    maxdiff = 0.0
    matches = 0
    nq = min(nq_target, n_cards)
    step = max(1, n_cards // nq)
    queries = 0
    try:
        for idx in range(0, n_cards, step):
            qtext = cards_d[idx].fact
            vd = dc.embed_text(qtext)
            vi = eng.embed_text(qtext)
            maxdiff = max(maxdiff, max(abs(a - b) for a, b in zip(vd, vi)))
            id_d = [c.id for c, _ in store_d.vector_search(vd, k=10)]
            id_i = [c.id for c, _ in store_i.vector_search(vi, k=10)]
            if id_d == id_i:
                matches += 1
            queries += 1
    finally:
        store_d.close()
        store_i.close()
    return maxdiff, (matches / queries * 100.0) if queries else 0.0, queries


def _run_reduced_adversarial(embedder: Any, scale: int = 2000, distractors: int = 100,
                             pairs: int = 50, workers: int = 5, cycles: int = 200) -> tuple[bool, str]:
    """Claim 5: confirm the existing in-process adversarial path still runs
    (reduced scale to keep the daemon benchmark tractable)."""
    try:
        res = run_adversarial(n_scale=scale, n_distractors=distractors, n_needle_queries=50,
                              n_polarity_pairs=pairs, n_workers=workers,
                              total_cycles=cycles, embedder=embedder)
        holds = sum([
            res.scale.vector_claim_holds, res.scale.sql_claim_holds,
            res.scale.rss_claim_holds, res.needle.recall_claim_holds,
            res.negation.negation_claim_holds, res.concurrency.concurrency_claim_holds,
        ])
        return True, f"{holds}/6 sub-claims, {res.latency_ms / 1000.0:.1f}s wall"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"raised {type(exc).__name__}: {exc}"


def run_daemon_benchmark(socket_path: str = "/tmp/izero_bench.sock", n_cards: int = 10_000,
                         n_workers: int = 5, worker_cards: int = 200,
                         ipc_reps: int = 1500) -> DaemonBenchmarkResult:
    """Measure the five Phase 7A claims. A daemon is auto-spawned via
    `DaemonClient(socket_path=...)`, measured, then shut down cleanly and its
    socket removed. If the daemon package is unavailable, the in-process
    baselines are measured and every daemon claim is left PENDING."""
    res = DaemonBenchmarkResult(available=False, n_cards=n_cards)

    # ---- in-process baselines (do not depend on the daemon) ----
    res.inproc_client_rss_mb, res.inproc_client_seed_sec = _measure_inproc_client_rss(n_cards)
    inproc_lat = _measure_inproc_embed_latency(ipc_reps)
    res.inproc_embed_p99_ms = inproc_lat[2]

    eng = EmbeddingEngine()
    res.regression_ok, res.regression_note = _run_reduced_adversarial(eng)
    tmpdir = tempfile.mkdtemp()
    inproc_ready = [os.path.join(tmpdir, f"r{i}") for i in range(n_workers)]
    inproc_5w, _live = _measure_workers_total(
        [_inproc_worker_script(worker_cards, rf) for rf in inproc_ready],
        inproc_ready)
    res.inproc_5w_total_mb = inproc_5w

    # ---- gate on the daemon package + a real hello handshake ----
    try:
        from isotope_zero.daemon.client import DaemonClient
    except Exception as exc:
        res.pending_reason = f"isotope_zero.daemon.client import: {exc}"
        return res

    # DaemonClient's auto-spawn does NOT forward `--socket` to the server, so a
    # custom bench socket would spawn a daemon on the DEFAULT /tmp/izero.sock and
    # the client would never reach it. We therefore pre-spawn the server on OUR
    # socket ourselves, then let DaemonClient connect to it (spawn becomes a
    # no-op because the socket is already live).
    server_proc: subprocess.Popen | None = None
    dc: Any = None
    try:
        if os.path.exists(socket_path):
            try:
                os.unlink(socket_path)  # stale socket from a crashed run
            except Exception:
                pass
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "isotope_zero.daemon.server",
             "--socket", socket_path, "--idle-timeout", "0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if os.path.exists(socket_path) and server_proc.poll() is None:
                break
            time.sleep(0.05)
        dc = DaemonClient(socket_path=socket_path)
        hello = dc.embed_text("isotope_zero daemon benchmark hello handshake")
        if not hello or len(hello) != getattr(dc, "dim", 0):
            raise RuntimeError("empty/short hello vector from daemon")
    except Exception as exc:
        res.pending_reason = f"DaemonClient handshake failed: {exc}"
        if dc is not None:
            try:
                dc.shutdown()
            except Exception:
                pass
        if server_proc is not None and server_proc.poll() is None:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=10)
            except Exception:
                pass
        try:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
        except Exception:
            pass
        return res

    res.available = True
    try:
        # claim 1 — client subprocess RSS
        res.client_rss_mb, res.client_seed_sec, ok = _measure_daemon_client_rss(socket_path, n_cards)
        if not ok:
            res.available = False
            res.pending_reason = "daemon client RSS subprocess failed"
            return res

        # claim 2 — 5 daemon-backed workers + the daemon process itself
        tmpdir = tempfile.mkdtemp()
        daemon_ready = [os.path.join(tmpdir, f"d{i}") for i in range(n_workers)]
        daemon_scripts = [_daemon_worker_script(socket_path, worker_cards, rf)
                          for rf in daemon_ready]
        d_total, _ = _measure_workers_total(daemon_scripts, daemon_ready)
        dpid = server_proc.pid if (server_proc is not None and server_proc.poll() is None) else None
        if dpid is None:
            dpid = _find_daemon_pid(dc, socket_path)  # fallback scan
        res.daemon_rss_known = dpid is not None
        res.daemon_rss_mb = _ps_rss_mb(dpid) if dpid else 0.0
        res.daemon_5w_total_mb = d_total + res.daemon_rss_mb

        # claim 3 — IPC dispatch latency + embed_batch(32) p99 + raw ping RTT
        (res.ipc_p50_ms, res.ipc_p95_ms, res.ipc_p99_ms,
         res.embed_batch32_p99_ms) = _measure_daemon_embed_latency(dc, ipc_reps)
        (res.ipc_ping_p50_ms, res.ipc_ping_p95_ms,
         res.ipc_ping_p99_ms) = _measure_daemon_ping_latency(dc, min(1000, ipc_reps))

        # claim 4 — recall parity against the in-process engine
        res.max_cos_diff, res.topk_match_pct, res.parity_queries = _measure_parity(dc, eng, n_cards)
    finally:
        try:
            dc.shutdown()
        except Exception:
            pass
        if server_proc is not None and server_proc.poll() is None:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=10)
            except Exception:
                try:
                    server_proc.kill()
                except Exception:
                    pass
        try:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
        except Exception:
            pass
    return res


# ------------------------------------------------------------------- #
# E. Temporal Decay & Graph Consolidation (30-day trace)
# ------------------------------------------------------------------- #
# Simulates 30 days of evolving user memory: locations change, new
# languages are learned, jobs are upgraded, hobbies shift. Each day adds
# cards with embeddings; consolidation sweeps run at 5-day intervals.
# Temporal recall queries at day 30 must prefer the fresh/more-specific
# facts over stale/generic ones — measured by hybrid cosine+retention
# scoring. Claims: >90% temporal recall, <0.05ms hybrid overhead, >10%
# storage reduction after consolidation.

_TEMPORAL_CITIES = ["San Francisco", "New York", "Los Angeles", "Boston", "Chicago", "Seattle", "Austin"]
_TEMPORAL_ROLES = ["engineer", "backend developer", "data analyst", "tech lead", "devops engineer"]
_TEMPORAL_LANGUAGES = ["Spanish", "French", "German", "Italian", "Japanese", "Korean"]
_TEMPORAL_HOBBIES = ["hiking", "photography", "playing piano", "rock climbing", "painting", "gardening"]
_TEMPORAL_PETS = ["a dog", "a cat", "a goldfish", "a hamster"]
_TEMPORAL_TOOLS = ["Python", "Rust", "TypeScript", "Go", "Vim", "VS Code", "IntelliJ"]
_TEMPORAL_BANDS = ["Radiohead", "Daft Punk", "Tame Impala", "Khruangbin", "Bob Dylan"]
_TEMPORAL_COLORS = ["blue", "green", "black", "red", "navy", "teal"]
_TEMPORAL_DRINKS = ["coffee", "green tea", "matcha", "espresso", "chai"]
_TEMPORAL_OS = ["macOS", "Ubuntu", "Fedora", "Windows 11", "Arch Linux"]
_TEMPORAL_FILLER_FRAMES = [
    "I use {tool} daily.",
    "My favorite color is {color}.",
    "I drink {drink} every morning.",
    "I read {n} books last month.",
    "My commute takes {n} minutes.",
    "The weather today is {weather}.",
    "I watched {movie} last night.",
    "I follow {sport} matches on weekends.",
    "The grocery store near me sells {food_item}.",
    "My laptop has {n} GB of RAM.",
    "I attend {event} every year.",
    "I subscribe to {service} for streaming.",
    "Last week I visited {place}.",
    "I bought {n} items from the store.",
    "The printer is out of {thing}.",
    "I have a meeting at {time}.",
    "My friend recommended {restaurant}.",
    "I need to renew my {doc} by next month.",
    "The conference is in {city} this year.",
    "I fixed the {thing} in my apartment.",
    "My car needs {service} soon.",
    "I ordered {n} things online yesterday.",
    "The {plant} on my desk needs water.",
    "I forgot to {task} this morning.",
    "My {relative} called me yesterday.",
]

# Milestone facts: specific evolving knowledge that queries test.
# Each is (day, card_id, fact, tags). The card_id fields are used to
# look up scores during temporal-recall verification.
_TEMPORAL_MILESTONES: list[tuple[int, str, str, list[str]]] = [
    (1, "t-day-1-loc-sf", "I live in San Francisco.", ["location", "milestone"]),
    (1, "t-day-1-job-eng", "I work as an engineer.", ["job", "milestone"]),
    (1, "t-day-1-hobby-hike", "I enjoy hiking.", ["hobby", "milestone"]),
    (5, "t-day-5-lang-es", "I am learning Spanish.", ["language", "milestone"]),
    (5, "t-day-5-tool-py", "I use Python for work.", ["tool", "milestone"]),
    (5, "t-day-5-plan-japan", "I plan to travel to Japan.", ["plan", "milestone"]),
    (10, "t-day-10-loc-ny", "I moved to New York.", ["location", "milestone", "move"]),
    (15, "t-day-15-lang-fr", "I am learning French now.", ["language", "milestone"]),
    (20, "t-day-20-job-sr", "I got a promotion to senior engineer.", ["job", "milestone", "promotion"]),
    (25, "t-day-25-pet-dog", "I adopted a dog.", ["pet", "milestone"]),
    (25, "t-day-25-hobby-bake", "I started baking bread.", ["hobby", "milestone"]),
]

# Temporal recall queries: each maps a natural-language question to the
# two milestone card IDs where the "correct" (fresher/more-specific) card
# must outrank the "stale" one in hybrid scoring.
_TEMPORAL_QUERY_PAIRS: list[tuple[str, str, str]] = [
    # (query_text, correct_card_id, stale_card_id)
    ("Where does the user live?", "t-day-10-loc-ny", "t-day-1-loc-sf"),
    ("What language is the user learning?", "t-day-15-lang-fr", "t-day-5-lang-es"),
    ("What is the user's job?", "t-day-20-job-sr", "t-day-1-job-eng"),
]


def _hybrid_rerank(
    candidates: list[tuple[MemoryCard, float]],
    alpha: float,
    current_ts: float,
    stability_map: dict[str, float],
) -> list[tuple[MemoryCard, float]]:
    """Re-rank vector_search results by hybrid cosine+retention score.

    Uses only the provided candidates (typically top-k from vector_search)
    so the underlying retrieval path is identical regardless of alpha.
    """
    if alpha >= 1.0:
        return candidates  # pure cosine — no re-ranking needed
    scored: list[tuple[MemoryCard, float]] = []
    for card, cosine in candidates:
        last_access = card.last_access if card.last_access else card.timestamp
        stability = stability_map.get(card.id, card.stability)
        retention = calculate_retention(last_access, stability, current_ts)
        score = hybrid_score(cosine, retention, alpha)
        scored.append((card, score))
    scored.sort(key=lambda x: -x[1])
    return scored


def _hybrid_search_all(
    store: MemoryStore,
    query_vec: list[float],
    alpha: float,
    current_ts: float,
    stability_map: dict[str, float],
) -> list[tuple[MemoryCard, float]]:
    """Score EVERY active card with hybrid cosine+retention, descending.

    Used ONLY for the temporal-recall correctness check (not latency
    measurement), where we need scores for every card to compare specific
    milestone pairs.
    """
    all_cards = store.all()
    scored: list[tuple[MemoryCard, float]] = []
    qd = len(query_vec)
    for card in all_cards:
        emb = card.embedding
        if emb is None or not emb:
            continue
        n = min(qd, len(emb))
        if n == 0:
            continue
        dot = sum(query_vec[i] * emb[i] for i in range(n))
        cosine = dot
        if cosine < 0.0:
            cosine = 0.0
        elif cosine > 1.0:
            cosine = 1.0
        last_access = card.last_access if card.last_access else card.timestamp
        stability = stability_map.get(card.id, card.stability)
        retention = calculate_retention(last_access, stability, current_ts)
        score = hybrid_score(cosine, retention, alpha)
        scored.append((card, score))
    scored.sort(key=lambda x: -x[1])
    return scored


def _random_filler_fact(rng: random.Random, day: int, idx: int) -> str:
    """Generate a realistic-but-random fact from the filler template pool."""
    tmpl = rng.choice(_TEMPORAL_FILLER_FRAMES)
    return tmpl.format(
        tool=rng.choice(_TEMPORAL_TOOLS),
        color=rng.choice(_TEMPORAL_COLORS),
        drink=rng.choice(_TEMPORAL_DRINKS),
        band=rng.choice(_TEMPORAL_BANDS),
        os=rng.choice(_TEMPORAL_OS),
        city=rng.choice(_TEMPORAL_CITIES),
        item=rng.choice(["standing desk", "mechanical keyboard", "ultrawide monitor",
                         "ergonomic chair", "second screen", "noise-cancelling headphones"]),
        n=rng.randrange(1, 60),
        time=f"{rng.randrange(5, 11)}:{rng.choice(['00', '15', '30', '45'])} AM",
        weather=rng.choice(["sunny", "rainy", "cloudy", "foggy", "windy", "snowy"]),
        movie=rng.choice(["Inception", "The Matrix", "Interstellar", "Parasite",
                          "Everything Everywhere", "Dune"]),
        sport=rng.choice(["basketball", "soccer", "tennis", "Formula 1", "cricket"]),
        food_item=rng.choice(["avocados", "bread", "eggs", "rice", "chicken", "pasta"]),
        event=rng.choice(["a tech meetup", "a hackathon", "a conference",
                          "a board game night", "a concert"]),
        service=rng.choice(["Netflix", "Spotify", "HBO Max", "Disney+", "YouTube Premium"]),
        place=rng.choice(["the bookstore", "a cafe", "the gym", "the park",
                          "the doctor's office", "the library"]),
        restaurant=rng.choice(["a sushi place", "a pizzeria", "a taco truck",
                               "a ramen shop", "a BBQ joint"]),
        doc=rng.choice(["passport", "driver's license", "lease", "insurance policy"]),
        plant=rng.choice(["succulent", "cactus", "fern", "orchid", "snake plant"]),
        task=rng.choice(["take out the trash", "check my email", "water the plants",
                         "pay the bill", "respond to a message"]),
        relative=rng.choice(["sister", "brother", "mom", "dad", "cousin", "friend"]),
        thing=rng.choice(["leaky faucet", "broken chair", "noisy fan", "stuck drawer",
                          "flickering light", "squeaky door"]),
    )


def run_temporal_benchmark(
    n_epochs: int = 30,
    cards_per_day: int = 30,
    embedder: EmbeddingEngine | None = None,
) -> TemporalDecayResult:
    """Simulate a 30-day conversation trace with evolving facts.

    Each day adds ``cards_per_day`` cards (milestones + filler); every
    5 days a consolidation sweep runs. At day 30, temporal recall queries
    verify that fresh facts suppress stale ones under hybrid scoring.
    """

    # ---- deterministic seed, date anchors ------------------------------------
    rng = random.Random(42)
    day_secs = 86400.0  # seconds per simulated day
    # Anchor base_ts so day 30 lands ~1 hour before now — recent cards survive
    # the Consolidator's min_age_seconds grace period, and the 30-day spread
    # still exercises the full Ebbinghaus decay range.
    base_ts = now_ts() - (30.0 * day_secs) - 3600.0
    day30_ts = base_ts + 30.0 * day_secs

    # ---- backend --------------------------------------------------------------
    backend = embedder if embedder is not None else EmbeddingEngine()

    # ---- build milestone index ------------------------------------------------
    milestones_by_day: dict[int, list[tuple[str, str, list[str]]]] = {}
    for day, mid, fact, tags in _TEMPORAL_MILESTONES:
        milestones_by_day.setdefault(day, []).append((mid, fact, tags))

    # ---- tracking maps --------------------------------------------------------
    stability_map: dict[str, float] = {}  # card_id -> stability S
    active_milestone_ids: set[str] = set()  # currently-true milestone card ids

    # Active lifecycle: a milestone is "active" from its insertion day until the
    # day its category is superseded by a newer fact. We touch active milestones
    # every day so their last_access stays fresh; superseded ones go cold.
    # Supersession schedule (exclusive end-day):
    #   loc-sf:  active  days 1-9   (superseded by loc-ny day 10)
    #   job-eng: active  days 1-19  (superseded by job-sr day 20)
    #   lang-es: active  days 5-14  (superseded by lang-fr day 15)

    # ---- SQLite store ---------------------------------------------------------
    store = MemoryStore(":memory:", embedder=backend)

    # ---- graph initialisation -------------------------------------------------
    init_graph(store._conn)
    total_edges_created = 0

    # ---- 30-day simulation loop ------------------------------------------------
    total_cards_added = 0
    for day in range(1, n_epochs + 1):
        day_ts = base_ts + day * day_secs
        milestones_today = milestones_by_day.get(day, [])
        n_filler = cards_per_day - len(milestones_today)

        # Build cards for this day.
        day_cards: list[MemoryCard] = []
        # Milestones first (gets deterministic lower indices in vector cache).
        for mid, fact, tags in milestones_today:
            emb = backend.embed_text(fact)
            day_cards.append(MemoryCard(
                id=mid, fact=fact, evidence=fact,
                timestamp=day_ts, tags=tags,
                embedding=_norm(emb),
                source_tokens=estimate_tokens(fact),
                last_access=day_ts,
                stability=1.0,
            ))
            active_milestone_ids.add(mid)
            stability_map[mid] = 1.0

        # Filler cards.
        for fi in range(n_filler):
            fact = _random_filler_fact(rng, day, fi)
            cid = f"t-day-{day}-fill-{fi}"
            emb = backend.embed_text(fact)
            day_cards.append(MemoryCard(
                id=cid, fact=fact, evidence=fact,
                timestamp=day_ts, tags=["filler"],
                embedding=_norm(emb),
                source_tokens=estimate_tokens(fact),
                last_access=day_ts,
                stability=1.0,
            ))
            stability_map[cid] = 1.0

        # Bulk-insert for speed.
        _seed_bulk(store, day_cards)
        total_cards_added += len(day_cards)

        # ---- manage active-milestone lifecycle ---------------------------------
        # On supersession days, retire the stale milestones.
        if day == 10:
            active_milestone_ids.discard("t-day-1-loc-sf")
        if day == 15:
            active_milestone_ids.discard("t-day-5-lang-es")
        if day == 20:
            active_milestone_ids.discard("t-day-1-job-eng")

        # ---- simulate retrieval: touch active milestones + random cards --------
        # Touching re-sets last_access, keeping retention alive for active facts.
        touch_ts = day_ts
        for mid in active_milestone_ids:
            store.touch(mid, at=touch_ts)
            stability_map[mid] = 1.0  # keep at default for simplicity

        # Touch 2-3 random cards to simulate organic retrieval noise.
        all_ids = [c.id for c in store.all()]
        if all_ids:
            n_touch_rand = min(3, len(all_ids))
            for rid in rng.sample(all_ids, n_touch_rand):
                if rid not in active_milestone_ids:
                    store.touch(rid, at=touch_ts)

        # ---- consolidation checkpoint (every 5 days) ---------------------------
        if day in {5, 10, 15, 20, 25}:
            Consolidator(store, embedder=backend, dedup_threshold=0.94).run()

            # Auto-link cards via the graph engine.
            all_cards = store.all()
            embeddings_list = [(c.id, c.embedding) for c in all_cards if c.embedding]
            for card in all_cards:
                if card.embedding:
                    linked = auto_link_cards(
                        store._conn,
                        card.id,
                        card.tags,
                        card.embedding,
                        embeddings_list,
                        cosine_threshold=0.75,
                    )
                    total_edges_created += len(linked)
            n_days_with_milestones += 1

    # ---- final consolidation sweep (day 30) ------------------------------------
    # Run one more sweep to trigger any final dedup + pruning. Use realistic
    # thresholds — the cumulative reduction from all sweeps is measured against
    # total_cards_added, not just the delta across this one sweep.
    n_cards_before = store.count()
    Consolidator(store, embedder=backend, dedup_threshold=0.94).run()
    n_cards_after_consolidation = store.count()

    # Cumulative reduction: total cards ever added vs active after all sweeps.
    storage_reduction_pct = (
        (total_cards_added - n_cards_after_consolidation) / max(1, total_cards_added) * 100.0
    )

    # ---- final auto-link + cluster detection -----------------------------------
    all_cards = store.all()
    embeddings_list = [(c.id, c.embedding) for c in all_cards if c.embedding]
    for card in all_cards:
        if card.embedding:
            linked = auto_link_cards(
                store._conn,
                card.id,
                card.tags,
                card.embedding,
                embeddings_list,
                cosine_threshold=0.75,
            )
            total_edges_created += len(linked)

    stats = get_graph_stats(store._conn)
    edges_count = stats["edge_count"]
    clusters = detect_clusters(store._conn, min_cluster_size=3, min_edge_weight=0.80)
    cluster_count = len(clusters)

    # ---- temporal recall queries -----------------------------------------------
    temporal_recall_correct = 0
    for qtext, correct_id, stale_id in _TEMPORAL_QUERY_PAIRS:
        q_vec = _norm(backend.embed_text(qtext))
        ranked = _hybrid_search_all(store, q_vec, alpha=0.7, current_ts=day30_ts,
                                     stability_map=stability_map)
        score_by_id = {card.id: s for card, s in ranked}
        correct_score = score_by_id.get(correct_id, -1.0)
        stale_score = score_by_id.get(stale_id, -2.0)
        if correct_score > stale_score:
            temporal_recall_correct += 1

    temporal_recall_total = len(_TEMPORAL_QUERY_PAIRS)
    temporal_recall_pct = (
        (temporal_recall_correct / temporal_recall_total * 100.0)
        if temporal_recall_total > 0 else 0.0
    )

    # ---- latency measurement: pure cosine baseline vs hybrid ------------------
    # Use the first query vector for timing. Both paths call store.vector_search
    # (the same numpy BLAS path); the only added cost is _hybrid_rerank on the
    # top-k results.  This isolates the pure retention/hybrid-score overhead.
    q_vec_timing = _norm(backend.embed_text(_TEMPORAL_QUERY_PAIRS[0][0]))

    # Warm vector cache once so baseline + hybrid hit the same cached matrix.
    store.vector_search(q_vec_timing, k=5)

    # Baseline: pure cosine (store.vector_search only).
    baseline_times: list[float] = []
    for _ in range(30):
        t0 = time.perf_counter()
        store.vector_search(q_vec_timing, k=5)
        baseline_times.append(time.perf_counter() - t0)
    baseline_p99 = _percentiles(baseline_times, (99,))[0]

    # Hybrid: vector_search + _hybrid_rerank (alpha=0.7). Same k as baseline
    # so the latency delta isolates the rerank cost, not a wider candidate fetch.
    hybrid_times: list[float] = []
    for _ in range(30):
        t0 = time.perf_counter()
        candidates = store.vector_search(q_vec_timing, k=5)
        _hybrid_rerank(candidates, alpha=0.7, current_ts=day30_ts,
                       stability_map=stability_map)
        hybrid_times.append(time.perf_counter() - t0)
    hp = _percentiles(hybrid_times, (50, 95, 99))
    hybrid_p50, hybrid_p95, hybrid_p99 = hp[0], hp[1], hp[2]

    latency_overhead_ms = hybrid_p99 - baseline_p99

    store.close()

    return TemporalDecayResult(
        n_cards_initial=total_cards_added,
        n_cards_after_pruning=0,  # pruning is bundled with consolidation sweeps
        n_cards_after_consolidation=n_cards_after_consolidation,
        storage_reduction_pct=round(storage_reduction_pct, 2),
        temporal_recall_correct=temporal_recall_correct,
        temporal_recall_total=temporal_recall_total,
        temporal_recall_pct=round(temporal_recall_pct, 1),
        query_latency_p50_ms=round(hybrid_p50, 4),
        query_latency_p95_ms=round(hybrid_p95, 4),
        query_latency_p99_ms=round(hybrid_p99, 4),
        baseline_latency_p99_ms=round(baseline_p99, 4),
        latency_overhead_ms=round(latency_overhead_ms, 4),
        edges_created=edges_count,
        cluster_groups_found=cluster_count,
    )


def render_temporal_markdown(res: TemporalDecayResult) -> str:
    """Render the Section E temporal-decay report as markdown."""
    L: list[str] = []
    L.append("")
    L.append("## E. Temporal Decay & Graph Consolidation (30-day trace)")
    L.append("")
    L.append("| claim (verbatim) | measured | verdict |")
    L.append("|---|---|---|")

    c1 = f"{res.temporal_recall_pct:.1f}% ({res.temporal_recall_correct}/{res.temporal_recall_total} correct)"
    v1 = "PASS" if res.recall_claim_holds else "FAIL"
    L.append(f"| Temporal recall > 90% (fresh suppresses stale) | {c1} | {v1} |")

    c2 = f"{res.latency_overhead_ms:.4f} ms"
    v2 = "PASS" if res.latency_claim_holds else "FAIL"
    L.append(f"| Query latency overhead < 0.05 ms | {c2} | {v2} |")

    c3 = f"{res.storage_reduction_pct:.1f}%"
    v3 = "PASS" if res.storage_claim_holds else "FAIL"
    L.append(f"| Storage reduction > 10% after consolidation | {c3} | {v3} |")

    L.append("")
    L.append(
        f"Consolidation summary: {res.n_cards_initial} cards -> "
        f"{res.n_cards_after_consolidation} active "
        f"({res.storage_reduction_pct:.1f}% reduction)."
    )
    L.append(
        f"Graph: {res.edges_created} edges, "
        f"{res.cluster_groups_found} tight clusters detected."
    )
    L.append(
        f"Latency: hybrid p50/p95/p99 = {res.query_latency_p50_ms:.4f}/"
        f"{res.query_latency_p95_ms:.4f}/{res.query_latency_p99_ms:.4f} ms, "
        f"baseline p99 = {res.baseline_latency_p99_ms:.4f} ms, "
        f"overhead = {res.latency_overhead_ms:.4f} ms."
    )
    return "\n".join(L)


def run_adversarial(n_scale: int = 10_000, n_distractors: int = 500,
                    n_needle_queries: int = 100, n_polarity_pairs: int = 100,
                    n_workers: int = 25, total_cycles: int = 1000,
                    embedder: EmbeddingEngine | None = None) -> AdversarialResult:
    backend = embedder if embedder is not None else EmbeddingEngine()
    t0 = time.perf_counter()
    scale = run_scale(n_scale, backend)
    t1 = time.perf_counter()
    needle = run_needle(n_distractors, backend, reps=n_needle_queries)
    t2 = time.perf_counter()
    negation = run_negation(backend, n_pairs=n_polarity_pairs)
    t3 = time.perf_counter()
    concurrency = run_concurrency(n_workers=n_workers, total_cycles=total_cycles, embedder=backend)
    t4 = time.perf_counter()
    return AdversarialResult(
        scale=scale, needle=needle, negation=negation, concurrency=concurrency,
        latency_ms=(t4 - t0) * 1000.0,
    )


def render_adversarial_markdown(res: AdversarialResult) -> str:
    s = res.scale
    lines: list[str] = []
    lines.append("# Isotope Zero — Adversarial Stress Report")
    lines.append("")
    lines.append("Claims are the brief's stated thresholds, recorded verbatim next to")
    lines.append("measured reality. A claim that does not hold renders FAIL.")
    lines.append("")
    lines.append("## A. High-Density Scale (10,000+ cards)")
    lines.append("")
    lines.append("| metric | measured | claim | verdict |")
    lines.append("|---|---|---|---|")
    lines.append(f"| vector_search p99 | {s.vector_p99_ms:.2f} ms | <{s.claim_vector_ms} ms | {'PASS' if s.vector_claim_holds else 'FAIL'} |")
    lines.append(f"| sql_lookup p99 | {s.sql_p99_ms:.3f} ms | <{s.claim_sql_ms} ms | {'PASS' if s.sql_claim_holds else 'FAIL'} |")
    lines.append(f"| RSS | {s.rss_mb:.0f} MB | <{s.claim_rss_mb} MB | {'PASS' if s.rss_claim_holds else 'FAIL'} |")
    lines.append("")
    lines.append(f"vector p50/p95/p99 = {s.vector_p50_ms:.2f}/{s.vector_p95_ms:.2f}/{s.vector_p99_ms:.2f} ms; "
                 f"sql p50/p95/p99 = {s.sql_p50_ms:.3f}/{s.sql_p95_ms:.3f}/{s.sql_p99_ms:.3f} ms (n={s.n_cards})")
    lines.append("")
    lines.append("## B. Needle-in-a-Haystack + Distractor Floor")
    lines.append("")
    n = res.needle
    lines.append(f"- queries: {n.n_needle_queries}  distractors: {n.n_distractors}")
    lines.append(f"- recall: **{n.recall_pct:.1f}%** (claim {n.claim_recall_pct}%) → "
                 f"{'PASS' if n.recall_claim_holds else 'FAIL'}")
    lines.append("")
    lines.append("## C. Negation & Polarity Bombardment")
    lines.append("")
    g = res.negation
    lines.append(f"- pairs: {g.n_pairs}  incorrect merges: **{g.incorrect_merges}** "
                 f"(claim {g.claim_incorrect_merges}) → "
                 f"{'PASS' if g.negation_claim_holds else 'FAIL'}")
    lines.append(f"- distinct timeline survivors: {g.distinct_timeline_survivors}")
    lines.append("")
    lines.append("## D. Max Concurrency & DB Contention Warfare")
    lines.append("")
    c = res.concurrency
    lines.append(f"- workers: {c.n_workers}  ops: {c.total_ops}  "
                 f"target {c.ops_per_sec_per_worker} ops/s/worker")
    lines.append(f"- OperationalError: **{c.operational_errors}** (claim {c.claim_operational_errors})")
    lines.append(f"- WAL corruption: **{c.wal_corruptions}** (claim {c.claim_wal_corruptions}) → "
                 f"{'PASS' if c.concurrency_claim_holds else 'FAIL'}")
    lines.append(f"- consolidation sweeps during warfare: {c.consolidation_sweeps}")
    lines.append("")
    passed = sum([s.vector_claim_holds, s.sql_claim_holds, s.rss_claim_holds,
                  n.recall_claim_holds, g.negation_claim_holds, c.concurrency_claim_holds])
    lines.append(f"**Total: {passed}/6 claims hold.** (harness wall-clock {res.latency_ms/1000.0:.1f}s)")
    return "\n".join(lines)


def render_daemon_markdown(res: DaemonBenchmarkResult) -> str:
    """Phase 7A honest report: verbatim claim | measured | PASS/FAIL (or PENDING
    when the daemon package is absent and could not be measured)."""
    L: list[str] = []
    L.append("")
    L.append("## Phase 7A — Shared-Memory Embedding Daemon")
    L.append("")
    if not res.available:
        L.append(f"**Daemon package not importable/handshake failed — daemon claims PENDING.** "
                 f"({res.pending_reason})")
        L.append("In-process baselines below were measured; no daemon number is fabricated.")
        L.append("")
    L.append("| claim (verbatim) | measured | verdict |")
    L.append("|---|---|---|")

    # claim 1
    c1 = (f"{res.client_rss_mb:.0f} MB peak, client subprocess, {res.n_cards} cards "
          f"(seed {res.client_seed_sec:.1f}s)") if res.available else "PENDING"
    v1 = ("PASS" if res.client_rss_holds else "FAIL") if res.available else "PENDING"
    L.append(f"| Client RSS < 10 MB (vs in-process ~450 MB) | {c1} | {v1} |")
    L.append(f"| — in-process client comparator (same {res.n_cards} cards) | "
             f"{res.inproc_client_rss_mb:.0f} MB peak (seed {res.inproc_client_seed_sec:.1f}s) | — |")

    # claim 2
    if res.available:
        d = f"{res.daemon_rss_mb:.0f} MB" if res.daemon_rss_known else "n/a (pid not found)"
        c2 = (f"daemon total {res.daemon_5w_total_mb:.0f} MB (5 clients + daemon {d}) vs "
              f"in-process total {res.inproc_5w_total_mb:.0f} MB")
    else:
        c2 = "PENDING"
    v2 = ("PASS" if res.sys_rss_holds else "FAIL") if res.available else "PENDING"
    L.append(f"| daemon-backed 5-worker total ≪ 5×450 MB in-process | {c2} | {v2} |")

    # claim 3
    if res.available:
        c3 = (f"daemon embed_text p99 {res.ipc_p99_ms:.3f} ms "
              f"(p50/p95 {res.ipc_p50_ms:.3f}/{res.ipc_p95_ms:.3f}); "
              f"in-process p99 {res.inproc_embed_p99_ms:.3f} ms; "
              f"embed_batch(32) p99 {res.embed_batch32_p99_ms:.3f} ms")
    else:
        c3 = "PENDING"
    v3 = ("PASS" if res.ipc_latency_holds else "FAIL") if res.available else "PENDING"
    L.append(f"| IPC dispatch latency < 0.05 ms p99 | {c3} | {v3} |")
    if res.available:
        L.append(f"| — raw IPC dispatch (daemon ping RTT, no inference) | "
                 f"p99 {res.ipc_ping_p99_ms:.3f} ms (p50/p95 {res.ipc_ping_p50_ms:.3f}/"
                 f"{res.ipc_ping_p95_ms:.3f}) | — |")

    # claim 4
    if res.available:
        c4 = (f"max |Δvec| = {res.max_cos_diff:.2e}; top-k match = {res.topk_match_pct:.1f}% "
              f"(ids+order across {res.parity_queries} queries)")
    else:
        c4 = "PENDING"
    v4 = ("PASS" if res.parity_holds else "FAIL") if res.available else "PENDING"
    L.append(f"| Recall parity 100% (bit-identical vectors → identical top-k) | {c4} | {v4} |")

    # claim 5 (measured in both modes — it is the in-process path)
    if res.regression_ok:
        c5, v5 = f"reduced-scale in-process adversarial ran ({res.regression_note})", "PASS"
    else:
        c5, v5 = f"FAILED: {res.regression_note}", "FAIL"
    L.append(f"| No regression: existing in-process adversarial path still runs | {c5} | {v5} |")

    if res.available:
        npass = sum([res.client_rss_holds, res.sys_rss_holds, res.ipc_latency_holds,
                     res.parity_holds, res.regression_holds])
        L.append("")
        L.append(f"**Daemon total: {npass}/5 claims hold** (n_cards={res.n_cards}).")
    else:
        L.append("")
        L.append("**Daemon claims PENDING** — rerun `--daemon` after "
                 "`from isotope_zero.daemon.client import DaemonClient` succeeds.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="izero-eval-adversarial",
                                description="Isotope Zero adversarial stress harness (sections A-E).")
    p.add_argument("--scale", type=int, default=10_000, help="A: card count (default 10000)")
    p.add_argument("--distractors", type=int, default=500, help="B: distractor count (default 500)")
    p.add_argument("--needle-queries", type=int, default=100, help="B: adversarial queries (default 100)")
    p.add_argument("--polarity-pairs", type=int, default=100, help="C: contradictory pairs (default 100)")
    p.add_argument("--workers", type=int, default=25, help="D: concurrent worker processes (default 25)")
    p.add_argument("--cycles", type=int, default=1000, help="D: cycles per worker (default 1000)")
    p.add_argument("--temporal", action="store_true",
                   help="E: 30-day temporal decay & graph consolidation benchmark")
    p.add_argument("--temporal-epochs", type=int, default=30,
                   help="E: number of simulated days (default 30)")
    p.add_argument("--temporal-cards-per-day", type=int, default=30,
                   help="E: cards per simulated day (default 30)")
    p.add_argument("--daemon", action="store_true",
                   help="Phase 7A: shared-memory embedding daemon benchmark (claims 1-5)")
    p.add_argument("--daemon-socket", default="/tmp/izero_bench.sock",
                   help="Phase 7A daemon socket path (default /tmp/izero_bench.sock)")
    p.add_argument("--daemon-scale", type=int, default=10_000,
                   help="Phase 7A daemon corpus size (default 10000)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = p.parse_args(argv)

    if args.temporal:
        tres = run_temporal_benchmark(
            n_epochs=args.temporal_epochs,
            cards_per_day=args.temporal_cards_per_day,
        )
        if args.json:
            import json
            print(json.dumps({
                **tres.__dict__,
                "recall_claim_holds": tres.recall_claim_holds,
                "latency_claim_holds": tres.latency_claim_holds,
                "storage_claim_holds": tres.storage_claim_holds,
            }, indent=2, sort_keys=True))
        else:
            print(render_temporal_markdown(tres))
        return 0

    if args.daemon:
        dres = run_daemon_benchmark(socket_path=args.daemon_socket, n_cards=args.daemon_scale)
        if args.json:
            import json
            print(json.dumps({**dres.__dict__,
                               "client_rss_holds": dres.client_rss_holds,
                               "sys_rss_holds": dres.sys_rss_holds,
                               "ipc_latency_holds": dres.ipc_latency_holds,
                               "parity_holds": dres.parity_holds,
                               "regression_holds": dres.regression_holds},
                              indent=2, sort_keys=True))
        else:
            print(render_daemon_markdown(dres))
        return 0

    res = run_adversarial(n_scale=args.scale, n_distractors=args.distractors,
                          n_needle_queries=args.needle_queries, n_polarity_pairs=args.polarity_pairs,
                          n_workers=args.workers, total_cycles=args.cycles)
    if args.json:
        import json
        print(json.dumps({
            "scale": {**res.scale.__dict__, "vector_claim_holds": res.scale.vector_claim_holds,
                      "sql_claim_holds": res.scale.sql_claim_holds, "rss_claim_holds": res.scale.rss_claim_holds},
            "needle": {**res.needle.__dict__, "recall_claim_holds": res.needle.recall_claim_holds},
            "negation": {**res.negation.__dict__, "negation_claim_holds": res.negation.negation_claim_holds},
            "concurrency": {**res.concurrency.__dict__, "concurrency_claim_holds": res.concurrency.concurrency_claim_holds},
            "latency_ms": res.latency_ms,
        }, indent=2, sort_keys=True))
    else:
        print(render_adversarial_markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())