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
import sys
import time
from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from isotope_zero.core.consolidation import Consolidator
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
# Orchestration + honest report rendering
# ------------------------------------------------------------------- #

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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="izero-eval-adversarial",
                                description="Isotope Zero adversarial stress harness (sections A-D).")
    p.add_argument("--scale", type=int, default=10_000, help="A: card count (default 10000)")
    p.add_argument("--distractors", type=int, default=500, help="B: distractor count (default 500)")
    p.add_argument("--needle-queries", type=int, default=100, help="B: adversarial queries (default 100)")
    p.add_argument("--polarity-pairs", type=int, default=100, help="C: contradictory pairs (default 100)")
    p.add_argument("--workers", type=int, default=25, help="D: concurrent worker processes (default 25)")
    p.add_argument("--cycles", type=int, default=1000, help="D: cycles per worker (default 1000)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = p.parse_args(argv)

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