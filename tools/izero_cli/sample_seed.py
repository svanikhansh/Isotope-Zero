"""Generate a real-schema sample Isotope Zero memory DB for izero-cli verification.

Creates a file-backed SQLite database with the EXACT `memories` schema used by
the prototypes (including the optional SQ8 quantization columns), seeds it with
a handful of cards carrying real float32 embeddings AND int8 quantized
embeddings, writes WAL by committing, and returns the path. The resulting DB is
safe to open with izero-cli's read-only URI mode.

This is a TEST FIXTURE, not shipped product code — lives outside the importable
izero_cli package so it never pollutes `pip install -e`.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
from array import array

# Matches prototypes/daemon_v0.7/isotope_zero/core/store.py exactly.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories(
  id TEXT PRIMARY KEY,
  fact TEXT NOT NULL,
  evidence TEXT,
  timestamp REAL,
  tags TEXT,
  source_tokens INTEGER DEFAULT 0,
  embedding BLOB,
  access_count INTEGER DEFAULT 0,
  last_access REAL,
  superseded_by TEXT
)
"""
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_fact ON memories(fact)",
    "CREATE INDEX IF NOT EXISTS idx_tags ON memories(tags)",
    "CREATE INDEX IF NOT EXISTS idx_memories_lookup ON memories(superseded_by, id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_memories_fact_nocase ON memories(fact COLLATE NOCASE, superseded_by, timestamp, id)",
]


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def _quantize(vec: list[float]) -> tuple[bytes, float]:
    """SQ8 symmetric scalar quantization, matching the prototype's quantize_vector."""
    import math as _m
    max_abs = max((abs(x) for x in vec), default=0.0)
    if max_abs == 0.0:
        return bytes(len(vec)), 1.0
    scale = max_abs / 127.0
    q = [max(-128, min(127, round(x / scale))) for x in vec]
    # array('b') = signed char (int8). tobytes() = packed little-endian int8.
    return array("b", q).tobytes(), float(scale)


def seed_sample_db(db_path: str | None = None) -> str:
    """Create + seed a sample DB. Returns the absolute path."""
    if db_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = tmp.name
        tmp.close()
    db_path = os.path.abspath(db_path)
    # Wipe any prior file so re-seeding is idempotent for verification.
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(db_path + suffix)
        except OSError:
            pass
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    cur = conn.cursor()
    cur.executescript(_SCHEMA)
    for idx in _INDEXES:
        cur.execute(idx)
    # SQ8 quantization columns (the simd_int8_v0.5 / quantization_v0.4 prototypes
    # add these via ALTER TABLE migration).
    cur.execute("ALTER TABLE memories ADD COLUMN q_embedding BLOB")
    cur.execute("ALTER TABLE memories ADD COLUMN q_scale REAL")

    base = os.path.getmtime(db_path) if os.path.exists(db_path) else 0.0
    now = base if base else 1700000000.0

    cards = [
        # id, fact, evidence, tags, embedding(float32, normalized), tokens, access_count, last_access, superseded_by
        ("card-001", "The user prefers dark mode for the terminal.",
         "user said: 'I love dark mode, light hurts my eyes'",
         ["preference", "ui", "theme"], _norm([0.9, 0.1, 0.2, 0.05, 0.0, 0.3]),
         8, 5, now - 120, None),
        ("card-002", "The project is written in Rust with a Python bridge via pyo3.",
         "README: 'this is a Rust project exposing a cdylib to Python'",
         ["tech", "stack", "rust"], _norm([0.0, 0.95, 0.1, 0.3, 0.0, 0.2]),
         12, 3, now - 3600, None),
        ("card-003", "The user works on AI agent memory systems.",
         "intro line: 'building isotope zero, an efficiency-first memory engine'",
         ["identity", "work"], _norm([0.1, 0.2, 0.9, 0.0, 0.1, 0.4]),
         9, 7, now - 30, None),
        ("card-004", "SQLite is used as the backing store with WAL journaling.",
         "store.py: 'PRAGMA journal_mode=WAL; synchronous=NORMAL;'",
         ["tech", "storage", "sqlite"], _norm([0.05, 0.3, 0.0, 0.92, 0.1, 0.1]),
         10, 2, now - 7200, None),
        ("card-005", "Embeddings are 384-dim float32 vectors packed via the array module.",
         "store.py: encode: array('f', vec).tobytes()",
         ["tech", "embeddings", "float32"], _norm([0.2, 0.0, 0.3, 0.0, 0.9, 0.1]),
         11, 4, now - 180, None),
        ("card-006", "The user prefers LIGHT mode now.",
         "user said: 'switch me to light mode'",
         ["preference", "ui", "change"], _norm([0.0, 0.9, 0.1, 0.05, 0.0, 0.4]),
         7, 0, 0.0, None),
        # A SUPERSEDED card (audit-trail folded) — must NOT surface in retrieval
        # but must be inspectable by `izero card`.
        ("card-001-old", "The user prefers dark mode for the terminal.",
         "superseded older copy folded into card-001",
         ["preference", "ui"], _norm([0.88, 0.12, 0.2, 0.05, 0.0, 0.3]),
         8, 1, now - 5000, "card-001"),
    ]

    for (cid, fact, evidence, tags, emb, stok, acc, la, sup) in cards:
        tags_json = json.dumps(tags) if tags else None
        emb_blob = array("f", emb).tobytes() if emb else None
        q_blob, q_scale = _quantize(emb) if emb else (None, None)
        cur.execute(
            "INSERT INTO memories(id, fact, evidence, timestamp, tags, source_tokens, "
            "embedding, access_count, last_access, superseded_by, q_embedding, q_scale) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, fact, evidence, now, tags_json, stok, emb_blob, acc, la or now, sup, q_blob, q_scale),
        )
    conn.commit()
    conn.close()
    # Touch the main file's mtime so `now` (base) makes sense for age calc.
    return db_path


def seed_large_db(db_path: str | None = None, n: int = 120) -> str:
    """Create a larger sample DB for benchmark/stats verification.

    Seeds `n` cards (default 120) with varied tags, timestamps spread across
    age buckets (<1h, <1d, <7d, >30d), access counts, and float32 embeddings.
    Has SQ8 columns but stores only float32 embeddings (so quantization status
    reads as float32). Returns the absolute path.
    """
    if db_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = tmp.name
        tmp.close()
    db_path = os.path.abspath(db_path)
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(db_path + suffix)
        except OSError:
            pass
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cur = conn.cursor()
    cur.executescript(_SCHEMA)
    for idx in _INDEXES:
        cur.execute(idx)
    cur.execute("ALTER TABLE memories ADD COLUMN q_embedding BLOB")
    cur.execute("ALTER TABLE memories ADD COLUMN q_scale REAL")

    import random as _r

    rng = _r.Random(1337)  # deterministic for reproducible benchmarks
    tags_pool = ["preference", "tech", "identity", "storage", "embeddings",
                 "ui", "rust", "sqlite", "work", "memory"]
    now = 1785900000.0  # fixed epoch for deterministic age buckets
    age_buckets = [60, 3600, 86400, 86400 * 7, 86400 * 35]  # <1h..>30d
    for i in range(n):
        # spread timestamps across age buckets for the stats histogram
        bucket = i % len(age_buckets)
        ts = now - age_buckets[bucket] - rng.uniform(0, 60)
        tags = rng.sample(tags_pool, k=rng.randint(1, 3))
        # deterministic 6-dim unit vector so norms are ~1.0
        base = [rng.uniform(-1, 1) for _ in range(6)]
        emb = _norm(base)
        fact = f"Sample fact #{i}: the agent learned about {tags[0]} and context variant {i}."
        evidence = f"evidence excerpt for card {i}"
        cur.execute(
            "INSERT INTO memories(id, fact, evidence, timestamp, tags, source_tokens, "
            "embedding, access_count, last_access, superseded_by, q_embedding, q_scale) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"card-{i:04d}", fact, evidence, ts, json.dumps(tags), len(fact) // 4,
             array("f", emb).tobytes(), rng.randint(0, 50), ts, None, None, None),
        )
    conn.commit()
    conn.close()
    return db_path


def seed_diff_pair(db1: str | None = None, db2: str | None = None) -> tuple[str, str]:
    """Create two DBs for `izero diff` verification.

    db1 = baseline (cards A, B, C, D — all live)
    db2 = after a session:
        - A unchanged, B superseded by B2 (modified), C deleted (gone entirely),
        - E added (new), D unchanged.
    So diff should report: Added=[E], Modified/Superseded=[B->B2], Deleted=[C].
    Returns (db1_path, db2_path).
    """
    if db1 is None:
        t1 = tempfile.NamedTemporaryFile(suffix=".db", delete=False); db1 = t1.name; t1.close()
    if db2 is None:
        t2 = tempfile.NamedTemporaryFile(suffix=".db", delete=False); db2 = t2.name; t2.close()
    db1 = os.path.abspath(db1); db2 = os.path.abspath(db2)
    for p in (db1, db2):
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(p + suffix)
            except OSError:
                pass

    now = 1785900000.0

    def _build(path, rows):
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        cur = conn.cursor()
        cur.executescript(_SCHEMA)
        for idx in _INDEXES:
            cur.execute(idx)
        for (cid, fact, sup, ts) in rows:
            tags_json = json.dumps(["diff", "test"])
            emb = array("f", _norm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])).tobytes()
            cur.execute(
                "INSERT INTO memories(id, fact, evidence, timestamp, tags, source_tokens, "
                "embedding, access_count, last_access, superseded_by) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (cid, fact, f"ev:{cid}", ts, tags_json, 5, emb, 0, ts, sup),
            )
        conn.commit(); conn.close()

    # baseline db1: A, B, C, D
    _build(db1, [
        ("card-A", "Alpha fact about rust.", None, now - 1000),
        ("card-B", "Beta fact about sqlite.", None, now - 900),
        ("card-C", "Gamma fact about embeddings.", None, now - 800),
        ("card-D", "Delta fact about memory.", None, now - 700),
    ])
    # session db2: A (same), B superseded by B2, C gone, D same, E new
    _build(db2, [
        ("card-A", "Alpha fact about rust.", None, now - 1000),
        ("card-B", "Beta fact about sqlite.", "card-B2", now - 900),
        ("card-B2", "Beta fact about sqlite WAL mode.", None, now - 100),
        ("card-D", "Delta fact about memory.", None, now - 700),
        ("card-E", "Epsilon fact about daemon IPC.", None, now - 50),
    ])
    return db1, db2


def seed_doctor_db(db_path: str | None = None) -> str:
    """Create a DB with deliberate anomalies for `izero doctor` verification.

    Anomalies seeded:
        - 1 zero-norm vector (all-zeros embedding) -> doctor flags null/zero-norm
        - 1 NULL embedding on a live card          -> flagged (no vector)
        - the rest healthy.
    No orphans in the strict sense (no separate vector table), so "orphaned
    vector entries" is reported as zero-norm/NULL vectors here.
    """
    if db_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = tmp.name
        tmp.close()
    db_path = os.path.abspath(db_path)
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(db_path + suffix)
        except OSError:
            pass
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cur = conn.cursor()
    cur.executescript(_SCHEMA)
    for idx in _INDEXES:
        cur.execute(idx)
    now = 1785900000.0
    rows = [
        ("healthy-1", "A healthy card.", _norm([1, 0, 0, 0, 0, 0])),
        ("healthy-2", "Another healthy card.", _norm([0, 1, 0, 0, 0, 0])),
        ("zero-norm", "A card with a zero vector.", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("null-emb", "A card with no embedding.", None),
    ]
    for (cid, fact, emb) in rows:
        emb_blob = array("f", emb).tobytes() if emb is not None else None
        cur.execute(
            "INSERT INTO memories(id, fact, evidence, timestamp, tags, source_tokens, "
            "embedding, access_count, last_access, superseded_by) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (cid, fact, f"ev:{cid}", now, json.dumps(["doctor", "test"]), 5, emb_blob, 0, now, None),
        )
    conn.commit(); conn.close()
    return db_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="sample", choices=["sample", "large", "diff", "doctor"])
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    if a.kind == "sample":
        print(seed_sample_db())
    elif a.kind == "large":
        print(seed_large_db(n=a.n))
    elif a.kind == "diff":
        d1, d2 = seed_diff_pair()
        print(d1)
        print(d2)
    elif a.kind == "doctor":
        print(seed_doctor_db())

