"""Shared SQLite helpers for the izero-cli command modules.

This consolidates the read-only safety model (re-exported from ``db.py``) and
adds the **write-access** helpers that the two mutating commands — ``import``
and ``vacuum`` — need. Read-only commands MUST go through ``open_ro``; mutating
commands MUST go through ``open_rw`` and are the only place izero-cli ever
opens a write-capable connection.

Isotope Zero schema (verified from prototypes/*/isotope_zero/core/store.py):

    memories(
        id TEXT PRIMARY KEY,
        fact TEXT NOT NULL,
        evidence TEXT,
        timestamp REAL,
        tags TEXT,            -- JSON array string
        source_tokens INTEGER DEFAULT 0,
        embedding BLOB,        -- packed float32 via array('f').tobytes()
        access_count INTEGER DEFAULT 0,
        last_access REAL,
        superseded_by TEXT
    )
    -- optional SQ8 columns (added via ALTER TABLE in quantized prototypes):
    --   q_embedding BLOB (packed int8), q_scale REAL
"""
from __future__ import annotations

import json
import os
import sqlite3
from array import array

# Re-export the read-only opener + shared decoders so command modules import
# everything from one place (a single, stable seam). These are the helpers the
# read-only commands rely on; they enforce mode=ro + query_only=ON.
from izero_cli.db import (
    open_ro,
    _safe_open,
    _table_columns,
    _parse_tags,
    _decode_float32,
    _decode_int8,
    _l2_norm,
    _human_size,
    _tables,
)

# The canonical column order for INSERTs (matches the base schema). Optional SQ8
# columns are appended only when present in the target DB (see _has_sq8).
_BASE_COLUMNS = (
    "id, fact, evidence, timestamp, tags, source_tokens, "
    "embedding, access_count, last_access, superseded_by"
)


def open_rw(db_path: str) -> sqlite3.Connection:
    """Open a SQLite database in READ-WRITE mode for the mutating commands.

    This is the deliberate counterpart to ``open_ro``. It is ONLY used by
    ``izero import`` and ``izero vacuum`` — both of which require write access
    by spec. The connection is opened with a timeout so a contending writer
    (a live agent) gets a bounded wait rather than an immediate SQLITE_BUSY.

    Raises sqlite3.Error on missing/invalid DB (callers wrap + render an error
    panel via the contract pattern). Does NOT set query_only — writes are the
    point.
    """
    abs_path = os.path.abspath(db_path)
    uri_path = abs_path.replace(" ", "%20")
    # rw: open file: URI without mode=ro. timeout gives bounded BUSY-retry.
    conn = sqlite3.connect(f"file:{uri_path}", uri=True, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def create_fresh_db(db_path: str) -> sqlite3.Connection:
    """Create a brand-new Isotope Zero memory DB (for ``import`` into a fresh path).

    Builds the canonical schema + the same indexes the prototypes use, opened
    read-write with WAL. Returns the connection (caller closes after seeding).
    """
    abs_path = os.path.abspath(db_path)
    # If the file exists, open it; else SQLite creates it via the bare path.
    conn = sqlite3.connect(abs_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        """
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
        );
        CREATE INDEX IF NOT EXISTS idx_fact ON memories(fact);
        CREATE INDEX IF NOT EXISTS idx_tags ON memories(tags);
        CREATE INDEX IF NOT EXISTS idx_memories_lookup ON memories(superseded_by, id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_memories_fact_nocase ON memories(fact COLLATE NOCASE, superseded_by, timestamp, id);
        """
    )
    return conn


def _has_sq8(conn: sqlite3.Connection) -> bool:
    """True if the DB has the SQ8 quantization columns (q_embedding, q_scale)."""
    cols = _table_columns(conn, "memories")
    return "q_embedding" in cols and "q_scale" in cols


def encode_float32(vec: list[float] | None) -> bytes | None:
    """Pack a float32 vector as a BLOB matching the prototype's array('f') format."""
    if vec is None:
        return None
    return array("f", vec).tobytes()


def insert_card(
    conn: sqlite3.Connection,
    *,
    id: str,
    fact: str,
    evidence: str | None = None,
    timestamp: float,
    tags: list[str] | None = None,
    source_tokens: int = 0,
    embedding: list[float] | None = None,
    access_count: int = 0,
    last_access: float | None = None,
    superseded_by: str | None = None,
) -> None:
    """Insert one card row. Caller manages the transaction (commit/rollback)."""
    tags_json = json.dumps(tags) if tags else None
    emb_blob = encode_float32(embedding)
    la = last_access if last_access is not None else timestamp
    conn.execute(
        f"INSERT OR REPLACE INTO memories({_BASE_COLUMNS}) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (id, fact, evidence, timestamp, tags_json, source_tokens,
         emb_blob, access_count, la, superseded_by),
    )


def db_file_size(db_path: str) -> int:
    """On-disk size of the main DB file in bytes (0 if missing)."""
    try:
        return os.path.getsize(db_path)
    except OSError:
        return 0


def wal_sidecar_sizes(db_path: str) -> tuple[int, int]:
    """Return (wal_bytes, shm_bytes) for the DB's WAL sidecars (0 if absent)."""
    try:
        wal = os.path.getsize(db_path + "-wal")
    except OSError:
        wal = 0
    try:
        shm = os.path.getsize(db_path + "-shm")
    except OSError:
        shm = 0
    return wal, shm


__all__ = [
    "open_ro", "open_rw", "create_fresh_db", "insert_card",
    "_safe_open", "_table_columns", "_parse_tags", "_decode_float32",
    "_decode_int8", "_l2_norm", "_human_size", "_tables", "_has_sq8",
    "encode_float32", "db_file_size", "wal_sidecar_sizes",
]
