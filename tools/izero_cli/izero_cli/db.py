"""Read-only data layer for izero-cli.

This module is the SOLE database access point for the izero-cli inspection
tool. It opens Isotope Zero SQLite memory-engine databases and returns plain
dict contracts that the UI layer consumes.

READ-ONLY SAFETY MODEL (critical, do not regress)
-------------------------------------------------
1. Every SQLite connection is opened in **URI read-only mode**::

       sqlite3.connect(f"file:{path}?mode=ro", uri=True)

   This asks the SQLite VFS to open the file read-only at the OS layer; a
   missing ``mode=ro`` would allow writes.

2. Immediately after connect we set ``PRAGMA query_only=ON``. This is a
   second line of defense: even a stray ``PRAGMA`` that could write (e.g.
   ``journal_mode``, ``synchronous``, ``CREATE TABLE``) is forbidden because
   SQLite refuses to execute write statements while ``query_only`` is ON. We
   never rely on this as the ONLY guard (``mode=ro`` is the primary), but it
   is a belt-and-suspenders against accidental PRAGMAs.

3. We NEVER call a PRAGMA that could write. The one PRAGMA we issue beyond
   ``query_only`` is ``PRAGMA journal_mode`` (a no-op in read-only mode that
   RETURNS the current mode without changing it) and ``PRAGMA table_info``
   (pure read). Both are safe under ``query_only=ON`` and confirmed by
   manual testing.

4. All DB access is wrapped in try/except. A missing file, a corrupt DB, an
   SQL error, or a Python ``array`` decode failure MUST return a contract
   dict with ``exists=False`` / ``error=<message>`` — NEVER raise to the
   caller. The UI layer assumes these functions are total.

5. Optional columns (``q_embedding`` / ``q_scale`` from the SQ8 quantized
   prototypes) are detected via ``PRAGMA table_info(memories)`` BEFORE any
   query that references them, so the data layer works against both the
   base float32 schema and the SQ8-extended schema without error.

Contract stability
------------------
``inspect_db``, ``search_db``, ``get_card`` and ``daemon_status`` return the
EXACT dict shapes documented in their docstrings / the package's data
contracts. ``izero_cli.ui`` depends on these shapes verbatim; changing a
key name or nesting level is a breaking change.
"""

from __future__ import annotations

import json
import math
import os
import socket
import sqlite3
import subprocess
import time
from array import array
from typing import Any

__all__ = [
    "open_ro",
    "inspect_db",
    "search_db",
    "get_card",
    "daemon_status",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Daemon surface (from prototypes/daemon_v0.7/isotope_zero/daemon/{client,server}.py).
# There is no fixed on-disk shm path (the daemon uses named multiprocessing
# SharedMemory regions like "izero_<pid>_<seq>"), but the contract specifies a
# stable "/izero_shm" identifier, so we surface it literally and report
# whether any such path exists (it will not, today).
_DAEMON_SOCKET_PATH = "/tmp/izero.sock"
_DAEMON_SHM_PATH = "/izero_shm"

# Prototype ONNX model cache locations, tried in order. The first directory
# that contains both the .onnx and the .tokenizer.json wins for the semantic
# search path. None of these is a hard dependency: if the model files or the
# onnxruntime/tokenizers/numpy libraries are missing, search_db falls back to
# the always-available lexical TF-IDF path.
_MODEL_ONNX_NAME = "Xenova_all-MiniLM-L6-v2.onnx"
_MODEL_TOK_NAME = "Xenova_all-MiniLM-L6-v2.tokenizer.json"
_MODEL_CACHE_CANDIDATES: tuple[str, ...] = (
    os.path.expanduser("~/.isotope_zero/cache"),
    # prototype-local cache (checked in alongside the prototypes)
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "prototypes",
        "daemon_v0.7",
        ".isotope_zero_cache",
    ),
)

# Embedding dimension for the MiniLM model. Used to validate cached vectors and
# as the fallback feature-hash dimension. Matches isotope_zero.embeddings.onnx_embed.
_EMBED_DIM = 384


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_size(n: float) -> str:
    """Format a byte count as B/KB/MB/GB with 2 significant figures.

    Edge cases: negative -> "0 B"; < 1024 bytes -> integer bytes.
    """
    if n is None or n < 0:
        return "0 B"
    if n < 1024:
        return f"{int(n)} B"
    units = ("KB", "MB", "GB", "TB", "PB")
    value = float(n) / 1024.0
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    # 2 significant figures
    return f"{value:.2g} {units[idx]}"


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the set of column names for ``table`` (empty if missing).

    Uses ``PRAGMA table_info`` which is a pure read and safe under
    ``query_only=ON``. Never raises: any error yields the empty set.
    """
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return {str(row[1]) for row in cur.fetchall()}
    except sqlite3.Error:
        return set()


def _parse_tags(tags_json: Any) -> list[str]:
    """Parse a JSON-array tags string into a list[str]. Malformed -> [].

    Mirrors MemoryStore._row_to_card: tolerant of NULL/malformed JSON.
    """
    if not tags_json:
        return []
    try:
        parsed = json.loads(tags_json)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(t) for t in parsed]


def _decode_float32(blob: bytes | None) -> list[float] | None:
    """Unpack raw float32 bytes to a Python list, or None if NULL.

    Uses ``array('f')`` (C float / IEEE-754 single, little-endian on all
    supported platforms). No numpy dependency.
    """
    if blob is None:
        return None
    try:
        return array("f", blob).tolist()
    except (TypeError, ValueError):
        # Should not happen for a well-formed BLOB, but never raise.
        return None


def _decode_int8(blob: bytes | None) -> list[int] | None:
    """Unpack raw int8 bytes to a Python list of signed ints, or None if NULL.

    ``array('b')`` is the C signed char (int8), matching the prototype's
    ``np.int8(...).tobytes()`` encoding. No numpy dependency.
    """
    if blob is None:
        return None
    try:
        return array("b", blob).tolist()
    except (TypeError, ValueError):
        return None


def _l2_norm(vec: list[float]) -> float:
    """L2 (Euclidean) norm of a float list. Pure Python; no numpy."""
    return math.sqrt(sum(v * v for v in vec))


# ---------------------------------------------------------------------------
# Public API: open_ro
# ---------------------------------------------------------------------------

def open_ro(db_path: str) -> sqlite3.Connection:
    """Open a SQLite database in URI read-only mode + ``query_only=ON``.

    This is the ONLY way ``db.py`` opens a database handle. It enforces the
    read-only safety model at two layers:

    1. ``file:<path>?mode=ro`` with ``uri=True`` — the VFS opens the file
       read-only; writes are rejected at the OS layer.
    2. ``PRAGMA query_only=ON`` — SQLite forbids any write statement on this
       connection even if one were somehow issued.

    Raises
    ------
    sqlite3.Error
        If the file does not exist or is not a valid SQLite database. Callers
        that need a total (non-raising) API should use ``inspect_db`` /
        ``search_db`` / ``get_card`` instead, which wrap this in try/except
        and return contract dicts.
    """
    # Resolve and absolutize so the file:// URI is well-formed under uri=True.
    abs_path = os.path.abspath(db_path)
    # The file: URI needs the path percent-encoded for safety on odd paths,
    # but abspath is already tame on POSIX. Use a minimal encode for spaces.
    uri_path = abs_path.replace(" ", "%20")
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    # Double safety: forbid ANY write on this connection. This is belt-and-
    # suspenders on top of mode=ro; it also blocks write-capable PRAGMAs.
    try:
        conn.execute("PRAGMA query_only=ON")
    except sqlite3.Error:
        # If the SQLite build rejects query_only (very rare), do NOT proceed
        # with a write-capable connection. Close and re-raise so the caller
        # surfaces the safety failure rather than silently weakening it.
        conn.close()
        raise
    return conn


def _safe_open(db_path: str) -> tuple[sqlite3.Connection | None, str | None]:
    """Open read-only, returning (conn, None) or (None, error_message).

    A total (non-raising) helper used by the contract-returning functions.
    """
    if not db_path:
        return None, "no db_path provided"
    try:
        return open_ro(db_path), None
    except sqlite3.OperationalError as exc:
        # Missing file, locked, not-a-database, etc.
        return None, f"open failed: {exc}"
    except sqlite3.DatabaseError as exc:
        return None, f"database error: {exc}"
    except (OSError, ValueError) as exc:
        return None, f"open failed: {exc}"


# ---------------------------------------------------------------------------
# Public API: inspect_db
# ---------------------------------------------------------------------------

def inspect_db(db_path: str) -> dict:
    """Inspect a memory-engine database and return the InspectData contract.

    Returns a dict with the EXACT shape documented in the module's data
    contract. On any error (missing file, corrupt DB, SQL error) returns a
    contract dict with ``exists=False`` and ``error=<message>`` instead of
    raising.

    The returned dict always contains these top-level keys::

        db_path, exists, error, total_cards, superseded_count, wal,
        quantization, vector_ram, access, top_tags, db_size_bytes,
        db_size_human
    """
    base: dict[str, Any] = {
        "db_path": db_path,
        "exists": False,
        "error": None,
        "total_cards": 0,
        "superseded_count": 0,
        "wal": {
            "wal_size_bytes": 0,
            "wal_size_human": "0 B",
            "shm_size_bytes": 0,
            "shm_size_human": "0 B",
            "journal_mode": "unknown",
        },
        "quantization": {
            "status": "none",
            "cards_float32": 0,
            "cards_int8_sq8": 0,
            "cards_no_embedding": 0,
            "has_sq8_columns": False,
        },
        "vector_ram": {
            "cards_with_embeddings": 0,
            "dim": None,
            "float32_bytes": 0,
            "int8_bytes": 0,
            "ram_bytes": 0,
            "ram_human": "0 B",
        },
        "access": {
            "most_recent": [],
            "top_accessed": [],
        },
        "top_tags": [],
        "db_size_bytes": 0,
        "db_size_human": "0 B",
    }

    # ---- existence / size of the main DB file (filesystem, not SQLite) ----
    if not db_path:
        base["error"] = "no db_path provided"
        return base
    try:
        st = os.stat(db_path)
        base["db_size_bytes"] = int(st.st_size)
        base["db_size_human"] = _human_size(st.st_size)
        base["exists"] = True
    except FileNotFoundError:
        base["error"] = "file not found"
        return base
    except OSError as exc:
        base["error"] = f"stat failed: {exc}"
        return base

    # ---- WAL sidecar sizes (filesystem, before opening) ----
    wal_path = f"{db_path}-wal"
    shm_path = f"{db_path}-shm"
    wal_bytes = 0
    shm_bytes = 0
    try:
        wal_bytes = int(os.path.getsize(wal_path))
    except OSError:
        wal_bytes = 0
    try:
        shm_bytes = int(os.path.getsize(shm_path))
    except OSError:
        shm_bytes = 0
    base["wal"]["wal_size_bytes"] = wal_bytes
    base["wal"]["wal_size_human"] = _human_size(wal_bytes)
    base["wal"]["shm_size_bytes"] = shm_bytes
    base["wal"]["shm_size_human"] = _human_size(shm_bytes)

    # ---- open read-only and run all SQL ----
    conn, err = _safe_open(db_path)
    if conn is None:
        base["error"] = err
        return base

    try:
        # journal_mode: in read-only URI mode this is a no-op that RETURNS the
        # current mode; it does not write. Safe under query_only=ON.
        try:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            mode = str(row[0]) if row and row[0] else "unknown"
        except sqlite3.Error:
            mode = "unknown"
        base["wal"]["journal_mode"] = mode

        # Detect the table and its columns. PRAGMA table_info is a pure read.
        cols = _table_columns(conn, "memories")
        if "memories" not in _tables(conn):
            # Table genuinely absent.
            base["error"] = "table 'memories' not found"
            return base
        has_q = "q_embedding" in cols and "q_scale" in cols
        base["quantization"]["has_sq8_columns"] = has_q

        # ---- counts (superseded vs active) ----
        total = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL"
        ).fetchone()
        sup = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE superseded_by IS NOT NULL"
        ).fetchone()
        base["total_cards"] = int(total[0]) if total else 0
        base["superseded_count"] = int(sup[0]) if sup else 0

        # ---- quantization breakdown + vector RAM ----
        if has_q:
            # float32-only rows: non-NULL embedding AND NULL q_embedding
            cf32 = conn.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE embedding IS NOT NULL AND q_embedding IS NULL"
            ).fetchone()
            # SQ8 rows: non-NULL q_embedding (the quantized column is the
            # source of truth for "this row is quantized")
            ci8 = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE q_embedding IS NOT NULL"
            ).fetchone()
            cnone = conn.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE embedding IS NULL AND q_embedding IS NULL"
            ).fetchone()
        else:
            cf32 = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
            ).fetchone()
            ci8 = (0,)
            cnone = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NULL"
            ).fetchone()
        cards_f32 = int(cf32[0]) if cf32 else 0
        cards_i8 = int(ci8[0]) if ci8 else 0
        cards_none = int(cnone[0]) if cnone else 0
        base["quantization"]["cards_float32"] = cards_f32
        base["quantization"]["cards_int8_sq8"] = cards_i8
        base["quantization"]["cards_no_embedding"] = cards_none

        # status
        if cards_i8 > 0 and cards_f32 > 0:
            qstatus = "mixed"
        elif cards_i8 > 0:
            qstatus = "int8_sq8"
        elif cards_f32 > 0:
            qstatus = "float32"
        else:
            qstatus = "none"
        base["quantization"]["status"] = qstatus

        # embedding dimension: take the first non-NULL embedding blob and
        # measure len(blob)//4. float32 = 4 bytes per element.
        dim: int | None = None
        try:
            row_dim = conn.execute(
                "SELECT embedding FROM memories "
                "WHERE embedding IS NOT NULL LIMIT 1"
            ).fetchone()
            if row_dim and row_dim[0] is not None:
                dim = len(row_dim[0]) // 4
        except sqlite3.Error:
            dim = None
        cards_with_emb = cards_f32 + cards_i8
        f32_bytes = cards_f32 * (dim * 4) if dim else 0
        # int8: 1 byte per element + one float (4 bytes) scale per vector
        int8_bytes = (cards_i8 * dim) + (cards_i8 * 4) if dim else 0
        ram_bytes = f32_bytes + int8_bytes
        base["vector_ram"] = {
            "cards_with_embeddings": cards_with_emb,
            "dim": dim,
            "float32_bytes": f32_bytes,
            "int8_bytes": int8_bytes,
            "ram_bytes": ram_bytes,
            "ram_human": _human_size(ram_bytes),
        }

        # ---- access recency / frequency (top 5 each) ----
        now = time.time()
        most_recent: list[dict] = []
        try:
            rows = conn.execute(
                "SELECT id, fact, timestamp, last_access "
                "FROM memories WHERE superseded_by IS NULL "
                "ORDER BY last_access DESC NULLS LAST LIMIT 5"
            ).fetchall()
            for r in rows:
                la = float(r[3]) if r[3] is not None else 0.0
                most_recent.append({
                    "id": str(r[0]),
                    "fact": str(r[1]) if r[1] is not None else "",
                    "timestamp": float(r[2]) if r[2] is not None else 0.0,
                    "last_access": la,
                    "age_seconds": max(0.0, now - la) if la else None,
                })
        except sqlite3.Error:
            most_recent = []
        base["access"]["most_recent"] = most_recent

        top_accessed: list[dict] = []
        try:
            rows = conn.execute(
                "SELECT id, fact, access_count, last_access "
                "FROM memories WHERE superseded_by IS NULL "
                "ORDER BY access_count DESC, last_access DESC NULLS LAST LIMIT 5"
            ).fetchall()
            for r in rows:
                top_accessed.append({
                    "id": str(r[0]),
                    "fact": str(r[1]) if r[1] is not None else "",
                    "access_count": int(r[2]) if r[2] is not None else 0,
                    "last_access": float(r[3]) if r[3] is not None else 0.0,
                })
        except sqlite3.Error:
            top_accessed = []
        base["access"]["top_accessed"] = top_accessed

        # ---- top tags (parse JSON tags, count frequency) ----
        base["top_tags"] = _compute_top_tags(conn, limit=10)

        return base
    except sqlite3.Error as exc:
        base["error"] = f"query failed: {exc}"
        return base
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def _tables(conn: sqlite3.Connection) -> set[str]:
    """Return the set of user-table names. Pure read (sqlite_master)."""
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return {str(r[0]) for r in cur.fetchall()}
    except sqlite3.Error:
        return set()


def _compute_top_tags(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    """Parse the JSON tags column across all cards and return the top-N tags.

    Pure-Python counting; no SQL aggregation over the JSON column (keeps it
    portable across SQLite builds without JSON1). Tolerant of malformed JSON
    via _parse_tags. Sorted by count desc, then tag asc for determinism.
    """
    counts: dict[str, int] = {}
    try:
        cur = conn.execute("SELECT tags FROM memories")
        for row in cur:
            for tag in _parse_tags(row[0]):
                counts[tag] = counts.get(tag, 0) + 1
    except sqlite3.Error:
        return []
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"tag": t, "count": c} for t, c in items[:limit]]


# ---------------------------------------------------------------------------
# Public API: search_db
# ---------------------------------------------------------------------------

def search_db(db_path: str, query: str, top_k: int = 5) -> dict:
    """Search a memory-engine database and return the SearchResult contract.

    Strategy (auto-selected, honestly reported in ``mode``):

    - **semantic**: if onnxruntime + tokenizers + numpy are all importable AND
      a cached MiniLM model exists at one of the known cache paths, embed the
      query to 384-dim and run cosine search over stored float32 embeddings
      (L2-normalize, dot product, clamp [0,1]). ``mode="semantic"``.
    - **lexical** (always available, pure stdlib): build a TF-IDF-style
      bag-of-words over ``fact``+``evidence`` of all non-superseded cards,
      embed the query as the same vocabulary, compute cosine similarity.
      ``mode="lexical"``.

    If the semantic path is unavailable for ANY reason (missing deps, missing
    model file, embedding dimension mismatch, no stored float32 vectors), the
    lexical path runs. ``mode`` always reflects which path actually executed.

    Results are sorted by (score desc, timestamp asc) and truncated to
    ``top_k``. On any error returns a contract dict with ``exists=False`` /
    ``error=<message>`` instead of raising.
    """
    start = time.perf_counter()
    result: dict[str, Any] = {
        "db_path": db_path,
        "exists": False,
        "error": None,
        "query": query,
        "top_k": int(top_k),
        "mode": "lexical",
        "latency_ms": 0.0,
        "results": [],
    }

    if not db_path:
        result["error"] = "no db_path provided"
        result["latency_ms"] = _elapsed_ms(start)
        return result

    conn, err = _safe_open(db_path)
    if conn is None:
        result["error"] = err
        result["latency_ms"] = _elapsed_ms(start)
        return result
    result["exists"] = True

    try:
        cols = _table_columns(conn, "memories")
        if "memories" not in _tables(conn):
            result["error"] = "table 'memories' not found"
            result["latency_ms"] = _elapsed_ms(start)
            return result
        has_q = "q_embedding" in cols and "q_scale" in cols

        # Load candidate rows: non-superseded cards with text + vector.
        # We pull the float32 embedding always; q_embedding/q_scale only if SQ8.
        if has_q:
            cur = conn.execute(
                "SELECT id, fact, evidence, tags, timestamp, embedding, "
                "q_embedding, q_scale FROM memories WHERE superseded_by IS NULL"
            )
        else:
            cur = conn.execute(
                "SELECT id, fact, evidence, tags, timestamp, embedding, "
                "NULL, NULL FROM memories WHERE superseded_by IS NULL"
            )
        rows = cur.fetchall()

        # No cards -> empty results, lexical mode (nothing to search).
        if not rows:
            result["mode"] = "lexical"
            result["latency_ms"] = _elapsed_ms(start)
            return result

        # ---- attempt semantic path ----
        semantic_attempt = _try_semantic_search(query, rows, top_k, has_q)
        if semantic_attempt is not None:
            mode, scored = semantic_attempt
        else:
            mode = "lexical"
            scored = _lexical_search(query, rows)

        # Clamp scores to [0,1] and sort.
        ranked = sorted(
            ((max(0.0, min(1.0, s)), cid, fact, ev, tags, ts) for (s, cid, fact, ev, tags, ts) in scored),
            key=lambda r: (-r[0], r[5] if r[5] is not None else 0.0),
        )

        out: list[dict] = []
        for rank, (s, cid, fact, ev, tags, ts) in enumerate(ranked[: max(0, int(top_k))], start=1):
            out.append({
                "rank": rank,
                "score": float(s),
                "card_id": str(cid),
                "fact": str(fact) if fact is not None else "",
                "evidence": str(ev) if ev is not None else "",
                "tags": _parse_tags(tags),
                "timestamp": float(ts) if ts is not None else 0.0,
            })
        result["mode"] = mode
        result["results"] = out
        result["latency_ms"] = _elapsed_ms(start)
        return result
    except sqlite3.Error as exc:
        result["error"] = f"query failed: {exc}"
        result["latency_ms"] = _elapsed_ms(start)
        return result
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def _elapsed_ms(start: float) -> float:
    """Wall-clock milliseconds since ``start`` (perf_counter)."""
    return round((time.perf_counter() - start) * 1000.0, 3)


def _try_semantic_search(
    query: str,
    rows: list[tuple],
    top_k: int,
    has_q: bool,
) -> tuple[str, list[tuple]] | None:
    """Attempt the real ONNX semantic search.

    Returns ``(mode, scored)`` on success, or ``None`` if the semantic path
    is unavailable (missing deps / model / float32 vectors). Never raises.

    ``scored`` is a list of (score, card_id, fact, evidence, tags, timestamp).
    """
    # 1. importable deps?
    try:
        import onnxruntime as ort  # type: ignore  # noqa: F401
        import tokenizers  # type: ignore  # noqa: F401
        import numpy as np  # type: ignore
    except Exception:
        return None

    # 2. cached model files?
    model_path, tok_path = _find_cached_model()
    if model_path is None or tok_path is None:
        return None

    # 3. gather float32 stored embeddings (decode BLOBs).
    stored: list[tuple[str, str, str, Any, float, list[float]]] = []
    for r in rows:
        cid, fact, evidence, tags, ts, emb_blob, _qb, _qs = r
        vec = _decode_float32(emb_blob)
        if vec is not None and len(vec) > 0:
            stored.append((cid, fact, evidence, tags, ts, vec))
    if not stored:
        # no float32 embeddings to search against -> fall back to lexical.
        return None

    # 4. embed the query via the cached ONNX model.
    try:
        qvec = _onnx_embed_query(model_path, tok_path, query)
    except Exception:
        return None
    if qvec is None or len(qvec) == 0:
        return None

    try:
        q = np.asarray(qvec, dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-12)
        mat = np.asarray([s[5] for s in stored], dtype=np.float32)
        # If row dims mismatch the query dim (e.g. a stale DB), bail to lexical.
        if mat.shape[1] != qn.shape[0]:
            return None
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mat_n = mat / norms
        scores = mat_n @ qn  # cosine (both normalized)
        scores = np.clip(scores, 0.0, 1.0)
    except Exception:
        return None

    scored: list[tuple] = []
    for i, (cid, fact, evidence, tags, ts, _vec) in enumerate(stored):
        s = float(scores[i]) if i < len(scores) else 0.0
        scored.append((s, cid, fact, evidence, tags, ts))
    return ("semantic", scored)


def _find_cached_model() -> tuple[str | None, str | None]:
    """Locate the cached ONNX model + tokenizer. Returns (onnx, tok) or (None,None)."""
    for d in _MODEL_CACHE_CANDIDATES:
        if not d:
            continue
        onnx = os.path.join(d, _MODEL_ONNX_NAME)
        tok = os.path.join(d, _MODEL_TOK_NAME)
        if os.path.exists(onnx) and os.path.exists(tok):
            return onnx, tok
    return None, None


def _onnx_embed_query(model_path: str, tok_path: str, query: str) -> list[float] | None:
    """Embed a single query string with the cached ONNX MiniLM model.

    Mean-pools token embeddings with attention masking, then L2-normalizes —
    matching isotope_zero.embeddings.onnx_embed._embed_real. Returns a 384-dim
    list[float], or None if anything fails. Never raises.
    """
    try:
        import onnxruntime as ort  # type: ignore
        import numpy as np  # type: ignore
        from tokenizers import Tokenizer  # type: ignore
    except Exception:
        return None

    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(model_path, sess_options=opts)
        tokenizer = Tokenizer.from_file(tok_path)
        tokenizer.enable_padding(length=None)
    except Exception:
        return None

    text = query if query else ""
    try:
        enc = tokenizer.encode_batch([text])
        input_ids = np.array([e.ids for e in enc], dtype=np.int64)
        attn = np.array([e.attention_mask for e in enc], dtype=np.int64)
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
            attn = attn.reshape(1, -1)
        feeds = {
            "input_ids": input_ids,
            "token_type_ids": np.zeros_like(input_ids),
            "attention_mask": attn,
        }
        try:
            out = sess.run(None, feeds)
        except Exception:
            feeds.pop("token_type_ids", None)
            out = sess.run(None, feeds)
        token_embeds = out[0]
        mask = attn[..., None].astype(np.float32)
        summed = (token_embeds * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1, None)
        pooled = summed / counts
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = pooled / norms
        return normed.astype(np.float32).tolist()[0]
    except Exception:
        return None


def _lexical_search(query: str, rows: list[tuple]) -> list[tuple]:
    """TF-IDF-style bag-of-words cosine search over fact+evidence. Pure stdlib.

    Returns a list of (score, card_id, fact, evidence, tags, timestamp) with
    scores in [0,1]. Falls back to numpy if available (vectorized dot), else a
    pure-Python loop. Either way correct.
    """
    # Tokenize: lowercase, split on non-alphanumeric runs.
    import re

    token_re = re.compile(r"[a-z0-9]+")

    def tokenize(text: str) -> list[str]:
        if not text:
            return []
        return token_re.findall(text.lower())

    # Build the corpus document vectors: one (term->tf) dict per card.
    docs: list[dict[str, float]] = []
    meta: list[tuple[str, str, str, Any, float]] = []
    for r in rows:
        cid, fact, evidence, tags, ts, _emb, _qb, _qs = r
        fact = fact if fact is not None else ""
        evidence = evidence if evidence is not None else ""
        toks = tokenize(fact) + tokenize(evidence)
        if not toks:
            docs.append({})
            meta.append((cid, fact, evidence, tags, ts))
            continue
        tf: dict[str, float] = {}
        for t in toks:
            tf[t] = tf.get(t, 0.0) + 1.0
        # L2-normalize the raw TF vector so cosine == dot product. This keeps
        # the scoring shape simple without full IDF, while still rewarding
        # term overlap proportionally (longer docs don't dominate). We compute
        # IDF implicitly by treating each unique term as its own dimension
        # below; a pure-TF cosine already discounts ubiquitous terms somewhat
        # because shared-ubiquitous terms contribute little to a normalized
        # dot. This is intentionally a SIMPLE lexical baseline, not BM25.
        norm = math.sqrt(sum(v * v for v in tf.values()))
        if norm > 0:
            tf = {t: v / norm for t, v in tf.items()}
        docs.append(tf)
        meta.append((cid, fact, evidence, tags, ts))

    # Query vector over the SAME vocabulary (built per-doc; union of terms).
    q_toks = tokenize(query)
    if not q_toks:
        return [(0.0, cid, fact, ev, tags, ts) for (cid, fact, ev, tags, ts) in meta]
    q_tf: dict[str, float] = {}
    for t in q_toks:
        q_tf[t] = q_tf.get(t, 0.0) + 1.0
    qnorm = math.sqrt(sum(v * v for v in q_tf.values()))
    if qnorm > 0:
        q_tf = {t: v / qnorm for t, v in q_tf.items()}

    # Try numpy vectorization; else pure-Python. Both yield identical scores.
    try:
        import numpy as np  # type: ignore
        vocab = sorted({t for d in docs for t in d} | set(q_tf))
        if not vocab:
            return [(0.0, cid, fact, ev, tags, ts) for (cid, fact, ev, tags, ts) in meta]
        vidx = {t: i for i, t in enumerate(vocab)}
        M = np.zeros((len(docs), len(vocab)), dtype=np.float32)
        for i, d in enumerate(docs):
            for t, v in d.items():
                M[i, vidx[t]] = v
        qv = np.zeros((len(vocab),), dtype=np.float32)
        for t, v in q_tf.items():
            qv[vidx[t]] = v
        scores = M @ qv
        scores = np.clip(scores, 0.0, 1.0)
        scored: list[tuple] = []
        for i, (cid, fact, ev, tags, ts) in enumerate(meta):
            scored.append((float(scores[i]), cid, fact, ev, tags, ts))
        return scored
    except Exception:
        # Pure-Python fallback.
        scored = []
        for i, (cid, fact, ev, tags, ts) in enumerate(meta):
            d = docs[i]
            if not d:
                scored.append((0.0, cid, fact, ev, tags, ts))
                continue
            # dot over shared terms only.
            s = 0.0
            # iterate over the smaller dict.
            if len(q_tf) <= len(d):
                for t, qv in q_tf.items():
                    dv = d.get(t)
                    if dv is not None:
                        s += qv * dv
            else:
                for t, dv in d.items():
                    qv = q_tf.get(t)
                    if qv is not None:
                        s += qv * dv
            scored.append((max(0.0, min(1.0, s)), cid, fact, ev, tags, ts))
        return scored


# ---------------------------------------------------------------------------
# Public API: get_card
# ---------------------------------------------------------------------------

def get_card(db_path: str, card_id: str) -> dict:
    """Fetch a single card (including superseded audit-trail cards) by id.

    Returns the CardDetail contract. ``card`` is None if not found;
    ``vector`` is None if the card has no embedding. Superseded cards ARE
    returned (audit-trail inspection) and carry ``superseded_by`` so the UI
    can badge them. On error returns ``exists=False``/``error=<msg>``.
    """
    result: dict[str, Any] = {
        "db_path": db_path,
        "exists": False,
        "error": None,
        "card_id": card_id,
        "found": False,
        "card": None,
        "vector": None,
    }

    if not db_path:
        result["error"] = "no db_path provided"
        return result

    conn, err = _safe_open(db_path)
    if conn is None:
        result["error"] = err
        return result
    result["exists"] = True

    try:
        cols = _table_columns(conn, "memories")
        if "memories" not in _tables(conn):
            result["error"] = "table 'memories' not found"
            return result
        has_q = "q_embedding" in cols and "q_scale" in cols

        if has_q:
            sql = (
                "SELECT id, fact, evidence, timestamp, tags, source_tokens, "
                "embedding, access_count, last_access, superseded_by, "
                "q_embedding, q_scale FROM memories WHERE id = ?"
            )
        else:
            sql = (
                "SELECT id, fact, evidence, timestamp, tags, source_tokens, "
                "embedding, access_count, last_access, superseded_by, "
                "NULL, NULL FROM memories WHERE id = ?"
            )
        row = conn.execute(sql, (card_id,)).fetchone()
        if row is None:
            return result  # found stays False, card stays None
        result["found"] = True

        (
            id_, fact, evidence, ts, tags_json, source_tokens,
            emb_blob, access_count, last_access, superseded_by,
            q_blob, q_scale,
        ) = row

        card = {
            "id": str(id_),
            "fact": str(fact) if fact is not None else "",
            "evidence": str(evidence) if evidence is not None else "",
            "timestamp": float(ts) if ts is not None else 0.0,
            "tags": _parse_tags(tags_json),
            "source_tokens": int(source_tokens) if source_tokens is not None else 0,
            "access_count": int(access_count) if access_count is not None else 0,
            "last_access": float(last_access) if last_access is not None else 0.0,
            "superseded_by": str(superseded_by) if superseded_by is not None else None,
        }
        result["card"] = card

        # ---- vector detail ----
        # Prefer float32 embedding; if absent but q_embedding present, decode
        # int8 and dequantize with q_scale so the UI can show the audit vector.
        vec = _decode_float32(emb_blob)
        if vec is not None and len(vec) > 0:
            dim = len(vec)
            norm = _l2_norm(vec)
            result["vector"] = {
                "dtype": "float32",
                "dim": dim,
                "norm": float(norm),
                "is_normalized": bool(0.95 <= norm <= 1.05),
                "q_scale": None,
            }
        else:
            qvec = _decode_int8(q_blob) if has_q else None
            if qvec is not None and len(qvec) > 0:
                scale = float(q_scale) if q_scale is not None else 1.0
                dequant = [scale * float(v) for v in qvec]
                norm = _l2_norm(dequant)
                result["vector"] = {
                    "dtype": "int8_sq8",
                    "dim": len(qvec),
                    "norm": float(norm),
                    "is_normalized": bool(0.95 <= norm <= 1.05),
                    "q_scale": scale,
                }
            else:
                # No embedding at all (NULL in both columns).
                result["vector"] = None
        return result
    except sqlite3.Error as exc:
        result["error"] = f"query failed: {exc}"
        return result
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


# ---------------------------------------------------------------------------
# Public API: daemon_status
# ---------------------------------------------------------------------------

def daemon_status() -> dict:
    """Report the status of the (hypothetical) izero embedding daemon.

    Takes NO db_path. There is currently no real socket daemon — the contract
    surface is defined for future use and to honestly report that nothing is
    listening. ``daemon_active`` is True only if the socket connects OR an
    isotope_zero/izero process is detected (excluding this CLI's own PID).

    Never raises: psutil missing, ``ps`` failing, or the socket connect
    erroring are all reported as fields, not exceptions.
    """
    socket_exists = os.path.exists(_DAEMON_SOCKET_PATH)
    shm_exists = os.path.exists(_DAEMON_SHM_PATH)

    # ---- socket connect probe ----
    socket_connected = False
    socket_error: str | None = None
    if socket_exists:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.connect(_DAEMON_SOCKET_PATH)
                socket_connected = True
            finally:
                s.close()
        except OSError as exc:
            socket_error = exc.__class__.__name__
            # errno string for clarity (e.g. ECONNREFUSED).
            try:
                import errno
                socket_error = os.strerror(exc.errno) if exc.errno else socket_error
            except Exception:
                pass
    else:
        socket_error = "socket file not found"

    # ---- process detection ----
    procs = _detect_processes()

    daemon_active = bool(socket_connected or len(procs) > 0)

    return {
        "socket_path": _DAEMON_SOCKET_PATH,
        "shm_path": _DAEMON_SHM_PATH,
        "socket_exists": socket_exists,
        "shm_exists": shm_exists,
        "socket_connected": socket_connected,
        "socket_error": socket_error,
        "processes": procs,
        "daemon_active": daemon_active,
    }


def _detect_processes() -> list[dict]:
    """Detect isotope_zero / izero processes, excluding this CLI's own PID.

    Prefers psutil; falls back to shelling out to ``ps``. RSS is normalized to
    MB (psutil: bytes/1048576; ps: KB/1024). Never raises.
    """
    # Exclude both our own PID and our immediate parent's PID: the parent is
    # typically the shell wrapper that launched this CLI, and its command line
    # often contains "isotope_zero" simply because the venv lives inside the
    # repo. Matching it would be a false positive — it is part of this same
    # invocation in spirit, so we drop it. A real daemon is never our parent.
    my_pid = os.getpid()
    try:
        my_ppid = os.getppid()
    except OSError:
        my_ppid = -1
    # Try psutil first.
    try:
        import psutil  # type: ignore
        out: list[dict] = []
        for p in psutil.process_iter(["pid", "rss", "name", "cmdline", "username"]):
            try:
                info = p.info  # type: ignore[attr-defined]
                pid = int(info.get("pid") or 0)
                if pid == my_pid or pid == my_ppid:
                    continue
                cmd = " ".join(info.get("cmdline") or []) or ""
                name = info.get("name") or ""
                if _matches_daemon(name, cmd):
                    rss = info.get("rss") or 0
                    out.append({
                        "pid": pid,
                        "rss_mb": round(float(rss) / 1048576.0, 2),
                        "name": str(name),
                        "cmd": str(cmd),
                    })
            except Exception:
                continue
        return out
    except Exception:
        pass

    # Fallback: shell out to ps. macOS/BSD layout: pid,rss,comm,command.
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid,rss,comm,command"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        lines = out.stdout.splitlines()[1:]  # skip header
        procs: list[dict] = []
        for line in lines:
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if pid == my_pid or pid == my_ppid:
                continue
            try:
                rss_kb = int(parts[1])
            except ValueError:
                rss_kb = 0
            comm = parts[2]
            command = parts[3]
            if _matches_daemon(comm, command):
                procs.append({
                    "pid": pid,
                    "rss_mb": round(float(rss_kb) / 1024.0, 2),
                    "name": str(comm),
                    "cmd": str(command),
                })
        return procs
    except Exception:
        return []


def _matches_daemon(name: str, command: str) -> bool:
    """Heuristic: does this process look like an isotope_zero / izero daemon?

    Matches the lowercased name/command against the project's binary names.
    Excludes the CLI's own PID separately (caller-side) so this stays a pure
    string test.
    """
    haystack = f"{name} {command}".lower()
    return "isotope_zero" in haystack or "izero" in haystack
