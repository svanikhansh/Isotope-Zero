"""Persistent memory store for isotope_zero.

A SQLite-backed store for `MemoryCard` objects. Stdlib-first: sqlite3, array,
json, os, threading. Embeddings are packed as raw float32 bytes via the
`array` module. `vector_search` uses a numpy-vectorized batch dot product over
a lazily-cached `(n_cards, dim)` float32 matrix WHEN numpy is importable (the
optional accelerator path); otherwise it transparently falls back to a
plain-Python dot-product loop so the store still runs with zero dependencies.

Connection strategy (documented):
    A SINGLE persistent `sqlite3.Connection` is held on the instance for the
    whole life of the `MemoryStore`, for BOTH the `:memory:` and file-backed
    cases. It is opened with `check_same_thread=False` and every public
    method acquires a `threading.Lock` around its DB work, so concurrent
    callers serialize on the lock (thread-safe-ish).

    Why one persistent connection for `:memory:`? An in-memory SQLite DB is
    tied to a single connection's lifetime — opening/closing per call would
    wipe the data between calls. Keeping one connection alive on the
    instance keeps the in-memory DB alive for the store's lifetime, which is
    exactly what a prototype needs. (Equivalently one could use a shared
    URI like `file:isotope_zero?mode=memory&cache=shared`; a bare held
    connection is simpler and sufficient.)

    Why one persistent connection for file-backed DBs? Simpler, no
    re-open/re-warm overhead, and fine for the read/write volumes a isotope_zero
    prototype targets. Per-call connections would also work but add nothing
    here.

Embedding blob format (documented):
    Embeddings are stored as packed float32 little-endian bytes using the
    stdlib `array` module:
        encode: array('f', vec).tobytes()
        decode: array('f', blob).tolist()
    This keeps numpy out of the hard-dependency set. `array('f')` uses the
    host float format which on every platform we support is IEEE-754
    little-endian float32. A card with `embedding is None` stores SQL NULL.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from array import array
from typing import Any

from isotope_zero.types import MemoryCard, now_ts

log = logging.getLogger("isotope_zero.store")

# Allowed lookup fields for sql_lookup, mapped to their DB column.
_SQL_LOOKUP_COLUMNS: dict[str, str] = {
    "fact": "fact",
    "evidence": "evidence",
    # "tags" handled separately (JSON array membership).
}

# Column list shared by every read-path SELECT. Kept as a single constant so
# all lookup statements select exactly the same columns in the same order,
# which guarantees `_row_to_card` produces identical cards regardless of which
# query path ran.
_SQL_LOOKUP_SELECT = (
    "id, fact, evidence, timestamp, tags, source_tokens, "
    "embedding, access_count, last_access"
)

# Exact-match fast path (see MemoryStore.sql_lookup). Uses `fact = ?` with
# COLLATE NOCASE so the `idx_memories_fact_nocase` index is usable.
_SQL_EXACT = (
    "SELECT {select} FROM memories "
    "WHERE superseded_by IS NULL AND {col} = ? COLLATE NOCASE "
    "ORDER BY timestamp ASC, id ASC"
)

# Substring LIKE fast path for `fact`. A leading-wildcard LIKE cannot use a
# b-tree for range constraints, but scanning `idx_fact` (which holds only the
# `fact` text + rowid, i.e. NO 384-float embedding blobs) and testing the LIKE
# against the indexed value is far cheaper than a full table scan that must
# slurp each row's embedding blob. Matching rowids are then fetched by id.
_SQL_SUBSTR_IDS = "SELECT id FROM memories INDEXED BY idx_fact WHERE {col} LIKE ? COLLATE NOCASE"

# Direct LIKE scan used for fields without a covering index (`evidence`) and
# as a fallback when the exact+two-phase approach yields too many matches.
# `NOT INDEXED` forces a plain sequential table scan: with the new indexes
# present SQLite's planner prefers an index-scan + rowid-lookup plan that is
# *slower* for an unconstrained LIKE, so we pin the cheaper table scan.
_SQL_SUBSTR_SCAN = (
    "SELECT {select} FROM memories NOT INDEXED "
    "WHERE {col} LIKE ? COLLATE NOCASE AND superseded_by IS NULL "
    "ORDER BY timestamp ASC, id ASC"
)

# Fetch full rows by id, re-applying the superseded-by audit filter and the
# canonical ordering. Reuses `batch_get`'s ordering (timestamp, id).
_SQL_IN_FETCH = (
    "SELECT {select} FROM memories "
    "WHERE id IN ({placeholders}) AND superseded_by IS NULL "
    "ORDER BY timestamp ASC, id ASC"
)

# Python-side tag membership scan (JSON array membership is not indexable).
_SQL_TAGS_SCAN = (
    "SELECT {select} FROM memories NOT INDEXED "
    "WHERE superseded_by IS NULL ORDER BY timestamp ASC, id ASC"
)

# If an id-set from the covering idx_fact scan exceeds this many rows we fall
# back to the single-pass LIKE scan rather than building a huge IN(...) list.
_SUBSET_MATCH_LIMIT = 900


class MemoryStore:
    """SQLite-backed store of `MemoryCard` records with SQL + vector lookup.

    See module docstring for the connection and embedding-blob strategies.

    **REQUIRED for multi-process / multi-connection deployments:** this store
    never sets a ``busy_timeout`` on any connection it opens, so SQLite's
    default of 0 applies — a writer that hits a locked database raises
    ``sqlite3.OperationalError: database is locked`` immediately. The store's
    own held connection is single-threaded (serialized by its lock) and so
    never needs one, but every EXTERNAL writer/consolidator sharing the DB
    file MUST run ``PRAGMA busy_timeout=<ms>`` (e.g. 5000) on its connection to
    wait out transient locks instead of failing under contention.
    """

    def __init__(self, db_path: str = ":memory:", embedder: Any = None) -> None:
        """Create/open the store.

        Args:
            db_path: SQLite path. ":memory:" keeps data in RAM for the life
                of this store (one held connection). Any other string is
                treated as a file path (opened/created on disk).
            embedder: Optional embeddings engine with `.embed_text(text) ->
                list[float]`. Stored on the instance for callers/router use;
                this class itself never calls it (cards arrive with their
                embedding already populated, or None).
        """
        self.db_path = db_path
        self.embedder = embedder
        self._is_memory = db_path == ":memory:"

        # One persistent connection for the life of the store. For
        # ":memory:" this is what keeps the in-memory DB alive across
        # method calls; for file-backed it's just the simplest choice.
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we drive txn via lock + explicit BEGIN where needed
        )
        self._lock = threading.Lock()
        # Prepared-statement cache: one cursor per SQL string, reused across
        # calls so repeated lookups skip SQL re-compilation. Every access is
        # under `self._lock` (all sql_lookup/read-path callers hold it), so
        # the shared cursors are never used concurrently.
        self._stmt_cache: dict[str, sqlite3.Cursor] = {}
        self._init_schema()

        # Vector-search cache: a lazily-refreshed (n_cards, dim) float32 matrix
        # of the non-NULL embeddings plus parallel id/timestamp metadata, used
        # by the numpy fast path of `vector_search`. Rebuilt on demand whenever
        # `_vec_dirty` is set (see `_mark_vec_dirty`). `_vec_hetero` latches
        # when rows have heterogeneous embedding lengths (not matrix-able) so
        # the search falls back to the pure-Python loop instead of re-probing.
        self._vec_matrix = None
        self._vec_ids = None
        self._vec_ts = None
        self._vec_dirty = True
        self._vec_hetero = False

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        """Create table + indexes if absent. Idempotent + migratory.

        For file-backed DBs we enable WAL (write-ahead logging) so that the
        Phase 3 background consolidation worker can sweep without blocking
        the read/write hot path: WAL permits concurrent readers alongside a
        single writer, drastically reducing lock contention versus the
        default rollback-journal mode. For ":memory:" WAL is a no-op.
        """
        with self._lock:
            cur = self._conn.cursor()
            # WAL for file-backed stores. ":memory:" ignores PRAGMA journal_mode.
            if not self._is_memory:
                try:
                    cur.execute("PRAGMA journal_mode=WAL;")
                    cur.execute("PRAGMA synchronous=NORMAL;")  # safe + fast under WAL
                except sqlite3.OperationalError:
                    pass  # WAL unsupported (rare) — silently fall back.
            # Local read-path tuning. These are per-connection settings applied
            # to the store's held connection. Each is a no-op (or harmless) on
            # ":memory:", and the try/except mirrors the WAL block so exotic
            # SQLite builds that reject a PRAGMA don't break construction.
            # We do NOT change journal_mode or synchronous semantics here.
            try:
                # 64MB page cache (negative value = KiB, not pages). Larger
                # cache = fewer page reads on the hot read path for both
                # file-backed and in-memory stores (in-memory pages live in
                # the page cache).
                cur.execute("PRAGMA cache_size = -64000;")
                # 256MB memory-mapped I/O for file-backed stores: the OS maps
                # the DB file into the address space so reads skip the
                # read()/copy path. SQLite only maps what fits in this budget;
                # on ":memory:" this is a no-op.
                cur.execute("PRAGMA mmap_size = 268435456;")
                # Use RAM (not a temp file) for temp tables/b-trees — e.g. the
                # "USE TEMP B-TREE FOR ORDER BY" sorts on the read path. Safe
                # here: the read path holds the store lock and result sorts
                # are bounded by match set size, not DB size.
                cur.execute("PRAGMA temp_store = MEMORY;")
            except sqlite3.OperationalError:
                pass  # PRAGMA rejected by an unusual build — silently skip.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memories(
                  id TEXT PRIMARY KEY,
                  fact TEXT NOT NULL,
                  evidence TEXT,
                  timestamp REAL,
                  tags TEXT,           -- JSON array string, e.g. '["a","b"]'
                  source_tokens INTEGER DEFAULT 0,
                  embedding BLOB,      -- packed float32 via array('f')
                  access_count INTEGER DEFAULT 0,  -- Phase 3: recall frequency
                  last_access REAL,                -- Phase 3: last read timestamp
                  superseded_by TEXT               -- Phase 3: audit trail; non-NULL => folded into that survivor's id
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_fact ON memories(fact)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tags ON memories(tags)")
            # Exact-match lookup index. Uses `superseded_by, id, timestamp`
            # so lookups filtered by the superseded-by audit trail are served
            # from the index, and a `superseded_by IS NULL` scan is index-
            # ordered (ready for the ORDER BY timestamp/id ordering) without
            # reading embedding blobs from rows.
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_lookup "
                "ON memories(superseded_by, id, timestamp)"
            )
            # NOCASE index over `fact` that ALSO carries (superseded_by,
            # timestamp, id). This is what makes the exact-match fast path in
            # `sql_lookup` genuinely index-usable: `fact = ? COLLATE NOCASE`
            # matches the NOCASE collation of this index (plain `idx_fact` is
            # BINARY-collated and is therefore NOT usable for a NOCASE
            # equality), and for a fixed `fact` the index is ordered
            # (superseded_by, timestamp, id) so it satisfies
            # `ORDER BY timestamp, id` with no temp sort.
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_fact_nocase "
                "ON memories(fact COLLATE NOCASE, superseded_by, timestamp, id)"
            )
            # Phase 3 migration: add access-tracking columns to pre-existing DBs
            # that predate them. `PRAGMA table_info` lets us detect absence
            # without a try/except per column.
            cols = {row[1] for row in cur.execute("PRAGMA table_info(memories)").fetchall()}
            if "access_count" not in cols:
                cur.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0")
            if "last_access" not in cols:
                cur.execute("ALTER TABLE memories ADD COLUMN last_access REAL")
            if "superseded_by" not in cols:
                cur.execute("ALTER TABLE memories ADD COLUMN superseded_by TEXT")
            # FTS5 virtual table for lexical pre-filtering in hybrid_vector_search.
            # `card_id` is UNINDEXED (we only ever look it up, never MATCH on it);
            # `content` is the BM25-ranked text surface = fact + " " + evidence, so
            # the lexical index matches what users recall by. `porter unicode61`
            # stem-strips (so "running"/"ran" fold together) and is unicode-aware.
            # IF NOT EXISTS makes a re-open a no-op on DBs that already have it.
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(
                    card_id UNINDEXED,
                    content,
                    tokenize='porter unicode61'
                )
                """
            )
            # Migration safety: backfill rows into cards_fts for existing DBs that
            # predate the FTS table. The `id NOT IN (SELECT card_id ...)` guard
            # makes a re-run a pure no-op (only rows missing from the FTS index get
            # inserted). Only non-superseded cards are indexed (folded/audit-trail
            # rows must never surface in lexical retrieval, mirroring the vector
            # and SQL paths' `superseded_by IS NULL` filter).
            cur.execute(
                """
                INSERT INTO cards_fts(card_id, content)
                SELECT id, fact || COALESCE(' ' || evidence, '')
                FROM memories
                WHERE id NOT IN (SELECT card_id FROM cards_fts)
                  AND superseded_by IS NULL
                """
            )
            cur.close()

    # ------------------------------------------------------------------ #
    # (De)serialization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _encode_embedding(vec: list[float] | None) -> bytes | None:
        """Pack a float list as raw float32 bytes, or None → SQL NULL."""
        if vec is None:
            return None
        # array('f') = C float (IEEE-754 single precision). On all supported
        # platforms this is little-endian; tobytes() gives the packed form.
        return array("f", vec).tobytes()

    @staticmethod
    def _decode_embedding(blob: bytes | None) -> list[float] | None:
        """Unpack raw float32 bytes back to a Python list, or None if NULL."""
        if blob is None:
            return None
        return array("f", blob).tolist()

    @staticmethod
    def _row_to_card(row: tuple[Any, ...]) -> MemoryCard:
        """Build a MemoryCard from a SELECT row tuple in column order.

        Column order must match the SELECT list used everywhere: id, fact,
        evidence, timestamp, tags, source_tokens, embedding, access_count,
        last_access. Older SELECTs that don't include the last two columns
        are handled by length-checking the row (defensive against any stale
        call sites).
        """
        (
            id_,
            fact,
            evidence,
            timestamp,
            tags_json,
            source_tokens,
            embedding_blob,
        ) = row[:7]
        access_count = int(row[7]) if len(row) > 7 and row[7] is not None else 0
        last_access = float(row[8]) if len(row) > 8 and row[8] is not None else 0.0
        superseded_by = str(row[9]) if len(row) > 9 and row[9] is not None else None
        tags: list[str] = []
        if tags_json:
            try:
                parsed = json.loads(tags_json)
                if isinstance(parsed, list):
                    tags = [str(t) for t in parsed]
            except (json.JSONDecodeError, TypeError):
                # Malformed tags JSON — treat as no tags rather than crash.
                tags = []
        return MemoryCard(
            id=id_,
            fact=fact,
            evidence=evidence if evidence is not None else "",
            timestamp=float(timestamp) if timestamp is not None else 0.0,
            tags=tags,
            embedding=MemoryStore._decode_embedding(embedding_blob),
            source_tokens=int(source_tokens) if source_tokens is not None else 0,
            access_count=access_count,
            last_access=last_access,
            superseded_by=superseded_by,
        )

    # ------------------------------------------------------------------ #
    # FTS5 sync helpers (callers MUST hold self._lock + pass their cursor)
    # ------------------------------------------------------------------ #
    def _fts_upsert(self, cur: sqlite3.Cursor, card: MemoryCard) -> None:
        """Mirror a card's text into the `cards_fts` index.

        Called from `add`/`update` (and re-runnable from any write path) with
        the ALREADY-OPEN cursor the write holds, so we neither re-acquire the
        lock nor open a second cursor. Delete-then-insert makes this an
        idempotent upsert that handles both insert and overwrite uniformly
        (FTS5 has no native upsert and a stale row from a prior content would
        otherwise linger).

        Superseded (folded/audit-trail) cards are NEVER searchable: if
        `card.superseded_by` is set we only DELETE the FTS row (if any) and
        skip the insert, so a card that just got superseded via `update` is
        dropped from the lexical index in the same write. The indexed content
        is `fact + " " + evidence` (evidence None -> just fact), matching the
        BM25 surface users recall by.
        """
        if card.superseded_by is not None:
            cur.execute("DELETE FROM cards_fts WHERE card_id = ?", (card.id,))
            return
        content = card.fact + " " + (card.evidence or "")
        cur.execute("DELETE FROM cards_fts WHERE card_id = ?", (card.id,))
        cur.execute(
            "INSERT INTO cards_fts(card_id, content) VALUES (?, ?)",
            (card.id, content),
        )

    def _fts_delete(self, cur: sqlite3.Cursor, card_id: str) -> None:
        """Drop a card's row from the `cards_fts` index (mirrors `delete`)."""
        cur.execute("DELETE FROM cards_fts WHERE card_id = ?", (card_id,))

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #
    def add(self, card: MemoryCard) -> None:
        """Persist a card + its embedding (BLOB, or NULL if embedding is None)."""
        log.debug("add id=%s fact=%r emb=%d", card.id, card.fact[:80], len(card.embedding) if card.embedding else 0)
        tags_json = json.dumps(list(card.tags)) if card.tags else None
        blob = self._encode_embedding(card.embedding)
        # A fresh write is itself a touch: default last_access to timestamp
        # when the caller didn't set it, so the decay scorer sees a vital card.
        last_access = card.last_access if card.last_access else card.timestamp
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO memories(id, fact, evidence, timestamp, tags, source_tokens, embedding, access_count, last_access, superseded_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    card.id,
                    card.fact,
                    card.evidence,
                    card.timestamp,
                    tags_json,
                    card.source_tokens,
                    blob,
                    card.access_count,
                    last_access,
                    card.superseded_by,
                ),
            )
            self._fts_upsert(cur, card)
            cur.close()
        self._mark_vec_dirty()

    def update(self, card: MemoryCard) -> None:
        """Upsert a card by id. If absent, inserts; if present, overwrites."""
        tags_json = json.dumps(list(card.tags)) if card.tags else None
        blob = self._encode_embedding(card.embedding)
        last_access = card.last_access if card.last_access else card.timestamp
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO memories(id, fact, evidence, timestamp, tags, source_tokens, embedding, access_count, last_access, superseded_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    fact = excluded.fact,
                    evidence = excluded.evidence,
                    timestamp = excluded.timestamp,
                    tags = excluded.tags,
                    source_tokens = excluded.source_tokens,
                    embedding = excluded.embedding,
                    access_count = excluded.access_count,
                    last_access = excluded.last_access,
                    superseded_by = excluded.superseded_by
                """,
                (
                    card.id,
                    card.fact,
                    card.evidence,
                    card.timestamp,
                    tags_json,
                    card.source_tokens,
                    blob,
                    card.access_count,
                    last_access,
                    card.superseded_by,
                ),
            )
            self._fts_upsert(cur, card)
            cur.close()
        self._mark_vec_dirty()

    def delete(self, memory_id: str) -> bool:
        """Delete by id. Returns True if a row was deleted, False if not found."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            deleted = cur.rowcount > 0
            self._fts_delete(cur, memory_id)
            cur.close()
        log.debug("delete id=%s deleted=%s", memory_id, deleted)
        self._mark_vec_dirty()
        return deleted

    # ------------------------------------------------------------------ #
    # Phase 3: access tracking + batch consolidation
    # ------------------------------------------------------------------ #
    def touch(self, memory_id: str, at: float | None = None) -> None:
        """Record a recall: increment access_count and set last_access.

        Called by the query router whenever a card surfaces as a hit, so the
        temporal-decay scorer can distinguish recalled (vital) cards from
        never-recalled (cold) ones. `at` defaults to now; tests pass a fixed
        timestamp for determinism. Idempotent on missing ids (no-op).
        """
        ts = now_ts() if at is None else at
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_access = ?
                WHERE id = ?
                """,
                (ts, memory_id),
            )
            cur.close()
        log.debug("touch id=%s at=%.3f", memory_id, ts)

    def batch_get(self, ids: list[str]) -> list[MemoryCard]:
        """Fetch many cards by id in one round-trip (consolidation helper)."""
        if not ids:
            return []
        # SQLite parameter limit is large; for prototype scales this is fine.
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                SELECT id, fact, evidence, timestamp, tags, source_tokens, embedding, access_count, last_access
                FROM memories WHERE id IN ({placeholders})
                ORDER BY timestamp ASC, id ASC
                """,
                tuple(ids),
            )
            rows = cur.fetchall()
            cur.close()
        return [self._row_to_card(r) for r in rows]

    def consolidate_memories(
        self,
        merged_cards: list[MemoryCard],
        deleted_ids: list[str],
        superseded_ids: dict[str, str] | None = None,
    ) -> int:
        """Apply a consolidation sweep atomically.

        - Upserts each `merged_cards` survivor (a merged/updated card; the
          consolidation engine is responsible for combining evidence and
          picking the survivor id). These overwrite the existing row for that
          id (typically one of the duplicate pair, now holding merged content).
        - Marks each id in `superseded_ids` (map: folded_id -> survivor_id) as
          superseded: the row is KEPT as an audit trail, with `superseded_by`
          set to the survivor it was folded into. Superseded rows never
          surface in `all()` / lookups / retrieval.
        - Deletes every id in `deleted_ids` (decay-pruned cards; folded cards
          are superseded, not hard-deleted).

        Everything runs in ONE explicit transaction so a consolidation sweep
        is atomic: either the whole merge+prune commits or nothing does,
        which keeps the DB consistent even if the worker is interrupted. The
        store already uses a single held connection + lock; this just bounds
        the transaction explicitly so concurrent readers (WAL) never see a
        half-applied sweep.

        Returns the number of rows hard-deleted (superseded rows remain in the
        table for the audit trail and are not counted).
        """
        if not merged_cards and not deleted_ids and not superseded_ids:
            return 0
        deleted = 0
        log.debug(
            "consolidate_memories merged=%d delete_ids=%d superseded=%d",
            len(merged_cards), len(deleted_ids), len(superseded_ids or {}),
        )
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                # Upsert survivors.
                for card in merged_cards:
                    tags_json = json.dumps(list(card.tags)) if card.tags else None
                    blob = self._encode_embedding(card.embedding)
                    last_access = card.last_access if card.last_access else card.timestamp
                    cur.execute(
                        """
                        INSERT INTO memories(id, fact, evidence, timestamp, tags, source_tokens, embedding, access_count, last_access, superseded_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            fact = excluded.fact,
                            evidence = excluded.evidence,
                            timestamp = excluded.timestamp,
                            tags = excluded.tags,
                            source_tokens = excluded.source_tokens,
                            embedding = excluded.embedding,
                            access_count = excluded.access_count,
                            last_access = excluded.last_access,
                            superseded_by = excluded.superseded_by
                        """,
                        (
                            card.id,
                            card.fact,
                            card.evidence,
                            card.timestamp,
                            tags_json,
                            card.source_tokens,
                            blob,
                            card.access_count,
                            last_access,
                            card.superseded_by,
                        ),
                    )
                # Supersession audit trail: keep the folded row, point it at
                # the survivor it folded into.
                if superseded_ids:
                    for mid, survived_by in superseded_ids.items():
                        cur.execute(
                            "UPDATE memories SET superseded_by = ? WHERE id = ?",
                            (survived_by, mid),
                        )
                # Hard-delete decay-pruned (and any legacy fold) ids.
                for mid in deleted_ids:
                    cur.execute("DELETE FROM memories WHERE id = ?", (mid,))
                    deleted += cur.rowcount
                cur.execute("COMMIT;")
            except Exception:
                cur.execute("ROLLBACK;")
                raise
            finally:
                cur.close()
        log.debug("consolidate_memories committed deleted=%d", deleted)
        self._mark_vec_dirty()
        return deleted


    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #
    def get(self, memory_id: str) -> MemoryCard | None:
        """Fetch a single card by id, or None if absent."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id, fact, evidence, timestamp, tags, source_tokens, embedding, access_count, last_access
                FROM memories WHERE id = ?
                """,
                (memory_id,),
            )
            row = cur.fetchone()
            cur.close()
        return self._row_to_card(row) if row is not None else None

    def all(self) -> list[MemoryCard]:
        """Return every card, ordered by timestamp ascending."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id, fact, evidence, timestamp, tags, source_tokens, embedding, access_count, last_access
                FROM memories NOT INDEXED WHERE superseded_by IS NULL
                ORDER BY timestamp ASC, id ASC
                """
            )
            rows = cur.fetchall()
            cur.close()
        return [self._row_to_card(r) for r in rows]

    def sql_lookup(self, field: str, value: str) -> list[MemoryCard]:
        """Exact/substring lookup on an indexed field.

        - field == "fact" or "evidence": case-insensitive LIKE substring match
          (SQL `fact LIKE '%value%' COLLATE NOCASE`), with an exact-match fast
          path for `fact` first (see below).
        - field == "tags": `value` is treated as a SINGLE tag and a card
          matches iff that tag string is present in the card's JSON tags
          array. Implemented PYTHON-SIDE (load candidate rows from SQL, then
          json.loads + membership test) so it is obviously correct and has
          no dependency on SQLite JSON1 availability. For the prototype
          scale (hundreds–thousands of cards) the extra scan is negligible.

        Any other `field` raises ValueError.

        Exact-match fast path (`fact` only, when `value` has no SQL wildcards):
        we FIRST try an index-usable equality `superseded_by IS NULL AND
        fact = ? COLLATE NOCASE`. Because a fact that *equals* `value` is also
        matched by `LIKE '%value%'`, the exact-match result set is a strict
        SUBSET of the substring result set — callers cannot observe a
        behavioral difference, only a faster result for exact keys. If the
        equality returns nothing we fall through to the substring path.

        Substring path for `fact`: a leading-wildcard LIKE cannot use a b-tree
        for range constraints, so we scan `idx_fact` (narrow: `fact` text +
        rowid, NO 384-float embedding blobs) testing the LIKE against the
        indexed value, then fetch the (few) matched rows by id. This avoids
        slurping every row's embedding BLOB during the scan.
        """
        if value is None:
            return []
        field = field.lower()
        if field in _SQL_LOOKUP_COLUMNS:
            col = _SQL_LOOKUP_COLUMNS[field]
            with self._lock:
                # Fast path: exact keys (no '%'/'_' wildcards) hit the NOCASE
                # index directly. Restricted to `fact`, which is the only
                # column with a matching index (evidence has none, so an
                # exact probe there would only add a wasted scan on a miss).
                if col == "fact" and "%" not in value and "_" not in value:
                    hits = self._lookup_exact_locked(col, value)
                    if hits is not None:
                        return hits
                if col == "fact":
                    return self._lookup_substring_fact_locked(value)
                return self._lookup_substring_locked(col, value)

        if field == "tags":
            # Python-side membership over a single tag.
            target = value
            results: list[MemoryCard] = []
            with self._lock:
                results = self._lookup_tags_locked(target)
            return results

        raise ValueError(
            f"sql_lookup field must be one of 'fact','tags','evidence'; got {field!r}"
        )

    # -- sql_lookup helpers (callers MUST hold self._lock) ----------------- #
    def _stmt_cursor(self, sql: str) -> sqlite3.Cursor:
        """Return a prepared cursor for `sql`, reusing a cached one if present.

        The cache is keyed by the SQL string, so each cursor only ever
        executes its own statement (cursors are not shared across different
        SQL). Every caller holds `self._lock`, so the shared cursors are never
        used concurrently. Bounded: only a handful of distinct statements are
        cached.
        """
        cur = self._stmt_cache.get(sql)
        if cur is None:
            cur = self._conn.cursor()
            self._stmt_cache[sql] = cur
        return cur

    def _lookup_exact_locked(self, col: str, value: str) -> list[MemoryCard] | None:
        """Index-usable exact match. Returns the cards, or None when no row
        matches so the caller can fall through to the substring path."""
        sql = _SQL_EXACT.format(select=_SQL_LOOKUP_SELECT, col=col)
        cur = self._stmt_cursor(sql)
        cur.execute(sql, (value,))
        rows = cur.fetchall()
        if not rows:
            return None
        return [self._row_to_card(r) for r in rows]

    def _lookup_substring_fact_locked(self, value: str) -> list[MemoryCard]:
        """Substring LIKE for `fact`: covering idx_fact scan + fetch by id."""
        pattern = f"%{value}%"
        sql = _SQL_SUBSTR_IDS.format(col="fact")
        cur = self._stmt_cursor(sql)
        cur.execute(sql, (pattern,))
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return []
        if len(ids) > _SUBSET_MATCH_LIMIT:
            # Pathologically broad match: fall back to the single-pass scan
            # rather than a huge IN(...) list.
            return self._lookup_substring_locked("fact", value)
        rows = self._fetch_by_ids_locked(ids)
        return [self._row_to_card(r) for r in rows]

    def _lookup_substring_locked(self, col: str, value: str) -> list[MemoryCard]:
        """Direct LIKE scan (fields without a covering index, e.g. evidence).
        `value` is the raw lookup term; the surrounding wildcards are added
        here so every caller passes the same raw value."""
        pattern = f"%{value}%"
        sql = _SQL_SUBSTR_SCAN.format(select=_SQL_LOOKUP_SELECT, col=col)
        cur = self._stmt_cursor(sql)
        cur.execute(sql, (pattern,))
        rows = cur.fetchall()
        return [self._row_to_card(r) for r in rows]

    def _fetch_by_ids_locked(self, ids: list[str]) -> list[tuple[Any, ...]]:
        """Fetch full rows for a small id list, re-applying the superseded-by
        audit filter and the canonical (timestamp, id) ordering. SQL varies
        with the placeholder count so it is NOT cached in the statement cache."""
        placeholders = ",".join("?" for _ in ids)
        sql = _SQL_IN_FETCH.format(select=_SQL_LOOKUP_SELECT, placeholders=placeholders)
        cur = self._conn.cursor()
        try:
            cur.execute(sql, ids)
            return cur.fetchall()
        finally:
            cur.close()

    def _lookup_tags_locked(self, target: str) -> list[MemoryCard]:
        """Python-side JSON-tags membership over the non-superseded rows."""
        results: list[MemoryCard] = []
        sql = _SQL_TAGS_SCAN.format(select=_SQL_LOOKUP_SELECT)
        cur = self._stmt_cursor(sql)
        cur.execute(sql)
        rows = cur.fetchall()
        for r in rows:
            card = self._row_to_card(r)
            if target in card.tags:
                results.append(card)
        return results

    # ------------------------------------------------------------------ #
    # Vector-search cache + numpy fast path
    # ------------------------------------------------------------------ #
    def _mark_vec_dirty(self) -> None:
        """Invalidate the cached vector-search matrix after any write that
        inserts/deletes a row or changes an embedding/timestamp.

        Called by add/update/delete/consolidate_memories (and by the eval
        harness's bulk seeder, which writes rows directly to the connection).
        `touch()` deliberately does NOT call this: it only bumps
        access_count/last_access, neither of which the matrix holds, so
        touch-driven reads must NOT force a rebuild. `_vec_hetero` is also
        cleared here so a later uniform embed store can re-attempt caching.
        """
        self._vec_dirty = True
        self._vec_hetero = False

    def _ensure_vec_cache(self, np) -> Any:
        """Return the cached (n, dim) float32 matrix, refreshing lazily.

        Builds on first use and whenever `_vec_dirty` is set: SELECTs only
        (id, timestamp, embedding) for rows with a non-NULL embedding,
        decodes each BLOB straight into a numpy float32 row (zero-Python
        unpacking), and stacks them once. Returns None when there are no
        embeddable rows or when rows have heterogeneous lengths (not
        representable as a dense matrix -> caller falls back to Python).
        """
        if self._vec_hetero:
            return None
        if (not self._vec_dirty) and self._vec_matrix is not None:
            return self._vec_matrix

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                # Match `all()` exactly: superseded (audit-trail folded) rows
                # must not surface in vector search either. NOT INDEXED keeps
                # this a sequential table scan (we read every embedding BLOB
                # anyway, so an index scan + rowid lookups would only add
                # random I/O).
                "SELECT id, timestamp, embedding FROM memories NOT INDEXED "
                "WHERE embedding IS NOT NULL AND superseded_by IS NULL "
                "ORDER BY timestamp ASC, id ASC"
            )
            rows = cur.fetchall()
            cur.close()
            # Clear the dirty flag while still holding the lock, right at fetch
            # time: any write that lands AFTER this point will set `_vec_dirty`
            # back to True after our clear, forcing a rebuild on the next call.
            # (Clearing only after the (lock-free) stack below would let a
            # mid-build write be silently clobbered by our stale matrix.)
            self._vec_dirty = False

        ids: list[str] = []
        ts: list[float] = []
        arrs = []
        dim: int | None = None
        for rid, rts, blob in rows:
            a = np.frombuffer(blob, dtype=np.float32)
            if dim is None:
                dim = a.shape[0]
            elif a.shape[0] != dim:
                # Heterogeneous embedding lengths are not matrix-able; latch and
                # let the search fall back to the pure-Python loop.
                self._vec_hetero = True
                self._vec_matrix = None
                return None
            ids.append(rid)
            ts.append(float(rts) if rts is not None else 0.0)
            arrs.append(a)

        self._vec_ids = ids
        self._vec_ts = ts
        self._vec_matrix = np.stack(arrs) if arrs else None
        return self._vec_matrix

    def vector_search(
        self, query_vec: list[float], k: int = 5
    ) -> list[tuple[MemoryCard, float]]:
        """Top-k cosine similarity search via dot product.

        Vectors are assumed L2-normalized by the embedder, so cosine
        similarity reduces to the dot product. When numpy is importable this
        runs as ONE vectorized batch matmul over a cached (n_cards, dim)
        float32 matrix (rows L2-normalized, so dot == cosine) followed by an
        argpartition top-k; otherwise it falls back to the pure-Python loop in
        `_vector_search_fallback`. Cards with NULL embedding are skipped. If
        `query_vec` is empty or all zeros, returns [] (degenerate query).

        Scores are clamped to [0, 1]: a normalized dot product lives in
        [-1, 1], but the store contract promises scores in [0, 1], so negative
        similarities are floored at 0.0 (and the rare >1 from float noise is
        capped at 1.0). Ordering is (score desc, timestamp asc); identical
        (score, timestamp) tiebreak on id asc for determinism.
        """
        if not query_vec or all(v == 0.0 for v in query_vec):
            return []

        try:
            import numpy as np
        except ImportError:
            np = None

        if np is None:
            return self._vector_search_fallback(query_vec, k)

        matrix = self._ensure_vec_cache(np)
        if matrix is None or matrix.shape[0] == 0:
            # No searchable rows, OR heterogeneous embedding lengths (not
            # matrix-able). Delegate to the pure-Python loop, which is correct
            # for both cases (empty -> [], hetero -> per-card min-dim dot).
            return self._vector_search_fallback(query_vec, k)

        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] == 0:
            return []

        com = matrix.shape[1]
        if q.shape[0] == com:
            scores = matrix @ q
        else:
            # Preserve the reference loop's min(qdim, emb_dim) semantics over
            # the first `n` components (matrix[:, :n] is a view, not a copy).
            n = min(q.shape[0], com)
            if n == 0:
                return []
            scores = matrix[:, :n] @ q[:n]
        np.clip(scores, 0.0, 1.0, out=scores)

        total = matrix.shape[0]
        kk = min(k, total) if k > 0 else 0
        if kk <= 0:
            return []

        # Top-k by score; then expand to EVERY row tied at the k-th boundary so
        # timestamp tie-breaking matches the reference exactly, and sort that
        # small candidate set by (score desc, timestamp asc, id asc).
        cand = np.argpartition(scores, -kk)[-kk:]
        thr = float(scores[cand].min())
        cand = np.flatnonzero(scores >= thr)
        entries = [
            (float(scores[i]), self._vec_ts[i], self._vec_ids[i])
            for i in cand.tolist()
        ]
        entries.sort(key=lambda e: (-e[0], e[1], e[2]))
        top = entries[:kk]

        want_ids = [e[2] for e in top]
        by_id = {c.id: c for c in self.batch_get(want_ids)}
        return [(by_id[e[2]], e[0]) for e in top if e[2] in by_id]

    def _vector_search_fallback(
        self, query_vec: list[float], k: int
    ) -> list[tuple[MemoryCard, float]]:
        """Pure-Python fallback used only when numpy is unavailable.

        Mirrors the original O(n x d) loop exactly: per-card dot product over
        `min(qdim, len(emb))`, clamp to [0, 1], sort by (score desc, timestamp
        asc), top-k. If the caller's matrix cache latched `_vec_hetero`, this
        path is also the correctness backstop for non-uniform embeddings.
        """
        cards = self.all()
        scored: list[tuple[MemoryCard, float]] = []
        qdim = len(query_vec)
        for card in cards:
            emb = card.embedding
            if emb is None or not emb:
                continue
            n = min(qdim, len(emb))
            dot = 0.0
            for i in range(n):
                dot += query_vec[i] * emb[i]
            if dot < 0.0:
                dot = 0.0
            elif dot > 1.0:
                dot = 1.0
            scored.append((card, dot))
        scored.sort(key=lambda item: (-item[1], item[0].timestamp))
        return scored[:k]

    # ------------------------------------------------------------------ #
    # Two-pass hybrid retrieval (FTS5 lexical pre-filter + vector re-rank)
    # ------------------------------------------------------------------ #
    def hybrid_vector_search(
        self,
        query_text: str,
        query_vec: list[float],
        top_k: int = 5,
        candidate_limit: int = 50,
        alpha: float = 0.3,
    ) -> list[tuple[MemoryCard, float]]:
        """Two-pass hybrid retrieval: FTS5 BM25 lexical pre-filter + vector re-rank.

        Pass 1 (lexical filter): SQLite FTS5 BM25 query over `cards_fts` for the
        query text, returning up to `candidate_limit` card_ids ranked by BM25
        (ORDER BY rank). Only non-superseded cards are candidates (superseded
        cards are never inserted into `cards_fts`).

        Fallback: if FTS5 returns fewer than `top_k` candidates (strict keyword
        mismatch -- e.g. the query is purely semantic and shares no tokens with
        any card), fall back to the full-matrix vector search
        (`self.vector_search(query_vec, k=top_k)`) so recall never silently drops
        to zero. This is the explicit spec'd fallback.

        Pass 2 (vector re-rank): fetch embeddings only for the candidate set,
        compute cosine (dot, vectors are L2-normalized) vs `query_vec` with
        numpy, and combine: ``Score = alpha * Norm(BM25) + (1 - alpha) * Cosine``.
        - Norm(BM25): FTS5's ``bm25()`` returns a NEGATIVE score (more negative =
          better). Convert to a [0,1]-ish positive rank score: take the candidate
          set's bm25 values, and normalize via min-max into [0,1] where the best
          (most-negative, i.e. highest |bm25|) maps to 1.0. Guard divide-by-zero
          (all equal bm25 -> all 1.0; chosen over 0.5 so a uniform-BM25 candidate
          set still gets full lexical weight rather than being zeroed out).
        - Cosine: raw dot, clamped to [0,1] (same convention as `vector_search`).
        Return the top `top_k` cards ordered by final combined score descending,
        with the SAME (score desc, timestamp asc, id asc) tie-break as
        `vector_search`. Return ``list[(MemoryCard, float)]`` where the float is
        the COMBINED score (so callers see the hybrid score, not just cosine).
        """
        # Degenerate query -> [], mirroring vector_search. (Empty query_vec is
        # also the cheap way to bail before touching numpy/FTS.)
        if not query_vec or all(v == 0.0 for v in query_vec):
            return []

        try:
            import numpy as np
        except ImportError:
            np = None

        # --- Pass 1: FTS5 BM25 lexical pre-filter --------------------------- #
        # Sanitize the query into a broad-recall OR-joined MATCH string. FTS5
        # treats several punctuation/codepoints as query operators (", *, OR,
        # AND, NOT, parens); raw user text would either error or be parsed as
        # boolean logic. We keep only [A-Za-z0-9]+ tokens and OR them so ANY
        # token matches (we want a wide candidate set for the vector re-rank,
        # not a strict AND). Lowercasing matches the unicode61 tokenizer's
        # case-folding so token equality holds.
        import re

        tokens = re.findall(r"[A-Za-z0-9]+", query_text or "")
        # De-duplicate while preserving order so the MATCH string stays compact
        # (a repeated token just re-asserts the same condition).
        seen: set[str] = set()
        uniq: list[str] = []
        for t in tokens:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                uniq.append(tl)
        if not uniq:
            # Pure punctuation/whitespace query -> no lexical signal at all;
            # go straight to the full vector scan.
            return self.vector_search(query_vec, k=top_k)
        match_str = " OR ".join(uniq)

        cand_rows: list[tuple[str, float]] = []  # (card_id, bm25)
        try:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    "SELECT card_id, bm25(cards_fts) FROM cards_fts "
                    "WHERE cards_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match_str, candidate_limit),
                )
                cand_rows = cur.fetchall()
                cur.close()
        except sqlite3.OperationalError:
            # Malformed MATCH (e.g. a token that still trips FTS5's parser after
            # our sanitize, or a build without FTS5) -> fall back to the full
            # vector scan so recall never silently drops to zero.
            return self.vector_search(query_vec, k=top_k)

        # Strict keyword mismatch: too few candidates to re-rank meaningfully.
        # The spec'd fallback is the full-matrix vector search.
        if len(cand_rows) < top_k:
            return self.vector_search(query_vec, k=top_k)

        cand_ids: list[str] = [r[0] for r in cand_rows]
        bm25_raw: dict[str, float] = {r[0]: float(r[1]) for r in cand_rows}

        # --- Pass 2: vector re-rank over the candidate set ------------------ #
        # Fetch candidate embeddings + timestamps in one round-trip. We pull
        # `embedding` and `timestamp` together (not via batch_get, which would
        # cost a second pass to re-read blobs) and re-apply the superseded-by
        # audit filter so a card superseded between pass 1 and now is dropped.
        placeholders = ",".join("?" for _ in cand_ids)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"SELECT id, timestamp, embedding FROM memories "
                f"WHERE id IN ({placeholders}) "
                f"AND superseded_by IS NULL AND embedding IS NOT NULL",
                tuple(cand_ids),
            )
            emb_rows = cur.fetchall()
            cur.close()

        if not emb_rows:
            # All candidates had NULL embeddings or were superseded between
            # passes; defer to the full vector scan.
            return self.vector_search(query_vec, k=top_k)

        # Preserve pass-1 (BM25) order as the canonical candidate order so the
        # re-rank is deterministic regardless of the IN(...) fetch order.
        emb_by_id: dict[str, tuple[float, bytes]] = {
            r[0]: (float(r[1]) if r[1] is not None else 0.0, r[2]) for r in emb_rows
        }
        # Filter cand_ids to those we actually fetched an embedding for, in
        # pass-1 order.
        scored_ids: list[str] = [cid for cid in cand_ids if cid in emb_by_id]
        if len(scored_ids) < top_k:
            return self.vector_search(query_vec, k=top_k)

        q = np.asarray(query_vec, dtype=np.float32) if np is not None else None

        # Heterogeneous embedding lengths in the candidate set are rare but must
        # not crash. When numpy is available we try the matrix path; if dims
        # disagree we fall back to a per-card min-dim dot (mirroring
        # `_vector_search_fallback`'s len handling). When numpy is unavailable
        # we go straight to the per-card pure-Python dot.
        cosine_by_id: dict[str, float] = {}
        if np is not None:
            arrs: list[Any] = []
            dim: int | None = None
            hetero = False
            for cid in scored_ids:
                blob = emb_by_id[cid][1]
                a = np.frombuffer(blob, dtype=np.float32)
                if dim is None:
                    dim = a.shape[0]
                elif a.shape[0] != dim:
                    hetero = True
                    break
                arrs.append(a)
            if not hetero and arrs:
                cand_matrix = np.stack(arrs)  # (k_cand, dim)
                com = cand_matrix.shape[1]
                if q.shape[0] == com:
                    scores = cand_matrix @ q
                else:
                    n = min(q.shape[0], com)
                    if n == 0:
                        return []
                    scores = cand_matrix[:, :n] @ q[:n]
                np.clip(scores, 0.0, 1.0, out=scores)
                for cid, s in zip(scored_ids, scores.tolist()):
                    cosine_by_id[cid] = float(s)
            else:
                # Heterogeneous dims (or numpy missing below): per-card dot.
                qdim = len(query_vec)
                for cid in scored_ids:
                    emb = array("f")
                    emb.frombytes(emb_by_id[cid][1])
                    e = emb.tolist()
                    n = min(qdim, len(e))
                    dot = 0.0
                    for i in range(n):
                        dot += query_vec[i] * e[i]
                    if dot < 0.0:
                        dot = 0.0
                    elif dot > 1.0:
                        dot = 1.0
                    cosine_by_id[cid] = dot
        else:
            qdim = len(query_vec)
            for cid in scored_ids:
                emb = array("f")
                emb.frombytes(emb_by_id[cid][1])
                e = emb.tolist()
                n = min(qdim, len(e))
                dot = 0.0
                for i in range(n):
                    dot += query_vec[i] * e[i]
                if dot < 0.0:
                    dot = 0.0
                elif dot > 1.0:
                    dot = 1.0
                cosine_by_id[cid] = dot

        # --- Normalize BM25 into [0,1] where best (most negative) -> 1.0 ---- #
        # bm25() returns <=0 with LOWER (more negative) = better match, so
        # best = min(raw), worst = max(raw). norm_i = (worst - raw_i) / (worst -
        # best) maps best -> 1.0, worst -> 0.0. Guard the all-equal case (worst
        # == best) by setting every norm to 1.0: a uniform-BM25 candidate set
        # keeps full lexical weight rather than being zeroed to 0.5-mid.
        raws = [bm25_raw[cid] for cid in scored_ids]
        best = min(raws)   # most negative = best
        worst = max(raws)  # closest to 0 = worst
        if worst == best:
            bm25_norm = {cid: 1.0 for cid in scored_ids}
        else:
            span = worst - best  # >= 0 (worst >= best since best is min)
            bm25_norm = {cid: (worst - bm25_raw[cid]) / span for cid in scored_ids}

        # --- Combine + final (score desc, timestamp asc, id asc) ordering --- #
        entries: list[tuple[float, float, str, str]] = []  # (combined, ts, id, id)
        for cid in scored_ids:
            cos = cosine_by_id.get(cid, 0.0)
            bm = bm25_norm.get(cid, 0.0)
            combined = alpha * bm + (1.0 - alpha) * cos
            ts, _ = emb_by_id[cid]
            entries.append((combined, ts, cid, cid))
        entries.sort(key=lambda e: (-e[0], e[1], e[2]))

        kk = min(top_k, len(entries)) if top_k > 0 else 0
        if kk <= 0:
            return []
        top = entries[:kk]

        want_ids = [e[3] for e in top]
        by_id = {c.id: c for c in self.batch_get(want_ids)}
        # batch_get re-applies the superseded-by filter and canonical ordering,
        # so a card superseded between pass 2 and this fetch simply drops.
        return [(by_id[e[3]], e[0]) for e in top if e[3] in by_id]

    # ------------------------------------------------------------------ #
    # Metrics / introspection
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        """Number of stored cards."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL")
            (n,) = cur.fetchone()
            cur.close()
        return int(n)

    def db_size_bytes(self) -> int:
        """On-disk size of the DB file.

        For ":memory:" there is no file, so returns 0. For a file-backed DB
        returns `os.path.getsize(db_path)`. If the path does not exist yet
        (no writes happened), returns 0.
        """
        if self._is_memory:
            return 0
        try:
            return os.path.getsize(self.db_path)
        except OSError:
            return 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Close the held connection. Safe to call once; no-op thereafter."""
        with self._lock:
            try:
                # Close prepared-statement cursors first (they hold references
                # into the connection we are about to close).
                for cur in self._stmt_cache.values():
                    try:
                        cur.close()
                    except sqlite3.Error:
                        pass
                self._stmt_cache.clear()
                self._conn.close()
            except sqlite3.ProgrammingError:
                pass

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # Tiny normalized embeddings (L2 norm 1) for a smoke test.
    def _norm(v: list[float]) -> list[float]:
        import math

        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v

    store = MemoryStore(db_path=":memory:")

    c1 = MemoryCard(
        id="card-1",
        fact="The user prefers dark mode.",
        evidence="user said: 'I love dark mode'",
        timestamp=now_ts(),
        tags=["preference", "ui"],
        embedding=_norm([1.0, 0.0, 0.0, 0.0]),
        source_tokens=6,
    )
    c2 = MemoryCard(
        id="card-2",
        fact="The project is written in Rust.",
        evidence="README says 'this is a Rust project'",
        timestamp=now_ts(),
        tags=["tech", "stack"],
        embedding=None,  # NULL embedding — must be skipped by vector_search
        source_tokens=10,
    )

    store.add(c1)
    store.add(c2)
    print("count after adds:", store.count())

    got = store.get("card-1")
    print("get(card-1) fact:", got.fact if got else None)
    print("get(card-1) embedding decoded:", got.embedding if got else None)

    missing = store.get("nope")
    print("get(nope):", missing)

    print("sql_lookup fact 'rust':", [c.fact for c in store.sql_lookup("fact", "rust")])
    print(
        "sql_lookup evidence 'dark':",
        [c.fact for c in store.sql_lookup("evidence", "dark")],
    )
    print(
        "sql_lookup tags 'ui':",
        [c.fact for c in store.sql_lookup("tags", "ui")],
    )

    # Vector search: query aligned with card-1's embedding.
    hits = store.vector_search(_norm([1.0, 0.0, 0.0, 0.0]), k=5)
    print("vector_search hits:", [(c.id, round(s, 3)) for c, s in hits])

    # Degenerate query → [].
    print("vector_search zeros:", store.vector_search([0.0, 0.0, 0.0, 0.0]))
    print("vector_search empty:", store.vector_search([]))

    # Update (upsert).
    c1_updated = MemoryCard(
        id="card-1",
        fact="The user prefers LIGHT mode now.",
        evidence="user said: 'switch me to light mode'",
        timestamp=now_ts(),
        tags=["preference", "ui", "change"],
        embedding=_norm([0.0, 1.0, 0.0, 0.0]),
        source_tokens=7,
    )
    store.update(c1_updated)
    got2 = store.get("card-1")
    print("after update fact:", got2.fact if got2 else None)
    print("after update tags:", got2.tags if got2 else None)

    # Delete.
    deleted = store.delete("card-2")
    print("delete(card-2):", deleted, "| count:", store.count())
    deleted_again = store.delete("card-2")
    print("delete(card-2) again:", deleted_again)

    print("db_size_bytes (in-memory):", store.db_size_bytes())

    store.close()
    print("smoke test OK")
