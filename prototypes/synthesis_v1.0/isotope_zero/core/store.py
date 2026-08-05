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
import math
import os
import re
import sqlite3
import threading
from array import array
from inspect import iscoroutine
from typing import Any

from isotope_zero.types import MemoryCard, now_ts
from isotope_zero.core import native
from isotope_zero.core import graph
from isotope_zero.core import history
from isotope_zero.core.decay import calculate_retention, update_stability, hybrid_score
try:
    from isotope_zero.core.dedup import content_aware_fingerprint
except ImportError:  # sibling module absent — dedup falls back to None
    content_aware_fingerprint = None  # type: ignore[assignment]

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
    "embedding, access_count, last_access, superseded_by, stability, importance, archived, scope, "
    "content_fingerprint, ttl_seconds, expiration_timestamp"
)

# Exact-match fast path (see MemoryStore.sql_lookup). Uses `fact = ?` with
# COLLATE NOCASE so the `idx_memories_fact_nocase` index is usable.
# Expiry filter: exclude cards whose ``expiration_timestamp`` is set AND has
# passed (``> unixepoch()``). NULL expiration => never expires (pre-TTL cards).
_SQL_EXACT = (
    "SELECT {select} FROM memories "
    "WHERE superseded_by IS NULL AND archived = 0 "
    "AND (expiration_timestamp IS NULL OR expiration_timestamp > unixepoch()) "
    "AND {col} = ? COLLATE NOCASE "
    "{scope}ORDER BY timestamp ASC, id ASC"
)

# Substring LIKE fast path for `fact`. A leading-wildcard LIKE cannot use a
# b-tree for range constraints, but scanning `idx_fact` (which holds only the
# `fact` text + rowid, i.e. NO 384-float embedding blobs) and testing the LIKE
# against the indexed value is far cheaper than a full table scan that must
# slurp each row's embedding blob. Matching rowids are then fetched by id.
_SQL_SUBSTR_IDS = (
    "SELECT id FROM memories INDEXED BY idx_fact "
    "WHERE {col} LIKE ? COLLATE NOCASE AND superseded_by IS NULL AND archived = 0 "
    "AND (expiration_timestamp IS NULL OR expiration_timestamp > unixepoch()) "
    "{scope}"
)

# Direct LIKE scan used for fields without a covering index (`evidence`) and
# as a fallback when the exact+two-phase approach yields too many matches.
# `NOT INDEXED` forces a plain sequential table scan: with the new indexes
# present SQLite's planner prefers an index-scan + rowid-lookup plan that is
# *slower* for an unconstrained LIKE, so we pin the cheaper table scan.
_SQL_SUBSTR_SCAN = (
    "SELECT {select} FROM memories NOT INDEXED "
    "WHERE {col} LIKE ? COLLATE NOCASE AND superseded_by IS NULL AND archived = 0 "
    "AND (expiration_timestamp IS NULL OR expiration_timestamp > unixepoch()) "
    "{scope}ORDER BY timestamp ASC, id ASC"
)

# Fetch full rows by id, re-applying the superseded-by audit filter and the
# canonical ordering. Reuses `batch_get`'s ordering (timestamp, id).
_SQL_IN_FETCH = (
    "SELECT {select} FROM memories "
    "WHERE id IN ({placeholders}) AND superseded_by IS NULL AND archived = 0 "
    "AND (expiration_timestamp IS NULL OR expiration_timestamp > unixepoch()) "
    "{scope}ORDER BY timestamp ASC, id ASC"
)

# Multi-tier scope filter clause appended to the sql_lookup SELECTs when the
# caller passes a non-None ``scope`` to ``sql_lookup``. A card is visible to a
# scoped query iff its stored scope matches the bound value, OR it carries
# the global ``'default'`` sentinel, OR it is NULL (the pre-migration row
# state). This is the same backward-compatible visibility rule
# ``isotope_zero.core.scoping.match_scope`` encodes (a global card is visible
# to every scoped query); the SQL form is the on-disk enforcement for the
# SQL lookup path. ``scope=None`` => empty string (no filter), preserving the
# original global behavior byte-for-byte.
_SCOPE_CLAUSE = "AND (scope = ? OR scope = 'default' OR scope IS NULL) "


def _scope_clause(scope: str | None) -> str:
    """Return the scope filter SQL fragment for ``sql_lookup`` templates.

    ``scope=None`` (the ``sql_lookup`` default) yields an empty string so the
    query is byte-identical to the pre-scoping path and the statement cache
    key is unchanged — pre-scoping callers see no behavioral change. A non-
    None scope yields ``_SCOPE_CLAUSE``; the caller binds the scope value as
    the trailing parameter. The clause tolerates a NULL ``scope`` column (pre-
    migration rows) by also matching NULL, so a not-yet-migrated DB stays
    searchable.
    """
    return _SCOPE_CLAUSE if scope is not None else ""

# Python-side tag membership scan (JSON array membership is not indexable).
_SQL_TAGS_SCAN = (
    "SELECT {select} FROM memories NOT INDEXED "
    "WHERE superseded_by IS NULL AND archived = 0 "
    "AND (expiration_timestamp IS NULL OR expiration_timestamp > unixepoch()) "
    "{scope}ORDER BY timestamp ASC, id ASC"
)

# If an id-set from the covering idx_fact scan exceeds this many rows we fall
# back to the single-pass LIKE scan rather than building a huge IN(...) list.
_SUBSET_MATCH_LIMIT = 900

# Over-fetch buffer for hybrid_search hydration. Fusion is asked for
# k + _HYDRATION_BUFFER candidates so a card hard-deleted between the
# (lock-released) fusion step and the batch_get hydration doesn't shrink the
# returned top-k below k — the next-best candidate hydrates in. Cheap (one
# extra row in the IN(...) SELECT) and bounds the TOCTOU window.
_HYDRATION_BUFFER = 4

# --------------------------------------------------------------------------- #
# FTS5 external-content sync triggers (Late Fusion hybrid search).
# `memories_fts` is `content='memories'`: it owns NO row data, only the
# tokenized `fact`. SQLite does NOT auto-mirror parent writes into an
# external-content FTS table, so these three triggers do it for every write
# path — including the eval harness's direct-to-connection bulk seeder that
# bypasses `add()/update()/delete()`. The `'delete'` command is the FTS5
# protocol for evicting a rowid from a content table without leaving a
# dangling back-pointer; a plain `DELETE FROM memories_fts` would corrupt
# the index. AFTER triggers fire once the base row is committed, so NEW/OLD
# rowids are always valid.
# --------------------------------------------------------------------------- #
_FTS_AIU_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS memories_fts_aiu
AFTER INSERT ON memories
BEGIN
    INSERT INTO memories_fts(rowid, fact, id) VALUES (new.rowid, new.fact, new.id);
END;
"""
_FTS_AAD_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS memories_fts_aad
AFTER DELETE ON memories
BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, fact, id)
    VALUES ('delete', old.rowid, old.fact, old.id);
END;
"""
_FTS_AU_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS memories_fts_au
AFTER UPDATE OF fact, id ON memories
BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, fact, id)
    VALUES ('delete', old.rowid, old.fact, old.id);
    INSERT INTO memories_fts(rowid, fact, id) VALUES (new.rowid, new.fact, new.id);
END;
"""


# --------------------------------------------------------------------------- #
# Late Fusion hybrid search: module-level helpers (pure functions; testable in
# isolation without a store). `_rrf_fusion` is the spec-mandated fusion; the
# entity/tokenize helpers are zero-dependency (no nltk/sklearn) so the store
# stays importable in its minimal config.
# --------------------------------------------------------------------------- #
def _rrf_fusion(
    semantic_hits: list[tuple[str, float]],
    bm25_hits: list[tuple[str, float]],
    entity_boosts: dict[str, float],
    alpha: float = 0.70,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of two ranked branches + an entity-graph boost.

    ``semantic_hits`` and ``bm25_hits`` are ``[(card_id, score)]`` lists, each
    ordered by the branch's NATIVE ranking (best first). Only the RANK
    position matters for fusion — the raw scores are discarded (they are not
    score-comparable across branches). ``entity_boosts`` maps ``card_id ->
    additive boost`` (0 for cards with none).

    Score(d) = α/(60 + r_vec(d)) + (1-α)/(60 + r_bm25(d)) + Boost(d)

    where ``r_*`` are 1-indexed ranks WITHIN each branch. A card absent from
    a branch contributes 0 from that branch's term (not a penalty). The ``60``
    constant is the RRF smoothing denominator from Cormack et al. (SIGIR 2009);
    it damps top-rank dominance so a noisy branch can't swamp a precise one.

    Returns the top-``k`` ``(card_id, fused_score)`` sorted by fused score
    desc. Ties are broken deterministically by id asc (callers re-apply the
    store's full timestamp/id tie-break after hydration).
    """
    # Rank by position within each branch (1-indexed). A dict accumulates the
    # two rank terms per card so a card appearing in BOTH branches gets both
    # contributions — the whole point of late fusion (a card strong in both
    # modalities should outrank one strong in only one).
    scores: dict[str, float] = {}
    for rank0, (cid, _score) in enumerate(semantic_hits):
        scores[cid] = scores.get(cid, 0.0) + alpha / (60.0 + (rank0 + 1))
    for rank0, (cid, _score) in enumerate(bm25_hits):
        scores[cid] = scores.get(cid, 0.0) + (1.0 - alpha) / (60.0 + (rank0 + 1))
    # Additive entity-graph boost on top of the fused RRF rank score. ONLY
    # cards already present in `scores` (i.e. surfaced by at least one branch)
    # are eligible — the branches carry the live-row filter (archived/
    # superseded/scope) and are the source of truth for "this card is
    # searchable". Applying a boost to a card NOT in `scores` would inject it
    # into the ranking at its raw boost value (~0.5), which dominates RRF
    # scores (~0.017) and lets an archived/out-of-scope graph neighbor hijack
    # the top of the results. So boosts are a RE-RANKING signal, never a
    # candidate-INTRODUCTION signal.
    for cid, boost in entity_boosts.items():
        if boost and cid in scores:
            scores[cid] = scores[cid] + float(boost)
    ranked = sorted(scores.items(), key=lambda e: (-e[1], e[0]))
    return ranked[:k] if k > 0 else ranked


# Minimal English stopword set — enough to keep query entities from being
# dominated by "the"/"a"/"is" without pulling a full NLTK list (zero-dep).
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that
    the to was were will with this these those i you he she they we us our
    your his her their my me him them not no but if then than so such too very
    can do does did done doing about above after again all also am any each
    few more most other some only own same s t don now into out up down off
    over under again further once here there when where why how what which who
    whom whose
    """.split()
)


def _extract_entities(query: str) -> list[str]:
    """Zero-dependency entity extraction: tokenize, lowercase, stopword-strip.

    Returns content-word tokens (len >= 2, alphabetic) in first-seen order.
    This is deliberately simple — it is a *recall* signal for the entity-graph
    boost, not a named-entity recognizer. Cards mentioning these tokens drive
    the `card_edges` neighborhood; a fancier NER would change which cards get
    boosted, not the fusion math itself.
    """
    if not query:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{1,}", query.lower())
    seen: list[str] = []
    seen_set: set[str] = set()
    for tok in tokens:
        if tok in _STOPWORDS or tok in seen_set:
            continue
        seen_set.add(tok)
        seen.append(tok)
    return seen


def _fts5_escape(token: str) -> str:
    """Escape a bare token for an FTS5 phrase. Doubles embedded quotes so a
    token like o'brien becomes o''brien inside a double-quoted phrase."""
    return token.replace('"', '""')


def _fts5_query(query: str) -> str:
    """Build an FTS5 MATCH expression from a free-text query.

    Each content-word entity becomes a double-quoted prefix term
    (``"tok"*``) joined by OR, so the BM25 branch recalls any card whose
    fact mentions any query entity. Prefix matching (``*``) lets a search
    for ``embed`` match ``embedding``. Returns "" (no MATCH) when there
    are no entities, so `hybrid_search` falls back to vector-only.
    """
    entities = _extract_entities(query)
    if not entities:
        return ""
    return " OR ".join('"%s"*' % _fts5_escape(e) for e in entities)


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

    def __init__(
        self,
        db_path: str = ":memory:",
        embedder: Any = None,
        use_daemon: bool = False,
        use_mmap: bool = False,
        enable_dedup: bool = False,
    ) -> None:
        """Create/open the store.

        Args:
            db_path: SQLite path. ":memory:" keeps data in RAM for the life
                of this store (one held connection). Any other string is
                treated as a file path (opened/created on disk).
            embedder: Optional embeddings engine with `.embed_text(text) ->
                list[float]`. Stored on the instance for callers/router use;
                this class itself never consumes it (cards arrive with their
                embedding already populated, or None).
            use_daemon: When True AND no explicit `embedder` is given, wrap the
                embedder in `isotope_zero.daemon.client.DaemonClient` so
                onnxruntime runs in a separate daemon process instead of this
                (client) process. Behavior-neutral at the default False; the
                lazy import keeps onnxruntime out of this module regardless.
            use_mmap: When True, `_ensure_vec_cache` builds the vector matrix
                via a file-backed `np.memmap` (`MmapVectorStore`) instead of a
                heap `np.stack`. Default **False**: the Phase 5 empirical
                finding is that the matrix is only ~15 MB at 10k cards (ONNX
                dominates RSS, not the matrix), the heap BLAS path is ~7 %
                FASTER than memmap, and the memmap view is not safe to rebuild
                while a concurrent reader holds it for a `matrix @ q` matmul
                (the single-connection store serializes writes, but a vector
                search runs the matmul outside `_ensure_vec_cache`'s lock, so
                an `add()`-driven `invalidate()` mid-matmul can corrupt the
                view -> SIGILL under heavy threaded load). mmap stays available
                as an opt-in for cold-start / large-scale probes; the default
                production path is the verified-fast heap BLAS.
        """
        self.db_path = db_path
        self.use_mmap = bool(use_mmap)
        # mem0 port (dedup): content-aware duplicate detection is OPT-IN. The
        # default False preserves the pre-port ``add()`` semantics — two cards
        # with identical fact+tags in the same scope both insert — which the
        # scope-isolation tests rely on. Set True to get mem0's
        # ``seen_hashes`` skip (same-scope fact+tags twin is folded into the
        # existing row, not a second row).
        self.enable_dedup = bool(enable_dedup)
        if use_daemon and embedder is None:
            from isotope_zero.daemon.client import DaemonClient

            embedder = DaemonClient()
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
        # Multi-tier scoping: parallel scope strings for the matrix rows (row i
        # <-> _vec_ids[i] <-> _vec_scopes[i]). Populated by `_ensure_vec_cache`
        # alongside _vec_ids/_vec_ts so `vector_search` can apply a boolean
        # mask to the scores in C (O(n), sub-0.1ms @10k) without a second SQL
        # round-trip per query. A scope->np.ndarray row-index cache is built
        # lazily on first use of a scope and invalidated on any write.
        self._vec_scopes: list[str] | None = None
        self._scope_index_cache: dict[str, Any] = {}
        # Phase 5: file-backed mmap backend, lazily built by `_ensure_vec_cache`
        # when `use_mmap=True`. `None` means not yet constructed; the heap
        # `np.stack` path is the fallback when `use_mmap=False` OR the mmap
        # build raises (hetero dim, build error). See `core.mmap_store`.
        self._mmap_store: Any = None

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
                  superseded_by TEXT,              -- Phase 3: audit trail; non-NULL => folded into that survivor's id
                  stability REAL DEFAULT 1.0,      -- Phase 7C: Ebbinghaus stability S
                  importance REAL DEFAULT 0.0,     -- Phase 7C: user-set importance [0.0, 1.0]
                  archived REAL DEFAULT 0.0,       -- Phase 7C: 0.0=live; >0.0=archived at timestamp
                  -- mem0 port (dedup + TTL): content-aware fingerprint (sha256 of
                  -- fact||sorted-tags) for exact-duplicate detection; ttl_seconds +
                  -- expiration_timestamp for time-bounded cards. All additive.
                  content_fingerprint TEXT,        -- sha256 hex; NULL => not yet computed
                  ttl_seconds INTEGER,             -- seconds from timestamp; NULL => never expires
                  expiration_timestamp REAL        -- timestamp + ttl_seconds; NULL => never expires
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
            # Phase 7C: Ebbinghaus decay & graph consolidation columns
            if "stability" not in cols:
                cur.execute("ALTER TABLE memories ADD COLUMN stability REAL DEFAULT 1.0")
            if "importance" not in cols:
                cur.execute("ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0.0")
            if "archived" not in cols:
                cur.execute("ALTER TABLE memories ADD COLUMN archived REAL DEFAULT 0.0")
            # Multi-tier scoping: add scope column for user_id/agent_id/run_id isolation
            if "scope" not in cols:
                cur.execute("ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'default'")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)")
            # mem0 port (dedup + TTL): add content-aware fingerprint + TTL columns.
            # Idempotent ALTER guarded by PRAGMA table_info (same pattern used for
            # stability/archived/scope above). These columns are DISJOINT from scope
            # (scope already exists; these are additive for dedup + expiry).
            if "content_fingerprint" not in cols:
                cur.execute("ALTER TABLE memories ADD COLUMN content_fingerprint TEXT")
            if "ttl_seconds" not in cols:
                cur.execute("ALTER TABLE memories ADD COLUMN ttl_seconds INTEGER")
            if "expiration_timestamp" not in cols:
                cur.execute("ALTER TABLE memories ADD COLUMN expiration_timestamp REAL")
            # Partial index: only rows with a fingerprint are dedup-eligible, so the
            # Composite (scope, content_fingerprint): the dedup SELECT in
            # `add()` filters by BOTH columns (a card is only a dup of a
            # same-scope twin), so this composite serves that lookup directly.
            # The partial WHERE keeps the index small and skips NULL rows.
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_scope_fingerprint "
                "ON memories(scope, content_fingerprint) "
                "WHERE content_fingerprint IS NOT NULL"
            )
            cur.close()
        # Phase 7C: initialize semantic graph edge table
        graph.init_graph(self._conn)
        # Initialize revision-history table (mem0 port): append-only audit
        # trail for UPDATE/DELETE on memories rows. Additive — CREATE TABLE
        # IF NOT EXISTS, never drops/renames. See isotope_zero.core.history.
        history.init_history(self._conn)
        # Initialize scope column in card_edges table
        self._init_scope_in_graph()
        # Late-fusion hybrid search: FTS5 external-content index over `fact`.
        # `content='memories', content_rowid='rowid'` makes the FTS table a
        # *secondary index* on the base table — it stores only the tokenized
        # fact text, not a second copy of the full row, so it stays small and
        # is always re-derivable from `memories`. Three triggers keep it in
        # sync with EVERY write path (add/update/delete AND the eval harness's
        # direct-to-connection bulk seeder, which bypasses the Python methods).
        # The special 'delete' command on the contentless rowid is the FTS5 way
        # to evict a row from an external-content table (a plain DELETE from
        # memories_fts would leave a dangling entry pointing at a gone rowid).
        cur = self._conn.cursor()
        try:
            cur.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
                "fact, id UNINDEXED, content='memories', content_rowid='rowid', "
                "tokenize='unicode61 remove_diacritics 2')"
            )
            cur.execute(_FTS_AIU_TRIGGER)
            cur.execute(_FTS_AAD_TRIGGER)
            cur.execute(_FTS_AU_TRIGGER)
            # Backfill: if a pre-existing DB has `memories` rows but the FTS
            # index is short (first run against an old store, or a corrupt
            # index), rebuild atomically via the 'rebuild' command. The
            # triggers handle ongoing sync after this; this only heals history.
            fts_count = cur.execute("SELECT count(*) FROM memories_fts").fetchone()[0]
            live_count = cur.execute(
                "SELECT count(*) FROM memories WHERE superseded_by IS NULL "
                "AND archived = 0 AND fact IS NOT NULL"
            ).fetchone()[0]
            if fts_count < live_count:
                cur.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                # 'rebuild' pulls archived/superseded rows from the content
                # table too; evict them so search stays scoped to live cards.
                stale = cur.execute(
                    "SELECT m.rowid, m.fact, m.id FROM memories m "
                    "JOIN memories_fts f ON f.rowid = m.rowid "
                    "WHERE m.superseded_by IS NOT NULL OR m.archived != 0 "
                    "OR m.fact IS NULL"
                ).fetchall()
                for rowid, fact, cid in stale:
                    cur.execute(
                        "INSERT INTO memories_fts(memories_fts, rowid, fact, id) "
                        "VALUES ('delete', ?, ?, ?)",
                        (rowid, fact if fact is not None else "", cid if cid is not None else ""),
                    )
            cur.close()
        except sqlite3.OperationalError as e:
            # FTS5 may be absent on a stripped SQLite build. Hybrid search then
            # degrades to vector-only (search_time skip) rather than failing at
            # store construction — a read-only store is still fully usable.
            log.warning("FTS5 unavailable; hybrid search will be vector-only: %s", e)
            cur.close()

    def _init_scope_in_graph(self) -> None:
        """Add the `scope` column + index to `card_edges` for multi-tier scoping.

        Mirrors the `memories.scope` migration on the graph edge table so every
        edge carries the same deterministic scope string as the cards it links.
        Idempotent: a no-op when the column already exists (new DBs get it via
        `graph.init_graph`; legacy DBs are migrated here). The index is always
        re-created with IF NOT EXISTS, which is harmless.
        """
        with self._lock:
            cur = self._conn.cursor()
            try:
                cols = {row[1] for row in cur.execute("PRAGMA table_info(card_edges)").fetchall()}
                if "scope" not in cols:
                    cur.execute("ALTER TABLE card_edges ADD COLUMN scope TEXT NOT NULL DEFAULT 'default'")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_card_edges_scope ON card_edges(scope)")
                self._conn.commit()
            finally:
                cur.close()

    def rebuild_fts(self) -> None:
        """Fully rebuild the FTS5 index from live `memories` rows.

        Used after bulk repairs / direct-row writes that left the FTS index
        inconsistent, or to reclaim space after many deletes. Safe under
        concurrent readers: the rebuild runs under `_lock`. Idempotent.

        FTS5 external-content caveat: `DELETE FROM memories_fts` is INVALID
        on a contentless/external-content table (it no-ops AND can corrupt the
        index). The supported atomic rebuild is the special `'rebuild'`
        command, which drops + re-seeds the index from the content table
        (`memories`) in one step. We follow it with a filtered re-seed so
        archived/superseded rows — which the triggers kept OUT at write time
        but `'rebuild'` would pull IN from the raw content table — are pruned.
        """
        with self._lock:
            cur = self._conn.cursor()
            try:
                # 'rebuild': drop the whole FTS index + re-populate from the
                # content table's CURRENT rows. Fast and atomic.
                cur.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                # The rebuild pulls ALL memories rows (including archived/
                # superseded), but those must stay out of search. Evict them
                # via the per-rowid 'delete' command (the only valid way to
                # remove a row from an external-content FTS table).
                stale = cur.execute(
                    "SELECT m.rowid, m.fact, m.id FROM memories m "
                    "JOIN memories_fts f ON f.rowid = m.rowid "
                    "WHERE m.superseded_by IS NOT NULL OR m.archived != 0 "
                    "OR m.fact IS NULL"
                ).fetchall()
                for rowid, fact, cid in stale:
                    cur.execute(
                        "INSERT INTO memories_fts(memories_fts, rowid, fact, id) "
                        "VALUES ('delete', ?, ?, ?)",
                        (rowid, fact if fact is not None else "", cid if cid is not None else ""),
                    )
            finally:
                cur.close()

    def invalidate_vector_cache(self) -> None:
        """Drop the cached vector matrix + per-scope masks so they rebuild on
        the next search.

        The Python-side vector cache (`_vec_matrix`, `_vec_ids`,
        `_scope_index_cache`) is only invalidated by the store's own write
        methods (`add`/`update`/`delete`/...), which call `_mark_vec_dirty`.
        A writer that goes AROUND those — the eval harness's direct-to-
        connection bulk seeder, or any other `conn.execute(...)` that changes
        `embedding`/`scope`/row presence — leaves the cache stale: new cards
        are invisible to search, deleted cards cause index drift, and changed
        `scope` values keep the old per-scope boolean mask. FTS5 stays correct
        (triggers fire on any write regardless of issuer) but the vector path
        does not. External writers should call this — the vector-side analogue
        of `rebuild_fts` — after any direct write that touches embeddings,
        scope, or row count. Idempotent; safe under concurrent readers.
        """
        self._mark_vec_dirty()

    def _init_scope_in_graph(self) -> None:
        """Add a `scope` column to `card_edges` to mirror `memories.scope`.

        Multi-tier scoping (user_id/agent_id/run_id isolation) added a `scope`
        column to `memories`; the same isolation applies to the semantic graph,
        so edges are scoped too. Idempotent: a no-op once the column exists.
        Default 'default' matches the `memories.scope` default so unscoped
        stores keep working with zero behavioral change.
        """
        with self._lock:
            cur = self._conn.cursor()
            try:
                edge_cols = {
                    row[1]
                    for row in cur.execute("PRAGMA table_info(card_edges)").fetchall()
                }
                if "scope" not in edge_cols:
                    cur.execute(
                        "ALTER TABLE card_edges ADD COLUMN scope "
                        "TEXT NOT NULL DEFAULT 'default'"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_card_edges_scope "
                        "ON card_edges(scope)"
                    )
            finally:
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
        last_access, superseded_by, stability, importance, archived, scope.
        Older SELECTs that don't include the trailing columns are handled by
        length-checking the row (defensive against any stale call sites).
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
        stability = float(row[10]) if len(row) > 10 and row[10] is not None else 1.0
        importance = float(row[11]) if len(row) > 11 and row[11] is not None else 0.0
        archived = float(row[12]) if len(row) > 12 and row[12] is not None else 0.0
        # Multi-tier scoping: column 13 (defensive default for legacy rows).
        scope = str(row[13]) if len(row) > 13 and row[13] is not None else "default"
        # mem0 port (dedup + TTL): columns 14-16 (defensive None for legacy rows).
        content_fingerprint = str(row[14]) if len(row) > 14 and row[14] is not None else None
        ttl_seconds = int(row[15]) if len(row) > 15 and row[15] is not None else None
        expiration_timestamp = (
            float(row[16]) if len(row) > 16 and row[16] is not None else None
        )
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
            stability=stability,
            importance=importance,
            archived=archived,
            scope=scope,
            content_fingerprint=content_fingerprint,
            ttl_seconds=ttl_seconds,
            expiration_timestamp=expiration_timestamp,
        )

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #
    def add(self, card: MemoryCard, scope: str | None = None) -> None:
        """Persist a card + its embedding (BLOB, or NULL if embedding is None).

        ``scope`` isolates the card behind a multi-tier boundary
        (``user_id=...&agent_id=...&run_id=...``); when a non-None string is
        passed it is stamped onto the card and stored so downstream retrieval
        can filter it. When omitted (``None``) the card keeps whatever scope it
        already carries (its dataclass default is ``"default"``), preserving
        backward compatibility for callers that pre-populate ``card.scope``.

        mem0 port (dedup + TTL):
          - **Content-aware dedup (opt-in).** When the store is constructed
            with ``enable_dedup=True``, ``add()`` computes the card's content
            fingerprint (``content_aware_fingerprint(fact, tags)`` — sha256 of
            ``fact + "||" + ",".join(sorted(tags))``, ports mem0's
            ``hashlib.md5(text.encode())`` at ``mem0/memory/main.py:1010`` to a
            content-aware sha256). If a LIVE (non-superseded, non-archived)
            card with the same fingerprint already exists **in the same scope**
            (mirroring mem0, which builds its ``seen_hashes`` set from
            scope-filtered ``existing_results``), the duplicate write is a
            no-op INSERT: we touch the existing card (bump access/stability)
            and return early. The default ``enable_dedup=False`` skips the
            fingerprint entirely and inserts every card — additive,
            off-by-default, preserving the pre-port ``add()`` semantics so the
            scope-isolation tests (which deliberately insert same-fact cards
            across tenants) keep working. When the dedup helper is
            unavailable (sibling module not importable), the fingerprint is
            skipped regardless of the flag (backward-compatible degrade).
          - **TTL.** When ``card.ttl_seconds`` is set, compute
            ``expiration_timestamp = card.timestamp + card.ttl_seconds`` so the
            retrieval path can exclude the card once ``now > expiration``. A
            ``None`` ``ttl_seconds`` (the default) means "never expires" — the
            pre-TTL behavior, preserved for every caller that doesn't pass it.
        """
        log.debug("add id=%s fact=%r emb=%d scope=%s", card.id, card.fact[:80], len(card.embedding) if card.embedding else 0, scope)
        # Stamp the scope onto the card so the in-memory object matches the row
        # and callers reading the returned/round-tripped card see the same value.
        # Only stamp when the caller EXPLICITLY passed a scope; otherwise leave
        # the card's existing scope untouched (the documented behavior).
        if scope is not None:
            card.scope = scope
        row_scope = card.scope  # what actually gets written to the row
        tags_json = json.dumps(list(card.tags)) if card.tags else None
        blob = self._encode_embedding(card.embedding)
        # A fresh write is itself a touch: default last_access to timestamp
        # when the caller didn't set it, so the decay scorer sees a vital card.
        last_access = card.last_access if card.last_access else card.timestamp
        # mem0 port: content-aware fingerprint for dedup. Computed BEFORE the
        # INSERT so the same value is both stored (for cross-batch dedup) and
        # used to short-circuit a duplicate write. ``None`` (=> skip dedup,
        # insert normally) in TWO cases: the dedup helper is unavailable
        # (import guarded at module top), OR ``self.enable_dedup`` is False
        # (the default — additive, off-by-default, preserving the pre-port
        # add() semantics where two same-scope/same-fact cards both insert).
        fp: str | None = None
        if self.enable_dedup and content_aware_fingerprint is not None:
            fp = content_aware_fingerprint(card.fact, sorted(card.tags))
        # TTL: stamp the expiration timestamp onto the card so the in-memory
        # object matches the row. Only when the caller set a ttl_seconds.
        expiration_ts: float | None = None
        if card.ttl_seconds is not None:
            expiration_ts = float(card.timestamp) + float(card.ttl_seconds)
        card.content_fingerprint = fp
        card.expiration_timestamp = expiration_ts
        # mem0 port (dedup): if a LIVE card with the same content fingerprint
        # already exists, skip the INSERT and touch the existing row instead
        # (ports mem0's ``seen_hashes`` skip at main.py:1011). The SELECT
        # filters to non-superseded, non-archived rows so a superseded/
        # archived twin does NOT suppress a fresh write of the same fact.
        #
        # The dup-SELECT runs INSIDE the lock (it shares the connection), but
        # the touch is deferred to AFTER the ``with self._lock`` block: touch
        # re-acquires ``self._lock`` (line 1070), and ``threading.Lock`` is
        # non-reentrant, so calling it here would self-deadlock. We capture the
        # dup id and fall through to the post-lock touch instead.
        touch_existing_id: str | None = None
        with self._lock:
            cur = self._conn.cursor()
            if fp is not None:
                # Scope the dedup query to the SAME scope: a card is only a
                # duplicate of another card in the SAME tenant boundary. This
                # mirrors mem0, which builds its `existing_hashes` set from
                # `existing_results` — and those results are already filtered
                # to the same user_id/agent_id/run_id before the hash is
                # compared (mem0/memory/main.py:993-1000 + :1004-1010). Two
                # cards with identical fact+tags but DIFFERENT scopes (e.g. the
                # same preference stored independently by two users) are
                # semantically distinct memories and must both be kept — the
                # whole point of multi-tier scoping is same-fact isolation
                # across tenants. ``row_scope`` is what gets written to this
                # row (the stamped scope, or the card's existing default).
                dup = cur.execute(
                    "SELECT id FROM memories "
                    "WHERE content_fingerprint = ? AND scope = ? "
                    "AND superseded_by IS NULL AND archived = 0",
                    (fp, row_scope),
                ).fetchone()
                if dup is not None:
                    # A live twin exists: skip the INSERT and touch it after
                    # releasing the lock (touch re-acquires self._lock, which is
                    # non-reentrant -> would self-deadlock if called in here).
                    touch_existing_id = str(dup[0])
            if touch_existing_id is None:
                cur.execute(
                    """
                    INSERT INTO memories(id, fact, evidence, timestamp, tags, source_tokens, embedding, access_count, last_access, superseded_by, stability, importance, archived, scope, content_fingerprint, ttl_seconds, expiration_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        card.stability,
                        card.importance,
                        card.archived,
                        row_scope,
                        fp,
                        card.ttl_seconds,
                        expiration_ts,
                    ),
                )
                # Phase 7C: best-effort graph auto-linking after insert
                try:
                    cur.execute(
                        "SELECT id, embedding FROM memories "
                        "WHERE embedding IS NOT NULL AND superseded_by IS NULL "
                        "AND archived = 0 AND id != ?",
                        (card.id,),
                    )
                    rows = cur.fetchall()
                    all_embeddings: list[tuple[str, list[float]]] = []
                    for r in rows:
                        other_emb = self._decode_embedding(r[1])
                        if other_emb is not None:
                            all_embeddings.append((r[0], other_emb))
                    graph.auto_link_cards(
                        self._conn, card.id, card.tags, card.embedding, all_embeddings
                    )
                except Exception:
                    pass  # best-effort: graph linking failure must not break insertion
            cur.close()
        if touch_existing_id is not None:
            # Dedup hit: bump the existing card's access_count/stability and
            # invalidate the vec cache (the touch may have changed ordering).
            # Done OUTSIDE the lock to avoid the self-deadlock described above.
            self._mark_vec_dirty()
            self.touch(touch_existing_id)
            return
        self._mark_vec_dirty()

    def update(self, card: MemoryCard) -> None:
        """Upsert a card by id. If absent, inserts; if present, overwrites.

        Carries the card's own ``scope`` field into the row; there is no
        separate ``scope`` argument (unlike ``add``) because update operates on
        an existing card that already carries its scope.
        """
        tags_json = json.dumps(list(card.tags)) if card.tags else None
        blob = self._encode_embedding(card.embedding)
        last_access = card.last_access if card.last_access else card.timestamp
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO memories(id, fact, evidence, timestamp, tags, source_tokens, embedding, access_count, last_access, superseded_by, stability, importance, archived, scope)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    fact = excluded.fact,
                    evidence = excluded.evidence,
                    timestamp = excluded.timestamp,
                    tags = excluded.tags,
                    source_tokens = excluded.source_tokens,
                    embedding = excluded.embedding,
                    access_count = excluded.access_count,
                    last_access = excluded.last_access,
                    superseded_by = excluded.superseded_by,
                    stability = excluded.stability,
                    importance = excluded.importance,
                    archived = excluded.archived,
                    scope = excluded.scope
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
                    card.stability,
                    card.importance,
                    card.archived,
                    card.scope,
                ),
            )
            cur.close()
        self._mark_vec_dirty()

    def delete(self, memory_id: str) -> bool:
        """Delete by id. Returns True if a row was deleted, False if not found."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            deleted = cur.rowcount > 0
            cur.close()
        log.debug("delete id=%s deleted=%s", memory_id, deleted)
        self._mark_vec_dirty()
        return deleted

    def delete_scope(self, scope: str) -> int:
        """Delete ALL memories + graph edges for a scope. Returns count deleted.

        Multi-tier scoping teardown: when a run/agent/user scope is retired,
        this atomically removes every card stored under that exact scope
        string AND every ``card_edges`` row carrying that scope, so no stale
        cross-scope links survive. Runs under one lock acquisition so a
        concurrent reader never sees a half-deleted scope. The FTS5
        external-content triggers (``memories_fts_aad``) fire per-row, keeping
        the full-text index in sync automatically.

        Returns the number of ``memories`` rows deleted. ``scope="default"``
        is NOT special-cased — it deletes default-scope cards just like any
        other (callers who want to protect it can guard at a higher layer).
        """
        with self._lock:
            cur = self._conn.cursor()
            # Delete graph edges for this scope first. Edges carry their own
            # scope column (see _init_scope_in_graph), so we filter on it
            # directly rather than JOINing to memories (which would miss edges
            # whose card was already deleted in a prior call).
            cur.execute("DELETE FROM card_edges WHERE scope = ?", (scope,))
            # Then the cards themselves.
            cur.execute("DELETE FROM memories WHERE scope = ?", (scope,))
            deleted = cur.rowcount
            cur.close()
        log.debug("delete_scope scope=%s deleted=%d", scope, deleted)
        self._mark_vec_dirty()
        return deleted

    def archive_card(self, card_id: str) -> bool:
        """Mark a card as archived (soft-delete). Returns True if archived.

        Sets ``archived`` to the current timestamp so the card is excluded from
        ``all()``, ``sql_lookup()``, ``vector_search()``, and ``count()``, but
        remains retrievable by direct ``get()`` for the audit trail. Idempotent:
        an already-archived card is a no-op. Returns False if the card was not
        found.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT archived FROM memories WHERE id = ?", (card_id,)
            )
            row = cur.fetchone()
            if row is None:
                cur.close()
                return False
            already_archived = float(row[0]) if row[0] is not None else 0.0
            if already_archived > 0.0:
                cur.close()
                log.debug("archive_card id=%s already archived", card_id)
                return True  # idempotent
            ts = now_ts()
            cur.execute(
                "UPDATE memories SET archived = ? WHERE id = ?", (ts, card_id)
            )
            cur.close()
        log.debug("archive_card id=%s at=%.3f", card_id, ts)
        self._mark_vec_dirty()
        return True

    # ------------------------------------------------------------------ #
    # Phase 3: access tracking + batch consolidation
    # ------------------------------------------------------------------ #
    def touch(self, memory_id: str, at: float | None = None) -> None:
        """Record a recall: increment access_count, set last_access, and update
        Ebbinghaus stability S via spaced-repetition dynamics.

        Called by the query router whenever a card surfaces as a hit, so the
        temporal-decay scorer can distinguish recalled (vital) cards from
        never-recalled (cold) ones. `at` defaults to now; tests pass a fixed
        timestamp for determinism. Idempotent on missing ids (no-op).
        """
        ts = now_ts() if at is None else at
        with self._lock:
            cur = self._conn.cursor()
            # Read current access_count, stability, importance to compute the
            # updated stability via update_stability.
            cur.execute(
                "SELECT access_count, stability, importance FROM memories WHERE id = ?",
                (memory_id,),
            )
            row = cur.fetchone()
            if row is None:
                cur.close()
                return
            cur_acc = (int(row[0]) if row[0] is not None else 0) + 1
            cur_stability = float(row[1]) if row[1] is not None else 1.0
            cur_importance = float(row[2]) if row[2] is not None else 0.0
            new_stability = update_stability(cur_stability, cur_acc, cur_importance)
            cur.execute(
                "UPDATE memories SET access_count = ?, last_access = ?, stability = ? WHERE id = ?",
                (cur_acc, ts, new_stability, memory_id),
            )
            cur.close()
        log.debug("touch id=%s at=%.3f stability=%.3f", memory_id, ts, new_stability)

    def purge_expired(self) -> int:
        """Delete every card whose TTL has elapsed. Returns the count deleted.

        mem0 port (TTL): hard-deletes rows where ``expiration_timestamp`` is set
        AND ``<= unixepoch()`` (now, in seconds). A row with a NULL
        ``expiration_timestamp`` (the default — no TTL set) is NEVER deleted by
        this method, preserving backward compatibility for every pre-TTL card.

        The DELETE fires the FTS5 ``memories_fts_aad`` trigger per row so the
        full-text index stays in sync automatically (no separate FTS eviction
        needed). The vector cache is invalidated by ``_mark_vec_dirty`` so the
        next search rebuilds without the purged rows.

        Called at the START of ``consolidate_memories`` so the consolidation
        sweep sees a current view of live rows. Also callable directly (e.g.
        by a maintenance task). Idempotent: a no-op (returns 0) when no expired
        rows exist.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "DELETE FROM memories "
                "WHERE expiration_timestamp IS NOT NULL "
                "AND expiration_timestamp <= unixepoch()"
            )
            deleted = cur.rowcount
            cur.close()
        if deleted:
            log.debug("purge_expired deleted=%d", deleted)
            self._mark_vec_dirty()
        return int(deleted) if deleted and deleted > 0 else 0

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
                SELECT id, fact, evidence, timestamp, tags, source_tokens, embedding, access_count, last_access, superseded_by, stability, importance, archived, scope, content_fingerprint, ttl_seconds, expiration_timestamp
                FROM memories WHERE id IN ({placeholders})
                ORDER BY timestamp ASC, id ASC
                """,
                tuple(ids),
            )
            rows = cur.fetchall()
            cur.close()
        return [self._row_to_card(r) for r in rows]

    def rebuild_graph(self, cosine_threshold: float = 0.75) -> int:
        """Rebuild the semantic ``card_edges`` graph from ALL live cards in one
        vectorized pass — the bulk-seed counterpart to per-``add()`` auto-linking.

        Drops existing edges and recomputes the full pairwise cosine matrix via
        a single NumPy ``M @ M.T`` matmul (``core.graph.bulk_link_cards``),
        instead of the naive O(n²) per-``add()`` path that reloaded every
        embedding on each insert. Use after bulk SQL seeding (e.g. the eval
        harness / benchmarks) to materialize the graph cheaply.

        Returns the number of edges written.
        """
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("DELETE FROM card_edges")
                rows = cur.execute(
                    "SELECT id, tags, embedding FROM memories "
                    "WHERE superseded_by IS NULL AND archived = 0"
                ).fetchall()
            finally:
                cur.close()
        cards: list[tuple[str, list[str], list[float] | None]] = []
        for r in rows:
            tags = json.loads(r[1]) if r[1] else []
            emb = self._decode_embedding(r[2])
            cards.append((r[0], tags, emb))
        return graph.bulk_link_cards(self._conn, cards, cosine_threshold)

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
        # mem0 port (TTL): purge expired cards at the START of consolidation so
        # the sweep sees a current view of live rows. Runs under the sweep's
        # transaction (the BEGIN IMMEDIATE below) so the purge + merge are atomic.
        # purge_expired() opens its own lock — call it BEFORE acquiring this
        # method's lock to avoid a self-deadlock. The purge is best-effort and
        # idempotent; any rows it misses here are caught on the next sweep.
        #
        # Async-subclass safety: ``AsyncMemoryEngine`` overrides ``purge_expired``
        # as ``async def`` (returns a coroutine). Calling it from this sync
        # method would leak an un-awaited coroutine (the purge would never
        # run). Detect that case and drive the coroutine to completion on a
        # fresh event loop — this method is invoked via ``asyncio.to_thread``
        # so no loop is running here, and a private loop is the documented way
        # to run one coroutine to completion from sync code. The sync path
        # (plain ``MemoryStore``) returns an int, not a coroutine, and is
        # unchanged.
        try:
            _purged = self.purge_expired()
            if iscoroutine(_purged):
                import asyncio as _asyncio

                _loop = _asyncio.new_event_loop()
                try:
                    _loop.run_until_complete(_purged)
                finally:
                    _loop.close()
        except Exception:  # noqa: BLE001 — purge must never break consolidation
            log.debug("purge_expired during consolidate_memories failed; continuing", exc_info=True)
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                # Upsert survivors.
                for card in merged_cards:
                    tags_json = json.dumps(list(card.tags)) if card.tags else None
                    blob = self._encode_embedding(card.embedding)
                    last_access = card.last_access if card.last_access else card.timestamp
                    # mem0 port: carry the dedup fingerprint + TTL onto the
                    # survivor so the merged row keeps its dedup identity +
                    # expiry (a merged survivor should still suppress future
                    # duplicates of the same fact and still expire on its TTL).
                    if content_aware_fingerprint is not None and not card.content_fingerprint:
                        card.content_fingerprint = content_aware_fingerprint(
                            card.fact, sorted(card.tags)
                        )
                    fp = card.content_fingerprint
                    exp_ts = card.expiration_timestamp
                    if card.ttl_seconds is not None and exp_ts is None:
                        exp_ts = float(card.timestamp) + float(card.ttl_seconds)
                    cur.execute(
                        """
                        INSERT INTO memories(id, fact, evidence, timestamp, tags, source_tokens, embedding, access_count, last_access, superseded_by, stability, importance, archived, content_fingerprint, ttl_seconds, expiration_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            fact = excluded.fact,
                            evidence = excluded.evidence,
                            timestamp = excluded.timestamp,
                            tags = excluded.tags,
                            source_tokens = excluded.source_tokens,
                            embedding = excluded.embedding,
                            access_count = excluded.access_count,
                            last_access = excluded.last_access,
                            superseded_by = excluded.superseded_by,
                            stability = excluded.stability,
                            importance = excluded.importance,
                            archived = excluded.archived,
                            content_fingerprint = excluded.content_fingerprint,
                            ttl_seconds = excluded.ttl_seconds,
                            expiration_timestamp = excluded.expiration_timestamp
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
                            card.stability,
                            card.importance,
                            card.archived,
                            fp,
                            card.ttl_seconds,
                            exp_ts,
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
        """Fetch a single card by id, or None if absent.

        Returns the row even if archived/superseded/expired — this is the audit-
        trail accessor (see ``archive_card`` docstring). Retrieval paths (``all``,
        ``sql_lookup``, ``vector_search``) apply the live filters themselves.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id, fact, evidence, timestamp, tags, source_tokens, embedding, access_count, last_access, superseded_by, stability, importance, archived, scope, content_fingerprint, ttl_seconds, expiration_timestamp
                FROM memories WHERE id = ?
                """,
                (memory_id,),
            )
            row = cur.fetchone()
            cur.close()
        return self._row_to_card(row) if row is not None else None

    def all(self) -> list[MemoryCard]:
        """Return every live (non-archived, non-superseded, non-expired) card,
        ordered by timestamp ascending."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id, fact, evidence, timestamp, tags, source_tokens, embedding, access_count, last_access, superseded_by, stability, importance, archived, scope, content_fingerprint, ttl_seconds, expiration_timestamp
                FROM memories NOT INDEXED
                WHERE superseded_by IS NULL AND archived = 0
                AND (expiration_timestamp IS NULL OR expiration_timestamp > unixepoch())
                ORDER BY timestamp ASC, id ASC
                """
            )
            rows = cur.fetchall()
            cur.close()
        return [self._row_to_card(r) for r in rows]

    def sql_lookup(self, field: str, value: str, scope: str | None = None) -> list[MemoryCard]:
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

        Multi-tier scoping (``scope``, ports ``mem0/memory/main.py:407``):
        when ``scope`` is a non-None string, ONLY cards visible to that scope
        are returned — a card matches iff its stored scope equals the bound
        value, OR equals the ``'default'`` global sentinel, OR is NULL (the
        pre-scoping row state). This is the same backward-compatible visibility
        rule :func:`isotope_zero.core.scoping.match_scope` encodes: a global
        card is visible to every scoped query, and ``scope=None`` (the default)
        disables filtering entirely so pre-scoping callers see no change.
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
                    hits = self._lookup_exact_locked(col, value, scope)
                    if hits is not None:
                        return hits
                if col == "fact":
                    return self._lookup_substring_fact_locked(value, scope)
                return self._lookup_substring_locked(col, value, scope)

        if field == "tags":
            # Python-side membership over a single tag.
            target = value
            results: list[MemoryCard] = []
            with self._lock:
                results = self._lookup_tags_locked(target, scope)
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

    def _lookup_exact_locked(self, col: str, value: str, scope: str | None = None) -> list[MemoryCard] | None:
        """Index-usable exact match. Returns the cards, or None when no row
        matches so the caller can fall through to the substring path.

        ``scope`` (default ``None`` = no filter) appends
        :data:`_SCOPE_CLAUSE` so only cards visible to the scope are returned.
        """
        sql = _SQL_EXACT.format(select=_SQL_LOOKUP_SELECT, col=col, scope=_scope_clause(scope))
        cur = self._stmt_cursor(sql)
        params: tuple[Any, ...] = (value, scope) if scope is not None else (value,)
        cur.execute(sql, params)
        rows = cur.fetchall()
        if not rows:
            return None
        return [self._row_to_card(r) for r in rows]

    def _lookup_substring_fact_locked(self, value: str, scope: str | None = None) -> list[MemoryCard]:
        """Substring LIKE for `fact`: covering idx_fact scan + fetch by id.

        ``scope`` (default ``None`` = no filter) is threaded into BOTH the
        id-scan and the fetch-by-id so an out-of-scope id never hydrates.
        """
        pattern = f"%{value}%"
        sql = _SQL_SUBSTR_IDS.format(col="fact", scope=_scope_clause(scope))
        cur = self._stmt_cursor(sql)
        params: tuple[Any, ...] = (pattern, scope) if scope is not None else (pattern,)
        cur.execute(sql, params)
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return []
        if len(ids) > _SUBSET_MATCH_LIMIT:
            # Pathologically broad match: fall back to the single-pass scan
            # rather than a huge IN(...) list.
            return self._lookup_substring_locked("fact", value, scope)
        rows = self._fetch_by_ids_locked(ids, scope)
        return [self._row_to_card(r) for r in rows]

    def _lookup_substring_locked(self, col: str, value: str, scope: str | None = None) -> list[MemoryCard]:
        """Direct LIKE scan (fields without a covering index, e.g. evidence).
        `value` is the raw lookup term; the surrounding wildcards are added
        here so every caller passes the same raw value.

        ``scope`` (default ``None`` = no filter) appends :data:`_SCOPE_CLAUSE`.
        """
        pattern = f"%{value}%"
        sql = _SQL_SUBSTR_SCAN.format(select=_SQL_LOOKUP_SELECT, col=col, scope=_scope_clause(scope))
        cur = self._stmt_cursor(sql)
        params: tuple[Any, ...] = (pattern, scope) if scope is not None else (pattern,)
        cur.execute(sql, params)
        rows = cur.fetchall()
        return [self._row_to_card(r) for r in rows]

    def _fetch_by_ids_locked(self, ids: list[str], scope: str | None = None) -> list[tuple[Any, ...]]:
        """Fetch full rows for a small id list, re-applying the superseded-by
        audit filter and the canonical (timestamp, id) ordering. SQL varies
        with the placeholder count so it is NOT cached in the statement cache.

        ``scope`` (default ``None`` = no filter) re-applies :data:`_SCOPE_CLAUSE`
        so the substring-fact path's id-list fetch can never hydrate an out-of-
        scope card that slipped into the id list (defense in depth — the id
        scan already filtered, but a re-check on fetch is cheap and correct).
        """
        placeholders = ",".join("?" for _ in ids)
        sql = _SQL_IN_FETCH.format(select=_SQL_LOOKUP_SELECT, placeholders=placeholders, scope=_scope_clause(scope))
        cur = self._conn.cursor()
        try:
            if scope is not None:
                cur.execute(sql, (*ids, scope))
            else:
                cur.execute(sql, ids)
            return cur.fetchall()
        finally:
            cur.close()

    def _lookup_tags_locked(self, target: str, scope: str | None = None) -> list[MemoryCard]:
        """Python-side JSON-tags membership over the non-superseded rows.

        ``scope`` (default ``None`` = no filter) appends :data:`_SCOPE_CLAUSE`
        to the candidate SELECT so only in-scope rows are loaded for the
        Python-side membership test.
        """
        results: list[MemoryCard] = []
        sql = _SQL_TAGS_SCAN.format(select=_SQL_LOOKUP_SELECT, scope=_scope_clause(scope))
        cur = self._stmt_cursor(sql)
        if scope is not None:
            cur.execute(sql, (scope,))
        else:
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

        Phase 5: also invalidates the `MmapVectorStore` (file-backed memmap)
        so its stale file + hot LRU are never read after a write. The memmap
        is rebuilt on the next `_ensure_vec_cache`.
        """
        self._vec_dirty = True
        self._vec_hetero = False
        # Any write may change which rows belong to which scope, so the
        # scope->row-index cache is stale too. Drop it; it rebuilds lazily.
        self._scope_index_cache.clear()
        if self._mmap_store is not None:
            self._mmap_store.invalidate()

    def _ensure_vec_cache(self, np) -> Any:
        """Return the cached (n, dim) float32 matrix, refreshing lazily.

        Builds on first use and whenever `_vec_dirty` is set: SELECTs only
        (id, timestamp, embedding) for rows with a non-NULL embedding,
        decodes each BLOB straight into a numpy float32 row (zero-Python
        unpacking), and stacks them once. Returns None when there are no
        embeddable rows or when rows have heterogeneous lengths (not
        representable as a dense matrix -> caller falls back to Python).

        Phase 5 (mmap path): when `use_mmap=True`, the matrix is a file-backed
        `np.memmap` built by `MmapVectorStore.build` instead of `np.stack`.
        The memmap is a drop-in: it supports `@` and `[:, :n]` slicing
        identically to a heap ndarray (numpy pages in cold rows on demand).
        The parallel `self._vec_ids` / `self._vec_ts` are populated from the
        mmap store's index so the post-score top-k selection is identical to
        the heap path. On ANY failure (hetero dim, build error, or
        `use_mmap=False`), the heap `np.stack` path is the transparent
        fallback — the caller sees the same matrix contract either way.
        """
        if self._vec_hetero:
            return None
        if (not self._vec_dirty) and self._vec_matrix is not None:
            return self._vec_matrix

        # Phase 5: try the file-backed mmap backend first when enabled. The
        # mmap store does its own SELECT + write + memmap-open under our lock.
        # On hetero-dim or build failure it raises _HeteroDim / a generic
        # Exception; we fall back to the heap path so search still works (and
        # latches hetero). NO 1-bit quantization — full float32 only.
        if self.use_mmap:
            try:
                from isotope_zero.core.mmap_store import MmapVectorStore, _HeteroDim

                if self._mmap_store is None:
                    self._mmap_store = MmapVectorStore(self.db_path, self._conn)
                with self._lock:
                    mm = self._mmap_store.build(np)
                    # Clear dirty while still holding the lock (same contract
                    # as the heap path below): any write after this point
                    # re-sets _vec_dirty via _mark_vec_dirty -> mmap.invalidate.
                    self._vec_dirty = False
                if mm is not None:
                    # Populate the parallel id/ts/scope lists from the mmap
                    # store so vector_search's top-k selection AND scope masking
                    # are identical to the heap path (same row order, same
                    # timestamps, same ids, same scopes).
                    self._vec_ids = list(self._mmap_store.ids)
                    self._vec_ts = list(self._mmap_store.ts)
                    self._vec_scopes = list(self._mmap_store.scopes)
                    self._scope_index_cache.clear()
                    self._vec_matrix = mm
                    return self._vec_matrix
                # mm is None -> no embeddable rows; fall through to heap path
                # which will also return None (and clear the parallel lists).
            except _HeteroDim:
                self._vec_hetero = True
                self._vec_matrix = None
                return None
            except Exception as exc:  # noqa: BLE001 — mmap must never break search
                log.warning("mmap vector build failed (%s); using heap path.", exc)
                # Fall through to the heap path below.

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                # Match `all()` exactly: superseded (audit-trail folded) and
                # archived rows must not surface in vector search either.
                # NOT INDEXED keeps this a sequential table scan (we read
                # every embedding BLOB anyway, so an index scan + rowid
                # lookups would only add random I/O). scope is fetched here so
                # `vector_search` can mask out-of-scope rows in C without a
                # second round-trip.
                "SELECT id, timestamp, embedding, scope FROM memories NOT INDEXED "
                "WHERE embedding IS NOT NULL AND superseded_by IS NULL AND archived = 0 "
                "AND (expiration_timestamp IS NULL OR expiration_timestamp > unixepoch()) "
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
        scopes: list[str] = []
        arrs = []
        dim: int | None = None
        for rid, rts, blob, scope_val in rows:
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
            scopes.append(scope_val if scope_val is not None else "default")
            arrs.append(a)

        self._vec_ids = ids
        self._vec_ts = ts
        self._vec_scopes = scopes
        # scope index cache rebuilds lazily on next vector_search(scope=...)
        self._scope_index_cache.clear()
        self._vec_matrix = np.stack(arrs) if arrs else None
        return self._vec_matrix

    def vector_search(
        self,
        query_vec: list[float],
        k: int = 5,
        alpha: float = 0.70,
        scope: str = "default",
    ) -> list[tuple[MemoryCard, float]]:
        """Top-k cosine similarity search via dot product, fused with Ebbinghaus
        temporal decay for recency-aware retrieval.

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
        capped at 1.0).

        When ``alpha < 1.0``, cosine scores are fused with Ebbinghaus retention
        (``R(t) = exp(- delta_t / S)``) via ``hybrid_score``:
        ``final = alpha * cosine + (1-alpha) * retention``. Top-k candidates are
        selected by cosine first (cheap), then re-ranked by hybrid score.
        ``alpha = 1.0`` is pure cosine — identical to the original behavior.
        Ordering is (score desc, timestamp asc); identical (score, timestamp)
        tiebreak on id asc for determinism.

        Multi-tier scoping (``scope``): when set (default ``"default"``), ONLY
        cards stored under that exact scope string are scored — out-of-scope
        slots are masked to a below-floor score before the top-k selection so
        they can never surface. The mask is a cached numpy boolean array
        applied in C, adding <0.1ms to the search path at 10k cards. Pass
        ``scope=None`` to search ALL scopes (cross-tenant global retrieval).
        """
        if not query_vec or all(v == 0.0 for v in query_vec):
            return []

        try:
            import numpy as np
        except ImportError:
            np = None

        if np is None:
            return self._vector_search_fallback(query_vec, k, alpha, scope)

        ranked = self._vector_search_ranked(query_vec, k, alpha, np, scope)
        if not ranked:
            return []
        # ranked is [(final_score, cos_score, ts, id)] sorted desc; hydrate
        # the top-k cards in one batched SELECT.
        want_ids = [e[3] for e in ranked]
        by_card = {c.id: c for c in self.batch_get(want_ids)}
        return [(by_card[e[3]], e[0]) for e in ranked if e[3] in by_card]

    def _scope_mask(self, scope: str, np, total: int) -> Any:
        """Return a cached boolean numpy mask: True where row scope == ``scope``.

        The mask is aligned to the current ``_vec_matrix`` rows (row i of the
        matrix corresponds to ``_vec_scopes[i]``). Built once per scope and
        cached in ``_scope_index_cache``; invalidated on every write via
        ``_mark_vec_dirty``. ``total`` is the live matrix row count, used only
        to detect a stale cache after a rebuild that re-used the same scope key
        (defensive — the cache is cleared on rebuild, so this is a belt-and-
        suspenders length check).

        Returns ``None`` when scopes are unavailable (e.g. a legacy cache built
        before the scope column existed, or numpy is missing) — callers treat
        None as "no masking", preserving the original global behavior.
        """
        cached = self._scope_index_cache.get(scope)
        if cached is not None and cached.shape[0] == total:
            return cached
        # Rebuild from the parallel scope list. ``_vec_scopes`` is populated by
        # ``_ensure_vec_cache`` alongside ``_vec_ids``/``_vec_ts``; if it's
        # None/empty (no embeddable rows yet) there is nothing to mask.
        scopes = self._vec_scopes
        if not scopes:
            return None
        if len(scopes) != total:
            # Cache + matrix out of sync (a concurrent rebuild between
            # _ensure_vec_cache and here). Treat as un-maskable this pass;
            # the next call rebuilds a consistent cache.
            return None
        mask = np.array([s == scope for s in scopes], dtype=np.bool_)
        self._scope_index_cache[scope] = mask
        return mask

    def _vector_search_ranked(
        self,
        query_vec: list[float],
        k: int,
        alpha: float,
        np,
        scope: str | None = "default",
    ) -> list[tuple[float, float, float, str]]:
        """Score-only vector search: returns ``[(final_score, cosine, ts, id)]``
        sorted by ``(final_score desc, ts asc, id asc)`` — NO card hydration.

        Used by ``hybrid_search`` (which needs only ids + ranks for RRF, then
        does its own single batched hydration at the very end) to avoid a
        redundant ``batch_get`` over the (up to ``top_n_per_branch``) vector
        candidates. ``vector_search`` itself calls this then hydrates the top-k.

        ``scope`` (default ``"default"``): multi-tier filter. When non-None,
        only rows whose stored scope matches are eligible — out-of-scope scores
        are floored to -1.0 (below the [0,1] clip floor) BEFORE argpartition so
        they never enter the top-k. ``scope=None`` disables scoping (global).
        """
        matrix = self._ensure_vec_cache(np)
        if matrix is None or matrix.shape[0] == 0:
            # No searchable rows, OR heterogeneous embedding lengths (not
            # matrix-able). Delegate to the pure-Python loop, which is correct
            # for both cases (empty -> [], hetero -> per-card min-dim dot).
            return self._vector_search_fallback_ranked(query_vec, k, alpha, scope)

        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] == 0:
            return []

        # Normalize to a (maybe-sliced) matrix aligned with the query. The
        # min(qdim, emb_dim) sub-slice case is passed through identically to
        # the original code (matrix[:, :n] is a view, not a copy).
        com = matrix.shape[1]
        if q.shape[0] == com:
            use_matrix = matrix
            use_q = q
        else:
            # Preserve the reference loop's min(qdim, emb_dim) semantics over
            # the first `n` components (matrix[:, :n] is a view, not a copy).
            n = min(q.shape[0], com)
            if n == 0:
                return []
            use_matrix = matrix[:, :n]
            use_q = q[:n]
        # Scores via the Smart Bridge: the float32 batch dot-product runs on
        # NumPy/BLAS (zero-copy, GIL-released C kernel) — measured ~9-115x
        # faster than the Rust extension's copy-to-release-GIL path. The result
        # is RAW (unclipped); clipping/top-k stays right below. ``batch_cosine_similarity``
        # always uses NumPy and never raises here (it's just ``matrix @ q``).
        scores = native.batch_cosine_similarity(use_q, use_matrix)
        np.clip(scores, 0.0, 1.0, out=scores)

        # Multi-tier scoping: mask out-of-scope rows BEFORE the top-k selection.
        # We floor non-matching scores at -1.0 (strictly below the [0,1] clip
        # floor above) so they can never survive the ``scores >= thr`` filter or
        # an argpartition's largest-k selection. The mask is a cached numpy
        # boolean array built once per scope (see ``_scope_mask``); the boolean
        # assignment is a single C-level vectorized pass over ``n`` float32 —
        # measured <0.1ms at 10k rows, satisfying the scoping perf contract.
        # ``scope is None`` => global (no masking).
        if scope is not None:
            mask = self._scope_mask(scope, np, total=scores.shape[0])
            if mask is not None:
                # ``mask`` is True for in-scope rows; floor the rest.
                scores[~mask] = -1.0

        # Effective top-k can never exceed the in-scope population. When the
        # scope is sparse this avoids argpartition's "select k largest" pulling
        # in floored -1.0 rows that would then be filtered by the threshold
        # check below — keeping kk honest shrinks the candidate set to reals.
        # When ``mask is None`` (scopes unavailable) we fall back to the full
        # population, matching the scope=None path.
        if scope is not None and mask is not None:
            in_scope_n = int(mask.sum())
            kk = min(k, in_scope_n) if k > 0 else 0
        else:
            total = matrix.shape[0]
            kk = min(k, total) if k > 0 else 0
        if kk <= 0:
            return []

        # Top-k by cosine score; then expand to EVERY row tied at the k-th
        # boundary so timestamp tie-breaking matches the reference exactly.
        cand = np.argpartition(scores, -kk)[-kk:]
        thr = float(scores[cand].min())
        # The threshold floor excludes any -1.0 (out-of-scope) survivors.
        cand = np.flatnonzero(scores >= thr)
        entries = [
            (float(scores[i]), self._vec_ts[i], self._vec_ids[i])
            for i in cand.tolist()
            # Defensive: never surface a floored out-of-scope row even if a
            # future mask/cache bug left a -1.0 inside `cand`.
            if float(scores[i]) >= 0.0
        ]

        if alpha < 1.0 and entries:
            # Phase 7C: fuse cosine with Ebbinghaus retention.
            # Batch-load last_access + stability for candidates only, then
            # re-rank by hybrid score.
            candidate_ids = [e[2] for e in entries]
            placeholders = ",".join("?" for _ in candidate_ids)
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    f"SELECT id, last_access, stability FROM memories WHERE id IN ({placeholders})",
                    candidate_ids,
                )
                rows = cur.fetchall()
                cur.close()
            decay_data: dict[str, tuple[float, float]] = {}
            for rid, la, stab in rows:
                decay_data[rid] = (
                    float(la) if la is not None else 0.0,
                    float(stab) if stab is not None else 1.0,
                )
            current_ts = now_ts()
            scored: list[tuple[float, float, float, str]] = []
            for cos_score, ts_val, cid in entries:
                la, stab = decay_data.get(cid, (0.0, 1.0))
                retention = calculate_retention(la, stab, current_ts=current_ts)
                final_score = hybrid_score(cos_score, retention, alpha)
                scored.append((final_score, cos_score, ts_val, cid))
            # Sort by hybrid score desc, then timestamp asc, id asc.
            scored.sort(key=lambda e: (-e[0], e[2], e[3]))
            return scored[:kk]
        else:
            # Pure cosine (alpha == 1.0 or no entries). Original behavior.
            entries.sort(key=lambda e: (-e[0], e[1], e[2]))
            top = entries[:kk]
            # Return as (final=cos, cos, ts, id) so the caller's hydration path
            # is uniform regardless of the alpha branch taken here.
            return [(e[0], e[0], e[1], e[2]) for e in top]

    def _vector_search_fallback_ranked(
        self, query_vec: list[float], k: int, alpha: float = 0.70, scope: str | None = "default"
    ) -> list[tuple[float, float, float, str]]:
        """Pure-Python fallback's score-only counterpart to
        ``_vector_search_ranked``. Mirrors the fallback's per-card min-dim dot
        semantics but returns ``(final, cos, ts, id)`` without hydrating cards.
        Used when numpy is unavailable OR embeddings are heterogeneous."""
        # Reuse the existing fallback for correctness, then strip the cards.
        # (Keeping the fallback single-sourced avoids a second copy of the
        # hetero-dim dot-product loop drifting from the original.)
        hydrated = self._vector_search_fallback(query_vec, k, alpha, scope)
        return [(s, s, c.timestamp, c.id) for c, s in hydrated]

    # ------------------------------------------------------------------ #
    # Late Fusion hybrid search (vector BLAS + FTS5 BM25 + entity boost).
    # ------------------------------------------------------------------ #
    def hybrid_search(
        self,
        query: str,
        query_vec: list[float],
        k: int = 5,
        alpha: float = 0.70,
        top_n_per_branch: int = 30,
        scope: str = "default",
    ) -> list[tuple[MemoryCard, float]]:
        """Late-fusion hybrid retrieval: vector cosine (BLAS) + FTS5 BM25,
        combined via Reciprocal Rank Fusion (RRF) with an entity-graph boost.

        Branches run in isolation, each returning up to ``top_n_per_branch``
        candidates by their NATIVE ranking (cosine score for the vector
        branch, BM25 relevance for the keyword branch). RRF then merges them
        by RANK position, not raw score — this is what makes the fusion
        score-comparable: a cosine of 0.99 vs a BM25 of 12.0 are not directly
        commensurable, but "rank 1 in the vector branch" and "rank 1 in the
        keyword branch" are. The ``60`` smoothing constant in the denominator
        (per Cormack et al., "Reciprocal Rank Fusion", SIGIR 2009) damps the
        influence of very high ranks so a rank-1 hit from a noisy branch
        doesn't swamp a rank-2 hit from a precise one.

        Per-card score (higher = better):
            Score(d) = α/(60 + r_vec(d)) + (1-α)/(60 + r_bm25(d)) + Boost(d)
        where ``r_*`` are 1-indexed ranks WITHIN each branch (a card absent
        from a branch contributes 0 from that branch), and ``Boost(d)`` is the
        entity-graph term (0 when the card has no query-entity-linked neighbors).

        ``top_n_per_branch`` defaults to 30 (was 60). RRF's rank-position
        weighting makes a branch's contribution decay as ~1/(60+r): at rank 30
        the BM25 term is ~0.0023 (vs ~0.017 at rank 1, with (1-α)=0.3), and at
        rank 60 it is ~0.0012 — sub-millith of the fused score, below the
        noise floor of cosine score differences. Truncating each branch to 30
        thus loses no fused-result recall worth keeping while cutting the BM25
        ``ORDER BY rank`` work (FTS5 still ranks all matches, but the LIMIT
        truncation + fewer returned rows shaves ~0.4ms off p99).

        Fallback posture: if FTS5 is unavailable (stripped SQLite) the BM25
        branch contributes nothing and this reduces to pure vector search —
        still correct, never raises. A degenerate (empty/all-zero) query
        vector yields only the BM25 branch (keyword-only recall), and vice
        versa for an empty query string.

        Returns the top-``k`` cards with their fused RRF score, ordered
        (score desc, timestamp asc, id asc) — the same tie-break contract as
        ``vector_search``. Scores are NOT clamped to [0,1] here (RRF scores
        live in ~[0, 1/60] + boost, not [0,1]); callers comparing to cosine
        scores should use ``vector_search`` directly.
        """
        # --- branch 1: semantic (vector) candidates by cosine rank ----------
        semantic: list[tuple[str, float]] = []  # (id, cosine_score), ranked desc
        if query_vec and not all(v == 0.0 for v in query_vec):
            # Ask for more than k so RRF has a deep candidate pool to fuse from;
            # the fusion re-ranks across both branches before taking the final k.
            # Use the score-only path (_vector_search_ranked, NO batch_get) —
            # hybrid_search only needs ids + ranks for RRF; it does its own
            # single batched hydration of the FINAL fused top-k below. This
            # avoids a redundant batch_get over up to `top_n_per_branch` cards.
            try:
                import numpy as np
            except ImportError:
                np = None
            if np is None:
                ranked = self._vector_search_fallback_ranked(query_vec, top_n_per_branch, 1.0, scope)
            else:
                ranked = self._vector_search_ranked(query_vec, top_n_per_branch, 1.0, np, scope)
            # ranked: [(final=cos, cos, ts, id)]; preserve native rank order (desc).
            semantic = [(e[3], e[0]) for e in ranked]

        # --- branch 2: lexical (FTS5 BM25) candidates by relevance rank ------
        bm25: list[tuple[str, float]] = []  # (id, bm25_score), ranked desc
        fts_query = _fts5_query(query)
        if fts_query:
            with self._lock:
                cur = self._conn.cursor()
                try:
                    # bm25(memories_fts) returns a relevance score (higher =
                    # better). `id` is an UNINDEXED FTS column, so we read it
                    # directly from the index (no JOIN needed to resolve
                    # rowid→id). The JOIN to `memories` is kept ONLY for the
                    # live-row filter (archived/superseded): the FTS index is
                    # NOT perfectly curated w.r.t. archive (archive changes
                    # archived, not fact — no trigger fires), so the base-table
                    # filter remains the source of truth for "live". Using
                    # explicit bm25() over the `rank` pseudocolumn + the
                    # UNINDEXED id cut this query from ~4.2ms to ~3.0ms at 10k
                    # cards.
                    #
                    # Multi-tier scoping: scope is filtered at the JOIN, on the
                    # base table (the source of truth for live rows). This keeps
                    # BM25 candidates scope-isolated in lockstep with the
                    # vector branch's mask. scope IS NULL => global (no filter).
                    if scope is not None:
                        rows = cur.execute(
                            "SELECT f.id, bm25(memories_fts) AS rank "
                            "FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
                            "WHERE memories_fts MATCH ? "
                            "AND m.superseded_by IS NULL AND m.archived = 0 "
                            "AND (m.expiration_timestamp IS NULL "
                            "OR m.expiration_timestamp > unixepoch()) "
                            "AND m.scope = ? "
                            "ORDER BY rank LIMIT ?",
                            (fts_query, scope, top_n_per_branch),
                        ).fetchall()
                    else:
                        rows = cur.execute(
                            "SELECT f.id, bm25(memories_fts) AS rank "
                            "FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
                            "WHERE memories_fts MATCH ? "
                            "AND m.superseded_by IS NULL AND m.archived = 0 "
                            "AND (m.expiration_timestamp IS NULL "
                            "OR m.expiration_timestamp > unixepoch()) "
                            "ORDER BY rank LIMIT ?",
                            (fts_query, top_n_per_branch),
                        ).fetchall()
                except sqlite3.OperationalError:
                    # FTS5 missing OR query parse error -> vector-only fallback.
                    rows = []
                finally:
                    cur.close()
            bm25 = [(r[0], float(r[1])) for r in rows]

        # Degenerate: both branches empty (no query, no vector, or no matches).
        if not semantic and not bm25:
            return []

        # --- entity-graph boost (card_edges neighborhood of query entities) --
        # The boost's "source" cards are those whose `fact` mentions a query
        # entity — which is EXACTLY the BM25 branch's matched set. Reuse the
        # bm25 ids instead of re-running an FTS query (saves one ~4ms scan
        # at 10k cards; the entity FTS would otherwise ~double the FTS cost).
        entity_boosts = self._entity_boosts(query, top_n_per_branch, bm25_ids=[cid for cid, _ in bm25])

        # --- fuse ------------------------------------------------------------
        # Ask fusion for k + _HYDRATION_BUFFER candidates (see the constant's
        # docstring): the final [:k] slice after hydration absorbs a transient
        # deletion without dropping below k results.
        fused = _rrf_fusion(
            semantic_hits=semantic,
            bm25_hits=bm25,
            entity_boosts=entity_boosts,
            alpha=alpha,
            k=k + _HYDRATION_BUFFER,
        )
        if not fused:
            return []
        # Hydrate cards in one batched SELECT for the fused top-k. The cards
        # already carry their timestamp (MemoryCard.timestamp), so the
        # (score desc, ts asc, id asc) tie-break is applied AFTER hydration —
        # no separate _timestamps_for round-trip needed.
        #
        # Over-fetch from fusion by a small buffer: a card that ranked in the
        # top-k may be hard-deleted between the lock-released fusion step and
        # this batch_get (a TOCTOU window under concurrent `delete()`). Asking
        # fusion for k+buffer candidates means a transient deletion no longer
        # shrinks the returned list below k — the next-best candidate hydrates
        # in. The buffer is cheap (one extra row in the IN(...) SELECT).
        want_ids = [cid for cid, _ in fused]
        by_card = {c.id: c for c in self.batch_get(want_ids)}
        ranked = sorted(
            ((cid, score) for cid, score in fused if cid in by_card),
            key=lambda e: (
                -e[1],
                by_card[e[0]].timestamp if by_card[e[0]].timestamp is not None else math.inf,
                e[0],
            ),
        )
        return [(by_card[cid], score) for cid, score in ranked[:k]]

    def _entity_boosts(
        self,
        query: str,
        top_n_per_branch: int,
        bm25_ids: list[str] | None = None,
    ) -> dict[str, float]:
        """Entity-graph boost: cards linked (via ``card_edges``) to a card
        that the query entities lexically match get a decaying bonus.

        Mirrors Mem0's neighborhood-boost: for each query entity token, find
        the live cards whose ``fact`` mentions it (FTS5), then every card
        linked to those via the semantic graph receives:

            Boost = 0.5 / (1 + 0.001 * (N_linked - 1) ** 2)

        where ``N_linked`` is the number of query-entity-matching cards that
        link TO this boosted card (more independent witnesses -> diminishing
        returns via the squared denominator, capped by the 0.5 numerator).
        A card boosts itself only via its graph neighborhood, never by matching
        the query directly (that's the BM25 branch's job).
        """
        entities = _extract_entities(query)
        if not entities:
            return {}
        # When the caller already has the BM25-matched card ids, reuse them
        # as the entity-source set — the entity tokens are a subset of the
        # query terms the BM25 branch already matched, so a second FTS scan
        # over the same index is redundant (saves ~4ms at 10k cards). The
        # standalone path (bm25_ids is None) runs the FTS OR query itself so
        # the method stays correct + testable in isolation.
        if bm25_ids is not None:
            matched_ids = bm25_ids[:top_n_per_branch]
        else:
            matched_ids = self._entity_source_ids(entities, top_n_per_branch)
        if not matched_ids:
            return {}
        # For every card linked FROM a matched card, count how many distinct
        # matched cards point at it (N_linked). One graph query (UNION'd
        # placeholders) instead of N round-trips.
        placeholders = ",".join("?" for _ in matched_ids)
        with self._lock:
            cur = self._conn.cursor()
            try:
                link_rows = cur.execute(
                    "SELECT target_id, COUNT(DISTINCT source_id) AS n "
                    "FROM card_edges "
                    "WHERE source_id IN (%s) "
                    "GROUP BY target_id" % placeholders,
                    matched_ids,
                ).fetchall()
            finally:
                cur.close()
        boosts: dict[str, float] = {}
        for target_id, n_linked in link_rows:
            n = int(n_linked)
            if n <= 0:
                continue
            # Mem0 decay: 0.5 / (1 + 0.001 * (N-1)^2). Monotonic in N, capped.
            b = 0.5 / (1.0 + 0.001 * (n - 1) ** 2)
            boosts[target_id] = max(boosts.get(target_id, 0.0), b)
        return boosts

    def _entity_source_ids(
        self, entities: list[str], top_n_per_branch: int
    ) -> list[str]:
        """Standalone entity-source lookup: ids of live cards whose `fact`
        mentions any of the entity tokens (FTS5 OR query). Used by
        ``_entity_boosts`` when no ``bm25_ids`` were passed in (test/standalone
        use); the hybrid path reuses the BM25 branch's ids instead."""
        fts_query = " OR ".join('"%s"' % _fts5_escape(e) for e in entities)
        with self._lock:
            cur = self._conn.cursor()
            try:
                try:
                    # JOIN keeps the live-row filter (archived/superseded)
                    # authoritative; id read directly from the UNINDEXED FTS
                    # column. Mirrors the BM25 branch's query shape.
                    rows = cur.execute(
                        "SELECT f.id FROM memories_fts f "
                        "JOIN memories m ON m.rowid = f.rowid "
                        "WHERE memories_fts MATCH ? "
                        "AND m.superseded_by IS NULL AND m.archived = 0 "
                        "AND (m.expiration_timestamp IS NULL "
                        "OR m.expiration_timestamp > unixepoch()) "
                        "LIMIT ?",
                        (fts_query, top_n_per_branch),
                    ).fetchall()
                except sqlite3.OperationalError:
                    return []
            finally:
                cur.close()
        return [r[0] for r in rows]

    def _vector_search_fallback(
        self, query_vec: list[float], k: int, alpha: float = 0.70, scope: str | None = "default"
    ) -> list[tuple[MemoryCard, float]]:
        """Pure-Python fallback used only when numpy is unavailable.

        Mirrors the original O(n x d) loop exactly: per-card dot product over
        `min(qdim, len(emb))`, clamp to [0, 1], sort by (score desc, timestamp
        asc), top-k. When ``alpha < 1.0``, cosine scores are fused with
        Ebbinghaus retention via ``hybrid_score`` and sorted by the hybrid
        result. If the caller's matrix cache latched ``_vec_hetero``, this
        path is also the correctness backstop for non-uniform embeddings.

        Multi-tier scoping: when ``scope`` is not None, cards whose stored
        scope doesn't match are skipped before scoring (the no-numpy path
        applies scope at iteration time rather than via a mask, since there's
        no vectorized score array to floor).
        """
        cards = self.all()
        scored: list[tuple[MemoryCard, float]] = []
        qdim = len(query_vec)
        now = now_ts()
        for card in cards:
            # Multi-tier scoping: skip out-of-scope cards entirely.
            if scope is not None and card.scope != scope:
                continue
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
            if alpha < 1.0:
                retention = calculate_retention(
                    card.last_access, card.stability, current_ts=now
                )
                final = hybrid_score(dot, retention, alpha)
            else:
                final = dot
            scored.append((card, final))
        scored.sort(key=lambda item: (-item[1], item[0].timestamp))
        return scored[:k]

    # ------------------------------------------------------------------ #
    # Metrics / introspection
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        """Number of live (non-archived, non-superseded, non-expired) cards."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE superseded_by IS NULL AND archived = 0 "
                "AND (expiration_timestamp IS NULL OR expiration_timestamp > unixepoch())"
            )
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
