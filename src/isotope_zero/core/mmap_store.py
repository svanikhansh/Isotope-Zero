"""Two-tier file-backed mmap vector store for Isotope Zero (Method 3).

Replaces the heap-resident ``(n, dim)`` float32 matrix with a file-backed
``np.memmap`` written to ``embeddings.bin`` next to the DB. The OS virtual
memory manager pages cold vectors in/out on demand; a small Hot LRU cache
(capacity 200 by default) keeps recently-accessed rows resident in heap
so the working set stays fast without pinning the whole matrix.

Layout of ``embeddings.bin``:
    contiguous float32, row-major, shape ``(n, dim)``. Row ``i`` holds the
    i-th non-superseded, non-NULL embedding in canonical (timestamp, id)
    order — the SAME order the heap path stacks via ``np.stack``. This makes
    the memmap a drop-in replacement for the heap matrix: ``matrix @ q`` and
    ``matrix[:, :n]`` work identically on the memmap view, and numpy reads
    pages on demand from the backing file (cold pages faulted in lazily).

Sidecar ``embeddings.bin.idx``: a JSON map ``{card_id: row_index}`` rebuilt
on every ``build()`` so a single ``SELECT id, embedding`` sweep suffices to
reconstruct both the file and the index. The DB (``memories`` table) remains
the single source of truth for card content; the memmap is a derived cache.

Honest memory model (see README "caveat"):
    The float32 matrix at 10k cards is only ~14.65 MB. The ~394 MB process
    RSS is dominated by the ONNX runtime (~360 MB), NOT the matrix. So mmap
    CANNOT reduce total RSS to 30 MB — at best it moves ~15 MB of matrix
    out of the heap, and a full vector scan pages the whole matrix back in
    (neutralizing the saving on the hot path). What we measure honestly:

      * heap matrix bytes (``np.stack`` path) vs mmap FILE bytes (the
        ceiling on resident pages; true resident is between 0 and file-size
        depending on access).
      * the Hot LRU resident heap bytes (``len(LRU) * dim * 4`` — at most
        200 * 384 * 4 = 307200 bytes = 0.3 MB, well under 30 MB).
      * cold page-fault latency (first scan after pages are evicted) vs
        warm (pages already resident / hot-cached).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from typing import Any

log = logging.getLogger("isotope_zero.mmap_store")


class MmapVectorStore:
    """File-backed float32 vector matrix + Hot LRU row cache.

    One of these is owned by ``MemoryStore`` when ``use_mmap=True``. The
    store's ``_ensure_vec_cache`` builds (or rebuilds) this object on demand
    and returns its ``matrix()`` view; ``vector_search`` runs ``matrix @ q``
    across the memmap. All mutators on ``MemoryStore`` that invalidate the
    heap cache (``_mark_vec_dirty``) also call this object's ``invalidate``.

    Thread-safety: every public method acquires ``self._lock``. The owning
    ``MemoryStore`` already serializes its write/scan path under its own
    lock, but this object is also safe to call directly (the LRU pop/get and
    the memmap rebuild are not re-entrant-safe otherwise).
    """

    def __init__(
        self,
        db_path: str,
        conn: Any,
        dim: int = 384,
        hot_capacity: int = 200,
        cache_dir: str | None = None,
    ) -> None:
        """Open/create the mmap backing files next to the DB.

        Args:
            db_path: the SQLite DB path the owning MemoryStore holds. Used
                ONLY to locate a writable directory for the sidecar files.
                ``:memory:`` is handled by writing the sidecars into a
                managed ``cache_dir`` (default ``.isotope_zero_cache/``).
            conn: the SQLite connection to read rows from during ``build``.
                Borrowed (not owned); caller owns its lifetime.
            dim: embedding dimension (must be uniform across rows; the
                store falls back to the heap path for hetero-dim data).
            hot_capacity: max rows kept resident in the Hot LRU cache.
            cache_dir: directory for the sidecar files. Defaults to a
                ``.isotope_zero_cache`` dir beside the DB (or the cwd for
                ``:memory:``). The repo-root .gitignore already ignores
                ``.isotope_zero_cache/`` and ``*.db``; the sidecar files
                (``embeddings.bin``, ``embeddings.bin.idx``) are created
                here, never checked in.
        """
        self._conn = conn
        self._dim = int(dim)
        self._hot_capacity = int(hot_capacity)
        self._lock = threading.Lock()
        self._dirty = True

        # Resolve a writable directory for the sidecar files.
        if cache_dir is not None:
            self._cache_dir = cache_dir
        elif db_path == ":memory:" or db_path == "":
            self._cache_dir = ".isotope_zero_cache"
        else:
            self._cache_dir = os.path.join(os.path.dirname(os.path.abspath(db_path)) or ".", ".isotope_zero_cache")
        os.makedirs(self._cache_dir, exist_ok=True)

        self._bin_path = os.path.join(self._cache_dir, "embeddings.bin")
        self._idx_path = os.path.join(self._cache_dir, "embeddings.bin.idx")

        # Lazy state — populated by build().
        self._memmap = None         # np.memmap (n, dim) float32 mode 'r+'
        self._ids: list[str] = []   # row_index -> card_id (parallel to rows)
        self._ts: list[float] = []  # row_index -> timestamp (parallel)
        self._scopes: list[str] = []  # row_index -> scope string (parallel)
        self._id_to_row: dict[str, int] = {}  # card_id -> row_index
        self._n: int = 0

        # Hot LRU: card_id -> np.float32 row (1-D, length dim). Resident heap
        # cost is len(LRU)*dim*4 bytes (<= hot_capacity*dim*4). OrderedDict
        # gives O(1) move-to-end + popitem(last=False) eviction.
        self._hot: "OrderedDict[str, Any]" = OrderedDict()

    # ------------------------------------------------------------------ #
    # Build / rebuild
    # ------------------------------------------------------------------ #
    def build(self, np: Any) -> Any:
        """(Re)build the bin file + index + memmap from the DB.

        SELECTs (id, timestamp, embedding) for non-superseded rows with a
        non-NULL embedding in canonical (timestamp, id) order — the SAME
        order the heap path stacks — so the memmap is row-identical to the
        heap matrix and ``matrix @ q`` produces identical scores. Writes
        rows contiguously to ``embeddings.bin`` via ``np.tofile`` (one
        ``np.stack`` + ``tofile`` for the whole set), persists the
        ``id -> row_index`` sidecar JSON, then opens the memmap read/write.

        Returns the ``np.memmap`` view (mode ``'r+'`` so a future ``flush``
        can write back, though search never mutates it). Caller MUST hold
        the owning store's lock around the SELECT + build so concurrent
        writers cannot interleave (the store clears its dirty flag at
        fetch-completion time, matching the heap path's contract).
        """
        cur = self._conn.cursor()
        try:
            cur.execute(
                # Match MemoryStore._ensure_vec_cache EXACTLY: NOT INDEXED
                # sequential scan (we read every embedding BLOB anyway),
                # superseded_by IS NULL (audit-trail folded rows excluded),
                # embedding IS NOT NULL (skip NULL-embedding cards),
                # ORDER BY timestamp ASC, id ASC (canonical). scope is fetched
                # so the owning store can mask out-of-scope rows in vector
                # search without a second round-trip (multi-tier scoping).
                "SELECT id, timestamp, embedding, scope FROM memories NOT INDEXED "
                "WHERE embedding IS NOT NULL AND superseded_by IS NULL "
                "ORDER BY timestamp ASC, id ASC"
            )
            rows = cur.fetchall()
        finally:
            cur.close()

        if not rows:
            # Empty matrix: write a zero-byte file so os.path.getsize == 0
            # and the memmap opens as shape (0, dim). Drop any stale index.
            self._memmap = None
            self._ids = []
            self._ts = []
            self._scopes = []
            self._id_to_row = {}
            self._n = 0
            self._hot.clear()
            try:
                os.remove(self._bin_path)
            except OSError:
                pass
            try:
                os.remove(self._idx_path)
            except OSError:
                pass
            self._dirty = False
            return None

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
                # Heterogeneous dims: signal failure to the caller, which
                # falls back to the heap/Python path. We do NOT latch here
                # — the owning store latches _vec_hetero on the same check.
                raise _HeteroDim(dim, a.shape[0])
            ids.append(rid)
            ts.append(float(rts) if rts is not None else 0.0)
            scopes.append(scope_val if scope_val is not None else "default")
            arrs.append(a)

        matrix = np.stack(arrs)  # (n, dim) float32, zero-copy views stacked
        # Write the contiguous float32 file. tofile writes row-major C order
        # by default, which is exactly the (n, dim) layout the memmap reads.
        matrix.tofile(self._bin_path)

        self._ids = ids
        self._ts = ts
        self._scopes = scopes
        self._id_to_row = {cid: i for i, cid in enumerate(ids)}
        self._n = len(ids)
        self._hot.clear()

        # Persist the id->row index sidecar (rebuild-from-scratch contract).
        with open(self._idx_path, "w") as f:
            json.dump(self._id_to_row, f)

        # Open the memmap read/write. mode='r+' requires the file to exist
        # (it does — we just wrote it). shape=(n, dim) so @ and slicing match
        # the heap matrix exactly.
        self._memmap = np.memmap(
            self._bin_path, dtype=np.float32, mode="r+",
            shape=(self._n, self._dim if dim is None else dim),
        )
        # Keep self._dim in sync with the actual data dim (in case the
        # caller's configured dim didn't match the embeddings).
        self._dim = int(self._memmap.shape[1])

        self._dirty = False
        return self._memmap

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #
    def matrix(self) -> Any:
        """Return the np.memmap view (or None if not built / empty).

        Usable directly as ``matrix @ q`` and ``matrix[:, :n]`` — numpy
        reads pages on demand from the backing file, so a cold scan pages
        in the whole matrix lazily (the resident-set grows toward the file
        size as pages are touched).
        """
        return self._memmap

    @property
    def ids(self) -> list[str]:
        """Row_index -> card_id (parallel to matrix rows)."""
        return self._ids

    @property
    def ts(self) -> list[float]:
        """Row_index -> timestamp (parallel to matrix rows)."""
        return self._ts

    @property
    def scopes(self) -> list[str]:
        """Row_index -> scope string (parallel to matrix rows).

        Multi-tier scoping: the owning store copies this into ``_vec_scopes``
        so ``vector_search(scope=...)`` can mask out-of-scope rows in C.
        """
        return self._scopes

    @property
    def n(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._dim

    # ------------------------------------------------------------------ #
    # Hot LRU cache
    # ------------------------------------------------------------------ #
    def hot_get(self, card_id: str, np: Any) -> Any | None:
        """Return the row for ``card_id``: LRU hit (move-to-end) or memmap
        fetch + insert (evicting the least-recently-used if over capacity).

        Returns a 1-D float32 array of length ``dim`` (a COPY of the memmap
        row, so the hot cache never aliases the mmap pages — the pages can
        be evicted without invalidating the cached copy). Returns ``None``
        when the card is not in the matrix (no row index / not built).
        """
        with self._lock:
            row = self._id_to_row.get(card_id)
            if row is None or self._memmap is None:
                return None
            cached = self._hot.get(card_id)
            if cached is not None:
                # Move-to-end on access (LRU).
                self._hot.move_to_end(card_id)
                return cached
            # Miss: fetch the row from the memmap (this page-faults the row
            # in if cold). Copy so the cached entry survives page eviction.
            vec = np.asarray(self._memmap[row], dtype=np.float32).copy()
            self._hot[card_id] = vec
            if len(self._hot) > self._hot_capacity:
                # Evict least-recently-used (the first item).
                self._hot.popitem(last=False)
            return vec

    def hot_warm(self, card_ids: list[str], np: Any) -> None:
        """Ensure the given card_ids are resident in the hot cache.

        Called by ``vector_search`` after the top-k are computed so the
        NEXT query for the same/nearby vectors hits the hot cache instead
        of re-faulting the memmap rows. No-op for ids not in the matrix.
        """
        with self._lock:
            for cid in card_ids:
                if cid in self._hot:
                    continue  # already resident
                row = self._id_to_row.get(cid)
                if row is None or self._memmap is None:
                    continue
                vec = np.asarray(self._memmap[row], dtype=np.float32).copy()
                self._hot[cid] = vec
                if len(self._hot) > self._hot_capacity:
                    self._hot.popitem(last=False)

    # ------------------------------------------------------------------ #
    # Invalidation / lifecycle
    # ------------------------------------------------------------------ #
    def invalidate(self) -> None:
        """Mark dirty; the next ``build()`` rebuilds the file + index + memmap.

        Called by ``MemoryStore._mark_vec_dirty`` on every add/update/delete/
        consolidate (and on direct-connection bulk writes via the adversarial
        harness). The hot LRU is also cleared so stale rows never surface.
        """
        with self._lock:
            self._dirty = True
            self._hot.clear()

    @property
    def dirty(self) -> bool:
        return self._dirty

    def flush(self) -> None:
        """Flush any pending memmap writes to disk (search never writes, but
        ``np.memmap`` documents flush() as the way to persist)."""
        with self._lock:
            if self._memmap is not None:
                try:
                    self._memmap.flush()
                except (ValueError, OSError):
                    pass

    def close(self) -> None:
        """Flush + close the memmap. Safe to call once; no-op thereafter."""
        with self._lock:
            if self._memmap is not None:
                try:
                    self._memmap.flush()
                except (ValueError, OSError):
                    pass
                # Deleting the memmap object releases the mapping. numpy
                # memmaps close the underlying file descriptor on GC.
                del self._memmap
                self._memmap = None
            self._hot.clear()

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Honest memory accounting
    # ------------------------------------------------------------------ #
    def resident_matrix_bytes(self) -> dict[str, int]:
        """Honest memory accounting for the matrix tier.

        macOS has no clean per-mapping resident-pages API exposed to Python,
        so we report the bounds honestly rather than fabricate a single
        number:

          * ``file_bytes``: ``os.path.getsize(embeddings.bin)`` — the
            CEILING on resident mmap pages (true resident is between 0 and
            this depending on how many pages have been faulted in).
          * ``hot_lru_bytes``: ``len(LRU) * dim * 4`` — the resident heap
            cost of the hot cache (at most ``hot_capacity * dim * 4``).
          * ``hot_capacity_bytes``: ``hot_capacity * dim * 4`` — the hard
            cap on the hot cache's heap cost.
          * ``n_rows``: number of rows in the matrix (for sanity).

        The heap-path comparison number (``np.stack(arrs).nbytes``) is
        reported by the owning ``MemoryStore`` when ``use_mmap=False``; it
        equals ``file_bytes`` for the same data (both are ``n*dim*4``).
        """
        try:
            file_bytes = os.path.getsize(self._bin_path) if os.path.exists(self._bin_path) else 0
        except OSError:
            file_bytes = 0
        hot_lru_bytes = len(self._hot) * self._dim * 4
        return {
            "file_bytes": file_bytes,
            "hot_lru_bytes": hot_lru_bytes,
            "hot_capacity_bytes": self._hot_capacity * self._dim * 4,
            "n_rows": self._n,
            "dim": self._dim,
        }

    def evict_pages_cold(self) -> None:
        """Best-effort: evict resident mmap pages so the next scan is cold.

        On macOS there is no ``posix_fadvise(POSIX_FADV_DONTNEED)``. The
        honest way to force a cold state is to drop the memmap mapping
        (which releases the pages) and reopen it: the next access faults
        the pages back in from the file. We do exactly that.
        """
        with self._lock:
            if self._memmap is None or self._n == 0:
                return
            shape = self._memmap.shape
            # Close + reopen: dropping the mapping lets the OS reclaim the
            # pages; the new memmap starts with no resident pages.
            try:
                self._memmap.flush()
            except (ValueError, OSError):
                pass
            del self._memmap
            # Re-open in read-only mode for the cold read (we never write to
            # the matrix during search). mode='r' still supports @ and slicing.
            import numpy as _np  # local; only needed for the cold-path probe
            self._memmap = _np.memmap(
                self._bin_path, dtype=_np.float32, mode="r", shape=shape,
            )


class _HeteroDim(Exception):
    """Internal signal: rows have heterogeneous embedding lengths.

    Raised by ``build()`` so the owning ``MemoryStore`` can latch
    ``_vec_hetero`` and fall back to the heap/Python path. NOT raised to
    user code — the store catches it in ``_ensure_vec_cache``.
    """

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(f"heterogeneous embedding dim: expected {expected}, got {got}")
        self.expected = expected
        self.got = got
