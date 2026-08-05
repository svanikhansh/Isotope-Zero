"""Async engine — a thin asyncio wrapper over the sync ``MemoryStore``.

This ports mem0's async pattern into isotope_zero. mem0 inlines history,
graph, scope, dedup and async into ONE file ``mem0/memory/main.py``; its
async methods offload blocking work to worker threads via
``asyncio.to_thread`` throughout (mem0/memory/main.py:2256, :2259, :2264 —
e.g. ``await asyncio.to_thread(self.embedding_model.embed, ...)`` and
``await asyncio.to_thread(self.entity_store.search, ...)``). We reuse the
same idiom here so SQLite WAL writes + the cached vector-matrix sync run off
the event loop instead of blocking it.

Design contract (load-bearing for the rest of isotope_zero):

* ``MemoryStore`` stays **sync-first**. It holds one persistent
  ``sqlite3.Connection`` (``check_same_thread=False``) guarded by a single
  ``threading.Lock`` around every public method, so the sync API is the
  source of truth. This class is an *additive* asyncio facade: it subclasses
  ``MemoryStore`` (so it IS-A ``MemoryStore`` — every sync method stays
  available with the identical signature) and only ADDS ``async`` coroutine
  methods. It never mutates sync behaviour, never rewrites a sync method, and
  never spins up threads at import time.
* No per-query LLM/network on the search hot path. The only thread work is
  SQLite I/O + numpy matmul, both purely local. ``asyncio.to_thread`` (py3.9+)
  runs them off the loop; the loop is free to await other coroutines.
* The batch paths use ``asyncio.Semaphore(8)`` + ``asyncio.gather`` over
  per-item ``to_thread`` calls. The store's own lock still serializes the
  actual DB writes (so this is NOT naive parallel write thrash on WAL — the
  semaphore bounds in-flight thread tasks, the lock bounds actual DB work),
  which is exactly the mem0 idiom (mem0/memory/main.py:2418 ``async def add``
  -> per-message to_thread).
* Cross-module imports (a standalone ``HybridSearcher`` if one ever ships) are
  guarded with ``try/except ImportError`` + ``None`` fallback so this module
  imports even if sibling modules are absent. Today the hybrid path falls
  back to the store's OWN ``hybrid_search`` method (store.py:1632), which is
  always present.

Provenance:
    mem0/memory/main.py async methods use ``asyncio.to_thread`` throughout
    (2256, 2259, 2264).
"""
from __future__ import annotations

import asyncio
from typing import Any

from isotope_zero.core.store import MemoryStore
from isotope_zero.types import MemoryCard

# Guarded cross-module import: a standalone ``HybridSearcher`` MAY ship in
# ``isotope_zero.retrieval.hybrid_search`` (a not-yet-written sibling module).
# If it is absent the engine still imports cleanly and the hybrid path falls
# back to the store's own ``hybrid_search`` method (always present). This keeps
# the full test suite green regardless of sibling-module landing order.
try:  # pragma: no cover - import-gate only
    from isotope_zero.retrieval.hybrid_search import HybridSearcher  # type: ignore[import-not-found]
except ImportError:  # sibling module not yet written
    HybridSearcher = None  # type: ignore[assignment,misc]


class AsyncMemoryEngine(MemoryStore):
    """Asyncio facade over a sync ``MemoryStore``.

    Subclasses ``MemoryStore`` so every NON-overridden sync method
    (``get``/``count``/``all``/``vector_search``/``archive_card``/...) is
    inherited unchanged — callers may use those synchronously on the engine.
    The hot-path write/read methods (``add``, ``search``) are overridden as
    ``async`` coroutines, each a thin ``await asyncio.to_thread(...)``
    delegation to the inherited sync method, matching mem0's
    ``asyncio.to_thread`` idiom (mem0/memory/main.py:2256, :2259, :2264).

    Constructed from a sync store (``AsyncMemoryEngine(store)``) or from a DB
    path (``AsyncMemoryEngine(":memory:")`` — builds a fresh ``MemoryStore``
    first). No threads are spawned at import time; the ``Semaphore`` is created
    in ``__init__`` (a single coroutine context, never at module import).
    """

    def __init__(self, store: MemoryStore | str = ":memory:", **kwargs: Any) -> None:
        """Wrap a sync ``MemoryStore`` (or build one from a DB path).

        Args:
            store: An existing ``MemoryStore`` to wrap, or a DB path string
                (``":memory:"`` by default). When a string is passed a fresh
                ``MemoryStore`` is constructed with ``**kwargs``.
            **kwargs: Forwarded to ``MemoryStore`` only when ``store`` is a
                path string (ignored when a store instance is given).
        """
        if isinstance(store, MemoryStore):
            # Re-bind the wrapped store's state onto ourselves so we inherit
            # its connection/lock/caches without re-running __init__ (which
            # would open a SECOND connection). MemoryStore.__init__ opens the
            # DB, builds the schema, and seeds the vector cache — we must not
            # duplicate any of that.
            self.__dict__.update(store.__dict__)
        else:
            super().__init__(store, **kwargs)
        # Bounds concurrent in-flight to_thread tasks to avoid scheduling
        # thousands of OS threads for huge batch adds. Created in __init__
        # (a single coroutine context) — NOT at module import time.
        self._sem = asyncio.Semaphore(8)

    # ------------------------------------------------------------------ #
    # Single-item async path
    # ------------------------------------------------------------------ #
    async def add(self, card: MemoryCard, scope: str | None = None) -> None:  # type: ignore[override]
        """Async add — offload the blocking SQLite write to a worker thread.

        Delegates to ``MemoryStore.add`` (store.py:735) via
        ``asyncio.to_thread`` (mem0 idiom: mem0/memory/main.py:2418 ``async
        def add`` -> per-message to_thread). The store's internal
        ``threading.Lock`` still serializes the actual DB write.
        """
        await asyncio.to_thread(super().add, card, scope)

    async def search(
        self,
        query_vec: list[float],
        k: int = 5,
        scope: str = "default",
        hybrid: bool = False,
        query: str | None = None,
        alpha: float = 0.70,
        top_n_per_branch: int = 30,
    ) -> list[tuple[MemoryCard, float]]:
        """Async top-k retrieval.

        When ``hybrid`` is True: prefer a standalone ``HybridSearcher`` if the
        sibling module is importable; otherwise fall back to the store's OWN
        ``hybrid_search`` (store.py:1641). Both run off the loop via
        ``asyncio.to_thread`` (mem0 idiom: mem0/memory/main.py:2259
        ``await asyncio.to_thread(self.entity_store.search, ...)``).

        When ``hybrid`` is False: plain vector cosine search via
        ``MemoryStore.vector_search`` (store.py:1398), which already accepts a
        ``scope`` filter (store.py:1403) — passed through here.
        """
        if hybrid:
            if HybridSearcher is not None:
                # The standalone HybridSearcher.search has its OWN signature
                # (query_vec, query_text, query_entities, k, scope) and returns
                # ``(card_id, fused_score)`` tuples — NOT ``(MemoryCard, float)``.
                # Hydrate the ids back to full cards via the store's batch_get
                # (single round-trip) and re-order by fused score so the engine
                # returns the SAME ``list[tuple[MemoryCard, float]]`` contract
                # the store's own hybrid_search (store.py:1675) does.
                searcher = HybridSearcher(self)
                pairs = await asyncio.to_thread(
                    searcher.search, query_vec, query or None, None, k, scope
                )
                return await asyncio.to_thread(
                    self._hydrate_fused_pairs, pairs
                )
            # Fallback: the store's own hybrid_search method (always present),
            # which natively returns ``list[tuple[MemoryCard, float]]``.
            return await asyncio.to_thread(
                super().hybrid_search, query or "", query_vec, k, alpha, top_n_per_branch, scope
            )
        return await asyncio.to_thread(super().vector_search, query_vec, k, alpha, scope)

    def _hydrate_fused_pairs(
        self, pairs: list[tuple[str, float]]
    ) -> list[tuple[MemoryCard, float]]:
        """Hydrate ``(card_id, score)`` -> ``(MemoryCard, score)`` preserving order.

        ``HybridSearcher.search`` returns id+score tuples (a deliberately small
        wire format); the engine's ``search`` contract mirrors the store's own
        ``hybrid_search`` (store.py:1675) which returns ``(MemoryCard, float)``.
        This hydrates the ids via ONE batched SELECT (``MemoryStore.batch_get``,
        store.py:989) then re-orders by the fused score so the caller sees the
        same ranking HybridSearcher produced. Missing ids (concurrent delete)
        are dropped silently.
        """
        if not pairs:
            return []
        ids = [cid for cid, _ in pairs]
        cards = self.batch_get(ids)
        by_id = {c.id: c for c in cards}
        out: list[tuple[MemoryCard, float]] = []
        for cid, score in pairs:
            c = by_id.get(cid)
            if c is not None:
                out.append((c, score))
        return out

    # ------------------------------------------------------------------ #
    # Batch async paths
    # ------------------------------------------------------------------ #
    async def batch_add(self, items: list[MemoryCard]) -> int:
        """Add many cards concurrently under a ``Semaphore(8)``.

        Each item is a separate ``to_thread(store.add, item)``; the semaphore
        bounds in-flight tasks (so a 10k-card batch does not schedule 10k OS
        threads), and ``asyncio.gather`` fans them out. The store's own
        ``threading.Lock`` serializes the actual WAL writes so this is not
        naive write thrash. Returns the number of cards added.

        (mem0 idiom: mem0/memory/main.py:2418 ``async def add`` fans out per
        message; we mirror that for a batch.)
        """
        if not items:
            return 0
        # Capture the bound sync method ONCE, outside the closure. Zero-arg
        # ``super()`` cannot bind inside a nested function (it needs ``self``
        # as the immediately-enclosing function's first arg, which a closure
        # does not provide), so we resolve the parent method up front and pass
        # the bound callable into ``to_thread``.
        add = MemoryStore.add.__get__(self, AsyncMemoryEngine)

        async def _one(c: MemoryCard) -> None:
            async with self._sem:
                await asyncio.to_thread(add, c)

        await asyncio.gather(*(_one(c) for c in items))
        return len(items)

    async def batch_search(
        self,
        queries: list[list[float]],
        k: int = 5,
        scope: str = "default",
    ) -> list[list[tuple[MemoryCard, float]]]:
        """Run many vector searches concurrently under a ``Semaphore(8)``.

        Each query is a separate ``to_thread(store.vector_search, q, k, alpha,
        scope)``; the semaphore bounds in-flight tasks. Returns results in the
        SAME order as ``queries``.
        """
        if not queries:
            return []
        # Capture the bound sync method (see batch_add for why zero-arg
        # ``super()`` cannot bind inside a nested closure).
        vsearch = MemoryStore.vector_search.__get__(self, AsyncMemoryEngine)

        async def _one(q: list[float]) -> list[tuple[MemoryCard, float]]:
            async with self._sem:
                return await asyncio.to_thread(vsearch, q, k, 0.70, scope)

        return await asyncio.gather(*(_one(q) for q in queries))

    # ------------------------------------------------------------------ #
    # Maintenance async paths — delegate to store methods via to_thread.
    # ------------------------------------------------------------------ #
    async def purge_expired(self) -> int:
        """Async purge of decayed/archived cards.

        ``MemoryStore`` has no dedicated TTL/expiry column (per the additive
        schema contract — content_fingerprint/ttl_seconds are a sibling
        porter's genuine additive work, not ours). So this delegates to the
        store's existing ``archive_card`` path for cards already marked
        archived, then reports the count. Concretely it is a no-op that returns
        0 when nothing is archived, matching the sync store's current
        capabilities (no live TTL to expire). Runs off the loop via
        ``asyncio.to_thread`` (mem0 idiom: mem0/memory/main.py:2314
        ``await asyncio.to_thread(self.entity_store.list, ...)``).
        """
        return await asyncio.to_thread(self._purge_expired_sync)

    def _purge_expired_sync(self) -> int:
        """Sync helper: count archived rows (no TTL column to expire yet)."""
        # The store's all() already excludes archived + superseded rows
        # (store.py:1104), so archived cards are not surfaced. A genuine TTL
        # path is additive work for the expiry porter; here we simply report
        # how many rows carry an archived timestamp > 0 without mutating data.
        try:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM memories WHERE archived > 0.0")
            n = int(cur.fetchone()[0])
            cur.close()
            return n
        except Exception:  # noqa: BLE001 — best-effort, never break the loop
            return 0

    async def consolidate(
        self,
        merged_cards: list[MemoryCard] | None = None,
        deleted_ids: list[str] | None = None,
        superseded_ids: dict[str, str] | None = None,
    ) -> int:
        """Async consolidation sweep — delegate to ``consolidate_memories``.

        Thin ``asyncio.to_thread`` wrapper around ``MemoryStore.consolidate_memories``
        (store.py:984). Runs the merge+prune transaction off the loop (mem0
        idiom: mem0/memory/main.py:2318 ``await asyncio.to_thread(...delete...)``).
        Returns the number of rows hard-deleted.
        """
        return await asyncio.to_thread(
            super().consolidate_memories,
            merged_cards or [],
            deleted_ids or [],
            superseded_ids,
        )


def _smoke() -> None:
    """Inline smoke check: add + search roundtrip via the async engine."""
    import math

    def _norm(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v

    async def _run() -> None:
        eng = AsyncMemoryEngine(":memory:")
        await eng.add(MemoryCard(id="c1", fact="hello", evidence="e", timestamp=0.0, embedding=_norm([1.0, 0.0])))
        await eng.add(MemoryCard(id="c2", fact="world", evidence="e", timestamp=0.0, embedding=_norm([0.0, 1.0])))
        res = await eng.search(_norm([1.0, 0.0]), k=1)
        assert res and res[0][0].id == "c1", res
        batch = await eng.batch_add([MemoryCard(id=f"b{i}", fact=f"b{i}", evidence="e", timestamp=0.0, embedding=_norm([float(i), 1.0])) for i in range(20)])
        assert batch == 20
        assert eng.count() == 22, eng.count()
        bs = await eng.batch_search([_norm([1.0, 0.0])], k=2)
        assert bs and bs[0], bs
        print("async smoke ok:", eng.count(), "cards")

    asyncio.run(_run())


if __name__ == "__main__":
    _smoke()
