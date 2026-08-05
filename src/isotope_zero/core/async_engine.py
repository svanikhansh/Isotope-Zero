"""Asyncio facade over the sync ``MemoryStore`` (v1.0 synthesis).

The shipped v1.0 store is synchronous (SQLite WAL + NumPy BLAS, with an
optional out-of-process embedding *daemon* that uses threads, not asyncio).
Agents built on ``asyncio`` (FastAPI, AutoGen, async LangGraph) need an
awaitable surface; this module provides a thin one that delegates every call
to the sync ``MemoryStore`` via ``loop.run_in_executor(None, ...)``.

Design (delegation, not reimplementation):
    Every method spawns the corresponding sync call onto the default
    ``ThreadPoolExecutor`` and awaits it. We do NOT open a second SQLite
    connection or take a second lock: the store is already opened with
    ``check_same_thread=False`` and serializes all access through its own
    ``threading.Lock``, so a thread-pool worker safely borrows the single
    held connection. This preserves the store's lock/WAL/mmap guarantees
    verbatim — the async facade inherits the exact semantics of the sync
    layer, including its concurrent-write serialization.

    The default executor caps worker threads at ``min(32, cpu+4)``; because
    the store lock serializes DB work, more workers than that just queue on
    the lock and add no throughput. We therefore accept the default; agents
    needing write parallelism should shard across stores, not threads.

Limitations:
    - Embedding generation is also delegated (the engine's ``embed_text`` is
      sync and CPU/IO-bound). For high-throughput async pipelines, prefer the
      daemon path (out-of-process) so embedding doesn't block a thread-pool
      worker; the facade forwards ``spawn_daemon`` through to the engine.
    - No native ``async for`` streaming of large result sets; ``all()`` and
      ``batch_get`` materialize. Fine for the card counts a prototype targets.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .store import MemoryStore

log = logging.getLogger("isotope_zero.async_engine")


class AsyncMemoryEngine:
    """Awaitable facade over a ``MemoryStore``.

    Construct with the same args as ``MemoryStore`` (plus an embedder) OR by
    wrapping an existing store. Exposes the read/write/housekeeping verbs as
    coroutines that delegate to the sync store on a thread-pool worker::

        eng = AsyncMemoryEngine(db_path="mem.db", embedder=my_engine)
        await eng.add(card)
        hits = await eng.vector_search(qvec, k=5)

    Async-context-manager support (``async with``) closes the store on exit.
    """

    def __init__(self, store: MemoryStore | None = None, **store_kwargs: Any) -> None:
        if store is not None:
            self._store = store
            self._owns_store = False  # caller manages lifetime
        else:
            # Mirror MemoryStore's constructor surface. The store itself does
            # NOT take an embedder at construction (the client wires it); but
            # the unified client does, so we accept and forward it.
            embedder = store_kwargs.pop("embedder", None)
            self._store = MemoryStore(embedder=embedder, **store_kwargs)
            self._owns_store = True

    # ------------------------------------------------------------------ #
    # Executor plumbing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _run(coro_callable, *args):
        """Spawn a sync callable onto the default executor and return a future.

        ``coro_callable`` is a zero-arg lambda building the sync call; we
        schedule it on the shared default ThreadPoolExecutor so the store's
        single held connection is accessed from a worker thread (safe under
        ``check_same_thread=False`` + the store's own ``_lock``).
        """
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, coro_callable, *args)

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #
    async def add(self, card, scope: str | None = None) -> None:
        await self._run(lambda: self._store.add(card, scope=scope))

    async def update(self, card) -> None:
        await self._run(self._store.update, card)

    async def delete(self, memory_id: str) -> bool:
        return await self._run(self._store.delete, memory_id)

    async def archive_card(self, card_id: str) -> bool:
        return await self._run(self._store.archive_card, card_id)

    async def touch(self, memory_id: str, at: float | None = None) -> None:
        await self._run(self._store.touch, memory_id, at)

    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #
    async def get(self, memory_id: str):
        return await self._run(self._store.get, memory_id)

    async def all(self):
        return await self._run(self._store.all)

    async def count(self) -> int:
        return await self._run(self._store.count)

    async def batch_get(self, ids: list[str]):
        return await self._run(self._store.batch_get, ids)

    async def vector_search(self, query_vec, k: int = 5, alpha: float = 0.70, scope: str = "default"):
        return await self._run(self._store.vector_search, query_vec, k, alpha, scope)

    async def hybrid_search(
        self,
        query: str,
        query_vec,
        k: int = 5,
        alpha: float = 0.70,
        top_n_per_branch: int = 30,
        scope: str = "default",
    ):
        return await self._run(
            self._store.hybrid_search, query, query_vec, k, alpha, top_n_per_branch, scope
        )

    # ------------------------------------------------------------------ #
    # Housekeeping
    # ------------------------------------------------------------------ #
    async def consolidate_memories(self, **kwargs):
        return await self._run(lambda: self._store.consolidate_memories(**kwargs))

    async def close(self) -> None:
        if self._owns_store:
            await self._run(self._store.close)

    # ------------------------------------------------------------------ #
    # Async context manager
    # ------------------------------------------------------------------ #
    async def __aenter__(self) -> "AsyncMemoryEngine":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @property
    def sync_store(self) -> MemoryStore:
        """Escape hatch to the underlying sync store for advanced callers."""
        return self._store
