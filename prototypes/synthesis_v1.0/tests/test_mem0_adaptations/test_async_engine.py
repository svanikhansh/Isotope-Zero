"""Tests for the asyncio facade over the sync MemoryStore (isotope_zero.core.async_engine).

Mem0 port: mem0 inlines history+graph+scope+dedup+async into ONE file
mem0/memory/main.py, and its async methods offload blocking work to worker
threads via ``asyncio.to_thread`` throughout (mem0/memory/main.py:2256, :2259,
:2264). isotope_zero ports that idiom into ``AsyncMemoryEngine``: a thin
asyncio wrapper that subclasses the sync ``MemoryStore`` and only ADDS
``async`` coroutine methods (each a ``await asyncio.to_thread(...)`` delegation
to the inherited sync method). These tests pin that contract.

Conventions: plain ``test_*`` functions (matching the rest of the suite —
asyncio.run is called explicitly inside each, NOT pytest-asyncio markers),
``:memory:`` store, and a small ``_card``/``_norm`` helper.
"""
from __future__ import annotations

import asyncio
import math

from isotope_zero.core.async_engine import AsyncMemoryEngine
from isotope_zero.core.store import MemoryStore
from isotope_zero.types import MemoryCard, now_ts


def _norm(v: list[float]) -> list[float]:
    """Unit-normalize a vector so cosine ~ dot product."""
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else list(v)


def _card(id: str, fact: str = "A fact.", embedding: list[float] | None = None) -> MemoryCard:
    return MemoryCard(
        id=id,
        fact=fact,
        evidence="evidence",
        timestamp=now_ts(),
        tags=[],
        embedding=embedding,
        source_tokens=5,
    )


def test_async_add_and_search_roundtrip():
    """async add() then async search() returns the just-added card on top."""
    eng = AsyncMemoryEngine(":memory:")

    async def run():
        await eng.add(_card("c1", fact="hello", embedding=_norm([1.0, 0.0])))
        await eng.add(_card("c2", fact="world", embedding=_norm([0.0, 1.0])))
        res = await eng.search(_norm([1.0, 0.0]), k=1)
        assert res, "search must return results"
        assert res[0][0].id == "c1", res
        # Score is a float (cosine * decay fusion), tuple shape (card, score).
        assert isinstance(res[0][1], float)

    asyncio.run(run())


def test_batch_add_fifty_cards_all_visible():
    """batch_add(50) -> store.all() reports exactly 50 live cards."""
    eng = AsyncMemoryEngine(":memory:")

    async def run():
        items = [
            _card(f"b{i}", fact=f"batch item {i}", embedding=_norm([float(i), 1.0]))
            for i in range(50)
        ]
        n = await eng.batch_add(items)
        assert n == 50
        # Inherited sync all() (MemoryStore.all, store.py:1104) excludes
        # archived/superseded; 50 fresh cards all surface.
        assert len(eng.all()) == 50, len(eng.all())
        # Inherited sync count() agrees.
        assert eng.count() == 50

    asyncio.run(run())


def test_batch_search_returns_results_in_order():
    """batch_search over many queries returns a result list per query, same order."""
    eng = AsyncMemoryEngine(":memory:")

    async def run():
        await eng.add(_card("c1", fact="alpha", embedding=_norm([1.0, 0.0])))
        await eng.add(_card("c2", fact="beta", embedding=_norm([0.0, 1.0])))
        queries = [_norm([1.0, 0.0]), _norm([0.0, 1.0]), _norm([1.0, 1.0])]
        results = await eng.batch_search(queries, k=2)
        assert len(results) == 3, results
        for r in results:
            assert isinstance(r, list)
            assert r, "each query must return at least one hit"
        # First query is closest to c1, second closest to c2.
        assert results[0][0][0].id == "c1", results[0]
        assert results[1][0][0].id == "c2", results[1]

    asyncio.run(run())


def test_second_asyncio_run_works_loop_reusable():
    """A fresh asyncio.run after the first works — loops are reusable, not one-shot."""
    eng = AsyncMemoryEngine(":memory:")

    async def first():
        await eng.add(_card("c1", fact="first", embedding=_norm([1.0, 0.0])))
        res = await eng.search(_norm([1.0, 0.0]), k=1)
        assert res and res[0][0].id == "c1"

    async def second():
        # State persists across loops (the store holds the connection).
        await eng.add(_card("c2", fact="second", embedding=_norm([0.0, 1.0])))
        res = await eng.search(_norm([1.0, 0.0]), k=2)
        assert res and len(res) == 2
        assert {r[0].id for r in res} == {"c1", "c2"}
        assert eng.count() == 2

    asyncio.run(first())
    asyncio.run(second())


def test_engine_is_a_memory_store_sync_api_intact():
    """AsyncMemoryEngine subclasses MemoryStore; non-overridden sync methods work unchanged.

    ``add``/``search`` are overridden as ``async`` coroutines (must be
    awaited). The rest of the sync API (``count``/``get``/``vector_search``/
    ``archive_card``/``all``) is inherited verbatim and works synchronously
    with no await — that is the backward-compat contract.
    """
    eng = AsyncMemoryEngine(":memory:")
    # IS-A relationship.
    assert isinstance(eng, MemoryStore)

    # add() is async — go through asyncio.run, not the bare call.
    async def seed():
        await eng.add(_card("s1", fact="sync add", embedding=_norm([1.0, 0.0])))

    asyncio.run(seed())
    # Non-overridden sync methods work directly (no await).
    assert eng.count() == 1
    res = eng.vector_search(_norm([1.0, 0.0]), k=1)
    assert res and res[0][0].id == "s1"
    assert eng.get("s1") is not None


def test_engine_wraps_existing_store_shares_connection():
    """Constructing from an existing MemoryStore reuses its connection (no second DB)."""
    store = MemoryStore(":memory:")
    store.add(_card("pre", fact="pre-existing", embedding=_norm([1.0, 0.0])))
    eng = AsyncMemoryEngine(store)
    # The wrapped card is visible through the engine (same connection).
    assert eng.count() == 1
    assert eng.get("pre") is not None

    async def run():
        res = await eng.search(_norm([1.0, 0.0]), k=1)
        assert res and res[0][0].id == "pre"

    asyncio.run(run())


def test_batch_add_empty_returns_zero():
    """batch_add([]) is a clean no-op returning 0, no gather on empty."""
    eng = AsyncMemoryEngine(":memory:")

    async def run():
        n = await eng.batch_add([])
        assert n == 0
        assert eng.count() == 0

    asyncio.run(run())


def test_batch_search_empty_returns_empty_list():
    """batch_search([]) is a clean no-op returning []."""
    eng = AsyncMemoryEngine(":memory:")

    async def run():
        assert await eng.batch_search([], k=5) == []

    asyncio.run(run())


def test_search_scope_filter_isolates_cards():
    """async search(scope=...) only sees cards in that scope (store.py:1403 filter)."""
    eng = AsyncMemoryEngine(":memory:")

    async def run():
        # a1 lives in the default scope; a2 lives under a different scope.
        await eng.add(_card("a1", fact="alpha", embedding=_norm([1.0, 0.0])))
        await eng.add(
            _card("a2", fact="beta", embedding=_norm([1.0, 0.0])), scope="user_id=U1"
        )
        # Default scope search sees only a1.
        default_res = await eng.search(_norm([1.0, 0.0]), k=5, scope="default")
        ids = {r[0].id for r in default_res}
        assert ids == {"a1"}, ids
        # Scoped search sees only a2.
        scoped_res = await eng.search(_norm([1.0, 0.0]), k=5, scope="user_id=U1")
        scoped_ids = {r[0].id for r in scoped_res}
        assert scoped_ids == {"a2"}, scoped_ids

    asyncio.run(run())


def test_hybrid_search_fallback_path_runs():
    """async search(hybrid=True) uses the store's own hybrid_search fallback."""
    eng = AsyncMemoryEngine(":memory:")

    async def run():
        await eng.add(_card("h1", fact="the quick brown fox", embedding=_norm([1.0, 0.0])))
        # hybrid=True with a query string — falls back to MemoryStore.hybrid_search
        # (store.py:1641) since the standalone HybridSearcher module is absent.
        res = await eng.search(
            _norm([1.0, 0.0]), k=1, hybrid=True, query="quick fox"
        )
        assert res, "hybrid fallback must return results"
        assert res[0][0].id == "h1"

    asyncio.run(run())


def test_consolidate_delegates_to_sync_consolidate_memories():
    """async consolidate() delegates to MemoryStore.consolidate_memories via to_thread."""
    eng = AsyncMemoryEngine(":memory:")

    async def run():
        await eng.add(_card("d1", fact="dup one", embedding=_norm([1.0, 0.0])))
        await eng.add(_card("d2", fact="dup two", embedding=_norm([1.0, 0.01])))
        # Consolidate: hard-delete d2, keep d1 as survivor.
        deleted = await eng.consolidate(
            merged_cards=[_card("d1", fact="merged dup", embedding=_norm([1.0, 0.0]))],
            deleted_ids=["d2"],
        )
        assert deleted == 1, deleted
        # d1 still live, d2 gone.
        assert eng.count() == 1
        assert eng.get("d1") is not None

    asyncio.run(run())


def test_purge_expired_reports_archived_count():
    """async purge_expired() reports the count of archived rows without mutating data."""
    eng = AsyncMemoryEngine(":memory:")

    async def run():
        await eng.add(_card("p1", fact="live", embedding=_norm([1.0, 0.0])))
        # Nothing archived yet.
        n = await eng.purge_expired()
        assert n == 0
        # Add a second card then archive it. add() is async (await it);
        # archive_card() is an inherited sync method (no await needed).
        await eng.add(_card("p2", fact="to archive", embedding=_norm([0.0, 1.0])))
        eng.archive_card("p2")
        n2 = await eng.purge_expired()
        assert n2 == 1, n2

    asyncio.run(run())
