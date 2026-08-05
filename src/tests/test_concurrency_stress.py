"""Concurrency stress tests for isotope_zero's store + router + consolidator.

Module invariant contract being proven
--------------------------------------
`MemoryStore` holds ONE persistent `sqlite3.Connection` (opened with
`check_same_thread=False`, `isolation_level=None` autocommit) and guards
EVERY public method with a single `threading.Lock`. Because all DB access is
serialized on that one lock, the classic SQLite "database is locked" /
"database table is locked" `sqlite3.OperationalError` is structurally
impossible under correct usage — there is never more than one writer in
flight at a time.

These tests do NOT attempt to prove true parallel throughput (the lock means
work is serialized — that is by design and is fine for a prototype store).
What these tests DO prove, under heavy concurrent load:

1. NO `sqlite3.OperationalError` ("database is locked" / "table is locked")
   ever surfaces, despite many threads contending.
2. NO other transient errors surface either.
3. The DB is not corrupted: a post-barrage `count()` and `query()` succeed
   and the data is consistent.
4. A background `Consolidator` sweep loop running every 50ms alongside live
   traffic does not corrupt state and stops cleanly.
5. A failed atomic consolidation transaction rolls back completely: the
   committed data from before the failed sweep survives untouched, and a
   fresh `MemoryStore` opened on the same file (WAL recovery) sees exactly
   the committed state.
6. Two concurrent `Consolidator.run()` calls both complete without error
   and without corrupting state (the `BEGIN IMMEDIATE` + lock serializes
   them).

WAL recovery (test 3) is exercised on a FILE-BACKED DB so the journal_mode
is actually WAL; in-memory DBs do not engage WAL the same way.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pytest

from isotope_zero.core.consolidation import Consolidator
from isotope_zero.retrieval.hybrid_search import QueryRouter
from isotope_zero.core.store import MemoryStore
from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.eval.adversarial import (
    _BUSY_TIMEOUT_MS,
    _SWEEP_HEARTBEAT_ID,
    _WARFARE_ROW_POOL,
    _warfare_worker,
)
from isotope_zero.types import MemoryCard, now_ts


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def engine() -> EmbeddingEngine:
    """Shared embedding engine (real or fallback — both are fine here)."""
    return EmbeddingEngine()


@pytest.fixture
def file_db_path():
    """A fresh file-backed SQLite DB path, cleaned up after the test.

    Uses NamedTemporaryFile so WAL mode is genuinely engaged (in-memory DBs
    do not use WAL the same way). The file is removed in teardown, including
    the WAL (-wal) and shm (-shm) sidecars SQLite may leave behind.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = tmp.name
    tmp.close()  # close the OS handle; MemoryStore will create the DB file
    # Ensure we start from a clean slate (NamedTemporaryFile creates an empty
    # file; SQLite is fine re-opening it).
    if os.path.exists(path):
        os.remove(path)
    yield path
    # Teardown: remove the DB + WAL/shm sidecars.
    for suffix in ("", "-wal", "-shm"):
        p = path + suffix
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_card(
    cid: str,
    fact: str,
    engine: EmbeddingEngine,
    timestamp: float | None = None,
    access_count: int = 0,
) -> MemoryCard:
    """Build a card with a real (or fallback) embedding from its fact text."""
    return MemoryCard(
        id=cid,
        fact=fact,
        evidence=f"evidence for {fact}",
        timestamp=timestamp if timestamp is not None else now_ts(),
        tags=["stress"],
        embedding=engine.embed_text(fact),
        source_tokens=len(fact.split()),
        access_count=access_count,
    )


# --------------------------------------------------------------------------- #
# Test 1: high-frequency parallel add+query, zero OperationalError
# --------------------------------------------------------------------------- #
def test_parallel_add_query_no_lock_errors(engine, file_db_path):
    """10 threads interleave ~1000 add+query ops; no OperationalError, no
    corruption, DB usable post-barrage.

    Proves the held-connection + single-lock invariant: because every public
    `MemoryStore` method serializes on the same `threading.Lock`, there is
    never more than one writer in flight, so "database is locked" /
    "database table is locked" `sqlite3.OperationalError` is structurally
    impossible. If one appeared, that would be a real bug in the lock design.
    """
    store = MemoryStore(file_db_path, embedder=engine)
    router = QueryRouter(store, engine)

    # Seed a handful of cards so queries have something to hit from the
    # start (otherwise the first wave of queries is a no-op scan).
    for i in range(10):
        store.add(_make_card(f"seed-{i}", f"seed fact number {i}", engine, timestamp=float(i)))

    NUM_THREADS = 10
    OPS_PER_THREAD = 100  # 1000 total ops across all threads
    errors: list[Exception] = []
    errors_lock = threading.Lock()
    op_counter = {"n": 0}
    counter_lock = threading.Lock()

    def _record(exc: Exception) -> None:
        with errors_lock:
            errors.append(exc)

    def writer(tid: int) -> int:
        """Hammer store.add with thread-unique card ids."""
        local = 0
        for j in range(OPS_PER_THREAD):
            try:
                cid = f"t{tid}-c{j}"
                fact = f"concurrent fact from thread {tid} step {j}"
                store.add(_make_card(cid, fact, engine))
                local += 1
                with counter_lock:
                    op_counter["n"] += 1
            except Exception as exc:  # noqa: BLE001 — we want ALL failures
                _record(exc)
                return local
        return local

    def reader(tid: int) -> int:
        """Hammer router.query with a budget-bounded lookup."""
        local = 0
        for j in range(OPS_PER_THREAD):
            try:
                # Vary the query so both SQL and vector paths get exercised.
                q = (
                    f"what is my fact about thread {tid}?"
                    if j % 2 == 0
                    else f"concurrent fact from thread {tid} step {j // 2}"
                )
                router.query(q, token_budget=100)
                local += 1
                with counter_lock:
                    op_counter["n"] += 1
            except Exception as exc:  # noqa: BLE001
                _record(exc)
                return local
        return local

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
        futures = []
        # Half writers, half readers.
        for t in range(NUM_THREADS):
            if t % 2 == 0:
                futures.append(pool.submit(writer, t))
            else:
                futures.append(pool.submit(reader, t))
            # Submit all at once for maximum contention.
        # Drain.
        for f in as_completed(futures):
            f.result(timeout=60)

    total_ops = op_counter["n"]

    # The core assertion: NO OperationalError ("database is locked") at all.
    lock_errors = [
        e for e in errors
        if isinstance(e, sqlite3.OperationalError)
    ]
    assert not lock_errors, (
        f"Found {len(lock_errors)} sqlite3.OperationalError under load: "
        f"{lock_errors[:3]!r}"
    )
    # And no other errors either — the barrage should be error-free.
    assert errors == [], f"unexpected errors under load: {errors[:5]!r}"

    # Post-barrage: the DB is still usable.
    final_count = store.count()
    assert final_count >= 10  # at least the seed cards survived
    # A query must succeed end-to-end.
    res = router.query("concurrent fact from thread", token_budget=100)
    assert res.latency_ms >= 0.0

    store.close()

    # Sanity: total operations were actually attempted (not a silent no-op).
    assert total_ops == NUM_THREADS * OPS_PER_THREAD, (
        f"expected {NUM_THREADS * OPS_PER_THREAD} ops, got {total_ops}"
    )


# --------------------------------------------------------------------------- #
# Test 2: background Consolidator loop (50ms) alongside traffic
# --------------------------------------------------------------------------- #
def test_background_consolidator_loop_alongside_traffic(engine, file_db_path):
    """A background Consolidator sweeping every 50ms runs alongside ~500
    mixed add+query ops across 4-6 threads. Zero exceptions from workers or
    the background thread; store.count() stays consistent; daemon stops
    cleanly and is no longer alive.

    Proves the off-hot-path consolidation worker does not corrupt state or
    raise under concurrent traffic, and that `stop()` joins the daemon
    thread promptly.
    """
    store = MemoryStore(file_db_path, embedder=engine)
    router = QueryRouter(store, engine)

    # Seed a few distinct cards so the consolidator has real work to consider
    # and so queries surface hits.
    for i in range(8):
        store.add(_make_card(f"seed-{i}", f"seed fact number {i}", engine, timestamp=float(i)))

    # min_age_seconds=0 so decay CAN prune never-recalled cards, exercising
    # the real sweep path (dedup + decay + atomic apply).
    cons = Consolidator(store, embedder=engine, min_age_seconds=0.0)
    cons.start_background_loop(interval_seconds=0.05)

    errors: list[Exception] = []
    errors_lock = threading.Lock()
    stop_flag = threading.Event()
    added = {"n": 0}
    added_lock = threading.Lock()

    def _record(exc: Exception) -> None:
        with errors_lock:
            errors.append(exc)

    def writer(tid: int) -> None:
        j = 0
        while not stop_flag.is_set() and j < 100:
            try:
                cid = f"bg-t{tid}-{j}"
                store.add(_make_card(cid, f"bg fact {tid} {j}", engine))
                with added_lock:
                    added["n"] += 1
                j += 1
            except Exception as exc:  # noqa: BLE001
                _record(exc)
                return

    def reader(tid: int) -> None:
        j = 0
        while not stop_flag.is_set() and j < 100:
            try:
                router.query(f"bg fact {tid} {j // 2}", token_budget=80)
                j += 1
            except Exception as exc:  # noqa: BLE001
                _record(exc)
                return

    threads = []
    for t in range(6):
        target = writer if t % 2 == 0 else reader
        th = threading.Thread(target=target, args=(t,))
        threads.append(th)
        th.start()

    # Let the workers + background loop run for a bounded, short time.
    time.sleep(0.6)
    stop_flag.set()
    for th in threads:
        th.join(timeout=10)

    # Stop the background loop and assert it is no longer alive.
    cons.stop(timeout=5)
    assert cons._thread is None or not cons._thread.is_alive(), (
        "background consolidator thread still alive after stop()"
    )

    # No exceptions from worker threads OR from the background sweep.
    assert errors == [], f"errors during background-loop traffic: {errors[:5]!r}"

    # Count is consistent: at least one card remains, and never exceeds the
    # total we wrote (consolidation may prune decayed dupes, so we only
    # assert a sane range + queryability).
    final_count = store.count()
    total_written = 8 + added["n"]  # seed + writer adds
    assert 1 <= final_count <= total_written, (
        f"final count {final_count} outside [1, {total_written}]"
    )
    # Store is still queryable post-sweep.
    res = router.query("bg fact", token_budget=100)
    assert res.latency_ms >= 0.0

    store.close()


# --------------------------------------------------------------------------- #
# Test 3: transaction rollback + WAL recovery on reconnect
# --------------------------------------------------------------------------- #
class _ExplodingStr:
    """A str-like object whose ``__str__`` raises, to force a mid-txn failure.

    SQLite's parameter binding calls ``str()`` on certain bound values; passing
    an instance whose ``__str__`` raises forces the INSERT inside the
    ``consolidate_memories`` transaction to fail AFTER ``BEGIN IMMEDIATE``,
    which exercises the ROLLBACK path. We use this instead of monkeypatching
    the connection because it forces a genuine, in-execution failure that the
    store's own try/except/ROLLBACK must handle.
    """

    def __str__(self) -> str:  # noqa: D401
        raise RuntimeError("induced mid-txn failure for rollback test")


def test_consolidate_memories_rollback_and_wal_recovery(engine, file_db_path):
    """A failed ``consolidate_memories`` rolls back fully; committed card A
    survives untouched; a fresh ``MemoryStore`` on the same file (WAL
    recovery) sees exactly the committed state.

    Proves (a) the atomic batch's ROLLBACK path actually rolls back, so a
    half-applied sweep is never committed, and (b) SQLite WAL recovers
    cleanly on reconnect — committed data survives, the rolled-back txn does
    not.
    """
    store = MemoryStore(file_db_path, embedder=engine)

    # (a) Seed a committed card A with a known fact.
    card_a = _make_card("card-A", "original", engine, timestamp=now_ts(), access_count=1)
    store.add(card_a)
    assert store.count() == 1
    got = store.get("card-A")
    assert got is not None and got.fact == "original"

    # (b) Sanity: a NORMAL consolidate_memories call (valid survivor + a
    # bogus deleted_id that won't raise) commits and leaves A correct.
    survivor = _make_card("card-B", "survivor fact", engine, timestamp=now_ts())
    deleted = store.consolidate_memories([survivor], ["nonexistent-id"])
    # bogus id deletes nothing; survivor upserted.
    assert deleted == 0
    assert store.get("card-A").fact == "original"
    assert store.get("card-B") is not None
    # count is now 2 (A + B survivor).
    assert store.count() == 2

    # (c) Rollback test: force a failure mid-transaction by passing a "bad
    # card" whose fact field is an object whose __str__ raises. The INSERT
    # inside consolidate_memories will fail AFTER BEGIN IMMEDIATE; the store
    # must ROLLBACK and re-raise, leaving the committed state intact.
    bad_card = MemoryCard(
        id="card-C",
        fact=_ExplodingStr(),  # type: ignore[arg-type]
        evidence="will fail",
        timestamp=now_ts(),
        tags=[],
        embedding=engine.embed_text("will fail"),
        source_tokens=2,
    )
    with pytest.raises(Exception):
        store.consolidate_memories([bad_card], [])

    # The rolled-back txn must NOT have committed anything: A and B are still
    # intact, C was never persisted.
    assert store.count() == 2, "rolled-back txn appears to have committed!"
    a_again = store.get("card-A")
    assert a_again is not None
    assert a_again.fact == "original", "card A mutated by a rolled-back txn!"
    assert a_again.access_count == 1
    assert store.get("card-B") is not None
    assert store.get("card-C") is None

    # (d) WAL recovery on reconnect: close this store, open a brand-new
    # MemoryStore on the SAME file path, and verify the committed data
    # survived while the rolled-back txn did not.
    store.close()

    # Give SQLite a moment to finalize WAL checkpoint on close.
    time.sleep(0.05)

    store2 = MemoryStore(file_db_path, embedder=engine)
    assert store2.count() == 2, "WAL recovery lost committed cards!"
    a_recovered = store2.get("card-A")
    assert a_recovered is not None
    assert a_recovered.fact == "original"
    assert a_recovered.access_count == 1
    b_recovered = store2.get("card-B")
    assert b_recovered is not None
    assert b_recovered.fact == "survivor fact"
    assert store2.get("card-C") is None, "rolled-back txn survived reconnect!"
    store2.close()


def test_consolidate_memories_rollback_via_mock_failure(engine, file_db_path):
    """A second rollback path: swap the store's held connection for a wrapper
    whose cursor raises on the DELETE step (the 2nd+ SQL statement inside the
    txn), forcing a ROLLBACK AFTER the survivor upsert has already run.

    `sqlite3.Connection` is a C object whose attributes are read-only, so we
    cannot `patch.object(store._conn, 'cursor', ...)`. Instead we replace
    `store._conn` (a plain Python instance attribute on `MemoryStore`) with a
    thin proxy that forwards everything to the real connection but hands back
    a flaky cursor. This exercises the store's own try/except/ROLLBACK path.

    Proves the rollback path covers failures that occur PARTWAY through the
    batch (not just on the first statement): already-upserted survivors must
    be rolled back, not committed.
    """
    store = MemoryStore(file_db_path, embedder=engine)
    card_a = _make_card("card-A", "committed-original", engine, timestamp=now_ts())
    store.add(card_a)

    survivor = _make_card("card-A", "would-be-overwrite", engine, timestamp=now_ts())
    # A real deleted_id that exists, so the DELETE statement actually runs.
    store.add(_make_card("card-D", "to be deleted", engine, timestamp=now_ts()))

    real_conn = store._conn

    class _FlakyCursor:
        """Wraps a real cursor; raises on DELETE to induce a mid-txn failure."""

        def __init__(self, real):
            self._real = real

        def execute(self, sql, *params):
            if isinstance(sql, str) and "DELETE" in sql.upper():
                raise sqlite3.OperationalError(
                    "induced failure on DELETE step"
                )
            return self._real.execute(sql, *params)

        def __getattr__(self, name):
            return getattr(self._real, name)

        def close(self):
            try:
                self._real.close()
            except Exception:  # noqa: BLE001
                pass

    class _FlakyConn:
        """Forwards all attribute access to the real connection except
        `cursor()`, which returns a flaky cursor."""

        def __getattr__(self, name):
            return getattr(real_conn, name)

        def cursor(self):
            return _FlakyCursor(real_conn.cursor())

        def close(self):
            real_conn.close()

    # Swap in the flaky connection for the duration of the failing call.
    flaky = _FlakyConn()
    store._conn = flaky  # type: ignore[assignment]
    try:
        with pytest.raises(sqlite3.OperationalError):
            store.consolidate_memories([survivor], ["card-D"])
    finally:
        # Restore the real connection so the post-rollback assertions hit the
        # genuine held connection (which still has the committed state).
        store._conn = real_conn

    # Rollback: A was NOT overwritten by the survivor (the upsert that ran
    # before the DELETE was rolled back), and D was NOT deleted.
    a = store.get("card-A")
    assert a is not None and a.fact == "committed-original", (
        "mid-txn upsert was committed instead of rolled back!"
    )
    assert store.get("card-D") is not None, "mid-txn delete leaked!"
    store.close()


# --------------------------------------------------------------------------- #
# Test 4 (bonus): two concurrent Consolidator.run() calls
# --------------------------------------------------------------------------- #
def test_two_concurrent_consolidator_runs(engine, file_db_path):
    """Two ``Consolidator.run()`` calls execute simultaneously from two
    threads. Both must complete without exception and without corrupting
    state. The ``BEGIN IMMEDIATE`` + single store lock serializes them: one
    may block briefly, but neither raises and the store is consistent
    afterwards.

    Proves the atomic apply path is safe under concurrent consolidations:
    no lost updates, no partial-applied sweeps, no lock errors.
    """
    store = MemoryStore(file_db_path, embedder=engine)
    # Seed cards including exact-fact duplicates so the consolidators have
    # real merge work to do, plus a couple of distinct facts.
    now = now_ts()
    for i in range(6):
        store.add(_make_card(f"dup-{i}", "exact duplicate fact", engine, timestamp=now - i))
    for i in range(4):
        store.add(_make_card(f"uniq-{i}", f"unique fact {i}", engine, timestamp=now - i))
    # A recalled card so decay doesn't prune everything.
    store.get("uniq-0")  # touch via read is not access_count; touch explicitly:
    store.touch("uniq-0")

    cons = Consolidator(store, embedder=engine, min_age_seconds=0.0)
    errors: list[Exception] = []
    errors_lock = threading.Lock()
    reports = []

    def _run_once() -> None:
        try:
            r = cons.run()
            # Stash a snapshot of the count right after this run, under the
            # lock, so we can inspect consistency without races.
            with errors_lock:
                reports.append((r, store.count()))
        except Exception as exc:  # noqa: BLE001
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=_run_once) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == [], f"concurrent consolidator errors: {errors!r}"
    assert len(reports) == 2

    # The store is consistent: the dup-0..5 cluster collapsed to ONE survivor
    # (the earliest-timestamp member, dup-5), and the 4 unique cards either
    # survived or were pruned by decay. Final count is sane and the store is
    # queryable. Because both runs operated on the same store, the second run
    # should have seen the first run's results (no re-merge needed), so the
    # final count equals the count after the last-completing run.
    final_count = store.count()
    # 6 dups -> 1 survivor; 4 uniques; some may be pruned by decay (never the
    # recalled one). So final is between 1 and 5.
    assert 1 <= final_count <= 5, f"unexpected final count {final_count}"

    # The earliest-timestamp dup survivor must still be present.
    survivor = store.get("dup-5")
    assert survivor is not None, "dedup survivor (earliest timestamp) lost!"
    assert survivor.fact == "exact duplicate fact"

    # The recalled unique card must survive (recalled cards are never pruned).
    assert store.get("uniq-0") is not None, "recalled card was pruned!"

    # Store is queryable.
    res = QueryRouter(store, engine).query("exact duplicate", token_budget=100)
    assert res.latency_ms >= 0.0

    store.close()


# --------------------------------------------------------------------------- #
# Test 5: multi-PROCESS contention warfare — zero errors with busy_timeout
# --------------------------------------------------------------------------- #
def test_multiprocess_contention_zero_errors_with_busy_timeout(file_db_path):
    """4 worker PROCESSES hammer a shared file-backed WAL DB's bounded row pool
    while a parent-side consolidator sweeps alongside every 50 ms. Both the
    workers and the sweep-store set ``PRAGMA busy_timeout=5000``, so SQLite
    waits out transient write locks instead of raising "database is locked":
    the outcome must be ZERO `sqlite3.OperationalError`, zero WAL corruption,
    and a DB that is still fully usable afterwards.

    Reuses the production `_warfare_worker` (and its `_warfare_worker`-style
    row pool / busy_timeout / heartbeat constants) so the test cannot drift
    from the battle-tested Section D logic. A deliberately small budget — 4
    workers x 200 cycles, 50 ms sweep — exercises genuine cross-process lock
    contention in a couple of seconds, so it is a permanent DEFAULT test (NOT
    gated behind IZERO_STRESS).

    Background (the documented finding this locks in): `MemoryStore` opens its
    connections with SQLite's default busy_timeout of 0, so a second writer
    that hits a locked database raises `sqlite3.OperationalError` immediately.
    The store's own held connection is single-threaded via its lock and never
    needs one, but every EXTERNAL writer/consolidator sharing the DB file MUST
    pin ``PRAGMA busy_timeout`` to wait out transient locks — which is exactly
    what the workers and the parent sweep-store do here.
    """
    db_path = file_db_path

    # Seed one card so worker SELECTs scan rows from the start. No ONNX needed:
    # workers embed via the deterministic `_warfare_worker` blob writer and the
    # sweep reuses the workers' blobs, so the test measures SQLite contention,
    # not the embedding path.
    store = MemoryStore(db_path, embedder=None)
    store.add(
        MemoryCard(
            id="seed",
            fact="seed fact",
            evidence="ev",
            timestamp=now_ts(),
            tags=["seed"],
            embedding=None,
            source_tokens=2,
        )
    )
    conn = store._conn
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    store.close()

    N_WORKERS = 4
    CYCLES_PER_WORKER = 200
    SWEEP_INTERVAL_S = 0.05  # 50 ms parent-side consolidation sweep

    args = [(db_path, w, CYCLES_PER_WORKER, 9999) for w in range(N_WORKERS)]
    pool = ProcessPoolExecutor(max_workers=N_WORKERS)
    futures = [pool.submit(_warfare_worker, a) for a in args]
    all_futures = list(futures)

    sweeps = 0
    sweep_errors: list[Exception] = []
    t_last = time.perf_counter()
    while futures:
        # Harvest finished workers as they complete.
        still_open = [f for f in futures if not f.done()]
        futures = still_open
        # Parent-side consolidator: open a SEPARATE connection with the REQUIRED
        # busy_timeout (the store itself sets none), run a full sweep, then
        # heartbeat-UPDATE so every pass takes a genuine write lock even when
        # dedup has nothing to merge.
        now = time.perf_counter()
        if now - t_last >= SWEEP_INTERVAL_S:
            sweep_store = MemoryStore(db_path, embedder=None)
            try:
                sweep_store._conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
                Consolidator(sweep_store, embedder=None).run()
                sweep_store.update(
                    MemoryCard(
                        id=_SWEEP_HEARTBEAT_ID,
                        fact=f"heartbeat {sweeps}",
                        evidence="sweep marker",
                        timestamp=now_ts(),
                        tags=["_sweep"],
                        embedding=None,
                        source_tokens=1,
                        last_access=now_ts(),
                    )
                )
                sweeps += 1
            except Exception as exc:  # noqa: BLE001 — capture any sweep failure
                sweep_errors.append(exc)
            finally:
                sweep_store.close()
            t_last = now
        if futures:
            time.sleep(0.002)
    pool.shutdown(wait=True)

    reports = [f.result() for f in all_futures]
    operational_errors = sum(r["operational_errors"] for r in reports)
    database_errors = sum(r["database_errors"] for r in reports)
    total_ops = sum(r["ops"] for r in reports)

    # The core assertion: cross-process contention yields ZERO lock errors.
    assert total_ops == N_WORKERS * CYCLES_PER_WORKER, (
        f"expected {N_WORKERS * CYCLES_PER_WORKER} ops, got {total_ops}"
    )
    assert operational_errors == 0, (
        f"{operational_errors} sqlite3.OperationalError under multi-process "
        f"contention despite busy_timeout={_BUSY_TIMEOUT_MS}"
    )
    assert database_errors == 0, f"{database_errors} sqlite3.DatabaseError"

    # The sweep must have actually run DURING warfare (proving the heartbeat
    # UPDATE + consolidation completed under contention) and raised nothing.
    assert sweeps >= 1, "no consolidation sweep ran alongside the workers"
    assert sweep_errors == [], (
        f"parent-side consolidation sweep raised: {sweep_errors[:5]!r}"
    )

    # WAL must be intact, and the DB usable + bounded by the shared row pool.
    check = sqlite3.connect(db_path)
    try:
        row = check.execute("PRAGMA quick_check").fetchone()
    finally:
        check.close()
    assert row is not None and str(row[0]) == "ok", (
        f"WAL corruption detected by PRAGMA quick_check: {row}"
    )

    verify = MemoryStore(db_path, embedder=None)
    try:
        final_count = verify.count()
        # Bounded shared row pool: at most the 200 pool rows + seed + heartbeat.
        assert final_count <= _WARFARE_ROW_POOL + 2, (
            f"DB grew to {final_count} rows; bounded row pool not honored "
            f"(pool={_WARFARE_ROW_POOL})"
        )
        assert verify.get("seed") is not None, "seed card lost during warfare"
    finally:
        verify.close()
