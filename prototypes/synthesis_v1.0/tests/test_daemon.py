"""Tests for the Phase 7A shared-memory embedding daemon.

Coverage:
    * hello handshake parity vs an in-process EmbeddingEngine (same cache)
    * bit-identical embedding parity across the 32-chunk boundary
    * auto-spawn when no daemon is pre-running
    * MemoryStore(use_daemon=True) wires in a DaemonClient end-to-end
    * shutdown() makes the daemon exit and unlink its socket

Teardown: kill any leftover ``isotope_zero.daemon.server`` children and unlink
the daemon socket so no daemon / shared-memory region leaks persist. The
concurrent benchmark agent uses ``/tmp/izero_bench.sock``, and the cleanup
below deliberately leaves any daemon whose command line references it alone.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import time

import pytest

from isotope_zero.core.store import MemoryStore
from isotope_zero.daemon.client import DaemonClient
from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.types import MemoryCard, now_ts

DEFAULT_SOCKET = "/tmp/izero.sock"


# ---------------------------------------------------------------------- #
# Cleanup helpers (isolate the tests from each other and from the
# concurrent benchmark agent's daemon).
# ---------------------------------------------------------------------- #
def _unlink_default_socket() -> None:
    try:
        if os.path.exists(DEFAULT_SOCKET):
            os.unlink(DEFAULT_SOCKET)
    except OSError:
        pass


def _kill_leftover_daemons() -> None:
    """SIGKILL leftover daemon processes, leaving the benchmark's alone."""
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return
    for line in out.splitlines():
        if "isotope_zero.daemon.server" not in line:
            continue
        if "izero_bench" in line:
            continue  # the concurrent benchmark agent's daemon — not ours
        parts = line.lstrip().split(None, 1)
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _daemon_running() -> bool:
    """True if something answers a connect on the daemon socket."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        s.connect(DEFAULT_SOCKET)
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


@pytest.fixture(autouse=True)
def _clean_slate():
    """Kill leftovers + unlink the socket before AND after each test, so each
    test starts with no pre-running daemon and nothing leaks out."""
    _kill_leftover_daemons()
    _unlink_default_socket()
    yield
    _kill_leftover_daemons()
    _unlink_default_socket()


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #
def test_hello():
    """DaemonClient hello reports is_real/dim matching an in-process engine
    using the same cache dir."""
    with DaemonClient(spawn=True) as client:
        ref = EmbeddingEngine()  # same default cache dir (.isotope_zero_cache)
        try:
            # The daemon loaded the real ONNX model because the cache exists.
            assert client.is_real is True
            assert ref.is_real is True
            assert client.is_real == ref.is_real
            assert client.dim == ref.dim
            assert client.dim == 384
        finally:
            del ref
        assert client.ping() is True
        # The daemon itself reports the same is_real through stats too.
        st = client.stats()
        assert st["ok"] is True
        assert st["is_real"] is True
        assert st["n_requests"] >= 1


def test_embed_bit_identical_parity():
    """Daemon vectors must EQUAL (==, no tolerance) the in-process engine's,
    including across the 32-chunk ONNX forward boundary."""
    texts = [
        "",
        "the user likes rust and tea",
        "x" * 500,
        "ünïcödé 🦀 茶 and 日本語 mixed text",
    ]
    texts += [
        f"distinct fact {i}: the user runs worker {i} in zone {i} with port {10000 + i}"
        for i in range(33)
    ]
    assert len(texts) == 37
    assert len(texts) > 32  # forces at least two embed chunks

    with DaemonClient(spawn=True) as client:
        ref = EmbeddingEngine()
        try:
            got = client.embed_batch(texts)
            want = ref.embed_batch(texts)
        finally:
            del ref

    assert len(got) == len(want) == len(texts)
    max_abs_diff = 0.0
    mismatches = 0
    for idx, (gi, wi) in enumerate(zip(got, want)):
        assert len(gi) == len(wi), f"text {idx}: dim mismatch {len(gi)} vs {len(wi)}"
        for a, b in zip(gi, wi):
            d = abs(a - b)
            max_abs_diff = max(max_abs_diff, d)
            if a != b:
                mismatches += 1
    assert mismatches == 0, (
        f"bit-parity FAIL: {mismatches} element mismatches, "
        f"max_abs_diff={max_abs_diff:.3e}"
    )


def test_auto_spawn():
    """With no daemon pre-running, a fresh DaemonClient auto-spawns one and
    answers a ping + embed."""
    assert not _daemon_running()  # fixture cleared the slate
    assert not os.path.exists(DEFAULT_SOCKET)
    client = DaemonClient(spawn=True)
    try:
        assert client.ping() is True
        vec = client.embed_text("an auto-spawned daemon answers embeds")
        assert len(vec) == client.dim
        assert client.is_real is True
    finally:
        client.close()


def test_store_use_daemon_flag():
    """MemoryStore(use_daemon=True) wires a DaemonClient; a card embedded via
    the daemon is stored and returned by vector_search."""
    store = MemoryStore(":memory:", use_daemon=True)
    try:
        assert isinstance(store.embedder, DaemonClient)
        assert store.embedder.is_real is True

        fact = "the user prefers the daemon architecture for embeddings"
        emb = store.embedder.embed_text(fact)
        assert len(emb) == store.embedder.dim

        store.add(
            MemoryCard(
                id="daemon-card-1",
                fact=fact,
                evidence="test evidence",
                timestamp=now_ts(),
                tags=["embedding", "daemon"],
                embedding=emb,
                source_tokens=8,
            )
        )
        hits = store.vector_search(emb, k=5)
        assert hits, "expected the stored card to be found by vector_search"
        assert hits[0][0].id == "daemon-card-1"
        assert hits[0][1] >= 0.99  # identical normalized vector -> ~1.0 cosine
    finally:
        store.close()


def test_cleanup():
    """shutdown() makes the daemon exit and unlink its socket."""
    client = DaemonClient(spawn=True)
    try:
        assert client.ping() is True
        assert os.path.exists(DEFAULT_SOCKET)
    finally:
        pass
    client.shutdown()
    client.close()

    deadline = time.time() + 5.0
    while os.path.exists(DEFAULT_SOCKET) and time.time() < deadline:
        time.sleep(0.05)
    assert not os.path.exists(DEFAULT_SOCKET), "socket not unlinked after shutdown"
    assert not _daemon_running(), "daemon still accepting connections after shutdown"