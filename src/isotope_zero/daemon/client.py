"""Shared-memory embedding daemon client — drop-in for ``EmbeddingEngine``.

Phase 7A: moves onnxruntime (~360MB RSS) out of client processes into a
dedicated daemon process. ``DaemonClient`` is a duck-typed replacement for
``isotope_zero.embeddings.onnx_embed.EmbeddingEngine``: it exposes
``.embed_text``, ``.embed_batch``, ``.is_real``, ``.dim`` and is accepted
anywhere the engine is (``MemoryStore``, ``QueryRouter``, ``Consolidator``).

HARD REQUIREMENT: this module imports NO third-party packages at module level
(no onnxruntime, no tokenizers, no numpy) — stdlib only — so importing it in a
client process does not load onnxruntime. Vectors travel over POSIX shared
memory (``multiprocessing.shared_memory``, stdlib).

Wire protocol (mirrors ``isotope_zero.daemon.server``):
    client -> daemon: ``struct.pack(">II", len(header_json), len(payload)) +
                       header_json + payload``
    daemon -> client: ``struct.pack(">I", len(reply_json)) + reply_json``
Commands: ping / hello / embed_batch / stats / shutdown.
"""
from __future__ import annotations

import array
import atexit
import json
import multiprocessing.shared_memory
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Any

_DEFAULT_SOCKET = "/tmp/izero.sock"
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIM = 384

_FRAME_STRUCT = struct.Struct(">II")
_RESP_STRUCT = struct.Struct(">I")
_MAX_RESP_LEN = 1 << 22  # 4 MiB


class DaemonClient:
    """RPC client for the shared-memory embedding daemon.

    Parameters
    ----------
    model_name:
        Embedding model the daemon should load. Only used when this client has
        to spawn the daemon itself (``spawn=True`` and no daemon reachable).
    socket_path:
        Unix domain socket the daemon listens on.
    spawn:
        If the socket is missing or refused, spawn a daemon via
        ``python -m isotope_zero.daemon.server`` and wait for it to come up.
    connect_timeout:
        Seconds to wait for a spawned daemon to become reachable.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        socket_path: str = _DEFAULT_SOCKET,
        spawn: bool = True,
        connect_timeout: float = 10.0,
    ) -> None:
        if not hasattr(socket, "AF_UNIX"):
            # The daemon is a POSIX transport (Unix-domain socket + shared
            # memory); Windows CPython builds don't expose socket.AF_UNIX.
            # Fail with intent instead of an AttributeError deep in connect.
            raise NotImplementedError(
                "DaemonClient requires POSIX Unix-domain sockets and shared "
                "memory and is unsupported on Windows. Use the in-process "
                "EmbeddingEngine (isotope_zero.embeddings.onnx_embed) instead."
            )
        self.model_name = model_name
        self.socket_path = socket_path
        self.spawn = spawn
        self.connect_timeout = float(connect_timeout)
        self._sock: socket.socket | None = None
        self._io_lock = threading.Lock()  # serializes frame exchange on the socket
        self._seq_lock = threading.Lock()
        self._seq = 0
        self._proc: subprocess.Popen | None = None  # handle if WE spawned the daemon
        self._is_real = False
        self._dim = _DEFAULT_DIM
        self._closed = False
        self._connect()
        atexit.register(self.close)

    # ------------------------------------------------------------------ #
    # Duck-typed embedder API
    # ------------------------------------------------------------------ #
    @property
    def is_real(self) -> bool:
        return self._is_real

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string. Empty input -> zero vector (engine parity)."""
        if text == "":
            return [0.0] * self._dim
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Returns ``list[list[float]]``, input-ordered.

        Creates a POSIX shared-memory segment, sends the texts to the daemon,
        reads the float32 vectors back row-major, and always unlinks the
        segment. Auto-reconnects (respawn + retry) once if the daemon dies
        mid-call.
        """
        if not texts:
            return []
        try:
            return self._embed_batch_once(texts)
        except OSError as exc:
            # One auto-reconnect: the daemon may have exited (idle timeout /
            # crash). Respawn + retry; if the retry fails the error propagates.
            if self._closed:
                raise
            self.close()
            try:
                self._connect()
            except Exception:
                raise ConnectionError(
                    f"embedding daemon unavailable after reconnect attempt: {exc}"
                ) from exc
            return self._embed_batch_once(texts)

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #
    def _connect(self) -> None:
        sock = self._try_connect()
        if sock is None and self.spawn:
            self._spawn_daemon()
            sock = self._poll_connect(self.connect_timeout)
        if sock is None:
            raise ConnectionError(
                f"cannot reach embedding daemon at {self.socket_path!r} "
                f"(spawn={self.spawn})"
            )
        self._sock = sock
        self._closed = False
        resp = self._send_cmd({"cmd": "hello"})
        if not resp.get("ok"):
            self.close()
            raise RuntimeError(f"daemon hello failed: {resp.get('error', resp)}")
        self._is_real = bool(resp.get("is_real", False))
        self._dim = int(resp.get("dim", _DEFAULT_DIM))

    def _try_connect(self) -> socket.socket | None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(5.0)
            s.connect(self.socket_path)
            s.settimeout(60.0)
            return s
        except OSError:
            try:
                s.close()
            except OSError:
                pass
            return None

    def _spawn_daemon(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "isotope_zero.daemon.server",
            "--model",
            self.model_name,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            self._proc = None

    def _poll_connect(self, timeout: float) -> socket.socket | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            s = self._try_connect()
            if s is not None:
                return s
            time.sleep(0.05)
        return None

    # ------------------------------------------------------------------ #
    # Wire I/O
    # ------------------------------------------------------------------ #
    def _send_cmd(self, header: dict[str, Any], payload: bytes = b"") -> dict[str, Any]:
        with self._io_lock:
            sock = self._sock
            if sock is None:
                raise ConnectionError("not connected to embedding daemon")
            hb = json.dumps(header).encode("utf-8")
            sock.sendall(_FRAME_STRUCT.pack(len(hb), len(payload)) + hb + payload)
            head = self._recv_exact(sock, _RESP_STRUCT.size)
            if head is None:
                raise ConnectionError("embedding daemon closed connection")
            (rlen,) = _RESP_STRUCT.unpack(head)
            if rlen < 0 or rlen > _MAX_RESP_LEN:
                raise ConnectionError("invalid daemon reply length")
            body = self._recv_exact(sock, rlen)
            if body is None:
                raise ConnectionError("embedding daemon closed connection")
            return json.loads(body.decode("utf-8"))

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            try:
                b = sock.recv(remaining)
            except socket.timeout:
                raise ConnectionError("embedding daemon timed out mid-reply") from None
            except OSError:
                return None
            if not b:
                return None
            chunks.append(b)
            remaining -= len(b)
        return b"".join(chunks)

    def _embed_batch_once(self, texts: list[str]) -> list[list[float]]:
        n = len(texts)
        dim = self._dim
        total = n * dim
        size = total * 4
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
        shm = None
        try:
            shm = multiprocessing.shared_memory.SharedMemory(
                create=True, size=size, name=f"izero_{os.getpid()}_{seq}"
            )
            header = {
                "cmd": "embed_batch",
                "shm": shm.name,
                "n": n,
                "dim": dim,
                "seq": seq,
            }
            payload = json.dumps(texts).encode("utf-8")
            resp = self._send_cmd(header, payload)
            if not resp.get("ok"):
                raise RuntimeError(f"embed_batch failed: {resp.get('error')}")
            # Daemon wrote the vectors into our shm region BEFORE sending the
            # ack, so reading here is safe. Read exactly `size` bytes.
            flat = array.array("f", shm.buf[:size].tobytes())
            return [flat[i * dim : (i + 1) * dim].tolist() for i in range(n)]
        finally:
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Introspection / lifecycle
    # ------------------------------------------------------------------ #
    def ping(self) -> bool:
        """Round-trip liveness check against the daemon."""
        return bool(self._send_cmd({"cmd": "ping"}).get("ok"))

    def stats(self) -> dict[str, Any]:
        """Ask the daemon for its request count / uptime / is_real."""
        return self._send_cmd({"cmd": "stats"})

    def close(self) -> None:
        """Close the client socket WITHOUT shutting down the daemon (it is
        shared across processes). Safe to call repeatedly."""
        self._closed = True
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def shutdown(self) -> None:
        """Ask the daemon to exit cleanly (unlink socket, os._exit) and close
        our socket. Used by tests / benchmark cleanup."""
        try:
            if self._sock is not None:
                try:
                    self._send_cmd({"cmd": "shutdown"})
                except (OSError, socket.error):
                    pass
        finally:
            self.close()

    def __enter__(self) -> "DaemonClient":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def __del__(self) -> None:  # pragma: no cover - best-effort
        try:
            self.close()
        except Exception:
            pass
