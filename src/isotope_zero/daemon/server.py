"""Shared-memory embedding daemon server for isotope_zero (Phase 7A).

Loads ONE `EmbeddingEngine` (the onnxruntime-heavy process) and serves
embedding requests over a Unix domain socket. Requests carry a
length-prefixed JSON control header; batch vectors are returned through a
client-created POSIX shared-memory segment (named in the header) rather than
over the socket, so large batches move zero-copy through the page cache.

Protocol
--------
Every client -> daemon message is framed as::

    struct.pack(">II", len(header_json), len(payload)) + header_json + payload

The header is a JSON object; the payload is command-specific raw bytes (for
``embed_batch`` it is the UTF-8 JSON array of texts). Every daemon -> client
reply is::

    struct.pack(">I", len(reply_json)) + reply_json

Commands (header JSON ``cmd`` field):

* ``ping``            -> ``{"ok": true}``
* ``hello``           -> ``{"ok": true, "is_real": <engine.is_real>,
                           "dim": <engine.dim>, "model": <model_name>}``
* ``embed_batch``     -> header ``{"cmd":"embed_batch","shm":<shm_name>,
                           "n":<n>,"dim":<dim>,"seq":<seq>}`` plus the JSON
                           array of texts as the payload. The daemon attaches
                           to the named shared-memory segment, writes the n*dim
                           float32 results row-major into it, then replies
                           ``{"ok":true,"n":n,"dim":dim,"seq":seq}`` (or
                           ``{"ok":false,"error":...}``).
* ``stats``           -> ``{"ok":true,"n_requests":...,"uptime_s":...,
                           "is_real":...}``
* ``shutdown``        -> ``{"ok":true}`` then the daemon unlinks the socket
                           and exits (``os._exit(0)``).

The daemon exits cleanly when it has received no requests for ``--idle-timeout``
seconds (default 300), or on SIGTERM/SIGINT.

This module is the ONNX process: it MAY import onnxruntime / numpy. It must
NOT import ``isotope_zero.daemon.client`` (no import cycles).
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing.shared_memory
import os
import signal
import socket
import struct
import threading
import time
from typing import Any

from ..embeddings.onnx_embed import EmbeddingEngine

log = logging.getLogger("isotope_zero.daemon.server")

_DEFAULT_SOCKET = "/tmp/izero.sock"
_DEFAULT_CACHE_DIR = ".isotope_zero_cache"
_DEFAULT_IDLE_TIMEOUT = 300.0
# Framing (see module docstring).
_FRAME_STRUCT = struct.Struct(">II")
_RESP_STRUCT = struct.Struct(">I")
# Sanity caps so a malformed peer cannot ask us to allocate unbounded buffers.
_MAX_HEADER_LEN = 1 << 22  # 4 MiB
_MAX_PAYLOAD_LEN = 1 << 30  # 1 GiB


class IzeroDaemon:
    """The embedding daemon: one heavy ONNX engine, threaded socket server."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        socket_path: str = _DEFAULT_SOCKET,
        cache_dir: str = _DEFAULT_CACHE_DIR,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
    ) -> None:
        self.model_name = model_name
        self.socket_path = socket_path
        self.cache_dir = cache_dir
        self.idle_timeout = float(idle_timeout)
        # The ONE heavy object: loads onnxruntime + the quantized model.
        self.engine = EmbeddingEngine(model_name, cache_dir=cache_dir)
        self._start = time.time()
        self._n_requests = 0
        self._n_lock = threading.Lock()
        self._embed_lock = threading.Lock()  # serialize engine calls (tokenizer is not thread-safe)
        self._last_request = time.time()
        self._stop = threading.Event()
        self._listener: socket.socket | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def n_requests(self) -> int:
        with self._n_lock:
            return self._n_requests

    def _bump_requests(self) -> None:
        with self._n_lock:
            self._n_requests += 1

    def _touch(self) -> None:
        self._last_request = time.time()

    def run(self) -> None:
        """Bind the socket and serve until shutdown / idle timeout."""
        self._unlink_stale_socket()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(self.socket_path)
        except OSError:
            # Bind race (socket file created between unlink and bind): retry once.
            self._unlink_stale_socket()
            listener.bind(self.socket_path)
        try:
            os.chmod(self.socket_path, 0o600)
        except OSError:  # pragma: no cover - permission edge, never fatal
            pass
        listener.listen(16)
        listener.settimeout(1.0)
        self._listener = listener
        log.info(
            "IzeroDaemon listening on %s model=%s is_real=%s dim=%d",
            self.socket_path, self.model_name, self.engine.is_real, self.engine.dim,
        )
        # Clean shutdown on SIGTERM/SIGINT (runs in the main thread).
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

        try:
            self._accept_loop()
        finally:
            self._cleanup()

    def _accept_loop(self) -> None:
        """Blocking accept loop in the main thread; one daemon thread/connection."""
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()  # type: ignore[union-attr]
            except socket.timeout:
                if self.idle_timeout > 0 and (time.time() - self._last_request) > self.idle_timeout:
                    log.info("idle timeout (%.1fs) reached; exiting cleanly", self.idle_timeout)
                    self._shutdown(0)
                    return
                continue
            except OSError:
                # Listener closed by _shutdown (signal / shutdown command).
                return
            threading.Thread(
                target=self._handle_connection, args=(conn,), daemon=True
            ).start()

    def _on_signal(self, signum: int, _frame: Any) -> None:  # pragma: no cover - signal path
        log.info("received signal %d; shutting down", signum)
        self._shutdown(0)

    def _unlink_stale_socket(self) -> None:
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except OSError:
            pass

    def _shutdown(self, code: int = 0) -> None:
        """Close listener, unlink the socket, and terminate the process."""
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        self._unlink_stale_socket()
        os._exit(code)

    def _cleanup(self) -> None:
        """Non-`os._exit` cleanup for the normal accept-loop-return path."""
        self._unlink_stale_socket()

    # ------------------------------------------------------------------ #
    # Socket helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            try:
                b = conn.recv(remaining)
            except OSError:
                return None
            if not b:
                return None  # peer closed
            chunks.append(b)
            remaining -= len(b)
        return b"".join(chunks)

    def _read_frame(self, conn: socket.socket) -> tuple[dict[str, Any], bytes] | None:
        head = self._recv_exact(conn, _FRAME_STRUCT.size)
        if head is None:
            return None
        hlen, plen = _FRAME_STRUCT.unpack(head)
        if hlen < 0 or plen < 0 or hlen > _MAX_HEADER_LEN or plen > _MAX_PAYLOAD_LEN:
            return None
        hb = self._recv_exact(conn, hlen)
        if hb is None:
            return None
        pb = self._recv_exact(conn, plen) if plen else b""
        if pb is None:
            return None
        try:
            header = json.loads(hb.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(header, dict):
            return None
        return header, pb

    def _send_resp(self, conn: socket.socket, resp: dict[str, Any]) -> None:
        body = json.dumps(resp).encode("utf-8")
        try:
            conn.sendall(_RESP_STRUCT.pack(len(body)) + body)
        except OSError:
            pass  # peer gone; nothing to do

    # ------------------------------------------------------------------ #
    # Connection handler
    # ------------------------------------------------------------------ #
    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                frame = self._read_frame(conn)
                if frame is None:
                    return
                header, payload = frame
                self._touch()
                self._bump_requests()
                cmd = header.get("cmd")
                if cmd == "ping":
                    self._send_resp(conn, {"ok": True})
                elif cmd == "hello":
                    self._send_resp(
                        conn,
                        {
                            "ok": True,
                            "is_real": self.engine.is_real,
                            "dim": self.engine.dim,
                            "model": self.model_name,
                        },
                    )
                elif cmd == "embed_batch":
                    self._handle_embed_batch(conn, header, payload)
                elif cmd == "stats":
                    self._send_resp(
                        conn,
                        {
                            "ok": True,
                            "n_requests": self.n_requests,
                            "uptime_s": round(time.time() - self._start, 3),
                            "is_real": self.engine.is_real,
                            "model": self.model_name,
                        },
                    )
                elif cmd == "shutdown":
                    self._send_resp(conn, {"ok": True})
                    log.info("shutdown command received; exiting")
                    self._shutdown(0)
                else:
                    self._send_resp(conn, {"ok": False, "error": f"unknown command {cmd!r}"})
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("connection handler error: %s", exc)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _handle_embed_batch(
        self, conn: socket.socket, header: dict[str, Any], payload: bytes
    ) -> None:
        shm_name = header.get("shm")
        n = int(header.get("n") or 0)
        dim = int(header.get("dim") or 0)
        seq = header.get("seq")
        try:
            if not isinstance(shm_name, str) or not shm_name:
                raise ValueError("missing shared-memory name in embed_batch header")
            texts = json.loads(payload.decode("utf-8")) if payload else []
            if not isinstance(texts, list):
                raise ValueError("embed_batch payload must be a JSON array of texts")
            if len(texts) != n:
                raise ValueError(f"payload has {len(texts)} texts but n={n}")
            with self._embed_lock:
                vectors = self.engine.embed_batch(texts)
            if len(vectors) != n:
                raise ValueError(f"engine returned {len(vectors)} vectors for n={n}")
            total = n * dim
            if total <= 0:
                raise ValueError("n and dim must be positive")
            shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
            try:
                if len(shm.buf) < total * 4:
                    raise ValueError(
                        f"shm segment too small: {len(shm.buf)} bytes < {total * 4}"
                    )
                import numpy as np

                arr = np.asarray(vectors, dtype=np.float32)
                if arr.shape != (n, dim):
                    raise ValueError(
                        f"engine produced shape {arr.shape}, expected ({n}, {dim})"
                    )
                # Zero-copy write into the client's shared-memory region.
                view = np.ndarray((total,), dtype=np.float32, buffer=shm.buf)
                view[:] = arr.reshape(-1)
            finally:
                try:
                    shm.close()
                except Exception:
                    pass
            self._send_resp(conn, {"ok": True, "n": n, "dim": dim, "seq": seq})
        except Exception as exc:
            log.warning("embed_batch failed: %s", exc)
            self._send_resp(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m isotope_zero.daemon.server``."""
    parser = argparse.ArgumentParser(
        prog="isotope_zero.daemon.server",
        description="Shared-memory ONNX embedding daemon for isotope_zero.",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="embedding model name")
    parser.add_argument("--socket", default=_DEFAULT_SOCKET, help="Unix socket path")
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR, help="model cache directory")
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=_DEFAULT_IDLE_TIMEOUT,
        help="exit after this many idle seconds (default 300)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    IzeroDaemon(
        model_name=args.model,
        socket_path=args.socket,
        cache_dir=args.cache_dir,
        idle_timeout=args.idle_timeout,
    ).run()


if __name__ == "__main__":  # pragma: no cover
    main()
