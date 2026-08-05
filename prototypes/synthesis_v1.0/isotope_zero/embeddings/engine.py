"""Hybrid IPC / in-process embedding engine (Phase 8 synthesis).

Tries the shared-memory embedding daemon first (centralizes the ~360 MB
onnxruntime in ONE process, so a fleet of client workers stays small — the
Phase 7A win: 5-worker RAM 776 MB -> 402 MB, 100 % recall parity). If the
daemon socket is unavailable, refused, OR a call errors mid-flight, it
TRANSPARENTLY and SILENTLY falls back to an in-process local ONNX session
(``EmbeddingEngine``). The caller always gets a vector — never an exception
for a transport failure.

The fallback is deliberately silent + lazy: the in-process
``EmbeddingEngine`` is only constructed the FIRST time the daemon path fails,
so onnxruntime stays out of the client process entirely while the daemon is
up (the whole point of the daemon). A single ``log.warning`` records the
transition; subsequent fallback calls re-use the already-built engine with no
further logging.

``HybridEmbeddingEngine`` is a duck-typed drop-in for both ``DaemonClient``
and ``EmbeddingEngine``: ``.embed_text``, ``.embed_batch``, ``.is_real``,
``.dim`` — accepted anywhere either is (``MemoryStore``, ``QueryRouter``,
``Consolidator``, ``IsotopeZero``).
"""
from __future__ import annotations

import atexit
import logging
from typing import Any

log = logging.getLogger("isotope_zero.embeddings.hybrid")

_DEFAULT_SOCKET = "/tmp/izero.sock"
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIM = 384


class HybridEmbeddingEngine:
    """Daemon-first embedding engine with silent in-process fallback.

    Parameters
    ----------
    model_name:
        Embedding model shorthand (default ``"all-MiniLM-L6-v2"``). Forwarded
        to whichever backend ends up active.
    socket_path:
        Unix domain socket the daemon listens on.
    spawn_daemon:
        When True (default), the ``DaemonClient`` may spawn the daemon itself
        if the socket is missing. Set False to force the in-process path
        (useful for tests / CI / sandboxed runs where spawning is undesirable).
    connect_timeout:
        Seconds to wait for a spawned daemon to become reachable before
        falling back to in-process.
    dim:
        Embedding dimension (default 384). Matches the model; used for the
        empty-input zero-vector contract and the in-process fallback engine.
    cache_dir:
        Local model cache dir forwarded to the in-process ``EmbeddingEngine``
        when the fallback is taken.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        socket_path: str = _DEFAULT_SOCKET,
        spawn_daemon: bool = True,
        connect_timeout: float = 10.0,
        dim: int = _DEFAULT_DIM,
        cache_dir: str = ".isotope_zero_cache",
    ) -> None:
        self.model_name = model_name
        self.socket_path = socket_path
        self.spawn_daemon = spawn_daemon
        self.connect_timeout = float(connect_timeout)
        self._dim = int(dim)
        self.cache_dir = cache_dir

        # Lazily-populated backends. Exactly one of these is the active path
        # once construction settles; the other stays None.
        self._daemon: Any = None
        self._in_process: Any = None

        # Active mode: "daemon" | "in_process" | "fallback_pseudo". Set once
        # during __init__ and again if a daemon call later forces a switch.
        self._mode: str = "fallback_pseudo"
        # Tracks whether we've already logged the fallback transition (avoid
        # spamming the log on every call once we've switched).
        self._fallback_logged: bool = False

        # Try the daemon path first. A failure here (socket refused + spawn
        # failed) is NOT fatal — it just means we start in-process / fallback.
        try:
            from isotope_zero.daemon.client import DaemonClient

            self._daemon = DaemonClient(
                model_name=model_name,
                socket_path=socket_path,
                spawn=spawn_daemon,
                connect_timeout=connect_timeout,
            )
            if self._daemon.is_real:
                self._mode = "daemon"
            else:
                # Daemon reachable but reports not-real (e.g. its own fallback
                # engaged). Still usable for parity, but record the real mode
                # honestly from the daemon's own state.
                self._mode = "daemon"
        except Exception as exc:  # noqa: BLE001 — daemon unavailability must never crash
            log.warning(
                "embedding daemon unavailable (%s); using in-process path.",
                exc,
            )
            self._daemon = None
            # Fall through: the first embed call will lazily build in-process.
            self._mode = "in_process"

        # If the daemon didn't come up AND spawn was disabled, we may already
        # know we're heading for in-process; build it lazily on first use so
        # importing this module never loads onnxruntime.
        atexit.register(self.close)

    # ------------------------------------------------------------------ #
    # Public API (duck-typed embedder)
    # ------------------------------------------------------------------ #
    @property
    def is_real(self) -> bool:
        """True if the active backend produces real ONNX vectors."""
        if self._mode == "daemon" and self._daemon is not None:
            return bool(self._daemon.is_real)
        if self._mode in ("in_process", "fallback_pseudo") and self._in_process is not None:
            return bool(self._in_process.is_real)
        return False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def mode(self) -> str:
        """Active backend: ``"daemon"`` | ``"in_process"`` | ``"fallback_pseudo"``."""
        return self._mode

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string, L2-normalized. Empty input -> zero vector.

        Routes to the daemon when available; on ANY failure silently switches
        to the in-process engine and retries once. Never raises for a
        transport failure — the caller always gets a vector.
        """
        if text == "":
            return [0.0] * self._dim
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Returns ``list[list[float]]``, input-ordered.

        Daemon-first with one transparent fallback to in-process. Never raises
        for a transport failure (the in-process engine's own fallback to
        deterministic pseudo-embeddings is the final safety net).
        """
        if not texts:
            return []

        # Daemon path.
        if self._mode == "daemon" and self._daemon is not None:
            try:
                return self._daemon.embed_batch(texts)
            except Exception as exc:  # noqa: BLE001 — ConnectionError, BrokenPipe, timeout, ...
                if not self._fallback_logged:
                    log.warning(
                        "daemon embed failed (%s); switching to in-process "
                        "fallback for this and subsequent calls.",
                        exc,
                    )
                    self._fallback_logged = True
                self._switch_to_in_process()

        # In-process path (also the destination after a daemon switch).
        if self._in_process is None:
            self._build_in_process()
        return self._in_process.embed_batch(texts)

    # ------------------------------------------------------------------ #
    # Internal: lazy in-process construction + mode switching
    # ------------------------------------------------------------------ #
    def _switch_to_in_process(self) -> None:
        """Drop the daemon and move to the in-process engine (lazy build)."""
        # Best-effort close of the dead daemon socket; ignore failures.
        try:
            if self._daemon is not None:
                self._daemon.close()
        except Exception:  # noqa: BLE001
            pass
        self._daemon = None
        self._mode = "in_process"

    def _build_in_process(self) -> None:
        """Lazily construct the in-process ``EmbeddingEngine``."""
        from isotope_zero.embeddings.onnx_embed import EmbeddingEngine

        self._in_process = EmbeddingEngine(
            model_name=self.model_name,
            dim=self._dim,
            cache_dir=self.cache_dir,
        )
        # If onnxruntime is absent the engine runs its own deterministic
        # pseudo-embedding fallback; record that honestly.
        if not self._in_process.is_real:
            self._mode = "fallback_pseudo"
        else:
            self._mode = "in_process"

    def close(self) -> None:
        """Release the daemon socket if owned. Idempotent."""
        if self._daemon is not None:
            try:
                self._daemon.close()
            except Exception:  # noqa: BLE001
                pass
            self._daemon = None
        # EmbeddingEngine has no explicit close; onnxruntime releases on GC.


def _smoke() -> None:  # pragma: no cover
    eng = HybridEmbeddingEngine(spawn_daemon=False)
    a = eng.embed_text("the user likes rust and tea")
    b = eng.embed_text("the user likes rust and tea")
    c = eng.embed_text("completely unrelated nonsense zzz")
    import math as _m

    def dot(x: list[float], y: list[float]) -> float:
        return sum(xi * yi for xi, yi in zip(x, y))

    print(f"mode={eng.mode} is_real={eng.is_real} dim={eng.dim}")
    print(f"first3(a)={a[:3]}")
    print(f"cos(a,b)={dot(a,b):.4f}  cos(a,c)={dot(a,c):.4f}")


if __name__ == "__main__":  # pragma: no cover
    _smoke()
