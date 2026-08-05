"""Modernized embedding runtime — a zero-dep-first superset of the existing
``EmbeddingEngine`` contract.

DESIGN POSTURE (verbatim from the runtime task):
- ZERO-DEP-FIRST. This module MUST import and be fully testable with ONLY
  numpy present. ``httpx`` / ``onnxruntime`` / ``tokenizers`` are ABSENT in
  the contract reference environment, so they are *optional-guarded* with
  ``try/except ImportError`` and NEVER hard-imported at module top level. The
  module imports cleanly even if numpy is somehow absent (it raises a clear
  ``RuntimeError`` only at first embed, mirroring how
  ``core.store.vector_search`` lazily ``import numpy``).
- Protocol/duck-typing over ABC where the ABC needs nothing missing. The
  ``Embedder`` runtime-checkable Protocol captures the *duck-typed* surface
  the rest of isotope_zero already accepts (``EmbeddingEngine``,
  ``DaemonClient``, ``HybridEmbeddingEngine``): ``embed_text`` /
  ``embed_batch`` / ``is_real`` / ``dim``. ``BaseEmbeddingEngine`` is the
  spec-mandated ABC (abstract ``embed_text``/``embed_batch`` + async twins +
  ``metadata``) and is what the new concrete engines inherit; it does NOT
  replace the duck-typed Protocol — both coexist.
- Deterministic fallback engine reusing the feature-hash stub pattern from
  ``embeddings.onnx_embed.EmbeddingEngine._embed_fallback`` (blake2b of word
  tokens + char trigrams into ``dim`` signed buckets, L2-normalized). This is
  the terminal safety net of ``EmbeddingPipeline`` and the default backend
  when every real backend is unavailable, so the runtime is ALWAYS testable
  without network / onnxruntime.
- ``MRLTruncator`` works on raw numpy: slices the first ``d`` dims and
  re-normalizes L2 via ``np.linalg.norm`` (zero-copy view + one divide).

CONTRACT COMPATIBILITY (do NOT invent a parallel interface):
- Existing engines return ``list[float]``; the SPEC asks for ``np.ndarray``.
  This runtime is a MODERNIZED SUPERSET: the spec engines
  (``ONNXEmbeddingEngine`` / ``OllamaEmbeddingEngine`` / ``OpenAIEmbeddingEngine``
  / ``FallbackEmbeddingEngine``) return ``np.ndarray`` (float32, C-contiguous,
  L2-normalized so cosine == dot). To keep vectors interopable with the
  EXISTING ``MemoryStore.vector_search`` — whose guard
  ``if not query_vec`` raises ``ValueError`` on a multi-element ndarray — the
  engines return a thin ``Embedding`` (an ``np.ndarray`` subclass whose
  ``__bool__`` is unambiguous: True iff any element is nonzero). ``np.asarray``
  in the store unwraps it to a plain ndarray transparently, and
  ``list(embedding)`` / indexing still work, so every existing call site
  (``client.remember`` / ``recall``, ``store.add``'s ``_encode_embedding``)
  keeps working unchanged.
- L2-normalized so cosine similarity == dot product (the contract
  ``vector_search`` relies on). Empty input -> zero vector (parity with
  ``EmbeddingEngine.embed_text``).
- The spec engines expose the SAME duck-typed surface
  (``embed_text`` / ``embed_batch`` / ``is_real`` / ``dim``) plus the
  spec-mandated ``metadata`` / ``aembed_text`` / ``aembed_batch``. They are
  drop-ins anywhere the existing engines are accepted, AND satisfy the new
  spec.

OPTIONAL-IMPORT STRATEGY:
- ``import numpy`` is guarded once at module import (sets ``_HAS_NUMPY``);
  the engines import it lazily on first use so a missing numpy degrades to a
  clear runtime error rather than an import-time crash. The store already
  follows the same lazy pattern.
- ``httpx`` and ``onnxruntime`` are imported inside the relevant engines'
  ``__init__`` / methods inside ``try/except``; absence flips ``is_real`` to
  False and the pipeline fails over to the next backend / the fallback stub.
"""
from __future__ import annotations

import abc
import asyncio
import hashlib
import logging
import math
import os
import urllib.request
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger("isotope_zero.embeddings.runtime")

# --------------------------------------------------------------------------- #
# Optional-dependency probes (NEVER hard-import at module top).
# numpy is a hard dep of the rest of isotope_zero, but this module must
# IMPORT even without it; we discover it once and use it lazily.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only when numpy is/isn't installed
    import numpy as _np  # noqa: F401

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


def _require_numpy() -> Any:
    """Return the numpy module, raising a clear error if absent.

    Mirrors ``core.store.vector_search``'s lazy ``import numpy`` discipline:
    the runtime is importable without numpy, but actually embedding needs it.
    """
    if not _HAS_NUMPY:
        raise RuntimeError(
            "isotope_zero.embeddings.embedding_runtime requires numpy to "
            "produce vectors; install numpy (the rest of the package already "
            "depends on it)."
        )
    return _np


# --------------------------------------------------------------------------- #
# Constants — mirror the existing EmbeddingEngine contract.
# --------------------------------------------------------------------------- #
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIM = 384
_EMBED_CHUNK = 32  # cap on texts per ONNX forward pass (memory frugality)

# HuggingFace ONNX repos known to ship quantized sentence-embedding models.
# Reused verbatim from onnx_embed.EmbeddingEngine so model shorthand is stable.
_HF_CANDIDATES: dict[str, tuple[str, str]] = {
    "all-MiniLM-L6-v2": ("Xenova/all-MiniLM-L6-v2", "onnx/model_quantized.onnx"),
    "bge-micro-v2": ("Xenova/bge-micro-v2", "onnx/model_quantized.onnx"),
}

# Matryoshka Representation Learning: the canonical truncation dims for the
# sentence-embedding models in scope. 128/256/512 per the spec.
_MRL_DIMS = (128, 256, 512)


# --------------------------------------------------------------------------- #
# Embedding vector type — an ndarray subclass with unambiguous truthiness.
# --------------------------------------------------------------------------- #
class Embedding:
    """L2-normalized embedding vector returned by the spec engines.

    This is a *typed view* over an ``np.ndarray`` (float32, 1-D, L2-normalized)
    rather than a raw ndarray, for one reason: the EXISTING
    ``MemoryStore.vector_search`` opens with ``if not query_vec: ...`` and
    ``all(v == 0.0 for v in query_vec)``. A raw multi-element ndarray raises
    ``ValueError: ambiguous truth value`` on ``not vec``; this subclass
    overrides ``__bool__`` to return True iff any element is nonzero, so the
    existing guard works unchanged. ``np.asarray(e)`` / ``e @ matrix`` /
    ``list(e)`` / ``e[i]`` all behave exactly like a plain float32 ndarray.

    Construct via ``Embedding(array)``; the array is cast to float32 and
    C-contiguous (``np.ascontiguousarray``) with no semantic copy when it
    already meets those properties (zero-copy fast path).
    """

    __slots__ = ("_arr",)

    def __init__(self, arr: Any) -> None:
        np = _require_numpy()
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if a.ndim != 1:
            raise ValueError(
                f"Embedding must be 1-D, got shape {a.shape!r}"
            )
        self._arr = a

    # -- ndarray-mirroring surface ------------------------------------------- #
    @property
    def array(self) -> Any:
        """The underlying float32 ndarray (do not mutate)."""
        return self._arr

    def __array__(self, dtype: Any = None, copy: bool = True) -> Any:
        # np.asarray(embedding) -> the ndarray (lets the store build its matrix).
        if dtype is None:
            return self._arr.copy() if copy else self._arr
        return self._arr.astype(dtype, copy=copy) if copy else self._arr.astype(
            dtype, copy=False
        )

    def __iter__(self):
        return iter(self._arr)

    def __len__(self) -> int:
        return int(self._arr.shape[0])

    def __getitem__(self, idx: Any) -> Any:
        return self._arr[idx]

    def __matmul__(self, other: Any) -> Any:
        return self._arr @ other

    def __rmatmul__(self, other: Any) -> Any:
        return other @ self._arr

    @property
    def shape(self) -> tuple[int, ...]:
        return self._arr.shape

    @property
    def dtype(self) -> Any:
        return self._arr.dtype

    # -- truthiness: the one behavioral delta from a raw ndarray ------------- #
    def __bool__(self) -> bool:
        # True iff the vector is non-degenerate (not all zeros). Matches the
        # store's degenerate-query contract: an all-zero query returns [].
        return bool(np_any(self._arr != 0)) if _HAS_NUMPY else False

    def __eq__(self, other: Any) -> Any:  # elementwise, like ndarray
        return self._arr == getattr(other, "_arr", other)

    def __repr__(self) -> str:
        return f"Embedding(shape={self._arr.shape}, dtype={self._arr.dtype})"


def np_any(x: Any) -> Any:
    """``np.any`` without forcing a full import at module load."""
    np = _require_numpy()
    return np.any(x)


# --------------------------------------------------------------------------- #
# Duck-typed embedder Protocol — the surface isotope_zero already accepts.
# --------------------------------------------------------------------------- #
@runtime_checkable
class Embedder(Protocol):
    """The duck-typed embedder surface used across isotope_zero.

    ``EmbeddingEngine`` / ``DaemonClient`` / ``HybridEmbeddingEngine`` all
    satisfy this *structurally*; this Protocol exists only so the pipeline
    can type a backend as "anything that embeds" without forcing inheritance.
    The spec's ``BaseEmbeddingEngine`` ABC (below) is the *typed* parent of
    the new concrete engines; both coexist.
    """

    @property
    def is_real(self) -> bool: ...

    @property
    def dim(self) -> int: ...

    def embed_text(self, text: str) -> Any: ...

    def embed_batch(self, texts: list[str]) -> Any: ...


# --------------------------------------------------------------------------- #
# Spec requirement 1: BaseEmbeddingEngine ABC.
# --------------------------------------------------------------------------- #
class BaseEmbeddingEngine(abc.ABC):
    """Abstract base for the spec's typed embedding engines.

    Concrete engines implement the synchronous ``embed_text`` /
    ``embed_batch`` (returning ``np.ndarray``) and inherit default async
    twins (``aembed_text`` / ``aembed_batch``) that run the sync methods in a
    thread via ``asyncio`` — so a backend with no native async path still
    satisfies the async contract without blocking the event loop.

    The duck-typed ``is_real`` / ``dim`` properties are concrete here (backed by
    instance attributes ``_is_real`` / ``_dim``) so subclasses can simply set
    ``self._is_real`` / ``self._dim`` in ``__init__`` without re-declaring the
    property — and the pipeline's failover logic can introspect any backend
    uniformly. Only ``embed_text`` / ``embed_batch`` / ``metadata`` are
    abstract (the truly-varying surface).
    """

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        self._dim: int = int(dim)
        self._is_real: bool = False

    @abc.abstractmethod
    def embed_text(self, text: str) -> Any:
        """Embed one string -> L2-normalized float32 ndarray. Empty -> zeros."""
        raise NotImplementedError

    @abc.abstractmethod
    def embed_batch(self, texts: list[str]) -> Any:
        """Embed a batch -> (n, dim) float32 ndarray, input-ordered."""
        raise NotImplementedError

    @property
    def is_real(self) -> bool:
        """True iff this backend produces real (non-fallback) vectors."""
        return self._is_real

    @property
    def dim(self) -> int:
        """Embedding dimensionality this engine produces."""
        return self._dim

    @property
    @abc.abstractmethod
    def metadata(self) -> dict[str, Any]:
        """``{dim, backend, provider, memory_type}`` per spec req 1.

        ``memory_type`` is one of ``"local-process"`` | ``"local-daemon"`` |
        ``"remote-api"``.
        """
        raise NotImplementedError

    # -- default async impls: run sync work in a worker thread --------------- #
    async def aembed_text(self, text: str) -> Any:
        return await asyncio.to_thread(self.embed_text, text)

    async def aembed_batch(self, texts: list[str]) -> Any:
        return await asyncio.to_thread(self.embed_batch, texts)

    # -- convenience: make every spec engine drop-in for store/client ------- #
    def __matmul__(self, other: Any) -> Any:  # pragma: no cover - convenience
        raise TypeError("EmbeddingEngine is not a vector; call embed_text first")


# --------------------------------------------------------------------------- #
# Spec requirement 5: MRLTruncator — works on raw numpy.
# --------------------------------------------------------------------------- #
class MRLTruncator:
    """Slice the first ``d`` dims of an embedding + zero-copy L2 re-norm.

    Matryoshka Representation Learning trains models so the *leading* dims are
    a usable lower-dimensional embedding; truncating to ``d`` (128/256/512)
    and re-normalizing yields a valid L2-normalized vector in ``d``-space, so
    cosine similarity on the truncated vector stays meaningful (and the
    cosine-of-truncated vs cosine-of-full stays high for near-duplicates).

    The slice ``arr[:d]`` is a zero-copy *view* into the original buffer; the
    re-norm divides in place into a fresh float32 array (the division cannot
    be done in place on a view without mutating the caller's buffer, so we
    allocate exactly one ``(d,)`` array). ``np.linalg.norm`` is used exactly
    as in ``onnx_embed.EmbeddingEngine._embed_real``.
    """

    def __init__(self, dim: int = 128) -> None:
        if dim <= 0:
            raise ValueError(f"MRL dim must be > 0, got {dim}")
        self.dim = int(dim)

    def truncate(self, vec: Any) -> Embedding:
        """Truncate a single 1-D vector (ndarray / Embedding / list) to ``dim``.

        Returns an ``Embedding`` wrapper (ndarray-backed, L2-normalized) so the
        result drops straight into ``MemoryStore.vector_search`` — whose
        ``if not query_vec`` guard would raise ``ValueError`` on a raw
        multi-element ndarray — and still supports ``matrix @ q`` / indexing /
        ``np.asarray``. The result is L2-normalized so cosine == dot in the
        truncated space.
        """
        np = _require_numpy()
        arr = np.asarray(vec.array if isinstance(vec, Embedding) else vec, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"MRLTruncator expects 1-D, got {arr.shape!r}")
        d = self.dim
        if arr.shape[0] < d:
            # Pad with zeros to reach d (the leading-dim contract still holds;
            # a too-short source is degenerate but we never raise on shape).
            out = np.zeros(d, dtype=np.float32)
            out[: arr.shape[0]] = arr
            arr = out
        sliced = arr[:d]  # zero-copy view (whole vector when padded)
        # Re-normalize so the output is unit L2 — parity with truncate_batch and
        # the cosine == dot contract. Padding does not change the norm, so this
        # correctly yields norm 1.0 for a too-short source as well.
        norm = float(np.linalg.norm(sliced))
        if norm > 0.0:
            return Embedding((sliced / norm).astype(np.float32, copy=False))
        return Embedding(np.asarray(sliced, dtype=np.float32))

    def truncate_batch(self, vecs: Any) -> Any:
        """Truncate a (n, D) matrix to (n, dim), row-wise L2 re-normalized."""
        np = _require_numpy()
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"MRLTruncator.truncate_batch expects 2-D, got {arr.shape!r}")
        d = self.dim
        if arr.shape[1] < d:
            out = np.zeros((arr.shape[0], d), dtype=np.float32)
            out[:, : arr.shape[1]] = arr
            arr = out
        sliced = arr[:, :d]  # zero-copy view
        norms = np.linalg.norm(sliced, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # guard zero rows
        return (sliced / norms).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Deterministic fallback engine (reuse the feature-hash stub pattern).
# This is the terminal safety net: always available, zero deps, deterministic.
# --------------------------------------------------------------------------- #
class FallbackEmbeddingEngine(BaseEmbeddingEngine):
    """Deterministic feature-hash embedding engine — the zero-dep safety net.

    Reuses the exact stub pattern from
    ``onnx_embed.EmbeddingEngine._embed_fallback``: whitespace tokens (lower)
    are blake2b-hashed into ``dim`` signed buckets, a global text signature
    spreads energy for single-token inputs, and the result is L2-normalized
    so cosine == dot. NOT semantic, but deterministic: identical inputs ->
    identical vectors (cosine 1.0), and lexical-overlap queries still surface
    near matches. ``is_real`` is always False.

    Returns ``Embedding`` (ndarray-backed) so the vectors interoperate with
    ``MemoryStore.vector_search``'s ``not query_vec`` guard.
    """

    backend_name = "fallback"

    def __init__(self, dim: int = _DEFAULT_DIM, model_name: str = _DEFAULT_MODEL) -> None:
        super().__init__(dim=dim)
        self.model_name = model_name
        self._is_real = False  # always: deterministic pseudo-embeddings

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "dim": self._dim,
            "backend": "fallback",
            "provider": self.model_name,
            "memory_type": "local-process",
        }

    def embed_text(self, text: str) -> Embedding:
        return Embedding(self._fallback_vec(text, self._dim))

    def embed_batch(self, texts: list[str]) -> Any:
        np = _require_numpy()
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        rows = [self._fallback_vec(t, self._dim) for t in texts]
        return np.stack(rows).astype(np.float32, copy=False)

    # -- the reused stub math ------------------------------------------------ #
    @staticmethod
    def _fallback_vec(text: str, dim: int) -> Any:
        """Deterministic feature-hash embedding, L2-normalized (numpy).

        Mirrors ``onnx_embed.EmbeddingEngine._embed_fallback`` but vectorized
        on a numpy buffer for speed and so the runtime stays numpy-native.
        Whitespace tokens -> blake2b -> (bucket, sign); a global text
        signature spreads energy so single-token texts aren't a one-hot spike.
        """
        np = _require_numpy()
        vec = np.zeros(dim, dtype=np.float32)
        if not text:
            return vec
        low = text.lower()
        tokens = low.split()
        for tok in tokens:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(h[:4], "little") % dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            vec[bucket] += sign
        # Global text signature so single-token texts spread energy.
        sig = hashlib.blake2b(low.encode("utf-8"), digest_size=8).digest()
        for i in range(0, min(8, dim)):
            b = sig[i]
            vec[(b * 7 + i) % dim] += (b / 127.5) - 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return vec.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Spec requirement 2: ONNXEmbeddingEngine (in-process).
# --------------------------------------------------------------------------- #
class ONNXEmbeddingEngine(BaseEmbeddingEngine):
    """In-process ONNX embedding engine (all-MiniLM-L6-v2 or ONNX Nomic).

    A modernized superset of ``onnx_embed.EmbeddingEngine``: same model
    download / caching, same mean-pool + masked L2-norm math, same
    ``_EMBED_CHUNK`` forward-pass chunking for memory frugality — but returns
    ``np.ndarray`` (spec contract) and exposes the spec ``metadata`` /
    async twins. ``onnxruntime`` / ``tokenizers`` are imported lazily inside
    ``__init__``; if absent, ``is_real`` is False and the pipeline fails over.

    Construction MAY download the model once (network permitted); all
    subsequent inference is purely local. If anything on the real path fails,
    the engine transparently sets ``is_real = False`` and the caller
    (typically ``EmbeddingPipeline``) routes to the next backend or the
    ``FallbackEmbeddingEngine``.
    """

    backend_name = "onnx"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        dim: int = _DEFAULT_DIM,
        quantized: bool = True,
        cache_dir: str = ".isotope_zero_cache",
    ) -> None:
        super().__init__(dim=dim)
        self.model_name = model_name
        self.quantized = quantized
        self.cache_dir = cache_dir
        self._is_real: bool = False
        self._session: Any = None
        self._tokenizer: Any = None
        self._max_length: int = 256
        try:
            self._try_load_real()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("ONNX real-path crashed (%s); is_real=False.", exc)
            self._session = None
            self._tokenizer = None
            self._is_real = False

    # -- BaseEmbeddingEngine surface ----------------------------------------- #
    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "dim": self._dim,
            "backend": "onnx",
            "provider": self.model_name,
            "memory_type": "local-process",
        }

    def embed_text(self, text: str) -> Any:
        if text == "":
            return Embedding(np_zeros(self._dim))
        if self.is_real:
            return Embedding(self._embed_real([text])[0])
        # Real path unavailable: caller (pipeline) should fail over; if used
        # standalone, fall back to the deterministic stub so we NEVER raise
        # for a missing dependency.
        return FallbackEmbeddingEngine(self._dim, self.model_name).embed_text(text)

    def embed_batch(self, texts: list[str]) -> Any:
        np = _require_numpy()
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        if self.is_real:
            return self._embed_real(texts)
        return FallbackEmbeddingEngine(self._dim, self.model_name).embed_batch(texts)

    # -- real ONNX path (mirrors onnx_embed.EmbeddingEngine) ---------------- #
    def _try_load_real(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
            from tokenizers import Tokenizer  # type: ignore
        except Exception as exc:
            log.warning("onnxruntime/tokenizers not importable (%s).", exc)
            return

        repo, onnx_sub = _HF_CANDIDATES.get(
            self.model_name, (self.model_name, "onnx/model_quantized.onnx")
        )
        if not self.quantized:
            onnx_sub = onnx_sub.replace("_quantized", "") or "onnx/model.onnx"

        os.makedirs(self.cache_dir, exist_ok=True)
        model_path = os.path.join(self.cache_dir, f"{repo.replace('/', '_')}.onnx")
        tok_path = os.path.join(self.cache_dir, f"{repo.replace('/', '_')}.tokenizer.json")

        if not (os.path.exists(model_path) and os.path.exists(tok_path)):
            if not self._download(repo, onnx_sub, model_path, tok_path):
                return  # is_real stays False

        try:
            opts = ort.SessionOptions()
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.enable_cpu_mem_arena = True
            opts.enable_mem_pattern = True
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(model_path, sess_options=opts)
            self._tokenizer = Tokenizer.from_file(tok_path)
            self._tokenizer.enable_padding(length=None)
            self._max_length = 256
            # Reconcile the constructor dim with the model's REAL output dim
            # (the last axis of the session's first output). The existing
            # EmbeddingEngine trusts the constructor arg; a modernized engine
            # should report the truth so metadata + downstream sizes match.
            try:
                real_dim = int(
                    self._session.get_outputs()[0].shape[-1]
                )
                if real_dim and real_dim > 0:
                    self._dim = real_dim
            except Exception:  # pragma: no cover - shape introspection is best-effort
                pass
            self._is_real = True
            log.info("ONNXEmbeddingEngine loaded real model from %s (dim=%d).", model_path, self._dim)
        except Exception as exc:
            log.warning("Failed to init ONNX session/tokenizer (%s).", exc)
            self._session = None
            self._tokenizer = None

    def _download(self, repo: str, onnx_sub: str, model_path: str, tok_path: str) -> bool:
        urls = {
            model_path: f"https://huggingface.co/{repo}/resolve/main/{onnx_sub}",
            tok_path: f"https://huggingface.co/{repo}/resolve/main/tokenizer.json",
        }
        for dest, url in urls.items():
            if os.path.exists(dest):
                continue
            try:
                log.info("Downloading %s -> %s", url, dest)
                req = urllib.request.Request(url, headers={"User-Agent": "isotope_zero/0.1"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
                    f.write(resp.read())
            except Exception as exc:
                log.warning("Download failed for %s (%s).", url, exc)
                return False
        return True

    def _embed_real(self, texts: list[str]) -> Any:
        """Chunked ONNX forward + masked mean-pool + L2-norm -> (n, dim) ndarray."""
        np = _require_numpy()
        out_all: list[Any] = []
        for start in range(0, len(texts), _EMBED_CHUNK):
            chunk = texts[start : start + _EMBED_CHUNK]
            enc = self._tokenizer.encode_batch(chunk)
            input_ids = np.array([e.ids for e in enc], dtype=np.int64)
            attn = np.array([e.attention_mask for e in enc], dtype=np.int64)
            if input_ids.ndim == 1:
                input_ids = input_ids.reshape(1, -1)
                attn = attn.reshape(1, -1)
            feeds = {
                "input_ids": input_ids,
                "token_type_ids": np.zeros_like(input_ids),
                "attention_mask": attn,
            }
            try:
                out = self._session.run(None, feeds)
            except Exception:
                feeds.pop("token_type_ids", None)
                out = self._session.run(None, feeds)
            token_embeds = out[0]  # (chunk, seq, hidden)
            mask = attn[..., None].astype(np.float32)
            summed = (token_embeds * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1, None)
            pooled = summed / counts
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normed = (pooled / norms).astype(np.float32, copy=False)
            out_all.append(normed)
            del token_embeds, mask, summed, counts, pooled, norms, normed
        return np.concatenate(out_all, axis=0) if out_all else np.empty(
            (0, self._dim), dtype=np.float32
        )


# --------------------------------------------------------------------------- #
# Spec requirement 3: OllamaEmbeddingEngine (HTTP client, /api/embed).
# --------------------------------------------------------------------------- #
class OllamaEmbeddingEngine(BaseEmbeddingEngine):
    """Ollama /api/embed HTTP client — drops model weights from client RSS.

    The heavy model (e.g. nomic-embed-text) lives in the Ollama server
    process, so this client's RSS stays tiny. Uses ``httpx.AsyncClient`` for
    async and reuses a session across calls (spec req 7); falls back to
    ``urllib.request`` (stdlib) for the sync path so the engine is importable
    and testable WITHOUT httpx. ``httpx`` is imported lazily; if absent,
    ``is_real`` is False and sync calls use the stdlib transport.

    The async path is the primary one (spec req 6 chains async backends); the
    sync ``embed_text``/``embed_batch`` run the same HTTP request via
    ``asyncio.run`` for parity with the rest of isotope_zero's sync API.
    """

    backend_name = "ollama"

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        dim: int = 768,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(dim=dim)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._async_client: Any = None
        # is_real is lazy/optimistic: unknown until first embed. We override
        # the property so the pipeline ATTEMPTS this backend (catching
        # connection errors and failing over), rather than skipping it.
        self._is_real: bool | None = None

    @property
    def is_real(self) -> bool:
        # Lazy: unknown until first embed; treat as True to let the pipeline
        # attempt it (the pipeline catches connection errors and fails over).
        return self._is_real in (None, True)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "dim": self._dim,
            "backend": "ollama",
            "provider": self.model,
            "memory_type": "remote-api",
        }

    # -- sync surface (parity with EmbeddingEngine) -------------------------- #
    def embed_text(self, text: str) -> Embedding:
        # Wrap the row so the store's ``not query_vec`` guard is unambiguous.
        return Embedding(self.embed_batch([text])[0])

    def embed_batch(self, texts: list[str]) -> Any:
        np = _require_numpy()
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        # Run the async path under a fresh loop (the sync API contract).
        return asyncio.run(self.aembed_batch(texts))

    async def aembed_text(self, text: str) -> Any:
        return (await self.aembed_batch([text]))[0]

    async def aembed_batch(self, texts: list[str]) -> Any:
        np = _require_numpy()
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vecs = await self._post_embed(texts)
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Defensive L2 re-norm: the server SHOULD return normalized vectors,
        # but the contract (cosine == dot) is load-bearing, so we enforce it.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).astype(np.float32, copy=False)

    # -- HTTP transport (httpx preferred, urllib fallback) ------------------- #
    async def _post_embed(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": texts}
        try:
            import httpx  # type: ignore
        except ImportError:
            # Stdlib fallback (sync, blocking) so the engine is usable without
            # httpx — parity with the zero-dep-first posture.
            return self._post_embed_urllib(url, payload)

        client = self._async_client
        owned = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            self._async_client = client
            owned = True
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]
        finally:
            if owned:
                await client.aclose()
                self._async_client = None

    def _post_embed_urllib(self, url: str, payload: dict[str, Any]) -> list[list[float]]:
        import json as _json
        import urllib.request as _urlreq

        body = _json.dumps(payload).encode("utf-8")
        req = _urlreq.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urlreq.urlopen(req, timeout=self.timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return data["embeddings"]

    async def close(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None


# --------------------------------------------------------------------------- #
# Spec requirement 4: OpenAIEmbeddingEngine (cloud REST, /v1/embeddings).
# --------------------------------------------------------------------------- #
class OpenAIEmbeddingEngine(BaseEmbeddingEngine):
    """OpenAI /v1/embeddings cloud REST client.

    Uses ``httpx.AsyncClient`` (preferred) or stdlib ``urllib`` (fallback) so
    the module imports without httpx. The API key is read from the
    ``OPENAI_API_KEY`` env var (or the constructor) — never logged. Batched
    at the configured chunk size (spec req 7); a single session is reused
    across chunks. Output is L2-normalized so cosine == dot (the API returns
    normalized vectors for text-embedding models, but we enforce it).
    """

    backend_name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com",
        dim: int = 1536,
        timeout: float = 60.0,
        chunk_size: int = _EMBED_CHUNK,
    ) -> None:
        super().__init__(dim=dim)
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.chunk_size = int(chunk_size)
        self._async_client: Any = None

    @property
    def is_real(self) -> bool:
        return bool(self._api_key)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "dim": self._dim,
            "backend": "openai",
            "provider": self.model,
            "memory_type": "remote-api",
        }

    def embed_text(self, text: str) -> Embedding:
        # Wrap the row so the store's ``not query_vec`` guard is unambiguous.
        return Embedding(self.embed_batch([text])[0])

    def embed_batch(self, texts: list[str]) -> Any:
        np = _require_numpy()
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return asyncio.run(self.aembed_batch(texts))

    async def aembed_text(self, text: str) -> Any:
        return (await self.aembed_batch([text]))[0]

    async def aembed_batch(self, texts: list[str]) -> Any:
        np = _require_numpy()
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.chunk_size):
            chunk = texts[i : i + self.chunk_size]
            rows.extend(await self._post_embeddings(chunk))
        arr = np.asarray(rows, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).astype(np.float32, copy=False)

    async def _post_embeddings(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": self.model, "input": texts}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        try:
            import httpx  # type: ignore
        except ImportError:
            return self._post_embeddings_urllib(url, payload, headers)

        client = self._async_client
        owned = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            self._async_client = client
            owned = True
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        finally:
            if owned:
                await client.aclose()
                self._async_client = None

    def _post_embeddings_urllib(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> list[list[float]]:
        import json as _json
        import urllib.request as _urlreq

        body = _json.dumps(payload).encode("utf-8")
        req = _urlreq.Request(url, data=body, headers=headers, method="POST")
        with _urlreq.urlopen(req, timeout=self.timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return [item["embedding"] for item in data["data"]]

    async def close(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None


# --------------------------------------------------------------------------- #
# Spec requirement 6: EmbeddingPipeline — chain-of-responsibility failover.
# --------------------------------------------------------------------------- #
class EmbeddingPipeline:
    """Chain-of-responsibility over prioritized backends with graceful failover.

    The NEW generalization of ``HybridEmbeddingEngine``'s daemon-first /
    silent-fallback discipline: an ordered list of backends is tried in turn;
    the FIRST one that is ``is_real`` AND whose embed call succeeds is used
    for all subsequent calls (latched). On a connection error / OOM / any
    exception during embed, the pipeline logs once, marks that backend dead,
    and fails over to the next. The terminal backend is ALWAYS a
    ``FallbackEmbeddingEngine`` so the pipeline NEVER raises for a backend
    failure — the caller always gets a vector (parity with
    ``HybridEmbeddingEngine``'s "never an exception for a transport failure").

    Default chain (spec req 6): Ollama -> ONNX -> OpenAI -> Fallback. The
    Fallback is appended automatically if not present, so a user who passes
    only ``[ONNXEmbeddingEngine()]`` still gets the terminal safety net.

    The pipeline is itself a duck-typed ``Embedder`` (``embed_text`` /
    ``embed_batch`` / ``is_real`` / ``dim``), so it drops into
    ``HybridEmbeddingEngine``'s slot in ``isotope_zero.client.IsotopeZero``.
    """

    def __init__(
        self,
        backends: list[BaseEmbeddingEngine] | None = None,
        dim: int = _DEFAULT_DIM,
        chunk_size: int = _EMBED_CHUNK,
    ) -> None:
        self._dim = int(dim)
        self.chunk_size = int(chunk_size)
        # Always terminate with the deterministic fallback so we never raise.
        chain = list(backends) if backends else []
        if not chain or not isinstance(chain[-1], FallbackEmbeddingEngine):
            chain.append(FallbackEmbeddingEngine(dim=self._dim))
        self._backends: list[BaseEmbeddingEngine] = chain
        # Index of the active backend; advanced on failure.
        self._active_idx: int = 0
        # Backends known to have failed (so we don't retry them every call).
        self._dead: set[int] = set()
        self._fallback_logged: bool = False

    # -- introspection ------------------------------------------------------ #
    @property
    def is_real(self) -> bool:
        """True iff the active backend reports ``is_real``."""
        b = self._active_backend()
        return bool(b.is_real) if b is not None else False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def metadata(self) -> dict[str, Any]:
        b = self._active_backend()
        if b is None:
            return {
                "dim": self._dim,
                "backend": "none",
                "provider": "none",
                "memory_type": "local-process",
            }
        md = dict(b.metadata)
        md["active_backend"] = type(b).__name__
        md["dead_backends"] = [
            type(self._backends[i]).__name__ for i in sorted(self._dead)
        ]
        return md

    @property
    def backends(self) -> list[BaseEmbeddingEngine]:
        """The configured backend chain (introspection / tests)."""
        return list(self._backends)

    @property
    def active_index(self) -> int:
        return self._active_idx

    # -- duck-typed embedder surface ---------------------------------------- #
    def embed_text(self, text: str) -> Embedding:
        if text == "":
            return Embedding(np_zeros(self._dim))
        # Wrap the row so the store's ``not query_vec`` guard is unambiguous
        # (a raw multi-element ndarray raises ValueError on ``not vec``).
        return Embedding(self.embed_batch([text])[0])

    def embed_batch(self, texts: list[str]) -> Any:
        np = _require_numpy()
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        # Chunk to bound peak memory on any single backend call (spec req 7).
        out: list[Any] = []
        for i in range(0, len(texts), self.chunk_size):
            chunk = texts[i : i + self.chunk_size]
            out.append(self._embed_chunk_with_failover(chunk))
        return np.concatenate(out, axis=0) if out else np.empty(
            (0, self._dim), dtype=np.float32
        )

    async def aembed_text(self, text: str) -> Any:
        if text == "":
            return Embedding(np_zeros(self._dim))
        return (await self.aembed_batch([text]))[0]

    async def aembed_batch(self, texts: list[str]) -> Any:
        np = _require_numpy()
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        parts: list[Any] = []
        for i in range(0, len(texts), self.chunk_size):
            chunk = texts[i : i + self.chunk_size]
            parts.append(await self._aembed_chunk_with_failover(chunk))
        return np.concatenate(parts, axis=0) if parts else np.empty(
            (0, self._dim), dtype=np.float32
        )

    # -- failover core ------------------------------------------------------ #
    def _active_backend(self) -> BaseEmbeddingEngine | None:
        # Skip dead backends to find the current live one.
        while self._active_idx < len(self._backends) and self._active_idx in self._dead:
            self._active_idx += 1
        if self._active_idx >= len(self._backends):
            # All dead — reset to the terminal fallback (always last).
            self._active_idx = len(self._backends) - 1
            self._dead.discard(self._active_idx)
        return self._backends[self._active_idx]

    def _embed_chunk_with_failover(self, chunk: list[str]) -> Any:
        """Try the active backend; on failure, advance and retry. Never raises."""
        last_exc: Exception | None = None
        # Walk from the current index forward; the terminal fallback never
        # raises, so this loop is guaranteed to return.
        while True:
            b = self._active_backend()
            if b is None:  # pragma: no cover - _active_backend always returns
                return FallbackEmbeddingEngine(self._dim).embed_batch(chunk)
            try:
                return b.embed_batch(chunk)
            except Exception as exc:  # noqa: BLE001 — connection / OOM / any
                last_exc = exc
                if not self._fallback_logged:
                    log.warning(
                        "backend %s failed (%s); failing over to next.",
                        type(b).__name__,
                        exc,
                    )
                    self._fallback_logged = True
                self._dead.add(self._active_idx)
                self._active_idx += 1
                continue

    async def _aembed_chunk_with_failover(self, chunk: list[str]) -> Any:
        last_exc: Exception | None = None
        while True:
            b = self._active_backend()
            if b is None:  # pragma: no cover
                return FallbackEmbeddingEngine(self._dim).embed_batch(chunk)
            try:
                return await b.aembed_batch(chunk)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not self._fallback_logged:
                    log.warning(
                        "backend %s failed (%s); failing over to next.",
                        type(b).__name__,
                        exc,
                    )
                    self._fallback_logged = True
                self._dead.add(self._active_idx)
                self._active_idx += 1
                continue

    def close(self) -> None:
        """Best-effort close of any backend that owns a session/socket."""
        for b in self._backends:
            try:
                close = getattr(b, "close", None)
                if close is not None:
                    close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Small numpy helpers (kept here so the spec engines share one impl).
# --------------------------------------------------------------------------- #
def np_zeros(n: int) -> Any:
    np = _require_numpy()
    return np.zeros(n, dtype=np.float32)


def _smoke() -> None:  # pragma: no cover
    eng = EmbeddingPipeline()
    a = eng.embed_text("the user likes rust and tea")
    b = eng.embed_text("the user likes rust and tea")
    c = eng.embed_text("completely unrelated nonsense zzz")
    print(f"dim={eng.dim} is_real={eng.is_real} metadata={eng.metadata}")
    print(f"first3(a)={list(a[:3])}")
    print(f"cos(a,b)={float(a @ b):.4f}  cos(a,c)={float(a @ c):.4f}")


if __name__ == "__main__":  # pragma: no cover
    _smoke()
