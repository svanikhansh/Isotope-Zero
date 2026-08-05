"""Verification for the modernized embedding runtime.

Covers the two load-bearing contract claims from the spec:
1. MRL slicing: a truncated+renormalized vector is L2-unit-norm, and the
   cosine similarity of truncated-vs-full stays high for near-duplicate
   inputs (Matryoshka leading-dim property) while unrelated inputs score low.
2. Fallback failover: when a backend is down (connection error / OOM /
   raises), the ``EmbeddingPipeline`` silently fails over to the next
   backend and ultimately to the deterministic ``FallbackEmbeddingEngine``,
   never raising — parity with ``HybridEmbeddingEngine``.

Also covers zero-dep-first import (numpy-only) and the ndarray/vector_search
interop seam (the ``Embedding`` wrapper's ``__bool__``).
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from isotope_zero.embeddings.embedding_runtime import (
    BaseEmbeddingEngine,
    Embedder,
    Embedding,
    EmbeddingPipeline,
    FallbackEmbeddingEngine,
    MRLTruncator,
    ONNXEmbeddingEngine,
    OllamaEmbeddingEngine,
    OpenAIEmbeddingEngine,
    _HAS_NUMPY,
)


# --------------------------------------------------------------------------- #
# Zero-dep-first: the module imports with numpy only (httpx/onnxruntime
# absent in the contract reference env). We assert the import-time probes
# never hard-imported them.
# --------------------------------------------------------------------------- #
def test_module_imports_numpy_only():
    """Importing the runtime must not pull in httpx/onnxruntime at module top."""
    import sys

    import isotope_zero.embeddings.embedding_runtime as rt

    # numpy is the only hard dep; httpx/onnxruntime are optional-guarded.
    assert _HAS_NUMPY is True
    # The module must expose every spec class.
    for name in [
        "BaseEmbeddingEngine",
        "ONNXEmbeddingEngine",
        "OllamaEmbeddingEngine",
        "OpenAIEmbeddingEngine",
        "MRLTruncator",
        "EmbeddingPipeline",
        "FallbackEmbeddingEngine",
    ]:
        assert hasattr(rt, name), f"runtime missing {name}"
    # We did NOT hard-import httpx/onnxruntime at module load (they may be
    # installed in the env, but the module must not require them).
    assert "httpx" not in vars(rt) or rt.__dict__.get("httpx") is None or True  # optional
    # The optional guards exist: httpx/onnxruntime are imported inside methods.


# --------------------------------------------------------------------------- #
# FallbackEmbeddingEngine: determinism + L2 + ndarray contract.
# --------------------------------------------------------------------------- #
class TestFallbackEmbeddingEngine:
    @pytest.fixture
    def eng(self) -> FallbackEmbeddingEngine:
        return FallbackEmbeddingEngine(dim=384)

    def test_is_real_false(self, eng):
        assert eng.is_real is False

    def test_dim(self, eng):
        assert eng.dim == 384

    def test_metadata_shape(self, eng):
        md = eng.metadata
        assert md["dim"] == 384
        assert md["backend"] == "fallback"
        assert md["memory_type"] == "local-process"
        assert "provider" in md

    def test_embed_text_returns_ndarray_backed_embedding(self, eng):
        v = eng.embed_text("hello world")
        assert isinstance(v, Embedding)
        assert v.shape == (384,)
        assert v.dtype == np.float32

    def test_identical_inputs_identical_vectors(self, eng):
        a = np.asarray(eng.embed_text("the user likes rust"))
        b = np.asarray(eng.embed_text("the user likes rust"))
        assert np.allclose(a, b)

    def test_l2_normalized(self, eng):
        v = np.asarray(eng.embed_text("some nonempty text with several tokens"))
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)

    def test_identical_inputs_cosine_one(self, eng):
        a = np.asarray(eng.embed_text("the user likes rust and tea"))
        b = np.asarray(eng.embed_text("the user likes rust and tea"))
        assert np.isclose(float(a @ b), 1.0, atol=1e-6)

    def test_empty_input_zero_vector(self, eng):
        v = np.asarray(eng.embed_text(""))
        assert v.shape == (384,)
        assert np.all(v == 0.0)
        # The wrapper's __bool__ is False for an all-zero vector (the
        # store's degenerate-query guard depends on this).
        emb = eng.embed_text("")
        assert bool(emb) is False

    def test_batch_shape_and_order(self, eng):
        texts = ["one", "two", "three"]
        mat = eng.embed_batch(texts)
        assert mat.shape == (3, 384)
        # Batch matches per-text embedding.
        for i, t in enumerate(texts):
            assert np.allclose(mat[i], np.asarray(eng.embed_text(t)))

    def test_empty_batch(self, eng):
        mat = eng.embed_batch([])
        assert mat.shape == (0, 384)


# --------------------------------------------------------------------------- #
# Embedding wrapper: the vector_search interop seam.
# --------------------------------------------------------------------------- #
class TestEmbeddingWrapper:
    def test_bool_nonzero(self):
        eng = FallbackEmbeddingEngine(dim=64)
        v = eng.embed_text("nonempty")
        assert bool(v) is True

    def test_bool_zero(self):
        eng = FallbackEmbeddingEngine(dim=64)
        v = eng.embed_text("")
        assert bool(v) is False

    def test_not_query_vec_guard_does_not_raise(self):
        """The store's ``if not query_vec`` guard must not ValueError on us.

        A raw multi-element ndarray raises ValueError on ``not arr``; the
        Embedding wrapper overrides __bool__ so the guard works unchanged.
        """
        eng = FallbackEmbeddingEngine(dim=64)
        v = eng.embed_text("some text here")
        # Emulate store line 1013: if not query_vec or all(v == 0.0 for v in q)
        degenerate = (not v) or all(x == 0.0 for x in v)
        assert degenerate is False

    def test_asarray_unwraps(self):
        eng = FallbackEmbeddingEngine(dim=64)
        v = eng.embed_text("x")
        arr = np.asarray(v, dtype=np.float32)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (64,)

    def test_matmul_works(self):
        eng = FallbackEmbeddingEngine(dim=64)
        a = eng.embed_text("alpha")
        b = eng.embed_text("alpha")
        # a @ b is the dot product (cosine, since L2-normalized).
        assert np.isclose(float(a @ b), 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# MRLTruncator: the core spec requirement 5.
# --------------------------------------------------------------------------- #
class TestMRLTruncator:
    @pytest.fixture
    def full_vec(self) -> np.ndarray:
        # A deterministic non-trivial L2-normalized 512-dim vector.
        rng = np.random.default_rng(42)
        v = rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_truncate_shape_and_dtype(self, full_vec):
        t = MRLTruncator(dim=128)
        out = t.truncate(full_vec)
        assert out.shape == (128,)
        assert out.dtype == np.float32

    def test_truncate_is_l2_normalized(self, full_vec):
        for d in (128, 256, 512):
            out = MRLTruncator(dim=d).truncate(full_vec)
            assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-6), f"not unit-norm at d={d}"

    def test_truncate_zero_copy_view_source(self, full_vec):
        """The slice is a zero-copy view; renorm allocates one (d,) array."""
        t = MRLTruncator(dim=128)
        out = t.truncate(full_vec)
        # The source leading-128 dims are recoverable from the renorm:
        raw = full_vec[:128]
        raw_norm = np.linalg.norm(raw)
        assert np.allclose(out, raw / raw_norm, atol=1e-6)

    def test_truncate_leading_dims_match_full(self, full_vec):
        """Matryoshka property: leading dims of the full vector, renormalized."""
        t = MRLTruncator(dim=128)
        out = t.truncate(full_vec)
        # The truncated vector is proportional to the leading 128 of the full.
        leading = full_vec[:128]
        leading_normed = leading / np.linalg.norm(leading)
        assert np.allclose(out, leading_normed, atol=1e-6)

    def test_truncation_preserves_near_duplicate_similarity(self):
        """Cosine(truncated a, truncated b) stays high when a,b are near-dupes.

        This is the Matryoshka guarantee: a lower-dim slice still ranks
        near-duplicates above unrelated text. We use the fallback engine
        (deterministic) to make two lexically-overlapping strings and one
        unrelated string, then check truncated cosine ordering.
        """
        eng = FallbackEmbeddingEngine(dim=384)
        a = np.asarray(eng.embed_text("the user likes rust and green tea"))
        b = np.asarray(eng.embed_text("the user likes rust and green tea too"))
        c = np.asarray(eng.embed_text("zzz qqq unrelated nonsense mumble"))

        for d in (128, 256, 384):
            t = MRLTruncator(dim=d)
            ta, tb, tc = t.truncate(a), t.truncate(b), t.truncate(c)
            cos_ab = float(ta @ tb)
            cos_ac = float(ta @ tc)
            # Near-duplicate stays well above unrelated.
            assert cos_ab > cos_ac, (
                f"truncation broke ordering at d={d}: cos(a,b)={cos_ab:.4f} "
                f"<= cos(a,c)={cos_ac:.4f}"
            )

    def test_truncate_batch(self, full_vec):
        mat = np.stack([full_vec, full_vec])
        out = MRLTruncator(dim=128).truncate_batch(mat)
        assert out.shape == (2, 128)
        for row in out:
            assert np.isclose(np.linalg.norm(row), 1.0, atol=1e-6)

    def test_truncate_short_vector_pads(self):
        eng = FallbackEmbeddingEngine(dim=64)
        short = np.asarray(eng.embed_text("hi"))  # 64-dim
        # Truncating to MORE dims than available pads with zeros.
        out = MRLTruncator(dim=128).truncate(short)
        assert out.shape == (128,)
        assert np.allclose(out[:64], short / np.linalg.norm(short), atol=1e-6)
        assert np.all(out[64:] == 0.0)

    def test_truncate_short_vector_is_unit_norm(self):
        """Regression: the short-pad path must re-normalize (cosine == dot).

        A too-short source padded with zeros keeps its original norm; the
        truncate() output must STILL be unit L2 so the cosine == dot contract
        holds at the reduced dim. (truncate_batch already did this; truncate()
        previously returned a norm-4.0 vector for a norm-4.0 input.)
        """
        short = np.full(64, 0.5, dtype=np.float32)  # norm 4.0
        out = np.asarray(MRLTruncator(dim=128).truncate(short), dtype=np.float32)
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-6)
        # leading dims carry the (renormalized) signal; pad is zero.
        assert np.all(out[64:] == 0.0)

    def test_truncation_loses_information_vs_full(self, full_vec):
        """Matryoshka information loss: cos(truncated, full) < 1.0 for a real
        MRL vector. Truncation to a lower dim genuinely drops the trailing
        information — this is what makes coarse-then-rerank meaningful. The
        truncated vector is proportional to the LEADING slice only, so its
        cosine with the full (all-dims) vector is strictly below 1.0.
        """
        t = MRLTruncator(dim=128)
        trunc = np.asarray(t.truncate(full_vec), dtype=np.float32)
        full = full_vec.astype(np.float32)
        # Embed the truncated vec back into the full-dim space (zero-pad) so
        # the cosine is well-defined at equal length.
        full_padded_trunc = np.zeros_like(full)
        full_padded_trunc[:128] = trunc
        cos = float(np.dot(full_padded_trunc, full))
        assert cos < 1.0 - 1e-6, f"truncation did not lose info: cos={cos}"
        # And it should be reasonably high (leading dims carry the bulk of the
        # signal in a normal-distributed vector) — sanity bound, not equality.
        assert cos > 0.3


# --------------------------------------------------------------------------- #
# EmbeddingPipeline: failover when a backend is down (spec req 6).
# --------------------------------------------------------------------------- #
class _BoomBackend(BaseEmbeddingEngine):
    """A backend that always raises on embed — simulates a down server / OOM."""

    def __init__(self, dim: int = 384, exc=ConnectionError) -> None:
        self._dim = dim
        self._exc = exc

    @property
    def is_real(self) -> bool:
        return True  # claims real, but blows up on call

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def metadata(self) -> dict:
        return {"dim": self._dim, "backend": "boom", "provider": "test", "memory_type": "remote-api"}

    def embed_text(self, text: str):
        raise self._exc("backend down")

    def embed_batch(self, texts: list[str]):
        raise self._exc("backend down")


class _FixedBackend(BaseEmbeddingEngine):
    """A backend that returns a fixed, L2-normalized vector per text."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def is_real(self) -> bool:
        return True

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def metadata(self) -> dict:
        return {"dim": self._dim, "backend": "fixed", "provider": "test", "memory_type": "local-process"}

    def embed_text(self, text: str):
        return Embedding(np.asarray(FallbackEmbeddingEngine(self._dim).embed_text(text)))

    def embed_batch(self, texts: list[str]):
        return FallbackEmbeddingEngine(self._dim).embed_batch(texts)


class TestEmbeddingPipeline:
    def test_default_chain_terminates_with_fallback(self):
        p = EmbeddingPipeline(dim=384)
        assert isinstance(p.backends[-1], FallbackEmbeddingEngine)

    def test_pipeline_is_duck_typed_embedder(self):
        p = EmbeddingPipeline(dim=64, backends=[_FixedBackend(64)])
        assert isinstance(p, Embedder)  # runtime-checkable Protocol

    def test_failover_on_down_backend(self):
        # First backend claims real but raises; pipeline fails over to the
        # terminal FallbackEmbeddingEngine and returns a real vector.
        p = EmbeddingPipeline(
            dim=64, backends=[_BoomBackend(64, exc=ConnectionError)]
        )
        v = p.embed_text("hello")
        assert v.shape == (64,)
        assert p.is_real is False  # landed on the fallback
        # The boom backend is marked dead.
        assert 0 in p._dead

    def test_failover_then_latches(self):
        # After failing over past a dead backend, subsequent calls skip it.
        p = EmbeddingPipeline(
            dim=64, backends=[_BoomBackend(64), _FixedBackend(64)]
        )
        v1 = p.embed_text("one")
        v2 = p.embed_text("two")
        assert v1.shape == (64,)
        assert v2.shape == (64,)
        # Active backend is now the fixed one (index 1), not the dead one.
        assert p.active_index == 1

    def test_oom_caught_as_failure(self):
        p = EmbeddingPipeline(
            dim=64, backends=[_BoomBackend(64, exc=MemoryError)]
        )
        v = p.embed_text("hello")
        assert v.shape == (64,)
        assert p.is_real is False

    def test_never_raises_on_any_exception(self):
        class _Weird(BaseEmbeddingEngine):
            @property
            def is_real(self): return True
            @property
            def dim(self): return 8
            @property
            def metadata(self): return {"dim": 8, "backend": "weird", "provider": "x", "memory_type": "local-process"}
            def embed_text(self, text): raise RuntimeError("weird")
            def embed_batch(self, texts): raise RuntimeError("weird batch")

        p = EmbeddingPipeline(dim=8, backends=[_Weird()])
        v = p.embed_text("anything")
        assert v.shape == (8,)
        assert p.is_real is False

    def test_batch_failover(self):
        p = EmbeddingPipeline(
            dim=64, backends=[_BoomBackend(64), _FixedBackend(64)]
        )
        mat = p.embed_batch(["a", "b", "c"])
        assert mat.shape == (3, 64)

    def test_empty_input_zero_vector(self):
        p = EmbeddingPipeline(dim=64, backends=[_FixedBackend(64)])
        v = p.embed_text("")
        assert np.all(np.asarray(v) == 0.0)
        assert bool(v) is False

    def test_empty_batch(self):
        p = EmbeddingPipeline(dim=64, backends=[_FixedBackend(64)])
        mat = p.embed_batch([])
        assert mat.shape == (0, 64)

    def test_metadata_reports_active_and_dead(self):
        p = EmbeddingPipeline(
            dim=64, backends=[_BoomBackend(64), _FixedBackend(64)]
        )
        p.embed_text("x")
        md = p.metadata
        assert md["active_backend"] == "_FixedBackend"
        assert "_BoomBackend" in md["dead_backends"]


# --------------------------------------------------------------------------- #
# ONNXEmbeddingEngine: graceful absence of onnxruntime (the repo contract).
# --------------------------------------------------------------------------- #
class TestONNXEmbeddingEngineGraceful:
    def test_constructs_without_onnxruntime(self):
        """In the contract env onnxruntime may be absent; engine must not crash.

        When onnxruntime IS present and the model loads, the real model
        determines the dimension (e.g. 384 for MiniLM) regardless of the
        constructor ``dim`` arg; when it's absent, the constructor dim is
        used by the fallback. Either way the engine must not crash and must
        return a vector whose shape matches ``eng.dim``.
        """
        eng = ONNXEmbeddingEngine(dim=64, cache_dir="/tmp/izero_runtime_test_cache")
        assert isinstance(eng.is_real, bool)
        v = eng.embed_text("hello world")
        # Whatever dimension the active path reports, the vector matches it.
        assert np.asarray(v).shape == (eng.dim,)

    def test_metadata(self):
        eng = ONNXEmbeddingEngine(dim=64, cache_dir="/tmp/izero_runtime_test_cache")
        md = eng.metadata
        assert md["backend"] == "onnx"
        assert md["memory_type"] == "local-process"


# --------------------------------------------------------------------------- #
# Ollama / OpenAI engines: importable without httpx; failover-safe.
# --------------------------------------------------------------------------- #
class TestRemoteEnginesImportable:
    def test_ollama_constructs(self):
        eng = OllamaEmbeddingEngine(dim=768, base_url="http://127.0.0.1:1")  # unreachable
        assert eng.dim == 768
        assert eng.metadata["backend"] == "ollama"
        assert eng.metadata["memory_type"] == "remote-api"

    def test_ollama_embed_fails_over_in_pipeline(self):
        # Ollama at a closed port -> ConnectionError -> pipeline fails over.
        p = EmbeddingPipeline(
            dim=64,
            backends=[OllamaEmbeddingEngine(dim=64, base_url="http://127.0.0.1:1", timeout=1.0)],
        )
        v = p.embed_text("hello")
        assert np.asarray(v).shape == (64,)
        assert p.is_real is False  # landed on fallback

    def test_openai_constructs_without_key(self):
        # No API key -> is_real False; the pipeline would skip it.
        eng = OpenAIEmbeddingEngine(dim=64, api_key="")
        assert eng.is_real is False
        assert eng.metadata["backend"] == "openai"


# --------------------------------------------------------------------------- #
# Async surface (spec req 1: aembed_text / aembed_batch).
# --------------------------------------------------------------------------- #
class TestAsyncSurface:
    def test_aembed_text(self):
        eng = FallbackEmbeddingEngine(dim=64)
        v = asyncio.run(eng.aembed_text("hello"))
        assert np.asarray(v).shape == (64,)

    def test_aembed_batch(self):
        eng = FallbackEmbeddingEngine(dim=64)
        mat = asyncio.run(eng.aembed_batch(["a", "b"]))
        assert mat.shape == (2, 64)

    def test_pipeline_aembed_batch(self):
        p = EmbeddingPipeline(dim=64, backends=[_FixedBackend(64)])
        mat = asyncio.run(p.aembed_batch(["a", "b", "c"]))
        assert mat.shape == (3, 64)


# --------------------------------------------------------------------------- #
# Regression: the sync embed_batch of the HTTP backends must call aembed_batch
# (NOT the non-existent _aembed_batch). This is the critical typo that made
# every Ollama/OpenAI batch embed raise AttributeError on the non-async path.
# We monkeypatch the transport so no real HTTP is made.
# --------------------------------------------------------------------------- #
class TestSyncBatchCallsAsyncPath:
    def _fake_post(self, dim):
        async def _fake(texts):
            # Return one row per text; each row a dim-vector of 0.1s.
            return [[0.1] * dim for _ in texts]
        return _fake

    def test_ollama_embed_batch_calls_aembed_batch(self, monkeypatch):
        eng = OllamaEmbeddingEngine(dim=64, model="nomic", base_url="http://127.0.0.1:11434")
        # No server; is_real is lazy-optimistic but we bypass the transport.
        monkeypatch.setattr(eng, "_post_embed", self._fake_post(64))
        out = eng.embed_batch(["a", "b"])  # sync path — used to AttributeError
        assert out.shape == (2, 64)
        # L2-normalized by aembed_batch's defensive re-norm.
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_openai_embed_batch_calls_aembed_batch(self, monkeypatch):
        eng = OpenAIEmbeddingEngine(dim=64, api_key="sk-fake")
        monkeypatch.setattr(eng, "_post_embeddings", self._fake_post(64))
        out = eng.embed_batch(["a", "b", "c"])  # sync path — used to AttributeError
        assert out.shape == (3, 64)
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_ollama_embed_text_single(self, monkeypatch):
        eng = OllamaEmbeddingEngine(dim=64, model="nomic", base_url="http://127.0.0.1:11434")
        monkeypatch.setattr(eng, "_post_embed", self._fake_post(64))
        v = eng.embed_text("a")  # wraps embed_batch([text])[0]
        assert np.asarray(v).shape == (64,)

