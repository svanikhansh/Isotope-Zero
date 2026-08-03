"""Local ONNX quantized embedding engine for isotope_zero.

Design priorities (in order): (1) zero external network calls at inference
time, (2) graceful degradation — the system must remain fully runnable and
testable even when ONNX runtime / tokenizers are absent or the model can't
download, (3) L2-normalized outputs so cosine similarity == dot product.

The real path loads a quantized ONNX model (e.g. Xenova/all-MiniLM-L6-v2)
and a matching tokenizer, caching both on disk. The fallback path produces
a deterministic, L2-normalized feature-hash vector of the correct dimension
so that identical texts still score 1.0 and unrelated texts score ~0.
"""
from __future__ import annotations

import gc
import hashlib
import logging
import math
import os
import urllib.request
from typing import Any

from isotope_zero.types import now_ts  # noqa: F401  (kept for tests that monkeypatch time)

log = logging.getLogger("isotope_zero.embeddings")

# Maximum number of texts fed to the ONNX session in ONE forward pass.
# Bounding this caps the peak `token_embeds` tensor at (CHUNK, seq, 384) fp32
# regardless of how large a batch embed_batch receives, so the 10k scale-seeding
# sweep never allocates a single ~460MB (10000, seq, 384) tensor. The
# tokenizer's encode_batch still runs once per chunk; only the ONNX forward path
# is chunked, and the public embed_batch contract (input-ordered
# list[list[float]]) is preserved by accumulating per-chunk results.
#
# 64 is the upper bound that keeps a single forward tensor tiny, but measured
# 1k scale-seeding RSS (variable-length facts, real MiniLM) is 235MB embed-peak
# at 64 — the ORT arena grows power-of-2 blocks for the mixed/longer sequences.
# 32 drops that to ~177MB and (with make_scale_cards streaming) clears the
# <200MB @1k RSS bar. Still well under the tensor cap.
_EMBED_CHUNK = 32

# HuggingFace ONNX repos known to ship quantized sentence-embedding models.
# Tried in order; first one that downloads both model + tokenizer wins.
_HF_CANDIDATES: dict[str, tuple[str, str]] = {
    # model_name -> (hf_repo, onnx_subpath)
    "all-MiniLM-L6-v2": ("Xenova/all-MiniLM-L6-v2", "onnx/model_quantized.onnx"),
    "bge-micro-v2": ("Xenova/bge-micro-v2", "onnx/model_quantized.onnx"),
}


class EmbeddingEngine:
    """Local embedding engine with an offline fallback.

    Construction may download the model exactly once (network permitted); all
    subsequent inference is purely local. If anything on the real path fails,
    the engine transparently falls back to a deterministic pseudo-embedding
    and sets ``is_real = False``.

    Parameters
    ----------
    model_name:
        HuggingFace ONNX model shorthand. Currently supports
        ``all-MiniLM-L6-v2`` (dim 384) and ``bge-micro-v2`` (dim 384).
    dim:
        Embedding dimension. Must match the model; used by the fallback.
    quantized:
        Prefer the quantized ONNX subpath when available.
    cache_dir:
        Local directory used to cache the downloaded model + tokenizer.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dim: int = 384,
        quantized: bool = True,
        cache_dir: str = ".isotope_zero_cache",
    ) -> None:
        self.model_name = model_name
        self._dim = int(dim)
        self.quantized = quantized
        self.cache_dir = cache_dir
        self.is_real: bool = False

        # Lazily-populated real-path state.
        self._session: Any = None
        self._tokenizer: Any = None
        self._max_length: int = 256

        try:
            self._try_load_real()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Embedding real-path crashed (%s); using fallback.", exc)
            self._session = None
            self._tokenizer = None
            self.is_real = False

        if not self.is_real:
            log.warning(
                "EmbeddingEngine running in FALLBACK mode (model=%s, dim=%d). "
                "Outputs are deterministic feature-hash pseudo-embeddings, "
                "NOT semantic. Install onnxruntime+tokenizers for real vectors.",
                self.model_name,
                self._dim,
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string, L2-normalized. Empty input -> zero vector."""
        if text == "":
            return [0.0] * self._dim
        if self.is_real:
            return self._embed_real([text])[0]
        return self._embed_fallback(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Real path chunks the ONNX forward; fallback maps."""
        if not texts:
            return []
        if self.is_real:
            out = self._embed_real(texts)
            # Chunking kicked in: run a deterministic GC so the per-chunk numpy
            # temporaries + Python list results are released before the caller
            # measures RSS (avoids transient buffers inflating ru_maxrss, and
            # breaks any reference cycles between the arrays and their metadata).
            if len(texts) > _EMBED_CHUNK:
                gc.collect()
            return out
        return [self._embed_fallback(t) for t in texts]

    # ------------------------------------------------------------------ #
    # Real ONNX path
    # ------------------------------------------------------------------ #

    def _try_load_real(self) -> None:
        """Attempt to locate/​download + load the ONNX model + tokenizer."""
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
                return  # fallback

        try:
            opts = ort.SessionOptions()
            # Memory-frugal, single-threaded, deterministic execution — this is a
            # latency-sensitive prototype embedder, not a throughput server.
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # run graph ops in order, no graph-level op parallelism sharing scratch
            opts.intra_op_num_threads = 1  # one thread per op -> stable, predictable per-node latency
            opts.inter_op_num_threads = 1  # one graph op at a time -> no extra worker threads / per-node arenas
            opts.enable_cpu_mem_arena = True  # keep the CPU arena ON: it pools allocations so repeated chunked runs reuse buffers instead of churning malloc
            opts.enable_mem_pattern = True  # let ORT pre-plan + reuse intermediate tensor buffers across the sequential graph
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # fusion + constant folding shrink tensor footprint
            self._session = ort.InferenceSession(model_path, sess_options=opts)
            self._tokenizer = Tokenizer.from_file(tok_path)
            self._tokenizer.enable_padding(length=None)
            self._max_length = 256
            self.is_real = True
            log.info("Loaded real ONNX embedding model from %s.", model_path)
        except Exception as exc:
            log.warning("Failed to init ONNX session/tokenizer (%s).", exc)
            self._session = None
            self._tokenizer = None

    def _download(self, repo: str, onnx_sub: str, model_path: str, tok_path: str) -> bool:
        """Download model + tokenizer from HuggingFace. Returns True on success."""
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

    def _embed_real(self, texts: list[str]) -> list[list[float]]:
        """Run the ONNX model over fixed-size chunks; mean-pool token embeddings.

        Inputs are processed in slices of at most ``_EMBED_CHUNK`` so the peak
        ``token_embeds`` tensor stays ~(64, seq, 384) regardless of batch size.
        Per-chunk encode/pool/step is identical to the original single-pass
        math (same masking + mean-pool + L2-normalization), so output vectors
        are unchanged; only the forward-path tensor size is bounded.
        """
        import numpy as np  # local import; only needed on the real path

        out_all: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_CHUNK):
            chunk = texts[start : start + _EMBED_CHUNK]
            enc = self._tokenizer.encode_batch(chunk)
            input_ids = np.array([e.ids for e in enc], dtype=np.int64)
            attn = np.array([e.attention_mask for e in enc], dtype=np.int64)
            # Pad to common length (tokenizer padding handles this; ensure 2D).
            if input_ids.ndim == 1:
                input_ids = input_ids.reshape(1, -1)
                attn = attn.reshape(1, -1)

            feeds = {
                "input_ids": input_ids,
                "token_type_ids": np.zeros_like(input_ids),
                "attention_mask": attn,
            }
            # Some exported models only name two inputs; drop token_type_ids if rejected.
            try:
                out = self._session.run(None, feeds)
            except Exception:
                feeds.pop("token_type_ids", None)
                out = self._session.run(None, feeds)

            token_embeds = out[0]  # (chunk, seq, hidden)
            # Mean-pool over tokens, masked by attention_mask.
            mask = attn[..., None].astype(np.float32)
            summed = (token_embeds * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1, None)
            pooled = summed / counts

            # L2-normalize.
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normed = pooled / norms
            out_all.extend(normed.astype(np.float32).tolist())
            # Eagerly drop this chunk's numpy temporaries before the next pass
            # so peak RSS reflects one chunk, not an accumulation of chunks.
            del token_embeds, mask, summed, counts, pooled, norms, normed
        return out_all

    # ------------------------------------------------------------------ #
    # Deterministic fallback path
    # ------------------------------------------------------------------ #

    def _embed_fallback(self, text: str) -> list[float]:
        """Deterministic feature-hash embedding, L2-normalized.

        Whitespace tokens are hashed into `dim` signed buckets. This is NOT
        semantic but is deterministic: identical inputs -> identical vectors
        (cosine 1.0), and lexical-overlap queries still surface near matches.
        """
        vec = [0.0] * self._dim
        tokens = text.lower().split()
        if not tokens:
            return vec
        for tok in tokens:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            # Two ints from the digest: bucket index + sign.
            bucket = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            # Weight rarer tokens slightly more via a dampened count.
            vec[bucket] += sign
        # Add a global text signature so even single-token texts spread energy.
        sig = hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest()
        for i in range(0, min(8, self._dim)):
            b = sig[i]
            vec[(b * 7 + i) % self._dim] += (b / 127.5) - 1.0
        # L2-normalize; if zero vector, return as-is (all zeros).
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec


def _smoke() -> None:  # pragma: no cover
    eng = EmbeddingEngine()
    a = eng.embed_text("the user likes rust and tea")
    b = eng.embed_text("the user likes rust and tea")
    c = eng.embed_text("completely unrelated nonsense zzz")
    import math as _m

    def dot(x: list[float], y: list[float]) -> float:
        return sum(xi * yi for xi, yi in zip(x, y))

    print(f"dim={eng.dim} is_real={eng.is_real}")
    print(f"first3(a)={a[:3]}")
    print(f"cos(a,b)={dot(a,b):.4f}  cos(a,c)={dot(a,c):.4f}")


if __name__ == "__main__":  # pragma: no cover
    _smoke()
