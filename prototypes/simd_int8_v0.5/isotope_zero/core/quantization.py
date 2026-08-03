"""SQ8 (int8 scalar quantization) for Isotope Zero embeddings.

Per-vector symmetric scalar quantization: a 384-dim float32 embedding
(1536 bytes) is compressed to 384 int8 values (384 bytes) plus a single
per-vector float scale, a 4x reduction in resident vector-cache footprint
(~75% RAM saved). Dequantization reconstructs an approximate float32 vector
with a bounded per-element error of <= scale (<= max(|vec|)/127).

This module is PURE quantization math -- it has no dependency on the store
and never touches SQLite. The store (``isotope_zero.core.store``) calls these
functions at write time (to populate ``q_embedding`` / ``q_scale`` columns)
and at search time (via ``vector_search_int8``).

Honesty note (do not paper over): ``int8_dot_product`` uses a genuine integer
matmul with an int32 accumulator (``q_matrix.astype(np.int32) @
q_query.astype(np.int32)``). numpy's ``@`` on integer arrays is NOT
BLAS-accelerated on most builds (BLAS accelerates float32/float64 only), so
numpy falls back to a generic integer loop. This may be SLOWER than the
float32 BLAS matmul used by ``vector_search``. That is a real measurement to
surface, not a bug to hide -- the point of this prototype is to find out
whether SQ8 actually delivers the claimed RAM savings AND acceptable latency.
"""
from __future__ import annotations

import numpy as np

__all__ = ["quantize_vector", "dequantize_vector", "int8_dot_product"]


def quantize_vector(vec: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize a float32 vector to int8 with a per-vector symmetric scale.

    Returns ``(q_vec int8, scale float)`` such that ``dequantize_vector(q_vec,
    scale)`` approximates ``vec``.

    Per-vector symmetric scale::

        scale = max(abs(vec)) / 127.0
        q_vec = clip(round(vec / scale), -128, 127).astype(np.int8)

    Divide-by-zero guard: when ``max(abs(vec)) == 0`` (an all-zero vector) the
    scale is set to ``1.0`` and ``q_vec`` is all zeros, so dequant returns the
    zero vector (mathematically correct: scale is irrelevant when every int8 is
    0).

    ``np.round`` (round-half-to-even) is used for the rounding step, matching
    the scope's ``floor(v/scale + 0.5)`` intent for all non-half values; the
    half-to-even tiebreak is immaterial for the per-element error bound.
    """
    v = np.asarray(vec, dtype=np.float32)
    max_abs = float(np.max(np.abs(v))) if v.size > 0 else 0.0
    if max_abs == 0.0:
        # All-zero vector: scale is irrelevant (every q entry is 0). Use 1.0
        # so callers never see a 0 scale (which would make dequant = 0
        # unconditionally -- correct here, but 1.0 is a safe, non-degenerate
        # convention that avoids any divide-by-zero downstream).
        return np.zeros(v.shape, dtype=np.int8), 1.0
    scale = max_abs / 127.0
    q = np.round(v / np.float32(scale))
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, float(scale)


def dequantize_vector(q_vec: np.ndarray, scale: float) -> np.ndarray:
    """Recover an approximate float32 vector: ``scale * q_vec.astype(float32)``.

    The per-element error ``|orig - dequant|`` is bounded by ``scale`` (i.e.
    ``<= max(|orig|) / 127``) for in-range originals; clipping at +/-127 can
    introduce slightly larger error only for originals exceeding the symmetric
    range, which cannot happen since ``scale = max/127`` puts the largest
    magnitude at exactly 127 before rounding.
    """
    return np.asarray(q_vec).astype(np.float32) * np.float32(scale)


def int8_dot_product(
    q_matrix: np.ndarray,
    q_query: np.ndarray,
    scales: np.ndarray,
    query_scale: float,
) -> np.ndarray:
    """Integer dot products of each matrix row vs the query, rescaled to float.

    Parameters
    ----------
    q_matrix : np.ndarray, shape (n, d), dtype int8
        Quantized matrix rows (one per stored card).
    q_query : np.ndarray, shape (d,), dtype int8
        Quantized query vector.
    scales : np.ndarray, shape (n,), float
        Per-row scale for the MATRIX rows (``scale_i``).
    query_scale : float
        The query's own per-vector scale (``scale_query``), passed SEPARATELY
        (not folded into ``scales``). The caller is expected to pass the raw
        per-row scales and the raw query scale; this function multiplies them
        together internally. This keeps the signature explicit and lets the
        caller reuse the cached per-row scales across many queries without
        recomputing ``scales * query_scale`` each time at the cache level.

    Returns
    -------
    np.ndarray, shape (n,), dtype float32
        ``scores[i] = scales[i] * query_scale * sum_j(q_matrix[i,j] *
        q_query[j])``. This is the approximate RAW dot product of the
        dequantized vectors (``dequant_i . dequant_q``), NOT a cosine: the
        dequantized vectors are not re-L2-normalized, so the magnitude drifts.
        The caller (``vector_search_int8``) divides by the product of the
        dequantized norms to recover a comparable [0,1] cosine score.

    Accumulator dtype
    -----------------
    The raw integer dot is computed as
    ``q_matrix.astype(np.int32) @ q_query.astype(np.int32)``. int32 is
    sufficient: the worst-case ``|sum| = d * 127 * 127 = 384 * 16129 ~= 6.2M``,
    well under int32's max of ~2.1e9. The explicit int32 cast is REQUIRED for
    overflow safety: numpy's default result dtype for ``int8 @ int8`` is
    build-dependent (numpy 2.x keeps it int8 to avoid surprising upcasts;
    older builds promoted to int32 or int64). int8 can only hold [-128, 127],
    so the un-cast path silently overflows on numpy 2.x -- the worst-case
    sum of ~6.2M wraps around modulo 256 and produces garbage scores. Pinning
    int32 avoids overflow on every build. Do NOT use plain
    ``q_matrix @ q_query`` without this cast.

    Performance caveat
    ------------------
    ``np.matmul``/``@`` on integer arrays is NOT BLAS-accelerated on most
    platforms (BLAS is float32/float64 only); numpy uses a generic integer
    loop. This may be SLOWER than the float32 BLAS matmul in ``vector_search``.
    That is a real, honest measurement to surface -- do not "fix" it by
    silently swapping to a float32 matmul here, as that would defeat the
    purpose of measuring genuine SQ8 search latency. The cast to int32 also
    allocates a transient (n, d) int32 buffer (~4x the int8 cache size)
    per call; the RESIDENT footprint metric (``quantized_matrix_nbytes``)
    still measures the int8 cache, which is the honest steady-state size.
    """
    # Explicit int32 cast for a deterministic, overflow-safe accumulator.
    raw_int = q_matrix.astype(np.int32) @ q_query.astype(np.int32)
    # Rescale: scales[i] * query_scale * raw_int[i]. Broadcast (n,) * scalar
    # then (n,) elementwise. Return float32 for downstream clip/sort parity
    # with the float32 vector_search path.
    combined_scale = np.asarray(scales, dtype=np.float32) * np.float32(query_scale)
    return raw_int.astype(np.float32) * combined_scale


# ---------------------------------------------------------------------- #
# Self-check: run via `python -m isotope_zero.core.quantization` or
# `python isotope_zero/core/quantization.py`. Exercises the round-trip
# error bound, the int8 dtype, and the dot-product rescale math on a
# hand-computable example.
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # 1. Round-trip error bound + int8 dtype.
    rng = np.random.default_rng(42)
    v = rng.standard_normal(384).astype(np.float32)
    v = v / np.linalg.norm(v)  # L2-normalized, like the embedder's output
    q, scale = quantize_vector(v)
    assert q.dtype == np.int8, f"expected int8, got {q.dtype}"
    assert q.shape == v.shape, f"shape mismatch {q.shape} vs {v.shape}"
    deq = dequantize_vector(q, scale)
    max_err = float(np.max(np.abs(v - deq)))
    assert max_err <= scale + 1e-6, (
        f"round-trip error {max_err} exceeds scale {scale} (bound = max/127)"
    )
    # The all-zero edge case: scale defaults to 1.0, q is all zeros.
    q0, s0 = quantize_vector(np.zeros(8, dtype=np.float32))
    assert q0.dtype == np.int8 and s0 == 1.0 and bool(np.all(q0 == 0)), (
        "all-zero vector must yield int8 zeros + scale 1.0"
    )
    print(f"[1] round-trip OK: max_err={max_err:.6f} <= scale={scale:.6f}, "
          f"dtype=int8, zero-vec edge OK")

    # 2. Dot-product rescale correctness on a hand-computable example.
    v1 = np.array([1.0, 2.0, 3.0, -4.0], dtype=np.float32)
    v2 = np.array([0.5, -1.0, 2.0, 1.0], dtype=np.float32)
    q1, s1 = quantize_vector(v1)
    q2, s2 = quantize_vector(v2)
    # Manual raw integer dot (int32 accumulator, the exact formula).
    manual_int = int((q1.astype(np.int32) * q2.astype(np.int32)).sum())
    # The function's rescaled output.
    got = int8_dot_product(q1[None, :], q2, np.array([s1], dtype=np.float32), s2)
    expected = manual_int * s1 * s2
    assert abs(float(got[0]) - expected) < 1e-6, (
        f"rescale mismatch: got {got[0]} expected {expected}"
    )
    # Compare to the float32 dot of the ORIGINALS -- within quantization error.
    float_dot = float(v1 @ v2)
    assert abs(float(got[0]) - float_dot) < 0.15, (
        f"int8 rescaled dot {float(got[0])} too far from float dot {float_dot}"
    )
    print(f"[2] dot rescale OK: int={manual_int} s1={s1:.6f} s2={s2:.6f} "
          f"-> {float(got[0]):.6f} (float dot={float_dot:.6f}, "
          f"err={abs(float(got[0])-float_dot):.6f})")

    # 3. Batch path + rescale matches per-row manual on a small matrix.
    M = np.stack([q1, q2])
    scales = np.array([s1, s2], dtype=np.float32)
    out = int8_dot_product(M, q1, scales, s1)
    # Row 0 is v1-vs-v1 (self): dequant dot, should be ~ v1.v1 = 1+4+9+16=30.
    self_dot = float(v1 @ v1)
    assert abs(float(out[0]) - self_dot) < 0.5, (
        f"self-dot {float(out[0])} vs float {self_dot}"
    )
    print(f"[3] batch OK: out={out.tolist()} (v1.v1 float={self_dot:.4f})")

    print("quantization self-check PASSED")
