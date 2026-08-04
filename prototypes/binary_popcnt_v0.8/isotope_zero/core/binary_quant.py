"""1-bit binary quantization + POPCNT Hamming search for Phase 7B.

Replaces float32 vector search with a 2-stage pipeline:
1. Stage 1: 1-bit sign-quantized vectors (384-dim -> 48 bytes) via XOR + POPCNT
   (Hamming) to filter top K x oversample_factor candidates.
2. Stage 2: Exact float32 cosine re-rank on the surviving candidates.

All operations are pure Python + numpy. No ONNX, no tokenizers, no
third-party imports beyond numpy.
"""
from __future__ import annotations

# Precomputed POPCNT lookup table (0..255 -> bit count).
# Built with plain Python so numpy is not required at import time;
# converted to np.ndarray inside binary_hamming_distance at first use.
_POPCNT: list[int] = [int.bit_count(i) for i in range(256)]


def quantize_1bit(vectors: "np.ndarray") -> "np.ndarray":
    """sign(v_i) -> {0,1}. Float32 (n, dim) -> packed uint8 (n, ceil(dim/8)).

    Each float is thresholded at zero: ``>= 0`` maps to ``1``, ``< 0`` maps
    to ``0``.  The resulting bit-matrix is packed into ``uint8`` bytes via
    ``np.packbits(axis=1)``, producing ``ceil(dim / 8)`` bytes per vector
    (48 bytes for the standard 384-dim embeddings).
    """
    import numpy as np

    bits = (vectors >= 0).astype(np.uint8)  # (n, dim)
    packed = np.packbits(bits, axis=1)  # (n, ceil(dim/8))
    return packed


def matrix_to_binary(embs: "np.ndarray") -> "np.ndarray":
    """Convert float32 (n, dim) to packed binary (n, ceil(dim/8))."""
    return quantize_1bit(embs)


def binary_hamming_distance(
    packed_a: "np.ndarray", packed_b: "np.ndarray"
) -> "np.ndarray":
    """Hamming distance via XOR + POPCNT. (n, B) vs (m, B) -> (n, m).

    Uses a precomputed 256-entry POPCNT lookup table for bit-counting each
    byte, applied element-wise over the XOR result and then summed per
    vector pair.  Both inputs must have the same number of byte columns
    (typically 48 for 384-dim embeddings).
    """
    import numpy as np

    popcnt = np.array(_POPCNT, dtype=np.uint8)
    # Broadcasting: (n, 1, B) ^ (1, m, B) -> (n, m, B)
    xor = packed_a[:, None, :] ^ packed_b[None, :, :]
    counts = popcnt[xor]  # (n, m, B) uint8
    return counts.sum(axis=2, dtype=np.int32)  # (n, m)


def binary_search(
    matrix_f32: "np.ndarray",
    q_f32: "np.ndarray",
    top_k: int,
    oversample_factor: int = 10,
) -> "tuple[list[int], list[int]]":
    """Exhaustive binary Hamming search on the FULL matrix.

    Quantizes both float32 inputs internally, computes the full Hamming
    distance matrix (n, 1), then selects the top
    ``top_k * oversample_factor`` candidates via argpartition + argsort.

    Args:
        matrix_f32: Float32 ``(n, dim)`` matrix of stored embedding vectors.
        q_f32: Float32 ``(dim,)`` query vector.
        top_k: Number of results to return (will not exceed ``n``).
        oversample_factor: Multiplier for the Stage-1 candidate pool size.
            The Stage 2 float32 re-ranker narrows this pool down to the
            final ``top_k``.

    Returns:
        ``(indices, distances)`` --- both sorted ascending by Hamming
        distance.  ``indices`` are row indices into ``matrix_f32``.
    """
    import numpy as np

    packed_matrix = quantize_1bit(matrix_f32)  # (n, B)
    packed_q = quantize_1bit(q_f32.reshape(1, -1))  # (1, B)
    dists = binary_hamming_distance(packed_matrix, packed_q)  # (n, 1)

    # Flatten from (n, 1) to (n,).
    dists_1d = dists.ravel()

    n = dists_1d.shape[0]
    pool_size = min(top_k * oversample_factor, n)
    if pool_size <= 0:
        return [], []

    # argpartition gives the `pool_size` smallest distances (unordered).
    part_idx = np.argpartition(dists_1d, pool_size - 1)[:pool_size]
    part_dists = dists_1d[part_idx]

    # Stable sort to get ascending order within the pool.
    sorted_loc = np.argsort(part_dists, kind="stable")
    final_idx = part_idx[sorted_loc].tolist()
    final_dists = part_dists[sorted_loc].tolist()

    return final_idx, final_dists


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    def _smoke() -> None:
        import numpy as np

        print("=== binary_quant smoke test ===")
        rng = np.random.RandomState(42)
        embs = rng.randn(10, 384).astype(np.float32)

        # 1-bit quantization shape + dtype.
        packed = quantize_1bit(embs)
        assert packed.dtype == np.uint8, f"expected uint8, got {packed.dtype}"
        assert packed.shape == (10, 48), f"expected (10,48), got {packed.shape}"
        print(f"  quantize_1bit: {embs.shape} -> {packed.shape} dtype={packed.dtype} OK")

        # matrix_to_binary alias matches quantize_1bit.
        packed2 = matrix_to_binary(embs)
        assert np.array_equal(packed, packed2), "matrix_to_binary must match quantize_1bit"
        print("  matrix_to_binary: matches quantize_1bit OK")

        # Identical vectors -> Hamming distance 0.
        d_same = binary_hamming_distance(packed[:1], packed[:1])
        assert d_same.shape == (1, 1), f"expected (1,1), got {d_same.shape}"
        assert int(d_same[0, 0]) == 0, (
            f"identical vectors must have distance 0, got {d_same[0, 0]}"
        )
        print(f"  identical Hamming distance: {int(d_same[0, 0])} OK")

        # Different vectors -> Hamming distance > 0.
        d_diff = binary_hamming_distance(packed[:1], packed[1:2])
        assert int(d_diff[0, 0]) > 0, (
            f"different vectors must have distance > 0, got {d_diff[0, 0]}"
        )
        print(f"  different Hamming distance: {int(d_diff[0, 0])} OK")

        # Full matrix-to-matrix distance (all-pairs).
        d_all_pairs = binary_hamming_distance(packed, packed)
        assert d_all_pairs.shape == (10, 10)
        # Diagonal (self-match) must be 0.
        assert np.all(np.diag(d_all_pairs) == 0), "diagonal must be 0"
        print(f"  all-pairs Hamming shape: {d_all_pairs.shape} OK")

        # binary_search returns the Stage-1 oversampled pool:
        # pool_size = min(top_k * oversample_factor, n) candidates.
        q = rng.randn(384).astype(np.float32)

        # Default oversample_factor=10 -> pool_size = min(3*10, 10) = 10.
        idx, d = binary_search(embs, q, top_k=3)
        assert len(idx) == 10, f"expected pool_size=10, got {len(idx)}"
        assert d[0] <= d[1] <= d[-1], f"results must be sorted ascending, got {d[:3]}..."
        print(f"  binary_search top_k=3 oversample=10: pool_size={len(idx)} min_dist={d[0]} OK")

        # top_k=10 oversample=1 -> pool_size = min(10, 10) = 10.
        idx_all, d_all = binary_search(embs, q, top_k=10)
        assert len(idx_all) == 10
        assert d_all[0] <= d_all[-1]
        print(f"  binary_search top_k=10: pool_size={len(idx_all)} min_dist={d_all[0]} max_dist={d_all[-1]} OK")

        # tight pool: oversample_factor=1, top_k=3 -> pool_size=3.
        idx_os, d_os = binary_search(embs, q, top_k=3, oversample_factor=1)
        assert len(idx_os) == 3
        print(f"  binary_search top_k=3 oversample=1: pool_size={len(idx_os)} min_dist={d_os[0]} OK")

        # Edge: empty matrix.
        empty = np.empty((0, 384), dtype=np.float32)
        idx_empty, d_empty = binary_search(empty, q, top_k=3)
        assert idx_empty == [] and d_empty == [], (
            f"empty matrix must return empty, got idx={idx_empty} dists={d_empty}"
        )
        print("  binary_search empty matrix: OK")

        print("=== smoke test PASSED ===")

    try:
        _smoke()
    except ImportError as e:
        print(f"SKIP (numpy not available): {e}")
        sys.exit(0)
