"""Smart Bridge between the pure-Python v0.1 engine and the Rust native core.

Phase 6 places each workload on its empirically-fastest path:

* **Vector dot-products** run on **NumPy/BLAS** (``matrix @ query``). BLAS
  operates zero-copy on the numpy buffers and releases the GIL around the C
  kernel; the Rust extension's ``batch_cosine_similarity_matrix`` has to clone
  the matrix into owned ``Vec<f32>`` to release the GIL (``PyReadonlyArray``
  borrows Python memory and is not ``Send``), which is ~9-115x slower. So the
  float32 cosine workload stays where it already wins.
* **Negation / polarity detection** runs on the **Rust native core**
  (``isotope_zero._native.are_negations``): a bit-for-bit port of the v0.1
  heuristic, compiled and GIL-released, with a clean pure-Python fallback.

The bridge is designed so that a **missing, half-built, or ABI-mismatched**
native module can never break the package:

* Import of ``isotope_zero._native`` is wrapped in a bare ``except`` that
  swallows every failure (``ImportError``, ``AttributeError``, ``OSError``,
  ``TypeError`` from a stale ABI, ...) and records the outcome in
  ``HAVE_NATIVE``.
* ``are_negations`` tries the native implementation first when ``HAVE_NATIVE``
  is True and transparently falls back to the pure-Python equivalent on any
  native failure.
* The pure-Python fallbacks are exact copies of the v0.1 reference logic, so
  negation behavior is bit-identical whether or not the native module loads.

Native API contract (single source of truth, mirrored in src/lib.rs):

    isotope_zero._native.batch_cosine_similarity(
        query: Vec<f32>, matrix_flat: Vec<f32>, dim: usize) -> list[float]
    isotope_zero._native.batch_cosine_similarity_matrix(
        query: numpy (dim,) float32, matrix: numpy (n, dim) float32) -> list[float]
    isotope_zero._native.are_negations(a: str, b: str) -> bool

The vector functions exist for parity probes / future quantized-SIMD work but
are NOT on the hot path; ``batch_cosine_similarity`` here always uses NumPy.
"""

from __future__ import annotations

try:  # noqa: E402  (import-time safety barrier is intentional)
    from isotope_zero import _native  # type: ignore[import-not-found]

    HAVE_NATIVE: bool = True
except Exception:  # noqa: BLE001 — a broken native module must never break import
    _native = None  # type: ignore[assignment]
    HAVE_NATIVE = False

__all__ = [
    "HAVE_NATIVE",
    "batch_cosine_similarity",
    "are_negations",
    "popcnt_hamming_search",
]


# --------------------------------------------------------------------------- #
# Vector path — NumPy/BLAS is the PRIMARY path (Smart Bridge)
# --------------------------------------------------------------------------- #
def batch_cosine_similarity(
    query_vec: "np.ndarray", matrix: "np.ndarray"
) -> "np.ndarray":
    """Raw (unclipped) cosine/dot scores of `query_vec` against every row.

    ``query_vec`` is a ``(dim,)`` float32 vector and ``matrix`` is a
    ``(n, dim)`` float32 array (rows L2-normalized by the embedder, so
    cosine == dot product). Returns a ``(n,)`` float32 array of RAW scores;
    the caller is responsible for clipping to [0, 1] and top-k selection, so
    every path is semantically identical.

    Smart Bridge routing (see Phase 6 report): the float32 batch dot-product
    runs on the **NumPy/BLAS** path, not the Rust extension. ``matrix @ q``
    calls into the platform BLAS (Accelerate on macOS) with **zero copy** of
    the numpy buffers and releases the GIL around the C kernel — measured
    ~9-115x faster than the Rust path, which had to clone the matrix into
    owned ``Vec<f32>`` to release the GIL (PyReadonlyArray borrows Python
    memory and is not ``Send``). Hand SIMD cannot beat zero-copy BLAS at
    these sizes, so the vector workload stays on the path that already wins.

    The Rust extension still exposes ``batch_cosine_similarity_matrix`` for
    parity probes and future quantized-SIMD work; it is not on the hot path.
    """
    import numpy as np

    # NumPy/BLAS primary path: raw dot product, NO clipping (caller clips).
    # np.matmul on float32 inputs yields float32 output.
    return np.asarray(matrix @ query_vec, dtype=np.float32)


# --------------------------------------------------------------------------- #
# POPCNT Hamming search path — Rust native primary, NumPy fallback
# --------------------------------------------------------------------------- #
def popcnt_hamming_search(
    matrix: "np.ndarray",
    q: "np.ndarray",
    top_k: int,
    oversample: int = 10,
) -> "tuple[list[int], list[int]]":
    """POPCNT-accelerated Hamming distance search over 1-bit binary vectors.

    Takes **already-packed** uint8 arrays: ``matrix`` of shape ``(n, 48)``
    and ``q`` of shape ``(48,)``.  Returns ``(indices, distances)`` sorted
    ascending by Hamming distance with ``min(top_k * oversample, n)``
    candidates.

    Tries ``_native.popcnt_hamming_search`` when the native module is
    present; on any failure (or when absent) falls back to the pure-Python /
    NumPy implementation via ``binary_quant``.
    """
    if HAVE_NATIVE:
        try:
            return _native.popcnt_hamming_search(
                matrix, q, top_k, oversample
            )
        except Exception:  # noqa: BLE001
            pass
    # Fallback: use the pure-Python binary Hamming distance.
    from isotope_zero.core.binary_quant import binary_hamming_distance

    if matrix.shape[0] == 0 or top_k <= 0:
        return [], []

    import numpy as np

    n = matrix.shape[0]
    target = min(top_k * oversample, n)
    # Binary Hamming distance: (n, 48) vs (1, 48) → (n, 1)
    dists = binary_hamming_distance(matrix, q.reshape(1, -1))
    dists_1d = dists.ravel()

    part_idx = np.argpartition(dists_1d, target - 1)[:target]
    part_dists = dists_1d[part_idx]
    sorted_loc = np.argsort(part_dists, kind="stable")
    final_idx = part_idx[sorted_loc].tolist()
    final_dists = part_dists[sorted_loc].tolist()
    return final_idx, final_dists


# --------------------------------------------------------------------------- #
# Negation path
# --------------------------------------------------------------------------- #
# Negation markers — verbatim copy from isotope_zero/core/consolidation.py.
# If one fact asserts X and another asserts NOT-X they are semantically
# opposite and must NEVER be merged, even when their embeddings are nearly
# identical.
_NEGATION_MARKERS: tuple[str, ...] = (
    "not", "no longer", "doesn't", "does not", "don't", "do not",
    "never", "isn't", "is not", "wasn't", "was not", "won't", "will not",
    "cannot", "can't", "neither", "nor", "without", "lacks", "stopped",
    "quit", "no more",
)


def _strip_negations(text: str) -> tuple[str, bool]:
    """Return (text-with-negation-markers-removed, was_any_negation_found).

    Lowercases and strips whitespace for comparison only; the returned text
    is only used to judge polarity, never stored.
    """
    t = " " + text.lower().strip() + " "
    found = False
    for marker in sorted(_NEGATION_MARKERS, key=len, reverse=True):
        needle = " " + marker + " "
        if needle in t:
            found = True
            t = t.replace(needle, " ")
    t = " ".join(t.split())
    return t, found


def _stem(tok: str) -> str:
    """Crude suffix stemmer for negation comparison only.

    Strips a trailing 'ing'/'ed'/'es'/'s' so morphological variants of the
    same verb ("uses"/"use"/"using") collapse to a common stem. Deliberately
    crude — it is only used to judge polarity equality, never stored, and a
    false collapse just means two negations are compared a little more
    liberally (which errs toward caution: not-merging).
    """
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def _are_negations_py(a: str, b: str) -> bool:
    """Pure-Python negation heuristic (verbatim from consolidation.py).

    True if `a` and `b` assert opposite polarities of the same fact:
    after removing negation markers from both sides, exactly one side
    originally contained a negation AND the denegated token sets overlap
    highly (>= 0.6 Jaccard after a crude stem).
    """
    if not a or not b:
        return False
    ta, neg_a = _strip_negations(a)
    tb, neg_b = _strip_negations(b)
    # Need a polarity difference: exactly one side is negated.
    if neg_a == neg_b:
        return False
    # The denegated texts must be near-identical (same core assertion).
    sa = {_stem(t) for t in ta.split()}
    sb = {_stem(t) for t in tb.split()}
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= 0.6


def are_negations(a: str, b: str) -> bool:
    """True if `a` and `b` assert opposite polarities of the same fact.

    Tries ``_native.are_negations`` when the native module is present; on any
    native failure (or when absent) falls back to the pure-Python heuristic,
    which is byte-for-byte the v0.1 reference logic.
    """
    if HAVE_NATIVE:
        try:
            return bool(_native.are_negations(a, b))
        except Exception:  # noqa: BLE001 — never propagate a native failure
            pass
    return _are_negations_py(a, b)
