"""Vector + negation kernels for the isotope_zero retrieval / triage paths.

As of v1.3.0 the package ships a **pure-Python wheel** — there is no Rust
native extension. Both workloads that once had a Rust fast path now run
entirely in Python, with no behavior change for any caller:

* **Vector dot-products** run on **NumPy/BLAS** (``matrix @ query``). BLAS
  operates zero-copy on the numpy buffers and releases the GIL around the C
  kernel — measured ~9-115x faster than the Rust extension's
  ``batch_cosine_similarity_matrix``, which has to clone the matrix into
  owned ``Vec<f32>`` to release the GIL (``PyReadonlyArray`` borrows Python
  memory and is not ``Send``). The float32 cosine workload always won on
  NumPy/BLAS, so removing the Rust path loses nothing here.
* **Negation / polarity detection** runs on the **pure-Python heuristic**
  (``_are_negations_py``): a verbatim copy of the v0.1 reference logic. The
  Rust ``are_negations`` port was bit-for-bit identical, so output is
  unchanged whether or not a native module was ever present.

``HAVE_NATIVE`` is retained as a module-level constant (always ``False``) for
API stability — downstream adapters or user code may read it. The historic
``isotope_zero._native`` extension (compiled by maturin from
``rust_bridge/``) was removed in v1.3.0; the int8 NEON measurement that
motivated it lives on as a frozen research artifact in
``prototypes/simd_int8_v0.5``.
"""

from __future__ import annotations

# v1.3.0: the Rust ``isotope_zero._native`` extension was removed. There is no
# native module to import; ``HAVE_NATIVE`` is a constant kept for API stability
# (downstream code may branch on it). Negation now always runs the pure-Python
# heuristic, which is byte-for-byte the v0.1 reference logic the Rust port
# reproduced.
HAVE_NATIVE: bool = False

__all__ = ["HAVE_NATIVE", "batch_cosine_similarity", "are_negations"]


# --------------------------------------------------------------------------- #
# Vector path — NumPy/BLAS is the (sole) path
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

    The float32 batch dot-product runs on **NumPy/BLAS**: ``matrix @ q``
    calls into the platform BLAS (Accelerate on macOS, OpenBLAS elsewhere)
    with **zero copy** of the numpy buffers and releases the GIL around the C
    kernel. Hand SIMD cannot beat zero-copy BLAS at these sizes, so the
    vector workload stays on the path that wins. (The historic Rust
    ``batch_cosine_similarity_matrix`` parity probe was removed in v1.3.0
    along with the crate; see the module docstring.)
    """
    import numpy as np

    # NumPy/BLAS primary path: raw dot product, NO clipping (caller clips).
    # np.matmul on float32 inputs yields float32 output.
    return np.asarray(matrix @ query_vec, dtype=np.float32)


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

    Runs the pure-Python heuristic (``_are_negations_py``), which is
    byte-for-byte the v0.1 reference logic. The historic Rust
    ``_native.are_negations`` port was bit-identical and was removed with the
    crate in v1.3.0; callers see no behavior change.
    """
    return _are_negations_py(a, b)
