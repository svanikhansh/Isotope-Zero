"""Scale-adaptive vector search dispatcher.

THESIS
------
A single vector store can serve the regime where a small int8 SIMD dot kernel
beats float32 BLAS (small N, the typical local agent-memory hot path), *and*
the regime where float32 BLAS pulls ahead at scale (large-N archive scan), by
routing each query to whichever kernel is faster at the *current* active count.
The dispatcher holds both an int8-quantized and a float32 copy of the matrix in
one pre-allocated buffer set and switches the active path with hysteresis as N
crosses measured crossover boundaries.

EMPIRICAL CROSSOVER (measured in prototypes/simd_int8_v0.5, README table L104)
------------------------------------------------------------------------------
Per-query p99 latency, int8 native NEON dot vs float32 BLAS (BLAS / int8, ms):

    N cards |  f32 BLAS | native NEON | ratio  | winner
    --------|-----------|-------------|--------|--------------------
      500   |  0.064    |   0.011     | 0.18x  | NEON (5.8x faster)
     1000   |  0.059    |   0.021     | 0.36x  | NEON (2.8x faster)
     2000   |  0.061    |   0.042     | 0.69x  | NEON (1.4x faster)
     4000   |  0.069    |   0.087     | 1.26x  | BLAS
    10000   |  0.162    |   0.357     | 2.20x  | BLAS (2.2x faster)

=> true crossover is between 2k and 4k cards. The dispatch thresholds below
straddle that measured band deliberately.

HYSTERESIS THRESHOLDS
---------------------
    INT8_THRESHOLD = 2000   # N <= 2000 -> force int8 path (NEON decisively faster)
    BLAS_THRESHOLD = 3000   # N > 3000  -> force BLAS path (BLAS decisively faster)
                             # 2000 < N <= 3000 -> hysteresis: keep the latched path,
                             # only switch when a boundary is actually crossed.

HONEST ZERO-ALLOC FRAMING
-------------------------
- BLAS path: ``np.dot(_f32[:_n], _q_f32, out=_scores[:_n])`` writes the full
  (n,) score vector into the PRE-ALLOCATED ``_scores`` buffer. Zero allocation
  in the dot-product hot path. The only unavoidable alloc is the small (k,)
  ``argpartition`` index array for top-k -- numpy provides no zero-alloc top-k.
- Native int8 path: the compiled PyO3 kernel
  (``isotope_zero._native.simd_int8_batch_dot`` from prototypes/simd_int8_v0.5)
  does NOT accept an ``out=`` buffer -- its Rust signature returns a fresh
  ``numpy::PyArray1::from_vec`` each call. So the native path allocates one
  (n,) float32 return array per call (plus transient Rust-side input copies,
  documented in that prototype's README as a known zero-copy target). We copy
  the returned scores into ``_scores`` for the shared top-k step. This is
  strictly better than the synthesis_v1.0 baseline (which allocates a fresh
  scores array on *both* paths), but it is NOT "literally zero allocations" on
  the native path -- only on the BLAS path.
- Numpy int8 fallback (correctness reference, used only when the native .so is
  absent): upcasts the (n,d) int8 matrix to int32 before the matmul, allocating
  a transient (n,d) int32 buffer per call. This is deliberately a correctness
  oracle, not a performance path -- it is SLOWER than BLAS at every scale
  (numpy int8/int32 upcast is not SIMD), which is expected and fine.

THREAD-SAFETY
-------------
One ``threading.RLock`` per instance. ``search`` acquires it for the whole
compute (safe buffer reuse of ``_scores``/``_q_f32``/``_q_i8`` and exclusion of
concurrent writes). This SERIALIZES concurrent searches -- a read-write lock for
true read concurrency is documented as future work. The guarantee is "no
corruption, no torn reads, no buffer races", NOT "parallel reads".

CONVENTIONS (match prototypes/synthesis_v1.0)
---------------------------------------------
- L2-normalized float32 embeddings, dim=384 (all-MiniLM-L6-v2).
- ``from __future__ import annotations``; double-quoted strings; numpy
  local-imported inside the methods that use it.
- Returns ``list[(id, float)]`` sorted score desc with deterministic tie-break.
- Tie-break here: (score desc, slot index asc). NOTE: synthesis_v1.0 uses
  (score desc, timestamp asc, id asc); this dispatcher has no per-row timestamp,
  so slot index asc gives the same determinism guarantee.
- Scores are clamped to [0, 1] to match the synthesis store contract (raw
  normalized dot lives in [-1,1]; the contract floors negatives at 0).
"""

from __future__ import annotations

import logging
import threading
from importlib import util as _importlib_util
from typing import Any, Callable

log = logging.getLogger("adaptive_search")

# Absolute path to the compiled int8 SIMD kernel from prototypes/simd_int8_v0.5.
# The venv's editable install resolves the ``isotope_zero`` package to
# synthesis_v1.0 (which has no _native ext), so we load the .so by path instead
# of relying on ``import isotope_zero._native``. The .so is a self-contained
# Mach-O arm64 PyO3 cdylib; loading it by path works regardless of which
# prototype the editable install points at.
_NATIVE_SO_PATH = (
    "/Users/svanikhansh/Documents/isotope_zero/prototypes/simd_int8_v0.5/"
    "isotope_zero/_native.abi3.so"
)


def _l2_normalize(vec: "Any") -> "Any":
    """Return an L2-normalized float32 copy of ``vec`` (no-op if already unit)."""
    import numpy as np

    v = np.asarray(vec, dtype=np.float32)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        return v
    return v / nrm


def quantize_int8_symmetric(vec: "Any") -> "tuple[Any, float]":
    """Symmetric per-vector int8 quantization.

    scale = max(abs(vec)) / 127.0  (1.0 guard for the all-zero vector)
    q    = round(vec / scale).clip(-128, 127).astype(int8)

    Dequant reconstruction:  a . b ~= scale_a * scale_b * (qa . qb)
    where (qa . qb) is an int32 dot of two int8 vectors.

    Clip range is [-128, 127] to match prototypes/simd_int8_v0.5's
    quantization.py and the native Rust kernel (signed int8 two's-complement
    range). The design brief wrote [-127, 127]; both are overflow-safe for the
    int32 accumulator, but matching the proven kernel makes the native and
    numpy paths bit-identical.
    """
    import numpy as np

    v = np.asarray(vec, dtype=np.float32)
    maxabs = float(np.max(np.abs(v))) if v.size else 0.0
    scale = maxabs / 127.0
    if scale <= 0.0:
        scale = 1.0  # div-by-zero guard for the all-zero vector
    q = np.round(v / scale).clip(-128, 127).astype(np.int8)
    return q, scale


def _try_import_native_kernel() -> "Callable[..., Any] | None":
    """Load the compiled int8 SIMD dot kernel if its .so is on disk.

    Returns the ``simd_int8_batch_dot`` callable, or None if the .so is absent
    or fails to load (silent at info level -- the numpy fallback is then used).

    The kernel signature (confirmed against the .so this session):
        simd_int8_batch_dot(matrix_i8 (n,d) int8, query_i8 (d,) int8,
                             scales (n,) f32, query_scale float)
            -> numpy float32 (n,) array of raw approximate dot products
               (out[i] = scales[i] * query_scale * sum_j matrix[i,j]*query[j])

    NOTE: the kernel does NOT accept an ``out=`` keyword. It returns a fresh
    (n,) float32 array each call. The native path therefore allocates one
    (n,) array per search (documented in the module docstring).
    """
    import os

    if not os.path.exists(_NATIVE_SO_PATH):
        log.debug("native int8 .so not found at %s; using numpy fallback", _NATIVE_SO_PATH)
        return None
    try:
        # The .so's PyInit symbol is baked as PyInit__native (matching the Rust
        # #[pymodule] name in simd_int8_v0.5). CPython resolves the init symbol
        # from the module name we load under, so we MUST load under the exact
        # name "_native" -- any other name yields
        # "dynamic module does not define module export function (PyInit__<name>)".
        # We deliberately do NOT insert into sys.modules (avoids shadowing any
        # real isotope_zero._native the venv might later resolve); the module
        # object just lives in this function's local scope.
        spec = _importlib_util.spec_from_file_location("_native", _NATIVE_SO_PATH)
        if spec is None or spec.loader is None:
            log.debug("could not build import spec for %s", _NATIVE_SO_PATH)
            return None
        mod = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        fn = getattr(mod, "simd_int8_batch_dot", None)
        if fn is None:
            log.debug(".so loaded but simd_int8_batch_dot attr missing")
            return None
        log.info("native int8 kernel loaded: %s", _NATIVE_SO_PATH)
        return fn  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001 -- any load failure -> fallback
        log.debug("native int8 kernel load failed (%s); using numpy fallback", exc)
        return None


def int8_dot_numpy(
    matrix_i8: "Any",
    q_i8: "Any",
    scales: "Any",
    q_scale: float,
    out: "Any",
) -> None:
    """Correctness-reference int8 dot product (numpy fallback).

    Writes ``out[:n] = scales[:n] * q_scale * int_dot`` as float32, where
    ``int_dot = matrix_i8.astype(int32) @ q_i8.astype(int32)``.

    DEVITATION FROM THE DESIGN BRIEF: the brief specified
    ``np.dot(int16) -> int32``. That OVERFLOWS -- for dim=384 and worst-case
    |127*127|=16129 per element, the per-row sum is 384*127*127 = 6,193,536,
    which exceeds int16's max of 32767 and produces silent wraparound to
    negative values. We use int32 accumulators, matching the proven
    ``quantization.py::int8_dot_product`` in prototypes/simd_int8_v0.5 and the
    native Rust kernel (i32 per row). This is overflow-safe for any dim <=
    ~16.7M at max int8 magnitude, and makes the fallback bit-identical to the
    native path (verified to ~1e-9 this session).

    ALLOCATES a transient (n,d) int32 buffer per call -- this is why the
    fallback is a correctness reference, not a performance path. The native
    kernel avoids this by reading int8 zero-copy and accumulating in Rust.
    """
    import numpy as np

    n = int(matrix_i8.shape[0])
    if n == 0:
        return
    # int32 accumulator: overflow-safe for dim*127*127 at dim=384 (~6.2M, well
    # under int32 max ~2.1e9). int16 would overflow here (verified).
    raw = matrix_i8.astype(np.int32) @ q_i8.astype(np.int32)  # (n,) int32, allocates
    out[:n] = raw.astype(np.float32) * scales[:n].astype(np.float32) * np.float32(q_scale)


class AdaptiveVectorSearch:
    """A scale-adaptive vector store that routes int8 vs BLAS by active count.

    Holds one float32 and one int8 copy of the matrix in a single pre-allocated
    buffer set; ``search`` picks the active path from the current N against the
    hysteresis thresholds, latches the decision, and runs the dot product into
    a reusable scores buffer. See the module docstring for the empirical
    crossover basis and the honest zero-alloc + thread-safety framing.
    """

    def __init__(
        self,
        dim: int = 384,
        capacity: int = 1024,
        int8_threshold: int = 2000,
        blas_threshold: int = 3000,
        use_native: bool = True,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if int8_threshold >= blas_threshold:
            raise ValueError("int8_threshold must be < blas_threshold")
        import numpy as np

        self._dim = int(dim)
        self._cap = int(capacity)
        self._int8_threshold = int(int8_threshold)
        self._blas_threshold = int(blas_threshold)
        # Matrix buffers (CAP, D). _f32 for BLAS; _i8 + _scales for int8.
        self._f32 = np.zeros((self._cap, self._dim), dtype=np.float32)
        self._i8 = np.zeros((self._cap, self._dim), dtype=np.int8)
        self._scales = np.zeros((self._cap,), dtype=np.float32)
        # External id per slot, and reverse map for O(1) delete.
        self._ids: list[Any] = [None] * self._cap
        self._id2slot: dict[Any, int] = {}
        # Free-slot stack for O(1) reuse; active count; latched path.
        self._free: list[int] = []
        self._n = 0
        self._path = "int8"
        # Pre-allocated per-query REUSABLE buffers (zero alloc in the hot path).
        self._scores = np.zeros((self._cap,), dtype=np.float32)
        self._q_f32 = np.zeros((self._dim,), dtype=np.float32)
        self._q_i8 = np.zeros((self._dim,), dtype=np.int8)
        self._q_scale = 0.0
        # One RLock per instance: serializes searches for correctness (see
        # docstring -- not parallel reads; RWLock is future work).
        self._lock = threading.RLock()
        # Native kernel (zero-alloc-ish fast path) or None (numpy fallback).
        self._kernel: "Callable[..., Any] | None" = (
            _try_import_native_kernel() if use_native else None
        )

    # -- capacity management ------------------------------------------------

    def _grow(self, new_cap: int) -> None:
        """Geometrically grow both matrices + scales + scores to ``new_cap``."""
        import numpy as np

        f32 = np.zeros((new_cap, self._dim), dtype=np.float32)
        i8 = np.zeros((new_cap, self._dim), dtype=np.int8)
        scales = np.zeros((new_cap,), dtype=np.float32)
        scores = np.zeros((new_cap,), dtype=np.float32)
        if self._n > 0:
            f32[: self._n] = self._f32[: self._n]
            i8[: self._n] = self._i8[: self._n]
            scales[: self._n] = self._scales[: self._n]
            scores[: self._n] = self._scores[: self._n]
        self._f32 = f32
        self._i8 = i8
        self._scales = scales
        self._scores = scores
        self._ids = self._ids + [None] * (new_cap - self._cap)
        self._cap = new_cap

    def _ensure_capacity(self) -> None:
        """Double capacity when active count hits the current capacity."""
        if self._n >= self._cap:
            self._grow(max(self._cap * 2, self._n + 1))

    # -- write paths --------------------------------------------------------

    def add(self, vec: "Any", id: "Any") -> None:
        """Add ``vec`` under ``id`` (O(1), amortized; overwrites if id exists).

        ``vec`` is L2-normalized before storage so the BLAS dot is a cosine.
        The int8 quantization of the normalized vec is stored alongside.
        """
        import numpy as np

        nrm = _l2_normalize(vec)
        with self._lock:
            if id in self._id2slot:
                slot = self._id2slot[id]
            else:
                self._ensure_capacity()
                if self._free:
                    slot = self._free.pop()
                else:
                    slot = self._n
                    self._n += 1
                self._id2slot[id] = slot
            self._f32[slot] = nrm
            q, scale = quantize_int8_symmetric(nrm)
            self._i8[slot] = q
            self._scales[slot] = scale
            self._ids[slot] = id

    def remove(self, id: "Any") -> bool:
        """Remove ``id`` via swap-and-pop so rows stay packed in [0, _n).

        O(1). Returns False if ``id`` is not present.
        """
        with self._lock:
            if id not in self._id2slot:
                return False
            slot = self._id2slot.pop(id)
            last = self._n - 1
            if slot != last:
                # Move the last active row into the freed slot to keep [0,_n)
                # contiguous (the dot product scans [:_n]).
                self._f32[slot] = self._f32[last]
                self._i8[slot] = self._i8[last]
                self._scales[slot] = self._scales[last]
                moved_id = self._ids[last]
                self._ids[slot] = moved_id
                self._id2slot[moved_id] = slot
            # Clear the vacated last slot (hygiene; not strictly required).
            self._ids[last] = None
            self._n = last
            return True

    # -- search -------------------------------------------------------------

    def search(self, query: "Any", k: int = 5) -> "list[tuple[Any, float]]":
        """Top-k search, routing to int8 or BLAS by active count with hysteresis.

        Path decision:
            N <= INT8_THRESHOLD(2000) -> "int8"
            N >  BLAS_THRESHOLD(3000) -> "blas"
            else -> keep ``self._path`` (hysteresis; switch only on boundary cross)

        The whole compute holds ``self._lock`` so the reusable ``_scores`` /
        ``_q_f32`` / ``_q_i8`` buffers are safe from concurrent writes and each
        other. Returns ``[(id, score)]`` sorted score desc; ties broken by slot
        index asc for determinism. Scores clamped to [0, 1].
        """
        import numpy as np

        with self._lock:
            n = self._n
            if n == 0 or k <= 0:
                return []
            # Stage the query into reusable buffers (no alloc in steady state).
            qn = _l2_normalize(query)
            self._q_f32[:] = qn
            qi8, qscale = quantize_int8_symmetric(qn)
            self._q_i8[:] = qi8
            self._q_scale = qscale

            # Dispatch with hysteresis.
            if n <= self._int8_threshold:
                path = "int8"
            elif n > self._blas_threshold:
                path = "blas"
            else:
                path = self._path  # hysteresis: keep the latched path
            self._path = path  # latch

            if path == "blas":
                # Zero-alloc dot: np.dot(matrix, query, out=_scores[:n]).
                np.dot(self._f32[:n], self._q_f32, out=self._scores[:n])
            else:
                if self._kernel is not None:
                    # Native kernel returns a fresh (n,) array (no out= kwarg);
                    # copy into the reusable scores buffer.
                    raw = self._kernel(
                        self._i8[:n], self._q_i8, self._scales[:n], self._q_scale
                    )
                    self._scores[:n] = raw
                else:
                    int8_dot_numpy(
                        self._i8[:n], self._q_i8, self._scales[:n], self._q_scale,
                        self._scores,
                    )
            scores = self._scores[:n]
            # Clamp to [0,1] to match the synthesis store contract.
            np.clip(scores, 0.0, 1.0, out=scores)

            k_eff = min(k, n)
            if k_eff == 0:
                return []
            # Top-k: argpartition (small (k,) alloc, unavoidable in numpy),
            # then argsort the candidates desc for deterministic ordering.
            idx = np.argpartition(scores, -k_eff)[-k_eff:]
            order = idx[np.argsort(scores[idx])[::-1]]
            # Tie-break: score desc (above) then slot index asc (stable under
            # equal scores since idx ascends and argsort is stable).
            return [(self._ids[int(j)], float(scores[int(j)])) for j in order]

    # -- introspection ------------------------------------------------------

    @property
    def n(self) -> int:
        """Active vector count."""
        return self._n

    @property
    def path(self) -> str:
        """Last-latched dispatch path ("int8" or "blas")."""
        return self._path

    @property
    def capacity(self) -> int:
        """Current allocated capacity (grows geometrically)."""
        return self._cap

    def stats(self) -> "dict[str, Any]":
        """Return {n, capacity, path, native_kernel_loaded}."""
        return {
            "n": self._n,
            "capacity": self._cap,
            "path": self._path,
            "native_kernel_loaded": self._kernel is not None,
        }


def _smoke() -> None:
    """Inline smoke test: add 1000 random vecs, search, print routing + results."""
    import numpy as np

    rng = np.random.default_rng(1337)
    s = AdaptiveVectorSearch(dim=384, capacity=1024)
    print("stats init:", s.stats())
    for i in range(1000):
        v = rng.standard_normal(384).astype(np.float32)
        v /= np.linalg.norm(v)
        s.add(v, f"card-{i}")
    q = rng.standard_normal(384).astype(np.float32)
    q /= np.linalg.norm(q)
    res = s.search(q, k=5)
    print("stats after 1000 adds:", s.stats())
    print("top-5 results (id, score):")
    for rid, sc in res:
        print(f"  {rid}: {sc:.6f}")
    # Sanity: results sorted desc.
    scs = [sc for _, sc in res]
    assert scs == sorted(scs, reverse=True), "results not sorted desc"
    # Sanity: all scores in [0,1].
    assert all(0.0 <= sc <= 1.0 for _, sc in res), "score out of [0,1]"
    # Routing: N=1000 <= 2000 -> int8.
    assert s.path == "int8", f"expected int8 at N=1000, got {s.path}"
    print("OK: smoke passed (routing=int8 at N=1000, scores sorted desc & in [0,1])")


if __name__ == "__main__":
    _smoke()
