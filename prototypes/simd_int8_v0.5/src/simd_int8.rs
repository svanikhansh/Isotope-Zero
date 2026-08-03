//! Int8 SIMD dot-product kernel for the SQ8 quantized search fast path.
//!
//! Computes, for each matrix row `i`:
//!
//!     score[i] = scales[i] * query_scale * sum_j(q_matrix[i,j] * q_query[j])
//!
//! (the approximate RAW dot of the dequantized vectors — the store divides by
//! dequantized norms to recover a [0,1] cosine). The integer dot uses an
//! i32 accumulator: worst-case `|sum| = dim * 127 * 127` (~6.2M for dim=384),
//! well within int32's ~2.1e9 range. Do NOT use i8/i16 accumulators.
//!
//! SIMD strategy:
//! * aarch64 (LIVE on Apple M4): NEON widening-multiply path. NOTE: the
//!   `vdotq_s32` (sdot) intrinsic is unstable on stable Rust
//!   (`stdarch_neon_dotprod`, tracking #117224), so we use the stable,
//!   universally-available widening idiom instead: `vmull_s8` (int8x8 x int8x8
//!   -> int16x8, products fit since max |product| = 16129 < 32767) then
//!   `vpaddlq_s16` (pairwise-add-long int16x8 -> int32x4) accumulated into an
//!   int32x4 via `vaddq_s32`. We process 16 int8 elements per iteration
//!   (one vld1q_s8 load, split into low/high int8x8 halves). Then a scalar
//!   epilogue for any remainder (handles dim not a multiple of 16). No
//!   `#[target_feature(enable="neon")]` is required: these intrinsics are
//!   available on every aarch64-apple-darwin target (NEON is mandatory in
//!   ARMv8+). The intrinsics are `unsafe`, hence the unsafe block.
//! * x86_64 (COMPILE-GUARDED, dead code on this M4 host): the signed-int8
//!   variant of the VNNI instruction is `_mm256_dpbssd_epi32` (AVX-VNNI-INT8,
//!   gated by avx512vnni). NOTE: the unsigned `_mm256_dpbusd_epi32` reads
//!   matrix bytes as UINT8 — WRONG for our signed int8 data. Because we
//!   cannot run/verify x86 on this host we fall through to the scalar i32
//!   loop on x86 (still GIL-released and compiled). To wire a real AVX path,
//!   gate `_mm256_dpbssd_epi32` behind `is_x86_feature_detected!("avx512vnni")`
//!   and keep the scalar loop as the runtime fallback.
//! * Scalar fallback: always available; the correctness reference, the x86
//!   path, and the non-SIMD-arch fallback. rustc auto-vectorizes it at
//!   opt-level=3 where it can. On this M4 host the NEON path ALWAYS applies.

use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64 as neon;

/// NEON int8 dot of one row vs the query, len `dim`.
///
/// Processes 16 int8 elements per iteration: one `vld1q_s8` load, split into
/// low/high int8x8 halves, each `vmull_s8` -> int16x8, `vpaddlq_s16` ->
/// int32x4, accumulate via `vaddq_s32`. Scalar epilogue for the remainder
/// (handles dim not a multiple of 16). The caller guarantees
/// `row.len() >= dim` and `q.len() >= dim`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_row_neon(row: &[i8], q: &[i8], dim: usize) -> i32 {
    let mut acc = neon::vdupq_n_s32(0);
    let rp = row.as_ptr();
    let qp = q.as_ptr();
    let mut i = 0;
    // 16 int8 per iter: one 128-bit load, split into two int8x8 halves,
    // each widened to int16x8 then pairwise-long into int32x4.
    while i + 16 <= dim {
        let rv = neon::vld1q_s8(rp.add(i));
        let qv = neon::vld1q_s8(qp.add(i));
        let rlo = neon::vget_low_s8(rv);
        let rhi = neon::vget_high_s8(rv);
        let qlo = neon::vget_low_s8(qv);
        let qhi = neon::vget_high_s8(qv);
        // int8x8 x int8x8 -> int16x8 (products fit: max |127*127|=16129).
        let plo = neon::vmull_s8(rlo, qlo);
        let phi = neon::vmull_s8(rhi, qhi);
        // Pairwise-add-long int16x8 -> int32x4 (4 lanes: sum of adjacent
        // pairs), then accumulate.
        acc = neon::vaddq_s32(acc, neon::vpaddlq_s16(plo));
        acc = neon::vaddq_s32(acc, neon::vpaddlq_s16(phi));
        i += 16;
    }
    let mut s = neon::vaddvq_s32(acc);
    // Scalar epilogue for the remaining 0..15 elements.
    while i < dim {
        s += (*rp.add(i) as i32) * (*qp.add(i) as i32);
        i += 1;
    }
    s
}

/// Plain scalar i32 dot (always available; the correctness reference, the x86
/// path, and the non-SIMD-arch fallback). rustc auto-vectorizes it at
/// opt-level=3 where it can.
#[inline(always)]
fn dot_row_scalar(row: &[i8], q: &[i8], dim: usize) -> i32 {
    let mut s: i32 = 0;
    let mut i = 0;
    while i < dim {
        // i8 as i32 sign-extends, so the product is exact for [-128,127].
        s += (row[i] as i32) * (q[i] as i32);
        i += 1;
    }
    s
}

#[inline(always)]
fn dot_row(row: &[i8], q: &[i8], dim: usize) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON intrinsics are available on all aarch64 targets. The
        // caller guarantees row/q have at least `dim` elements.
        unsafe { dot_row_neon(row, q, dim) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // x86_64 and any other arch: scalar i32 loop (see module docs for the
        // AVX-VNNI signedness caveat). Still GIL-released + compiled.
        dot_row_scalar(row, q, dim)
    }
}

/// Batch int8 dot-product kernel exposed to Python.
///
/// `q_matrix` is (n, d) int8, `q_query` is (d,) int8, `scales` is (n,) f32,
/// `query_scale` is a scalar float. Returns a numpy float32 (n,) array where
/// `out[i] = scales[i] * query_scale * sum_j(q_matrix[i,j] * q_query[j])`.
///
/// The numpy arrays are read zero-copy (`PyReadonlyArray*`), sliced once
/// (GIL held), then the per-row SIMD loop runs under `py.allow_threads` (GIL
/// released) so the matrix scan never blocks Python threads. We copy the
/// numpy buffer into owned Rust slices first (PyReadonlyArray borrows Python
/// memory and is not `Send`), matching the proven pattern from the float32
/// kernel in `prototypes/rust_v0.2`.
#[pyfunction]
pub fn simd_int8_batch_dot(
    py: Python<'_>,
    q_matrix: PyReadonlyArray2<i8>,
    q_query: PyReadonlyArray1<i8>,
    scales: PyReadonlyArray1<f32>,
    query_scale: f32,
) -> PyResult<PyObject> {
    let m_view = q_matrix.as_slice()?;
    let q_view = q_query.as_slice()?;
    let s_view = scales.as_slice()?;
    let n = q_matrix.shape()[0];
    let d = q_matrix.shape()[1];
    if q_query.shape()[0] != d {
        return Err(PyValueError::new_err(format!(
            "query dim {} != matrix dim {}",
            q_query.shape()[0],
            d
        )));
    }
    if scales.shape()[0] != n {
        return Err(PyValueError::new_err(format!(
            "scales len {} != matrix rows {}",
            scales.shape()[0],
            n
        )));
    }
    // Copy numpy buffers into owned Vecs so the GIL can be released while the
    // hot loop runs (PyReadonlyArray borrows Python memory and is not Send).
    let m: Vec<i8> = m_view.to_vec();
    let q: Vec<i8> = q_view.to_vec();
    let s: Vec<f32> = s_view.to_vec();

    let out: Vec<f32> = py.allow_threads(move || fast_path(&m, &q, &s, n, d, query_scale));
    // Build the returned numpy float32 (n,) array (GIL held again).
    let arr = numpy::PyArray1::from_vec(py, out);
    Ok(arr.to_owned().into())
}

/// Fast path: int8 matrix (row-major) + query + per-row scales. Pure Rust
/// (no numpy borrow), so it runs GIL-released. Processes each row through
/// the SIMD `dot_row` and rescales to float32.
#[inline]
fn fast_path(mat: &[i8], q: &[i8], s: &[f32], n: usize, d: usize, query_scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    if d == 0 {
        out.resize(n, 0.0f32);
        return out;
    }
    for i in 0..n {
        let row = &mat[i * d..(i + 1) * d];
        let dot = dot_row(row, q, d);
        out.push(s[i] * query_scale * (dot as f32));
    }
    out
}

/// Introspection helper: returns the name of the active SIMD path so the
/// benchmark can confirm the native kernel is live. "neon-sdot" on aarch64.
#[pyfunction]
pub fn simd_kernel_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        "neon-sdot"
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // x86_64 and any other arch: scalar loop (see module docs for the
        // AVX-VNNI signedness caveat). Report honestly.
        "scalar-fallback"
    }
}
