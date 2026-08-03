//! Batch dot-product core for the vector-similarity fast path.
//!
//! Input rows and the query are L2-normalized upstream, so `dot == cosine`;
//! we compute raw (unclamped) scores, exactly like the numpy reference path
//! that the caller clips to [0, 1].
//!
//! Performance strategy (measured on Apple M4, 10 cores):
//! * aarch64: explicit NEON with 4 independent 128-bit FMA accumulators
//!   (16 floats per iteration; 384 = 24 full iterations), remainder handled
//!   with a 4-wide FMA tail + scalar epilogue.
//! * other targets: a plain contiguous loop that rustc auto-vectorizes at
//!   opt-level=3 (AVX2/AVX-512 on x86-64).
//! * batches large enough to amortize split overhead run over rows with
//!   rayon; smaller batches stay single-threaded (measured crossover).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64 as neon;

use rayon::prelude::*;

/// Above this many row-elements (`rows * dim`) we parallelize over rows with
/// rayon; below it a single thread wins. Tuned empirically — see the timing
/// probe results in the build report.
const PAR_MIN_ELEMENTS: usize = 1 << 18; // 262_144

/// Dot product of `row[0..dim]` against `q[0..dim]`.
///
/// `dim` is clamped to `q.len()` by the caller so `q` is never over-read.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn row_dot(q: &[f32], row: &[f32], dim: usize) -> f32 {
    unsafe {
        let mut acc0 = neon::vdupq_n_f32(0.0);
        let mut acc1 = neon::vdupq_n_f32(0.0);
        let mut acc2 = neon::vdupq_n_f32(0.0);
        let mut acc3 = neon::vdupq_n_f32(0.0);
        let qp = q.as_ptr();
        let rp = row.as_ptr();
        let mut i = 0;
        // 4 x 128-bit FMA per iteration -> 16 floats; 4 independent
        // accumulators keep the dependency chains short.
        while i + 16 <= dim {
            let q0 = neon::vld1q_f32(qp.add(i));
            let q1 = neon::vld1q_f32(qp.add(i + 4));
            let q2 = neon::vld1q_f32(qp.add(i + 8));
            let q3 = neon::vld1q_f32(qp.add(i + 12));
            let r0 = neon::vld1q_f32(rp.add(i));
            let r1 = neon::vld1q_f32(rp.add(i + 4));
            let r2 = neon::vld1q_f32(rp.add(i + 8));
            let r3 = neon::vld1q_f32(rp.add(i + 12));
            acc0 = neon::vfmaq_f32(acc0, q0, r0);
            acc1 = neon::vfmaq_f32(acc1, q1, r1);
            acc2 = neon::vfmaq_f32(acc2, q2, r2);
            acc3 = neon::vfmaq_f32(acc3, q3, r3);
            i += 16;
        }
        let mut acc = neon::vaddq_f32(
            neon::vaddq_f32(acc0, acc1),
            neon::vaddq_f32(acc2, acc3),
        );
        while i + 4 <= dim {
            let q0 = neon::vld1q_f32(qp.add(i));
            let r0 = neon::vld1q_f32(rp.add(i));
            acc = neon::vfmaq_f32(acc, q0, r0);
            i += 4;
        }
        let mut s = neon::vaddvq_f32(acc);
        while i < dim {
            s += *qp.add(i) * *rp.add(i);
            i += 1;
        }
        s
    }
}

/// Scalar fallback for non-aarch64 targets; rustc auto-vectorizes this at
/// opt-level=3.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn row_dot(q: &[f32], row: &[f32], dim: usize) -> f32 {
    let mut s = 0.0f32;
    let mut i = 0;
    while i < dim {
        s += q[i] * row[i];
        i += 1;
    }
    s
}

/// One raw dot product per row of `matrix` (treated as `len/dim` rows of
/// length `dim`), against `query`. Picks rayon vs single-thread by batch size.
pub fn dot_batch(query: &[f32], matrix: &[f32], dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let d = dim.min(query.len());
    let elems = matrix.len() / dim * dim; // floored element count
    if elems >= PAR_MIN_ELEMENTS {
        matrix
            .par_chunks_exact(dim)
            .map(|row| row_dot(query, row, d))
            .collect()
    } else {
        matrix
            .chunks_exact(dim)
            .map(|row| row_dot(query, row, d))
            .collect()
    }
}
