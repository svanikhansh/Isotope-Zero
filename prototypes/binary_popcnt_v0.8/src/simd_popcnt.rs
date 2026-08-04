/// POPCNT-accelerated Hamming distance search over packed binary vectors.
///
/// Each vector: 384 bits packed into 48 uint8 (matrix shape n×48).
/// Distance = XOR each byte → popcount → sum.
/// Stage 1 over-fetches top_k × oversample_factor for float32 re-rank.
///
/// ARM64: NEON vcntq_u8 (stable Rust) + horizontal sum.
/// x86:   u64::count_ones() on XOR'd chunks (stable, fast enough).
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// 384 bits / 8 = 48 bytes per packed vector.
const BYTES_PER_VECTOR: usize = 48;

/// Compute Hamming distance between two 48-byte packed binary vectors.
///
/// Dispatches to NEON intrinsics on ARM64 (aarch64), scalar `count_ones()` on
/// every other architecture.
#[inline]
fn hamming_48(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: caller guarantees a.len() >= 48 and b.len() >= 48.
        // In practice the chunks_exact(48) iterator never produces shorter
        // slices, so every call site is safe.
        unsafe { hamming_48_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        hamming_48_scalar(a, b)
    }
}

/// ARM64 NEON: 3 × 128-bit registers, XOR + vcnt + horizontal sum.
///
/// # Safety
///
/// `a` and `b` must each be at least 48 bytes long.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hamming_48_neon(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::aarch64::*;

    // 48 bytes = 3 × 16-byte NEON registers (no alignment requirement for
    // vld1 — unaligned loads are fine on Apple Silicon and all modern ARM).
    let va0 = vld1q_u8(a.as_ptr());
    let vb0 = vld1q_u8(b.as_ptr());
    let va1 = vld1q_u8(a.as_ptr().add(16));
    let vb1 = vld1q_u8(b.as_ptr().add(16));
    let va2 = vld1q_u8(a.as_ptr().add(32));
    let vb2 = vld1q_u8(b.as_ptr().add(32));

    // XOR → POPCNT per lane (8-bit lane → 0..8 per lane).
    let pop0 = vcntq_u8(veorq_u8(va0, vb0));
    let pop1 = vcntq_u8(veorq_u8(va1, vb1));
    let pop2 = vcntq_u8(veorq_u8(va2, vb2));

    // Horizontal sum: each register summed to a u16.
    let sum0 = vaddlvq_u8(pop0) as u32;
    let sum1 = vaddlvq_u8(pop1) as u32;
    let sum2 = vaddlvq_u8(pop2) as u32;

    sum0 + sum1 + sum2
}

/// Scalar fallback: XOR 8-byte chunks → `u64::count_ones()`.
///
/// 48 bytes = 6 × 8-byte chunks. On x86_64 `count_ones()` compiles to the
/// `POPCNT` instruction (when target-cpu includes it), which is fast enough
/// that hand-written SIMD is not worth the maintenance burden.
#[inline]
#[allow(dead_code)]
fn hamming_48_scalar(a: &[u8], b: &[u8]) -> u32 {
    let mut total: u32 = 0;
    // Process six 8-byte chunks — 48 bytes total.
    for chunk in 0..6 {
        let off = chunk * 8;
        // SAFETY: chunks_exact(48) guarantees we always have at least 48 bytes;
        // each 8-byte window is always in-bounds.
        let va = u64::from_ne_bytes(
            a[off..off + 8].try_into().unwrap(),
        );
        let vb = u64::from_ne_bytes(
            b[off..off + 8].try_into().unwrap(),
        );
        total += (va ^ vb).count_ones();
    }
    total
}

/// POPCNT-accelerated Hamming distance search over packed binary vectors.
///
/// Args:
///     matrix: 2-D uint8 numpy array of shape ``(n, 48)`` — packed sign-quantized
///         embeddings (384 bits per vector).
///     q: 1-D uint8 numpy array of shape ``(48,)`` — packed query vector.
///     top_k: Number of final results the caller wants (used to size the
///         Stage-1 oversampled pool).
///     oversample: How many ``× top_k`` candidates to return.
///         E.g., ``top_k=10, oversample=10`` returns 100 candidates.
///
/// Returns:
///     ``(indices: list[int], distances: list[int])`` — both sorted ascending
///     by Hamming distance.  ``indices`` are row indices into ``matrix``.
///
/// The GIL is released during the entire computation so Python threads can
/// make progress.
#[pyfunction]
pub fn popcnt_hamming_search(
    py: Python<'_>,
    matrix: PyReadonlyArray2<u8>, // (n, 48)
    q: PyReadonlyArray1<u8>,      // (48,)
    top_k: usize,
    oversample: usize,
) -> PyResult<(Vec<i64>, Vec<u32>)> {
    let q_slice = q.as_slice()?;
    let m_slice = matrix.as_slice()?;
    let shape = matrix.shape();
    let n_rows = shape[0];
    let n_cols = shape[1];

    // --- Validation -------------------------------------------------
    if q_slice.len() != BYTES_PER_VECTOR {
        return Err(PyValueError::new_err(format!(
            "query must be exactly {} bytes (384 bits), got {}",
            BYTES_PER_VECTOR,
            q_slice.len()
        )));
    }
    if n_cols != BYTES_PER_VECTOR {
        return Err(PyValueError::new_err(format!(
            "matrix columns must be exactly {} bytes (384 bits), got {}",
            BYTES_PER_VECTOR, n_cols
        )));
    }

    let target = top_k.saturating_mul(oversample);
    if target == 0 || n_rows == 0 {
        return Ok((vec![], vec![]));
    }
    let target_count = target.min(n_rows);

    // Copy numpy buffers so the GIL can be released (PyReadonlyArray borrows
    // Python memory and is not Send).
    let m_copy: Vec<u8> = m_slice.to_vec();
    let q_copy: Vec<u8> = q_slice.to_vec();

    let (indices, distances) = py.allow_threads(move || {
        // Max-heap keyed by Reverse(distance) → keeps the target_count
        // *smallest* distances.  O(n log k) with k == target_count.
        let mut heap: BinaryHeap<Reverse<(u32, usize)>> =
            BinaryHeap::with_capacity(target_count + 1);

        for (idx, row) in m_copy.chunks_exact(BYTES_PER_VECTOR).enumerate() {
            let d = hamming_48(row, &q_copy);
            heap.push(Reverse((d, idx)));
            if heap.len() > target_count {
                heap.pop(); // evict the furthest candidate
            }
        }

        // Drain into a Vec and sort ascending.
        let mut top: Vec<(u32, usize)> = heap
            .into_iter()
            .map(|Reverse(pair)| pair)
            .collect();
        top.sort_unstable_by_key(|(d, _)| *d);

        let indices: Vec<i64> = top.iter().map(|(_, i)| *i as i64).collect();
        let distances: Vec<u32> = top.iter().map(|(d, _)| *d).collect();
        (indices, distances)
    });

    Ok((indices, distances))
}
