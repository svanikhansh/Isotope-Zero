//! Phase 6 native core for Isotope Zero.
//!
//! Module name `_native` must match the last component of the maturin
//! `module-name = "isotope_zero._native"` so Python can import it as
//! `isotope_zero._native`.
//!
//! Exposed functions (the Python bridge's single source of truth):
//! 1. `batch_cosine_similarity(query, matrix_flat, dim)` — Vec-based fallback.
//! 2. `batch_cosine_similarity_matrix(query, matrix)` — zero-copy numpy fast
//!    path; never round-trips through Python float objects.
//! 3. `are_negations(a, b)` — bit-for-bit port of the consolidation polarity
//!    heuristic.

mod negation;
mod simd_popcnt;
mod vector;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Vec-based fallback: `matrix_flat` is `n_rows * dim` floats laid out
/// row-major; returns one raw dot product per row. Rows and query are
/// L2-normalized upstream, so `dot == cosine`. Scores are NOT clamped.
#[pyfunction]
fn batch_cosine_similarity(
    py: Python<'_>,
    query: Vec<f32>,
    matrix_flat: Vec<f32>,
    dim: usize,
) -> Vec<f32> {
    // The Vecs are owned (Send); release the GIL for the whole batch.
    py.allow_threads(move || vector::dot_batch(&query, &matrix_flat, dim))
}

/// Zero-copy fast path: reads the `(n, dim)` float32 numpy matrix directly
/// from numpy's buffer (no per-element Python float conversion), copies the
/// buffers into owned Rust slices, then releases the GIL around the SIMD
/// batch.
#[pyfunction]
fn batch_cosine_similarity_matrix(
    py: Python<'_>,
    query: PyReadonlyArray1<f32>,
    matrix: PyReadonlyArray2<f32>,
) -> PyResult<Vec<f32>> {
    let q_view = query.as_slice()?;
    let m_view = matrix.as_slice()?;
    let cols = matrix.shape()[1];
    let dim = query.shape()[0];
    if cols != dim {
        return Err(PyValueError::new_err(format!(
            "matrix column count ({}) must equal query dimension ({})",
            cols, dim
        )));
    }
    // Copy numpy buffers into owned Vecs so the GIL can be released while the
    // hot loop runs (PyReadonlyArray borrows Python memory and is not Send).
    let qv: Vec<f32> = q_view.to_vec();
    let mv: Vec<f32> = m_view.to_vec();
    Ok(py.allow_threads(move || vector::dot_batch(&qv, &mv, dim)))
}

/// Bit-for-bit port of `consolidation._are_negations`.
#[pyfunction]
fn are_negations(py: Python<'_>, a: String, b: String) -> bool {
    py.allow_threads(move || negation::are_negations(&a, &b))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(batch_cosine_similarity_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(are_negations, m)?)?;
    m.add_function(wrap_pyfunction!(
        simd_popcnt::popcnt_hamming_search,
        m
    )?)?;
    Ok(())
}
