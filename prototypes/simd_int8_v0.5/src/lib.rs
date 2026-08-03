//! Native isotope_core module: PyO3 bindings for the int8 SIMD kernel.
//!
//! Exposes `simd_int8_batch_dot` to Python as `isotope_zero._native.simd_int8_batch_dot`.
use pyo3::prelude::*;
use pyo3::types::PyModule;
mod simd_int8;

/// Register the native extension. Called by maturin at import time.
/// The function MUST be named `_native` (not `isotope_core`) so pyo3 emits the
/// `PyInit__native` symbol that matches `module-name = "isotope_zero._native"`
/// in pyproject.toml. A mismatch here produces `ImportError: dynamic module
/// does not define module export function (PyInit__native)` at import time.
#[pymodule]
fn _native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simd_int8::simd_int8_batch_dot, m)?)?;
    m.add_function(wrap_pyfunction!(simd_int8::simd_kernel_name, m)?)?;
    m.add("__simdkernel_version__", "0.5.0")?;
    Ok(())
}
