# Phase 6 — Rust v0.2 prototype (Smart Bridge)

> **Frozen reference implementation.** This directory holds the Phase 6
> `isotope-zero` **v0.2.0a1** baseline (commit `2e07d1c`, archived via
> `git mv` so full commit history is preserved). The repository root is
> reserved for the next architectural iteration; this archive can be inspected
> or run independently.

## Architecture — the Smart Bridge

Phase 6 places each workload on its empirically-fastest path, chosen from
**measured** benchmarks rather than assumption:

| workload | path | why |
|---|---|---|
| **Vector dot-product** (cosine similarity) | **NumPy / BLAS** — `matrix @ query` | Zero-copy operation on the numpy buffers, GIL released around the platform BLAS C kernel (Accelerate on macOS). Measured ~9–115× faster than the Rust extension's copy-to-release-GIL path. |
| **Negation / polarity detection** | **Rust** (`isotope_zero._native.are_negations`) | A bit-for-bit port of the v0.1 Python heuristic, compiled and GIL-released via PyO3, with a clean pure-Python fallback when the `.so` is absent or ABI-mismatched. No BLAS equivalent exists, so compiled Rust wins. |

**Why the Rust vector path was rejected:** `PyReadonlyArray` borrows Python
memory and is not `Send`, so releasing the GIL with `py.allow_threads` forces
the Rust code to **clone the entire matrix** into owned `Vec<f32>` (≈15 MB at
10k×384) before computing. NumPy's `matrix @ q` does the same dot-product
zero-copy and releases the GIL around the C kernel — so hand-rolled NEON SIMD
cannot beat zero-copy BLAS at these sizes. The Rust extension still exports
`batch_cosine_similarity_matrix` for parity probes and future quantized-SIMD
work, but it is **off the hot path**.

## Layout

```
prototypes/rust_v0.2/
├── Cargo.toml        # maturin + PyO3 0.21 (abi3-py310) + rayon + numpy 0.21
├── Cargo.lock        # pinned, for reproducible native builds
├── pyproject.toml    # maturin build backend, isotope-zero v0.2.0a1
├── src/
│   ├── lib.rs        # PyO3 module `_native`, GIL-released dispatch
│   ├── vector.rs     # NEON 4-accumulator FMA dot batch (+ rayon crossover)
│   └── negation.rs   # zero-copy &str tokenizer + integer Jaccard polarity
├── isotope_zero/     # the Python package (core/native.py = the bridge seam)
└── tests/            # 129 passed / 5 skipped (5 gated behind IZERO_STRESS=1)
```

## Run it

A Rust toolchain and `maturin` are required. The venv's editable install must
be (re)pointed at this directory after the move.

```bash
cd prototypes/rust_v0.2

# Build & install the native extension into the project venv (release mode).
export PATH="$HOME/.cargo/bin:$PATH"   # rustup toolchain lives here
.venv/bin/maturin develop --release    # produces isotope_zero/_native.abi3.so

# Re-point the editable install + console scripts at this directory.
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m pip install -e .

# Full test suite (129 passed / 5 skipped expected).
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m pytest tests/ -q

# Adversarial stress + competitor benchmark (10k cards, 25-proc concurrency).
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m isotope_zero.eval.adversarial
```

Use the **absolute** venv path (`…/isotope_zero/.venv/bin/python`) after `cd`-ing
in — a relative `.venv/bin/python` resolves against the new cwd and fails.

## Verified results (captured at freeze, commit `2e07d1c`)

Adversarial suite, 10,000 cards, 25-worker concurrency warfare:

| metric | target | measured | verdict |
|---|---|---|---|
| Vector search p99 @10k | < 0.30 ms | 0.30 ms (p50/p95 = 0.20/0.24) | **PASS** |
| Negation correctness | 0 incorrect merges | 0 / 100 distinct survivors | **PASS** (Rust) |
| SQL lookup p99 @10k | < 0.80 ms | 0.66 ms | **PASS** |
| Needle recall | 100% | 100% | **PASS** |
| Concurrency (25 proc, 25k ops) | 0 err / 0 corrupt | 0 / 0 | **PASS** |
| RSS @10k | < 200 MB (v0.1 claim) | 394 MB | **FAIL** (structural ONNX floor) |

The RSS floor is structural — the ONNX MiniLM embedder (~135 MB) plus genuine
10k×384 float32 embeddings — and the Smart Bridge does not touch it. It is
reported honestly, not relaxed. It would only move by quantizing or dropping
the ONNX embedder, which is out of scope for this prototype.

Negation parity is bit-for-bit: Rust `_native.are_negations` matches the
pure-Python `consolidation._are_negations` on 85/85 adversarial pairs.

See the repository root `README.md` for the project thesis and the sibling
`prototypes/python_v0.1/` archive for the pure-Python v0.1.0 baseline this
supersedes.
