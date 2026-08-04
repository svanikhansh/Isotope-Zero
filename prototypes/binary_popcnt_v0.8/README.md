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

---

# Phase 7B — 1-Bit Binary POPCNT Engine

## Architecture

Phase 7B adds a **2-stage binary->float32 vector search pipeline** as a
capstone experiment in extreme compression for large-scale memory retrieval:

```
Query embed (float32 384-dim)
        |
        v
Stage 1: 1-bit sign quantization (>=0 → 1, <0 → 0)
         → packbits → uint8 (48 bytes per card)
         → XOR + POPCNT (Hamming distance) over ALL cards
         → select top (k × oversample_factor) candidates
        |
        v
Stage 2: float32 cosine re-rank (BLAS matmul)
         → only on surviving candidates (typically ~100)
         → top-k by exact cosine score
```

Key modules:
- `isotope_zero/core/binary_quant.py` — quantize_1bit, binary_hamming_distance (numpy fallback)
- `src/simd_popcnt.rs` — native NEON POPCNT kernel (Rust PyO3 `_native.popcnt_hamming_search`)
- `isotope_zero/core/store.py` — `MemoryStore.vector_search_binary_rerank(query, k, oversample_factor)`

## Measured Results (100,000 cards)

Each claim from the Phase 7B brief is measured honestly at 100k cards:

| claim (verbatim) | measured | verdict |
|---|---|---|
| Binary matrix footprint < 5.0 MB (vs float32 ~153.6 MB) | 4.80 MB (96.9% reduction) | **PASS** |
| POPCNT Hamming p99 < 0.05 ms | 2.462 ms | **FAIL** |
| 2-stage pipeline p99 < 0.10 ms | 2.838 ms | **FAIL** |
| Recall@10 > 95% vs float32 BLAS | 0.0% | **FAIL** |

**1/4 claims hold.**

### Comparison

| metric | 1-Bit POPCNT | Float32 BLAS |
|---|---|---|
| Matrix footprint | 4.80 MB | 153.60 MB |
| Vector search p99 | 2.838 ms | 2.832 ms |
| Stage 1 (POPCNT) p99 | 2.462 ms | -- |
| Stage 2 (re-rank) p99 | 0.141 ms | -- |

### Analysis

- **Footprint is the clear win**: 32x compression (153.6 MB → 4.80 MB) exactly
  as expected from float32→uint8 packing. At 100k cards the binary matrix sits
  just under the 5.0 MB claim boundary.
- **Latency is no better than BLAS**: The 2-stage pipeline (2.838 ms) is
  essentially identical to direct float32 BLAS (2.832 ms). Stage 1 (POPCNT)
  dominates at 2.462 ms — it must scan every row's 48-byte packed vector.
  Stage 2 re-rank on just 1,000 candidates is fast (0.141 ms) but cannot
  compensate for the O(n) Stage 1 scan.
- **Recall is zero**: Sign-bit quantization on 384-dim ONNX MiniLM embeddings
  discards too much information. The Hamming distance ordering is orthogonal
  to the cosine similarity ordering — none of the float32 top-10 appear in
  the POPCNT candidate pool of 1,000. Binary quantization works for embeddings
  specifically trained for binarization, not for off-the-shelf float32 models.
- **Conclusion**: Binary POPCNT trades 32x footprint for zero-recall retrieval.
  As a pre-filter, it requires embedding-aware quantization training or
  product-quantization codebooks — sign-bit thresholding of pre-trained float32
  embeddings is insufficient. The matrix drop is real and significant (4.80 MB),
  but without recall the pipeline cannot replace float32 BLAS search.

## Run it

```bash
# Binary benchmark at N cards (appended to the adversarial report)
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python \
    -m isotope_zero.eval.adversarial --binary-popcnt 100000

# Benchmark only (skip sections A-D by using minimal scale + cycles)
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python \
    -m isotope_zero.eval.adversarial --scale 0 --cycles 0 --binary-popcnt 100000

# Smaller scale for faster iteration
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python \
    -m isotope_zero.eval.adversarial --binary-popcnt 10000
```
