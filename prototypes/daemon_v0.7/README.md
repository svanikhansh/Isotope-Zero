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

## Phase 7A — Shared-Memory Embedding Daemon

> **This directory is `prototypes/daemon_v0.7/`.** Phase 7A moves the heavy
> onnxruntime embedding engine (~360 MB RSS) out of every client process and
> into ONE dedicated daemon process, so client processes import no third-party
> packages and collapse toward <10 MB RSS.

### Architecture

- **The daemon loads the ONNX `EmbeddingEngine` once.** `isotope_zero.daemon.server`
  runs `python -m isotope_zero.daemon.server` (or `--socket <path>` for a custom
  socket) and serves embedding requests over a Unix domain socket
  (`/tmp/izero.sock` by default; the benchmark harness uses its own
  `/tmp/izero_bench.sock`).
- **POSIX shared-memory handoff.** Batch vectors are NOT streamed over the
  socket. The client creates a `multiprocessing.shared_memory` segment, names it
  in the request header, and the daemon writes the float32 results row-major
  into that segment; the client reads them back zero-copy through the page cache.
- **Auto-spawn.** `DaemonClient(socket_path=...)` connects to a running daemon or
  spawns one via `python -m isotope_zero.daemon.server`. Known limitation: the
  auto-spawn command does not forward `--socket`, so a **custom socket path must
  be pre-spawned** (`python -m isotope_zero.daemon.server --socket <path>`)
  before the client connects — the benchmark does exactly this.
- **Duck-typed client.** `isotope_zero.daemon.client.DaemonClient` exposes
  `.embed_text/.embed_batch/.is_real/.dim`, imports ONLY stdlib, and drops in
  anywhere an `EmbeddingEngine` is accepted (`MemoryStore`, `QueryRouter`,
  `Consolidator`).

### How to run

```bash
cd prototypes/daemon_v0.7

# Server (loads ONNX once; add --socket for a custom path):
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m isotope_zero.daemon.server

# Client in any process:
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -c "
from isotope_zero.daemon.client import DaemonClient
from isotope_zero.core.store import MemoryStore
dc = DaemonClient(socket_path='/tmp/izero_bench.sock')
store = MemoryStore(':memory:', embedder=dc)   # or MemoryStore(..., use_daemon=True)
"

# Full Phase 7A benchmark (5 claims, honest PASS/FAIL):
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m isotope_zero.eval.adversarial --daemon
```

### Measured results (10,000 cards, captured at this prototype)

Claims recorded verbatim next to measured reality — a claim that does not hold
renders FAIL:

| claim (verbatim) | measured | verdict |
|---|---|---|
| Client RSS < 10 MB (vs in-process ~450 MB) | **275 MB** peak (client subprocess, 10k cards seeded via daemon) | **FAIL** |
| — in-process client comparator (same 10k cards) | 409 MB peak | — |
| daemon-backed 5-worker total ≪ 5×450 MB in-process | daemon **402 MB** (5 clients + daemon 174 MB) vs in-process 776 MB | **PASS** |
| IPC dispatch latency < 0.05 ms p99 | daemon embed_text p99 **1.117 ms**; in-process p99 1.037 ms | **FAIL** |
| — raw IPC dispatch (daemon ping RTT, no inference) | p99 **0.016 ms** | — |
| — daemon embed_batch(32) p99 | 18.238 ms | — |
| Recall parity 100% (bit-identical vectors → identical top-k) | max \|Δvec\| = **0.00e+00**; top-k match **100.0%** (ids+order, 50 queries) | **PASS** |
| No regression: existing in-process adversarial path still runs | reduced-scale run 5/6 sub-claims | **PASS** |

**Daemon total: 3/5 claims hold.**

Honest reading:

- **Claim 1 FAILs structurally.** The client process floor is python + numpy
  (~35 MB) + sqlite3 + the daemon client + the 10k-card store (matrix + card
  lists) ≈ 274 MB. The <10 MB target ignores the cost of holding a 10k-card
  store and the numpy/SQLite runtime a real client needs; only the ~360 MB
  onnxruntime share is removed by the daemon.
- **Claim 2 PASSes decisively.** 5 daemon-backed workers + the daemon (398 MB)
  is ~2× less than 5 in-process clients (776 MB) and far under the 5×450 MB
  claim — the daemon amortizes the ONNX footprint across all clients.
- **Claim 3 FAILs honestly.** The 0.05 ms target applies to pure socket
  dispatch, which MEASURES at p99 0.017 ms (raw `ping` RTT). But the claim's own
  unit — a warm `embed_text` round-trip — includes the daemon's ONNX inference
  (~1.1 ms), which is on the hot path either way and matches in-process p99
  (1.045 ms). The daemon adds no measurable latency over in-process embedding.
- **Claim 4 PASSes bit-exactly.** Vectors round-trip through shared memory as
  float32 identical to in-process output; top-k ids AND order match 100%.
- **Claim 5 PASSes** — the in-process adversarial path is unchanged.

The 5-worker RSS test spawns daemon-backed client workers on the harness's own
socket; the parallel daemon on `/tmp/izero.sock` is never touched. Run the
harness with `--json` for machine-readable values.

These figures were independently re-verified (a second full `--daemon` run:
275 MB / 402 MB / 1.117 ms — same 3/5 verdict). One harness bug surfaced and
was fixed during that verification: the benchmark pre-spawns the daemon
**without** `--idle-timeout`, so the daemon's default 300 s idle-exit fired
mid-run — during claim 4's parity step the benchmark embeds 10k cards
in-process (minutes of no daemon traffic) while the daemon sits idle, then the
parity query loop hits a dead socket (`BrokenPipeError`). `run_daemon_benchmark`
now pre-spawns with `--idle-timeout 0` so the daemon lives for the whole run.
