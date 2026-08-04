# Two-Tier mmap Storage v0.6 Prototype — File-Backed Vectors (Method 3)

This folder is the **v0.6.0a1 prototype** of Isotope Zero implementing
**Method 3: Two-Tier Memory-Mapped (mmap) Storage**. Scaffolded from
`prototypes/rust_v0.2/` (the Phase 6 Smart Bridge baseline: float32 BLAS
vector search + Rust negation). Method 3 is a *storage backend* change
layered on top — it does not touch the search math or the negation kernel.

## Thesis

Process RSS sits at ~394 MB at 10k cards. The thesis attributes this to
"all float32 vector matrices permanently allocated in heap memory" and
proposes replacing the heap matrix with file-backed `np.memmap` stored in
`embeddings.bin`, with a Hot RAM LRU cache (N=200) for active cards. The OS
virtual memory manager would page cold vectors in/out on demand, keeping
total matrix memory strictly bounded (< 30 MB heap).

### The honest caveat (measured up front)

**The thesis premise is factually wrong about where the RSS comes from.**
At 10k cards the float32 matrix is only **14.65 MB** — the ~394 MB RSS is
dominated by the **ONNX runtime model weights + thread pool (~360 MB)**,
not the matrix. So mmap (moving 15 MB to disk) **cannot** reduce RSS from
394→30 MB; at best it saves ~15 MB, and a full vector scan pages the whole
matrix back in (neutralizing the saving on the hot path). The honest
questions this prototype actually answers:

1. Does mmap change the *resident* matrix footprint vs heap (mmap pages are
   not counted as RSS until touched)?
2. Does cold page-fault latency penalize the vector scan?
3. Does the Hot LRU cache (N=200) keep the working set resident and fast?

**No fabricated "RSS dropped to 30 MB" result will be reported.** The ONNX
dominance is a structural fact established across Methods 1–4. What this
prototype measures is the *matrix-tier* memory delta and the latency cost of
demand-paging, reported truthfully beside the claim.

## What changed from v0.2

- `isotope_zero/core/mmap_store.py` (new): `MmapVectorStore` — writes 384-dim
  float32 embeddings contiguously to `embeddings.bin`, opens it as
  `np.memmap(mode='r+')`, maintains a Hot LRU cache (N=200) of recently
  accessed card vectors. Search runs `matrix @ q` across the memmap view
  (cold pages read on demand) with the hot cache sliced first.
- `isotope_zero/core/store.py`: `MemoryStore._ensure_vec_cache` and
  `vector_search` now route through the mmap backend when configured, with
  the heap BLAS path as fallback for hetero-dim / empty / disabled cases.
- `isotope_zero/eval/adversarial.py`: benchmarks Heap Storage vs mmap
  Storage across resident matrix memory, hot/cold p99 latency, and recall
  accuracy.

## The build pipeline (must use absolute paths)

```bash
export PATH="$HOME/.cargo/bin:$PATH"   # rustc/cargo NOT on PATH by default
cd prototypes/mmap_v0.6
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/maturin develop --release
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m pytest tests/ -q   # 129 passed / 5 skipped
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m isotope_zero.eval.adversarial
```

Always use the **absolute** venv path after `cd`-ing in — a relative
`.venv/bin/python` resolves against the new cwd and fails.

## Measured results (captured at freeze)

Adversarial suite, 10,000 cards, 25-worker concurrency. Claims recorded
verbatim beside measured reality; FAIL for any that don't hold.

| metric | Heap path (baseline) | mmap path (Method 3) | claim | verdict |
|---|---|---|---|---|
| resident matrix MB @10k | 15.36 | 15.36 (file ceiling) | < 30 MB | **PASS** (both) |
| total process RSS MB @10k | 452 | 452 | < 30 MB (thesis headline) | **FAIL** — ONNX ~360 MB dominates, not the matrix |
| hot p99 (LRU warm) | 0.29 ms | 0.31 ms | (no regress) | INFO — mmap ~7% slower, not faster |
| cold p99 (memmap close+reopen) | n/a | 0.29 ms | (no claim) | INFO — 1.04–1.07× hot (OS file cache nullifies cold penalty) |
| recall matches baseline | n/a | yes (100%) | 100% | **PASS** |
| Hot LRU resident MB | n/a | 0.008 (cap 0.307) | — | trivially bounded |

**Total: 7/9 claims hold.** The two matrix-tier claims split exactly as the
caveat predicted: the matrix-tier `< 30 MB` claim PASSES (the 14.65 MB matrix
was never the bottleneck — for *either* path), and the thesis headline
`total RSS < 30 MB` FAILS honestly at 452 MB (ONNX runtime ~360 MB dominates;
mmap cannot subtract the matrix from an RSS it doesn't own). The remaining
FAILs (the pre-existing structural ones) are unchanged.

### Honest verdict — correct backend, refuted thesis

**(1) Correctness — PASS.** The mmap backend produces **bit-identical** search
results to the heap path (same top-k ids, same order, max score diff = 0.0,
verified independently). Superseded cards excluded, `update`/`delete`/`touch`
invalidation correct, hetero-dim falls back gracefully. 129/5 green.

**(2) Matrix-tier resident memory — PASS, but meaningless.** The matrix is
14.65 MB either way (heap `np.stack` bytes == mmap file bytes == n*dim*4).
The Hot LRU resident is 0.008 MB (cap 0.307 MB), trivially under 30 MB — but
"bounding" a 15 MB matrix to "under 30 MB" is a tautology; the heap path
already satisfies it. mmap did not reduce the resident matrix footprint in
any way that matters.

**(3) Total RSS — FAIL, structurally.** mmap did not reduce RSS; it *raised*
it. Independent measurement at 10k: heap path ~416–452 MB peak, mmap path
~456–472 MB — **mmap is ~40 MB HIGHER, not lower.** The ~360 MB ONNX runtime
(model weights + arena + threads) dominates, and the 14.65 MB matrix is a
rounding error against it. mmap added the memmap mapping overhead on top of
the unchanged ONNX footprint instead of subtracting the matrix from RSS. The
thesis premise — "RSS ~394 MB because all float32 vector matrices are
permanently allocated in heap" — is factually wrong, confirmed by
measurement.

**(4) Cold-scan latency — nullified by the OS.** The predicted cold-page-fault
penalty did not materialize: cold p99 0.29 ms vs hot 0.29 ms (1.04–1.07× across
runs). This is *not* because page faults are free — it's because on macOS,
dropping the process memmap mapping and reopening it does **not** evict the
file's pages from the OS unified buffer cache. The 14.65 MB matrix is tiny
relative to available RAM, so the kernel keeps it fully cached regardless of
the process mapping state. The "cold" path just re-maps pages already resident
in the OS file cache. A genuine cold penalty would require `posix_fadvise
(POSIX_FADV_DONTNEED)` (unavailable on macOS) or evicting the OS file cache
itself, which is outside userland's control. The LRU's speedup over a cold
scan is marginal (1.04×) precisely because the OS cache already does the job
the LRU was meant to do.

**(5) Latency vs heap — slightly slower, not faster.** mmap hot p50 0.082 ms vs
heap BLAS 0.073 ms at 5k (~13% slower), 0.31 vs 0.29 ms p99 at 10k. The memmap
view adds a thin indirection vs a contiguous heap array; the matrix math
(`matrix @ q`) is identical, so there is no latency win, only a minor cost.

### Bottom line

The mmap backend is **production-safe** (correct, no regressions, clean
invalidation, graceful fallback) and **does bound the resident LRU heap to
<0.3 MB**. But the thesis claim that it reduces total process RSS to <30 MB is
**structurally impossible given ONNX dominance**, and the honest measurements
confirm it: RSS *rose* ~40 MB, and the cold-scan penalty is nullified by the
OS file cache at the 10k-card scale. The LRU and demand-paging would only
start mattering at 1M+ cards (~1.5 GB matrix), where the matrix overflows the
OS file cache — a scale this prototype does not reach, and beyond the product's
local-agent-memory use case.

The structural RSS ceiling (~360 MB ONNX) is the same wall Methods 1–4 hit; no
matrix-tier optimization can breach it. The only lever for total RSS is the
embedding backend itself (a smaller ONNX model, or a quantized/embedding
engine) — not the vector storage tier.

## Siblings

- `prototypes/rust_v0.2/` — the Phase 6 Smart Bridge baseline this builds on.
- `prototypes/quantization_v0.4/` + `prototypes/simd_int8_v0.5/` — Methods 2/4,
  the int8 quantization prototypes (4× RAM reduction at the matrix tier).
