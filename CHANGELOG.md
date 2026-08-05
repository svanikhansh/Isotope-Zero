# Changelog

All notable changes to Isotope Zero are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are dated
when released.

The project evolves through a deliberate **R&D arc** — each prototype phase
validated or refuted a specific hypothesis before being promoted into the
shipped tree. Entries below mark which capabilities are **accelerators**
(optional native code / model downloads) versus **fallbacks** (always-on,
zero-dependency paths that keep the package runnable with nothing installed).

---

## [1.0.0] - 2026-08-05 — Grand Synthesis

The synthesis release: every validated prototype phase is promoted into one
shipped, locally-runnable package. **Local-first by default, zero network
calls for core operations, $0 inference.**

### Added
- **v1.0.0 production layout.** The `prototypes/synthesis_v1.0` tree is
  consolidated into a proper `src/isotope_zero/` distribution package with a
  maturin/PyO3 build backend and a single `pyproject.toml`. The prototype
  trees remain under `prototypes/` as archived, reproducible research artifacts.
- **`izero-cli`** — a read-only terminal inspection + maintenance tool for
  Isotope Zero memory engines, installable via a POSIX `install.sh` one-liner
  or an `npm` wrapper (`npm i -g izero-cli`). The npm package provisions a
  private Python venv on `postinstall` and proxies argv/stdio/signals through.
- **Multi-tier scoping** (`scope="user_123&agent_456&run_789"`) across the
  store and adapters, enabling row-level isolation without collection-per-tenant.
- **Scale-Adaptive Search.** A runtime router selects the cheapest vector
  backend for the corpus size: Int8 quantized + SIMD (POPCNT) for small
  corpora, float32 BLAS GEMM (`matrix @ query`) for large ones. The router is
  benchmarked, not guessed — see `prototypes/adaptive_dispatch_v1.1`.
- **Hybrid Retrieval (Late Fusion).** An FTS5 inverted index (BM25) is fused
  with the semantic vector branch via Reciprocal Rank Fusion (RRF), plus an
  entity-graph boost using the Mem0 decay formula `0.5/(1+0.001*(N-1)²)`.
  External-content FTS5 triggers keep the index in sync on every write path,
  including the eval harness's direct-to-connection bulk seeder.
- **MRL (Matryoshka Representation Learning) embeddings.** Dimension-slicing
  support so callers can trade recall for storage/latency at a single embedder.
- **Ebbinghaus retention decay + hybrid score fusion.** `calculate_retention`
  and `hybrid_score` promote fresh cards and suppress stale ones automatically;
  `consolidate()` prunes decayed cards, driving the **98.5%** active-context
  reduction headline.
- **Semantic graph & knowledge compaction.** A `card_edges` table records
  semantic (cosine ≥ 0.75) and shared-tag (Jaccard) relations; `auto_link_cards`
  populates it on every write; `detect_clusters` finds connected components for
  consolidation folding.
- **Rust/PyO3 native accelerator (`_native`).** A 22-pattern negation guard
  and vector helpers compiled via maturin (abi3, cp310+). **Fallback:** a
  pure-Python `native.py` Smart Bridge keeps the package fully runnable when
  Rust is absent or the extension fails to load — only the accelerator is lost.
- **Shared-memory embedding daemon.** A single process owns a quantized ONNX
  embedder; callers share it via multiprocessing shared memory, amortizing the
  model load across the process group.
- **Framework adapters.** Drop-in providers for LangChain, LlamaIndex,
  AutoGen, and CrewAI.
- **Benchmark scorecard.** Four claims (parity 100%, temporal recall 100%,
  storage reduction 98.5%, recall-latency overhead ≤ 0.06 ms), four passes —
  reproduced from `python -m isotope_zero.eval.benchmark`.

### Changed
- CI now runs the `src/` package (previously `prototypes/python_v0.1`) across
  a 3-OS × 3-Python matrix, with the Rust accelerator built via maturin and a
  pure-Python Smart Bridge fallback keeping the suite green when Rust is absent.
- Registered pytest marks (`perf`/`stress`/`fuzz`/`integration`/`onnx`/`slow`)
  and silenced the `asyncio_mode` config warning in both `pyproject.toml`s.
- Adapters now resolve the feature-complete `synthesis_v1.0` prototype store by
  default (`IZERO_ENGINE_PATH`), so scope + hybrid search flow through the
  LangChain / LlamaIndex / AutoGen / CrewAI providers.

### Performance (measured, reference host)
- Cold start **< 50 ms ready-to-serve** (35 ms measured: import + schema +
  store open; zero network).
- Vector read p99 @ 10k cards: **0.284 ms** (float32 BLAS, zero-copy).
- Client process RSS: **~28 MB** (27.8 MB measured at init, before the optional
  ONNX embedder loads). The embedding backend is the only RSS lever: +~95 MB
  in-process on first embed, or centralized in the ~360 MB shared daemon so
  clients stay small. Zero-dep stub path (no onnxruntime): ~37 MB idle /
  ~53 MB at 200 cards.

### Dependencies
- **1 hard dep** (numpy) vs. 150+ for the remote-API reference; `sqlite3` is
  stdlib. The `onnx`/`ollama`/`openai`/`mcp`/`fallback`/`adapters` extras are
  opt-in; `pip install isotope-zero` runs with zero optional extras via the
  deterministic fallback embedder.

---

## [0.9.0a1] — Phase 7C: Ebbinghaus Decay & Graph Consolidation

### Added
- `calculate_retention`, `update_stability`, `hybrid_score` — the decay model.
- `consolidate_memories` — cluster-fold dedup with newest-wins survivor,
  evidence union, and summed access pressure.
- `card_edges` schema + `auto_link_cards` / `detect_clusters` /
  `prune_stale_edges`.

---

## [0.8.0] — Binary POPCNT Engine (Phase 7B)

### Added
- A 1-bit binary POPCNT vector-search engine for the small-corpus tier of the
  scale-adaptive router. Quantize to 1-bit, count matching bits as similarity.

---

## [0.7.0] — Shared-Memory Embedding Daemon (Phase 7A)

### Added
- A shared-memory embedding daemon so one model load serves many workers,
  dropping the per-call embed cost toward the ONNX forward pass alone.

---

## [0.6.0] — mmap Storage (Phase 6, refuted as default)

### Notes
- An mmap-backed vector matrix was investigated as the production storage path.
  **Refuted as the default:** the concurrency-safe heap BLAS path is the
  recommended production default; mmap is retained as an EXPERIMENTAL opt-in.
  See the `use_mmap` honesty note in the README.

---

## [0.5.0] — Native Int8 NEON SIMD (Phase 5, conditional win)

### Notes
- Int8 quantized vector search with NEON SIMD acceleration. **Conditional win:**
  faster than float32 BLAS on small corpora and ARM; not a universal default.
  Promoted into the scale-adaptive router's small-corpus tier rather than the
  sole backend.

---

## [0.2.0] — Rust Native Core (Phase 2)

### Added
- The initial Rust/PyO3 `_native` bridge: negation detection patterns and
  vector helpers, with a pure-Python fallback when the extension is unavailable.

---

## [0.1.0] — SQLite Baseline (Phase 1)

### Added
- The original SQLite-backed `MemoryStore` for `MemoryCard` objects: packed
  float32 embeddings via the stdlib `array` module, a single persistent
  connection with a `threading.Lock`, and a plain-Python dot-product vector
  search. The zero-dependency foundation everything else is built on.
