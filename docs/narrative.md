# Isotope Zero — Narrative

A developer-facing account of what isotope_zero is, how it works, why it looks
the way it does, and where it is going. Honest technical prose, not marketing.
Measured numbers are measured; audit-derived numbers are labeled as such.

---

## 1. What it is

Isotope Zero is a **local-first, offline, zero-API-key cognitive memory layer
for AI agents**. It is a Python library that gives an agent long-term memory:
the agent stores facts, and later retrieves the ones that are still relevant —
with recent, important, frequently-recalled facts promoted and stale,
duplicated ones suppressed. It is deliberately *not* a hosted vector database
and *not* an LLM-backed memory service.

Three properties separate it from the mainstream agent-memory tools:

- **No network.** Every core operation (write, read, consolidate) is a local
  function call against a SQLite file. There is no embedding API call, no
  vector-store round-trip, no telemetry ping. `pip install isotope-zero` + one
  file is the entire footprint.
- **No API bill.** Embeddings come from a local ONNX model
  (`all-MiniLM-L6-v2`, dim 384) via `onnxruntime`, never from
  `text-embedding-3-small`. Inference cost is `$0`.
- **Your data stays on your machine.** The memory DB is a file you own. There
  is no sync, no cloud copy, no "platform" to trust. If the file is deleted,
  the memory is gone — which is exactly the point.

The cost of that choice is that the semantic model is a fixed local one and the
knowledge graph is lightweight. Section 5 compares that honestly against mem0.

Measured envelope: vector read **p99 ~0.3 ms @ 10k cards** (0.284 ms on the
reference host, 0.31 ms in `scripts/verify_perf.py`), cold start **< 50 ms**
(35 ms measured: import + schema + store open), client process **~28 MB RSS**
before the optional embedder loads. Core has **one hard dependency** (numpy);
`sqlite3` is stdlib.

## 2. How it works

Four layers do the actual work, in `src/isotope_zero/core/` and
`src/isotope_zero/embeddings/`.

### SQLite WAL for ACID and persistence

`MemoryStore` (`core/store.py`) is a single SQLite database holding cards as
rows — fact, evidence, tags, timestamps, stability/importance/archived flags,
and a packed float32 embedding BLOB — plus a `card_edges` graph table and an
FTS5 index for lexical search. For file-backed DBs it opens in **WAL mode**
(`PRAGMA journal_mode=WAL`, `store.py:399`), which lets a background
consolidation worker sweep without blocking the read/write hot path, and sets
`synchronous=NORMAL` (safe under WAL) plus `busy_timeout=5000` so a contended
`BEGIN IMMEDIATE` waits instead of raising instantly (`store.py:400`, `436`).
Writes are serialized by one held connection and one `threading.Lock`; readers
go through the same connection, so there is a single point of truth and no
connection pool.

### NumPy/BLAS for sub-ms vector search

The hot read path is a single float32 `matrix @ query` over the lazily-cached
`(n, dim)` matrix of normalized embeddings (`_ensure_vec_cache`,
`store.py:1333`; dispatch in `vector_search`, `store.py:1447`). Because rows
are L2-normalized by the embedder, dot product *is* cosine similarity, and the
matmul runs on the platform BLAS (Accelerate on macOS, OpenBLAS elsewhere)
with **zero copy** of the numpy buffers and the GIL released around the C
kernel (`core/native.py:42-65`). That is the entire trick: no hand-written
kernel can beat zero-copy BLAS at these sizes. Top-k then re-ranks by the
hybrid score (cosine fused with recency).

### ONNX for local embeddings

`HybridEmbeddingEngine` (`embeddings/engine.py:36`) is daemon-first: a Unix
domain socket (`/tmp/izero.sock`) serves one process that owns the ~360 MB
`onnxruntime` load, so a fleet of client workers stays small. If the daemon is
unreachable it silently falls back to an in-process ONNX engine, and if no
model is available at all it degrades to a deterministic feature-hash stub —
the package is runnable with nothing installed.

### Ebbinghaus decay + consolidation for cognitive maintenance

Two components give the memory a temporal life cycle:

- **Decay** (`core/decay.py`): `calculate_retention` (`decay.py:24`) scores a
  card by Ebbinghaus retention `R(t) = exp(-Δt/(S·H))` with a 24 h base
  half-life; `touch()` (`store.py:968`) bumps access count and grows stability
  `S` non-linearly on every recall. `recall` fuses cosine and retention at
  `alpha = 0.70` (`decay.py:105`), so a 3-second-old fact outranks a
  semantically identical 30-day-old one.
- **Consolidation** (`core/consolidation.py`): the `Consolidator`
  (`consolidation.py:222`) detects near-duplicate clusters and folds them into
  survivors with an audit trail (`superseded_by`), prunes decayed cards, and is
  negation-aware (`are_negations`, `core/native.py:138`) so "X is my favorite"
  and "I no longer like X" are never merged even when their embeddings are
  nearly identical. One sweep measured a **98.5%** active-context reduction
  (5518 → 83 tokens, 199 merged).

The whole loop is `remember` → `recall` → `touch` → `consolidate` exposed by
the `IsotopeZero` facade in `client.py`.

## 3. The v1.3.0 pivot: why the Rust native extension is gone

This is the honest story, and it is the credibility hook of the project.

Isotope Zero shipped its first two releases (v1.0.0, v1.1.x) with a
Rust/PyO3 native extension (`isotope_zero._native`) built by maturin across a
cibuildwheel ×3–4 platform matrix. **In v1.3.0 that crate was removed and the
package became a pure-Python wheel.** The reasons are empirical, not aesthetic:

1. **The SIMD fast path never shipped to pip users.** The int8 NEON kernel
   (`simd_int8_batch_dot`) that motivated the native build was a prototype-only
   artifact (`prototypes/simd_int8_v0.5`) that was never compiled into the
   wheel. For every pip-installed user the Rust crate only ever contained
   float32 parity probes and one negation function.
2. **BLAS was faster.** The Rust extension's `batch_cosine_similarity_matrix`
   had to clone the 10k×384 matrix into an owned `Vec<f32>` to release the GIL
   (`PyReadonlyArray` borrows Python memory and is not `Send`). NumPy's
   `matrix @ q` releases the GIL around zero-copy BLAS and was measured
   **~9–115× faster**. The hot vector path always won on NumPy/BLAS.
3. **The negation port was redundant.** The Rust `are_negations` was
   bit-for-bit identical to the pure-Python heuristic, so removing it changed
   nothing for any caller (`core/native.py:138-146`).

What the crate *did* buy was full native-build cost: a maturin/pyo3 version-pin
surface, a three-platform cibuildwheel matrix, and a **Linux aarch64 gap** where
QEMU cross-compiles hung past 3 hours. Removing it flips the build to
`setuptools` and produces a single **universal `py3-none-any` wheel** — which
installs natively on aarch64 Linux (and everywhere else) with no compilation.
The aarch64 gap closed for free. Behavior is bit-identical on every code path
that mattered: cosine scores (already NumPy/BLAS) and negation output
(pure-Python == the Rust port).

The decision is documented in three places, each of which tells the same story:
`CHANGELOG.md` v1.3.0, the `core/native.py` module docstring, and the
superseding note in `docs/architecture.md:74-87`. `HAVE_NATIVE` is retained as
a constant `False` so downstream code that branches on it still works
(`core/native.py:34`).

One real fix rode along: in `engine/adaptive_search.py` the dispatcher used to
route small-N queries to a numpy **int8 correctness oracle** that upcasts to
int32 and is slower than BLAS at every scale. With the native kernel gone, the
dispatcher now routes to zero-copy BLAS at every N unless a developer wires the
prototype `.so` in via `IZERO_INT8_NATIVE_SO` (`adaptive_search.py:434-436`).

## 4. Architecture

```
                    ┌────────────────────────────────────────────────────┐
   SURFACE          │  izero CLI · izero-cli (read-only) · MCP server    │
                    │  adapters: LangChain · LlamaIndex · AutoGen · CrewAI│
                    └────────────────────────┬───────────────────────────┘
                                             │ remember / recall / touch / consolidate
                    ┌────────────────────────▼───────────────────────────┐
   FACADE           │  IsotopeZero client (client.py)                     │
                    └──────────────┬─────────────────────────┬───────────┘
                                   │ embed_text              │ cards & queries
                    ┌──────────────▼──────────┐   ┌──────────▼────────────────────────┐
   ENGINE           │ HybridEmbeddingEngine   │   │ MemoryStore (core/store.py)         │
                    │ (embeddings/engine.py)  │   │  SQLite WAL + float32 matrix cache  │
                    │ daemon-first, silent    │   │  native.py   — matrix @ q → BLAS    │
                    │ in-process ONNX fallback│   │  adaptive_search.py — N-routed      │
                    │                        │   │  decay.py · consolidation.py         │
                    └──────────────┬──────────┘   └──────────────┬─────────────────────┘
                                   │ /tmp/izero.sock             │ one sqlite3 conn (lock)
                    ┌──────────────▼──────────┐   ┌──────────────▼─────────────────────┐
   STORAGE          │ Embedding daemon        │   │ SQLite (WAL)                       │
                    │ onnxruntime, once/process│  │  memories + card_edges + FTS5      │
                    │ all-MiniLM-L6-v2, dim 384│ │  NumPy/BLAS matrix @ query          │
                    │ ~360 MB, kept in ONE proc│  │  (Accelerate / OpenBLAS)           │
                    └─────────────────────────┘   └────────────────────────────────────┘
```

The invariant that shaped the layout: **the embedding backend, not the storage
tier, is the only resident-set lever.** The float32 matrix is ~15 MB at 10k
cards; `onnxruntime` is ~360 MB. The daemon exists to centralize the 360 MB
once per host, so the client process that actually runs the agent stays small.
Every storage-tier optimization this project measured (int8 quantization, mmap,
1-bit POPCNT) attacked the 15 MB matrix and was, at best, a wash — that is the
measured reason `use_mmap=False` is the default and int8 is a research variant
(`docs/architecture.md` documents all eight phases and the refutations).

## 5. Where we are vs mem0

Honest comparison, not cherry-picked. Isotope Zero figures are measured in this
repo; mem0 figures are from `MEM0_COMPARATIVE_AUDIT.md` (audit-derived from the
mem0 codebase, or vendor-stated).

| Dimension | isotope_zero | mem0 |
|---|---|---|
| Vector read p99 @ 10k | **0.28–0.31 ms** (measured, local) | 50–200 ms (network RTT + API + rerank) |
| Cold start | **< 50 ms** (35 ms measured) | 2–5 s (LLM client + API handshake) |
| Per-op cost | **$0** (local ONNX) | per-call embedding API bill |
| Network calls | **0** for core ops | every add/search hits remote APIs |
| Data residency | **your file, your machine** | facts flow to embedding + vector-store APIs |
| Dependency weight | **1 hard dep** (numpy); sqlite3 stdlib | 150+ packages, 22 vector stores, 24 LLM clients |
| Fact reconciliation | deterministic negation + semantic consolidation | LLM-driven V3 additive extraction + MD5 dedup |
| Temporal forgetting | **Ebbinghaus decay, built-in** | none (manual) |
| Knowledge compaction | **consolidate() — 98.5% reduction (measured)** | additive only |
| Graph memory | `card_edges`: cosine + shared-tag edges, BFS clusters | **spaCy entity-linking → `linked_memory_ids`** — richer entity model |
| Multi-tenancy | multi-tier `scope=` row isolation; single-tenant by design | **payload metadata filtering; production multi-tenancy** |
| Managed hosting | none — you own the DB file | **hosted platform + fleet API** |
| Ecosystem | LangChain/LlamaIndex/AutoGen/CrewAI, MCP, CLI | **larger OSS + platform, more frameworks, skills/plugins** |

**Where we win:** cold start, latency, cost, privacy, dependency weight,
offline operation. These are structural — a local in-process architecture beats
a remote-API architecture on all four, and no feature work erases that.

**Where mem0 wins:** graph/entity memory (an LLM can extract entities and links
that a deterministic pipeline cannot), managed hosting, and ecosystem breadth
(more integrations, more tooling). Multi-tenancy maturity matters at
production scale; isotope_zero's `scope=` filtering is real but single-tenant
in spirit.

## 6. The REROUTE roadmap

After v1.1.x shipped, the work split into phases. **Phases 1–6 are complete.**
Each is a commit or a small set of commits on `main`.

- **Phase 1 — Stop the bleeding** (`a0c1858`): the main-CI red jobs were a
  real concurrency double-fault in `consolidate_memories`: a contended
  `BEGIN IMMEDIATE` failed instantly (no `busy_timeout`), then the exception
  handler called `ROLLBACK` on a transaction that never began, masking the
  real error with `cannot rollback - no transaction is active`. Fixed by
  guarding the rollback behind `conn.in_transaction` and setting
  `busy_timeout` on the store's own connection (`store.py:428-436`, `1120-1130`).
  Release pipeline hardened (post-publish smoke, pinned actions, OIDC).
- **Phase 2 — The Rust reckoning** (`310708b`, `8ae7840`): the decision to
  remove the Rust extension and ship the pure-Python wheel. Section 3 is the
  full story.
- **Phase 3 — Concurrency correctness** (`28c29e6`): `AdaptiveVectorSearch`
  search reads made lock-free via per-call local buffers, with a single lock
  guarding writes only (`adaptive_search.py:388-463`).
- **Phase 4 — Close the capability gap vs mem0** (`5782b68`): added
  `recover_card()` (restores archived cards, symmetric with `archive_card()`,
  `store.py:935`) and closed the scoping + hybrid-retrieval items in
  `MEM0_COMPARATIVE_AUDIT.md`. Multi-tier `scope=` isolation and RRF hybrid
  retrieval had shipped in v1.0.0; this phase marked them CLOSED.
- **Phase 5 — Independent verification** (`6254e34`): reproducible perf and
  correctness scripts that don't trust the benchmark suite:
  `scripts/verify_perf.py` (measured on this machine: vector p99 **0.31 ms**
  @ 10k, cold-start import **12.5 ms**, RSS **186 MB** with ONNX) and
  `scripts/verify_negation.py` (47-case curated negation audit: **precision
  1.0, recall 0.864**; the 3 false negatives are documented stemmer
  limitations, zero false positives).
- **Phase 6 — Documentation & repo hygiene**: reconciling the last Rust-era
  references with the pure-Python wheel, cleaning the docs narrative (this
  file included), and the setup.py/MANIFEST.in packaging cleanup.

**What's next.** Phase 7 is the go-to-market foundation (installer story, npm
launcher, release cadence). Open technical items, stated plainly: mem0's
entity-linking graph remains a genuine gap (Section 5); the MCP server and
adapters are thinner than mem0's ecosystem; and there is a known test-hygiene
loose end — `test_dashboard_once_empty` currently reads the *real*
`~/.isotope_zero/isotope_zero.db` and fails when that file has cards, instead
of an isolated temp DB.

## 7. Credits

Isotope Zero is the product of an eight-phase R&D program, each phase a
single-purpose prototype under `prototypes/` that validated or refuted one
hypothesis against a measured claims table. The refuted phases (BM25
pre-filter, int8 quantization, mmap, 1-bit POPCNT) are not failures; they are
the project's results — they foreclose the obvious storage-tier optimizations
and prove where the real levers are. See `docs/architecture.md` and
`docs/complete_history.md` for the full record.

Core design and implementation by Svanik Kolli, with the REROUTE engineering
phases co-authored with Claude Code (per commit attribution). The mem0
comparison is grounded in a line-level audit of the `mem0_repo/` tree
(`MEM0_COMPARATIVE_AUDIT.md`), not vendor marketing.
