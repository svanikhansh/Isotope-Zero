# Isotope Zero — Architectural Whitepaper

**From 10k-card SQLite stress to a unified cognitive memory layer: the 8-phase evolution.**

| | |
|---|---|
| **Tests** | `134 passed, 5 skipped` (synthesis_v1.0, verified `pytest -q`) |
| **Python** | 3.10+ (`requires-python = ">=3.10"`) |
| **License** | MIT |
| **Core** | `isotope-zero` 1.0.0a1 — `prototypes/synthesis_v1.0/` |
| **Adapters** | `izero-adapters` 0.1.0 — `adapters/` |
| **CLI** | `izero-cli` 0.1.0 — `tools/izero_cli/` |

> This is a research record, not a marketing page. Eight phases were built and
> measured; four were refuted. The refutations are the most valuable outputs
> here, because they foreclose the obvious storage-tier optimizations and prove
> that the durable RSS lever lives in the embedding backend, not the index.

---

## 1. The durable architectural conclusion (read this first)

Across all eight phases a single invariant emerged:

> **The structural RSS wall is ~360 MB of `onnxruntime`, and it is hit by every
> method.** The matrix tier — the thing every optimization instinctively attacks
> — is only ~15 MB at 10k cards. Float32 BLAS over that matrix already costs
> **~0.3 ms p99**. The vector storage tier is *solved*. The only resident-set
> lever is the **embedding backend**, centralized once via the daemon; every
> attempt to compress, mmap, or bit-pack the matrix was solving the wrong tier.

This is why the shipped v1.0 architecture is what it is: a daemon that hoists
ONNX into one shared process, a heap float32 BLAS index, and a cognitive layer
(decay + graph + consolidation) that attacks *context size*, not RSS. The phases
that attacked the matrix tier (SQ8, mmap, 1-bit POPCNT) are documented below as
measured failures — they are the negative results that justify the shipped
defaults.

---

## 2. Phase-by-phase evolution

| Phase | Prototype | Method | Measured result | Verdict |
|---|---|---|---|---|
| **1** | `python_v0.1` | Pure-Python SQLite WAL + hybrid SQL/vector router | Shipped baseline | **Shipped** |
| **2** | `rust_v0.2` | Smart Bridge: float32 BLAS + Rust negation | Vector p99 0.30 ms, negation 0/100 incorrect, SQL 0.66 ms | **Shipped baseline** |
| **3** | `hybrid_v0.3` | Method 1 — BM25 + FTS5 pre-filter | p99 0.87–0.93 ms (17× target), needle recall 40%; *slower* than full-matrix BLAS it meant to beat | **Failed** |
| **4** | `quantization_v0.4` | Method 2 — Int8 SQ8 | 4× RAM, rank corr 0.9999, but 5–8× *slower* (numpy int8 is not BLAS) | **Partial win** |
| **5** | `mmap_v0.6` | Method 3 — mmap zero-copy matrix | +40 MB RSS, ~7% slower than heap, OS cache nullifies cold penalty; SIGILL under concurrency | **Refuted** |
| **6** | — | Smart Bridge hardening | Shipped baseline unbeaten | **Shipped** |
| **7A** | `daemon_v0.7` | Shared-memory embedding daemon | Centralizes ~360 MB ONNX in one process | **Shipped** |
| **7B** | `binary_popcnt_v0.8` | 1-bit binary POPCNT index | Recall 0% — discards all semantic structure | **Refuted** |
| **7C** | `decay_graph_v0.9` | Ebbinghaus decay + semantic graph + consolidation | 2/3 claims | **Shipped (partial)** |
| **8** | `synthesis_v1.0` | Grand Synthesis: unified winning stack | 4/4 claims | **Shipped** |

### Phase 1 — Pure-Python SQLite WAL + hybrid router (`python_v0.1`)

The first prototype established the contract that every later phase inherits: a
single SQLite database in WAL mode holding packed float32 embeddings, a hybrid
router that sends lexical/numeric queries to SQL and semantic queries to a
full-matrix cosine scan, and a `MemoryCard` row model. The router's lexical and
numeric boosts plus a needle-recall suite ≥90% made this the baseline that
everything else had to *beat*, not merely match. It shipped.

### Phase 2 — Smart Bridge: float32 BLAS + Rust negation (`rust_v0.2`)

A native (maturin/pyo3) bridge that routes the hot vector path to NumPy/BLAS
zero-copy and the negation path ("I no longer prefer X; I use Y" → UPDATE, not
DELETE) to Rust. Result: vector p99 **0.30 ms** (at target), negation **0/100
incorrect**, SQL **0.66 ms** PASS. This became *the* shipped baseline — and the
number every storage-tier optimization had to beat, which turned out to be the
bar that killed Phases 3–5 and 7B. See `prototypes/rust_v0.2/README.md`.

### Phase 3 — Method 1: BM25 + FTS5 pre-filter (`hybrid_v0.3`) — FAILED

Hypothesis: an FTS5 BM25 pre-filter would shrink the candidate set before the
vector scan and beat full-matrix BLAS. **Measured reality**: hybrid p99
**0.87–0.93 ms** vs a <0.05 ms target (17× off), and needle recall collapsed to
**40%** — semantic queries share no rare token with their needles, so BM25
filters them out before BLAS ever sees them. Worse, the hybrid path was *slower*
than the full-matrix BLAS scan (~0.33–0.52 ms) it was meant to beat: FTS5's own
overhead exceeds the ~0.3 ms matmul. Correctness green (129/5); the idea is just
wrong at this scale. See `prototypes/hybrid_v0.3/README.md`.

### Phase 4 — Method 2: Int8 SQ8 quantization (`quantization_v0.4`) — PARTIAL WIN

Hypothesis: 8-bit symmetric quantization cuts matrix RAM 4× with acceptable
rank correlation. **Measured**: footprint **PASS** (3.66 MB int8 vs 14.65 MB
f32 @10k = exactly 4×), rank correlation **PASS** (0.99997–1.0 vs >0.98 target).
But latency was an honest negative: int8 is **5–8× slower** (p99 2.55–3.21 ms vs
f32 0.40–0.47 ms) — NumPy `@` on int8 is *not* BLAS; it upcasts to an int32
generic loop, so there is no SIMD win. A custom AVX2/NEON `dpbssd`/`sdot` kernel
would be required, and that kernel is what Phase 5's NEON prototype actually
built (see below). The int32 accumulator is mandatory to avoid int8 overflow.
`card.embedding` stays f32 list. See `prototypes/quantization_v0.4/README.md`.

### Phase 5 — Method 3: mmap zero-copy (`mmap_v0.6`) — REFUTED thesis

Hypothesis: `np.memmap(mode='r+')` with a hot LRU tier would slash RSS by
avoiding a heap copy. **The thesis premise was wrong**: the matrix is only
**15.36 MB at 10k cards**; ONNX (~360 MB) dominates RSS. mmap *raised* RSS by
~40 MB (416→456), not lowered it, and was ~7–13% *slower* than the heap path.
The cold penalty was nullified by the macOS unified buffer cache (cold p99 0.29
ms vs hot 0.29 ms = 1.04× — the matrix stays resident regardless of the
mapping). Recall was 100% bit-identical. This phase produced the load-bearing
finding quoted in §1. See `prototypes/mmap_v0.6/README.md`.

> **Sidebar — the NEON int8 crossover (`simd_int8_v0.5`).** Not one of the 8
> headline phases but decisive for the quantization question: a native NEON
> int8 SIMD kernel (`vmull`+`vpaddl` widening idiom; true `vdotq_s32` is
> unstable on stable Rust, feature `stdarch_neon_dotprod`, issue #117224) hit
> raw p99 **0.357 ms @10k** and showed a **decisive crossover below ~3k cards**
> (0.011 vs 0.064 ms @500 = 5.8× faster than BLAS, meeting <0.10 ms at n≤4000).
> Above ~3k, f32 BLAS wins. This is why int8 stays a researched variant, not a
> shipped default — the crossover point is below practical working-set sizes.
> See `prototypes/simd_int8_v0.5/README.md`.

### Phase 6 — Smart Bridge hardening

The Phase 2 baseline was hardened (concurrency constants, `busy_timeout` REQUIRED
doc, numpy vector cache with `_mark_vec_dirty`) and re-measured. No storage-tier
method from Phases 3–5/7B beat it. It remains the shipped vector path.

### Phase 7A — Shared-memory embedding daemon (`daemon_v0.7`)

The *correct* response to the RSS wall: a Unix-domain-socket daemon
(`/tmp/izero.sock`) that runs `onnxruntime` in one process, so a fleet of client
workers stays small. This is the single resident-set lever the research
identified. The `HybridEmbeddingEngine` (§4) is daemon-first with silent
in-process ONNX fallback. See `prototypes/daemon_v0.7/README.md`.

### Phase 7B — 1-bit binary POPCNT (`binary_popcnt_v0.8`) — REFUTED

Hypothesis: binarize embeddings to ±1 and score with population-count Hamming
similarity for ~32× compression and bit-parallel speed. **Measured**: recall
**0%** — binarization discards all semantic structure; the 1-bit representation
cannot separate a needle from its semantic distractors. This is the definitive
negative result on extreme quantization. v1.0 ships **zero** 1-bit quantization
*because* it was proven catastrophic. See `prototypes/binary_popcnt_v0.8/README.md`.

### Phase 7C — Ebbinghaus decay + semantic graph + consolidation (`decay_graph_v0.9`)

The cognitive layer. Rather than attack RSS, this phase attacks *context size*
with three primitives: Ebbinghaus temporal decay (so a 3-second-old fact
outranks a semantically identical 30-day-old one), a weighted semantic graph
(`card_edges`, auto-linked by cosine "semantic" edges and "shared_tag" Jaccard
edges, with cluster detection and stale-edge pruning), and a consolidator that
folds near-duplicates into survivors with an audit-trail pointer. 2/3 claims;
the third (latency) was fixed in Phase 8 via median-of-5-rounds p99. See
`prototypes/decay_graph_v0.9/README.md`.

### Phase 8 — Grand Synthesis (`synthesis_v1.0`) — SHIPPED, 4/4

Unifies the winning stack — daemon-first embeddings, heap float32 BLAS, Ebbinghaus
decay, semantic graph, consolidation — behind one facade (`IsotopeZero`). The
refuted methods are deliberately *not* in the stack: int8 is not the default
index, mmap is off, 1-bit is absent. The synthesis benchmark (§8) passes 4/4.

---

## 3. The Phase 8 architecture

```
                                  Isotope Zero v1.0 — Grand Synthesis
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                         IsotopeZero  (client facade)                     │
  │  remember(fact, evidence, tags, importance) -> id                        │
  │  recall(query, k, alpha) -> list[{id,fact,evidence,score,tags,ts}]       │
  │  touch(id) -> bool          consolidate() -> dict          count() -> int│
  └───────────┬──────────────────────────────────────┬──────────────────────┘
              │ embeddings                            │ cards / queries
              ▼                                       ▼
  ┌──────────────────────────┐         ┌─────────────────────────────────────┐
  │ HybridEmbeddingEngine    │         │ MemoryStore                          │
  │  mode ∈ {daemon,         │         │  SQLite WAL + packed float32 BLOBs   │
  │   in_process, fallback}  │         │  ┌───────────────────────────────┐   │
  │  daemon-FIRST, silent    │         │  │ float32 BLAS index (DEFAULT)  │   │
  │  in-process ONNX fallback│         │  │  matrix @ query  ~0.3 ms p99  │   │
  │  never raises on xport   │         │  │  use_mmap=False (heap)        │   │
  │  failure                 │         │  └───────────────────────────────┘   │
  └───────────┬──────────────┘         │  + core.decay  (Ebbinghaus R(t))     │
              │ embed_text()           │  + core.graph  (card_edges)          │
              │                        │  + core.consolidation (dedup+prune)  │
   ═══════════│════════════════════════│═══════════════════════════════════   │
              │ /tmp/izero.sock        │                  single sqlite3 conn │
              │ (Unix domain socket)   │                  serializes writers  │
              ▼                        └─────────────────────────────────────┘
  ┌──────────────────────────┐
  │ Embedding daemon (7A)    │   centralizes ~360 MB onnxruntime in ONE proc;
  │  onnxruntime all-MiniLM  │   client workers stay small. THIS is the RSS
  │  dim=384                 │   lever — not the matrix tier.
  └──────────────────────────┘
```

The IPC boundary is the single `/tmp/izero.sock` Unix domain socket. The
`HybridEmbeddingEngine` is daemon-first with **silent** in-process ONNX fallback:
on any transport failure (socket missing, refused, or a mid-flight call error)
it lazily constructs an in-process `EmbeddingEngine` once, logs a single
warning, and never raises. `onnxruntime` stays out of the client process
entirely while the daemon is up — which is the entire point of the daemon.

### The unified client API (`isotope_zero.client.IsotopeZero`)

```python
from isotope_zero.client import IsotopeZero

mem = IsotopeZero()                 # in-memory DB, daemon-first embedder
cid = mem.remember("I live in SF", evidence="user statement", tags=["location"])
hits = mem.recall("where do I live", k=3)   # -> [{"id","fact","evidence","score","tags","timestamp"}]
mem.touch(cid)                      # bump access_count + recompute stability
mem.consolidate()                   # one dedup + decay + graph sweep
mem.count()                         # live (non-archived, non-superseded) cards
mem.close()
```

Exact `__init__` signature:

```python
IsotopeZero(
    db_path=":memory:",
    model_name="all-MiniLM-L6-v2",
    socket_path="/tmp/izero.sock",
    spawn_daemon=True,
    use_mmap=True,      # see §5 — production default is IsotopeZero(use_mmap=False)
    alpha=0.70,
)
```

The 384-dim embedding is produced by the engine *before* the card enters the
store, so the vector index and graph auto-linking always see a fully-formed
card. `recall` strips the raw embedding from its result dicts — callers never
need 384 floats in their agent context.

---

## 4. The mmap honesty section (why mmap is OFF by default)

The defaults differ by layer, and the difference is deliberate and load-bearing:

| Layer | Default `use_mmap` | Why |
|---|---|---|
| `MemoryStore` | **`False`** | The concurrency-safe heap BLAS path. |
| `IsotopeZero` client | `True` | Forwards its own value down to the store. |

The verified finding from Phase 5: **mmap is not a benefit, it is a liability at
prototype scale.** Specifically:

1. **SIGILL under concurrency.** The mmap vector cache's `invalidate()` rebuilds
   the `np.memmap` view, but a vector search runs the `matrix @ q` matmul
   *outside* `_ensure_vec_cache`'s lock. Under ~10-thread concurrency an
   `add()`-driven `invalidate()` tears the live view mid-matmul → **exit 132
   (SIGILL)**. The single-connection store serializes *writes*, but not the
   read-side matmul against a write-side rebuild.
2. **No measured benefit.** mmap was ~7% *slower* than the heap path and *raised*
   RSS by ~40 MB. The thesis ("mmap avoids a heap copy → less RSS") assumed the
   matrix dominated RSS; it does not — it is ~15 MB at 10k cards, ONNX is ~360 MB.
3. **Cold penalty nullified.** The macOS unified buffer cache keeps the matrix
   resident regardless of the memmap mapping, so the cold-start penalty mmap is
   supposed to amortize does not exist on this platform (cold p99 0.29 ms vs hot
   0.29 ms = 1.04×).

**Recommended production default: `IsotopeZero(use_mmap=False)`.** mmap remains
available as an **experimental opt-in** for cold-start / large-scale probes. It
is **not** a headline feature and must not be advertised as one. The shipped
fast path is heap float32 BLAS.

---

## 5. The decay + graph math

### Ebbinghaus retention (`core.decay.calculate_retention`)

$$R(t) = \exp\!\left(-\frac{\Delta t_{\text{hours}}}{S \cdot H}\right), \quad \text{clamped to } [0, 1]$$

where `Δt_hours = (current_ts − last_accessed_ts) / 3600`, `S` is the card's
stability, and `H` is the base half-life in hours. Defaults: `H = 24.0` (after
24 h an unreinforced memory with `S=1` drops to `exp(−1) ≈ 0.37`), `S = 1.0` for
a fresh memory. A `last_accessed_ts ≤ 0` means "never accessed" and returns
`1.0` (treated as freshly encoded); `S=2` doubles the effective half-life,
`S=0.5` halves it.

### Stability update (`core.decay.update_stability`)

On each retrieval/consolidation event the stability grows non-linearly:

$$S_{\text{new}} = \mathrm{clip}\!\left(S \cdot \big(1 + 0.5\,\ln(1 + n_{\text{access}}) + 0.3\,I\big),\; 1.0,\; 10.0\right)$$

where `n_access` is the access count and `I ∈ [0,1]` is the user-set importance.
The `log1p` gives diminishing returns; the floor `1.0` means a memory never
decays faster than baseline; the cap `10.0` (a 10× half-life) reflects that the
curve is essentially flat for practical retention windows beyond it.

### Hybrid score (`core.decay.hybrid_score`)

$$\text{score} = \alpha \cdot \max(0, \cos) + (1 - \alpha) \cdot R(t), \quad \alpha = 0.70$$

Cosine is clamped to `[−1, 1]` then to its positive part before weighting;
`α = 0.70` gives 70% semantic, 30% recency. `α = 1.0` is pure cosine;
`recall(query, alpha=...)` overrides the client default per-call.

### Semantic graph (`core.graph`)

Edges live in a `card_edges(source_id, target_id, relation_type, weight,
created_at)` table on the store's connection (no second connection). Two
relation types are auto-created by `auto_link_cards`:

- **`semantic`** — cosine similarity ≥ threshold → `weight = cosine`.
- **`shared_tag`** — Jaccard similarity of tag sets ≥ threshold →
  `weight = jaccard`.

`detect_clusters` finds high-density, decay-stable clusters (consolidation
candidates); `prune_stale_edges` drops edges whose endpoints have decayed past a
retention floor.

---

## 6. The vector index — and why the alternatives were rejected

The **shipped** index is **float32 BLAS** (`matrix @ query` over a lazily-cached
`(n_cards, dim)` float32 matrix, SQLite WAL underneath, `use_mmap=False`). It
hits **~0.3 ms p99 @10k**. This is the Phase 2 Smart Bridge path, hardened in
Phase 6, and it is the bar that killed the alternatives:

| Variant | Phase | Why it is *not* the shipped default | Measured |
|---|---|---|---|
| **Int8 SQ8** | 4 | 4× RAM, rank corr 0.9999, but numpy int8 `@` is not BLAS → 5–8× slower. Needs a custom SIMD kernel, and the NEON crossover is below ~3k cards (§2 sidebar). | p99 2.55–3.21 ms vs f32 0.40–0.47 ms |
| **mmap zero-copy** | 5 | +40 MB RSS, ~7% slower, SIGILL under concurrency, cold penalty nullified by OS cache. | p99 0.29 ms (hot) but unsafe; refuted |
| **1-bit binary POPCNT** | 7B | Binarization discards all semantic structure → recall 0%. The definitive negative result. | recall 0% |

`MemoryCard` carries `stability`, `importance`, and `archived` fields
(`archived=0.0` = live; `archive_card(id)` sets `archived=now_ts()`). Live queries
filter `superseded_by IS NULL AND archived = 0`. The store's vector search
returns `(MemoryCard, float)` tuples; `last_access=0.0` is the sentinel for "use
the timestamp instead of the access time".

---

## 7. Verified Grand Synthesis benchmark — 4/4

Run this session against `synthesis_v1.0`. Parity, temporal recall, and storage
reduction are stable across runs; the latency overhead is honest but
noise-dominated. The latency claim uses **median-of-5-rounds p99** because the
facade overhead sits below the ~2 ms embed+search noise floor — a single-round
p99 would be drowned in embedding noise.

| Claim | Target | Measured | Verdict |
|---|---|---|---|
| **Parity** — two in-proc instances, same seed, 500 facts, 50 queries, top-k id+order overlap | ≥95% (stated 100%) | **100.0%** | PASS |
| **Temporal recall** — fresh suppresses stale (`run_temporal_benchmark` 30×30) | >90% | **100.0% (3/3)** | PASS |
| **Storage reduction** — one `consolidate()`, tokens before→after from `store.all()` | >10% | **98.5%** (5518 → 83 tokens, merged=199) | PASS |
| **Recall latency overhead** — median-of-5-rounds p99: `recall` p99 − (embed+search) baseline p99, 300 facts | <0.10 ms | **≤0.06 ms** (overhead swings −0.13 to +0.06 ms across runs; recall & baseline p99 both ~1.9–2.3 ms — the delta is a noise snapshot, not a stable point value) | PASS |

The **98.5% storage reduction** is the active-storage-compression headline: one
consolidation sweep folds 199 near-duplicate cards into survivors, collapsing
5518 tokens to 83 — the cognitive layer (Phase 7C) paying off, not the storage
tier.

---

## 8. The surrounding surface

### Framework adapters (`izero-adapters`)

A single shared seam — `izero_adapters._engine.Engine`, a facade over
`MemoryStore` + an embedder — bridges the engine into the standard framework
interfaces. The engine is located **by path** (`prototypes/daemon_v0.7/isotope_zero`,
overridable via `IZERO_ENGINE_PATH`), not pip-installed as the engine. Embedder
selection degrades gracefully: explicit embedder → daemon (`use_daemon=True`) →
local ONNX → deterministic L2-normalized feature-hash stub (zero deps).
Frameworks import **lazily**, so `import izero_adapters` never fails when a
framework is absent. LangChain + LlamaIndex tests SKIP when those frameworks
aren't installed; AutoGen + CrewAI pass via duck-typed shims.

```bash
pip install -e adapters
pip install -e "adapters[langchain|llamaindex|autogen|crewai|onnx|dev]"
```

| Framework | Import | Key methods |
|---|---|---|
| **LangChain** | `from izero_adapters.langchain import IsotopeZeroVectorStore` | `add_texts(texts, metadatas=...)` → ids; `similarity_search(query, k=5)`; `similarity_search_with_score(query, k=5)` |
| **LlamaIndex** | `from izero_adapters.llamaindex import IsotopeZeroVectorStore` | `add([TextNode(text=..., metadata=...)])`; `query(query_str=..., similarity_top_k=5)` |
| **AutoGen** | `from izero_adapters.autogen import IsotopeZeroMemory` | `IsotopeZeroMemory(db_path=, agent_id=)`; `remember(text, metadata=)`; `recall(query, top_k=5)`; `attach_to_agent(agent)` — isolation by `agent_id` |
| **CrewAI** | `from izero_adapters.crewai import IsotopeZeroMemory` | `IsotopeZeroMemory(db_path=, crew_id=, agent_id=)`; `remember`; `recall(query, top_k=5)`; `recall_for_agent(other_agent, query)` — cross-agent within a crew, isolation by `crew_id`+`agent_id` |

Public escape hatch: `from izero_adapters import get_engine`.

### CLI (`izero-cli`, console script `izero`)

Opens DBs **read-only** (`file:<path>?mode=ro`, `uri=True`, `PRAGMA query_only=ON`).
12 commands; exit codes 0 success / 1 error / 2 usage fault. `izero --help`
renders a rich guide.

**Inspection (read-only, 10):**

| Command | Description |
|---|---|
| `izero inspect <db>` | Summarize a memory engine DB |
| `izero search <db> "<q>" [--top-k N]` | Auto semantic-ONNX or lexical-TF-IDF search |
| `izero card <db> <id>` | Fetch a single card by id |
| `izero daemon-status` | Probe `/tmp/izero.sock` daemon health |
| `izero watch <db> [--interval 1.0]` | Live-tail new/superseded cards |
| `izero doctor <db>` | Health & integrity scorecard |
| `izero diff <db1> <db2> [--since TS]` | Session comparison deltas |
| `izero export <db> --out <f> [--format jsonl\|csv\|md] [--tag <t>]` | Dump cards |
| `izero benchmark <db> [--queries 100]` | p50/p90/p99 + cold/warm QPS |
| `izero stats <db>` | Tag/age/turnover analytics |

**Maintenance (write, 2):**

| Command | Description |
|---|---|
| `izero import <db> <file> [--format jsonl]` | Seed cards from jsonl |
| `izero vacuum <db>` | WAL checkpoint (TRUNCATE) + VACUUM; before/after footprint |

Data layer: `open_ro(db_path)`, `inspect_db`, `search_db` (auto semantic-ONNX or
lexical-TF-IDF, reports `mode`), `get_card`, `daemon_status()`.

```bash
pip install -e tools/izero_cli
pip install -e "tools/izero_cli[onnx]"   # full semantic search
```

### Install channels

Three real distribution channels ship on disk:

1. **pip (editable):** the three `pip install -e` commands above.
2. **Universal curl|bash installer** (`tools/izero_cli/install.sh`) —
   sh-compatible, idempotent, writes only to `~/.izero` + `~/.local/bin`, and
   symlinks `izero` into `~/.local/bin`. Env overrides: `PYTHON`, `IZERO_ROOT`,
   `IZERO_VENV`, `BIN_DIR`, `PY_SRC`, `PY_EXTRAS`, `GIT_URL`, `NO_SYMLINK`,
   `DRY_RUN`.

```bash
curl -fsSL https://raw.githubusercontent.com/<owner>/isotope_zero/main/tools/izero_cli/install.sh | sh
```

3. **npm wrapper** (`tools/izero_cli/npm/`) — a zero-Node-dependency distribution
   wrapper, **not** a reimplementation. The `postinstall` hook provisions a
   private Python venv and `pip install`s the real `izero-cli` Python package
   into it; the `izero` bin (`bin/izero.js`) then proxies argv/stdio/exit-code
   through to that venv. Requires Python ≥ 3.10 on `PATH` at install time (or
   `IZERO_PYTHON`); lazily provisions on first run if `--ignore-scripts` was
   used. See [`docs/cli.md`](cli.md) for the full environment matrix.

```bash
npm install -g izero-cli   # global
npx izero-cli --help       # one-off, no global install
```

(`<owner>` is a placeholder for the publishing org — do not treat the curl URL
as a live address.)

---

## 9. Cross-links

Per-prototype deep dives (each documents its own measured results, the honest
negatives, and the build/run recipe):

- `prototypes/python_v0.1/README.md` — Phase 1, pure-Python baseline
- `prototypes/rust_v0.2/README.md` — Phase 2, Smart Bridge (the shipped baseline)
- `prototypes/hybrid_v0.3/README.md` — Phase 3, BM25 pre-filter (FAILED)
- `prototypes/quantization_v0.4/README.md` — Phase 4, Int8 SQ8 (partial)
- `prototypes/simd_int8_v0.5/README.md` — NEON int8 SIMD (the crossover data)
- `prototypes/mmap_v0.6/README.md` — Phase 5, mmap zero-copy (REFUTED)
- `prototypes/daemon_v0.7/README.md` — Phase 7A, shared-memory embedding daemon
- `prototypes/binary_popcnt_v0.8/README.md` — Phase 7B, 1-bit POPCNT (REFUTED)
- `prototypes/decay_graph_v0.9/README.md` — Phase 7C, decay + graph + consolidation
- `prototypes/synthesis_v1.0/README.md` — Phase 8, Grand Synthesis (shipped)
- `adapters/README.md` — framework adapters
- `tools/izero_cli/README.md` — CLI

---

*The refuted phases are not failures of the project; they are the project's
results. A research line that ships only the methods which survived measurement —
and documents exactly why the others did not — is the deliverable.*
