# Isotope Zero Integration Contract (authoritative)

All modules import shared types from `isotope_zero.types` and token math from `isotope_zero.tokens`. Do not duplicate these. Package: `isotope-zero` on PyPI, module `isotope_zero`, console scripts `izero`, `izero-mcp`, `izero-benchmark`. Logger namespace: `isotope_zero.*`.

## Types (already defined in isotope_zero/types.py — import, do not redefine)

- `ActionType` enum: `ADD`, `UPDATE`, `DELETE`.
- `ActionResult(action, confidence, escalated=False, target_id=None, reasoning="")` — `confidence` in `[0.0, 1.0]`; `escalated=True` when the fast heuristic deferred to the mocked LLM path.
- `MemoryCard(id, fact, evidence, timestamp, tags=[], embedding=None, source_tokens=0)` — `.to_dict()` serializes all fields.
  - Phase 3 access tracking: `access_count=0`, `last_access=0.0` (0.0 ⇒ treat as the card's `timestamp`; feeds the temporal-decay vitality score).
- `QueryHit(card, score, route, token_cost)` — `route` is `"sql"` | `"vector"`; `score` in `[0,1]`.
- `QueryResult(hits, route_used, tokens_used, tokens_saved_vs_raw, latency_ms, budget_exhausted)`.
- `ConsolidationReport(merged_cards, decayed_cards, survivors, tokens_before, tokens_after, tokens_reclaimed, latency_ms, pruned_mean_vitality=0.0)` — outcome of one consolidation sweep.
- `now_ts() -> float` — wall-clock timestamp, wrapped so tests/benchmarks can monkeypatch determinism.

## Token math
- `from isotope_zero.tokens import estimate_tokens` — `estimate_tokens(text: str) -> int`. Stdlib-only regex splitter (~4 chars/token), sub-ms, the single source of truth used on every write and read.

## Module responsibilities

### isotope_zero/embeddings/onnx_embed.py  →  class `EmbeddingEngine`
- `EmbeddingEngine(model_name="all-MiniLM-L6-v2", dim=384, quantized=True, cache_dir=".isotope_zero_cache")`
- `.embed_text(text: str) -> list[float]` — empty input → zero vector.
- `.embed_batch(texts: list[str]) -> list[list[float]]`.
- `.dim -> int`  (property).
- `.is_real -> bool` — `True` on the real ONNX path; `False` on the deterministic feature-hash fallback.
- Must be **zero network calls** at inference time. Load ONNX + tokenizer from local cache; if absent, download ONCE from HuggingFace at construction, then cache to `cache_dir`.
- Must degrade gracefully: if onnxruntime/tokenizers unavailable OR download fails, fall back to a deterministic, L2-normalized hash-based pseudo-embedding of the correct `dim` so identical inputs still score 1.0. Log a warning. Set `.is_real = False`.
- Embeddings MUST be L2-normalized so cosine similarity = dot product. Supports `all-MiniLM-L6-v2` (dim 384) and `bge-micro-v2` (dim 384).

### isotope_zero/core/triage.py  →  functions
- `classify_action(input_text: str, context: str = "") -> ActionResult`
  - Fast stdlib-only heuristics first (keyword/phrase signals for ADD/UPDATE/DELETE; `_ESCALATE_THRESHOLD = 0.6`).
  - On low confidence (<0.6), escalate to a *mocked* LLM path (a deterministic re-examination, NOT a real API call) and set `escalated=True`.
- `compress_to_card(raw_input: str, embedding: list[float] | None = None) -> MemoryCard`
  - Produces `{id (uuid4 hex), fact (compressed), evidence (minimal verbatim quote), timestamp (now_ts()), tags (auto-extracted via regex), embedding, source_tokens (estimate_tokens(raw_input))}`.
  - Conservative: `fact` never invents beyond a verbatim substring; `evidence` is the shortest substring justifying the fact. Single combined fact per input (downstream layers may split).

### isotope_zero/core/store.py  →  class `MemoryStore`
- `MemoryStore(db_path=":memory:", embedder=None)` — `embedder` stored on the instance for callers/router use; the store itself never calls it.
- Write path:
  - `.add(card: MemoryCard) -> None` — persist card + embedding (packed float32 `array('f')` BLOB; `None` embedding ⇒ SQL NULL). Fresh writes default `last_access` to `timestamp`.
  - `.update(card: MemoryCard) -> None` — upsert by id (insert if absent, overwrite if present).
  - `.delete(memory_id: str) -> bool` — `True` if a row was deleted.
- Read path:
  - `.get(memory_id: str) -> MemoryCard | None`
  - `.all() -> list[MemoryCard]` — ordered by timestamp ASC.
  - `.sql_lookup(field: str, value: str) -> list[MemoryCard]` — `field` ∈ {`fact`, `evidence`, `tags`}; fact/evidence use case-insensitive LIKE substring match; `tags` does Python-side JSON-array membership for a single tag. Other fields raise `ValueError`.
  - `.vector_search(query_vec: list[float], k: int = 5) -> list[tuple[MemoryCard, float]]` — pure-Python dot product on L2-normalized vectors; scores clamped to `[0,1]`; NULL embeddings skipped; degenerate query → `[]`.
- Phase 3 access tracking + batch consolidation:
  - `.touch(memory_id: str, at: float | None = None) -> None` — increment `access_count`, set `last_access`; idempotent on missing ids. Called by the router on every surviving hit.
  - `.batch_get(ids: list[str]) -> list[MemoryCard]` — fetch many by id in one round-trip (consolidation helper).
  - `.consolidate_memories(merged_cards: list[MemoryCard], deleted_ids: list[str]) -> int` — apply a consolidation sweep ATOMICALLY (single `BEGIN IMMEDIATE` + `COMMIT`): upsert survivors, delete folded+decayed ids. Returns rows deleted.
- Metrics / lifecycle:
  - `.count() -> int`
  - `.db_size_bytes() -> int` — 0 for `:memory:`; `os.path.getsize` for file-backed.
  - `.close() -> None` — close the held connection; safe to call once.
- Schema: table `memories(id TEXT PK, fact TEXT NOT NULL, evidence TEXT, timestamp REAL, tags TEXT(json), source_tokens INTEGER, embedding BLOB, access_count INTEGER, last_access REAL)`. Index on `fact`, `tags`. File-backed DBs enable WAL + `synchronous=NORMAL`; `:memory:` ignores the PRAGMA. One persistent connection held on the instance (`check_same_thread=False`) guarded by a `threading.Lock`; `access_count`/`last_access` columns auto-added to pre-existing DBs.

### isotope_zero/core/router.py  →  class `QueryRouter`
- `QueryRouter(store: MemoryStore, embedder: EmbeddingEngine)`
- `.query(query: str, token_budget: int = 300) -> QueryResult`
  - Route: SQL path first for explicit state lookups (patterns like "what is/who is/current/my X"); else vector path (embed query once, top-k cosine scan).
  - Budget-aware: gather candidates, sort by marginal value per token, halt when `token_budget` reached OR marginal value drops (diminishing-returns gate; dead-zero vector hits dropped).
  - `tokens_saved_vs_raw` = estimate_tokens of the full raw history minus `tokens_used`.
  - Phase 3: calls `store.touch(h.card.id)` for every hit that survived budget truncation, so the vitality scorer can tell recalled (vital) cards from cold ones.

### isotope_zero/core/consolidation.py  →  class `Consolidator`  (Phase 3 — ADD)
Off-hot-path housekeeping engine: dedup + temporal decay + atomic sweep. Holds NO store lock across planning (works on plain Python `MemoryCard` snapshots from `store.all()` / `store.batch_get()`); only the final `store.consolidate_memories(...)` call takes the lock for one atomic transaction.
- `Consolidator(store: MemoryStore, embedder=None, *, dedup_threshold=0.88, token_overlap_floor=0.70, w_recency=0.7, w_access=0.3, decay_lambda=ln(2)/(30d)≈2.675e-7/s, vitality_floor=0.05, min_age_seconds=3600.0, max_evidence_fragments=3)`
  - `embedder` falls back to `store.embedder` when not passed (may be in fallback mode — exact-fact + token-overlap dedup still work).
- `.vitality(card: MemoryCard, now: float | None = None) -> float`
  - `S = w_recency * exp(-lambda * delta_t) + w_access * ln(1 + access_count)` where `delta_t = now - last_access`. `last_access==0.0` treated as `timestamp`. Higher = more vital.
- `.run() -> ConsolidationReport` — one full synchronous sweep: snapshot all cards, plan dedup (union-find clusters of near-duplicates), plan decay (prune only if vitality < floor AND never-recalled AND past `min_age_seconds` grace period — recalled cards NEVER pruned), apply ONCE via `store.consolidate_memories(...)`, return the report. Recalled cards are never pruned.
- `.dry_run() -> dict[str, Any]` — compute the same plan WITHOUT touching the store; returns JSON-serializable `{proposed_merges, proposed_deletions:{dedup,decay}, summary}`. Used by `izero dry-run-consolidation`.
- `.run_async() -> ConsolidationReport` — `asyncio.to_thread(self.run)`.
- `.start_background_loop(interval_seconds=300.0) -> None` — daemon thread running `.run()` every interval; `.stop(timeout=None) -> None` cancels it.
- Dedup decision order (cheapest/certain first): (0) negation guard — opposite-polarity facts ("uses Mac" vs "does not use Mac") NEVER merge; (1) exact case-insensitive `fact` equality; (2) fact token-overlap ≥ `token_overlap_floor`; (3) cosine of embeddings > `dedup_threshold` (only meaningful with a real embedder; `None` embeddings skip this). Survivor = earliest-timestamp member, with unioned evidence/tags, summed `access_count`, max `last_access`, fact re-embedded when an embedder is available.

### isotope_zero/mcp/server.py  →  MCP server
- `IsotopeZeroServer(db_path: str | None = None, embedder: EmbeddingEngine | None = None)` — wires `MemoryStore` + `QueryRouter` + triage. `db_path` defaults to `$ISOTOPE_ZERO_DB` or `.isotope_zero_cache/isotope_zero.sqlite`; `embedder` defaults to `EmbeddingEngine()`.
- Methods backing the MCP tools:
  - `.add_memory(content: str) -> dict` — triage + compress + embed the fact + persist (DELETE path best-effort deletes by target; UPDATE upserts onto a matched id).
  - `.query_memory(query: str, token_budget: int = 300) -> dict` — route SQL-first/vector-second; returns hits + `tokens_saved_vs_raw`.
  - `.delete_memory(memory_id: str) -> dict`.
  - `.get_metrics() -> dict` — DB size, card count, `embedding_is_real`, cumulative tokens saved vs raw.
  - `.run_consolidation() -> dict` — one `Consolidator(...).run()` sweep; returns merged/decayed/survivors/tokens reclaimed.
- `build_mcp_app(server: IsotopeZeroServer | None = None)` — builds the MCP server app (supports `mcp>=2.0` `MCPServer` and `mcp 1.x` `FastMCP`; raises a clear ImportError if `mcp` is absent). Tools registered: `add_memory(content)`, `query_memory(query, token_budget=300)`, `delete_memory(memory_id)`, `get_metrics()`, `run_consolidation()`.
- `main()` — run the MCP server over stdio. Console script: `izero-mcp`.

### isotope_zero/eval/benchmark.py
- `run_benchmark(db_path=":memory:", embedder=None) -> BenchmarkResult` and `main()` that prints a Markdown table. `render_markdown(r)` formats it.
- Phase 3 scaling: `run_scaling_benchmark(db_path=":memory:", embedder=None, seed_count=1200) -> ScalingResult` (seeds 1000+ redundant cards, runs one consolidation sweep, asserts ≥25% context reduction at 100% recall). `render_scaling_markdown(s)` formats it. `main()` runs both.
- Metrics: **tokens per useful fact (isotope_zero vs raw)**, latency per op (write / SQL read / vector read, mean+median), $ savings vs OpenAI `text-embedding-3-small` (local embeddings cost $0), correctness floor (100% recall on structured fact queries). Console script: `izero-benchmark`.

### isotope_zero/cli/debug.py  →  `izero` CLI  (Phase 4 — ADD)
- `main(argv: list[str] | None = None) -> int` — argparse entry point; returns a process exit code. Console script: `izero`. Calls `configure_logging()` first so DEBUG records surface on stderr.
- Default DB path: explicit `--db PATH`, else `~/.isotope_zero/isotope_zero.db` if it exists, else `:memory:` (empty in-memory store so the CLI is always runnable). A missing explicit `--db` path is reported gracefully (exit 1, no crash).
- Subcommands:
  - `inspect [--db PATH] [--json] [--top N]` — human-readable (or JSON) store report: card count, DB size, embedding mode (REAL ONNX / FALLBACK / none), average embedding dim, top-N lowest-vitality decay candidates (pure vitality ranking, no grace-period filtering), total fact+evidence token footprint.
  - `dry-run-consolidation [--db PATH] [--limit N]` — preview what `Consolidator.run()` WOULD do (proposed merges + decay deletions) WITHOUT committing; prints `Consolidator.dry_run()` as JSON. NOTHING is written to the DB.

### isotope_zero/diagnostics.py  (Phase 4 — ADD)
- `configure_logging() -> str` — configure the `isotope_zero` logger namespace from `ISOTOPE_ZERO_LOG_LEVEL` (alias `IZERO_LOG_LEVEL` accepted, case-insensitive; default `WARNING`). Idempotent — re-calling with the same env value is a no-op; a changed value reconfigures. One stderr handler, `propagate=False` so stdout (used by MCP stdio transport) stays clean. Call once from any entrypoint (`izero`, `izero-mcp`, `python -m isotope_zero...`) before core modules run.
- `get_level() -> str` — current configured level name (or the env default if not yet configured).

## Concurrency note for parallel builders
Each builder works on its OWN file(s) only. The shared files (`types.py`, `tokens.py`, `__init__.py`, `pyproject.toml`) are ALREADY written — do not modify them. Import from them.
