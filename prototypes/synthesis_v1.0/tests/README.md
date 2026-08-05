# isotope_zero — Test Suite

**330 tests collected · 325 passed · 5 skipped** (verified at the v1.0.0 release, with
`onnxruntime` installed and the cached ONNX weights present, so real semantic embeddings
were active). The 5 skips are all the `IZERO_STRESS` opt-in stress tests
(`IZERO_STRESS=1` re-enables them).

## Running the suite

```bash
# From prototypes/synthesis_v1.0/, with the test extra installed:
pip install -e ".[test]"
PYTHONPATH=. pytest tests/ -q

# The full audit suite (core + mem0 adaptations + adapters + fuzz + stress):
PYTHONPATH=. pytest tests/ tests/test_mem0_adaptations/ -q
```

## Test layout

| Path | What it covers |
|---|---|
| `tests/test_storage.py` | Core store CRUD, embedding blob (de)serialisation |
| `tests/test_scope.py` | Multi-tier scoping (user/agent/run isolation) |
| `tests/test_hybrid_search.py` | Late-fusion RRF (FTS5 + vector + entity), perf SLA |
| `tests/test_consolidation.py` | Near-duplicate cosine merge + decay prune |
| `tests/test_needle_recall.py` | Recall@5 floor at 10k cards (semantic accuracy) |
| `tests/test_extreme_stress.py` | 10k-card ingestion + retrieval under budget |
| `tests/test_concurrency_stress.py` | Threaded read/write serialization on the lock |
| `tests/test_fuzzing.py` | `hypothesis` property-based invariants |
| `tests/test_daemon.py` | Shared-memory embedding daemon lifecycle |
| `tests/test_mem0_adaptations/` | The 8 mem0-ported subsystems (dedup, TTL, history, scoping, async, hybrid, prompts, graph) |
| `adapters/tests/` | LangChain / LlamaIndex / CrewAI / AutoGen adapters |

## Environmental test dependency

The store is **zero-dependency at runtime** (stdlib + optional numpy). But the
*test suite* verifies real semantic behavior, which requires real embeddings.
~10 tests need the optional **`onnxruntime`** + **`tokenizers`** packages so
the embedder runs in real ONNX mode (`is_real=True`):

- `test_daemon.py` (×4) — daemon `is_real` assertions
- `test_needle_recall.py` (×4) — recall@5 ≥ 0.90 floors
- `test_hybrid_search.py::TestHybridPerf` — p99 < 5ms at 10k
- `test_consolidation.py::test_near_duplicate_cosine_merges` — cosine threshold

**Without** these packages the embedder falls back to deterministic
feature-hash pseudo-embeddings (non-semantic). The semantic-accuracy tests
can't pass then: the semantic-regression tests **skip** (`skipif` on
`EmbeddingEngine().is_real`), while the daemon-`is_real` and recall-floor
tests **fail** outright because they assert real-embedding behavior. Install
the extras to run the full green suite.

### Recommended: install the `[test]` extra

```bash
pip install -e ".[test]"          # pytest, pytest-asyncio, hypothesis, psutil
pip install onnxruntime tokenizers   # the semantic engine (+ cached ONNX weights)
```

The ONNX model weights (`Xenova_all-MiniLM-L6-v2.onnx`) are cached under
`.isotope_zero_cache/` and auto-download on first run. Once installed, all 494
tests pass with 0 failures.

### Mark-based selection

Custom marks are registered in `pyproject.toml` (`[tool.pytest.ini_options].markers`):

```bash
pytest -m "not slow"           # skip >5s tests
pytest -m "perf"               # only the SLA benchmarks
pytest -m "onnx"               # only the tests requiring real embeddings
```

## Honest note on the async config

`asyncio_mode = "auto"` is set in `pyproject.toml`. When `pytest-asyncio` is
absent, pytest emits a (now-silenced) `PytestConfigWarning`. Async tests use
`anyio`'s backend, which is a runtime dep — so the mode is a no-op without the
plugin and the warning is benign. Installing the `[test]` extra resolves it
cleanly.
