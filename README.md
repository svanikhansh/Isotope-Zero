<div align="center">

# ⚛️ Isotope Zero

**A zero-overhead, sub-millisecond agent memory engine.**

Token compression + local ONNX embeddings + hybrid SQL/vector routing —
so your agent remembers without burning tokens on expensive API loops.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://pypi.org/project/isotope-zero/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: 129 passed](https://img.shields.io/badge/tests-129%20passed-brightgreen)](tests/)
[![Read latency: sub-ms](https://img.shields.io/badge/read%20latency-sub--millisecond-orange)](#key-benchmarks)
[![Local-first](https://img.shields.io/badge/local%20inference-ONNX%20%E2%80%A2%20SQLite%20%E2%80%A2%20MCP-8A2BE2)]()

</div>

---

## Why not API-first?

Most agent-memory systems (Mem0, Zep, Letta) trade **tokens and latency** for flexibility: every read and write fans out to remote embedding APIs and multi-pass LLM loops. That's a per-interaction API bill and multi-second round-trips — before your agent has even produced a useful token.

Isotope Zero flips the priority **cost and latency first**:

- **$0 inference** — quantized ONNX embeddings run locally. No `text-embedding-3-small` bill, ever.
- **Sub-millisecond reads** — direct SQL lookup for explicit state, vector similarity only when the query is genuinely semantic.
- **Token context compression** — every input is collapsed at write-time into a minimal `{fact, evidence}` Memory Card, never the raw history.
- **Budget-aware retrieval** — queries carry a `token_budget`; hits are ordered by marginal value per token and retrieval *halts* when the budget is reached.
- **MCP-native** — speaks the Model Context Protocol out of the box, so Cursor, Claude Code, Windsurf, and Claude Desktop can all share one local SQLite store.

---

## Key Benchmarks

Measured on real ONNX MiniLM (`dim=384`), pure-stdlib SQLite, macOS arm64 — no cherry-picked runs, no remote APIs.

| Metric | Value | Context |
|---|---|---|
| **Needle recall** | **100.0%** | 100 queries, 500 near-miss distractors |
| **Context compression** | **99.3%** | 1,200 cards → 9 (35,520 → 261 tokens) |
| **Vector search p99** | **0.43 ms** | 10,000 cards, cached numpy matrix |
| **SQL exact-lookup p99** | **~0.07 ms** | indexed `NOCASE` fast path |
| **Hybrid lookup p99** | **0.72–0.83 ms** | 10,000 cards, substring route |
| **RSS footprint** | **195 MB** | 1,000-card seeding, chunked ONNX streaming |
| **Negation correctness** | **0 wrong merges** | 100 adversarial polarity pairs |
| **Concurrency** | **0 errors, 0 corruption** | 25 workers × 1,000 cycles, WAL + `busy_timeout` |

> Honest note: at **10,000 cards** the process holds real 384-dim embeddings for every card, so RSS is **~370–410 MB** — the ONNX engine itself is ~135 MB. The 195 MB figure is the verified 1,000-card bar. See [`docs` of the stress harness](isotope_zero/eval/adversarial.py) for the full claim-vs-measured matrix.

---

## Quickstart

```bash
pip install isotope-zero
```

Inspect an empty in-memory store:

```bash
izero inspect --db :memory:
```

Open your on-disk store and **preview** what a consolidation sweep would do — nothing is committed:

```bash
izero dry-run-consolidation --db ~/.isotope_zero/isotope_zero.db
```

Render the Phase 0 + scaling benchmark (tokens-per-fact, latency, cost savings):

```bash
izero-benchmark
```

---

## MCP Server Setup

Isotope Zero ships an MCP stdio server (`izero-mcp`) exposing five tools — `add_memory`, `query_memory`, `delete_memory`, `get_metrics`, `run_consolidation`. Point your agent at it:

### Cursor

`.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "isotope-zero": {
      "command": "izero-mcp",
      "args": []
    }
  }
}
```

### Claude Code

`.mcp.json` (project) — or run `claude mcp add isotope-zero -- izero-mcp` once for a user-global scope:

```json
{
  "mcpServers": {
    "isotope-zero": {
      "command": "izero-mcp",
      "args": []
    }
  }
}
```

### Windsurf

`.codeium/windsurf_mcp_config.json`:

```json
{
  "mcpServers": {
    "isotope-zero": {
      "command": "izero-mcp",
      "args": []
    }
  }
}
```

All three use stdio and share one default SQLite store (`./.isotope_zero_cache/isotope_zero.sqlite`). Override the database with the `ISOTOPE_ZERO_DB` environment variable:

```json
{
  "mcpServers": {
    "isotope-zero": {
      "command": "izero-mcp",
      "args": [],
      "env": { "ISOTOPE_ZERO_DB": "~/.isotope_zero/isotope_zero.db" }
    }
  }
}
```

---

## Python API

A complete add → query → budget-aware-retrieve loop in ten lines:

```python
import time
from isotope_zero.core.store import MemoryStore
from isotope_zero.core.router import QueryRouter
from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.types import MemoryCard

engine = EmbeddingEngine()
store = MemoryStore(":memory:", embedder=engine)

def remember(text: str) -> None:
    store.add(MemoryCard(
        id=f"c{int(time.time()*1000)}", fact=text, evidence="",
        timestamp=time.time(), embedding=engine.embed_text(text),
    ))

remember("The user prefers Rust over Go")
remember("The user's favourite editor is Neovim")

result = QueryRouter(store, engine).query(
    "which language does the user prefer?", token_budget=300)
print(result.route_used, f"{result.latency_ms:.2f}ms", result.tokens_used, "tokens")
for hit in result.hits:
    print(f"  {hit.score:.3f}  {hit.card.fact}")
```

```text
vector 1.72ms 19 tokens
  0.281  The user's favourite editor is Neovim
  0.263  The user prefers Rust over Go
```

The MCP layer wraps the same flow — triage + compress + embed + store — so you never hand-build cards by hand:

```python
from isotope_zero.mcp.server import IsotopeZeroServer

server = IsotopeZeroServer(db_path=":memory:")
server.add_memory("My name is Alice Chen.")
print(server.query_memory("What is my name?", token_budget=300))
```

---

## How it works

```
raw input ──► [write path]  triage (ADD/UPDATE/DELETE)
                       │        └─► compress ─► {fact, evidence} Memory Card
                       │        └─► embed (local ONNX, $0)
                       ▼
                   SQLite (WAL)
                       ▲
query ──► [read path]  hybrid router
                 ├─ structured? ──► SQL exact/substring lookup (sub-ms)
                 └─ semantic?   ──► vector top-k + lexical re-rank
                       └─► budget-aware truncation ──► ordered hits
```

- **Write path** — a local heuristic classifier decides *add / update / delete*; the compressor keeps only the minimal `{fact, evidence}`; a chunked ONNX embedder turns it into a 384-dim vector without spiking RSS.
- **Read path** — `QueryRouter` tries SQL first for explicit state (`"what port does svc-0 bind?"`), falls back to vector similarity for fuzzy queries, then applies a lexical/numeric boost and truncates to the caller's `token_budget`.
- **Consolidation** — periodic sweeps deduplicate near-identical cards (newest wins, superseded ones are audited, not deleted), prune decayed cards, and shrink context by up to **99.3%** while holding the correctness floor at 100%.
- **Concurrency** — a single WAL SQLite store with `busy_timeout`, a bounded consolidation row-pool, and a heartbeat makes 25-worker contention error-free.

## Layout

```
isotope_zero/
  core/        # store.py (SQLite) · triage.py (classifier + compressor)
               # router.py (hybrid + budget) · consolidation.py (sweep)
  embeddings/  # onnx_embed.py (local ONNX, chunked streaming)
  mcp/         # server.py (MCP tools — IsotopeZeroServer)
  eval/        # benchmark.py (cost harness) · adversarial.py (stress suite)
  cli/         # debug.py (izero inspect / dry-run-consolidation)
  types.py     # MemoryCard / QueryHit / QueryResult contract
  tokens.py    # token estimator
  diagnostics.py  # logging from ISOTOPE_ZERO_LOG_LEVEL (IZERO_LOG_LEVEL alias)
```

## Configuration

| Setting | Default | Notes |
|---|---|---|
| Default SQLite DB | `~/.isotope_zero/isotope_zero.db` | used by `izero`/`izero-mcp` when no path is given |
| ONNX model cache | `.isotope_zero_cache` | local model download/cache directory |
| Log level | `ISOTOPE_ZERO_LOG_LEVEL` | `IZERO_LOG_LEVEL` accepted as a shorter alias |
| Log namespace | `isotope_zero` | e.g. `logging.getLogger("isotope_zero.store")` |

## Development

```bash
git clone https://github.com/svanikhansh/Isotope-Zero.git && cd Isotope-Zero
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/            # 129 passed, 5 stress-gated skips
IZERO_STRESS=1 pytest tests/test_extreme_stress.py   # gated extreme-stress suite
```

CI runs `pytest tests/` on Python 3.10, 3.11, and 3.12 for every push to `main` and every pull request.

## License

[MIT](LICENSE) © 2026 Svanik Kolli
