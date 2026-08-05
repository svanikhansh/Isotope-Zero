# izero-cli

Isolated terminal inspection + maintenance tool for Isotope Zero memory engines.

`izero-cli` opens Isotope Zero SQLite memory-engine databases in URI read-only
mode (`file:<path>?mode=ro`, `uri=True`) with `PRAGMA query_only=ON` as a second
line of defense. The ten inspection commands never write to the inspected
database; the two maintenance commands (`import`, `vacuum`) require write access
and use explicit read-write connections with transactional safety.

It is decoupled from `prototypes/` — it inspects/maintains engine DBs but never
modifies core storage schemas or prototype source.

## Install

```bash
pip install -e tools/izero_cli
```

## Optional extras

- `numpy` / `vector` — vectorized TF-IDF + semantic cosine math (soft dep; the
  data layer degrades to pure-Python loops when absent).
- `onnx` — real ONNX semantic search via `onnxruntime` + `tokenizers`. Without
  it, `search` falls back to the always-available lexical TF-IDF path.

```bash
pip install -e "tools/izero_cli[onnx]"   # full semantic search
```

## Commands (12)

**Inspection (read-only):**

| Command | Description |
|---|---|
| `izero inspect <db>` | Summarize a memory engine DB (cards, schema, size). |
| `izero search <db> "<q>" [--top-k N]` | Semantic/lexical search over a memory engine. |
| `izero card <db> <id>` | Fetch a single memory card by id. |
| `izero daemon-status` | Report the Isotope Zero daemon health. |
| `izero watch <db> [--interval 1.0]` | Live-tail new/superseded cards as they are written. |
| `izero doctor <db>` | Health & integrity scorecard (vectors, WAL, FTS5, IPC). |
| `izero diff <db1> <db2> [--since TS]` | Session comparison: added/superseded/deleted deltas. |
| `izero export <db> --out <f> [--format jsonl\|csv\|md] [--tag <t>]` | Dump cards to a portable format. |
| `izero benchmark <db> [--queries 100]` | Search latency p50/p90/p99 + cold/warm QPS. |
| `izero stats <db>` | Tag/age/turnover memory analytics. |

**Maintenance (write access):**

| Command | Description |
|---|---|
| `izero import <db> <file> [--format jsonl]` | Seed cards from jsonl into a fresh/existing DB. |
| `izero vacuum <db>` | WAL checkpoint (TRUNCATE) + VACUUM; before/after footprint. |

`izero --help` renders a rich command guide. Exit codes: 0 success, 1 error,
2 usage fault.

## Data layer

`izero_cli.db` exposes a small, stable, contract-returning API:

- `open_ro(db_path)` — read-only connection (the only way db.py opens a DB).
- `inspect_db(db_path)` — summary: counts, WAL, quantization, vector RAM,
  access recency/frequency, top tags.
- `search_db(db_path, query, top_k=5)` — auto-selects semantic (ONNX) or
  lexical (TF-IDF) search; reports which path ran in `mode`.
- `get_card(db_path, card_id)` — single card detail incl. decoded vector.
- `daemon_status()` — probe the (currently hypothetical) `/tmp/izero.sock`
  embedding daemon and detect isotope_zero processes.

All data-layer functions return plain dicts; none raise on missing/corrupt DBs.

## Command modules

The eight newer commands live under `izero_cli.commands/`, each in its own
module exporting `cmd(args) -> int`. Shared helpers:

- `izero_cli.commands._dbutil` — `open_ro` / `open_rw` / `create_fresh_db` /
  `insert_card` + vector decoders and size helpers. `open_rw` is the *only*
  write-capable opener and is used solely by `import` and `vacuum`.
- `izero_cli.commands._uiutil` — the shared rich palette + badge/table helpers
  so all commands render as one system.

All read-only commands use `open_ro` (`mode=ro` + `query_only=ON`); the two
mutating commands use `open_rw` (WAL mode, `timeout=30s`, transactional).

## Fixtures

`sample_seed.py` builds verification DBs: `seed_sample_db()` (7-card mixed),
`seed_large_db(n=120)` (benchmark/stats), `seed_diff_pair()` (added/
superseded/deleted delta), `seed_doctor_db()` (zero-norm + null-vector
anomalies). Run `python sample_seed.py --kind {sample,large,diff,doctor}`.

