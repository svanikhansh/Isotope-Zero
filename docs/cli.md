# izero-cli

> Tests 134 passed · Python 3.10+ · MIT

`izero-cli` is the isolated terminal inspection + maintenance tool for Isotope
Zero memory engines. It is decoupled from `prototypes/` — it inspects and
maintains engine SQLite databases, but **never modifies core storage schemas or
prototype source**. Its job is to let you see what an agent remembered, debug a
live memory DB, compare two sessions, export cards for fine-tuning, and compact
a fragmented store — all from the terminal, without touching the engine code.

The defining property of the tool is **read-only by default**. Every inspection
connection is opened in URI read-only mode with a second line of defense on top:

```python
sqlite3.connect(f"file:{path}?mode=ro", uri=True)   # VFS opens read-only
conn.execute("PRAGMA query_only=ON")                 # forbids any write statement
```

The ten inspection commands never write to the inspected database. The two
maintenance commands (`import`, `vacuum`) require write access and use a
separate, deliberately narrow read-write connection with transactional safety
(see [§7 — The read-only safety model](#7-the-read-only-safety-model)).

| | |
|---|---|
| Package | `izero-cli` v0.1.0 (`name="izero-cli"`, MIT) |
| Console script | `izero` → `izero_cli.main:main` |
| Requires | Python ≥ 3.10, `rich>=13.0` |
| Source | [`tools/izero_cli/`](../tools/izero_cli/) |
| Commands | 12 (10 read-only inspection, 2 write maintenance) |
| Exit codes | 0 success · 1 error · 2 usage fault |

---

## 1. Install

There are three install channels. The first two are the canonical Python paths;
the third is a universal wrapper for JavaScript-side consumers.

### 1a. pip (editable, from a source checkout)

```bash
pip install -e tools/izero_cli
izero --help
```

### 1b. pip with ONNX extras (real semantic search)

`search` has two modes — **semantic** (ONNX MiniLM cosine) and **lexical**
(TF-IDF, pure stdlib, always available). The ONNX path needs `onnxruntime` +
`tokenizers` + `numpy`; without them `search` silently falls back to lexical and
reports `mode: "lexical"`. Install the extras to get the semantic path:

```bash
pip install -e "tools/izero_cli[onnx]"   # onnxruntime + tokenizers + numpy
```

Other optional extras (all soft dependencies — the data layer degrades to
pure-Python loops when absent):

```bash
pip install -e "tools/izero_cli[numpy]"   # vectorized TF-IDF / cosine math
pip install -e "tools/izero_cli[all]"     # numpy + onnxruntime + tokenizers
```

### 1c. Universal curl | bash installer

A sh-compatible, idempotent installer that provisions a private venv in
`~/.izero` and symlinks `izero` into `~/.local/bin`. It writes **only** to
`~/.izero` and `~/.local/bin` — never to core prototype source.

```bash
curl -fsSL https://raw.githubusercontent.com/<owner>/isotope_zero/main/tools/izero_cli/install.sh | sh
# or, from a source checkout:
sh tools/izero_cli/install.sh
```

> The URL uses the `<owner>/isotope_zero` placeholder. Substitute your GitHub
> owner/org (the published repo lives under `your-org/isotope_zero`).

**Environment overrides** (all optional; configure via env, the script takes no
positional args):

| Variable | Purpose | Default |
|---|---|---|
| `PYTHON` | Python interpreter to use | `python3` |
| `IZERO_ROOT` | Install root (venv lives here) | `~/.izero` |
| `IZERO_VENV` | Explicit venv path | `$IZERO_ROOT/venv` |
| `BIN_DIR` | Where the `izero` symlink goes | `~/.local/bin` |
| `PY_SRC` | Source dir to `pip install` from | dir of `install.sh` |
| `PY_EXTRAS` | Optional extras, e.g. `onnx` | (empty) |
| `GIT_URL` | Git URL to install from instead of a local dir | (empty → falls back to the `izero-cli` PyPI package) |
| `NO_SYMLINK` | `1` to skip creating the symlink | (empty) |
| `DRY_RUN` | `1` to print the plan and exit 0 | (empty) |

Re-running the installer upgrades the package in place. Uninstall with
`rm -rf "$IZERO_ROOT" && rm -f "$BIN_DIR/izero"`.

### 1d. npm wrapper (for JavaScript-side consumers)

A thin npm package wraps the Python CLI: on `postinstall` it provisions a
private Python venv **inside the package directory** and `pip install`s the real
`izero-cli` into it. The `izero` bin then proxies argv, stdio, exit codes, and
signals through to that venv. It does **not** reimplement the CLI in JavaScript.

```bash
npm install -g izero-cli
izero --help
# one-off, no global install:
npx izero-cli --help
```

Requires Python ≥ 3.10 on `PATH` (used only at install time to build the private
venv). npm-side env vars: `IZERO_PYTHON`, `IZERO_PY_SRC`, `IZERO_GIT_URL`,
`IZERO_PY_EXTRAS`, `IZERO_NO_VENV`, `npm_config_izero_skip_postinstall` (CI
opt-out; the bin lazily provisions on first run instead).

See [`tools/izero_cli/npm/README.md`](../tools/izero_cli/npm/README.md) for the
full npm wrapper reference.

---

## 2. The 12 commands

### Inspection (read-only — 10)

These open the inspected DB through `open_ro` (`mode=ro` + `query_only=ON`).
They never write, never set `journal_mode`, never hold a write lock.

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

### Maintenance (write access — 2)

These go through the read-write opener `open_rw` (WAL mode, `synchronous=NORMAL`,
`timeout=30s`, **no** `query_only`). They are the *only* place `izero-cli` ever
opens a write-capable connection.

| Command | Description |
|---|---|
| `izero import <db> <file> [--format jsonl]` | Seed cards from jsonl into a fresh/existing DB. |
| `izero vacuum <db>` | WAL checkpoint (TRUNCATE) + VACUUM; before/after footprint. |

`izero --help` renders a rich command guide (a `rich` panel with a per-command
table). Exit codes: 0 success, 1 error, 2 usage fault.

---

## 3. Per-command reference

Every command below is shown with its exact signature and a realistic example.
No flags are invented — what you see is what `argparse` accepts in `main.py`.

### 3.1 `izero inspect <db>`

Summarize a memory engine DB: total/superseded card counts, WAL + SHM sidecar
sizes, journal mode, quantization breakdown (float32 vs int8-SQ8 vs none),
vector RAM estimate, access recency/frequency (top 5 each), and top tags.

```bash
izero inspect prototypes/synthesis_v1.0/.isotope_zero_cache/mem.db
```

Output shape (the `inspect_db` contract): `db_path`, `exists`, `error`,
`total_cards`, `superseded_count`, `wal`, `quantization`, `vector_ram`,
`access`, `top_tags`, `db_size_bytes`, `db_size_human`.

### 3.2 `izero search <db> "<q>" [--top-k N]`

Search a memory engine and return the top-k ranked cards. Auto-selects the
search path and **honestly reports which one ran** in the `mode` field:

- **semantic** — if `onnxruntime` + `tokenizers` + `numpy` are all importable
  **and** a cached MiniLM model exists at one of the known cache paths
  (`~/.isotope_zero/cache` or the prototype-local cache), embed the query to
  384-dim and run cosine search over stored float32 embeddings
  (L2-normalize, dot product, clamp `[0,1]`).
- **lexical** — always available, pure stdlib: TF-IDF-style bag-of-words over
  `fact`+`evidence` of all non-superseded cards, cosine similarity. Falls back
  to numpy vectorization when present, else a pure-Python loop (identical
  scores either way).

If the semantic path is unavailable for *any* reason (missing deps, missing
model file, embedding dim mismatch, no stored float32 vectors), the lexical path
runs. `mode` always reflects the path that actually executed. Results are sorted
by (score desc, timestamp asc) and truncated to `top_k`. Default `--top-k` is 5.

```bash
izero search mem.db "prefers python over rust" --top-k 10
```

### 3.3 `izero card <db> <id>`

Fetch a single memory card by id, including its decoded vector. Superseded
(audit-trail) cards **are** returned and carry their `superseded_by` target so
the UI can badge them. Vector detail reports `dtype` (`float32` or `int8_sq8`),
`dim`, `norm`, `is_normalized` (norm in `[0.95, 1.05]`), and `q_scale` for SQ8
rows (dequantized as `scale * v`).

```bash
izero card mem.db 7f3a2c1b9d8e4f0a1b2c3d4e5f6a7b8c
```

### 3.4 `izero daemon-status`

Report the health of the Isotope Zero embedding daemon. Takes **no** `db_path`.
Probes the socket at `/tmp/izero.sock` (connect test) and detects
`isotope_zero`/`izero` processes (via `psutil`, falling back to `ps`; excludes
the CLI's own PID and its parent's PID so a shell wrapper whose path contains
"isotope_zero" is not a false positive). `daemon_active` is `True` if the socket
connects **or** a matching process is found.

```bash
izero daemon-status
```

### 3.5 `izero watch <db> [--interval 1.0]`

A polling-based WAL tailer. It opens the DB read-only on a **fresh connection
each poll** (so committed WAL frames are always visible and no lock is held
across polls) and streams newly-created or newly-superseded cards as an agent
writes them in another terminal. Three event kinds: `🟢 NEW` (live card
arrived), `➕ AUDIT` (a card arrived already-folded into the audit trail), and
`🟡 SUPERSEDED → <new_id>` (a live card just got folded). Stop with `Ctrl-C`
(exit 0).

```bash
# Terminal 1: an agent writes memories
# Terminal 2: watch them appear live
izero watch mem.db --interval 0.5
```

### 3.6 `izero doctor <db>`

A read-only health & integrity scorecard across six checks, each returning
`pass` / `warn` / `fail` with a one-line metric and actionable recommendation:

1. **Vector integrity** — NULL / zero-norm / SQ8-inconsistent embeddings.
2. **Fragmentation** — WAL/SHM sidecar sizes, journal mode, freelist/page bloat
   (`bloat > 20%` → warn, `> 40%` → fail; recommends `izero vacuum`). These
   PRAGMAs (`page_count` / `freelist_count` / `journal_mode`) are pure reads
   under `query_only=ON`; doctor never issues `wal_checkpoint`.
3. **FTS5 consistency** — virtual-table row count vs live cards (never creates
   an FTS5 table; "not enabled" is a pass).
4. **SQLite integrity** — `PRAGMA integrity_check` (a pure read under
   `query_only`).
5. **Daemon IPC** — `daemon_status()` probe (missing socket = warn, not fail).
6. **Audit references** — dangling `superseded_by` pointers.

Returns 0 in all cases (it reports, it does not fail the CLI) unless the DB is
missing or unopenable, which renders an error panel and returns 1.

```bash
izero doctor mem.db
```

### 3.7 `izero diff <db1> <db2> [--since TS]`

Compare two memory-engine DBs (both opened read-only) and report the session
delta, categorized by card id:

- **Added** — in db2's active set, not in db1's active set, not a supersede
  replacement target.
- **Superseded/Modified** — a card live in db1 that db2 has folded
  (`superseded_by` went `NULL → target`), grouped as `card-X → card-X2`; plus
  same id active in both but `fact` text differs.
- **Deleted/Forgotten** — in db1's active set and absent from db2 entirely (not
  even as a superseded audit row) — fully purged.

`--since TS` restricts the comparison to cards whose `timestamp >= TS` (the
session window), applied to both sets before deriving active. `TS` is a float
epoch. No differences in the window → a green panel, exit 0.

```bash
izero diff session_start.db session_end.db --since 1722700000.0
```

### 3.8 `izero export <db> --out <f> [--format jsonl|csv|md] [--tag <t>]`

Dump stored memory cards to a portable file. Both active and superseded
(audit-trail) cards are exported; superseded ones are marked via the
`superseded_by` field. `--tag` filters to cards whose parsed JSON tags contain
the given tag (case-insensitive). The DB is opened read-only; the **only** write
is to the user-specified output file.

- `jsonl` (default) — one JSON object per line; fine-tuning-friendly shape
  (`id`, `text`, `evidence`, `timestamp`, `tags`, `metadata`, `vector_norm`,
  `dim`).
- `csv` — header + one row per card; tags pipe-joined; stdlib `csv` quotes
  properly.
- `md` — human-readable report with a header (db path, export time, card count)
  and per-card sections (fact, evidence blockquote, field table).

```bash
izero export mem.db --out cards.jsonl
izero export mem.db --out tagged.csv --format csv --tag python
izero export mem.db --out report.md --format md
```

### 3.9 `izero benchmark <db> [--queries 100]`

Run N sample searches and report cold vs warm p50/p90/p99 latency plus
throughput (QPS). Strictly read-only: each query goes through `search_db`, which
opens its own read-only connection internally. The query set is derived
**deterministically** from the live cards' `fact` text (seeded RNG, seed 42) so
the benchmark is representative of real content and reproducible without a
hard-coded query list. Two passes run back-to-back: a cold pass (first-touch
caches) and a warm pass (identical second pass). A `cold→warm` delta line shows
the median latency improvement. Needs ≥ 2 live cards; fewer renders an amber
"too few cards" panel and exits 0.

```bash
izero benchmark mem.db --queries 200
```

### 3.10 `izero stats <db>`

A three-panel dashboard over the **non-superseded active** cards:

1. **Tag distribution** — JSON tags parsed + counted, top 12 with proportional
   `█` bars.
2. **Age distribution** — cards bucketed by `timestamp` age into `<1h`, `<1d`,
   `<7d`, `<30d`, `>30d`, drawn as a histogram with proportional bars.
3. **Turnover & activity** — total/active/superseded counts, supersede ratio,
   mean `access_count`, never-accessed count, an "updates" proxy (superseded
   count), and a cards-per-day rate over the min..max timestamp span.

```bash
izero stats mem.db
```

### 3.11 `izero import <db> <file> [--format jsonl]`

Seed memory cards from a JSONL file into an existing or freshly-created DB.
**Mutating.** If the target DB does not exist, it is created with the canonical
`memories` schema + the same indexes the prototypes use; if it exists but lacks
a `memories` table, that is a hard error (we won't seed into a stranger schema).

All inserts run inside a **single transaction** (`BEGIN … COMMIT`). On any
exception the transaction is rolled back — a bad mid-file row can never leave
the DB half-written. `insert_card` uses `INSERT OR REPLACE`, so a duplicate `id`
updates the existing card in place (counted as "updated", not "imported").
Individual malformed JSONL rows are skipped + tallied as "skipped/invalid"; they
never abort the whole import. Only setup/open/commit failures abort.

JSONL row contract (one JSON object per line):

| field | required | default | notes |
|---|---|---|---|
| `id` | yes | — | non-empty str |
| `text` *or* `fact` | yes | — | non-empty str; `text` takes precedence |
| `evidence` | no | `None` | str |
| `timestamp` | no | now | int/float → float |
| `tags` | no | `[]` | list |
| `source_tokens` | no | `len(text)//4` | int ≥ 0 |
| `embedding` | no | `None` | list[float], packed float32 via `array('f')` |
| `access_count` | no | `0` | int ≥ 0 |
| `last_access` | no | `timestamp` | int/float → float |
| `superseded_by` | no | `None` | str |

```bash
izero import fresh.db seed.jsonl
izero import existing.db more.jsonl --format jsonl
```

### 3.12 `izero vacuum <db>`

Flush the WAL and reclaim disk space from purged cards. **Mutating.** Snapshots
the before state read-only (db/WAL/SHM sizes, `page_count`, `freelist_count`),
then on a read-write connection runs two **auto-commit** statements
(`wal_checkpoint(TRUNCATE)` and `VACUUM` — these cannot run inside a transaction
or under `query_only`), then snapshots the after state and renders a
before/after delta table. A summary line reports reclaimed MB and WAL frames
checkpointed. An already-compact DB (delta < 1 KB and zero freelist pages)
renders a green "already compact" note. A DB locked by a live writer past the
30 s timeout surfaces `SQLITE_BUSY` as an error panel, exit 1.

```bash
izero vacuum mem.db
```

---

## 4. Data layer API (programmatic use)

`izero_cli.db` exposes a small, stable, contract-returning API for programmatic
use. Every function returns a **plain dict**; **none raise** on missing/corrupt
DBs — they return a contract dict with `exists=False` / `error=<message>`
instead. This totality is load-bearing: the UI layer assumes these functions
never throw.

```python
from izero_cli.db import open_ro, inspect_db, search_db, get_card, daemon_status
```

### `open_ro(db_path) -> sqlite3.Connection`

Open a SQLite database in URI read-only mode + `PRAGMA query_only=ON`. This is
the **only** way `db.py` opens a database handle. It enforces the read-only
safety model at two layers (VFS `mode=ro` + SQLite `query_only`). Raises
`sqlite3.Error` if the file is missing or not a valid SQLite DB — callers that
need a non-raising API should use `inspect_db` / `search_db` / `get_card`, which
wrap this in try/except.

### `inspect_db(db_path) -> dict`

Summary: counts (total/superseded), WAL, quantization status (float32 / int8_sq8
/ mixed / none, plus `cards_float32`, `cards_int8_sq8`, `cards_no_embedding`,
`has_sq8_columns`), vector RAM (`cards_with_embeddings`, `dim`, `float32_bytes`,
`int8_bytes`, `ram_bytes`, `ram_human`), access recency/frequency (top 5 each),
top tags (10), and db size. Optional SQ8 columns (`q_embedding` / `q_scale`) are
detected via `PRAGMA table_info(memories)` **before** any query that references
them, so the data layer works against both the base float32 schema and the
SQ8-extended schema without error.

### `search_db(db_path, query, top_k=5) -> dict`

Auto-selects semantic (ONNX) or lexical (TF-IDF) search and reports which path
ran in `mode`. Returns `db_path`, `exists`, `error`, `query`, `top_k`, `mode`,
`latency_ms` (wall-clock, perf_counter), and `results` — a list of dicts
(`rank`, `score`, `card_id`, `fact`, `evidence`, `tags`, `timestamp`) sorted by
(score desc, timestamp asc), truncated to `top_k`. Scores clamped to `[0,1]`.

### `get_card(db_path, card_id) -> dict`

Single card detail (including superseded audit-trail cards). Returns `db_path`,
`exists`, `error`, `card_id`, `found`, `card` (None if not found), and `vector`
(None if no embedding). `card` carries `id`, `fact`, `evidence`, `timestamp`,
`tags`, `source_tokens`, `access_count`, `last_access`, `superseded_by`.
`vector` carries `dtype`, `dim`, `norm`, `is_normalized`, `q_scale` (for SQ8
rows, dequantized as `scale * v`).

### `daemon_status() -> dict`

Probe the embedding daemon. Takes no `db_path`. Returns `socket_path`
(`/tmp/izero.sock`), `shm_path` (`/izero_shm`), `socket_exists`, `shm_exists`,
`socket_connected`, `socket_error`, `processes` (list of `{pid, rss_mb, name,
cmd}`), and `daemon_active` (`True` if socket connected **or** a matching
process is found). Never raises — `psutil` missing, `ps` failing, or a socket
connect error are all reported as fields.

---

## 5. Exit codes

| Code | Meaning | When |
|---|---|---|
| `0` | success | command completed (even if a check reports `warn`/`fail`, e.g. `doctor`) |
| `1` | error | DB missing/unopenable, query failed, write failed, I/O error |
| `2` | usage fault | `argparse` rejected the invocation (missing required arg, unknown subcommand) |

`izero --help` (and bare `izero`, and `-h`/`--help` anywhere in argv) renders
the rich command guide and exits 0. `izero --version` prints `izero 0.1.0`.

---

## 6. `izero --help`

Bare `izero`, `izero --help`, or `izero -h` renders a `rich` panel titled
"⚡ izero — Isotope Zero memory inspector" with a three-column table
(Command · Description · Usage) covering all 12 commands, plus a dim footer
reminding you that most commands are read-only (`mode=ro` + `query_only=ON`),
that `import` and `vacuum` require write access, and the exit-code legend. The
help path is fast: it lazy-imports nothing from `db` or `ui`, so a broken
`rich`/`sqlite` install cannot poison `--help`.

---

## 7. The read-only safety model

This is the core invariant of `izero-cli` and the reason it is safe to run
against a **live** memory engine while an agent is writing to it. The model is
enforced in `izero_cli/db.py` and `izero_cli/commands/_dbutil.py`.

### Two layers of read-only defense

1. **URI read-only mode** — `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`.
   This asks the SQLite VFS to open the file read-only at the **OS layer**; a
   missing `mode=ro` would allow writes.
2. **`PRAGMA query_only=ON`** — set immediately after connect. This is
   belt-and-suspenders on top of `mode=ro`: SQLite refuses to execute **any**
   write statement on this connection, including write-capable PRAGMAs
   (`journal_mode`, `synchronous`, `CREATE TABLE`). If the SQLite build rejects
   `query_only` (very rare), `open_ro` closes the connection and **re-raises**
   so the caller surfaces the safety failure rather than silently weakening it.

### `open_ro` is the only read path

Every read-only command (`inspect`, `search`, `card`, `daemon-status`, `watch`,
`doctor`, `diff`, `export`, `benchmark`, `stats`) goes through `open_ro`. The
contract-returning functions (`inspect_db`, `search_db`, `get_card`,
`daemon_status`) wrap `open_ro` in `_safe_open`, which returns
`(conn, None)` or `(None, error_message)` — so a missing/corrupt DB yields a
contract dict, never a traceback.

### `open_rw` only for `import` and `vacuum`

The read-write opener `open_rw` is the deliberate counterpart to `open_ro`. It
opens a `file:` URI **without** `mode=ro`, sets `journal_mode=WAL` +
`synchronous=NORMAL` + `timeout=30s` (bounded BUSY wait), and does **not** set
`query_only` — writes are the point. It is used **solely** by `import` and
`vacuum`, both of which require write access by spec.

### Transactional safety

- `import` wraps all inserts in a single `BEGIN … COMMIT`. Any exception →
  `rollback` + error panel, exit 1. A bad mid-file row can never leave the DB
  half-written.
- `vacuum` runs `wal_checkpoint(TRUNCATE)` and `VACUUM` as **auto-commit**
  statements (they cannot run inside a transaction or under `query_only`). Any
  error → error panel, exit 1. `SQLITE_BUSY` (a live writer held the lock past
  the 30 s timeout) surfaces as `sqlite3.OperationalError`.

### What the CLI never does

- Never modifies core storage schemas or prototype source.
- Never sets a write-capable PRAGMA on a read-only connection.
- Never holds a connection (or a lock) across `watch` polls — each iteration
  opens and closes its own read-only connection, so a transient `SQLITE_BUSY`
  from a live writer skips that poll with a warning rather than killing the
  stream.
- Never creates an FTS5 table (`doctor` only *reports* FTS5 drift).

---

## 8. See also

- [`tools/izero_cli/README.md`](../tools/izero_cli/README.md) — the CLI's own
  README (command list, data layer, fixtures).
- [`tools/izero_cli/install.sh`](../tools/izero_cli/install.sh) — the universal
  installer source.
- [`tools/izero_cli/npm/README.md`](../tools/izero_cli/npm/README.md) — the npm
  wrapper reference.
- [`README.md`](../README.md) — the Isotope Zero project root.
- The unified client API (`IsotopeZero`) and the 8-phase research evolution are
  documented in the engine docs under `prototypes/synthesis_v1.0/`.
