# isotope-zero (npm)

> Global `izero` CLI launcher for [isotope-zero](https://pypi.org/project/isotope-zero/) — a sub-millisecond, local-first cognitive memory layer for AI agents.

This npm package is a **thin launcher**. It manages an isolated Python environment with `isotope-zero` installed, then delegates every command straight to the Python `izero` CLI. All the real logic — hybrid retrieval, Ebbinghaus decay, knowledge compaction — lives in the Python package. The npm wrapper just makes it installable from npm for JS-first environments and CI.

```bash
npm install -g isotope-zero
izero inspect            # human-readable store report
izero dry-run-consolidation   # preview a consolidation sweep (no commits)
izero --help
```

## Why an npm package?

isotope-zero ships to [PyPI](https://pypi.org/project/isotope-zero/) as a Python package with a Rust native extension. The npm package lets you install and run the same `izero` CLI via npm, without touching your Python setup:

```bash
npm install -g isotope-zero   # ← one command, works in any shell
izero --help
```

The launcher guarantees the Python runtime exists (see [How it works](#how-it-works)), so `npm i -g` is the only step.

## Requirements

- **Node.js ≥ 18** (to run the launcher).
- **Python 3.10+** must be available on your system — isotope-zero is a Python/Rust package. The launcher will locate or create a Python environment automatically; it does not bundle a Python interpreter.

If you have no Python at all, install one first:

| OS | Command |
|---|---|
| macOS | `brew install python@3.12` |
| Ubuntu/Debian | `apt install python3` |
| Windows | [python.org](https://python.org) installer, or `winget install Python.Python.3.12` |

## How it works

On first run, the launcher resolves a Python interpreter that can `import isotope_zero`, in this order:

1. **`uv`** (preferred) — if [uv](https://astral.sh/uv) is on your PATH, the launcher builds a dedicated isolated env at `~/.izero/venv` with `isotope-zero` pinned to the npm package version. Reproducible, isolated from your system Python.
2. **Bootstrap `uv`** — if `uv` isn't installed, the launcher downloads the standalone `uv` binary into `~/.izero/bin`, then proceeds as (1).
3. **System Python + `pip install --user`** — if `uv` can't be bootstrapped (offline, sandboxed), the launcher falls back to your system `python3` and installs `isotope-zero` with `--user`. May fail on Homebrew/system Pythons under PEP 668 (`externally-managed-environment`) — in that case, install `uv` and re-run.
4. **No Python** — the launcher prints a clear, actionable error telling you how to install Python.

Every subsequent run reuses the cached env (a version stamp at `~/.izero/.installed-version`), so warm starts are instant.

### Version lockstep

The npm package version always matches the pinned `isotope-zero` PyPI version: `npm i -g isotope-zero@1.0.0` installs exactly `isotope-zero==1.0.0` in the venv. The two never drift. Run `izero doctor` to verify.

## Commands

All commands are delegated to the Python `izero` CLI (`python -m isotope_zero.cli.debug`), so the behavior is identical to running the Python console script directly.

| Command | Description |
|---|---|
| `izero inspect` | Human-readable store report: card count, size, decay state, token usage. |
| `izero dry-run-consolidation` | Preview a consolidation sweep — nothing is committed to the DB. |
| `izero --help` | Full Python CLI help. |

### Launcher-private commands

These are prefixed `--izero-` so they never collide with the Python CLI's arguments:

| Command | Description |
|---|---|
| `izero doctor` | Diagnostics: resolved Python path, `uv` status, venv location, `isotope-zero` version, and a self-test import. Use this when `izero` won't start. |
| `izero --izero-version` | Print the npm launcher version (`-V` alias). |
| `izero --izero-self-test` | Resolve the env and verify `import isotope_zero` works; exits 0/1. |

## Troubleshooting

```bash
izero doctor   # always run this first — it pinpoints the failure
```

| Symptom | Fix |
|---|---|
| `isotope-zero requires Python 3.10+` | Install Python (see Requirements). |
| `PEP 668 'externally-managed'` | Install [uv](https://astral.sh/uv) (`curl -LsSf https://astral.sh/uv/install.sh \| sh` on macOS/Linux) and re-run — the launcher will use an isolated env. |
| Version mismatch in `doctor` | The venv has a different `isotope-zero` than the npm pin. Delete `~/.izero` and re-run; the launcher rebuilds. |
| Slow first run | Expected — the first run builds the venv and installs the wheel (~10-30s with `uv`). Subsequent runs are instant. |

## License

MIT © Svanik Kolli
