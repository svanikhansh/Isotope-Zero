# izero-cli (npm wrapper)

Universal npm distribution wrapper for **izero-cli** — the read-only terminal
inspection + maintenance tool for Isotope Zero memory engines.

This package does **not** reimplement the CLI in JavaScript. On install, the
`postinstall` hook provisions a private Python virtual environment inside this
package and `pip install`s the real `izero-cli` (a Python package) into it. The
`izero` bin then proxies through to that venv, forwarding argv, stdio, exit
codes, and signals.

## Install

```bash
npm install -g izero-cli
izero --help
```

> Requires **Python ≥ 3.10** on your `PATH` (used only at install time to build
> the private venv; never invoked on every run). If `python3` isn't found,
> install Python 3.10+ or set `IZERO_PYTHON=/path/to/python3` and re-run
> `npm rebuild izero-cli`.

## One-off / no global install

```bash
npx izero-cli --help
```

## What it does

1. `postinstall` finds Python ≥ 3.10 (or `IZERO_PYTHON`).
2. Creates `.venv/` inside this package directory.
3. `pip install`s izero-cli from (in priority order):
   - `IZERO_PY_SRC` — explicit source dir (has `pyproject.toml`), or
   - the bundled peer source (`../` when shipped inside a repo checkout), or
   - `IZERO_GIT_URL` — a git URL, or
   - the published `izero-cli` package on PyPI.
4. The `izero` bin spawns `.venv/bin/izero` with your args.

## Environment

| Variable | Purpose |
|---|---|
| `IZERO_PYTHON` | Python interpreter to use (default: `python3`, then `python`). |
| `IZERO_PY_SRC` | Absolute path to the izero-cli source dir. |
| `IZERO_GIT_URL` | Git URL to pip-install from instead of PyPI. |
| `IZERO_PY_EXTRAS` | Optional extras, e.g. `onnx` for ONNX semantic search. |
| `IZERO_NO_VENV` | `1` = assume the venv already exists; skip provisioning. |
| `npm_config_izero_skip_postinstall` | `1` = skip postinstall entirely (CI opt-out). The bin will lazily provision on first run instead. |

## Safety / scope

- Only writes inside **this package's own directory** (`.venv/`).
- Never touches `$HOME`, core prototype source, or any other project files.
- Idempotent: re-running upgrades the installed CLI.

## Uninstall

```bash
npm uninstall -g izero-cli   # removes the package + its private .venv
```

See the main CLI README at `../README.md` for the full command reference.
