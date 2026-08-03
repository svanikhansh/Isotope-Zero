# Python v0.1 Prototype (frozen)

This folder is the frozen **v0.1.0 pure-Python reference implementation** of
Isotope Zero. It is fully functional and kept as a runnable baseline — a
**129-test green** milestone — while the repository evolves toward
**Phase 6 (Hybrid Rust / Native Core)**.

> Everything here was cut from the repository root via `git mv`, so the full
> commit history of `isotope_zero/`, `tests/`, and `pyproject.toml` is
> preserved and the prototype can be run or inspected at any time.

## What's here

- `isotope_zero/` — the package: SQLite store, hybrid SQL/vector router,
  local ONNX embeddings, consolidation, MCP server, CLI.
- `tests/` — the full suite (**129 passed / 5 skipped**, the 5 gated behind
  `IZERO_STRESS=1`).
- `pyproject.toml` — packaging for the v0.1.0 wheel/sdist.
- `CONTRACT.md` — the project contract preserved alongside the prototype.

## Run it

```bash
cd prototypes/python_v0.1
python -m pip install -e .        # or: pip install -e ".[dev]"
izero inspect --db :memory:
python -m pytest tests/           # 129 passed / 5 skipped
IZERO_STRESS=1 python -m pytest tests/test_extreme_stress.py   # gated suite
```

### Quick results captured at freeze

- Needle recall **100%** @ 500 distractors
- Context compression **99.3%** (1,200 → 9 cards)
- Vector search p99 **0.43 ms** @ 10k cards
- SQL exact-lookup p99 **~0.07 ms**; hybrid p99 **0.72–0.83 ms**
- RSS **195 MB** @ 1k cards (10k ≈ 370–410 MB — the honest structural floor)
- 0 wrong merges on 100 adversarial polarity pairs; 0 errors / 0 corruption
  under 25-worker concurrency

The upstream documentation that shipped with v0.1.0 lives in the repository
root (`README.md`, `LICENSE`).