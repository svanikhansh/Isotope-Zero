"""isotope_zero debug CLI — inspect the store and dry-run consolidation.

A lightweight, pure-stdlib CLI with two subcommands:

    inspect
        Human-readable store report: card count, DB size, embedding mode,
        average embedding dimension, top-N lowest-vitality decay
        candidates, and the total token footprint. ``--json`` emits the same
        data as a JSON object.

    dry-run-consolidation
        Preview what a ``Consolidator.run()`` sweep WOULD do — proposed
        merges and decay deletions — WITHOUT committing anything to the DB.
        The plan dict returned by ``Consolidator.dry_run()`` is already
        JSON-serializable, so we just pretty-print it.

Design notes:
    - Pure stdlib (argparse, json, os, sys) + project modules. No new deps.
    - Does NOT require a real ONNX embedder. ``inspect`` reports the
      embedding mode (REAL ONNX / FALLBACK / none) from ``store.embedder``;
      ``dry-run-consolidation`` passes ``store.embedder`` to the
      Consolidator, which falls back to exact-fact + token-overlap dedup
      when no real embeddings are present.
    - Default DB path: ``~/.isotope_zero/isotope_zero.db`` if it exists, else
      ``:memory:`` (an empty in-memory store). ``--db PATH`` overrides.
    - A missing explicit ``--db PATH`` is reported gracefully (no crash).

Run:
    .venv/bin/python -m isotope_zero.cli.debug inspect --db :memory:
    .venv/bin/python -m isotope_zero.cli.debug dry-run-consolidation --db :memory:
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any

from isotope_zero.core.consolidation import Consolidator
from isotope_zero.core.store import MemoryStore
from isotope_zero.diagnostics import configure_logging
from isotope_zero.tokens import estimate_tokens
from isotope_zero.types import MemoryCard, now_ts

# Default on-disk store location. Used only when no ``--db`` is given AND the
# file exists; otherwise we fall back to an empty ``:memory:`` store so the
# CLI is always runnable with zero setup.
_DEFAULT_DB: str = os.path.expanduser("~/.isotope_zero/isotope_zero.db")

# Seconds per day, for the age-in-days display.
_SECS_PER_DAY: float = 86400.0


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _resolve_db_path(db_path: str | None) -> str:
    """Pick the DB path: explicit override, else default-if-present else :memory:."""
    if db_path is not None:
        return db_path
    return _DEFAULT_DB if os.path.exists(_DEFAULT_DB) else ":memory:"


def _open_store(db_path: str) -> MemoryStore | None:
    """Open a MemoryStore at ``db_path``.

    Returns None for a non-``:memory:`` path that does not exist on disk, so
    the caller can report it gracefully instead of letting sqlite3 silently
    create an empty file (which would surprise a user who mistyped a path).
    For ``:memory:`` the store is always openable (and empty).
    """
    if db_path != ":memory:" and not os.path.exists(db_path):
        return None
    return MemoryStore(db_path)


def _human_bytes(n: int) -> str:
    """Bytes -> human-readable string (B / KB / MB)."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _embedding_mode(store: MemoryStore) -> str:
    """REAL ONNX / FALLBACK / none — from the store's attached embedder."""
    emb = getattr(store, "embedder", None)
    if emb is None:
        return "none (no embedder attached)"
    if getattr(emb, "is_real", False):
        return "REAL ONNX"
    return "FALLBACK"


def _trunc(s: str, n: int) -> str:
    """Truncate a string to n chars (display-only; JSON keeps full values)."""
    s = str(s)
    return s if len(s) <= n else s[:n]


# ---------------------------------------------------------------------- #
# inspect
# ---------------------------------------------------------------------- #
def _cmd_inspect(store: MemoryStore, top: int, as_json: bool) -> int:
    """Print a human-readable (or JSON) store report."""
    cards = store.all()
    total = len(cards)
    size_bytes = store.db_size_bytes()
    mode = _embedding_mode(store)

    # Average embedding dimension across cards that carry an embedding.
    with_emb = [c for c in cards if c.embedding]
    if with_emb:
        avg_dim = sum(len(c.embedding) for c in with_emb) / len(with_emb)
        emb_json: dict[str, Any] = {
            "avg_dim": round(avg_dim, 2),
            "cards_with_embeddings": len(with_emb),
        }
        emb_line = (
            f"avg embedding dim: {avg_dim:.1f} "
            f"(across {len(with_emb)} cards with embeddings)"
        )
    else:
        emb_json = {
            "avg_dim": None,
            "cards_with_embeddings": 0,
            "note": "no embeddings",
        }
        emb_line = "no embeddings"

    # Top-N decay candidates: the cards with the LOWEST vitality score.
    # Pure vitality ranking (no grace-period / never-recalled filtering) —
    # this is a diagnostic view, not the actual prune decision.
    cons = Consolidator(store)
    now = now_ts()
    scored = [(cons.vitality(c, now=now), c) for c in cards]
    scored.sort(key=lambda item: item[0])
    candidates = scored[:top] if top > 0 else scored
    decay_rows: list[dict[str, Any]] = [
        {
            "id": c.id,
            "fact": c.fact,
            "vitality": round(v, 4),
            "access_count": c.access_count,
            "age_days": round(max(0.0, now - c.timestamp) / _SECS_PER_DAY, 1),
        }
        for v, c in candidates
    ]

    # Total token footprint: fact + evidence tokens across all cards.
    tokens = sum(estimate_tokens(c.fact) + estimate_tokens(c.evidence) for c in cards)

    if as_json:
        payload = {
            "store": {
                "db_path": store.db_path,
                "total_cards": total,
                "db_size_bytes": size_bytes,
                "db_size_human": _human_bytes(size_bytes),
                "embedding_mode": mode,
            },
            "embeddings": emb_json,
            "decay_candidates": decay_rows,
            "token_footprint": tokens,
        }
        print(json.dumps(payload, indent=2))
        return 0

    # Text report — clean aligned columns via f-string padding.
    print("=== isotope_zero inspect ===")
    print(f"store path:        {store.db_path}")
    print(f"total cards:       {total}")
    print(f"db size:           {size_bytes} bytes ({_human_bytes(size_bytes)})")
    print(f"embedding mode:    {mode}")
    print()
    print(emb_line)
    print()
    print(f"top {len(decay_rows)} decay candidates (lowest vitality):")
    if decay_rows:
        print(
            f"  {'id':<12}  {'fact':<60}  {'vital':>6}  "
            f"{'acc':>4}  {'age_d':>7}"
        )
        for r in decay_rows:
            print(
                f"  {_trunc(r['id'], 12):<12}  {_trunc(r['fact'], 60):<60}  "
                f"{r['vitality']:>6.4f}  {r['access_count']:>4}  "
                f"{r['age_days']:>7.1f}"
            )
    else:
        print("  (no cards)")
    print()
    print(
        f"token footprint:   {tokens} tokens "
        f"(fact+evidence across all cards)"
    )
    return 0


# ---------------------------------------------------------------------- #
# dry-run-consolidation
# ---------------------------------------------------------------------- #
def _cmd_dry_run(store: MemoryStore, limit: int) -> int:
    """Preview the consolidation plan without touching the DB."""
    plan = Consolidator(store, embedder=getattr(store, "embedder", None)).dry_run()

    # Optional truncation of the merges + decay lists, with a "… and X more"
    # note appended so the elision is visible.
    if limit and limit > 0:
        merges = plan.get("proposed_merges", [])
        if len(merges) > limit:
            extra = len(merges) - limit
            plan["proposed_merges"] = merges[:limit]
            plan["proposed_merges_truncated"] = f"... and {extra} more"

        deletions = plan.get("proposed_deletions", {}) or {}
        decay = deletions.get("decay", [])
        if len(decay) > limit:
            extra = len(decay) - limit
            plan["proposed_deletions"]["decay"] = decay[:limit]
            plan["proposed_deletions"]["decay_truncated"] = f"... and {extra} more"

    print("=== isotope_zero dry-run-consolidation ===")
    print("NOTE: NOTHING was committed to the DB. This is a plan preview only.")
    print()
    print(json.dumps(plan, indent=2))
    return 0


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """argparse entry point. Returns a process exit code."""
    # Wire up diagnostic logging from ISOTOPE_ZERO_LOG_LEVEL before anything else,
    # so DEBUG records from triage/store/consolidation surface on stderr.
    configure_logging()

    parser = argparse.ArgumentParser(
        prog="isotope_zero",
        description=(
            "isotope_zero debug CLI: inspect the memory store and preview "
            "consolidation sweeps without committing."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = sub.add_parser(
        "inspect",
        help="human-readable store report (card count, size, decay, tokens)",
    )
    p_inspect.add_argument(
        "--db",
        default=None,
        help=(
            "DB path override. Default: ~/.isotope_zero/isotope_zero.db if it exists, "
            "else :memory: (an empty in-memory store)."
        ),
    )
    p_inspect.add_argument(
        "--json",
        action="store_true",
        help="emit the report as a JSON object instead of aligned text",
    )
    p_inspect.add_argument(
        "--top",
        type=int,
        default=5,
        help="number of decay candidates to show (default 5)",
    )

    # dry-run-consolidation
    p_dry = sub.add_parser(
        "dry-run-consolidation",
        help="preview a consolidation sweep; NOTHING is committed to the DB",
    )
    p_dry.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )
    p_dry.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "truncate proposed_merges and decay lists to N entries each "
            "(default 0 = no truncation)."
        ),
    )

    args = parser.parse_args(argv)
    db_path = _resolve_db_path(args.db)

    if args.command == "inspect":
        store = _open_store(db_path)
        if store is None:
            print(f"DB path does not exist: {db_path}", file=sys.stderr)
            return 1
        try:
            return _cmd_inspect(store, args.top, args.json)
        finally:
            store.close()

    if args.command == "dry-run-consolidation":
        store = _open_store(db_path)
        if store is None:
            print(f"DB path does not exist: {db_path}", file=sys.stderr)
            return 1
        try:
            return _cmd_dry_run(store, args.limit)
        finally:
            store.close()

    return 0


# ---------------------------------------------------------------------- #
# Inline smoke test
# ---------------------------------------------------------------------- #
def _smoke_test() -> None:
    """Seed a temp file DB with 3 cards (2 duplicates + 1 decayed) and run
    both subcommands to confirm non-empty output.

    Uses a temp file DB (not :memory:) so WAL engages, matching the
    production on-disk path. Two cards share an identical fact (exact-fact
    dedup path) and one card is 200 days old + never recalled (decay path),
    so the dry-run plan is non-empty without needing a real ONNX embedder.
    """
    import math

    def _norm(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v

    # Temp file DB so WAL mode engages (the in-memory path ignores the PRAGMA).
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name
    try:
        store = MemoryStore(db_path)
        t0 = now_ts()
        # dup-1 + dup-2: identical fact -> exact-fact dedup will fold dup-2.
        store.add(
            MemoryCard(
                id="dup-1",
                fact="The user prefers dark mode.",
                evidence="user said: 'I love dark mode'",
                timestamp=t0 - 100,
                tags=["preference", "ui"],
                embedding=_norm([1.0, 0.0, 0.0, 0.0]),
                source_tokens=6,
                access_count=2,
                last_access=t0 - 50,
            )
        )
        store.add(
            MemoryCard(
                id="dup-2",
                fact="The user prefers dark mode.",
                evidence="settings log: dark theme enabled",
                timestamp=t0 - 90,
                tags=["preference"],
                embedding=_norm([0.99, 0.01, 0.0, 0.0]),
                source_tokens=5,
                access_count=1,
                last_access=t0 - 40,
            )
        )
        # stale-1: 200 days old, never recalled -> vitality well below the
        # 0.05 floor, past the 1h grace period -> decay candidate.
        store.add(
            MemoryCard(
                id="stale-1",
                fact="An old throwaway note.",
                evidence="debug log line 42",
                timestamp=t0 - (200 * _SECS_PER_DAY),
                tags=["debug"],
                embedding=_norm([0.0, 0.0, 1.0, 0.0]),
                source_tokens=5,
                access_count=0,
                last_access=0.0,
            )
        )
        store.close()
    except Exception:
        try:
            os.unlink(db_path)
        finally:
            pass
        raise

    try:
        print("\n\n===== SMOKE: inspect =====")
        main(["inspect", "--db", db_path])
        print("\n\n===== SMOKE: dry-run-consolidation =====")
        main(["dry-run-consolidation", "--db", db_path])
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
