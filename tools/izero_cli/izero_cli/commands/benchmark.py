"""izero benchmark <db_path> [--queries 100] — local latency harness.

Runs N sample searches against a memory engine and reports cold vs warm
p50/p90/p99 latency stats plus throughput (QPS). It is strictly READ-ONLY:
``izero_cli.db.search_db`` opens its own read-only connections internally, and
this module never opens a write-capable handle.

The query set is derived deterministically from the live cards' ``fact`` text
(seeded RNG for reproducibility), so the benchmark is representative of real
content without depending on a hard-coded query list.

Returns 0 on success or when the DB has too few cards to benchmark; returns 1
only if the DB is missing/unopenable (rendered via error_panel).
"""
from __future__ import annotations

import argparse
import random
import time

from izero_cli.commands._dbutil import _safe_open, _human_size, db_file_size
from izero_cli.commands._uiutil import (
    LAVENDER,
    CYAN,
    GOLD,
    DIM,
    GREEN,
    AMBER,
    CORAL,
    WHITE,
    ROUNDED_BOX,
    CONSOLE,
    Group,
    Panel,
    Table,
    Text,
    title,
    error_panel,
)
from izero_cli.db import search_db


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def _percentile(sorted_lat: list[float], p: int) -> float:
    """Percentile via nearest-rank on an already-sorted list.

    idx = (p/100) * (len-1); falls back to the last element on the empty edge
    case. Never raises on a non-empty sorted list.
    """
    if not sorted_lat:
        return 0.0
    idx = (p / 100.0) * (len(sorted_lat) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_lat) - 1)
    frac = idx - lo
    return round(sorted_lat[lo] + (sorted_lat[hi] - sorted_lat[lo]) * frac, 3)


def _build_queries(facts: list[str], n: int, seed: int = 42) -> list[str]:
    """Build a deterministic query list of length n from card fact texts.

    If there are fewer distinct facts than n, cycle through them with light
    variation (a numeric suffix on a fact fragment) so the lexical/semantic
    path still has real content to match, yet the run is reproducible. Uses a
    seeded Random instance — never the unseeded global RNG.
    """
    rng = random.Random(seed)
    if not facts:
        return []
    # Trim each fact to a representative fragment (first clause / ~60 chars).
    frags = [_fragment(f) for f in facts if f]
    if not frags:
        return []
    # Deterministic shuffle so the query order is reproducible, not DB order.
    rng.shuffle(frags)
    queries: list[str] = []
    for i in range(n):
        base = frags[i % len(frags)]
        if i >= len(frags):
            # Light variation on repeat cycles keeps content representative
            # while remaining deterministic. Drop the tail and add a word hint.
            base = base + f" variant {i // len(frags)}"
        queries.append(base)
    return queries


def _fragment(fact: str, limit: int = 60) -> str:
    """Reduce a fact to a search-representative fragment.

    Split on the first sentence boundary; clip to ``limit`` chars so a very
    long fact doesn't become a degenerate full-text match. Always returns a
    non-empty string.
    """
    if not fact:
        return ""
    head = fact.split(".")[0].strip() or fact
    return head[:limit] if len(head) > limit else head


def _run_pass(db_path: str, queries: list[str]) -> tuple[list[float], float]:
    """Execute one pass over the query list; return (latencies_ms, wall_seconds)."""
    latencies: list[float] = []
    wall_start = time.time()
    for q in queries:
        res = search_db(db_path, q, top_k=5)
        latencies.append(float(res.get("latency_ms") or 0.0))
    wall = time.time() - wall_start
    return latencies, wall


def _detect_mode(db_path: str, queries: list[str]) -> str:
    """Probe search_db once to report which search mode is active.

    Falls back to "lexical" if the probe errors or returns no mode field.
    """
    if not queries:
        return "lexical"
    res = search_db(db_path, queries[0], top_k=5)
    mode = res.get("mode")
    if isinstance(mode, str) and mode:
        return mode
    return "lexical"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt_ms(x: float) -> Text:
    """A gold-styled latency cell with the ms unit."""
    return Text(f"{x:.2f} ms", style=GOLD)


def _fmt_qps(x: float) -> Text:
    """A cyan-styled QPS cell."""
    return Text(f"{x:.1f}", style=CYAN)


def _render_pass_row(tbl: Table, label: str, lats: list[float], wall: float) -> None:
    """Add one benchmark pass row to the stats table."""
    s = sorted(lats)
    p50 = _percentile(s, 50)
    p90 = _percentile(s, 90)
    p99 = _percentile(s, 99)
    qps = (len(lats) / wall) if wall > 0 else 0.0
    tbl.add_row(label, _fmt_ms(p50), _fmt_ms(p90), _fmt_ms(p99), _fmt_qps(qps))


def _render(
    n: int,
    mode: str,
    card_count: int,
    db_path: str,
    cold: tuple[list[float], float],
    warm: tuple[list[float], float],
) -> None:
    # Summary line.
    summary = Text(f"{n} queries  ·  mode ", style=WHITE)
    summary.append(mode, style=CYAN)
    summary.append(f"  ·  {card_count} card{'s' if card_count != 1 else ''}  ·  ",
                   style=WHITE)
    summary.append(db_path, style=DIM)
    summary.append(f"  ({_human_size(db_file_size(db_path))})", style=DIM)

    tbl = Table(box=ROUNDED_BOX, border_style=DIM, show_header=True,
                header_style=f"bold {LAVENDER}", padding=(0, 1))
    tbl.add_column("Pass", style=WHITE, no_wrap=True)
    tbl.add_column("p50", no_wrap=True)
    tbl.add_column("p90", no_wrap=True)
    tbl.add_column("p99", no_wrap=True)
    tbl.add_column("QPS", no_wrap=True)
    _render_pass_row(tbl, "cold", cold[0], cold[1])
    _render_pass_row(tbl, "warm", warm[0], warm[1])

    # cold vs warm delta line (median latency improvement).
    cold_p50 = _percentile(sorted(cold[0]), 50)
    warm_p50 = _percentile(sorted(warm[0]), 50)
    delta = Text("cold→warm  ", style=DIM)
    if cold_p50 > 0:
        diff = cold_p50 - warm_p50
        pct = (diff / cold_p50) * 100.0 if cold_p50 else 0.0
        sign = "↓" if diff >= 0 else "↑"
        color = GREEN if diff >= 0 else CORAL
        delta.append(f"p50 {cold_p50:.2f} → {warm_p50:.2f} ms  ", style=DIM)
        delta.append(f"{sign} {abs(diff):.2f} ms ({abs(pct):.0f}%)",
                     style=f"bold {color}")
    else:
        delta.append("no latency data", style=DIM)

    body = Group(summary, Text(""), tbl, Text(""), delta)
    panel = Panel(body, title=title("⚡ izero benchmark"),
                  border_style=LAVENDER, box=ROUNDED_BOX, padding=(1, 2))
    CONSOLE.print(panel)


def _render_too_few(card_count: int, db_path: str) -> None:
    panel = Panel(
        Text(f"Only {card_count} card{'s' if card_count != 1 else ''} in "
             f"{db_path} — need ≥2 to benchmark meaningfully.",
             style=f"bold {AMBER}"),
        title=title("⚡ izero benchmark"),
        border_style=AMBER, box=ROUNDED_BOX, padding=(1, 2),
    )
    CONSOLE.print(panel)


# --------------------------------------------------------------------------- #
# Dispatcher entry point
# --------------------------------------------------------------------------- #
def cmd(args: argparse.Namespace) -> int:
    """`izero benchmark <db_path> [--queries N]` — render latency stats (0
    unless the DB is missing/unopenable)."""
    db_path = args.db_path
    try:
        n = int(getattr(args, "queries", 100))
    except (TypeError, ValueError):
        n = 100
    if n < 1:
        n = 1

    # Open read-only to harvest facts + card count. search_db opens its own
    # ro connections per query, so this handle is only for set construction.
    conn, err = _safe_open(db_path)
    if conn is None:
        CONSOLE.print(error_panel(err or "open failed", "benchmark"))
        return 1
    try:
        cur = conn.execute(
            "SELECT fact FROM memories WHERE superseded_by IS NULL"
        )
        facts = [str(r[0]) for r in cur.fetchall() if r[0]]
        card_count = len(facts)
    except Exception as exc:
        CONSOLE.print(error_panel(f"benchmark failed: {exc}", "benchmark"))
        return 1
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if card_count < 2:
        _render_too_few(card_count, db_path)
        return 0

    queries = _build_queries(facts, n, seed=42)
    mode = _detect_mode(db_path, queries)
    cold = _run_pass(db_path, queries)   # first pass — cold caches
    warm = _run_pass(db_path, queries)   # identical second pass — warm caches
    _render(n, mode, card_count, db_path, cold, warm)
    return 0
