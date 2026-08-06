"""isotope_zero CLI — live TUI memory dashboard.

``izero dashboard --db PATH`` opens a read-only, auto-refreshing overview of a
store bound to a folder. It mirrors the read-only ``inspect`` / ``stats`` surface
(card count, DB size, embedding mode, vitality histogram, decay candidates) but
renders it as a single panel that refreshes in place until the user presses
``Ctrl-C``. Nothing is written — the store is read once per frame.

Two render transports, chosen at runtime:

    rich  (optional ``[dashboard]`` extra)
        ``rich.live.Live`` for smooth in-place refresh + clean terminal restore
        on exit. Imported lazily inside ``_run_rich`` so the module imports with
        zero optional deps.

    plain (stdlib only — the fallback when ``rich`` is absent)
        ``\033[2J\033[H`` clear-and-reprint + ``time.sleep``. Works everywhere,
        mild flicker, no dependencies.

``--once`` prints one static frame to stdout and exits 0 — scriptable, pipeable,
and the deterministic path the tests exercise (no live loop, no terminal control).

Design notes:
    - Reuses the data paths already used by ``inspect``/``stats``/``dry-run``;
      invents no new queries and mutates nothing. ``Consolidator.dry_run()`` is
      pure (reads ``store.all()`` once, computes the plan in memory, never
      writes), so calling it every frame is safe.
    - Strict missing-DB guard (``create=False``), mirroring ``inspect``: a path
      that does not exist is reported as "DB path does not exist" and exit 1,
      rather than spinning an empty dashboard over a typo.
    - Vitality split matches ``_cmd_stats``: >=0.66 fresh / >=0.33 aging /
      <0.33 decayed.

Run:
    izero dashboard --db ./my-project/mem.db          # live (Ctrl-C to exit)
    izero dashboard --db ./my-project/mem.db --once   # one static frame
    izero dashboard --db ./my-project/mem.db --interval 1.0
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable

from ..core.consolidation import Consolidator
from .debug import IsotopeZero, _embedding_mode, _human_bytes, _open_client

# Vitality bucket thresholds (mirror _cmd_stats in debug.py).
_FRESH, _AGING = 0.66, 0.33
_SECS_PER_DAY = 86_400.0

# The vitality bar is fixed-width so the panel doesn't jitter as counts shift.
_BAR_WIDTH = 20


# --------------------------------------------------------------------------- #
# State collection — pure, testable, no terminal I/O
# --------------------------------------------------------------------------- #
def _collect_state(
    client: IsotopeZero, cons: Consolidator, db_path: str
) -> dict[str, Any]:
    """Gather every panel field into a plain dict (one read pass, no writes).

    ``db_path`` is the path the caller resolved (e.g. the literal ``--db``
    value or the default) — shown verbatim in the title so the user sees what
    they typed, not an internal attribute.

    Returned shape::

        {
          "db_path": str, "size_human": str, "mode": str, "count": int,
          "histogram": {"fresh": int, "aging": int, "decayed": int},
          "recent": [{"id": str, "fact": str, "age_days": float}],   # newest-first, <=5
          "decay": [{"id", "fact", "vitality", "age_days"}],          # dry_run shape, <=5
          "reclaimable_tokens": int, "tokens_total": int,
          "rendered_at": float,                                       # local clock, display only
        }
    """
    store = client.store
    cards = store.all()
    now = time.time()

    fresh = aging = decayed = 0
    for card in cards:
        v = cons.vitality(card, now=now)
        if v >= _FRESH:
            fresh += 1
        elif v >= _AGING:
            aging += 1
        else:
            decayed += 1

    # Recent adds: newest-first by timestamp.
    recent_cards = sorted(cards, key=lambda c: (-c.timestamp, c.id))[:5]
    recent = [
        {
            "id": c.id,
            "fact": c.fact,
            "age_days": round(max(0.0, now - c.timestamp) / _SECS_PER_DAY, 1),
        }
        for c in recent_cards
    ]

    # Decay candidates + reclaimable tokens from a pure in-memory dry-run.
    plan = cons.dry_run()
    decay = plan.get("proposed_deletions", {}).get("decay", [])[:5]
    summary = plan.get("summary", {})
    reclaimable = int(summary.get("tokens_reclaimed_approx", 0))
    tokens_total = int(summary.get("tokens_before", 0))

    return {
        "db_path": db_path,
        "size_human": _human_bytes(store.db_size_bytes()),
        "mode": _embedding_mode(store),
        "count": len(cards),
        "histogram": {"fresh": fresh, "aging": aging, "decayed": decayed},
        "recent": recent,
        "decay": decay,
        "reclaimable_tokens": reclaimable,
        "tokens_total": tokens_total,
        "rendered_at": now,
    }


# --------------------------------------------------------------------------- #
# Rendering — one shared frame builder, no dependency on a TUI library
# --------------------------------------------------------------------------- #
def _bar(fresh: int, aging: int, decayed: int, width: int = _BAR_WIDTH) -> str:
    """A fixed-width unicode block bar (fresh > aging > decayed). No external dep."""
    total = max(1, fresh + aging + decayed)
    # Fractional widths rounded to nearest filled cell; the last segment absorbs
    # the rounding remainder so the bar is always exactly ``width`` wide.
    f = round(fresh / total * width)
    d = round(decayed / total * width)
    a = width - f - d
    a = max(0, min(width - f - d, a))
    if f + a + d != width:  # guard against an off-by-one; rebalance aging
        a = width - f - d
    # Densest blocks for fresh (high vitality), lighter for decayed.
    return "█" * f + "▒" * a + "░" * d


def _truncate(s: str, n: int) -> str:
    s = str(s)
    if n <= 0:
        return ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _render_frame(state: dict[str, Any], interval: float) -> str:
    """Build the dashboard panel as a plain string (both transports use this)."""
    hist = state["histogram"]
    f_, a_, d_ = hist["fresh"], hist["aging"], hist["decayed"]
    bar = _bar(f_, a_, d_) if state["count"] else "░" * _BAR_WIDTH

    # Title: isotope_zero · <db path> — right-padded to panel width.
    title_src = state["db_path"] or ":memory:"
    title = f"isotope_zero · {title_src}"

    lines: list[str] = []
    lines.append(
        f"cards: {state['count']:<6} size: {state['size_human']:<9} "
        f"mode: {state['mode']}"
    )
    lines.append(
        f"vitality: {bar}  fresh {f_} / aging {a_} / decay {d_}"
    )
    tokens_line = f"tokens: {state['tokens_total']} total"
    if state["reclaimable_tokens"] > 0:
        tokens_line += f"  ·  ~{state['reclaimable_tokens']} reclaimable"
    lines.append(tokens_line)

    # Recent adds.
    if state["recent"]:
        lines.append("recent:")
        for r in state["recent"]:
            lines.append(f"  · {_truncate(r['fact'], 54)}  ({r['age_days']}d)")
    else:
        lines.append("recent: (none)")

    # Decay candidates (count = full dry_run length, not just the 5 shown).
    decay = state["decay"]
    n_decay = len(decay)
    if n_decay:
        lines.append(f"decay candidates ({n_decay}):")
        for c in decay:
            lines.append(
                f"  · {_truncate(c['fact'], 50)}  v={c['vitality']:.2f} "
                f"age={c['age_days']}d"
            )
    else:
        lines.append("decay candidates: none")

    # Footer: refresh interval + exit hint.
    footer = f"refresh {interval:g}s   ·   [Ctrl-C to exit]"

    # Box-draw: compute inner width from the longest content line, then pad.
    inner = max(len(title), *(len(l) for l in lines), len(footer))
    top = "┌ " + title + " " * (inner - len(title)) + " ┐"
    bot = "└ " + footer + " " * (inner - len(footer)) + " ┘"
    body = ["│ " + l + " " * (inner - len(l)) + " │" for l in lines]
    return "\n".join([top, *body, bot])


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #
def _run_plain(supplier: Callable[[], dict[str, Any]], interval: float) -> int:
    """Stdlib-only refresh loop: clear screen, print frame, sleep. Ctrl-C exits."""
    try:
        while True:
            state = supplier()
            frame = _render_frame(state, interval)
            # \033[2J = clear screen, \033[H = home cursor.
            sys.stdout.write("\033[2J\033[H" + frame + "\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        # Clear the partial frame so the shell prompt isn't left mid-panel.
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        return 0


def _run_rich(supplier: Callable[[], dict[str, Any]], interval: float) -> int:
    """Smooth in-place refresh via rich.live (lazy import; degrades to _run_plain)."""
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.text import Text
    except ImportError:
        return _run_plain(supplier, interval)

    console = Console()
    # refresh_per_second capped at 1/interval so a long interval doesn't spin.
    rps = max(1, int(1.0 / max(interval, 0.1)))
    try:
        with Live(
            Text(_render_frame(supplier(), interval)),
            console=console,
            refresh_per_second=rps,
            screen=True,  # alternate screen buffer → terminal restored on exit
        ) as live:
            while True:
                live.update(Text(_render_frame(supplier(), interval)))
                time.sleep(interval)
    except KeyboardInterrupt:
        return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_dashboard(
    db_path: str, interval: float = 2.0, once: bool = False
) -> int:
    """Open the store at ``db_path`` and run the live (or one-shot) dashboard.

    Returns 1 + a stderr message when ``db_path`` does not exist (mirrors
    ``inspect``), 0 on clean Ctrl-C / one-shot success.
    """
    if interval <= 0:
        print("interval must be > 0 seconds", file=sys.stderr)
        return 1

    client = _open_client(db_path, create=False)
    if client is None:
        print(f"DB path does not exist: {db_path}", file=sys.stderr)
        return 1

    cons = Consolidator(client.store)

    def supplier() -> dict[str, Any]:
        return _collect_state(client, cons, db_path)

    try:
        if once:
            sys.stdout.write(_render_frame(supplier(), interval) + "\n")
            sys.stdout.flush()
            return 0
        return _run_rich(supplier, interval)
    except KeyboardInterrupt:
        return 0
    finally:
        client.close()
