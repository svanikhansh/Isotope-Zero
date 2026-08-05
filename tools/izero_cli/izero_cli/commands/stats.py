"""`izero stats <db_path>` — deep statistical breakdown of memory cards.

Renders a three-panel dashboard over the NON-superseded active cards:

  1. 📊 tag distribution  — JSON tags parsed + counted, top entries with bars.
  2. ⏳ age distribution  — cards bucketed by ``timestamp`` age into <1h, <1d,
     <7d, <30d, >30d, drawn as a histogram with proportional █ bars.
  3. 🔁 turnover & activity — totals, superseded ratio, mean access_count,
     never-accessed count, an "updates" proxy (superseded count), and a
     cards-per-day rate over the min..max timestamp span.

Read-only safety (critical): the single connection goes through ``open_ro``
(mode=ro + query_only=ON). The command NEVER writes, NEVER sets journal_mode.
All DB access is wrapped in try/except; a missing/corrupt DB renders an
``error_panel`` and returns exit 1 — never a traceback.

Dispatcher contract: ``cmd(args: argparse.Namespace) -> int`` (0 ok, 1 error).
``args.db_path`` is the DB path. main.py owns argparse parsing; we only
consume.
"""
from __future__ import annotations

import os
import time
from typing import Any

# Lazy imports of the shared seams live inside ``cmd`` so an import-time
# failure (e.g. a broken rich install) doesn't poison `izero --help`.


# Age histogram buckets, in ascending age. Each entry is (label, upper_bound_s).
# A card lands in the first bucket whose bound exceeds its age; the final
# bucket is open-ended (>30d). Boundaries match the spec exactly.
_AGE_BUCKETS: tuple[tuple[str, float], ...] = (
    ("<1h", 3600.0),
    ("<1d", 86400.0),
    ("<7d", 86400.0 * 7),
    ("<30d", 86400.0 * 30),
    (">30d", float("inf")),
)


def _bucket_age(age_s: float) -> int:
    """Return the index into ``_AGE_BUCKETS`` for a card age in seconds."""
    for i, (_label, bound) in enumerate(_AGE_BUCKETS):
        if age_s < bound:
            return i
    return len(_AGE_BUCKETS) - 1  # safety: anything >= inf lands in >30d


def _bar(count: int, max_count: int, width: int = 24) -> str:
    """A proportional █ bar. Scales to ``width`` cells; 0 count -> empty."""
    if max_count <= 0 or count <= 0:
        return ""
    n = round((count / max_count) * width)
    return "█" * max(0, min(width, n))


def cmd(args: Any) -> int:
    """Entry point for `izero stats`. Returns 0 on success, 1 on error."""
    # Lazy imports: keep `izero --help` fast and decoupled from rich/sqlite.
    import sqlite3

    from izero_cli.commands._dbutil import open_ro, _parse_tags
    from izero_cli.commands._uiutil import (
        AMBER,
        CONSOLE,
        CYAN,
        DIM,
        GOLD,
        GREEN,
        LAVENDER,
        Group,
        Panel,
        ROUNDED_BOX,
        Table,
        Text,
        error_panel,
        title,
    )

    db_path: str = args.db_path

    # Missing DB -> error panel, exit 1. Never traceback.
    if not db_path or not os.path.exists(db_path):
        CONSOLE.print(
            error_panel(f"database not found: {db_path or '(empty)'}", "stats")
        )
        return 1

    # --- open read-only and compute everything in one connection ------------ #
    try:
        conn = open_ro(db_path)
    except sqlite3.Error as exc:
        CONSOLE.print(error_panel(f"open failed: {exc}", "stats"))
        return 1
    except OSError as exc:
        CONSOLE.print(error_panel(f"open failed: {exc}", "stats"))
        return 1

    # Pull the columns we need for all three panels in a single pass — cheaper
    # than three separate queries and keeps the snapshot consistent. We fetch
    # ALL rows (active + superseded) and partition in Python so the turnover
    # panel can report both counts from one consistent read.
    try:
        rows = conn.execute(
            "SELECT fact, timestamp, tags, access_count, superseded_by "
            "FROM memories"
        ).fetchall()
    except sqlite3.Error as exc:
        CONSOLE.print(error_panel(f"query failed: {exc}", "stats"))
        return 1
    finally:
        # Always close, success or error. We've materialized rows into a list,
        # so rendering is fully offline and never touches the connection.
        try:
            conn.close()
        except sqlite3.Error:
            pass

    now = time.time()

    # Partition into active vs superseded for the turnover panel.
    active_rows: list[tuple] = []
    superseded_count = 0
    for fact, ts, tags_json, access_count, sup_by in rows:
        if sup_by is not None:
            superseded_count += 1
            continue
        active_rows.append((fact, ts, tags_json, access_count))
    total_rows = len(rows)
    active_count = len(active_rows)

    # ---- 1. tag/category distribution -------------------------------------- #
    tag_counts: dict[str, int] = {}
    for _fact, _ts, tags_json, _acc in active_rows:
        for tag in _parse_tags(tags_json):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    # Sort desc by count, then tag asc for determinism.
    tag_items = sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_tags = tag_items[:12]
    max_tag_count = tag_items[0][1] if tag_items else 0

    # ---- 2. age distribution histogram ------------------------------------- #
    bucket_counts = [0] * len(_AGE_BUCKETS)
    for _fact, ts, _tags, _acc in active_rows:
        age = now - float(ts) if ts is not None else float("inf")
        if age < 0:
            age = 0.0  # a future-dated card counts as brand-new.
        bucket_counts[_bucket_age(age)] += 1
    max_bucket = max(bucket_counts) if bucket_counts else 0

    # ---- 3. turnover & update frequency ------------------------------------ #
    access_counts = [int(acc) if acc is not None else 0
                     for _f, _t, _tg, acc in active_rows]
    mean_access = (sum(access_counts) / len(access_counts)) if access_counts else 0.0
    never_accessed = sum(1 for a in access_counts if a == 0)

    # cards-per-day over the min..max timestamp span of ACTIVE cards.
    timestamps = [float(ts) for _f, ts, _tg, _a in active_rows if ts is not None]
    if len(timestamps) >= 2:
        span_s = max(timestamps) - min(timestamps)
        span_days = span_s / 86400.0 if span_s > 0 else 0.0
        cards_per_day = (active_count / span_days) if span_days > 0 else float(active_count)
    elif len(timestamps) == 1:
        cards_per_day = float(active_count)  # single card -> rate is undefined, show count
    else:
        cards_per_day = 0.0

    sup_ratio = (superseded_count / total_rows) if total_rows > 0 else 0.0
    # "updates" proxy: each superseded card represents a prior card replaced.
    updates_proxy = superseded_count

    # ===================================================================== #
    # RENDER
    # ===================================================================== #

    # --- Panel 1: tag distribution ----------------------------------------- #
    tag_tbl = Table(
        box=ROUNDED_BOX,
        border_style=DIM,
        header_style=f"bold {CYAN}",
        padding=(0, 1),
        show_edge=True,
    )
    tag_tbl.add_column("tag", style=LAVENDER, no_wrap=True, max_width=28)
    tag_tbl.add_column("count", style=GOLD, no_wrap=True, justify="right")
    tag_tbl.add_column("bar", style=GREEN, no_wrap=True)
    if top_tags:
        for tag, count in top_tags:
            tag_tbl.add_row(tag, str(count), _bar(count, max_tag_count))
    else:
        tag_tbl.add_row(Text("no tags found", style=DIM), "", "")

    # --- Panel 2: age distribution histogram ------------------------------- #
    age_tbl = Table(
        box=ROUNDED_BOX,
        border_style=DIM,
        header_style=f"bold {CYAN}",
        padding=(0, 1),
        show_edge=True,
    )
    age_tbl.add_column("bucket", style=LAVENDER, no_wrap=True)
    age_tbl.add_column("count", style=GOLD, no_wrap=True, justify="right")
    age_tbl.add_column("histogram", style=CYAN, no_wrap=True)
    for (label, _bound), count in zip(_AGE_BUCKETS, bucket_counts):
        age_tbl.add_row(label, str(count), _bar(count, max_bucket, width=24))

    # --- Panel 3: turnover & activity grid --------------------------------- #
    turn = Table.grid(padding=(0, 3))
    turn.add_column(style=DIM)
    turn.add_column(style=GOLD)
    turn.add_row("total cards", str(total_rows))
    turn.add_row("active cards", str(active_count))
    turn.add_row("superseded", str(superseded_count))
    turn.add_row("supersede ratio", f"{sup_ratio:.1%}")
    turn.add_row("mean access_count", f"{mean_access:.2f}")
    turn.add_row("never accessed", str(never_accessed))
    turn.add_row("updates (proxy)", str(updates_proxy))
    turn.add_row("cards / day", f"{cards_per_day:.2f}")

    # Group the three panels under one lavender-titled dashboard. Using a
    # vertical stack so each panel keeps its own rounded border + title.
    dashboard = Group(
        Panel(
            tag_tbl,
            title=title("📊 tag distribution"),
            border_style=DIM,
            box=ROUNDED_BOX,
            padding=(1, 2),
        ),
        Panel(
            age_tbl,
            title=title("⏳ age distribution"),
            border_style=DIM,
            box=ROUNDED_BOX,
            padding=(1, 2),
        ),
        Panel(
            turn,
            title=title("🔁 turnover & activity"),
            border_style=DIM,
            box=ROUNDED_BOX,
            padding=(1, 2),
        ),
    )
    CONSOLE.print(Panel(
        dashboard,
        title=title("📈 izero stats — memory card distribution"),
        subtitle=f"[{DIM}]{db_path}[/]",
        border_style=LAVENDER,
        box=ROUNDED_BOX,
        padding=(1, 2),
    ))
    return 0
