"""izero diff — session delta between two Isotope Zero memory DBs.

Compares two memory-engine databases (both opened read-only via ``open_ro``)
and reports cards **added**, **superseded/modified**, and **deleted/forgotten**
between them. Optionally restricts the comparison to cards whose timestamp is
``>= --since`` (the session window).

Delta categories (by card id):
    - Added            : in db2's active set, not in db1's active set, and not a
                         supersede replacement target.
    - Superseded/Modified : a card live in db1 that db2 has folded (superseded_by
                         went NULL -> target), grouped as "card-X -> card-X2";
                         PLUS same id active in both but `fact` text differs.
    - Deleted/Forgotten : in db1's active set and absent from db2 entirely (not
                         even as a superseded audit row) — fully purged.

Read-only: never writes; opens two read-only connections. Errors render an
``error_panel`` and return 1.
"""
from __future__ import annotations

from argparse import Namespace

from izero_cli.commands._dbutil import _safe_open, _parse_tags
from izero_cli.commands._uiutil import (
    LAVENDER, CYAN, GOLD, WHITE, DIM, GREEN, AMBER, CORAL,
    ROUNDED_BOX, CONSOLE, Group, Panel, Table, Text,
    trunc, iso, title, error_panel,
)

# Card tuple: (fact, timestamp, superseded_by, tags).
_Card = tuple[str, float, str | None, list[str]]


def _load_cards(conn) -> dict[str, _Card]:
    """Load every card row as ``{id: (fact, timestamp, superseded_by, tags)}``.

    Pulls active AND superseded rows so the deleted-vs-superseded distinction
    can be made downstream. Tolerant of NULL columns.
    """
    cards: dict[str, _Card] = {}
    cur = conn.execute(
        "SELECT id, fact, timestamp, superseded_by, tags FROM memories"
    )
    for r in cur.fetchall():
        cid = str(r[0])
        fact = str(r[1]) if r[1] is not None else ""
        ts = float(r[2]) if r[2] is not None else 0.0
        sup = str(r[3]) if r[3] is not None else None
        tags = _parse_tags(r[4])
        cards[cid] = (fact, ts, sup, tags)
    return cards


def cmd(args: Namespace) -> int:
    """``izero diff <db1> <db2> [--since <timestamp>]`` -> exit code."""
    db1: str = args.db1
    db2: str = args.db2
    since = getattr(args, "since", None)

    # Open BOTH dbs read-only (open_ro, via the non-raising _safe_open seam).
    conn1, err1 = _safe_open(db1)
    if conn1 is None:
        CONSOLE.print(error_panel(err1 or "cannot open db1", "diff"))
        return 1
    conn2, err2 = _safe_open(db2)
    if conn2 is None:
        try:
            conn1.close()
        except Exception:
            pass
        CONSOLE.print(error_panel(err2 or "cannot open db2", "diff"))
        return 1

    try:
        all1 = _load_cards(conn1)
        all2 = _load_cards(conn2)
    except Exception as exc:
        CONSOLE.print(error_panel(f"query failed: {exc}", "diff"))
        return 1
    finally:
        for c in (conn1, conn2):
            try:
                c.close()
            except Exception:
                pass

    # Apply the --since session window to both sets (before deriving active).
    since_f: float | None = None
    if since is not None:
        try:
            since_f = float(since)
        except (TypeError, ValueError):
            CONSOLE.print(error_panel(f"invalid --since value: {since}", "diff"))
            return 1
        all1 = {k: v for k, v in all1.items() if v[1] >= since_f}
        all2 = {k: v for k, v in all2.items() if v[1] >= since_f}

    # Active cards = superseded_by IS NULL.
    active1 = {k: v for k, v in all1.items() if v[2] is None}
    active2 = {k: v for k, v in all2.items() if v[2] is None}

    # --- Superseded transitions: live in db1, folded in db2 (NULL -> target) ---
    # (old_id, new_id, old_fact, new_fact)
    superseded: list[tuple[str, str, str, str]] = []
    sup_targets: set[str] = set()
    for cid, (fact1, _ts, sup, _tags) in all2.items():
        if sup is not None and cid in active1:
            new_fact = all2[sup][0] if sup in all2 else ""
            superseded.append((cid, sup, fact1, new_fact))
            sup_targets.add(sup)

    # --- Modified: same id active in both, fact text differs (not superseded) ---
    seen_sup = {s[0] for s in superseded}
    # (id, old_fact, new_fact)
    modified: list[tuple[str, str, str]] = []
    for cid, (fact1, _ts, _sup, _tags) in active1.items():
        if cid in seen_sup or cid not in active2:
            continue
        if fact1 != active2[cid][0]:
            modified.append((cid, fact1, active2[cid][0]))

    # --- Added: in db2 active, not in db1 active, not a supersede target ---
    added: list[tuple[str, str]] = []
    for cid, (fact, _ts, _sup, _tags) in active2.items():
        if cid not in active1 and cid not in sup_targets:
            added.append((cid, fact))

    # --- Deleted/Forgotten: in db1 active, absent from db2 entirely ---
    deleted: list[tuple[str, str]] = []
    for cid, (fact, _ts, _sup, _tags) in active1.items():
        if cid not in all2:
            deleted.append((cid, fact))

    # Stable ordering for reproducible output.
    added.sort(key=lambda x: x[0])
    superseded.sort(key=lambda x: x[0])
    modified.sort(key=lambda x: x[0])
    deleted.sort(key=lambda x: x[0])

    n_add = len(added)
    n_sup = len(superseded) + len(modified)
    n_del = len(deleted)

    # --- No differences -> green panel ---
    if n_add == 0 and n_sup == 0 and n_del == 0:
        CONSOLE.print(
            Panel(
                Text("✓ no differences in the comparison window", style=GREEN),
                title=title("↔️ izero diff — session delta"),
                subtitle=f"[{DIM}]{db1}  ↔  {db2}[/]",
                border_style=GREEN,
                box=ROUNDED_BOX,
                padding=(1, 2),
            )
        )
        return 0

    # --- Header grid: paths, counts, window ---
    header = Table.grid(padding=(0, 2))
    header.add_column(style=DIM)
    header.add_column(style=WHITE)
    header.add_row("db1", f"{db1}  ({len(active1)} active)")
    header.add_row("db2", f"{db2}  ({len(active2)} active)")
    if since_f is not None:
        header.add_row("since", f"{iso(since_f)}  ({since_f})")

    sections: list = [header, Text("")]

    # --- Added (green) ---
    if added:
        t = Table(header_style=f"bold {GREEN}", border_style=DIM,
                  box=ROUNDED_BOX, padding=(0, 1))
        t.add_column("card id", style=CYAN, no_wrap=True, max_width=20)
        t.add_column("fact preview", style="white", max_width=50)
        for cid, fact in added:
            t.add_row(trunc(cid, 20), trunc(fact, 50))
        sections.append(
            Panel(t, title=f"[bold {GREEN}]🟢 Added ({n_add})[/]",
                  border_style=DIM, box=ROUNDED_BOX))

    # --- Superseded / Modified (amber) ---
    if superseded or modified:
        t = Table(header_style=f"bold {AMBER}", border_style=DIM,
                  box=ROUNDED_BOX, padding=(0, 1))
        t.add_column("card", style=CYAN, no_wrap=True, max_width=24)
        t.add_column("change", style="white", max_width=54)
        for old_id, new_id, old_fact, new_fact in superseded:
            t.add_row(
                Text(f"{old_id} → {new_id}", style=AMBER),
                trunc(new_fact or old_fact, 54),
            )
        for cid, old_fact, new_fact in modified:
            t.add_row(
                trunc(cid, 24),
                Text(f"{trunc(old_fact, 24)} → {trunc(new_fact, 24)}", style="white"),
            )
        sections.append(
            Panel(t, title=f"[bold {AMBER}]🟡 Superseded/Modified ({n_sup})[/]",
                  border_style=DIM, box=ROUNDED_BOX))

    # --- Deleted / Forgotten (coral) ---
    if deleted:
        t = Table(header_style=f"bold {CORAL}", border_style=DIM,
                  box=ROUNDED_BOX, padding=(0, 1))
        t.add_column("card id", style=CYAN, no_wrap=True, max_width=20)
        t.add_column("fact preview", style="white", max_width=50)
        for cid, fact in deleted:
            t.add_row(trunc(cid, 20), trunc(fact, 50))
        sections.append(
            Panel(t, title=f"[bold {CORAL}]🔴 Deleted/Forgotten ({n_del})[/]",
                  border_style=DIM, box=ROUNDED_BOX))

    # --- Footer totals ---
    footer = Table.grid(padding=(0, 2))
    footer.add_column(style=DIM)
    footer.add_column(style=GOLD)
    footer.add_row("added", str(n_add))
    footer.add_row("superseded/modified", str(n_sup))
    footer.add_row("deleted", str(n_del))
    sections.append(Text(""))
    sections.append(footer)

    CONSOLE.print(
        Panel(
            Group(*sections),
            title=title("↔️ izero diff — session delta"),
            subtitle=f"[{DIM}]{db1}  ↔  {db2}[/]",
            border_style=DIM,
            box=ROUNDED_BOX,
            padding=(1, 2),
        )
    )
    return 0
