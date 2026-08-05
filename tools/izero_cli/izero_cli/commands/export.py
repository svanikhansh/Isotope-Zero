"""izero export — dump memory cards to a portable file (jsonl/csv/md).

Reads a memory-engine DB **read-only** (via ``open_ro``) and writes the cards
to a user-specified output file. Both active and superseded (audit-trail) cards
are exported; superseded ones are marked via the ``superseded_by`` field / an
explicit marker so downstream tooling can filter them.

Output formats:
    - jsonl : one JSON object per line (fine-tuning friendly).
    - csv   : header + one row per card (stdlib csv, properly quoted).
    - md    : a human-readable report with per-card sections.

Optional ``--tag`` filters to cards whose parsed JSON tags contain the given
tag (case-insensitive).

Safety: the DB is opened read-only and never written; the only write is to the
user-specified output FILE. Unwritable output / missing DB -> error_panel + 1.
"""
from __future__ import annotations

import csv
import json
import os
import time
from argparse import Namespace
from typing import Any

from izero_cli.commands._dbutil import (
    _safe_open, _parse_tags, _decode_float32, _l2_norm, _human_size,
)
from izero_cli.commands._uiutil import (
    LAVENDER, CYAN, GOLD, WHITE, DIM, GREEN, AMBER, CORAL,
    ROUNDED_BOX, CONSOLE, Group, Panel, Table, Text,
    badge, trunc, iso, title, error_panel,
)


def _load_rows(conn) -> list[dict[str, Any]]:
    """Load every card (active + superseded) with decoded vector info.

    Ordered by (timestamp, id) for stable, reproducible output. Each row dict
    carries: id, fact, evidence, timestamp, tags, source_tokens, access_count,
    last_access, superseded_by, vector_norm, dim.
    """
    cur = conn.execute(
        "SELECT id, fact, evidence, timestamp, tags, source_tokens, "
        "embedding, access_count, last_access, superseded_by "
        "FROM memories ORDER BY timestamp ASC, id ASC"
    )
    rows: list[dict[str, Any]] = []
    for r in cur.fetchall():
        vec = _decode_float32(r[6])  # float32 embedding BLOB
        if vec is not None and len(vec) > 0:
            vnorm = float(_l2_norm(vec))
            dim = len(vec)  # == len(blob) // 4
        else:
            vnorm = None
            dim = None
        rows.append({
            "id": str(r[0]),
            "fact": str(r[1]) if r[1] is not None else "",
            "evidence": str(r[2]) if r[2] is not None else "",
            "timestamp": float(r[3]) if r[3] is not None else 0.0,
            "tags": _parse_tags(r[4]),
            "source_tokens": int(r[5]) if r[5] is not None else 0,
            "access_count": int(r[7]) if r[7] is not None else 0,
            "last_access": float(r[8]) if r[8] is not None else 0.0,
            "superseded_by": str(r[9]) if r[9] is not None else None,
            "vector_norm": vnorm,
            "dim": dim,
        })
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    """One JSON object per line; fine-tuning friendly shape."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            obj = {
                "id": r["id"],
                "text": r["fact"],
                "evidence": r["evidence"],
                "timestamp": r["timestamp"],
                "tags": r["tags"],
                "metadata": {
                    "source_tokens": r["source_tokens"],
                    "access_count": r["access_count"],
                    "last_access": r["last_access"],
                    "superseded_by": r["superseded_by"],
                },
                "vector_norm": r["vector_norm"],
                "dim": r["dim"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_csv(rows: list[dict[str, Any]], path: str) -> None:
    """Header + one row per card; tags pipe-joined; csv stdlib quotes properly."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "fact", "evidence", "timestamp", "tags",
            "source_tokens", "access_count", "last_access",
            "superseded_by", "vector_norm", "dim",
        ])
        for r in rows:
            w.writerow([
                r["id"], r["fact"], r["evidence"], r["timestamp"],
                "|".join(r["tags"]),
                r["source_tokens"], r["access_count"], r["last_access"],
                r["superseded_by"] or "",
                "" if r["vector_norm"] is None else f"{r['vector_norm']:.6f}",
                "" if r["dim"] is None else r["dim"],
            ])


def _write_md(rows: list[dict[str, Any]], path: str, db_path: str) -> None:
    """Human-readable report: header, db path, export time, per-card sections."""
    now = time.time()
    lines: list[str] = [
        "# izero memory export",
        "",
        f"DB: `{db_path}`",
        f"Exported: {iso(now)}  (epoch {now:.0f})",
        f"Cards: {len(rows)}",
        "",
    ]
    for r in rows:
        sup_marker = ""
        if r["superseded_by"] is not None:
            sup_marker = f"  (SUPERSEDED → {r['superseded_by']})"
        lines.append(f"## {r['id']}{sup_marker}")
        lines.append("")
        lines.append(r["fact"])
        lines.append("")
        # evidence as a blockquote
        ev = r["evidence"] or "—"
        for el in ev.splitlines() or ["—"]:
            lines.append(f"> {el}")
        lines.append("")
        lines.append("| field | value |")
        lines.append("| --- | --- |")
        lines.append(f"| timestamp | {iso(r['timestamp'])} ({r['timestamp']}) |")
        lines.append(f"| tags | {', '.join(r['tags']) if r['tags'] else '—'} |")
        lines.append(f"| source_tokens | {r['source_tokens']} |")
        lines.append(f"| access_count | {r['access_count']} |")
        lines.append(f"| last_access | {iso(r['last_access'])} |")
        sup_val = r["superseded_by"] if r["superseded_by"] is not None else "—"
        lines.append(f"| superseded_by | {sup_val} |")
        vn = f"{r['vector_norm']:.6f}" if r["vector_norm"] is not None else "—"
        lines.append(f"| vector_norm | {vn} |")
        dim = str(r["dim"]) if r["dim"] is not None else "—"
        lines.append(f"| dim | {dim} |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def cmd(args: Namespace) -> int:
    """``izero export <db_path> --out <file> [--format jsonl|csv|md] [--tag <tag>]``."""
    db_path: str = args.db_path
    out: str = args.out
    fmt = getattr(args, "format", None) or "jsonl"
    tag = getattr(args, "tag", None)

    # Open read-only (open_ro via the non-raising seam).
    conn, err = _safe_open(db_path)
    if conn is None:
        CONSOLE.print(error_panel(err or "cannot open db", "export"))
        return 1

    try:
        rows = _load_rows(conn)
    except Exception as exc:
        CONSOLE.print(error_panel(f"query failed: {exc}", "export"))
        return 1
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Tag filter (case-insensitive match on parsed JSON tags).
    if tag:
        tag_l = tag.lower()
        rows = [r for r in rows if any(t.lower() == tag_l for t in r["tags"])]

    # Write the output file (the ONLY write in this read-only command).
    try:
        if fmt == "jsonl":
            _write_jsonl(rows, out)
        elif fmt == "csv":
            _write_csv(rows, out)
        elif fmt == "md":
            _write_md(rows, out, db_path)
        else:
            CONSOLE.print(error_panel(f"unknown format: {fmt}", "export"))
            return 1
    except OSError as exc:
        CONSOLE.print(error_panel(f"cannot write {out}: {exc}", "export"))
        return 1

    size = 0
    try:
        size = os.path.getsize(out)
    except OSError:
        size = 0

    # Confirmation panel.
    fmt_color = CYAN if fmt == "jsonl" else (AMBER if fmt == "csv" else LAVENDER)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=DIM)
    grid.add_column(style=WHITE)
    grid.add_row("format", badge(fmt, fmt_color))
    grid.add_row("output", Text(out, style=CYAN))
    grid.add_row("cards", Text(str(len(rows)), style=GOLD))
    if tag:
        grid.add_row("tag filter", Text(tag, style=LAVENDER))
    grid.add_row("file size", Text(_human_size(size), style=GOLD))

    CONSOLE.print(
        Panel(
            Group(grid, Text(""), Text(f"wrote {len(rows)} cards to {out}", style=GREEN)),
            title=title("📦 izero export"),
            subtitle=f"[{DIM}]{db_path}[/]",
            border_style=DIM,
            box=ROUNDED_BOX,
            padding=(1, 2),
        )
    )
    return 0
