"""`izero import <db_path> <file> [--format jsonl]` — seed cards from a file.

Seeds Isotope Zero memory cards from a JSONL file into an existing or
freshly-created DB. This is one of the two MUTATING izero-cli subcommands
(the other is `vacuum`); it therefore goes through the write-capable
``open_rw`` / ``create_fresh_db`` seams — never ``open_ro``.

Safety model (critical):
    - All inserts run inside a single transaction (BEGIN … COMMIT). On ANY
      exception the transaction is rolled back and an error panel is rendered,
      so a bad mid-file row can never leave the DB half-written.
    - ``insert_card`` uses ``INSERT OR REPLACE`` so a duplicate ``id`` updates
      the existing card in place; replacements are counted as "updated" rather
      than "imported".
    - Individual malformed JSONL rows are skipped + tallied (invalid count);
      they never abort the whole import. Only setup/open/commit failures abort.

JSONL row contract (one JSON object per line):
    required: ``id`` (non-empty str), ``text`` or ``fact`` (non-empty str)
    optional: ``evidence``, ``timestamp`` (default now), ``tags`` (list,
    default []), ``source_tokens`` (default len(text)//4), ``embedding``
    (list[float], packed via ``encode_float32``), ``access_count`` (default
    0), ``last_access`` (default timestamp), ``superseded_by`` (default None)

Dispatcher contract: ``cmd(args: argparse.Namespace) -> int`` (0 ok, 1 error).
``main.py`` owns argparse; we consume ``args.db_path``, ``args.file``,
``args.format`` (default ``jsonl``; only jsonl is required).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

# Lazy imports of the shared seams live inside ``cmd`` so an import-time
# failure (e.g. a broken rich install) never poisons `izero --help`.


def _row_text(row: dict[str, Any]) -> str | None:
    """Extract the card body from ``text`` or ``fact``. None if neither/empty."""
    txt = row.get("text")
    if txt is None:
        txt = row.get("fact")
    if not isinstance(txt, str) or not txt.strip():
        return None
    return txt


def _parse_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Validate + normalize one JSONL row to insert_card kwargs.

    Returns None if the row is invalid (caller counts it as skipped). All
    optional fields are defaulted here so the insert call stays uniform.
    """
    rid = row.get("id")
    if not isinstance(rid, str) or not rid.strip():
        return None
    text = _row_text(row)
    if text is None:
        return None

    ts = row.get("timestamp")
    if not isinstance(ts, (int, float)) or ts is None:
        ts = time.time()
    ts = float(ts)

    tags = row.get("tags")
    if not isinstance(tags, list):
        tags = []

    stok = row.get("source_tokens")
    if not isinstance(stok, int) or stok < 0:
        stok = len(text) // 4

    emb = row.get("embedding")
    if emb is not None and not (isinstance(emb, list) and all(
            isinstance(x, (int, float)) for x in emb)):
        emb = None

    acc = row.get("access_count")
    if not isinstance(acc, int) or acc < 0:
        acc = 0

    la = row.get("last_access")
    if not isinstance(la, (int, float)):
        la = ts
    else:
        la = float(la)

    sup = row.get("superseded_by")
    if not isinstance(sup, str):
        sup = None

    evidence = row.get("evidence")
    if not isinstance(evidence, str):
        evidence = None

    return {
        "id": rid,
        "fact": text,
        "evidence": evidence,
        "timestamp": ts,
        "tags": tags,
        "source_tokens": stok,
        "embedding": emb,
        "access_count": acc,
        "last_access": la,
        "superseded_by": sup,
    }


def cmd(args: Any) -> int:
    """Entry point for `izero import`. Returns 0 on success, 1 on error."""
    # Lazy imports: keep `izero --help` fast and decoupled from rich/sqlite.
    import sqlite3

    from izero_cli.commands._dbutil import (
        create_fresh_db,
        insert_card,
        open_ro,
        open_rw,
        _tables,
    )
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
        Table,
        Text,
        badge,
        error_panel,
        title,
    )

    db_path: str = args.db_path
    file_path: str = args.file
    fmt: str = getattr(args, "format", "jsonl") or "jsonl"

    # --- missing input file → error panel, exit 1 --------------------------- #
    if not file_path or not os.path.exists(file_path):
        CONSOLE.print(error_panel(
            f"input file not found: {file_path or '(empty)'}", "import"))
        return 1

    # Only jsonl is supported (the default). A non-jsonl format is a usage
    # error surfaced as an error panel rather than a traceback.
    if fmt != "jsonl":
        CONSOLE.print(error_panel(
            f"unsupported format: {fmt!r} (only 'jsonl' is supported)", "import"))
        return 1

    # --- open or create the target DB --------------------------------------- #
    conn: sqlite3.Connection | None = None
    try:
        if os.path.exists(db_path):
            conn = open_rw(db_path)
            # Confirm the canonical memories table is present; a non-izero DB
            # is a hard error (we'd otherwise seed into a stranger schema).
            if "memories" not in _tables(conn):
                conn.close()
                CONSOLE.print(error_panel(
                    f"target DB has no 'memories' table: {db_path}", "import"))
                return 1
        else:
            conn = create_fresh_db(db_path)
    except sqlite3.Error as exc:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        CONSOLE.print(error_panel(f"open/create failed: {exc}", "import"))
        return 1
    except OSError as exc:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        CONSOLE.print(error_panel(f"open/create failed: {exc}", "import"))
        return 1

    # --- read + parse the input file ---------------------------------------- #
    imported = 0
    updated = 0
    skipped = 0
    # Track which ids we've already seen THIS run so we can classify the
    # INSERT OR REPLACE outcome: a pre-existing id (either in the DB before
    # the run, or already replaced earlier in this same run) counts as an
    # update; a brand-new id counts as an import.
    preexisting_ids: set[str] = set()
    seen_this_run: set[str] = set()
    try:
        cur = conn.execute("SELECT id FROM memories")
        for (cid,) in cur.fetchall():
            preexisting_ids.add(str(cid))
    except sqlite3.Error:
        # Reading existing ids is an optimization for accurate counts only;
        # if it fails we still import correctly (counts skew toward updated).
        pass

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            raw_lines = fh.readlines()
    except OSError as exc:
        try:
            conn.close()
        except sqlite3.Error:
            pass
        CONSOLE.print(error_panel(f"read failed: {exc}", "import"))
        return 1

    parsed_rows: list[dict[str, Any]] = []
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue  # blank lines are not "invalid", just ignored
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            skipped += 1
            continue
        if not isinstance(obj, dict):
            skipped += 1
            continue
        norm = _parse_row(obj)
        if norm is None:
            skipped += 1
            continue
        parsed_rows.append(norm)

    # --- single transaction: BEGIN, insert all, COMMIT ---------------------- #
    try:
        conn.execute("BEGIN")
        for norm in parsed_rows:
            is_update = norm["id"] in preexisting_ids or norm["id"] in seen_this_run
            insert_card(
                conn,
                id=norm["id"],
                fact=norm["fact"],
                evidence=norm["evidence"],
                timestamp=norm["timestamp"],
                tags=norm["tags"],
                source_tokens=norm["source_tokens"],
                embedding=norm["embedding"],
                access_count=norm["access_count"],
                last_access=norm["last_access"],
                superseded_by=norm["superseded_by"],
            )
            if is_update:
                updated += 1
            else:
                imported += 1
            seen_this_run.add(norm["id"])
        conn.commit()
    except sqlite3.Error as exc:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        try:
            conn.close()
        except sqlite3.Error:
            pass
        CONSOLE.print(error_panel(
            f"import failed (rolled back): {exc}", "import"))
        return 1
    finally:
        # Only close if we didn't already bail (the success path closes below).
        pass

    # --- final card count + close ------------------------------------------- #
    final_count = 0
    try:
        final_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    except sqlite3.Error:
        final_count = -1
    try:
        conn.close()
    except sqlite3.Error:
        pass

    # --- render the result panel -------------------------------------------- #
    # Body: input file, db path, format badge, then the count metrics.
    meta_tbl = Table.grid(padding=(0, 2))
    meta_tbl.add_column(style=DIM)
    meta_tbl.add_column(style="white")
    meta_tbl.add_row("input", Text(file_path, style=CYAN))
    meta_tbl.add_row("db", Text(db_path, style=CYAN))
    meta_tbl.add_row("format", badge(fmt, CYAN))

    counts_tbl = Table.grid(padding=(0, 2))
    counts_tbl.add_column(style=DIM)
    counts_tbl.add_column()
    counts_tbl.add_row(
        "📦 cards imported",
        Text(str(imported), style=GREEN),
    )
    if updated:
        counts_tbl.add_row(
            "↻ cards updated",
            Text(str(updated), style=GOLD),
        )
    counts_tbl.add_row(
        "⚠ skipped / invalid",
        Text(str(skipped), style=AMBER),
    )
    fc_str = str(final_count) if final_count >= 0 else "—"
    counts_tbl.add_row(
        "🧠 final card count",
        Text(fc_str, style=GOLD),
    )

    body = Group(meta_tbl, Text(""), counts_tbl)

    # Empty input is a soft success: amber "0 cards imported", not an error.
    border = LAVENDER if (imported or updated) else AMBER
    CONSOLE.print(Panel(
        body,
        title=title("📦 izero import — seed cards"),
        subtitle=f"[{DIM}]{db_path}[/]",
        border_style=border,
        padding=(1, 2),
    ))
    return 0
