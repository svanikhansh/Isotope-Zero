"""`izero vacuum <db_path>` — compact the DB and flush the WAL.

Maintenance command. This is one of the two MUTATING izero-cli subcommands
(the other is `import`); it therefore goes through the write-capable
``open_rw`` seam — never ``open_ro``.

What it does:
    1. Snapshot the BEFORE state (read-only): main DB file size, WAL/SHM
       sidecar sizes, ``page_count`` and ``freelist_count``.
    2. Open read-write (no ``query_only``) and run, as auto-commit statements:
         a. ``PRAGMA wal_checkpoint(TRUNCATE)`` — flush + truncate the WAL.
         b. ``VACUUM`` — rebuild the file, reclaiming freelist pages.
    3. Snapshot the AFTER state and compute deltas.

Safety model (critical):
    - ``PRAGMA wal_checkpoint(TRUNCATE)`` and ``VACUUM`` CANNOT run inside an
      open transaction and CANNOT run under ``query_only``. So they are issued
      as auto-commit statements on a connection from ``open_rw`` (WAL mode,
      synchronous=NORMAL, timeout=30s for a bounded BUSY wait, NO
      query_only). They are deliberately NOT wrapped in BEGIN/COMMIT.
    - Any error → error panel + exit 1. Never a traceback.
    - Missing DB, or a DB locked by a live writer past the 30s timeout →
      error panel + exit 1 (SQLITE_BUSY surfaces as sqlite3.OperationalError).

Dispatcher contract: ``cmd(args: argparse.Namespace) -> int`` (0 ok, 1 error).
``main.py`` owns argparse; we consume ``args.db_path``.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any

# Lazy imports of the shared seams live inside ``cmd`` so an import-time
# failure (e.g. a broken rich install) never poisons `izero --help`.


def _snapshot(db_path: str) -> dict[str, int] | None:
    """Read-only BEFORE/AFTER snapshot.

    Returns a dict with db_bytes, wal_bytes, shm_bytes, page_count,
    freelist_count; or None if the DB can't be opened read-only (caller
    surfaces the error). Uses ``open_ro`` so the snapshot is truly read-only
    and never contends with a writer for the lock.
    """
    from izero_cli.commands._dbutil import (
        db_file_size,
        open_ro,
        wal_sidecar_sizes,
    )

    db_bytes = db_file_size(db_path)
    wal_bytes, shm_bytes = wal_sidecar_sizes(db_path)
    try:
        conn = open_ro(db_path)
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        freelist_count = conn.execute("PRAGMA freelist_count").fetchone()[0]
        conn.close()
    except sqlite3.Error:
        return None
    return {
        "db_bytes": db_bytes,
        "wal_bytes": wal_bytes,
        "shm_bytes": shm_bytes,
        "page_count": int(page_count),
        "freelist_count": int(freelist_count),
    }


def _fmt_bytes(n: int) -> str:
    """Compact human byte size for the before/after table (B/KB/MB)."""
    from izero_cli.commands._dbutil import _human_size

    return _human_size(n)


def cmd(args: Any) -> int:
    """Entry point for `izero vacuum`. Returns 0 on success, 1 on error."""
    from izero_cli.commands._dbutil import open_rw
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
        error_panel,
        title,
    )

    db_path: str = args.db_path

    # --- missing DB → error panel, exit 1 ----------------------------------- #
    if not db_path or not os.path.exists(db_path):
        CONSOLE.print(error_panel(
            f"database not found: {db_path or '(empty)'}", "vacuum"))
        return 1

    # --- BEFORE state (read-only) ------------------------------------------- #
    before = _snapshot(db_path)
    if before is None:
        CONSOLE.print(error_panel(
            f"could not read DB (locked or corrupt): {db_path}", "vacuum"))
        return 1

    # --- write phase: wal_checkpoint(TRUNCATE) + VACUUM (auto-commit) ------- #
    # These MUST run as auto-commit statements: no BEGIN/COMMIT wrapper, and
    # they require a non-query_only connection (open_rw satisfies both).
    checkpointed_frames = 0
    wal_blocked = False
    try:
        conn = open_rw(db_path)
        try:
            # PRAGMA wal_checkpoint(TRUNCATE) returns
            # (blocked, log_frames, checkpointed_frames).
            row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if row is not None and len(row) >= 3:
                wal_blocked = bool(row[0])
                checkpointed_frames = int(row[2])
            # VACUUM as an auto-commit statement (rebuilds the file).
            conn.execute("VACUUM")
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        # SQLITE_BUSY (writer held the lock past timeout) or locked/corrupt.
        CONSOLE.print(error_panel(
            f"vacuum failed (DB busy or locked): {exc}", "vacuum"))
        return 1
    except sqlite3.Error as exc:
        CONSOLE.print(error_panel(f"vacuum failed: {exc}", "vacuum"))
        return 1
    except OSError as exc:
        CONSOLE.print(error_panel(f"vacuum failed: {exc}", "vacuum"))
        return 1

    # --- AFTER state (read-only) -------------------------------------------- #
    after = _snapshot(db_path)
    if after is None:
        # VACUUM succeeded but the post-snapshot read failed — unlikely, but
        # surface it rather than rendering stale numbers.
        CONSOLE.print(error_panel(
            "vacuum completed but post-snapshot read failed", "vacuum"))
        return 1

    # --- deltas ------------------------------------------------------------- #
    db_delta = after["db_bytes"] - before["db_bytes"]
    wal_delta = after["wal_bytes"] - before["wal_bytes"]
    shm_delta = after["shm_bytes"] - before["shm_bytes"]
    # freelist_delta = after - before (negative == pages reclaimed == good).
    freelist_delta = after["freelist_count"] - before["freelist_count"]
    # freelist_reclaimed is the positive count of pages freed (for the summary).
    freelist_reclaimed = before["freelist_count"] - after["freelist_count"]
    page_delta = after["page_count"] - before["page_count"]

    # Reclaimed bytes ≈ freelist pages reclaimed * page_size. The page size is
    # not in the snapshot; derive it from the db delta if available, else read
    # it once. We use it only for the summary line's "reclaimed X.X MB".
    reclaimed_bytes = 0
    try:
        from izero_cli.commands._dbutil import open_ro

        rc = open_ro(db_path)
        page_size = int(rc.execute("PRAGMA page_size").fetchone()[0])
        rc.close()
        reclaimed_bytes = max(0, freelist_reclaimed) * page_size
    except sqlite3.Error:
        reclaimed_bytes = max(0, -db_delta)

    # --- render the before/after panel -------------------------------------- #
    def _delta_txt(delta: int, *, reclaimed_good: bool = False,
                   as_bytes: bool = True) -> Text:
        """Format a Δ cell. Reclaimed (negative growth) renders green.

        ``as_bytes=True`` renders the magnitude via ``_human_size`` (for byte
        metrics); ``as_bytes=False`` renders the raw integer (for page counts).
        """
        if delta == 0:
            return Text("0", style=DIM)
        style = GOLD
        if delta < 0:
            style = GREEN if reclaimed_good else GREEN
        mag = _fmt_bytes(abs(delta)) if as_bytes else str(abs(delta))
        sign = "+" if delta > 0 else "-"
        return Text(f"{sign}{mag}", style=style)

    tbl = Table(
        header_style=f"bold {CYAN}",
        border_style=DIM,
        box=None,
        padding=(0, 2),
        show_header=True,
    )
    tbl.add_column("Metric", style=DIM, no_wrap=True)
    tbl.add_column("Before", style="white", no_wrap=True, justify="right")
    tbl.add_column("After", style="white", no_wrap=True, justify="right")
    tbl.add_column("Δ", no_wrap=True, justify="right")

    tbl.add_row(
        "DB file size",
        Text(_fmt_bytes(before["db_bytes"]), style=GOLD),
        Text(_fmt_bytes(after["db_bytes"]), style=GOLD),
        _delta_txt(db_delta, reclaimed_good=True),
    )
    tbl.add_row(
        "WAL size",
        Text(_fmt_bytes(before["wal_bytes"]), style=GOLD),
        Text(_fmt_bytes(after["wal_bytes"]), style=GOLD),
        _delta_txt(wal_delta, reclaimed_good=True),
    )
    tbl.add_row(
        "SHM size",
        Text(_fmt_bytes(before["shm_bytes"]), style=GOLD),
        Text(_fmt_bytes(after["shm_bytes"]), style=GOLD),
        _delta_txt(shm_delta, reclaimed_good=True),
    )
    tbl.add_row(
        "freelist pages",
        Text(str(before["freelist_count"]), style="white"),
        Text(str(after["freelist_count"]), style="white"),
        _delta_txt(freelist_delta, reclaimed_good=True, as_bytes=False),
    )
    tbl.add_row(
        "page count",
        Text(str(before["page_count"]), style="white"),
        Text(str(after["page_count"]), style="white"),
        _delta_txt(page_delta, as_bytes=False),
    )

    # --- summary line ------------------------------------------------------- #
    reclaimed_mb = reclaimed_bytes / (1024.0 * 1024.0)
    wal_flush_note = (
        f"WAL flushed ({checkpointed_frames} frames checkpointed)"
        if checkpointed_frames > 0
        else "WAL already clean"
    )
    summary = Text.assemble(
        Text("reclaimed ", style=DIM),
        Text(f"{reclaimed_mb:.2f} MB", style=GREEN),
        Text(", ", style=DIM),
        Text(wal_flush_note, style=GOLD),
    )

    # Already-compact DB: delta < 1KB → green "already compact" note.
    already_compact = abs(db_delta) < 1024 and before["freelist_count"] == 0
    if already_compact:
        summary = Text.assemble(
            Text("already compact · ", style=GREEN),
            summary,
        )

    body = Group(tbl, Text(""), summary)

    border = LAVENDER if not already_compact else GREEN
    CONSOLE.print(Panel(
        body,
        title=title("🧹 izero vacuum — compact & flush WAL"),
        subtitle=f"[{DIM}]{db_path}[/]",
        border_style=border,
        padding=(1, 2),
    ))
    return 0
