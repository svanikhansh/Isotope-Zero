"""izero doctor <db_path> — read-only health & integrity diagnostic.

Computes a Health Scorecard across six checks and renders actionable
recommendations:

  1. Vector integrity   — NULL / zero-norm / SQ8-inconsistent embeddings.
  2. Fragmentation      — WAL sidecars, journal mode, freelist/page bloat.
  3. FTS5 consistency   — virtual-table row count vs live cards (never creates).
  4. SQLite integrity   — PRAGMA integrity_check (a pure read under query_only).
  5. Daemon IPC         — izero_cli.db.daemon_status() (missing socket = warn).
  6. Audit references   — dangling superseded_by pointers.

Strictly READ-ONLY: every connection goes through open_ro (mode=ro +
query_only=ON). It never writes, never runs wal_checkpoint / VACUUM — doctor
only *reports* fragmentation. Returns 0 in all cases (it reports, it does not
fail the CLI) UNLESS the DB itself is missing or unopenable, in which case an
error_panel is rendered and 1 is returned.
"""
from __future__ import annotations

import argparse
import sqlite3

from izero_cli.commands._dbutil import (
    _safe_open,
    _table_columns,
    _tables,
    _decode_float32,
    _l2_norm,
    _human_size,
    db_file_size,
    wal_sidecar_sizes,
)
from izero_cli.commands._uiutil import (
    LAVENDER,
    CYAN,
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
from izero_cli.db import daemon_status


# --------------------------------------------------------------------------- #
# Check primitives — each returns a dict: {status, metric, rec}
#   status: "pass" | "warn" | "fail"
#   metric: one-line key metric string
#   rec:    one-line actionable recommendation ("" for a clean pass)
# --------------------------------------------------------------------------- #
def _check_vector_integrity(
    conn: sqlite3.Connection, has_sq8: bool
) -> dict:
    """Flag live cards with NULL, zero-norm, or SQ8-inconsistent embeddings."""
    cols = "id, embedding" + (", q_embedding, q_scale" if has_sq8 else ", NULL, NULL")
    try:
        rows = conn.execute(
            f"SELECT {cols} FROM memories WHERE superseded_by IS NULL"
        ).fetchall()
    except sqlite3.Error as exc:
        return {"status": "fail", "metric": f"query failed: {exc}",
                "rec": "Inspect the memories table schema."}

    null_emb = 0
    zero_norm = 0
    sq8_incon = 0
    offenders: list[str] = []
    for r in rows:
        cid = str(r[0])
        emb_blob = r[1]
        if emb_blob is None:
            null_emb += 1
            offenders.append(f"{cid} (no embedding)")
            continue
        vec = _decode_float32(emb_blob)
        if vec is None or len(vec) == 0:
            null_emb += 1
            offenders.append(f"{cid} (undecodable embedding)")
            continue
        if _l2_norm(vec) == 0.0:
            zero_norm += 1
            offenders.append(f"{cid} (zero-norm vector)")
        if has_sq8:
            qe, qs = r[2], r[3]
            # Inconsistent = exactly one of (q_embedding, q_scale) is NULL.
            if (qe is None) != (qs is None):
                sq8_incon += 1
                offenders.append(f"{cid} (SQ8 half-set)")

    total = len(rows)
    bad = null_emb + zero_norm + sq8_incon
    metric = f"{bad} anomaly / {total} live card{'s' if total != 1 else ''}"
    if bad == 0:
        return {"status": "pass", "metric": metric, "rec": ""}
    sample = ", ".join(offenders[:3]) + (" …" if len(offenders) > 3 else "")
    return {"status": "warn", "metric": metric,
            "rec": f"Re-embed/fix: {sample}"}


def _check_fragmentation(
    conn: sqlite3.Connection, db_path: str
) -> dict:
    """WAL sidecar sizes + journal mode + freelist/page bloat ratio.

    bloat = freelist_count / page_count. amber >20%, coral >40%. These PRAGMAs
    (page_count / freelist_count / journal_mode) are pure reads under
    query_only=ON; no wal_checkpoint is ever issued.
    """
    wal_bytes, shm_bytes = wal_sidecar_sizes(db_path)
    try:
        mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
    except sqlite3.Error:
        mode = "unknown"
    try:
        page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
    except sqlite3.Error:
        page_count = 0
    try:
        freelist = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
    except sqlite3.Error:
        freelist = 0

    ratio = (freelist / page_count) if page_count else 0.0
    pct = ratio * 100.0
    metric = (f"bloat {pct:.1f}% · WAL {_human_size(wal_bytes)} · "
              f"SHM {_human_size(shm_bytes)} · {mode}")
    if pct > 40.0:
        status, rec = "fail", f"Run `izero vacuum` to reclaim {pct:.0f}% page bloat"
    elif pct > 20.0:
        status, rec = "warn", f"Run `izero vacuum` to reclaim {pct:.0f}% page bloat"
    else:
        status, rec = "pass", ""
    return {"status": status, "metric": metric, "rec": rec}


def _find_fts5_tables(conn: sqlite3.Connection) -> set[str]:
    """Locate FTS5 virtual tables (by VIRTUAL TABLE … fts5 SQL), fallback to
    a spec-literal startswith('fts') filter on _tables. Never creates one."""
    fts: set[str] = set()
    try:
        cur = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table'"
        )
        for name, sql in cur.fetchall():
            sql_txt = (sql or "").lower()
            if "virtual" in sql_txt and "fts5" in sql_txt:
                fts.add(str(name))
    except sqlite3.Error:
        pass
    if not fts:  # fallback: literal name-prefix filter
        fts = {t for t in _tables(conn) if t.lower().startswith("fts")}
    return fts


def _check_fts5(conn: sqlite3.Connection) -> dict:
    """Compare FTS5 row counts to live memories; flag drift. No-FTS = pass."""
    fts_tables = _find_fts5_tables(conn)
    if not fts_tables:
        return {"status": "pass", "metric": "not enabled (ok)", "rec": ""}
    try:
        live = int(conn.execute(
            "SELECT count(*) FROM memories WHERE superseded_by IS NULL"
        ).fetchone()[0])
    except sqlite3.Error:
        live = -1
    drift: list[str] = []
    total_fts = 0
    for t in sorted(fts_tables):
        try:
            n = int(conn.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0])
        except sqlite3.Error:
            n = -1
        total_fts += max(n, 0)
        if n != live:
            drift.append(f"{t}={n}")
    metric = (f"fts rows {total_fts} vs live {live} "
              f"({len(fts_tables)} table{'s' if len(fts_tables) != 1 else ''})")
    if drift:
        return {"status": "warn", "metric": metric,
                "rec": f"FTS index drift ({', '.join(drift)}) — rebuild the index"}
    return {"status": "pass", "metric": metric, "rec": ""}


def _check_integrity(conn: sqlite3.Connection) -> dict:
    """PRAGMA integrity_check — a read-only probe returning 'ok' or error text."""
    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.Error as exc:
        return {"status": "fail", "metric": f"check failed: {exc}",
                "rec": "DB may be corrupt — restore from backup"}
    text = "; ".join(str(r[0]) for r in rows).strip() if rows else ""
    if text.lower() == "ok":
        return {"status": "pass", "metric": "ok", "rec": ""}
    from izero_cli.commands._uiutil import trunc
    return {"status": "fail", "metric": trunc(text, 64),
            "rec": "Integrity errors detected — restore from backup or run `izero vacuum`"}


def _check_daemon() -> dict:
    """Daemon IPC probe. A missing socket is a WARNING (daemon is optional)."""
    try:
        st = daemon_status()
    except Exception as exc:  # daemon_status never raises, but be safe
        return {"status": "warn", "metric": f"probe failed: {exc}",
                "rec": "Daemon status unavailable (optional)"}
    socket_exists = bool(st.get("socket_exists"))
    connected = bool(st.get("socket_connected"))
    procs = st.get("processes") or []
    if connected:
        return {"status": "pass", "metric": "socket connected", "rec": ""}
    if procs:
        return {"status": "warn",
                "metric": f"{len(procs)} proc(s), socket not connected",
                "rec": "Daemon process running but socket unreachable — check the socket path"}
    if socket_exists:
        return {"status": "warn", "metric": "stale socket (not connected)",
                "rec": "Stale daemon socket — remove it or restart the daemon"}
    return {"status": "warn", "metric": "no socket, no process",
            "rec": "Daemon not running (optional) — start it for hot embeddings"}


def _check_supersede_refs(conn: sqlite3.Connection) -> dict:
    """For each superseded card, verify its superseded_by target exists."""
    try:
        rows = conn.execute(
            "SELECT id, superseded_by FROM memories WHERE superseded_by IS NOT NULL"
        ).fetchall()
    except sqlite3.Error as exc:
        return {"status": "fail", "metric": f"query failed: {exc}",
                "rec": "Inspect the memories table."}
    if not rows:
        return {"status": "pass", "metric": "0 superseded card", "rec": ""}
    dangling: list[str] = []
    for cid, sup in rows:
        try:
            found = conn.execute(
                "SELECT 1 FROM memories WHERE id = ?", (sup,)
            ).fetchone()
        except sqlite3.Error:
            found = None
        if not found:
            dangling.append(f"{cid} → missing {sup}")
    metric = f"{len(rows)} superseded, {len(dangling)} dangling"
    if not dangling:
        return {"status": "pass", "metric": metric, "rec": ""}
    sample = "; ".join(dangling[:3]) + (" …" if len(dangling) > 3 else "")
    return {"status": "warn", "metric": metric,
            "rec": f"Dangling supersede ref: {sample}"}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _status_badge(status: str) -> Text:
    if status == "pass":
        return Text("✓ pass", style=f"bold {GREEN}")
    if status == "warn":
        return Text("⚠ warn", style=f"bold {AMBER}")
    return Text("✗ fail", style=f"bold {CORAL}")


def _overall_badge(checks: list[tuple[str, dict]]) -> Text:
    statuses = [c["status"] for _, c in checks]
    if "fail" in statuses:
        return Text("✗ unhealthy", style=f"bold {CORAL}")
    if "warn" in statuses:
        return Text("⚠ needs attention", style=f"bold {AMBER}")
    return Text("✓ healthy", style=f"bold {GREEN}")


def _render(checks: list[tuple[str, dict]], db_path: str) -> None:
    tbl = Table(box=ROUNDED_BOX, border_style=DIM, show_header=True,
                header_style=f"bold {LAVENDER}", padding=(0, 1), expand=True)
    tbl.add_column("Check", style=WHITE, no_wrap=True)
    tbl.add_column("Status", no_wrap=True)
    tbl.add_column("Detail", style="white", overflow="fold")
    tbl.add_column("Recommendation", style=DIM, overflow="fold")
    for name, c in checks:
        rec = c.get("rec") or "—"
        tbl.add_row(name, _status_badge(c["status"]), c.get("metric", ""), rec)

    head = Text(f"{db_path}  ", style=f"bold {CYAN}")
    head.append(f"({_human_size(db_file_size(db_path))})", style=DIM)

    overall = _overall_badge(checks)
    footer = Text("overall  ", style=DIM)
    footer.append_text(overall)
    footer.append(f"  · {len(checks)} checks", style=DIM)

    body = Group(head, Text(""), tbl, Text(""), footer)
    panel = Panel(body, title=title("🏥 izero doctor — health scorecard"),
                  border_style=LAVENDER, box=ROUNDED_BOX, padding=(1, 2))
    CONSOLE.print(panel)


# --------------------------------------------------------------------------- #
# Dispatcher entry point
# --------------------------------------------------------------------------- #
def _run_check(fn) -> dict:
    """Run one check; never let a crash abort the whole scorecard."""
    try:
        return fn()
    except Exception as exc:
        return {"status": "fail", "metric": f"check crashed: {exc}",
                "rec": "This check could not complete — investigate"}


def cmd(args: argparse.Namespace) -> int:
    """`izero doctor <db_path>` — render the health scorecard (0 unless DB unopenable)."""
    db_path = args.db_path
    conn, err = _safe_open(db_path)
    if conn is None:
        CONSOLE.print(error_panel(err or "open failed", "doctor"))
        return 1
    try:
        if "memories" not in _tables(conn):
            CONSOLE.print(error_panel("table 'memories' not found", "doctor"))
            return 1
        cols = _table_columns(conn, "memories")
        has_sq8 = "q_embedding" in cols and "q_scale" in cols
        checks: list[tuple[str, dict]] = [
            ("Vector integrity", _run_check(lambda: _check_vector_integrity(conn, has_sq8))),
            ("Fragmentation", _run_check(lambda: _check_fragmentation(conn, db_path))),
            ("FTS5 consistency", _run_check(lambda: _check_fts5(conn))),
            ("SQLite integrity", _run_check(lambda: _check_integrity(conn))),
            ("Daemon IPC", _run_check(_check_daemon)),
            ("Audit references", _run_check(lambda: _check_supersede_refs(conn))),
        ]
        _render(checks, db_path)
        return 0
    except Exception as exc:
        CONSOLE.print(error_panel(f"doctor failed: {exc}", "doctor"))
        return 1
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
