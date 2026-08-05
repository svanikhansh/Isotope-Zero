"""`izero watch <db_path> [--interval 1.0]` — live memory feed.

A polling-based WAL tailer. It opens the DB read-only on a fresh connection
each poll (never holding one open across polls, so committed WAL frames are
always visible) and streams newly-created or newly-superseded cards as an
agent writes them in another terminal.

Read-only safety (critical): every connection goes through ``open_ro``
(mode=ro + query_only=ON). The command NEVER writes, NEVER sets journal_mode,
and NEVER holds a write lock. Each poll iteration opens and closes its own
connection inside a try/except, so a transient lock or a corrupt frame can't
crash the stream.

Dispatcher contract: ``cmd(args: argparse.Namespace) -> int`` (0 ok, 1 error).
``args.db_path`` is the DB path; ``args.interval`` is the poll interval in
seconds (float, default 1.0). main.py owns argparse parsing; we only consume.
"""
from __future__ import annotations

import os
import time
from typing import Any

# Lazy imports of the shared seams live inside ``cmd`` so an import-time
# failure (e.g. a broken rich install) doesn't poison `izero --help`. The
# command modules are only loaded when their subcommand is actually run.


def _snapshot(conn) -> dict[str, dict[str, Any]]:
    """Fetch the current id -> (fact, dim, superseded_by) state in one pass.

    A single query over ALL rows (live + audit-trail folded) so we can detect
    both NEW cards and NEWLY SUPERSEDED cards in one pass. Embedding length is
    derived from the float32 BLOB (4 bytes/element); SQ8-only rows fall back to
    the int8 column when the float32 column is NULL. Pure read.
    """
    cur = conn.execute(
        "SELECT id, fact, embedding, q_embedding, superseded_by FROM memories"
    )
    state: dict[str, dict[str, Any]] = {}
    for cid, fact, emb_blob, q_blob, sup_by in cur.fetchall():
        dim: int | None = None
        if emb_blob is not None:
            try:
                dim = len(emb_blob) // 4  # float32 = 4 bytes/element
            except TypeError:
                dim = None
        if dim is None and q_blob is not None:
            try:
                dim = len(q_blob)  # int8 = 1 byte/element
            except TypeError:
                dim = None
        state[str(cid)] = {
            "fact": str(fact) if fact is not None else "",
            "dim": dim,
            "superseded_by": str(sup_by) if sup_by is not None else None,
        }
    return state


def cmd(args: Any) -> int:
    """Entry point for `izero watch`. Returns 0 on clean stop, 1 on setup error."""
    # Lazy imports: keep `izero --help` fast and decoupled from rich/sqlite.
    import sqlite3

    from izero_cli.commands._dbutil import open_ro
    from izero_cli.commands._uiutil import (
        AMBER,
        CONSOLE,
        CYAN,
        DIM,
        GOLD,
        GREEN,
        LAVENDER,
        Panel,
        ROUNDED_BOX,
        Text,
        error_panel,
        title,
        trunc,
    )

    db_path: str = args.db_path
    interval: float = float(getattr(args, "interval", 1.0) or 1.0)

    # Missing DB at start -> error panel, exit 1. Never traceback.
    if not db_path or not os.path.exists(db_path):
        CONSOLE.print(
            error_panel(f"database not found: {db_path or '(empty)'}", "watch")
        )
        return 1

    # --- establish baseline ------------------------------------------------- #
    # Open read-only, snapshot every id + its supersede status, close. This is
    # the "before" state; subsequent polls diff against it. ``open_ro`` raises
    # on a missing/corrupt file; ``_snapshot`` raises on a missing table. Both
    # are sqlite3.Error/OSError and render the same error panel — the message
    # distinguishes open vs read so the user knows which step failed.
    try:
        conn = open_ro(db_path)
    except sqlite3.Error as exc:
        CONSOLE.print(error_panel(f"open failed: {exc}", "watch"))
        return 1
    except OSError as exc:
        CONSOLE.print(error_panel(f"open failed: {exc}", "watch"))
        return 1
    try:
        baseline = _snapshot(conn)
    except sqlite3.Error as exc:
        CONSOLE.print(error_panel(f"read failed: {exc}", "watch"))
        return 1
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass

    seen_ids: set[str] = set(baseline.keys())
    last_supersede: dict[str, str | None] = {
        cid: info["superseded_by"] for cid, info in baseline.items()
    }

    # --- startup banner ----------------------------------------------------- #
    banner_body = Text.assemble(
        Text("db   ", style=DIM), Text(db_path, style=CYAN), Text("\n"),
        Text("poll ", style=DIM), Text(f"{interval:.2f}s", style=GOLD), Text("\n"),
        Text("live ", style=DIM),
        Text(f"{len(baseline)} cards at baseline", style="white"),
    )
    CONSOLE.print(Panel(
        banner_body,
        title=title("👀 izero watch — live memory feed"),
        border_style=LAVENDER,
        box=ROUNDED_BOX,
        padding=(1, 2),
    ))

    events: list[dict[str, Any]] = []

    # --- poll loop ---------------------------------------------------------- #
    # Re-open read-only EVERY iteration so we always see WAL-committed changes
    # and never hold a connection (or a lock) across polls. KeyboardInterrupt is
    # the only expected exit; we render a dim summary and return 0.
    try:
        while True:
            time.sleep(interval)
            # Open fresh read-only each poll, snapshot, close — all inside one
            # try/finally so a snapshot failure can't leak the connection.
            try:
                try:
                    conn = open_ro(db_path)
                    current = _snapshot(conn)
                except sqlite3.Error as exc:
                    # Transient lock / corrupt frame: warn but keep streaming.
                    # A live writer can cause brief SQLITE_BUSY; we don't want
                    # to kill the watch over one bad poll.
                    CONSOLE.print(Text(
                        f"  ⚠ poll skipped: {exc}", style=AMBER))
                    continue
                except OSError as exc:
                    CONSOLE.print(Text(
                        f"  ⚠ poll skipped: {exc}", style=AMBER))
                    continue
            finally:
                # ``conn`` is bound iff open_ro succeeded; guard with locals().
                if "conn" in locals():
                    try:
                        conn.close()
                    except sqlite3.Error:
                        pass

            new_state: dict[str, dict[str, Any]] = current

            # NEW cards: ids present now but not seen before. A freshly-folded
            # audit-trail card is also "new" to the stream and is classified by
            # whether it arrives already-superseded.
            for cid, info in new_state.items():
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                last_supersede[cid] = info["superseded_by"]
                if info["superseded_by"] is not None:
                    # Arrived already folded -> audit-trail entry.
                    kind = "audit"
                    glyph = "➕ AUDIT"
                    color = DIM
                else:
                    kind = "new"
                    glyph = "🟢 NEW"
                    color = GREEN
                ev = {
                    "kind": kind,
                    "card_id": cid,
                    "fact": trunc(info["fact"], 60),
                    "dim": info["dim"],
                    "superseded_by": info["superseded_by"],
                    "ts": time.time(),
                    "glyph": glyph,
                    "color": color,
                }
                events.append(ev)
                _emit(ev)

            # NEWLY SUPERSEDED: ids whose superseded_by changed NULL->something
            # or to a different value since the last poll. Only consider ids we
            # already knew about (newly-arrived folded cards handled above).
            for cid, info in new_state.items():
                if cid not in last_supersede:
                    continue
                old_sup = last_supersede[cid]
                new_sup = info["superseded_by"]
                if new_sup == old_sup:
                    continue
                last_supersede[cid] = new_sup
                if new_sup is not None:
                    # NULL -> something (or changed): a live card just got folded.
                    kind = "superseded"
                    glyph = f"🟡 SUPERSEDED → {new_sup}"
                    color = AMBER
                else:
                    # something -> NULL: shouldn't normally happen (audit trail
                    # is append-only), but report it as a state change anyway.
                    kind = "superseded"
                    glyph = "🟡 UNSUPERSEDED (unexpected)"
                    color = AMBER
                ev = {
                    "kind": kind,
                    "card_id": cid,
                    "fact": trunc(info["fact"], 60),
                    "dim": info["dim"],
                    "superseded_by": new_sup,
                    "ts": time.time(),
                    "glyph": glyph,
                    "color": color,
                }
                events.append(ev)
                _emit(ev)
    except KeyboardInterrupt:
        # Graceful stop: dim summary, exit 0. Never traceback on Ctrl-C.
        CONSOLE.print(Text(
            f"⏹ watch stopped ({len(events)} events streamed)", style=DIM))
        return 0


def _emit(event: dict[str, Any]) -> None:
    """Render one feed line to the shared CONSOLE with the event's color."""
    from izero_cli.commands._uiutil import CONSOLE, Text

    ts_str = time.strftime("%H:%M:%S", time.localtime(event["ts"]))
    dim_str = str(event["dim"]) if event["dim"] is not None else "—"
    line = Text.assemble(
        Text(f"{ts_str}  ", style="dim"),
        Text(f'{event["glyph"]}  ', style=event["color"]),
        Text(f'{event["card_id"]}  ', style="bold #00d7ff"),
        Text(f"dim={dim_str}  ", style="dim"),
        Text(f'"{event["fact"]}"', style="white"),
    )
    CONSOLE.print(line)
