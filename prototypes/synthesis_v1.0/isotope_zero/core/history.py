"""Revision history tracker for isotope_zero (mem0 port).

Ports mem0's per-memory update/delete audit trail into isotope_zero's
SQLite backbone. mem0 inlines history tracking into ``Memory.update`` /
``_delete_memory`` via ``self.db.add_history(memory_id, prev_value, data,
"UPDATE", ...)`` calls (mem0/memory/main.py:2064 for UPDATE, :2096 for
DELETE). There the ``prev_value``/``new_value`` pair captures the before/
after text of every mutation so a caller can audit or revert.

This module adapts that pattern to isotope_zero's single-file SQLite store:
instead of a separate history DB, it lives in the SAME WAL connection the
``MemoryStore`` already holds. ``init_history(conn)`` creates the
``memories_history`` table (additive — ``CREATE TABLE IF NOT EXISTS``, never
drops/renames), and ``MemoryHistoryTracker`` records + reverts mutations on
demand. ``rollback(history_id)`` re-applies ``old_fact`` to the live
``memories`` row, marks the history row ``rolled_back=1``, and inserts a new
``rollback`` event recording the revert — mirroring how mem0's audit trail
is append-only (you never lose a record, even the revert itself).

Design rules (per repo constraints):
  - Pure stdlib: ``sqlite3``, ``uuid4``, ``time``, ``dataclasses``.
  - Operates on the store's ``sqlite3.Connection`` — does NOT create its own.
  - No threading (the store serializes callers on its lock).
  - Double-quoted strings, ``from __future__ import annotations``, typed
    signatures, docstrings cite mem0 source path:line.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from uuid import uuid4


# ------------------------------------------------------------------ #
# Schema
# ------------------------------------------------------------------ #

_CREATE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS memories_history (
    history_id   TEXT PRIMARY KEY,
    card_id      TEXT,
    scope        TEXT,
    old_fact     TEXT,
    new_fact     TEXT,
    event_type   TEXT,
    created_at   REAL,
    rolled_back  INTEGER DEFAULT 0
);
"""

_CREATE_HISTORY_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_history_card ON memories_history(card_id);"
)


# ------------------------------------------------------------------ #
# Data model
# ------------------------------------------------------------------ #

@dataclass
class HistoryRecord:
    """A single immutable mutation event on one memory card."""

    history_id: str
    card_id: str
    scope: str
    old_fact: str | None
    new_fact: str | None
    event_type: str
    created_at: float
    rolled_back: int = 0


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def init_history(conn: sqlite3.Connection) -> None:
    """Create the ``memories_history`` table + index if absent. Idempotent.

    Additive only: ``CREATE TABLE IF NOT EXISTS`` never touches an existing
    table, and the index is likewise guarded. Called from
    ``MemoryStore._init_schema`` right after ``graph.init_graph``.
    """
    cur = conn.cursor()
    try:
        cur.execute(_CREATE_HISTORY_TABLE)
        cur.execute(_CREATE_HISTORY_INDEX)
        conn.commit()
    finally:
        cur.close()


class MemoryHistoryTracker:
    """Append-only audit trail for mutations on ``memories`` rows.

    Each ``record(...)`` call inserts a new ``memories_history`` row capturing
    the before/after fact text for an UPDATE (or ``new_fact=None`` for a
    DELETE). ``rollback(history_id)`` reverts one such row: it writes
    ``old_fact`` back onto the live ``memories`` row (only if the card still
    exists and is not superseded), flips the source row's ``rolled_back`` flag
    to 1, and inserts a fresh ``rollback`` event recording the revert. The
    trail therefore never loses a record — exactly like mem0's audit trail
    (mem0/memory/main.py:2064-2073, :2096-2104), where every mutation is
    logged and a revert is itself a logged mutation.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(
        self,
        card_id: str,
        scope: str,
        old_fact: str | None,
        new_fact: str | None,
        event_type: str,
    ) -> str:
        """Insert one history row. Returns the new ``history_id``.

        ``event_type`` is free-form but conventionally ``"UPDATE"`` or
        ``"DELETE"`` (matching mem0's ``add_history(..., "UPDATE", ...)`` at
        main.py:2068 and ``"DELETE"`` at :2100).
        """
        history_id = uuid4().hex
        created_at = time.time()
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO memories_history "
                "(history_id, card_id, scope, old_fact, new_fact, event_type, "
                " created_at, rolled_back) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (history_id, card_id, scope, old_fact, new_fact, event_type, created_at),
            )
            self._conn.commit()
        finally:
            cur.close()
        return history_id

    def get_history(self, card_id: str) -> list[HistoryRecord]:
        """Return every history row for ``card_id`` oldest-first.

        Ordered by ``created_at ASC`` so a caller replaying the trail sees
        mutations in the order they happened. Rolled-back rows are included
        (they are still part of the audit trail).
        """
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT history_id, card_id, scope, old_fact, new_fact, "
                "event_type, created_at, rolled_back "
                "FROM memories_history WHERE card_id = ? "
                "ORDER BY created_at ASC",
                (card_id,),
            )
            rows = cur.fetchall()
        finally:
            cur.close()
        return [
            HistoryRecord(
                history_id=str(r[0]),
                card_id=str(r[1]),
                scope=str(r[2]) if r[2] is not None else "default",
                old_fact=str(r[3]) if r[3] is not None else None,
                new_fact=str(r[4]) if r[4] is not None else None,
                event_type=str(r[5]) if r[5] is not None else "",
                created_at=float(r[6]) if r[6] is not None else 0.0,
                rolled_back=int(r[7]) if r[7] is not None else 0,
            )
            for r in rows
        ]

    def rollback(self, history_id: str) -> bool:
        """Revert the mutation recorded at ``history_id``.

        Steps (all on the store's held connection):
          1. SELECT the history row — must exist AND have ``rolled_back=0``.
             Returns ``False`` for an unknown id or an already-rolled row.
          2. UPDATE the live ``memories`` row: set ``fact=old_fact`` ONLY if
             the card still exists AND is not superseded (``superseded_by IS
             NULL``). A rolled-back UPDATE onto a folded/archived card would
             silently resurrect a dead fact, which we refuse. Returns
             ``False`` if the card is gone or superseded.
          3. Flip the source history row ``rolled_back=1``.
          4. INSERT a fresh ``rollback`` event recording the revert, so the
             trail stays append-only (the revert itself is auditable).

        Returns ``True`` only when the revert fully succeeded.
        """
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT history_id, card_id, scope, old_fact, new_fact, "
                "event_type, created_at, rolled_back "
                "FROM memories_history WHERE history_id = ?",
                (history_id,),
            )
            row = cur.fetchone()
            if row is None:
                # Unknown history id — nothing to revert.
                return False
            rolled_back = int(row[7]) if row[7] is not None else 0
            if rolled_back != 0:
                # Already rolled back — idempotent refusal.
                return False
            card_id = str(row[1])
            scope = str(row[2]) if row[2] is not None else "default"
            old_fact = row[3]
            event_type = str(row[5]) if row[5] is not None else ""

            # Revert the live row — only if it still exists and is live
            # (not superseded). We refuse to resurrect a folded card.
            cur.execute(
                "UPDATE memories SET fact = ? WHERE id = ? "
                "AND superseded_by IS NULL",
                (old_fact, card_id),
            )
            if cur.rowcount == 0:
                # Card is gone or superseded — do not mark the history row
                # rolled back, so a later caller can still see it as pending.
                return False

            # Mark the source history row as rolled back.
            cur.execute(
                "UPDATE memories_history SET rolled_back = 1 "
                "WHERE history_id = ?",
                (history_id,),
            )

            # Append a fresh audit row recording the revert itself.
            rb_id = uuid4().hex
            rb_ts = time.time()
            cur.execute(
                "INSERT INTO memories_history "
                "(history_id, card_id, scope, old_fact, new_fact, event_type, "
                " created_at, rolled_back) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    rb_id,
                    card_id,
                    scope,
                    old_fact,            # old_fact of the revert = old_fact
                    None,                # new_fact n/a for a rollback marker
                    "rollback",
                    rb_ts,
                ),
            )
            self._conn.commit()
        finally:
            cur.close()
        return True


# ------------------------------------------------------------------ #
# Smoke test
# ------------------------------------------------------------------ #

def _smoke() -> None:
    """Inline smoke: create table, record two updates, roll one back."""
    conn = sqlite3.connect(":memory:")
    init_history(conn)
    # Minimal live memories table so rollback's UPDATE has a target.
    conn.execute(
        "CREATE TABLE memories(id TEXT PRIMARY KEY, fact TEXT, superseded_by TEXT)"
    )
    conn.execute("INSERT INTO memories(id, fact, superseded_by) VALUES ('c1', 'v0', NULL)")
    conn.commit()
    t = MemoryHistoryTracker(conn)
    h1 = t.record("c1", "default", "v0", "v1", "UPDATE")
    conn.execute("UPDATE memories SET fact='v1' WHERE id='c1'")
    conn.commit()
    h2 = t.record("c1", "default", "v1", "v2", "UPDATE")
    conn.execute("UPDATE memories SET fact='v2' WHERE id='c1'")
    conn.commit()
    hist = t.get_history("c1")
    assert len(hist) == 2, f"expected 2 rows, got {len(hist)}"
    assert t.rollback(h2) is True
    assert t.rollback("no-such-id") is False
    assert t.rollback(h2) is False  # already rolled
    fact = conn.execute("SELECT fact FROM memories WHERE id='c1'").fetchone()[0]
    assert fact == "v1", f"expected v1 after rollback, got {fact}"
    hist2 = t.get_history("c1")
    assert len(hist2) == 3, f"expected 3 rows after rollback, got {len(hist2)}"
    assert hist2[-1].event_type == "rollback"
    print("history smoke OK")


if __name__ == "__main__":
    _smoke()
