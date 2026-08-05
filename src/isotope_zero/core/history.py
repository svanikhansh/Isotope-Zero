"""Time-travel history + rollback for ``MemoryStore`` (v1.0 synthesis).

The shipped v1.0 line tracked per-card *access* metadata (``access_count``,
``last_access``, ``stability``) inside ``store.MemoryStore`` but exposed no
versioned *content* history: an ``update()`` overwrote the prior ``fact``/
``evidence`` with no recoverable prior state, and ``archive_card`` was a
one-way soft-delete. This module closes that gap with a first-class
``History`` handle the unified client composes as ``client.history``.

Design (append-only snapshot log; no store.py write-path churn):
    A sidecar ``memories_history`` table holds an append-only log of card
    *snapshots*. ``History.snapshot(card_id)`` is called by the client BEFORE
    every mutating op (``remember``-update, ``touch``-with-revision,
    ``archive``) — it copies the card's current full state into the log with
    an autoincrement version id and a wall-clock ``seen_at``. Nothing in
    ``store.py`` itself is modified: the snapshot is driven from the client
    facade, which already mediates writes. This keeps the store's lock, WAL,
    and mmap vector index untouched and avoids adding triggers that would
    fire on the eval harness's direct-to-connection seeder.

    ``rollback(card_id, to_version=None)`` restores a prior snapshot: it reads
    the chosen (or latest-but-one) snapshot, re-applies it via ``store.update``
    (which the client snapshots first — so the rollback itself is also
    logged, making time-travel reversible), and returns the restored card.
    A rollback to a version that doesn't exist raises ``KeyError``.

Why snapshots rather than a trigger-based CDC: the store is concurrently
written by the consolidator (which hard-deletes decay-pruned ids and marks
superseded-by pointers) and by the eval harness's bulk seeder. Trigger-based
history would double-log those internal mutations and interleave with
user-driven revisions. Snapshotting only at the *client* facade captures the
revisions a user actually wants to undo, which is the whole point of
``client.history.rollback``.

Limitations (documented, not fixed here):
    - Snapshots are best-effort: if the store was mutated directly (bypassing
      the client facade) between two ``snapshot`` calls, the log will not
      capture those intermediate states. This is acceptable for the v1.0
      client-mediated model; a trigger backstop can be added in v1.1.
    - Hard-deleted cards (consolidator prune) lose their log rows with the row
      only if the history table is also pruned — it is NOT, so a rolled-back
      card resurrects even after a prune, which is the desired audit behavior.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from isotope_zero.types import MemoryCard, now_ts

log = logging.getLogger("isotope_zero.history")


class History:
    """Append-only versioned history log + ``rollback`` time-travel handle.

    Composed by the unified client as ``client.history``. Owns a private
    ``memories_history`` table created lazily on first use against the store's
    connection. Thread-safe via the store's existing ``_lock`` (every snapshot
    / list / rollback acquires it), so concurrent client callers serialize
    correctly.
    """

    def __init__(self, store: Any) -> None:
        # ``store`` is a ``MemoryStore``; typed as Any to dodge a circular
        # import (store.py imports from this module's package indirectly via
        # the client; this module imports from types only).
        self._store = store
        self._init_table()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _init_table(self) -> None:
        """Create the sidecar history table if absent. Idempotent."""
        sql = """
        CREATE TABLE IF NOT EXISTS memories_history (
            version     INTEGER NOT NULL,
            card_id     TEXT    NOT NULL,
            fact        TEXT,
            evidence    TEXT,
            timestamp   REAL,
            tags        TEXT,
            embedding   BLOB,
            access_count INTEGER,
            last_access REAL,
            superseded_by TEXT,
            stability   REAL,
            importance  REAL,
            archived    REAL,
            scope       TEXT,
            seen_at     REAL    NOT NULL,
            PRIMARY KEY (card_id, version)
        );
        """
        with self._store._lock:
            cur = self._store._conn.cursor()
            cur.execute(sql)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_hist_card ON memories_history(card_id, version)"
            )
            cur.close()
            self._store._conn.commit()

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #
    def snapshot(self, card_id: str) -> int | None:
        """Log the CURRENT state of ``card_id`` and return its new version id.

        Called by the client facade before a mutating op so the pre-mutation
        state is recoverable. Returns ``None`` (and logs nothing) when the
        card does not currently exist — e.g. the very first ``remember`` of a
        new id has no prior state to snapshot. The version id is
        per-card-sequential (1, 2, 3 … for that card_id).
        """
        # Read the live card under the store lock so the snapshot reflects a
        # consistent point-in-time state.
        with self._store._lock:
            row = self._store._conn.execute(
                "SELECT id, fact, evidence, timestamp, tags, embedding, "
                "access_count, last_access, superseded_by, stability, "
                "importance, archived, scope FROM memories WHERE id = ?",
                (card_id,),
            ).fetchone()
            if row is None:
                return None
            # Next per-card version = max(version)+1, or 1 for the first.
            cur = self._store._conn.cursor()
            cur.execute(
                "SELECT COALESCE(MAX(version), 0) FROM memories_history WHERE card_id = ?",
                (card_id,),
            )
            next_version = (cur.fetchone()[0] or 0) + 1
            cur.execute(
                """
                INSERT INTO memories_history
                  (version, card_id, fact, evidence, timestamp, tags, embedding,
                   access_count, last_access, superseded_by, stability,
                   importance, archived, scope, seen_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    next_version,
                    row[0], row[1], row[2], row[3], row[4], row[5],
                    row[6], row[7], row[8], row[9], row[10], row[11], row[12],
                    now_ts(),
                ),
            )
            cur.close()
            self._store._conn.commit()
        log.debug("snapshot card_id=%s version=%d", card_id, next_version)
        return next_version

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def versions(self, card_id: str) -> list[int]:
        """All logged version ids for ``card_id``, ascending."""
        with self._store._lock:
            rows = self._store._conn.execute(
                "SELECT version FROM memories_history WHERE card_id = ? ORDER BY version ASC",
                (card_id,),
            ).fetchall()
        return [r[0] for r in rows]

    def get_version(self, card_id: str, version: int) -> MemoryCard:
        """Reconstruct the ``MemoryCard`` for a logged version.

        Raises ``KeyError`` if no such (card_id, version) snapshot exists.
        """
        with self._store._lock:
            row = self._store._conn.execute(
                "SELECT fact, evidence, timestamp, tags, embedding, "
                "access_count, last_access, superseded_by, stability, "
                "importance, archived, scope FROM memories_history "
                "WHERE card_id = ? AND version = ?",
                (card_id, version),
            ).fetchone()
        if row is None:
            raise KeyError(f"no history version {version} for card {card_id!r}")
        return MemoryCard(
            id=card_id,
            fact=row[0],
            evidence=row[1],
            timestamp=row[2],
            tags=json.loads(row[3]) if row[3] else [],
            embedding=self._store._decode_embedding(row[4]) if row[4] else None,
            access_count=row[5],
            last_access=row[6],
            superseded_by=row[7],
            stability=row[8],
            importance=row[9],
            archived=row[10],
            scope=row[11],
        )

    # ------------------------------------------------------------------ #
    # Rollback
    # ------------------------------------------------------------------ #
    def rollback(self, card_id: str, to_version: int | None = None) -> MemoryCard:
        """Restore ``card_id`` to a prior logged state.

        With ``to_version=None`` restores the latest-but-one snapshot (the
        state immediately before the most recent mutation) — the common
        "undo my last edit" case. With an explicit version, restores that
        exact snapshot. The rollback itself snapshots the current state first
        (so the rollback is reversible), then re-applies the chosen version via
        ``store.update``. Returns the restored card.

        Raises ``KeyError`` if no snapshots exist for the card, or the chosen
        version does not exist.
        """
        versions = self.versions(card_id)
        if not versions:
            raise KeyError(f"no history to roll back for card {card_id!r}")
        if to_version is None:
            # latest-but-one (the most recent snapshot is the pre-current
            # state, i.e. exactly the "undo" target).
            to_version = versions[-1]
        elif to_version not in versions:
            raise KeyError(f"no history version {to_version} for card {card_id!r}")
        restored = self.get_version(card_id, to_version)
        # Snapshot the CURRENT state first so the rollback is itself undoable.
        # (snapshot is a no-op if the card was hard-deleted between the last
        # mutation and now — that's fine; we resurrect via update below.)
        try:
            self.snapshot(card_id)
        except Exception as exc:  # never block a rollback on a snapshot failure
            log.debug("pre-rollback snapshot failed for %s: %s", card_id, exc)
        # Re-apply: update re-inserts if the row was deleted (resurrection).
        self._store.update(restored)
        self._store._mark_vec_dirty()
        log.info(
            "rollback card_id=%s to_version=%d fact=%r",
            card_id, to_version, restored.fact[:80],
        )
        return restored
