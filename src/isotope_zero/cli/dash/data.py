"""isotope_zero CLI — Dashboard data collection (shared by web + TUI).

The single source of truth for what a dashboard frame can show. Pure read
logic over ``MemoryStore`` + ``Consolidator`` — never writes, never spawns a
daemon, never touches the terminal. Both the browser dashboard (``izero
serve``) and the terminal TUI (``izero dashboard``) consume the exact dict
this module produces, so the two surfaces can never drift on the data.

The shape mirrors the proven legacy ``cli/dashboard.py::_collect_state``
contract (which the ``test_dashboard.py`` suite pins), then extends it with
the fields a richer dashboard needs (tags distribution, per-card vitality,
bucket percentages). Everything is JSON-serializable.

Vitality buckets (mirror ``_cmd_stats``): >=0.66 fresh, 0.33–0.66 aging,
<0.33 decayed. ``Consolidator.dry_run()`` is pure (reads ``store.all()``
once, computes in memory, never writes) so calling it per frame is safe.
"""

from __future__ import annotations

import time
from typing import Any

from isotope_zero.core.consolidation import Consolidator

_FRESH, _AGING = 0.66, 0.33
_SECS_PER_DAY = 86_400.0
_RECENT_LIMIT = 5
_DECAY_LIMIT = 5


def _embedding_mode(store) -> str:
    """REAL ONNX / FALLBACK / none — from the store's attached embedder."""
    emb = getattr(store, "embedder", None)
    if emb is None:
        return "none (no embedder attached)"
    if getattr(emb, "is_real", False):
        return "REAL ONNX"
    return "FALLBACK"


def _human_bytes(n: int) -> str:
    """1234567 -> '1.2 MiB' (stdlib-only, mirrors debug._human_bytes)."""
    if n < 0:
        return "0 B"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _bucket(v: float) -> str:
    """Map a vitality score to its bucket name."""
    if v >= _FRESH:
        return "fresh"
    if v >= _AGING:
        return "aging"
    return "decayed"


def collect_state(store, db_path: str) -> dict[str, Any]:
    """Gather every dashboard field into a plain JSON-serializable dict.

    One read pass over ``store.all()`` plus a pure in-memory dry-run. No
    writes, no terminal I/O, safe to call on every refresh frame.

    Returned shape (superset of the legacy contract)::

        {
          "db_path": str,          # verbatim, shown in the title
          "size_bytes": int, "size_human": str,
          "mode": str,             # REAL ONNX / FALLBACK / none (...)
          "count": int,            # total live cards
          "histogram": {"fresh": int, "aging": int, "decayed": int},
          "pct": {"fresh": float, "aging": float, "decayed": float},  # 0-100
          "recent": [{"id","fact","tags","age_days","vitality","bucket"}],  # <=5 newest
          "decay": [{"id","fact","vitality","age_days","bucket"}],          # <=5 lowest
          "decay_total": int,      # full dry-run decay length, not just shown
          "tags": {"tag": count, ...},   # count desc, tag asc tiebreak
          "top_tags": [{"tag","count"}], # <=8 for a distribution strip
          "tokens_total": int, "reclaimable_tokens": int,
          "merged_cards": int,     # dry-run proposed dedup merges
          "rendered_at": float,    # local clock, display only
        }
    """
    cards = store.all()
    cons = Consolidator(store)
    now = time.time()

    fresh = aging = decayed = 0
    vital_by_id: dict[str, float] = {}
    for card in cards:
        v = cons.vitality(card, now=now)
        vital_by_id[card.id] = v
        bucket = _bucket(v)
        if bucket == "fresh":
            fresh += 1
        elif bucket == "aging":
            aging += 1
        else:
            decayed += 1

    total = max(1, len(cards))

    # Recent adds: newest-first by timestamp.
    recent_cards = sorted(cards, key=lambda c: (-c.timestamp, c.id))[:_RECENT_LIMIT]
    recent = [
        {
            "id": c.id,
            "fact": c.fact,
            "tags": list(c.tags),
            "age_days": round(max(0.0, now - c.timestamp) / _SECS_PER_DAY, 1),
            "vitality": round(vital_by_id.get(c.id, 0.0), 4),
            "bucket": _bucket(vital_by_id.get(c.id, 0.0)),
        }
        for c in recent_cards
    ]

    # Decay candidates + reclaimable tokens from a pure in-memory dry-run.
    plan = cons.dry_run()
    decay_all = plan.get("proposed_deletions", {}).get("decay", [])
    decay = [
        {
            "id": d.get("id", ""),
            "fact": d.get("fact", ""),
            "vitality": round(float(d.get("vitality", 0.0)), 4),
            "age_days": d.get("age_days", 0.0),
            "bucket": _bucket(float(d.get("vitality", 0.0))),
        }
        for d in decay_all[:_DECAY_LIMIT]
    ]
    summary = plan.get("summary", {})
    reclaimable = int(summary.get("tokens_reclaimed_approx", 0))
    tokens_total = int(summary.get("tokens_before", 0))
    merged_cards = int(summary.get("merged_cards", 0))

    # Tag distribution (count desc, tag asc tiebreak).
    tag_counts: dict[str, int] = {}
    for card in cards:
        for tag in card.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    tag_dist = dict(sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    top_tags = [{"tag": t, "count": c} for t, c in list(tag_dist.items())[:8]]

    return {
        "db_path": db_path or ":memory:",
        "size_bytes": store.db_size_bytes(),
        "size_human": _human_bytes(store.db_size_bytes()),
        "mode": _embedding_mode(store),
        "count": len(cards),
        "histogram": {"fresh": fresh, "aging": aging, "decayed": decayed},
        "pct": {
            "fresh": round(fresh / total * 100, 1),
            "aging": round(aging / total * 100, 1),
            "decayed": round(decayed / total * 100, 1),
        },
        "recent": recent,
        "decay": decay,
        "decay_total": len(decay_all),
        "tags": tag_dist,
        "top_tags": top_tags,
        "tokens_total": tokens_total,
        "reclaimable_tokens": reclaimable,
        "merged_cards": merged_cards,
        "rendered_at": now,
    }


def missing_db(db_path: str) -> bool:
    """True when ``db_path`` is a file-backed path that does not exist.

    Mirrors the strict missing-DB guard of ``inspect``/legacy dashboard: a
    typo'd path is reported, not spun as an empty dashboard. ``:memory:`` is
    always constructible and never "missing".
    """
    import os

    return db_path != ":memory:" and not os.path.exists(db_path)


__all__ = ["collect_state", "missing_db", "_embedding_mode", "_human_bytes"]
