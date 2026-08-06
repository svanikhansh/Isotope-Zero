"""isotope_zero CLI — inspect, manage, and recall the memory store.

A lightweight, pure-stdlib CLI wrapping the high-level ``IsotopeZero`` client.
Diagnostic subcommands (``inspect``, ``dry-run-consolidation``) read the store
directly; the memory surface (``add``/``active``/``recall``/``search``/``list``/
``get``/``forget``/``touch``/``tags``/``stats``) dispatches through the client
so writes embed, recall/search use the hybrid router, and lifecycle (``touch``)
goes through the real access-tracking path.

Subcommands:

    inspect
        Human-readable store report: card count, DB size, embedding mode,
        average embedding dimension, top-N lowest-vitality decay
        candidates, and the total token footprint. ``--json`` emits the same
        data as a JSON object.

    dry-run-consolidation
        Preview what a ``Consolidator.run()`` sweep WOULD do — proposed
        merges and decay deletions — WITHOUT committing anything to the DB.
        The plan dict returned by ``Consolidator.dry_run()`` is already
        JSON-serializable, so we just pretty-print it.

    add <fact> [--evidence] [--tags] [--scope] [--id] [--json]
        Embed + persist one fact (creates the DB if missing). ``--id`` upserts.

    active [--top] [--tags] [--min-vitality] [--json]
        Cards ranked by vitality DESC (warmest first) — the inverse of
        inspect's decay-candidate view.

    recall <query> [--k] [--alpha] [--json]
        Semantic retrieval (cosine + Ebbinghaus retention).

    search <query> [--k] [--alpha] [--json]
        Hybrid retrieval (vector + BM25 + entity-graph boost).

    list [--tags] [--limit] [--json]
        All cards, newest-first.

    get <id> [--json]   |   forget <id> [--yes]   |   touch <id>
        Single-card read / delete / mark-accessed.

    tags [--json]   |   stats [--json]
        Tag distribution; compact store summary.

    dashboard [--db PATH] [--interval N] [--once]
        Live, auto-refreshing read-only TUI overview (count, size, mode,
        vitality bar, recent adds, decay candidates). ``--once`` prints one
        static frame. Smooth refresh via the optional ``[dashboard]`` (rich)
        extra; pure-stdlib clear-and-reprint fallback otherwise.

Design notes:
    - Pure stdlib (argparse, json, os, sys) + project modules. No new deps
      (the ``dashboard`` subcommand gains an optional ``rich`` extra).
    - Does NOT require a real ONNX embedder. The ``IsotopeZero`` engine falls
      back to deterministic pseudo-embeddings when no ONNX runtime is present,
      so ``recall``/``search`` work with zero optional deps (degraded semantic
      quality, ranked results still returned). ``inspect`` reports the mode.
    - Default DB path: ``~/.isotope_zero/isotope_zero.db`` if it exists, else
      ``:memory:`` (an empty in-memory store). ``--db PATH`` overrides.
    - A missing explicit ``--db PATH`` is reported gracefully (no crash) for
      read commands; ``add`` creates it (the expected write behavior).

Run:
    .venv/bin/python -m isotope_zero.cli.debug inspect --db :memory:
    .venv/bin/python -m isotope_zero.cli.debug add "Paris is the capital of France" --tags geo
    .venv/bin/python -m isotope_zero.cli.debug recall "capital of France"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any

from ..client import IsotopeZero
from ..core.consolidation import Consolidator
from ..core.store import MemoryStore
from ..diagnostics import configure_logging
from ..tokens import estimate_tokens
from ..types import MemoryCard, now_ts

# Default on-disk store location. Used only when no ``--db`` is given AND the
# file exists; otherwise we fall back to an empty ``:memory:`` store so the
# CLI is always runnable with zero setup.
_DEFAULT_DB: str = os.path.expanduser("~/.isotope_zero/isotope_zero.db")

# Seconds per day, for the age-in-days display.
_SECS_PER_DAY: float = 86400.0


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _resolve_db_path(db_path: str | None) -> str:
    """Pick the DB path: explicit override, else default-if-present else :memory:."""
    if db_path is not None:
        return db_path
    return _DEFAULT_DB if os.path.exists(_DEFAULT_DB) else ":memory:"


def _open_store(db_path: str) -> MemoryStore | None:
    """Open a MemoryStore at ``db_path``.

    Returns None for a non-``:memory:`` path that does not exist on disk, so
    the caller can report it gracefully instead of letting sqlite3 silently
    create an empty file (which would surprise a user who mistyped a path).
    For ``:memory:`` the store is always openable (and empty).
    """
    if db_path != ":memory:" and not os.path.exists(db_path):
        return None
    return MemoryStore(db_path)


def _human_bytes(n: int) -> str:
    """Bytes -> human-readable string (B / KB / MB)."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _embedding_mode(store: MemoryStore) -> str:
    """REAL ONNX / FALLBACK / none — from the store's attached embedder."""
    emb = getattr(store, "embedder", None)
    if emb is None:
        return "none (no embedder attached)"
    if getattr(emb, "is_real", False):
        return "REAL ONNX"
    return "FALLBACK"


def _trunc(s: str, n: int) -> str:
    """Truncate a string to n chars (display-only; JSON keeps full values)."""
    s = str(s)
    return s if len(s) <= n else s[:n]


def _open_client(db_path: str, create: bool = True) -> IsotopeZero | None:
    """Open an IsotopeZero client (embedder attached) at ``db_path``.

    By default (``create=True``) a missing on-disk DB is opened as an empty
    store: ``add`` creates it on first write, and read/lifecycle commands
    (``get``/``forget``/``recall``/…) see an empty store and report "no cards"
    rather than "DB does not exist" — matching the user expectation that a
    not-yet-created store is equivalent to an empty one. Pass ``create=False``
    to keep the strict missing-DB guard (used by the diagnostic commands,
    which surface "DB does not exist" as meaningful output).

    For ``:memory:`` the client is always constructible (and empty).

    Built with ``spawn_daemon=False, use_mmap=False`` so the CLI never tries to
    start the shared-memory embedding daemon or mmap a vector index file — the
    engine still embeds (real ONNX if available, else deterministic fallback),
    which is all the recall/search/add path needs.
    """
    if not create and db_path != ":memory:" and not os.path.exists(db_path):
        return None
    return IsotopeZero(db_path, spawn_daemon=False, use_mmap=False)


def _parse_tags(csv: str | None) -> list[str]:
    """``"a, b"`` -> ``["a", "b"]`` (stripped, empties dropped). None/``""`` -> []."""
    if not csv:
        return []
    return [p.strip() for p in csv.split(",") if p.strip()]


def _filter_by_tags(cards: list[MemoryCard], tags: list[str]) -> list[MemoryCard]:
    """Keep cards whose ``card.tags`` intersect ``tags``; empty ``tags`` -> all."""
    if not tags:
        return list(cards)
    want = set(tags)
    return [c for c in cards if want.intersection(c.tags)]


def _vitality_row(card: MemoryCard, now: float, vitality: float | None = None) -> dict[str, Any]:
    """One normalized row for vitality-ranked views.

    ``age_days`` is measured from the card's creation ``timestamp`` (not
    ``last_access``) so it answers "how old is this memory". ``vitality`` may
    be passed in (avoids a redundant ``Consolidator.vitality`` call when the
    caller already scored the card) or left None to compute nothing here —
    callers that need a score always supply it, so None is preserved verbatim.
    """
    return {
        "id": card.id,
        "fact": card.fact,
        "vitality": round(vitality, 4) if vitality is not None else None,
        "access_count": card.access_count,
        "age_days": round(max(0.0, now - card.timestamp) / _SECS_PER_DAY, 1),
    }


def _confirm(prompt: str, yes: bool) -> bool:
    """Yes/no confirmation. Returns True without prompting when ``yes`` or non-tty stdin."""
    if yes or not sys.stdin.isatty():
        return True
    print(prompt, end=" ")
    sys.stdout.flush()
    try:
        line = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        return False
    return line.strip().lower().startswith("y")


def _age_days(timestamp: float, now: float) -> float:
    """Card age in days, floored at 0. Shared by the list/recall/search views."""
    return round(max(0.0, now - timestamp) / _SECS_PER_DAY, 1)


def _run_client(db_path: str, fn, *cmd_args: Any, create: bool = True) -> int:
    """Open an ``IsotopeZero`` client, guard a missing path, run ``fn``, close.

    Shared dispatch wrapper for the subcommands that operate through the
    high-level client (everything except ``inspect`` / ``dry-run-consolidation``,
    which keep using ``_open_store``). By default opens create-or-empty so a
    missing DB is treated as an empty store (see ``_open_client``); returns 1
    with a stderr message only when ``create=False`` and the path is absent.
    Otherwise calls ``fn(client, *cmd_args)`` and returns its exit code,
    closing the client in all cases.
    """
    client = _open_client(db_path, create=create)
    if client is None:
        print(f"DB path does not exist: {db_path}", file=sys.stderr)
        return 1
    try:
        return fn(client, *cmd_args)
    finally:
        client.close()


# ---------------------------------------------------------------------- #
# inspect
# ---------------------------------------------------------------------- #
def _cmd_inspect(store: MemoryStore, top: int, as_json: bool) -> int:
    """Print a human-readable (or JSON) store report."""
    cards = store.all()
    total = len(cards)
    size_bytes = store.db_size_bytes()
    mode = _embedding_mode(store)

    # Average embedding dimension across cards that carry an embedding.
    with_emb = [c for c in cards if c.embedding]
    if with_emb:
        avg_dim = sum(len(c.embedding) for c in with_emb) / len(with_emb)
        emb_json: dict[str, Any] = {
            "avg_dim": round(avg_dim, 2),
            "cards_with_embeddings": len(with_emb),
        }
        emb_line = (
            f"avg embedding dim: {avg_dim:.1f} "
            f"(across {len(with_emb)} cards with embeddings)"
        )
    else:
        emb_json = {
            "avg_dim": None,
            "cards_with_embeddings": 0,
            "note": "no embeddings",
        }
        emb_line = "no embeddings"

    # Top-N decay candidates: the cards with the LOWEST vitality score.
    # Pure vitality ranking (no grace-period / never-recalled filtering) —
    # this is a diagnostic view, not the actual prune decision.
    cons = Consolidator(store)
    now = now_ts()
    scored = [(cons.vitality(c, now=now), c) for c in cards]
    scored.sort(key=lambda item: item[0])
    candidates = scored[:top] if top > 0 else scored
    decay_rows: list[dict[str, Any]] = [
        {
            "id": c.id,
            "fact": c.fact,
            "vitality": round(v, 4),
            "access_count": c.access_count,
            "age_days": round(max(0.0, now - c.timestamp) / _SECS_PER_DAY, 1),
        }
        for v, c in candidates
    ]

    # Total token footprint: fact + evidence tokens across all cards.
    tokens = sum(estimate_tokens(c.fact) + estimate_tokens(c.evidence) for c in cards)

    if as_json:
        payload = {
            "store": {
                "db_path": store.db_path,
                "total_cards": total,
                "db_size_bytes": size_bytes,
                "db_size_human": _human_bytes(size_bytes),
                "embedding_mode": mode,
            },
            "embeddings": emb_json,
            "decay_candidates": decay_rows,
            "token_footprint": tokens,
        }
        print(json.dumps(payload, indent=2))
        return 0

    # Text report — clean aligned columns via f-string padding.
    print("=== isotope_zero inspect ===")
    print(f"store path:        {store.db_path}")
    print(f"total cards:       {total}")
    print(f"db size:           {size_bytes} bytes ({_human_bytes(size_bytes)})")
    print(f"embedding mode:    {mode}")
    print()
    print(emb_line)
    print()
    print(f"top {len(decay_rows)} decay candidates (lowest vitality):")
    if decay_rows:
        print(
            f"  {'id':<12}  {'fact':<60}  {'vital':>6}  "
            f"{'acc':>4}  {'age_d':>7}"
        )
        for r in decay_rows:
            print(
                f"  {_trunc(r['id'], 12):<12}  {_trunc(r['fact'], 60):<60}  "
                f"{r['vitality']:>6.4f}  {r['access_count']:>4}  "
                f"{r['age_days']:>7.1f}"
            )
    else:
        print("  (no cards)")
    print()
    print(
        f"token footprint:   {tokens} tokens "
        f"(fact+evidence across all cards)"
    )
    return 0


# ---------------------------------------------------------------------- #
# dry-run-consolidation
# ---------------------------------------------------------------------- #
def _cmd_dry_run(store: MemoryStore, limit: int) -> int:
    """Preview the consolidation plan without touching the DB."""
    plan = Consolidator(store, embedder=getattr(store, "embedder", None)).dry_run()

    # Optional truncation of the merges + decay lists, with a "… and X more"
    # note appended so the elision is visible.
    if limit and limit > 0:
        merges = plan.get("proposed_merges", [])
        if len(merges) > limit:
            extra = len(merges) - limit
            plan["proposed_merges"] = merges[:limit]
            plan["proposed_merges_truncated"] = f"... and {extra} more"

        deletions = plan.get("proposed_deletions", {}) or {}
        decay = deletions.get("decay", [])
        if len(decay) > limit:
            extra = len(decay) - limit
            plan["proposed_deletions"]["decay"] = decay[:limit]
            plan["proposed_deletions"]["decay_truncated"] = f"... and {extra} more"

    print("=== isotope_zero dry-run-consolidation ===")
    print("NOTE: NOTHING was committed to the DB. This is a plan preview only.")
    print()
    print(json.dumps(plan, indent=2))
    return 0


# ---------------------------------------------------------------------- #
# add — remember a new fact (embedder attached)
# ---------------------------------------------------------------------- #
def _cmd_add(client: IsotopeZero, fact: str, evidence: str, tags: list[str],
             scope: str | None, card_id: str | None, as_json: bool) -> int:
    """Embed + persist one fact via ``client.remember``; print the new card id."""
    # An explicit --id that already exists means this is an overwrite (upsert),
    # not a create. An auto-generated id is always a create.
    existed = card_id is not None and client.store.get(card_id) is not None
    cid = client.remember(
        fact,
        evidence=evidence,
        tags=tags or None,
        scope=scope,
        card_id=card_id,
    )
    card = client.store.get(cid)
    if as_json:
        payload: dict[str, Any] = {
            "id": cid,
            "fact": fact,
            "evidence": evidence,
            "tags": list(tags) if tags else [],
            "scope": scope,
            "created": not existed,
        }
        if card is not None:
            payload["timestamp"] = card.timestamp
        print(json.dumps(payload, indent=2))
        return 0
    print(f"remembered {cid}")
    return 0


# ---------------------------------------------------------------------- #
# active — warmest memories (highest vitality first)
# ---------------------------------------------------------------------- #
def _cmd_active(client: IsotopeZero, top: int, tags: list[str],
                min_vitality: float, as_json: bool) -> int:
    """Rank all cards by vitality DESC, filter by tags/min-vitality, take top N."""
    cards = client.store.all()
    cards = _filter_by_tags(cards, tags)
    cons = Consolidator(client.store)
    now = now_ts()
    scored = [(cons.vitality(c, now=now), c) for c in cards]
    # HIGHEST vitality first = warmest memories (inverse of inspect's decay view).
    # scored is [(vitality_float, MemoryCard), ...] so item[0] is the score and
    # item[1] is the card; the primary key negates the score (DESC), tiebreakers
    # read off the card (item[1].timestamp, item[1].id).
    scored.sort(key=lambda item: (-item[0], item[1].timestamp, item[1].id))
    rows = [
        _vitality_row(c, now, v) for v, c in scored if v >= min_vitality
    ]
    rows = rows[:top] if top > 0 else rows

    if as_json:
        print(json.dumps(rows, indent=2))
        return 0

    print("=== isotope_zero active (highest vitality) ===")
    if not rows:
        print("  (no cards)")
        return 0
    print(
        f"  {'id':<12}  {'fact':<60}  {'vital':>6}  "
        f"{'acc':>4}  {'age_d':>7}"
    )
    for r in rows:
        print(
            f"  {_trunc(r['id'], 12):<12}  {_trunc(r['fact'], 60):<60}  "
            f"{r['vitality']:>6.4f}  {r['access_count']:>4}  "
            f"{r['age_days']:>7.1f}"
        )
    return 0


# ---------------------------------------------------------------------- #
# recall — semantic retrieval
# ---------------------------------------------------------------------- #
def _print_hits(title: str, hits: list[dict[str, Any]], now: float) -> None:
    """Shared aligned-column printer for recall/search results."""
    print(f"=== isotope_zero {title} ===")
    if not hits:
        print("  (no hits)")
        return
    print(
        f"  {'#':>2}  {'id':<12}  {'fact':<60}  "
        f"{'score':>7}  {'age_d':>7}"
    )
    for rank, h in enumerate(hits, 1):
        print(
            f"  {rank:>2}  {_trunc(h['id'], 12):<12}  "
            f"{_trunc(h['fact'], 60):<60}  {h['score']:>7.4f}  "
            f"{_age_days(h['timestamp'], now):>7.1f}"
        )


def _cmd_recall(client: IsotopeZero, query: str, k: int, alpha: float | None,
                as_json: bool) -> int:
    """Semantic recall: embed the query, return top-k by fused score."""
    hits = client.recall(query, k=k, alpha=alpha)
    now = now_ts()
    if as_json:
        out = [dict(h, age_days=_age_days(h["timestamp"], now)) for h in hits]
        print(json.dumps(out, indent=2))
        return 0
    _print_hits("recall", hits, now)
    return 0


# ---------------------------------------------------------------------- #
# search — hybrid (vector + BM25 + graph)
# ---------------------------------------------------------------------- #
def _cmd_search(client: IsotopeZero, query: str, k: int, alpha: float | None,
                as_json: bool) -> int:
    """Hybrid search; same output shape as recall."""
    hits = client.search(query, k=k, alpha=alpha)
    now = now_ts()
    if as_json:
        out = [dict(h, age_days=_age_days(h["timestamp"], now)) for h in hits]
        print(json.dumps(out, indent=2))
        return 0
    _print_hits("search", hits, now)
    return 0


# ---------------------------------------------------------------------- #
# list — all cards, newest first, optional tag filter
# ---------------------------------------------------------------------- #
def _cmd_list(client: IsotopeZero, tags: list[str], limit: int,
              as_json: bool) -> int:
    """List cards newest-first (timestamp desc), filtered by tags, capped at limit."""
    cards = client.store.all()
    cards = _filter_by_tags(cards, tags)
    cards.sort(key=lambda c: (-c.timestamp, c.id))
    if limit and limit > 0:
        cards = cards[:limit]
    now = now_ts()
    rows = [
        {
            "id": c.id,
            "fact": c.fact,
            "tags": list(c.tags),
            "timestamp": c.timestamp,
            "age_days": _age_days(c.timestamp, now),
        }
        for c in cards
    ]

    if as_json:
        print(json.dumps(rows, indent=2))
        return 0

    print("=== isotope_zero list (newest first) ===")
    if not rows:
        print("  (no cards)")
        return 0
    print(
        f"  {'id':<12}  {'fact':<60}  {'tags':<20}  {'age_d':>7}"
    )
    for r in rows:
        print(
            f"  {_trunc(r['id'], 12):<12}  {_trunc(r['fact'], 60):<60}  "
            f"{_trunc(','.join(r['tags']), 20):<20}  {r['age_days']:>7.1f}"
        )
    return 0


# ---------------------------------------------------------------------- #
# get — full detail for one card
# ---------------------------------------------------------------------- #
def _cmd_get(client: IsotopeZero, card_id: str, as_json: bool) -> int:
    """Print full detail for one card, or an error if it is missing."""
    card = client.store.get(card_id)
    if card is None:
        print(f"no card with id: {card_id}", file=sys.stderr)
        return 1
    cons = Consolidator(client.store)
    now = now_ts()
    vitality = cons.vitality(card, now=now)

    if as_json:
        payload = {
            "id": card.id,
            "fact": card.fact,
            "evidence": card.evidence,
            "tags": list(card.tags),
            "timestamp": card.timestamp,
            "last_access": card.last_access,
            "access_count": card.access_count,
            "vitality": round(vitality, 4),
            "source_tokens": card.source_tokens,
            "age_days": _age_days(card.timestamp, now),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"=== isotope_zero get {card.id} ===")
    print(f"id:            {card.id}")
    print(f"fact:          {card.fact}")
    print(f"evidence:      {card.evidence}")
    print(f"tags:          {', '.join(card.tags) if card.tags else '(none)'}")
    print(f"timestamp:     {card.timestamp}")
    print(f"last_access:   {card.last_access}")
    print(f"access_count:  {card.access_count}")
    print(f"vitality:      {vitality:.4f}")
    print(f"source_tokens: {card.source_tokens}")
    return 0


# ---------------------------------------------------------------------- #
# forget — delete one card
# ---------------------------------------------------------------------- #
def _cmd_forget(client: IsotopeZero, card_id: str, yes: bool) -> int:
    """Confirm (unless --yes), then delete one card."""
    if client.store.get(card_id) is None:
        print(f"no card with id: {card_id}", file=sys.stderr)
        return 1
    if not _confirm(f"forget {card_id}?", yes):
        print("aborted")
        return 1
    deleted = client.store.delete(card_id)
    if deleted:
        print(f"deleted {card_id}")
        return 0
    print(f"no card with id: {card_id}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------- #
# touch — bump access tracking on one card
# ---------------------------------------------------------------------- #
def _cmd_touch(client: IsotopeZero, card_id: str) -> int:
    """Record a recall on a card; print refreshed or not-found."""
    ok = client.touch(card_id)
    if ok:
        print(f"refreshed {card_id}")
        return 0
    print(f"no card with id: {card_id}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------- #
# tags — tag distribution
# ---------------------------------------------------------------------- #
def _cmd_tags(client: IsotopeZero, as_json: bool) -> int:
    """Aggregate all card tags into {tag: count}, sorted by count desc."""
    counts: dict[str, int] = {}
    for card in client.store.all():
        for tag in card.tags:
            counts[tag] = counts.get(tag, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    if as_json:
        print(json.dumps(dict(ordered), indent=2))
        return 0

    print("=== isotope_zero tags ===")
    if not ordered:
        print("  (no tags)")
        return 0
    print(f"  {'tag':<24}  {'count':>5}")
    for tag, n in ordered:
        print(f"  {_trunc(tag, 24):<24}  {n:>5}")
    return 0


# ---------------------------------------------------------------------- #
# stats — store overview
# ---------------------------------------------------------------------- #
def _cmd_stats(client: IsotopeZero, as_json: bool) -> int:
    """Count, DB size, embedding mode, token footprint, tags, vitality histogram."""
    cards = client.store.all()
    count = client.count()
    size_bytes = client.store.db_size_bytes()
    mode = _embedding_mode(client.store)
    tokens = sum(estimate_tokens(c.fact) + estimate_tokens(c.evidence) for c in cards)

    # Tag distribution (count desc, tag asc tiebreak).
    tag_counts: dict[str, int] = {}
    for card in cards:
        for tag in card.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    tag_dist = dict(sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    # Vitality histogram: fresh / aging / decayed buckets.
    cons = Consolidator(client.store)
    now = now_ts()
    fresh = aging = decayed = 0
    for card in cards:
        v = cons.vitality(card, now=now)
        if v >= 0.66:
            fresh += 1
        elif v >= 0.33:
            aging += 1
        else:
            decayed += 1
    histogram = {"fresh": fresh, "aging": aging, "decayed": decayed}
    histogram_desc = {
        "fresh": ">=0.66",
        "aging": "0.33-0.66",
        "decayed": "<0.33",
    }

    if as_json:
        payload = {
            "count": count,
            "db_size_bytes": size_bytes,
            "db_size_human": _human_bytes(size_bytes),
            "embedding_mode": mode,
            "token_footprint": tokens,
            "tags": tag_dist,
            "tag_distribution": tag_dist,
            "vitality_histogram": histogram,
            "vitality_buckets": histogram_desc,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("=== isotope_zero stats ===")
    print(f"count:             {count}")
    print(f"db size:           {size_bytes} bytes ({_human_bytes(size_bytes)})")
    print(f"embedding mode:    {mode}")
    print(f"token footprint:   {tokens} tokens (fact+evidence)")
    print()
    print("tag distribution:")
    if tag_dist:
        print(f"  {'tag':<24}  {'count':>5}")
        for tag, n in tag_dist.items():
            print(f"  {_trunc(tag, 24):<24}  {n:>5}")
    else:
        print("  (no tags)")
    print()
    print("vitality histogram:")
    print(f"  fresh (>=0.66):   {fresh}")
    print(f"  aging (0.33-0.66):{aging}")
    print(f"  decayed (<0.33):  {decayed}")
    return 0


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """argparse entry point. Returns a process exit code."""
    # Wire up diagnostic logging from ISOTOPE_ZERO_LOG_LEVEL before anything else,
    # so DEBUG records from triage/store/consolidation surface on stderr.
    configure_logging()

    parser = argparse.ArgumentParser(
        prog="isotope_zero",
        description=(
            "isotope_zero CLI: inspect, manage, and recall the memory store. "
            "Run a subcommand with -h for its flags."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = sub.add_parser(
        "inspect",
        help="human-readable store report (card count, size, decay, tokens)",
    )
    p_inspect.add_argument(
        "--db",
        default=None,
        help=(
            "DB path override. Default: ~/.isotope_zero/isotope_zero.db if it exists, "
            "else :memory: (an empty in-memory store)."
        ),
    )
    p_inspect.add_argument(
        "--json",
        action="store_true",
        help="emit the report as a JSON object instead of aligned text",
    )
    p_inspect.add_argument(
        "--top",
        type=int,
        default=5,
        help="number of decay candidates to show (default 5)",
    )

    # dry-run-consolidation
    p_dry = sub.add_parser(
        "dry-run-consolidation",
        help="preview a consolidation sweep; NOTHING is committed to the DB",
    )
    p_dry.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )
    p_dry.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "truncate proposed_merges and decay lists to N entries each "
            "(default 0 = no truncation)."
        ),
    )

    # add
    p_add = sub.add_parser(
        "add",
        help="remember a new fact (embeds it); prints the new card id",
    )
    p_add.add_argument("fact", help="the fact to remember")
    p_add.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )
    p_add.add_argument(
        "--evidence",
        default="",
        help="smallest quote justifying the fact (default empty)",
    )
    p_add.add_argument(
        "--tags",
        default=None,
        help="comma-separated tag list, e.g. 'ui,preference'",
    )
    p_add.add_argument(
        "--scope",
        default=None,
        help="multi-tier scope string stamped onto the card",
    )
    p_add.add_argument(
        "--id",
        dest="card_id",
        default=None,
        help="explicit card id (upsert); default is a fresh uuid4",
    )
    p_add.add_argument(
        "--json",
        action="store_true",
        help="emit the new card as a JSON object instead of just the id",
    )

    # active
    p_active = sub.add_parser(
        "active",
        help="warmest memories ranked by vitality (highest first)",
    )
    p_active.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )
    p_active.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of cards to show (default 10)",
    )
    p_active.add_argument(
        "--tags",
        default=None,
        help="comma-separated tag filter (intersect); e.g. 'ui,preference'",
    )
    p_active.add_argument(
        "--min-vitality",
        type=float,
        default=0.0,
        help="drop cards below this vitality floor (default 0.0 = keep all)",
    )
    p_active.add_argument(
        "--json",
        action="store_true",
        help="emit the rows as a JSON array instead of aligned text",
    )

    # recall
    p_recall = sub.add_parser(
        "recall",
        help="semantic retrieval: embed a query and return top-k hits",
    )
    p_recall.add_argument("query", help="the query to recall against")
    p_recall.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )
    p_recall.add_argument(
        "--k",
        type=int,
        default=5,
        help="number of hits to return (default 5)",
    )
    p_recall.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="cosine/retention fusion weight (default: client default 0.70)",
    )
    p_recall.add_argument(
        "--json",
        action="store_true",
        help="emit the hits as a JSON array instead of aligned text",
    )

    # search
    p_search = sub.add_parser(
        "search",
        help="hybrid search: semantic vector + BM25 + entity-graph boost",
    )
    p_search.add_argument("query", help="the query to search for")
    p_search.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )
    p_search.add_argument(
        "--k",
        type=int,
        default=5,
        help="number of hits to return (default 5)",
    )
    p_search.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="vector/BM25 fusion weight (default: client default 0.70)",
    )
    p_search.add_argument(
        "--json",
        action="store_true",
        help="emit the hits as a JSON array instead of aligned text",
    )

    # list
    p_list = sub.add_parser(
        "list",
        help="list cards newest-first, optionally filtered by tags",
    )
    p_list.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )
    p_list.add_argument(
        "--tags",
        default=None,
        help="comma-separated tag filter (intersect); e.g. 'ui,preference'",
    )
    p_list.add_argument(
        "--limit",
        type=int,
        default=50,
        help="max cards to show (default 50)",
    )
    p_list.add_argument(
        "--json",
        action="store_true",
        help="emit the rows as a JSON array instead of aligned text",
    )

    # get
    p_get = sub.add_parser(
        "get",
        help="print full detail for one card",
    )
    p_get.add_argument("id", help="card id to fetch")
    p_get.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )
    p_get.add_argument(
        "--json",
        action="store_true",
        help="emit the card as a JSON object instead of aligned text",
    )

    # forget
    p_forget = sub.add_parser(
        "forget",
        help="delete one card (prompts unless --yes)",
    )
    p_forget.add_argument("id", help="card id to delete")
    p_forget.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )
    p_forget.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt",
    )

    # touch
    p_touch = sub.add_parser(
        "touch",
        help="record a recall on a card (bump access tracking)",
    )
    p_touch.add_argument("id", help="card id to touch")
    p_touch.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )

    # tags
    p_tags = sub.add_parser(
        "tags",
        help="aggregate tag distribution across all cards",
    )
    p_tags.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )
    p_tags.add_argument(
        "--json",
        action="store_true",
        help="emit the distribution as a JSON object instead of aligned text",
    )

    # stats
    p_stats = sub.add_parser(
        "stats",
        help="store overview: count, size, mode, tokens, tags, vitality histogram",
    )
    p_stats.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )
    p_stats.add_argument(
        "--json",
        action="store_true",
        help="emit the overview as a JSON object instead of aligned text",
    )

    # dashboard — live read-only TUI overview of a store (auto-refresh).
    p_dash = sub.add_parser(
        "dashboard",
        help="live, auto-refreshing TUI overview of a store (read-only)",
    )
    p_dash.add_argument(
        "--db",
        default=None,
        help="DB path override (same default as inspect).",
    )
    p_dash.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="refresh interval in seconds (default 2.0).",
    )
    p_dash.add_argument(
        "--once",
        action="store_true",
        help="print one static frame and exit (scriptable; no live loop).",
    )

    args = parser.parse_args(argv)
    db_path = _resolve_db_path(args.db)

    if args.command == "inspect":
        store = _open_store(db_path)
        if store is None:
            print(f"DB path does not exist: {db_path}", file=sys.stderr)
            return 1
        try:
            return _cmd_inspect(store, args.top, args.json)
        finally:
            store.close()

    if args.command == "dry-run-consolidation":
        store = _open_store(db_path)
        if store is None:
            print(f"DB path does not exist: {db_path}", file=sys.stderr)
            return 1
        try:
            return _cmd_dry_run(store, args.limit)
        finally:
            store.close()

    if args.command == "dashboard":
        # Lazy import keeps `import isotope_zero.cli.debug` cheap (rich, if
        # present, is only pulled in when the dashboard actually runs).
        from .dashboard import run_dashboard

        return run_dashboard(db_path, args.interval, args.once)

    # add / active / recall / search / list / get / forget / touch / tags /
    # stats — all dispatch through the high-level IsotopeZero client via
    # _run_client, which owns the None-guard + try/finally close. recall and
    # search (and add) need the attached embedder; the rest don't call it but
    # pay only the cheap engine construction cost, so we use _open_client
    # everywhere for one consistent open/close path.
    if args.command == "add":
        return _run_client(
            db_path, _cmd_add, args.fact, args.evidence,
            _parse_tags(args.tags), args.scope, args.card_id, args.json,
        )
    if args.command == "active":
        return _run_client(
            db_path, _cmd_active, args.top, _parse_tags(args.tags),
            args.min_vitality, args.json,
        )
    if args.command == "recall":
        return _run_client(
            db_path, _cmd_recall, args.query, args.k, args.alpha, args.json,
        )
    if args.command == "search":
        return _run_client(
            db_path, _cmd_search, args.query, args.k, args.alpha, args.json,
        )
    if args.command == "list":
        return _run_client(
            db_path, _cmd_list, _parse_tags(args.tags), args.limit, args.json,
        )
    if args.command == "get":
        return _run_client(db_path, _cmd_get, args.id, args.json)
    if args.command == "forget":
        return _run_client(db_path, _cmd_forget, args.id, args.yes)
    if args.command == "touch":
        return _run_client(db_path, _cmd_touch, args.id)
    if args.command == "tags":
        return _run_client(db_path, _cmd_tags, args.json)
    if args.command == "stats":
        return _run_client(db_path, _cmd_stats, args.json)

    return 0


# ---------------------------------------------------------------------- #
# Inline smoke test
# ---------------------------------------------------------------------- #
def _smoke_test() -> None:
    """Seed a temp file DB with 3 cards (2 duplicates + 1 decayed) and run
    both subcommands to confirm non-empty output.

    Uses a temp file DB (not :memory:) so WAL engages, matching the
    production on-disk path. Two cards share an identical fact (exact-fact
    dedup path) and one card is 200 days old + never recalled (decay path),
    so the dry-run plan is non-empty without needing a real ONNX embedder.
    """
    import math

    def _norm(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v

    # Temp file DB so WAL mode engages (the in-memory path ignores the PRAGMA).
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name
    try:
        store = MemoryStore(db_path)
        t0 = now_ts()
        # dup-1 + dup-2: identical fact -> exact-fact dedup will fold dup-2.
        store.add(
            MemoryCard(
                id="dup-1",
                fact="The user prefers dark mode.",
                evidence="user said: 'I love dark mode'",
                timestamp=t0 - 100,
                tags=["preference", "ui"],
                embedding=_norm([1.0, 0.0, 0.0, 0.0]),
                source_tokens=6,
                access_count=2,
                last_access=t0 - 50,
            )
        )
        store.add(
            MemoryCard(
                id="dup-2",
                fact="The user prefers dark mode.",
                evidence="settings log: dark theme enabled",
                timestamp=t0 - 90,
                tags=["preference"],
                embedding=_norm([0.99, 0.01, 0.0, 0.0]),
                source_tokens=5,
                access_count=1,
                last_access=t0 - 40,
            )
        )
        # stale-1: 200 days old, never recalled -> vitality well below the
        # 0.05 floor, past the 1h grace period -> decay candidate.
        store.add(
            MemoryCard(
                id="stale-1",
                fact="An old throwaway note.",
                evidence="debug log line 42",
                timestamp=t0 - (200 * _SECS_PER_DAY),
                tags=["debug"],
                embedding=_norm([0.0, 0.0, 1.0, 0.0]),
                source_tokens=5,
                access_count=0,
                last_access=0.0,
            )
        )
        store.close()
    except Exception:
        try:
            os.unlink(db_path)
        finally:
            pass
        raise

    try:
        print("\n\n===== SMOKE: inspect =====")
        main(["inspect", "--db", db_path])
        print("\n\n===== SMOKE: dry-run-consolidation =====")
        main(["dry-run-consolidation", "--db", db_path])
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
