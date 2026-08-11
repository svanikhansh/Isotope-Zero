"""isotope_zero CLI — Legacy formatting functions for debug.py compatibility.

This module provides the plain-text formatting functions used by the original
argparse-based CLI (debug.py). These are kept separate from the new renderer
system to maintain backward compatibility with the CLI SPEC tests.
"""

from __future__ import annotations

import sys
from typing import Any, Sequence

# Display helpers from a stdlib-only local module — NOT from .debug. Importing
# from .debug at module top would transitively pull the whole embedding/engine
# stack into render.py's import graph, violating the "stdlib-only, zero new
# core deps" philosophy of this module. _fmtutil is dependency-free, so render
# stays cheap to import and genuinely decoupled from the engine.
from .._fmtutil import age_days as _age_days
from .._fmtutil import human_bytes as _human_bytes
from .._fmtutil import trunc as _trunc

# Display widths for the --verbose (technical) columns — kept identical to the
# pre-redesign aligned tables so ``--verbose`` is a faithful "old behavior" restore.
_ID_W = 12
_FACT_W = 60
_TAG_W = 24


# --------------------------------------------------------------------------- #
# Glyph resolution — Unicode marks with ASCII fallback for non-UTF-8 stdout.
# --------------------------------------------------------------------------- #
def _glyph(name: str) -> str:
    """Return a Unicode glyph, falling back to ASCII if stdout can't encode it.

    ``name`` is one of: ``ok``, ``bullet``, ``arrow``. The fallback keeps the CLI
    from crashing on Windows cp1252 / ``LC_ALL=C`` pipes where ``✓`` isn't encodable.
    """
    table = {"ok": "✓", "bullet": "·", "arrow": "→"}
    preferred = table[name]
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        preferred.encode(enc)
        return preferred
    except (UnicodeEncodeError, LookupError):
        return {"ok": "ok", "bullet": "-", "arrow": "->"}[name]


def _short_id(full_id: str) -> str:
    """First 6 chars of an id — enough to disambiguate and copy back into ``get``.

    The full id is always available via ``--json`` or ``get <id>``; the short form
    is just a visual hint in list/recall/search so a user can see distinctness
    without a uuid column dominating the line.
    """
    return full_id[:6]


def _wrap_fact(fact: str, width: int = 70) -> str:
    """Truncate a fact to ``width`` chars with an ellipsis (display only).

    JSON keeps full values; this is purely so a long fact doesn't blow out the
    one-line-per-memory layout. Mirrors ``debug._trunc`` but adds the ellipsis.
    """
    fact = str(fact).replace("\n", " ").strip()
    return fact if len(fact) <= width else fact[: width - 1] + "…"


# --------------------------------------------------------------------------- #
# Ranked hits — recall / search / active share this shape.
# --------------------------------------------------------------------------- #
def format_hits(
    hits: Sequence[dict[str, Any]],
    now: float,
    verbose: bool = False,
) -> str:
    """Numbered list of hits, fact-first.

    Default (Mem0-style)::

        1.  The user prefers dark mode for UI.            (a1b2c3)
        2.  Python is the user's favorite language.       (d4e5f6)

    ``verbose=True`` restores the pre-redesign technical columns (rank, short id,
    score, age) — a faithful "old behavior" restore for power users / scripts that
    relied on the aligned table (though ``--json`` is the stable script contract).
    """
    if not hits:
        return "No memories found."

    if verbose:
        lines = [
            f"  {'#':>2}  {'id':<{_ID_W}}  {'fact':<{_FACT_W}}  "
            f"{'score':>7}  {'age_d':>7}"
        ]
        for rank, h in enumerate(hits, 1):
            lines.append(
                f"  {rank:>2}  {_trunc(h['id'], _ID_W):<{_ID_W}}  "
                f"{_trunc(h['fact'], _FACT_W):<{_FACT_W}}  "
                f"{h['score']:>7.4f}  {_age_days(h['timestamp'], now):>7.1f}"
            )
        return "\n".join(lines)

    bullet = _glyph("bullet")
    out = []
    for rank, h in enumerate(hits, 1):
        out.append(f"{rank}.  {_wrap_fact(h['fact'])}  {bullet} {_short_id(h['id'])}")
    return "\n".join(out)


def format_list_rows(
    rows: Sequence[dict[str, Any]],
    now: float,
    verbose: bool = False,
    filtered: bool = False,
) -> str:
    """Numbered list of cards, newest first, fact-first.

    Default shows tags inline only when present::

        1.  <fact>  · ui, preference
        2.  <fact>                 (a1b2c3)

    ``verbose=True`` restores the aligned id/fact/tags/age table.

    ``filtered=True`` means a tag filter was applied. An empty result then reads
    "No memories found." (the filter matched nothing) rather than "No memories
    yet." (which implies a truly empty store and would mislead a user whose store
    has cards but none bearing the requested tag).
    """
    if not rows:
        return "No memories found." if filtered else "No memories yet."

    if verbose:
        lines = [
            f"  {'id':<{_ID_W}}  {'fact':<{_FACT_W}}  "
            f"{'tags':<{_TAG_W}}  {'age_d':>7}"
        ]
        for r in rows:
            lines.append(
                f"  {_trunc(r['id'], _ID_W):<{_ID_W}}  "
                f"{_trunc(r['fact'], _FACT_W):<{_FACT_W}}  "
                f"{_trunc(','.join(r['tags']), _TAG_W):<{_TAG_W}}  "
                f"{r['age_days']:>7.1f}"
            )
        return "\n".join(lines)

    bullet = _glyph("bullet")
    out = []
    for rank, r in enumerate(rows, 1):
        tags = ", ".join(r["tags"])
        suffix = f"  {bullet} {tags}" if tags else f"  {bullet} {_short_id(r['id'])}"
        out.append(f"{rank}.  {_wrap_fact(r['fact'])}{suffix}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Single-card views.
# --------------------------------------------------------------------------- #
def format_card(
    card_id: str,
    fact: str,
    evidence: str,
    tags: Sequence[str],
    timestamp: float,
    last_access: float,
    access_count: int,
    vitality: float,
    source_tokens: int,
    now: float,
    verbose: bool = False,
) -> str:
    """Full detail for one card, fact as the headline.

    Default is a small block (fact, evidence, tags, id). ``verbose=True``
    appends the technical fields (timestamp, last_access, access_count,
    vitality, source_tokens, age) that the pre-redesign ``get`` always printed.

    Evidence is rendered in the *default* view (not just verbose) because it is
    content the user stored — the justifying quote behind the fact — not chrome
    like score floats or uuid columns. Dropping it would be a regression: the
    pre-redesign ``get`` always printed ``evidence:``, and ``--json`` still
    returns it. The hero-line goal is about removing *decoration*, not content.
    """
    lines = [
        _wrap_fact(fact, width=100),
        f"evidence: {evidence if evidence else '(none)'}",
        f"tags:     {', '.join(tags) if tags else '(none)'}",
        f"id:       {card_id}",
    ]
    if verbose:
        lines.append(f"timestamp:     {timestamp}")
        lines.append(f"last_access:   {last_access}")
        lines.append(f"access_count:  {access_count}")
        lines.append(f"vitality:      {vitality:.4f}")
        lines.append(f"source_tokens: {source_tokens}")
        lines.append(f"age_days:      {_age_days(timestamp, now):.1f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Action confirmations — add / forget / touch.
# --------------------------------------------------------------------------- #
def format_add(card_id: str, created: bool) -> str:
    """``✓ remembered`` on line 1; the id on its own trailing line (line 2).

    The trailing ``id: <uuid>`` line is what ``test_cli._id_from_plain`` parses —
    it takes the last non-blank line's last whitespace-separated token, so this
    two-line shape stays parseable while the human-visible hero line is the
    friendly ``✓ remembered`` (not a bare uuid).
    """
    ok = _glyph("ok")
    verb = "remembered" if created else "updated"
    return f"{ok} {verb}\nid: {card_id}"


def format_forget(card_id: str) -> str:
    """``✓ forgot`` (the id is already known — the user passed it)."""
    return f"{_glyph('ok')} forgot"


def format_touch(card_id: str) -> str:
    """``✓ refreshed``."""
    return f"{_glyph('ok')} refreshed"


# --------------------------------------------------------------------------- #
# Aggregates — tags / stats.
# --------------------------------------------------------------------------- #
def format_tags(tag_counts: dict[str, int], verbose: bool = False) -> str:
    """Tags as a wrapped ``tag (n)  ·  tag (n)`` line, count-desc.

    Empty → ``No tags yet.`` The pre-redesign aligned ``tag / count`` table is
    unnecessary for ≤ a few dozen tags and reads as spreadsheet, not prose, so
    it lives behind ``verbose=True`` — the same clean/verbose split the other
    table-bearing commands (list, recall, search, stats) use. Verbose restores
    the aligned ``tag / count`` table, count-desc, for power users.
    """
    if not tag_counts:
        return "No tags yet."
    if verbose:
        lines = [f"  {'tag':<{_TAG_W}}  {'count':>5}"]
        for tag, n in tag_counts.items():
            lines.append(f"  {_trunc(tag, _TAG_W):<{_TAG_W}}  {n:>5}")
        return "\n".join(lines)
    bullet = _glyph("bullet")
    parts = [f"{tag} ({n})" for tag, n in tag_counts.items()]
    return f" {bullet} ".join(parts)


def format_stats(
    count: int,
    size_bytes: int,
    embedding_mode: str,
    tokens: int,
    tag_dist: dict[str, int],
    histogram: dict[str, int],
    verbose: bool = False,
) -> str:
    """Short prose summary line; technical detail behind ``--verbose``.

    Default::

        12 memories · 4.2 KB · fallback embeddings · ~280 tokens

    ``verbose=True`` appends the tag-distribution table + vitality histogram the
    pre-redesign ``stats`` always printed.
    """
    bullet = _glyph("bullet")
    noun = "memory" if count == 1 else "memories"
    mode = embedding_mode.lower()
    line = (
        f"{count} {noun} {bullet} {_human_bytes(size_bytes)} {bullet} "
        f"{mode} embeddings {bullet} ~{tokens} tokens"
    )
    if not verbose:
        return line

    parts = [line, "", "tag distribution:"]
    if tag_dist:
        for tag, n in tag_dist.items():
            parts.append(f"  {_trunc(tag, _TAG_W):<{_TAG_W}}  {n:>5}")
    else:
        parts.append("  (no tags)")
    parts.append("")
    parts.append("vitality histogram:")
    parts.append(f"  fresh (>=0.66):    {histogram.get('fresh', 0)}")
    parts.append(f"  aging (0.33-0.66): {histogram.get('aging', 0)}")
    parts.append(f"  decayed (<0.33):   {histogram.get('decayed', 0)}")
    return "\n".join(parts)


__all__ = [
    "format_hits",
    "format_list_rows",
    "format_card",
    "format_add",
    "format_forget",
    "format_touch",
    "format_tags",
    "format_stats",
]