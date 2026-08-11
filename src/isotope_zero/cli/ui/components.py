"""isotope_zero CLI — Reusable UI components.

High-level components built on top of glyphs and theme for consistent,
beautiful output across all renderers (plain, rich, JSON).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.console import Console, RenderableType

from .glyphs import (
    glyph, build_vitality_bar, build_box, truncate,
    rich_ok, rich_bullet, rich_arrow, rich_selected, rich_unselected,
)
from .theme import IZERO_THEME, vitality_color, tag_color


# --------------------------------------------------------------------------- #
# Data Classes for Structured Output
# --------------------------------------------------------------------------- #
@dataclass
class MemoryRow:
    """A single memory card row for list/recall/search output."""
    rank: int
    id: str
    fact: str
    score: float | None = None
    vitality: float | None = None
    tags: Sequence[str] = ()
    age_days: float | None = None
    evidence: str = ""
    timestamp: float | None = None
    access_count: int = 0
    last_access: float | None = None
    source_tokens: int = 0
    scope: str = "default"
    created_at: str | None = None
    accessed_at: str | None = None


@dataclass
class StatRow:
    """A key-value stat row for stats/inspect output."""
    key: str
    value: str
    style: str = "izero.body"


@dataclass
class TagCount:
    """Tag with count for tag distribution."""
    tag: str
    count: int


@dataclass
class VitalityHistogram:
    """Vitality histogram data."""
    fresh: int = 0
    aging: int = 0
    decayed: int = 0


# --------------------------------------------------------------------------- #
# Plain Text Renderers (stdlib only, no rich)
# --------------------------------------------------------------------------- #
def render_memory_rows_plain(
    rows: Sequence[MemoryRow],
    show_score: bool = False,
    show_vitality: bool = False,
    show_tags: bool = True,
    show_id: bool = True,
    max_fact_width: int = 70,
) -> str:
    """Render memory rows as beautiful plain text (no rich)."""
    if not rows:
        return "No memories found."

    lines = []
    bullet = glyph("bullet")
    arrow = glyph("arrow")

    for row in rows:
        # Rank
        prefix = f"{row.rank}."

        # Fact (hero)
        fact = truncate(row.fact, max_fact_width)

        # Build suffix parts
        parts = [fact]

        if show_tags and row.tags:
            tags_str = ", ".join(row.tags)
            parts.append(f"{bullet} {tags_str}")
        elif show_id:
            parts.append(f"{bullet} {row.id[:6]}")

        if show_score and row.score is not None:
            parts.append(f"{arrow} {row.score:.4f}")

        if show_vitality and row.vitality is not None:
            bar = build_vitality_bar(
                1 if row.vitality >= 0.66 else 0,
                1 if 0.33 <= row.vitality < 0.66 else 0,
                1 if row.vitality < 0.33 else 0,
                width=10,
            )
            parts.append(f"{arrow} {bar} {row.vitality:.2f}")

        if row.age_days is not None:
            parts.append(f"{arrow} {row.age_days:.1f}d")

        lines.append(f"{prefix}  " + "  ".join(parts))

    return "\n".join(lines)


def render_memory_card_plain(card: MemoryRow, verbose: bool = False) -> str:
    """Render a single memory card in detail."""
    lines = [
        truncate(card.fact, width=100),
        f"evidence: {card.evidence if card.evidence else '(none)'}",
        f"tags:     {', '.join(card.tags) if card.tags else '(none)'}",
        f"id:       {card.id}",
    ]

    if verbose:
        if card.timestamp is not None:
            lines.append(f"timestamp:     {card.timestamp}")
        if card.last_access is not None:
            lines.append(f"last_access:   {card.last_access}")
        lines.append(f"access_count:  {card.access_count}")
        if card.vitality is not None:
            lines.append(f"vitality:      {card.vitality:.4f}")
        lines.append(f"source_tokens: {card.source_tokens}")
        if card.age_days is not None:
            lines.append(f"age_days:      {card.age_days:.1f}")

    return "\n".join(lines)


def render_stats_plain(
    count: int,
    size_bytes: int,
    embedding_mode: str,
    tokens: int,
    tag_dist: Sequence[TagCount] = (),
    histogram: VitalityHistogram | None = None,
    verbose: bool = False,
) -> str:
    """Render stats overview as plain text."""
    from ._fmtutil import human_bytes

    bullet = glyph("bullet")
    noun = "memory" if count == 1 else "memories"
    mode = embedding_mode.lower()

    line = f"{count} {noun} {bullet} {human_bytes(size_bytes)} {bullet} {mode} embeddings {bullet} ~{tokens} tokens"

    if not verbose:
        return line

    parts = [line, "", "tag distribution:"]
    if tag_dist:
        for tc in tag_dist:
            parts.append(f"  {tc.tag:<24}  {tc.count:>5}")
    else:
        parts.append("  (no tags)")

    if histogram:
        parts.append("")
        parts.append("vitality histogram:")
        parts.append(f"  fresh (>=0.66):    {histogram.fresh}")
        parts.append(f"  aging (0.33-0.66): {histogram.aging}")
        parts.append(f"  decayed (<0.33):   {histogram.decayed}")

    return "\n".join(parts)


def render_tags_plain(tag_counts: Sequence[TagCount], verbose: bool = False) -> str:
    """Render tag distribution as plain text."""
    if not tag_counts:
        return "No tags yet."

    if verbose:
        lines = ["  tag                     count"]
        for tc in tag_counts:
            lines.append(f"  {tc.tag:<24}  {tc.count:>5}")
        return "\n".join(lines)

    bullet = glyph("bullet")
    parts = [f"{tc.tag} ({tc.count})" for tc in tag_counts]
    return f" {bullet} ".join(parts)


def render_inspect_plain(
    db_path: str,
    total: int,
    size_bytes: int,
    embedding_mode: str,
    avg_dim: float | None,
    cards_with_emb: int,
    decay_rows: list[dict],
    tokens: int,
) -> str:
    """Render inspect output as plain text."""
    from ._fmtutil import human_bytes, trunc

    lines = [
        "=== isotope_zero inspect ===",
        f"store path:        {db_path}",
        f"total cards:       {total}",
        f"db size:           {size_bytes} bytes ({human_bytes(size_bytes)})",
        f"embedding mode:    {embedding_mode}",
        "",
    ]

    if avg_dim is not None:
        lines.append(f"avg embedding dim: {avg_dim:.1f} (across {cards_with_emb} cards with embeddings)")
    else:
        lines.append("no embeddings")

    lines.append("")
    lines.append(f"top {len(decay_rows)} decay candidates (lowest vitality):")

    if decay_rows:
        lines.append(f"  {'id':<12}  {'fact':<60}  {'vital':>6}  {'acc':>4}  {'age_d':>7}")
        for r in decay_rows:
            lines.append(
                f"  {trunc(r['id'], 12):<12}  {trunc(r['fact'], 60):<60}  "
                f"{r['vitality']:>6.4f}  {r['access_count']:>4}  {r['age_days']:>7.1f}"
            )
    else:
        lines.append("  (no cards)")

    lines.append("")
    lines.append(f"token footprint:   {tokens} tokens (fact+evidence across all cards)")

    return "\n".join(lines)


def render_dashboard_plain(state: dict[str, Any], interval: float) -> str:
    """Render dashboard frame as plain text (used by both transports)."""
    # This mirrors the existing dashboard._render_frame but with new glyphs/theme
    hist = state["histogram"]
    f_, a_, d_ = hist["fresh"], hist["aging"], hist["decayed"]
    bar = build_vitality_bar(f_, a_, d_) if state["count"] else glyph("bar_empty") * 20

    title_src = state["db_path"] or ":memory:"
    title = f"isotope_zero · {title_src}"

    lines = []
    lines.append(f"cards: {state['count']:<6} size: {state['size_human']:<9} mode: {state['mode']}")
    lines.append(f"vitality: {bar}  fresh {f_} / aging {a_} / decay {d_}")
    tokens_line = f"tokens: {state['tokens_total']} total"
    if state["reclaimable_tokens"] > 0:
        tokens_line += f"  ·  ~{state['reclaimable_tokens']} reclaimable"
    lines.append(tokens_line)

    if state["recent"]:
        lines.append("recent:")
        for r in state["recent"]:
            lines.append(f"  {glyph('bullet')} {truncate(r['fact'], 54)}  ({r['age_days']}d)")
    else:
        lines.append("recent: (none)")

    decay = state["decay"]
    n_decay = len(decay)
    if n_decay:
        lines.append(f"decay candidates ({n_decay}):")
        for c in decay:
            lines.append(f"  {glyph('bullet')} {truncate(c['fact'], 50)}  v={c['vitality']:.2f} age={c['age_days']}d")
    else:
        lines.append("decay candidates: none")

    footer = f"refresh {interval:g}s  ·  [Ctrl-C to exit]"

    return build_box(title, lines, footer, heavy=True)


# --------------------------------------------------------------------------- #
# Rich Renderers (beautiful TUI output)
# --------------------------------------------------------------------------- #
def rich_memory_rows(
    rows: Sequence[MemoryRow],
    show_score: bool = False,
    show_vitality: bool = False,
    show_tags: bool = True,
    console: Console | None = None,
) -> Table:
    """Create a rich Table for memory rows."""
    table = Table(
        show_header=False,
        box=None,
        pad_edge=False,
        collapse_padding=True,
        style="izero.body",
    )

    # Columns: rank, fact, tags/id, score, vitality, age
    table.add_column("rank", style="izero.number", justify="right", width=3)
    table.add_column("fact", style="izero.hero", ratio=1, overflow="ellipsis")
    table.add_column("meta", style="izero.dim", width=40)
    if show_score:
        table.add_column("score", style="izero.accent", justify="right", width=8)
    if show_vitality:
        table.add_column("vitality", style="izero.dim", justify="center", width=12)
    table.add_column("age", style="izero.timestamp", justify="right", width=8)

    for row in rows:
        # Meta column: tags or short id
        if show_tags and row.tags:
            # Color each tag differently
            meta_parts = []
            for i, tag in enumerate(row.tags):
                meta_parts.append(f"[{tag_color(i)}]{tag}[/]")
            meta = " " + glyph("bullet") + " " + ", ".join(meta_parts)
        else:
            meta = " " + glyph("bullet") + " " + f"[izero.id]{row.id[:6]}[/]"

        cells = [
            f"{row.rank}.",
            row.fact,
            meta,
        ]
        if show_score and row.score is not None:
            cells.append(f"{row.score:.4f}")
        if show_vitality and row.vitality is not None:
            # Build vitality bar as rich segments
            bar_segments = build_vitality_bar(
                1 if row.vitality >= 0.66 else 0,
                1 if 0.33 <= row.vitality < 0.66 else 0,
                1 if row.vitality < 0.33 else 0,
                width=10,
                use_rich=True,
            )
            bar_text = Text.assemble(*bar_segments)
            cells.append(bar_text)
        if row.age_days is not None:
            cells.append(f"{row.age_days:.1f}d")

        table.add_row(*cells)

    return table


def rich_memory_card(card: MemoryRow, verbose: bool = False) -> Panel:
    """Create a rich Panel for a single memory card."""
    lines = [
        f"[izero.hero]{card.fact}[/]",
        f"[izero.evidence]evidence:[/] {card.evidence if card.evidence else '[izero.dim](none)[/]'}",
        f"[izero.evidence]tags:[/]     {', '.join(f'[izero.tag]{t}[/]' for t in card.tags) if card.tags else '[izero.dim](none)[/]'}",
        f"[izero.evidence]id:[/]       [izero.id]{card.id}[/]",
    ]

    if verbose:
        if card.timestamp is not None:
            lines.append(f"[izero.evidence]timestamp:[/]     {card.timestamp}")
        if card.last_access is not None:
            lines.append(f"[izero.evidence]last_access:[/]   {card.last_access}")
        lines.append(f"[izero.evidence]access_count:[/]  {card.access_count}")
        if card.vitality is not None:
            style = vitality_style(card.vitality)
            bar = build_vitality_bar(
                1 if card.vitality >= 0.66 else 0,
                1 if 0.33 <= card.vitality < 0.66 else 0,
                1 if card.vitality < 0.33 else 0,
                width=20,
                use_rich=True,
            )
            bar_text = Text.assemble(*bar)
            lines.append(f"[izero.evidence]vitality:[/]      {bar_text} [izero.id]{card.vitality:.4f}[/]")
        lines.append(f"[izero.evidence]source_tokens:[/] {card.source_tokens}")
        if card.age_days is not None:
            lines.append(f"[izero.evidence]age_days:[/]      {card.age_days:.1f}")

    content = "\n".join(lines)
    return Panel(
        content,
        title="[izero.panel.title]memory[/]",
        border_style="izero.panel.border",
        style="izero.panel",
        padding=(1, 2),
    )


def rich_stats(
    count: int,
    size_bytes: int,
    embedding_mode: str,
    tokens: int,
    tag_dist: Sequence[TagCount] = (),
    histogram: VitalityHistogram | None = None,
    verbose: bool = False,
) -> RenderableType:
    """Create rich renderable for stats."""
    from ._fmtutil import human_bytes
    from rich.columns import Columns
    from rich.text import Text

    bullet = Text(glyph("bullet"), style="izero.glyph.bullet")
    noun = "memory" if count == 1 else "memories"
    mode = embedding_mode.lower()

    header = Text.assemble(
        (f"{count} ", "izero.number"),
        (noun, "izero.hero"),
        (" ", "izero.body"),
        bullet,
        (" ", "izero.body"),
        (human_bytes(size_bytes), "izero.accent"),
        (" ", "izero.body"),
        bullet,
        (" ", "izero.body"),
        (f"{mode} embeddings", "izero.info"),
        (" ", "izero.body"),
        bullet,
        (" ", "izero.body"),
        (f"~{tokens} tokens", "izero.accent"),
    )

    if not verbose:
        return header

    # Verbose: add tag distribution and histogram
    parts = [header, Text("")]

    # Tag distribution table
    if tag_dist:
        tag_table = Table(show_header=False, box=None, pad_edge=False, style="izero.dim")
        tag_table.add_column("tag", style="izero.tag", width=24)
        tag_table.add_column("count", style="izero.number", justify="right", width=8)
        for tc in tag_dist:
            tag_table.add_row(tc.tag, str(tc.count))
        parts.append(Text("tag distribution:", style="izero.subtitle"))
        parts.append(tag_table)
    else:
        parts.append(Text("tag distribution: (no tags)", style="izero.dim"))

    # Vitality histogram
    if histogram:
        hist_table = Table(show_header=False, box=None, pad_edge=False, style="izero.dim")
        hist_table.add_column("bucket", style="izero.dim", width=20)
        hist_table.add_column("count", style="izero.number", justify="right", width=8)
        # Fresh
        bar_fresh = Text.assemble(*build_vitality_bar(histogram.fresh, 0, 0, width=10, use_rich=True))
        hist_table.add_row(Text.assemble("fresh (>=0.66) ", bar_fresh), str(histogram.fresh))
        # Aging
        bar_aging = Text.assemble(*build_vitality_bar(0, histogram.aging, 0, width=10, use_rich=True))
        hist_table.add_row(Text.assemble("aging (0.33-0.66) ", bar_aging), str(histogram.aging))
        # Decayed
        bar_decay = Text.assemble(*build_vitality_bar(0, 0, histogram.decayed, width=10, use_rich=True))
        hist_table.add_row(Text.assemble("decayed (<0.33) ", bar_decay), str(histogram.decayed))

        parts.append(Text(""))
        parts.append(Text("vitality histogram:", style="izero.subtitle"))
        parts.append(hist_table)

    from rich.console import Group
    return Group(*parts)


def rich_dashboard(state: dict[str, Any], interval: float) -> Panel:
    """Create a rich Panel for the dashboard."""
    hist = state["histogram"]
    f_, a_, d_ = hist["fresh"], hist["aging"], hist["decayed"]

    # Build vitality bar
    bar_segments = build_vitality_bar(f_, a_, d_, width=20, use_rich=True)
    bar_text = Text.assemble(*bar_segments)

    lines = [
        Text.assemble(
            ("cards: ", "izero.dim"),
            (f"{state['count']:<6}", "izero.number"),
            (" size: ", "izero.dim"),
            (f"{state['size_human']:<9}", "izero.accent"),
            (" mode: ", "izero.dim"),
            (state['mode'], "izero.info"),
        ),
        Text.assemble(
            ("vitality: ", "izero.dim"),
            bar_text,
            ("  fresh ", "izero.dim"),
            (str(f_), "izero.bar.fresh"),
            (" / aging ", "izero.dim"),
            (str(a_), "izero.bar.aging"),
            (" / decay ", "izero.dim"),
            (str(d_), "izero.bar.decay"),
        ),
    ]

    tokens_line = Text.assemble(
        ("tokens: ", "izero.dim"),
        (str(state['tokens_total']), "izero.number"),
        (" total", "izero.dim"),
    )
    if state["reclaimable_tokens"] > 0:
        tokens_line.append("  ·  ~")
        tokens_line.append(str(state['reclaimable_tokens']), "izero.warning")
        tokens_line.append(" reclaimable", "izero.dim")
    lines.append(tokens_line)

    # Recent
    if state["recent"]:
        lines.append(Text("recent:", style="izero.subtitle"))
        for r in state["recent"]:
            lines.append(Text.assemble(
                ("  ", "izero.body"),
                (glyph("bullet") + " ", "izero.glyph.bullet"),
                (truncate(r['fact'], 54), "izero.hero"),
                ("  (", "izero.dim"),
                (f"{r['age_days']}d", "izero.timestamp"),
                (")", "izero.dim"),
            ))
    else:
        lines.append(Text.assemble(("recent: ", "izero.dim"), ("(none)", "izero.dim")))

    # Decay
    decay = state["decay"]
    n_decay = len(decay)
    if n_decay:
        lines.append(Text.assemble(
            ("decay candidates (", "izero.dim"),
            (str(n_decay), "izero.number"),
            (")", "izero.dim"),
        ))
        for c in decay:
            style = vitality_style(c['vitality'])
            lines.append(Text.assemble(
                ("  ", "izero.body"),
                (glyph("bullet") + " ", "izero.glyph.bullet"),
                (truncate(c['fact'], 50), "izero.hero"),
                ("  v=", "izero.dim"),
                (f"{c['vitality']:.2f}", style),
                (" age=", "izero.dim"),
                (f"{c['age_days']}d", "izero.timestamp"),
            ))
    else:
        lines.append(Text.assemble(("decay candidates: ", "izero.dim"), ("none", "izero.dim")))

    footer = Text.assemble(
        ("refresh ", "izero.hint"),
        (f"{interval:g}s", "izero.number"),
        ("  ·  ", "izero.hint"),
        ("[Ctrl-C to exit]", "izero.hint"),
        ("  ·  ", "izero.hint"),
        ("[?] help", "izero.hint"),
    )

    from rich.console import Group
    content = Group(*lines)

    return Panel(
        content,
        title=Text.assemble(("isotope_zero · ", "izero.panel.title"), (state["db_path"] or ":memory:", "izero.accent")),
        subtitle=footer,
        border_style="izero.panel.border",
        style="izero.panel",
        padding=(1, 2),
    )


def rich_menu(banner: str, entries: list[tuple[str, Any]], selected_idx: int) -> Panel:
    """Create a rich Panel for the interactive menu."""
    from rich.console import Group

    lines = [Text(banner, style="izero.panel.title"), Text("")]
    lines.append(Text("What would you like to do?", style="izero.subtitle"))
    lines.append(Text(""))

    for i, (label, _) in enumerate(entries):
        if i == selected_idx:
            prefix = Text.assemble((glyph("selected") + " ", "izero.selected"))
            label_text = Text(label, style="izero.hero")
        else:
            prefix = Text.assemble((glyph("unselected") + " ", "izero.dim"))
            label_text = Text(label, style="izero.body")
        lines.append(Text.assemble(prefix, label_text))

    lines.append(Text(""))
    lines.append(Text.assemble(
        ("↑↓ navigate · ", "izero.hint"),
        ("enter to run · ", "izero.hint"),
        ("q to quit · ", "izero.hint"),
        ("/ to search", "izero.hint"),
    ))

    from rich.console import Group
    content = Group(*lines)

    return Panel(
        content,
        border_style="izero.panel.border",
        style="izero.panel",
        padding=(1, 2),
    )


# --------------------------------------------------------------------------- #
# JSON Renderers (machine contract - unchanged)
# --------------------------------------------------------------------------- #
def render_json(obj: Any) -> str:
    """Render any object as pretty JSON."""
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Action Confirmation Renderers
# --------------------------------------------------------------------------- #
def render_add_result(card_id: str, created: bool, as_json: bool = False) -> str:
    """Render add command result."""
    if as_json:
        return render_json({"id": card_id, "created": created})

    verb = "remembered" if created else "updated"
    return f"{glyph('ok')} {verb}\nid: {card_id}"


def render_forget_result(card_id: str, as_json: bool = False) -> str:
    """Render forget command result."""
    if as_json:
        return render_json({"id": card_id, "deleted": True})
    return f"{glyph('ok')} forgot"


def render_touch_result(card_id: str, as_json: bool = False) -> str:
    """Render touch command result."""
    if as_json:
        return render_json({"id": card_id, "refreshed": True})
    return f"{glyph('ok')} refreshed"