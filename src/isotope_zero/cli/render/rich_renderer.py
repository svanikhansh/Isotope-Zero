"""isotope_zero CLI — Rich Renderer.

Full-featured TUI output using rich. Supports panels, tables, live updates,
animations, and the complete black/beige/white theme.
"""

from __future__ import annotations

from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align

from .base import BaseRenderer
from ..ui.components import (
    MemoryRow,
    StatRow,
    TagCount,
    VitalityHistogram,
)

from ..ui.theme import get_console, get_theme, tag_color, vitality_style
from ..ui.glyphs import (
    glyph,
    rich_ok,
    rich_bullet,
    rich_arrow,
    rich_selected,
    rich_unselected,
    build_vitality_bar,
    box_top,
    box_bottom,
    box_line,
    build_box,
)


class RichRenderer(BaseRenderer):
    """Rich TUI renderer — full featured with theme, panels, tables."""

    def __init__(self, use_color: bool = True, verbose: bool = False):
        super().__init__(use_color=use_color, verbose=verbose)
        self.console = get_console()
        self.theme = get_theme()

    def _styled(self, text: str, style: str) -> Text:
        """Create a styled Text object."""
        return Text(text, style=style)

    def render_memory_rows(
        self,
        rows: Sequence[MemoryRow],
        show_score: bool = False,
        show_vitality: bool = False,
        show_tags: bool = True,
    ) -> str:
        if not rows:
            return self._styled("(no memories)", "izero.dim").plain

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("#", style="izero.dim", width=3)
        table.add_column("fact", style="izero.body", ratio=1)
        if show_tags:
            table.add_column("tags", style="izero.tag", width=30)
        if show_score:
            table.add_column("score", style="izero.accent", width=8)
        if show_vitality:
            table.add_column("vitality", width=22)

        for i, row in enumerate(rows, 1):
            cells = [str(i), row.fact]

            if show_tags:
                tags_str = ", ".join(row.tags) if row.tags else ""
                cells.append(tags_str)

            if show_score:
                cells.append(f"{row.score:.4f}" if row.score is not None else "")

            if show_vitality:
                if row.vitality is not None:
                    bar = build_vitality_bar(
                        1 if row.vitality >= 0.66 else 0,
                        1 if 0.33 <= row.vitality < 0.66 else 0,
                        1 if row.vitality < 0.33 else 0,
                        width=16,
                        use_rich=True,
                    )
                    vitality_text = Text()
                    for segment_text, segment_style in bar:
                        vitality_text.append(segment_text, style=segment_style)
                    cells.append(vitality_text)
                else:
                    cells.append("")

            table.add_row(*cells)

        # Capture output
        with self.console.capture() as capture:
            self.console.print(table)
        return capture.get()

    def render_memory_card(self, card: MemoryRow, verbose: bool = False) -> str:
        content = []

        # Fact as hero
        content.append(self._styled(card.fact, "izero.body"))
        content.append("")

        # Evidence
        if card.evidence:
            content.append(self._styled("evidence:", "izero.dim"))
            content.append(self._styled(f"  {card.evidence}", "izero.body"))
            content.append("")

        # Tags
        if card.tags:
            content.append(self._styled("tags:", "izero.dim"))
            tags_rich = Text()
            for i, tag in enumerate(card.tags):
                if i > 0:
                    tags_rich.append(", ")
                tags_rich.append(tag, style=f"izero.tag")
            content.append(self._styled("  ", "izero.body") + tags_rich)
            content.append("")

        # ID
        content.append(self._styled(f"id: {card.id}", "izero.id"))

        # Verbose details
        if verbose:
            content.append("")
            if card.score is not None:
                content.append(self._styled(f"score: {card.score:.4f}", "izero.accent"))
            if card.vitality is not None:
                bar = build_vitality_bar(
                    1 if card.vitality >= 0.66 else 0,
                    1 if 0.33 <= card.vitality < 0.66 else 0,
                    1 if card.vitality < 0.33 else 0,
                    width=20,
                    use_rich=True,
                )
                vitality_text = Text()
                for segment_text, segment_style in bar:
                    vitality_text.append(segment_text, style=segment_style)
                vitality_text.append(f" {card.vitality:.2f}", style=vitality_style(card.vitality))
                content.append(Text("vitality: ") + vitality_text)
            if card.created_at:
                content.append(self._styled(f"created: {card.created_at}", "izero.timestamp"))
            if card.accessed_at:
                content.append(self._styled(f"accessed: {card.accessed_at}", "izero.timestamp"))
            if card.access_count is not None:
                content.append(self._styled(f"accesses: {card.access_count}", "izero.number"))

        with self.console.capture() as capture:
            for c in content:
                self.console.print(c)
        return capture.get()

    def render_stats(
        self,
        count: int,
        size_bytes: int,
        embedding_mode: str,
        tokens: int,
        tag_dist: Sequence[TagCount] = (),
        histogram: VitalityHistogram | None = None,
        verbose: bool = False,
    ) -> str:
        with self.console.capture() as capture:
            # Hero line
            size_kb = size_bytes / 1024
            hero = f"{count} memories  ·  {size_kb:.1f} KB  ·  {embedding_mode}  ·  ~{tokens} tokens"
            self.console.print(self._styled(hero, "izero.hero"))

            if verbose:
                # Tags
                if tag_dist:
                    self.console.print("")
                    self.console.print(self._styled("tag distribution:", "izero.dim"))
                    tag_table = Table(show_header=False, box=None, padding=(0, 1))
                    tag_table.add_column("tag", style="izero.tag")
                    tag_table.add_column("count", style="izero.number", justify="right")
                    for tc in tag_dist:
                        tag_table.add_row(tc.tag, str(tc.count))
                    self.console.print(tag_table)

                # Vitality histogram
                if histogram:
                    self.console.print("")
                    self.console.print(self._styled("vitality histogram:", "izero.dim"))
                    vit_table = Table(show_header=False, box=None, padding=(0, 1))
                    vit_table.add_column("category", style="izero.body")
                    vit_table.add_column("bar", width=22)
                    vit_table.add_column("count", style="izero.number", justify="right")

                    if histogram.fresh:
                        bar = build_vitality_bar(histogram.fresh, 0, 0, width=16, use_rich=True)
                        vit_table.add_row("fresh (≥0.66)", Text("").join([Text(s, style=st) for s, st in bar]), str(histogram.fresh))
                    if histogram.aging:
                        bar = build_vitality_bar(0, histogram.aging, 0, width=16, use_rich=True)
                        vit_table.add_row("aging (0.33-0.66)", Text("").join([Text(s, style=st) for s, st in bar]), str(histogram.aging))
                    if histogram.decayed:
                        bar = build_vitality_bar(0, 0, histogram.decayed, width=16, use_rich=True)
                        vit_table.add_row("decayed (<0.33)", Text("").join([Text(s, style=st) for s, st in bar]), str(histogram.decayed))

                    self.console.print(vit_table)

        return capture.get()

    def render_tags(self, tag_counts: Sequence[TagCount], verbose: bool = False) -> str:
        if not tag_counts:
            return self._styled("(no tags)", "izero.dim").plain

        with self.console.capture() as capture:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("tag", style="izero.tag")
            table.add_column("count", style="izero.number", justify="right")
            for tc in tag_counts:
                table.add_row(tc.tag, str(tc.count))
            self.console.print(table)

        return capture.get()

    def render_inspect(
        self,
        db_path: str,
        total: int,
        size_bytes: int,
        embedding_mode: str,
        avg_dim: float | None,
        cards_with_emb: int,
        decay_rows: list[dict],
        tokens: int,
    ) -> str:
        with self.console.capture() as capture:
            # Header panel
            header_lines = [
                f"db: {db_path}",
                f"total: {total} cards",
                f"size: {size_bytes / 1024:.1f} KB",
                f"mode: {embedding_mode}",
                f"tokens: ~{tokens}",
            ]
            footer = f"avg_dim: {avg_dim:.0f}" if avg_dim else ""
            panel = Panel(
                "\n".join(header_lines),
                title="inspect",
                style="izero.panel",
                border_style="izero.panel.border",
                subtitle=footer,
            )
            self.console.print(panel)
            self.console.print("")

            # Embedding coverage
            self.console.print(self._styled("embedding coverage:", "izero.dim"))
            self.console.print(f"  {cards_with_emb} / {total} cards have embeddings")
            self.console.print("")

            # Decay candidates
            if decay_rows:
                self.console.print(self._styled("decay candidates:", "izero.dim"))
                table = Table(show_header=True, box=None, padding=(0, 1))
                table.add_column("fact", style="izero.body", ratio=1)
                table.add_column("vitality", style="izero.number", width=10)
                table.add_column("age", style="izero.timestamp", width=8)
                table.add_column("id", style="izero.id", width=14)

                for d in decay_rows:
                    fact = d.get("fact", "")[:70]
                    vitality = d.get("vitality", 0.0)
                    age = d.get("age_days", 0)
                    vid = d.get("id", "?")[:12]
                    vit_style = vitality_style(vitality)
                    table.add_row(
                        fact,
                        f"{vitality:.2f}",
                        f"{age}d",
                        vid,
                    )
                self.console.print(table)
            else:
                self.console.print(self._styled("no decay candidates", "izero.success"))

        return capture.get()

    def render_dry_run(self, plan: dict[str, Any], limit: int = 0) -> str:
        with self.console.capture() as capture:
            cards = plan.get("cards", [])

            if limit > 0:
                cards = cards[:limit]

            self.console.print(self._styled(f"consolidation dry-run: {len(cards)} cards would be processed", "izero.hero"))
            self.console.print("")

            table = Table(show_header=True, box=None, padding=(0, 1))
            table.add_column("action", style="izero.body", width=8)
            table.add_column("fact", style="izero.body", ratio=1)

            for card in cards:
                action = card.get("action", "keep")
                fact = card.get("fact", "")[:70]
                action_style = {
                    "keep": "izero.success",
                    "merge": "izero.warning",
                    "delete": "izero.error",
                }.get(action, "izero.body")
                table.add_row(
                    self._styled(action.upper(), action_style).plain,
                    fact,
                )

            self.console.print(table)

            if limit > 0 and len(plan.get("cards", [])) > limit:
                self.console.print(self._styled(f"... ({len(plan.get('cards', [])) - limit} more)", "izero.dim"))

        return capture.get()

    def render_dashboard(self, state: dict[str, Any], interval: float) -> str:
        with self.console.capture() as capture:
            cards = state.get("cards", 0)
            size = state.get("size_bytes", 0)
            mode = state.get("embedding_mode", "unknown")
            vitality = state.get("vitality", {})
            fresh = vitality.get("fresh", 0)
            aging = vitality.get("aging", 0)
            decayed = vitality.get("decayed", 0)
            tokens = state.get("tokens", 0)
            recent = state.get("recent", [])
            decay_candidates = state.get("decay_candidates", [])

            # Build vitality bar
            bar = build_vitality_bar(fresh, aging, decayed, width=30, use_rich=True)
            bar_text = Text()
            for segment_text, segment_style in bar:
                bar_text.append(segment_text, style=segment_style)

            # Top panel
            top_lines = [
                f"cards: {cards}  ·  size: {size / 1024:.1f} KB  ·  mode: {mode}",
                "vitality: " + bar_text.plain + f"  fresh {fresh} / aging {aging} / decay {decayed}",
                f"tokens: {tokens} total  ·  ~{state.get('reclaimable_tokens', 0)} reclaimable",
            ]
            panel = Panel(
                "\n".join(top_lines),
                title=f"isotope_zero · {state.get('db_path', '?')}",
                style="izero.panel",
                border_style="izero.panel.border",
                subtitle=f"refresh {interval}s  ·  Ctrl-C to exit",
            )
            self.console.print(panel)

            # Recent
            if recent:
                self.console.print("")
                self.console.print(self._styled("recent:", "izero.dim"))
                for r in recent:
                    fact = r.get("fact", "")[:70]
                    age = r.get("age_days", 0)
                    self.console.print(
                        self._styled(glyph("bullet"), "izero.glyph.bullet") + " " +
                        self._styled(fact, "izero.body") + " " +
                        self._styled(f"({age:.1f}d)", "izero.timestamp")
                    )

            # Decay candidates
            if decay_candidates:
                self.console.print("")
                self.console.print(self._styled(f"decay candidates ({len(decay_candidates)}):", "izero.dim"))
                table = Table(show_header=True, box=None, padding=(0, 1))
                table.add_column("fact", style="izero.body", ratio=1)
                table.add_column("vitality", style="izero.number", width=10)
                table.add_column("age", style="izero.timestamp", width=8)
                for d in decay_candidates:
                    fact = d.get("fact", "")[:70]
                    vitality = d.get("vitality", 0)
                    age = d.get("age_days", 0)
                    vit_style = vitality_style(vitality)
                    table.add_row(
                        fact,
                        f"{vitality:.2f}",
                        f"{age}d",
                    )
                self.console.print(table)

        return capture.get()

    def render_menu(self, banner: str, entries: list[tuple[str, Any]], selected_idx: int) -> str:
        with self.console.capture() as capture:
            # Banner panel
            panel = Panel(
                banner,
                title="isotope_zero",
                style="izero.panel",
                border_style="izero.panel.border",
                subtitle="welcome back",
            )
            self.console.print(panel)
            self.console.print("")

            # Menu entries
            for i, (label, _) in enumerate(entries):
                is_selected = i == selected_idx
                if is_selected:
                    prefix, prefix_style = rich_selected()
                    label_style = "izero.selected"
                else:
                    prefix, prefix_style = rich_unselected()
                    label_style = "izero.body"
                self.console.print(
                    self._styled("  ", "izero.body") +
                    self._styled(prefix, prefix_style) +
                    self._styled("  ", "izero.body") +
                    self._styled(label, label_style)
                )

            self.console.print("")
            self.console.print(self._styled("↑↓ navigate · enter to run · q to quit · / to search commands", "izero.hint"))

        return capture.get()

    def render_add_result(self, card_id: str, created: bool) -> str:
        status = "remembered" if created else "updated"
        with self.console.capture() as capture:
            self.console.print(
                self._styled(glyph("ok"), "izero.glyph.ok") + " " +
                self._styled(status, "izero.hero") + " " +
                self._styled("·", "izero.glyph.bullet") + " " +
                self._styled(card_id[:12], "izero.id")
            )
        return capture.get()

    def render_forget_result(self, card_id: str) -> str:
        with self.console.capture() as capture:
            self.console.print(
                self._styled(glyph("ok"), "izero.glyph.ok") + " " +
                self._styled("forgotten", "izero.hero") + " " +
                self._styled("·", "izero.glyph.bullet") + " " +
                self._styled(card_id[:12], "izero.id")
            )
        return capture.get()

    def render_touch_result(self, card_id: str) -> str:
        with self.console.capture() as capture:
            self.console.print(
                self._styled(glyph("ok"), "izero.glyph.ok") + " " +
                self._styled("touched", "izero.hero") + " " +
                self._styled("·", "izero.glyph.bullet") + " " +
                self._styled(card_id[:12], "izero.id")
            )
        return capture.get()