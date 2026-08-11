"""isotope_zero CLI — Plain Text Renderer.

Human-readable output using only stdlib. Works without rich installed.
Graceful degradation for non-UTF-8 terminals.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import BaseRenderer
from ..ui.components import (
    MemoryRow,
    StatRow,
    TagCount,
    VitalityHistogram,
)

from ..ui.glyphs import (
    glyph,
    can_unicode,
    truncate,
    build_vitality_bar,
    box_top,
    box_bottom,
    box_line,
    build_box,
)


class PlainRenderer(BaseRenderer):
    """Plain text renderer — stdlib only, no external dependencies."""

    def __init__(self, use_color: bool = True, verbose: bool = False):
        super().__init__(use_color=use_color, verbose=verbose)
        self._unicode = can_unicode()

    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color if enabled and supported."""
        if not self.use_color:
            return text
        # ANSI codes for our palette
        colors = {
            "beige": "\033[38;2;245;240;225m",
            "white": "\033[38;2;255;255;255m",
            "muted": "\033[38;2;139;125;107m",
            "accent": "\033[38;2;212;196;168m",
            "success": "\033[38;2;168;212;168m",
            "warning": "\033[38;2;232;212;168m",
            "error": "\033[38;2;232;168;168m",
            "info": "\033[38;2;168;200;232m",
            "reset": "\033[0m",
            "bold": "\033[1m",
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def render_memory_rows(
        self,
        rows: Sequence[MemoryRow],
        show_score: bool = False,
        show_vitality: bool = False,
        show_tags: bool = True,
    ) -> str:
        if not rows:
            return self._color("(no memories)", "muted")

        lines = []
        for i, row in enumerate(rows, 1):
            parts = []

            # Index
            parts.append(self._color(f"{i}.", "muted"))

            # Fact (truncated if needed)
            fact = truncate(row.fact, 80)
            parts.append(self._color(fact, "beige"))

            # Tags
            if show_tags and row.tags:
                tags_str = ", ".join(row.tags)
                parts.append(self._color(f"· {tags_str}", "muted"))

            # Score
            if show_score and row.score is not None:
                parts.append(self._color(f"· {row.score:.4f}", "accent"))

            # Vitality
            if show_vitality and row.vitality is not None:
                bar = build_vitality_bar(
                    1 if row.vitality >= 0.66 else 0,
                    1 if 0.33 <= row.vitality < 0.66 else 0,
                    1 if row.vitality < 0.33 else 0,
                    width=10,
                )
                color = "success" if row.vitality >= 0.66 else ("warning" if row.vitality >= 0.33 else "error")
                parts.append(self._color(f"· [{bar}]", color))

            # ID (verbose)
            if self.verbose:
                parts.append(self._color(f"· {row.id[:12]}", "id"))

            lines.append("  ".join(parts))

        return "\n".join(lines)

    def render_memory_card(self, card: MemoryRow, verbose: bool = False) -> str:
        lines = []

        # Fact (the hero)
        lines.append(self._color(card.fact, "beige"))
        lines.append("")

        # Evidence
        if card.evidence:
            lines.append(self._color("evidence:", "muted"))
            lines.append(f"  {self._color(card.evidence, 'white')}")
            lines.append("")

        # Tags
        if card.tags:
            lines.append(self._color("tags:", "muted"))
            tags_str = ", ".join(card.tags)
            lines.append(f"  {self._color(tags_str, 'info')}")
            lines.append("")

        # ID
        lines.append(self._color(f"id: {card.id}", "id"))

        # Verbose details
        if verbose:
            lines.append("")
            if card.score is not None:
                lines.append(self._color(f"score: {card.score:.4f}", "accent"))
            if card.vitality is not None:
                color = "success" if card.vitality >= 0.66 else ("warning" if card.vitality >= 0.33 else "error")
                bar = build_vitality_bar(
                    1 if card.vitality >= 0.66 else 0,
                    1 if 0.33 <= card.vitality < 0.66 else 0,
                    1 if card.vitality < 0.33 else 0,
                    width=20,
                )
                lines.append(self._color(f"vitality: [{bar}] {card.vitality:.2f}", color))
            if card.created_at:
                lines.append(self._color(f"created: {card.created_at}", "timestamp"))
            if card.accessed_at:
                lines.append(self._color(f"accessed: {card.accessed_at}", "timestamp"))
            if card.access_count is not None:
                lines.append(self._color(f"accesses: {card.access_count}", "number"))

        return "\n".join(lines)

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
        lines = []

        # Hero line
        size_kb = size_bytes / 1024
        hero = f"{count} memories  ·  {size_kb:.1f} KB  ·  {embedding_mode}  ·  ~{tokens} tokens"
        lines.append(self._color(hero, "beige"))

        if not verbose:
            return "\n".join(lines)

        lines.append("")

        # Tags
        if tag_dist:
            lines.append(self._color("tag distribution:", "muted"))
            for tc in tag_dist:
                lines.append(f"  {self._color(tc.tag, 'info')}  {self._color(str(tc.count), 'number')}")
            lines.append("")

        # Vitality histogram
        if histogram:
            lines.append(self._color("vitality histogram:", "muted"))
            if histogram.fresh:
                bar = build_vitality_bar(histogram.fresh, 0, 0, width=20)
                lines.append(f"  {self._color('[fresh]', 'success')}  {self._color(bar, 'success')}  {self._color(str(histogram.fresh), 'number')}")
            if histogram.aging:
                bar = build_vitality_bar(0, histogram.aging, 0, width=20)
                lines.append(f"  {self._color('[aging]', 'warning')}  {self._color(bar, 'warning')}  {self._color(str(histogram.aging), 'number')}")
            if histogram.decayed:
                bar = build_vitality_bar(0, 0, histogram.decayed, width=20)
                lines.append(f"  {self._color('[decayed]', 'error')}  {self._color(bar, 'error')}  {self._color(str(histogram.decayed), 'number')}")

        return "\n".join(lines)

    def render_tags(self, tag_counts: Sequence[TagCount], verbose: bool = False) -> str:
        if not tag_counts:
            return self._color("(no tags)", "muted")

        lines = []
        for tc in tag_counts:
            lines.append(f"  {self._color(tc.tag, 'info')}  {self._color(str(tc.count), 'number')}")
        return "\n".join(lines)

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
        lines = []

        # Header box
        lines.append(build_box(
            "inspect",
            [
                f"db: {db_path}",
                f"total: {total} cards",
                f"size: {size_bytes / 1024:.1f} KB",
                f"mode: {embedding_mode}",
                f"tokens: ~{tokens}",
            ],
            f"avg_dim: {avg_dim:.0f}" if avg_dim else "",
            heavy=True,
        ))
        lines.append("")

        # Embedding coverage
        lines.append(self._color("embedding coverage:", "muted"))
        lines.append(f"  {cards_with_emb} / {total} cards have embeddings")
        lines.append("")

        # Decay candidates
        if decay_rows:
            lines.append(self._color("decay candidates:", "muted"))
            for d in decay_rows:
                vid = d.get("id", "?")[:12]
                fact = truncate(d.get("fact", ""), 60)
                vitality = d.get("vitality", 0.0)
                age = d.get("age_days", 0)
                color = "error" if vitality < 0.33 else "warning"
                lines.append(f"  {self._color(fact, 'white')}  {self._color(f'v={vitality:.2f} age={age}d', color)}  {self._color(vid, 'id')}")
        else:
            lines.append(self._color("no decay candidates", "success"))

        return "\n".join(lines)

    def render_dry_run(self, plan: dict[str, Any], limit: int = 0) -> str:
        lines = []
        cards = plan.get("cards", [])

        if limit > 0:
            cards = cards[:limit]

        lines.append(self._color(f"consolidation dry-run: {len(cards)} cards would be processed", "beige"))
        lines.append("")

        for card in cards:
            action = card.get("action", "keep")
            fact = truncate(card.get("fact", ""), 70)
            color = "success" if action == "keep" else ("warning" if action == "merge" else "error")
            lines.append(f"  {self._color(action.upper(), color)}  {self._color(fact, 'white')}")

        if limit > 0 and len(plan.get("cards", [])) > limit:
            lines.append(f"  {self._color('...', 'muted')}  ({len(plan.get('cards', [])) - limit} more)")

        return "\n".join(lines)

    def render_dashboard(self, state: dict[str, Any], interval: float) -> str:
        # Dashboard in plain mode is just a static snapshot
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

        lines = []

        # Top bar
        bar = build_vitality_bar(fresh, aging, decayed, width=30)
        lines.append(build_box(
            f"isotope_zero · {state.get('db_path', '?')}",
            [
                f"cards: {cards}  ·  size: {size / 1024:.1f} KB  ·  mode: {mode}",
                f"vitality: {bar}  fresh {fresh} / aging {aging} / decay {decayed}",
                f"tokens: {tokens} total  ·  ~{state.get('reclaimable_tokens', 0)} reclaimable",
            ],
            f"refresh {interval}s  ·  Ctrl-C to exit",
            heavy=True,
        ))

        # Recent
        if recent:
            lines.append("")
            lines.append(self._color("recent:", "muted"))
            for r in recent:
                fact = truncate(r.get("fact", ""), 70)
                age = r.get("age_days", 0)
                lines.append(f"  {self._color(glyph('bullet'), 'accent')} {self._color(fact, 'white')}  {self._color(f'({age:.1f}d)', 'muted')}")

        # Decay candidates
        if decay_candidates:
            lines.append("")
            lines.append(self._color(f"decay candidates ({len(decay_candidates)}):", "muted"))
            for d in decay_candidates:
                fact = truncate(d.get("fact", ""), 70)
                vitality = d.get("vitality", 0)
                age = d.get("age_days", 0)
                color = "error" if vitality < 0.33 else "warning"
                lines.append(f"  {self._color(glyph('bullet'), 'accent')} {self._color(fact, 'white')}  {self._color(f'v={vitality:.2f} age={age}d', color)}")

        return "\n".join(lines)

    def render_menu(self, banner: str, entries: list[tuple[str, Any]], selected_idx: int) -> str:
        lines = []

        # Banner box
        lines.append(build_box("isotope_zero", [banner], heavy=True))
        lines.append("")

        # Menu entries
        for i, (label, _) in enumerate(entries):
            is_selected = i == selected_idx
            prefix = self._color(glyph("selected"), "success") if is_selected else self._color(glyph("unselected"), "muted")
            label_color = "white" if is_selected else "beige"
            lines.append(f"  {prefix}  {self._color(label, label_color)}")

        lines.append("")
        lines.append(self._color("↑↓ navigate · enter to run · q to quit · / to search commands", "hint"))

        return "\n".join(lines)

    def render_add_result(self, card_id: str, created: bool) -> str:
        status = "remembered" if created else "updated"
        return f"{self._color(glyph('ok'), 'success')} {self._color(status, 'beige')}  {self._color('·', 'muted')}  {self._color(card_id[:12], 'id')}"

    def render_forget_result(self, card_id: str) -> str:
        return f"{self._color(glyph('ok'), 'success')} {self._color('forgotten', 'beige')}  {self._color('·', 'muted')}  {self._color(card_id[:12], 'id')}"

    def render_touch_result(self, card_id: str) -> str:
        return f"{self._color(glyph('ok'), 'success')} {self._color('touched', 'beige')}  {self._color('·', 'muted')}  {self._color(card_id[:12], 'id')}"