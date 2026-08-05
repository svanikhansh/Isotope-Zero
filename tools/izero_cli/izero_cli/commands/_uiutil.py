"""Shared rich UI helpers for the izero-cli command modules.

Re-exports the established palette and micro-helpers from ``ui.py`` so each
command module imports from one stable seam (``izero_cli.commands._uiutil``)
rather than reaching into private ``ui._underscore`` names. Keeping a single
aesthetic source of truth means all 12 commands render as one system.

Palette (enforced across all commands):
    lavender  #af87ff  — headers & titles
    cyan      #00d7ff  — highlights, ids, badges
    gold      bold gold1 — metrics (latency ms, memory MB, scores)
    white     bold white — primary text / metrics
    dim       grey     — borders, muted text
    green3    emerald  — healthy / success
    gold3     amber    — warning
    red3      coral    — error

Glyphs: ⚡ latency · 🧠 cards · 🏥/🩺 doctor · 🔍 search · 📊 stats/bench ·
        🧹 vacuum · 📦 export/import · 👀 watch · ↔️ diff
"""
from __future__ import annotations

from rich.box import ROUNDED
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Palette (single source of truth — same constants as ui.py).
LAVENDER = "#af87ff"
CYAN = "#00d7ff"
GOLD = "bold gold1"
WHITE = "bold white"
DIM = "dim"
GREEN = "green3"
AMBER = "gold3"
CORAL = "red3"

ROUNDED_BOX = ROUNDED

# Module-level console shared by all command modules.
CONSOLE = Console()

# --- re-exported helpers from ui.py (the canonical implementations) ---------- #
from izero_cli.ui import (  # noqa: E402  — re-export seam
    _badge as badge,
    _trunc as trunc,
    _human_age as human_age,
    _iso as iso,
    _score_color as score_color,
    _title as title,
    _error_panel as error_panel,
)


def section_title(text: str) -> Text:
    """Lavender bold title with a glyph. Alias of ``title`` for readability."""
    return title(text)


__all__ = [
    "LAVENDER", "CYAN", "GOLD", "WHITE", "DIM", "GREEN", "AMBER", "CORAL",
    "ROUNDED_BOX", "CONSOLE", "Console", "Group", "Panel", "Table", "Text",
    "badge", "trunc", "human_age", "iso", "score_color", "title",
    "section_title", "error_panel",
]
