"""isotope_zero CLI — Void Black / White / Ocean Blue theme definition.

Colors:
    Void          #000000  — pure dark background
    White         #FFFFFF  — primary content, crisp reading
    Ocean-light   #90CAF9  — accent, active elements, highlights (material blue 200)
    Ocean         #2196F3  — primary brand blue (material blue 500)
    Ocean-dark    #1565C0  — elevated surfaces, panels, gradient endpoint
    Ocean-deep    #0D47A1  — deep accent, strong highlights
    Charcoal      #1A1A1A  — surface background, borders, dock/glass

Inspired by Material Design blue scale + shadcn/21st.dev layered-surface
principles: near-invisible tints, 1px borders, no heavy shadows,
two-tier typography, semantic tokens for every surface.

All rich style names follow the ``izero.*`` convention so the renderer
and TUI can reference them by name.
"""

from __future__ import annotations

from rich.theme import Theme

# --------------------------------------------------------------------------- #
# Color Palette
# --------------------------------------------------------------------------- #
VOID = "#000000"
WHITE = "#FFFFFF"
OCEAN_LIGHT = "#90CAF9"          # accent / active elements / highlights
OCEAN = "#2196F3"                # primary brand blue
OCEAN_DARK = "#1565C0"           # elevated surfaces / panels / gradient endpoint
OCEAN_DEEP = "#0D47A1"           # deep accent / strong highlights
CHARCOAL = "#1A1A1A"             # surface background / borders / glass

# Semantic colors harmonized with the ocean-blue palette
SUCCESS = "#90CAF9"              # ocean-light = success (blue native)
WARNING = "#FFB74D"              # warm amber (orange 300)
ERROR = "#EF5350"                # warm red (red 400)
INFO = "#4FC3F7"                 # sky blue (light blue 300)

# --------------------------------------------------------------------------- #
# Rich Theme
# --------------------------------------------------------------------------- #
IZERO_THEME = Theme({
    # Base palette
    "izero.void": VOID,
    "izero.white": WHITE,
    "izero.ocean": OCEAN,
    "izero.ocean_light": OCEAN_LIGHT,
    "izero.ocean_dark": OCEAN_DARK,
    "izero.ocean_deep": OCEAN_DEEP,
    "izero.charcoal": CHARCOAL,

    # Semantic
    "izero.success": SUCCESS,
    "izero.warning": WARNING,
    "izero.error": ERROR,
    "izero.info": INFO,

    # UI surfaces
    "izero.panel": f"on {CHARCOAL}",
    "izero.panel.border": f"{OCEAN_LIGHT} on {CHARCOAL}",
    "izero.panel.title": f"bold {WHITE} on {CHARCOAL}",
    "izero.title": f"bold {WHITE} on {CHARCOAL}",
    "izero.subtitle": f"{OCEAN_LIGHT} on {CHARCOAL}",
    "izero.muted": f"{OCEAN_LIGHT} on {CHARCOAL}",       # secondary / labels
    "izero.dim": f"{OCEAN_DARK} on {CHARCOAL}",          # tertiary / borders

    # Glyphs / sparkline (bar glyphs drawn with these styles)
    "izero.bar.fresh": f"{SUCCESS} on {CHARCOAL}",
    "izero.bar.aging": f"{WARNING} on {CHARCOAL}",
    "izero.bar.decay": f"{ERROR} on {CHARCOAL}",
    "izero.bar.empty": f"{CHARCOAL} on {CHARCOAL}",

    # Status / vitality badges
    "izero.vitality.fresh": f"bold {SUCCESS} on {CHARCOAL}",
    "izero.vitality.aging": f"bold {WARNING} on {CHARCOAL}",
    "izero.vitality.decay": f"bold {ERROR} on {CHARCOAL}",

    # Tag chips (frequency-coded)
    "izero.tag.high": f"bold {SUCCESS} on {CHARCOAL}",    # top 33%
    "izero.tag.mid": f"bold {INFO} on {CHARCOAL}",        # mid 33%
    "izero.tag.low": f"bold {OCEAN_DARK} on {CHARCOAL}",  # bottom 33%
    "izero.tag": f"{OCEAN_LIGHT} on {CHARCOAL}",          # generic inline tag

    # Table styles
    "izero.table.header": f"bold {WHITE} on {CHARCOAL}",
    "izero.table.cell": f"{WHITE} on {CHARCOAL}",
    "izero.table.border": f"{OCEAN_LIGHT} on {CHARCOAL}",

    # Alert / notice banner
    "izero.alert.info": f"{INFO} on {CHARCOAL}",
    "izero.alert.success": f"{SUCCESS} on {CHARCOAL}",
    "izero.alert.warning": f"{WARNING} on {CHARCOAL}",
    "izero.alert.error": f"{ERROR} on {CHARCOAL}",

    # Dock / footer
    "izero.dock.bg": f"on {CHARCOAL}",
    "izero.dock.active": f"bold {OCEAN_LIGHT} on {CHARCOAL}",
    "izero.dock.inactive": f"{OCEAN_DARK} on {CHARCOAL}",

    # Search / input
    "izero.input.border": f"{OCEAN_LIGHT} on {CHARCOAL}",
    "izero.input.placeholder": f"{OCEAN_DARK} on {CHARCOAL}",

    # Body / accent / number / hero (TUI backward-compat)
    "izero.body": f"{WHITE} on {CHARCOAL}",
    "izero.accent": f"{OCEAN_LIGHT} on {CHARCOAL}",
    "izero.number": f"bold {WHITE} on {CHARCOAL}",
    "izero.hero": f"bold {WHITE} on {CHARCOAL}",
    "izero.hint": f"{OCEAN_DARK} on {CHARCOAL}",
    "izero.id": f"{OCEAN_LIGHT} on {CHARCOAL}",
    "izero.timestamp": f"{OCEAN_DARK} on {CHARCOAL}",
    "izero.evidence": f"{OCEAN_LIGHT} on {CHARCOAL}",
    "izero.selected": f"bold {WHITE} on {OCEAN_DEEP}",

    # Glyphs (inline icons used in components)
    "izero.glyph.arrow": f"{OCEAN_LIGHT} on {CHARCOAL}",
    "izero.glyph.bullet": f"{OCEAN_LIGHT} on {CHARCOAL}",
    "izero.glyph.ok": f"{SUCCESS} on {CHARCOAL}",

    # Skeleton / loading
    "izero.skeleton": f"{CHARCOAL} on {CHARCOAL}",
    "izero.shimmer": f"{OCEAN_LIGHT} on {CHARCOAL}",
})


# --------------------------------------------------------------------------- #
# Vitals helpers — vitality + tag styles for components/renderers
# --------------------------------------------------------------------------- #

def vitality_color(vitality: float) -> str:
    """Return a hex color for a vitality score (0→1).

    Three tiers: fresh (≥0.66 ocean-light), aging (≥0.33 amber), decay (<0.33 red).
    """
    if vitality is None:
        return CHARCOAL
    if vitality >= 0.66:
        return SUCCESS
    if vitality >= 0.33:
        return WARNING
    return ERROR


def vitality_style(vitality: float) -> str:
    """Return a rich style name for a vitality score (0→1)."""
    if vitality is None:
        return "izero.bar.empty"
    if vitality >= 0.66:
        return "izero.vitality.fresh"
    if vitality >= 0.33:
        return "izero.vitality.aging"
    return "izero.vitality.decay"


# Rotating tag palette (cycle through tag tiers by index)
_TAG_PALETTE = ["izero.tag.high", "izero.tag.mid", "izero.tag.low", "izero.tag"]


def tag_color(index: int) -> str:
    """Return a rotating rich style name for the nth tag (0-indexed)."""
    return _TAG_PALETTE[index % len(_TAG_PALETTE)]


# --------------------------------------------------------------------------- #
# Console / theme factories
# --------------------------------------------------------------------------- #

def get_theme() -> Theme:
    """Return the shared isotope_zero Rich theme."""
    return IZERO_THEME


def get_console() -> "Console":  # type: ignore[name-defined]
    """Return a Console configured with the isotope_zero theme.

    Lazily imports ``rich.console.Console`` so the module stays importable
    in environments where rich is absent (the TUI guards on rich availability).
    """
    from rich.console import Console

    return Console(theme=IZERO_THEME)


# Expose for downstream users (tests, plugins)
__all__ = [
    "VOID",
    "WHITE",
    "OCEAN_LIGHT",
    "OCEAN",
    "OCEAN_DARK",
    "OCEAN_DEEP",
    "CHARCOAL",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "INFO",
    "IZERO_THEME",
    "vitality_color",
    "vitality_style",
    "tag_color",
    "get_theme",
    "get_console",
]