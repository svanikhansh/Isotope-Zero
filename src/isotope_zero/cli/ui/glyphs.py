"""isotope_zero CLI — Unicode glyph system with ASCII fallback.

Provides a consistent glyph vocabulary across all renderers with graceful
degradation for non-UTF-8 terminals (Windows cp1252, LC_ALL=C pipes).
"""

from __future__ import annotations

import sys
from typing import Literal

# --------------------------------------------------------------------------- #
# Glyph Definitions
# --------------------------------------------------------------------------- #
# Each glyph has a Unicode preferred form and an ASCII fallback
_GLYPHS: dict[str, dict[Literal["unicode", "ascii"], str]] = {
    # Status
    "ok": {"unicode": "✓", "ascii": "ok"},
    "check": {"unicode": "✓", "ascii": "✓"},  # alias
    "cross": {"unicode": "✗", "ascii": "x"},
    "warning": {"unicode": "⚠", "ascii": "!"},
    "info": {"unicode": "ℹ", "ascii": "i"},

    # Bullets & markers
    "bullet": {"unicode": "·", "ascii": "-"},
    "arrow": {"unicode": "→", "ascii": "->"},
    "arrow_right": {"unicode": "▸", "ascii": ">"},
    "arrow_down": {"unicode": "▾", "ascii": "v"},
    "chevron_right": {"unicode": "›", "ascii": ">"},
    "chevron_down": {"unicode": "▼", "ascii": "v"},

    # Selection
    "selected": {"unicode": "❯", "ascii": ">"},
    "unselected": {"unicode": " ", "ascii": " "},
    "radio_on": {"unicode": "◉", "ascii": "(*)"},
    "radio_off": {"unicode": "○", "ascii": "( )"},
    "checkbox_on": {"unicode": "☑", "ascii": "[x]"},
    "checkbox_off": {"unicode": "☐", "ascii": "[ ]"},

    # Box drawing (light)
    "box_tl": {"unicode": "┌", "ascii": "+"},
    "box_tr": {"unicode": "┐", "ascii": "+"},
    "box_bl": {"unicode": "└", "ascii": "+"},
    "box_br": {"unicode": "┘", "ascii": "+"},
    "box_h": {"unicode": "─", "ascii": "-"},
    "box_v": {"unicode": "│", "ascii": "|"},
    "box_t": {"unicode": "┬", "ascii": "+"},
    "box_b": {"unicode": "┴", "ascii": "+"},
    "box_l": {"unicode": "├", "ascii": "+"},
    "box_r": {"unicode": "┤", "ascii": "+"},
    "box_x": {"unicode": "┼", "ascii": "+"},

    # Box drawing (heavy)
    "box_heavy_tl": {"unicode": "┏", "ascii": "+"},
    "box_heavy_tr": {"unicode": "┓", "ascii": "+"},
    "box_heavy_bl": {"unicode": "┗", "ascii": "+"},
    "box_heavy_br": {"unicode": "┛", "ascii": "+"},
    "box_heavy_h": {"unicode": "━", "ascii": "="},
    "box_heavy_v": {"unicode": "┃", "ascii": "|"},

    # Bars (vitality)
    "bar_full": {"unicode": "█", "ascii": "#"},
    "bar_medium": {"unicode": "▒", "ascii": "="},
    "bar_low": {"unicode": "░", "ascii": "-"},
    "bar_empty": {"unicode": " ", "ascii": " "},

    # Misc
    "ellipsis": {"unicode": "…", "ascii": "..."},
    "star": {"unicode": "★", "ascii": "*"},
    "heart": {"unicode": "♥", "ascii": "<3"},
    "sparkles": {"unicode": "✨", "ascii": "* *"},
    "gear": {"unicode": "⚙", "ascii": "[~]"},
    "database": {"unicode": "🗄", "ascii": "[DB]"},
    "memory": {"unicode": "🧠", "ascii": "[MEM]"},
    "search": {"unicode": "🔍", "ascii": "[?]"},
    "lightning": {"unicode": "⚡", "ascii": "[!]"},
    "rocket": {"unicode": "🚀", "ascii": "[>>]"},
}

# --------------------------------------------------------------------------- #
# Encoding Detection
# --------------------------------------------------------------------------- #
def _can_encode_unicode() -> bool:
    """Check if stdout can encode Unicode characters."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "✓".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# Cache the result
_CAN_UNICODE = _can_encode_unicode()


def can_unicode() -> bool:
    """Return True if the terminal supports Unicode output."""
    return _CAN_UNICODE


# --------------------------------------------------------------------------- #
# Glyph Resolution
# --------------------------------------------------------------------------- #
def glyph(name: str) -> str:
    """Get a glyph by name, with automatic Unicode/ASCII fallback.

    Args:
        name: Glyph name (e.g., "ok", "bullet", "arrow", "box_tl")

    Returns:
        The appropriate glyph for the current terminal capabilities.
    """
    if name not in _GLYPHS:
        return name  # Return as-is if unknown

    mode = "unicode" if _CAN_UNICODE else "ascii"
    return _GLYPHS[name][mode]


def glyph_unicode(name: str) -> str:
    """Force Unicode glyph (for testing or explicit Unicode output)."""
    if name not in _GLYPHS:
        return name
    return _GLYPHS[name]["unicode"]


def glyph_ascii(name: str) -> str:
    """Force ASCII glyph (for testing or explicit ASCII output)."""
    if name not in _GLYPHS:
        return name
    return _GLYPHS[name]["ascii"]


# --------------------------------------------------------------------------- #
# Convenience Functions for Common Patterns
# --------------------------------------------------------------------------- #
def ok() -> str:
    """Success checkmark."""
    return glyph("ok")


def bullet() -> str:
    """List bullet."""
    return glyph("bullet")


def arrow() -> str:
    """Right arrow."""
    return glyph("arrow")


def selected() -> str:
    """Selection marker for menus."""
    return glyph("selected")


def unselected() -> str:
    """Unselected marker for menus."""
    return glyph("unselected")


def box(corner: str = "tl") -> str:
    """Box drawing character."""
    return glyph(f"box_{corner}")


def box_heavy(corner: str = "tl") -> str:
    """Heavy box drawing character."""
    return glyph(f"box_heavy_{corner}")


def bar_segment(level: Literal["full", "medium", "low", "empty"]) -> str:
    """Vitality bar segment."""
    return glyph(f"bar_{level}")


def truncate(text: str, width: int, suffix: str | None = None) -> str:
    """Truncate text to width with ellipsis glyph."""
    if suffix is None:
        suffix = glyph("ellipsis")
    text = str(text)
    if len(text) <= width:
        return text
    if width <= len(suffix):
        return suffix[:width]
    return text[: width - len(suffix)] + suffix


# --------------------------------------------------------------------------- #
# Rich Segment Builders (for rich-based renderers)
# --------------------------------------------------------------------------- #
def rich_ok() -> tuple[str, str]:
    """Rich segment: (text, style) for success glyph."""
    return (glyph("ok"), "izero.glyph.ok")


def rich_bullet() -> tuple[str, str]:
    """Rich segment: (text, style) for bullet."""
    return (glyph("bullet"), "izero.glyph.bullet")


def rich_arrow() -> tuple[str, str]:
    """Rich segment: (text, style) for arrow."""
    return (glyph("arrow"), "izero.glyph.arrow")


def rich_selected() -> tuple[str, str]:
    """Rich segment: (text, style) for selected menu item."""
    return (glyph("selected"), "izero.selected")


def rich_unselected() -> tuple[str, str]:
    """Rich segment: (text, style) for unselected menu item."""
    return (glyph("unselected"), "izero.dim")


# --------------------------------------------------------------------------- #
# Vitality Bar Builder
# --------------------------------------------------------------------------- #
def build_vitality_bar(
    fresh: int,
    aging: int,
    decayed: int,
    width: int = 20,
    use_rich: bool = False,
) -> str | list[tuple[str, str]]:
    """Build a fixed-width vitality bar.

    Args:
        fresh: Count of fresh cards (vitality >= 0.66)
        aging: Count of aging cards (0.33 <= vitality < 0.66)
        decayed: Count of decayed cards (vitality < 0.33)
        width: Total bar width in characters
        use_rich: If True, return list of (text, style) segments for rich

    Returns:
        String (plain) or list of rich segments.
    """
    total = max(1, fresh + aging + decayed)

    # Calculate segment widths (round to nearest, last absorbs remainder)
    f = round(fresh / total * width)
    d = round(decayed / total * width)
    a = width - f - d
    a = max(0, min(width - f - d, a))

    # Adjust for any rounding errors
    if f + a + d != width:
        a = width - f - d

    if use_rich:
        segments = []
        if f > 0:
            segments.append((glyph_unicode("bar_full") * f, "izero.bar.fresh"))
        if a > 0:
            segments.append((glyph_unicode("bar_medium") * a, "izero.bar.aging"))
        if d > 0:
            segments.append((glyph_unicode("bar_low") * d, "izero.bar.decay"))
        return segments

    # Plain text
    return (
        glyph("bar_full") * f +
        glyph("bar_medium") * a +
        glyph("bar_low") * d
    )


# --------------------------------------------------------------------------- #
# Box Drawing Helpers
# --------------------------------------------------------------------------- #
def box_top(title: str, width: int, heavy: bool = False) -> str:
    """Build top border with centered title."""
    prefix = box_heavy("tl") if heavy else box("tl")
    h = box_heavy("h") if heavy else box("h")
    suffix = box_heavy("tr") if heavy else box("tr")

    title_padded = f" {title} "
    if len(title_padded) > width - 2:
        title_padded = title_padded[:width - 2]

    padding = width - len(title_padded)
    left_pad = padding // 2
    right_pad = padding - left_pad

    return prefix + h * left_pad + title_padded + h * right_pad + suffix


def box_bottom(footer: str, width: int, heavy: bool = False) -> str:
    """Build bottom border with left-aligned footer."""
    prefix = box_heavy("bl") if heavy else box("bl")
    h = box_heavy("h") if heavy else box("h")
    suffix = box_heavy("br") if heavy else box("br")

    footer_padded = f" {footer} "
    if len(footer_padded) > width - 2:
        footer_padded = footer_padded[:width - 2]

    padding = width - len(footer_padded)
    return prefix + footer_padded + h * padding + suffix


def box_line(content: str, width: int, heavy: bool = False) -> str:
    """Build a content line with side borders."""
    v = box_heavy("v") if heavy else box("v")
    padded = content + " " * (width - len(content))
    return f"{v} {padded} {v}"


def build_box(
    title: str,
    lines: list[str],
    footer: str = "",
    heavy: bool = False,
) -> str:
    """Build a complete box with title, body lines, and footer.

    Args:
        title: Title text (centered in top border)
        lines: Body content lines
        footer: Footer text (left-aligned in bottom border)
        heavy: Use heavy box drawing characters

    Returns:
        Complete box as a string.
    """
    # Calculate inner width
    inner_width = max(
        len(title) + 2,  # title needs at least this much
        *(len(l) for l in lines),
        len(footer) + 2,
    )

    parts = []
    parts.append(box_top(title, inner_width, heavy))
    for line in lines:
        parts.append(box_line(line, inner_width, heavy))
    if footer:
        parts.append(box_bottom(footer, inner_width, heavy))
    else:
        # Bottom without footer
        prefix = box_heavy("bl") if heavy else box("bl")
        h = box_heavy("h") if heavy else box("h")
        suffix = box_heavy("br") if heavy else box("br")
        parts.append(prefix + h * inner_width + suffix)

    return "\n".join(parts)