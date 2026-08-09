"""Tiny stdlib-only display helpers shared by the CLI modules.

``debug.py`` (the command dispatcher), ``render.py`` (the human-readable
formatters), and ``dashboard.py`` (the live TUI) all need the same handful of
display helpers: byte sizing, age-in-days, and string truncation. Keeping them
here — in a module with **only stdlib imports** — means ``render.py`` no longer
has to ``from .debug import …`` at module top, which transitively pulled the
entire embedding/engine stack into render.py's import graph. Now render.py is
genuinely stdlib + this local helper module only, honoring the "zero new core
deps" philosophy of the clean-output redesign.

These are deliberately trivial (2–4 lines each) and dependency-free so any CLI
module can import them without cost.
"""

from __future__ import annotations

_SECS_PER_DAY: float = 86400.0


def human_bytes(n: int) -> str:
    """Bytes -> human-readable string (B / KB / MB / GB / TB)."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n < 1024 * 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024 * 1024):.2f} GB"
    return f"{n / (1024 * 1024 * 1024 * 1024):.2f} TB"


def age_days(timestamp: float, now: float) -> float:
    """Age in days from ``timestamp`` to ``now``, floored at 0."""
    return round(max(0.0, now - timestamp) / _SECS_PER_DAY, 1)


def trunc(s: str, n: int) -> str:
    """Truncate a string to ``n`` chars (display-only; JSON keeps full values)."""
    s = str(s)
    return s if len(s) <= n else s[:n]
