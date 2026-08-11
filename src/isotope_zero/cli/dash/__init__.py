"""isotope_zero CLI — Dashboard package (web + TUI surfaces).

``dash.data``       — shared read-only state collection (single source of truth)
``dash.tui``        — terminal TUI dashboard surface (rich)
``dash.web``        — browser dashboard served at localhost (stdlib http.server)
"""

from __future__ import annotations

from .data import collect_state, missing_db, _embedding_mode, _human_bytes

__all__ = ["collect_state", "missing_db", "_embedding_mode", "_human_bytes"]
