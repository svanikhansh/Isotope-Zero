"""isotope_zero CLI — Dashboard command (live TUI).

Serves the shared dashboard surface in the terminal: a rich multi-panel live
TUI (KPI row, vitality bar, tags strip, recent + decay tables) refreshed in
place. Both this TUI and the browser dashboard (``izero serve``) consume the
exact same state dict from ``dash.data.collect_state``, so the two surfaces
can never drift.

    izero dashboard [--db PATH] [--interval N] [--once] [--no-color]

    --once      prints one static frame and exits (scriptable/pipable)
    --interval  refresh cadence in seconds (default 2.0)
    --no-color  force the stdlib clear-and-reprint transport (no rich)

The missing-DB guard is strict: a typo'd ``--db`` path is reported on stderr
with exit 1, never spun as an empty dashboard.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer

from ..core.context import (
    db_option,
    json_option,
    verbose_option,
    no_color_option,
    inject_context,
)
from ..dash import collect_state, missing_db
from ..dash.tui import run_live


def register(app: typer.Typer) -> None:
    """Register the dashboard command with the typer app."""

    @app.command("dashboard")
    def dashboard_cmd(
        ctx: typer.Context,
        interval: Annotated[float, typer.Option("--interval", "-i", help="Refresh interval in seconds")] = 2.0,
        once: Annotated[bool, typer.Option("--once", help="Render one frame and exit (scriptable)")] = False,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """Live overview of the memory store (terminal TUI)."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color)

        db_path = cli_ctx.db_path or ":memory:"
        if missing_db(db_path):
            print(f"DB path does not exist: {db_path}", file=sys.stderr)
            raise typer.Exit(1)

        from isotope_zero.client import IsotopeZero

        client = IsotopeZero(db_path, spawn_daemon=False, use_mmap=False)
        try:
            # The data source (embedded + vitality-scored) is shared with the
            # browser surface; the TUI transport handles --once / rich / plain.
            sys.exit(
                run_live(
                    client.store,
                    db_path,
                    interval=interval,
                    once=once,
                    force_plain=cli_ctx.no_color or not sys.stdout.isatty(),
                )
            )
        finally:
            client.close()


__all__ = ["register"]
