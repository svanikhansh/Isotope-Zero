"""isotope_zero CLI — Serve command (browser dashboard).

Serves the shared dashboard surface in the browser at
``http://localhost:<port>`` via a stdlib-only HTTP server. The page is one
self-contained HTML document (inline CSS/JS, no CDN, no web fonts) that
refreshes live over Server-Sent Events.

    izero serve [--db PATH] [--port N] [--interval N] [--open]

Read-only and local-first: binds to 127.0.0.1 by default, never writes to the
store, and pushes the same state dict the terminal TUI renders
(``dash.data.collect_state``), so the two surfaces always agree.
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
from ..dash import missing_db
from ..dash.web import DEFAULT_PORT, DEFAULT_INTERVAL, run_server


def register(app: typer.Typer) -> None:
    """Register the serve command with the typer app."""

    @app.command("serve")
    def serve_cmd(
        ctx: typer.Context,
        port: Annotated[int, typer.Option("--port", "-p", help="Port to bind (default 8930)")] = DEFAULT_PORT,
        interval: Annotated[float, typer.Option("--interval", "-i", help="Refresh interval in seconds")] = DEFAULT_INTERVAL,
        open_browser: Annotated[bool, typer.Option("--open", "-o", help="Open the dashboard in a browser")] = False,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """Browser dashboard served at localhost."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color)

        db_path = cli_ctx.db_path or ":memory:"
        if missing_db(db_path):
            print(f"DB path does not exist: {db_path}", file=sys.stderr)
            raise typer.Exit(1)

        from isotope_zero.client import IsotopeZero

        client = IsotopeZero(db_path, spawn_daemon=False, use_mmap=False)
        try:
            sys.exit(
                run_server(
                    client.store,
                    db_path,
                    host="127.0.0.1",
                    port=port,
                    interval=interval,
                    open_browser=open_browser,
                )
            )
        finally:
            client.close()


__all__ = ["register"]
