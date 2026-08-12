"""isotope_zero CLI — Hook Command.

Claude Code lifecycle hook for automatic memory capture.
"""

from __future__ import annotations

import json
import sys
import typer
from typing import Annotated, Optional

from ..core.context import (
    CLIContext,
    inject_context,
    db_option,
    json_option,
    verbose_option,
    no_color_option,
)
from ..render import get_renderer_from_context
from ..ui.glyphs import glyph


def register(app: typer.Typer) -> None:
    """Register hook command with the typer app."""

    @app.command("hook")
    def hook_cmd(
        ctx: typer.Context,
        event: Annotated[str, typer.Argument(help="Hook event type (session_start, user_prompt, file_read, block_memory_write, bash_output, stop, pre_compact)")],
        db: str = db_option(),
        as_json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """Claude Code lifecycle hook — captures context to memory.

        Reads hook payload from stdin (JSON), processes it, and writes
        hook output to stdout. This is the single entrypoint used by
        the bash hook wrappers in integrations/izero-plugin/hooks/.
        """
        cli_ctx = inject_context(ctx, db, as_json, verbose, no_color)
        renderer = get_renderer_from_context(cli_ctx)

        # Read hook payload from stdin
        raw = sys.stdin.read()
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            payload = {}

        # Process hook using the integrations hook engine
        from isotope_zero.integrations.hooks import handle_hook

        rc = handle_hook(event, payload, cli_ctx.db_path, _emit=True)

        # The handler prints JSON to stdout when _emit=True
        # Just exit with the appropriate code
        raise typer.Exit(rc)


__all__ = ["register"]