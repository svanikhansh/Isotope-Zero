"""isotope_zero CLI — Plugin Command.

Editor plugin installation and management.
"""

from __future__ import annotations

import typer
from typing import Annotated, Optional
from pathlib import Path

from ..core.context import (
    CLIContext,
    inject_context,
    db_option,
    json_option,
    verbose_option,
    no_color_option,
    dry_run_option,
    force_option,
)
from ..render import get_renderer_from_context
from ..ui.glyphs import glyph


def register(app: typer.Typer) -> None:
    """Register plugin command with the typer app."""

    plugin_app = typer.Typer(help="Editor plugin management")
    app.add_typer(plugin_app, name="plugin")

    @plugin_app.command("install")
    def install_cmd(
        ctx: typer.Context,
        target: Annotated[Optional[str], typer.Option("--target", "-t", help="Target directory (default: auto-detect)")] = None,
        editor: Annotated[Optional[str], typer.Option("--editor", "-e", help="Editor: vscode, cursor, windsurf, zed")] = None,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
        dry_run: bool = dry_run_option(),
        force: bool = force_option(),
    ):
        """Install the editor plugin."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color, dry_run, force)
        renderer = get_renderer_from_context(cli_ctx)

        from isotope_zero.integrations import install_plugin

        # Auto-detect editor if not specified
        if editor is None:
            editor = _detect_editor()

        if cli_ctx.json:
            result = install_plugin(editor=editor, target=target, dry_run=cli_ctx.dry_run)
            print(renderer.render_json(result))
        else:
            print(f"{glyph('rocket')} Installing {editor or 'auto-detected'} plugin...")
            result = install_plugin(editor=editor, target=target, dry_run=cli_ctx.dry_run)

            if result.get("success"):
                print(f"{glyph('ok')} Plugin installed to {result.get('path', 'unknown')}")
                if result.get("files"):
                    for f in result["files"]:
                        print(f"  {glyph('bullet')} {f}")
            else:
                print(f"{glyph('cross')} Install failed: {result.get('error', 'Unknown error')}")
                raise typer.Exit(1)

    @plugin_app.command("uninstall")
    def uninstall_cmd(
        ctx: typer.Context,
        editor: Annotated[Optional[str], typer.Option("--editor", "-e", help="Editor: vscode, cursor, windsurf, zed")] = None,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
        force: bool = force_option(),
    ):
        """Uninstall the editor plugin."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color, False, force)
        renderer = get_renderer_from_context(cli_ctx)

        if editor is None:
            editor = _detect_editor()

        if not cli_ctx.force:
            confirm = typer.confirm(f"{glyph('warning')} Uninstall {editor} plugin?")
            if not confirm:
                print("Cancelled")
                raise typer.Exit(0)

        from isotope_zero.integrations import uninstall_plugin

        if cli_ctx.json:
            result = uninstall_plugin(editor=editor)
            print(renderer.render_json(result))
        else:
            result = uninstall_plugin(editor=editor)
            if result.get("success"):
                print(f"{glyph('ok')} Plugin uninstalled")
            else:
                print(f"{glyph('cross')} Uninstall failed: {result.get('error', 'Unknown error')}")
                raise typer.Exit(1)

    @plugin_app.command("status")
    def status_cmd(
        ctx: typer.Context,
        editor: Annotated[Optional[str], typer.Option("--editor", "-e", help="Editor: vscode, cursor, windsurf, zed")] = None,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """Check plugin installation status."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color)
        renderer = get_renderer_from_context(cli_ctx)

        if editor is None:
            editor = _detect_editor()

        from isotope_zero.integrations import plugin_status

        result = plugin_status(editor=editor)

        if cli_ctx.json:
            print(renderer.render_json(result))
        else:
            if result.get("installed"):
                print(f"{glyph('ok')} {editor} plugin installed at {result.get('path')}")
                print(f"  Version: {result.get('version')}")
            else:
                print(f"{glyph('cross')} {editor} plugin not installed")


def _detect_editor() -> str:
    """Auto-detect the editor from environment."""
    import os

    # Check common editor env vars
    if "VSCODE_PID" in os.environ or "TERM_PROGRAM" in os.environ and os.environ.get("TERM_PROGRAM") == "vscode":
        return "vscode"
    if "CURSOR_TRACE_ID" in os.environ:
        return "cursor"
    if "WINDSURF" in os.environ:
        return "windsurf"
    if "ZED" in os.environ:
        return "zed"

    # Default
    return "vscode"


__all__ = ["register"]