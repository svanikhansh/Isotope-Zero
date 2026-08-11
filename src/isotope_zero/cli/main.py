"""isotope_zero CLI — Main Entry Point.

Typer-based CLI with global flags, subcommands, and the new renderer system.
"""

from __future__ import annotations

import os
import sys
import typer
from typing import Optional, Annotated

from .core.context import (
    CLIContext,
    inject_context,
    db_option,
    json_option,
    verbose_option,
    no_color_option,
)
from .core.hooks import get_hook_registry, TimingHook
from .commands import register_commands
from .render import get_renderer_from_context
from .ui.glyphs import glyph
from .ui.theme import get_console, get_theme


# Create the main typer app
app = typer.Typer(
    name="izero",
    help="isotope_zero — local-first cognitive memory layer for AI agents",
    add_completion=True,
    no_args_is_help=False,  # We handle no-args in menu callback
    rich_markup_mode="rich",
)

# Register global hooks
get_hook_registry().register(TimingHook())


# Global options callback
@app.callback(invoke_without_command=True)
def global_callback(
    ctx: typer.Context,
    db: str = db_option(),
    json: bool = json_option(),
    verbose: bool = verbose_option(),
    no_color: bool = no_color_option(),
    version: Annotated[bool, typer.Option("--version", "-V", help="Show version and exit")] = False,
    antigravity: Annotated[bool, typer.Option("--antigravity", help="Open XKCD 353")] = False,
    zen: Annotated[bool, typer.Option("--zen", help="Print a programming koan")] = False,
):
    """Global options and easter eggs."""
    # Handle version
    if version:
        from isotope_zero import __version__
        print(f"isotope_zero {__version__}")
        if verbose:
            print("""
memory persists
through silicon and time
recall is truth
""")
        raise typer.Exit(0)

    # Handle antigravity easter egg
    if antigravity:
        import webbrowser
        webbrowser.open("https://xkcd.com/353/")
        raise typer.Exit(0)

    # Handle zen easter egg
    if zen:
        koans = [
            "The best code is no code at all.",
            "A bug found in production is a lesson learned twice.",
            "Premature optimization is the root of all evil — but so is premature pessimization.",
            "The only constant in software is change. The only constant in memory is decay.",
            "To remember everything is to remember nothing.",
        ]
        import random
        print(random.choice(koans))
        raise typer.Exit(0)

    # Create base context (will be enriched by subcommands)
    cli_ctx = CLIContext(
        db_path=db,
        json=json,
        verbose=verbose,
        no_color=no_color,
    )
    ctx.obj = cli_ctx


# Register all command modules
register_commands(app)


# Custom help with theme
def _print_rich_help():
    """Print rich-styled help."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from .ui.theme import get_console

    console = get_console()

    panel = Panel(
        Text.assemble(
            ("\n", ""),
            ("isotope_zero", "izero.hero"),
            (" ", ""),
            ("local-first cognitive memory layer", "izero.subtitle"),
            ("\n\n", ""),
            ("Commands:\n", "izero.dim"),
            ("  add          ", "izero.body"),
            ("Remember a fact\n", "izero.dim"),
            ("  recall       ", "izero.body"),
            ("Semantic retrieval\n", "izero.dim"),
            ("  search       ", "izero.body"),
            ("Hybrid search (vector+BM25+graph)\n", "izero.dim"),
            ("  list         ", "izero.body"),
            ("All cards, newest first\n", "izero.dim"),
            ("  get          ", "izero.body"),
            ("Full card detail\n", "izero.dim"),
            ("  forget       ", "izero.body"),
            ("Delete a card\n", "izero.dim"),
            ("  touch        ", "izero.body"),
            ("Refresh access tracking\n", "izero.dim"),
            ("  stats        ", "izero.body"),
            ("Store overview\n", "izero.dim"),
            ("  tags         ", "izero.body"),
            ("Tag distribution\n", "izero.dim"),
            ("  inspect      ", "izero.body"),
            ("Diagnostic report\n", "izero.dim"),
            ("  dry-run-consolidation", "izero.body"),
            ("Preview consolidation\n", "izero.dim"),
            ("  dashboard    ", "izero.body"),
            ("Live TUI overview\n", "izero.dim"),
            ("  hook         ", "izero.body"),
            ("Claude Code lifecycle hook\n", "izero.dim"),
            ("  plugin       ", "izero.body"),
            ("Editor plugin management\n", "izero.dim"),
            ("\n", ""),
            ("Global flags:\n", "izero.dim"),
            ("  --db PATH    ", "izero.body"),
            ("Database path\n", "izero.dim"),
            ("  --json       ", "izero.body"),
            ("Machine-readable output\n", "izero.dim"),
            ("  -v, --verbose", "izero.body"),
            ("Technical detail\n", "izero.dim"),
            ("  --no-color   ", "izero.body"),
            ("Disable colors\n", "izero.dim"),
            ("  --version    ", "izero.body"),
            ("Show version\n", "izero.dim"),
            ("  --antigravity", "izero.body"),
            ("Open XKCD 353\n", "izero.dim"),
            ("  --zen        ", "izero.body"),
            ("Programming koan\n", "izero.dim"),
        ),
        title="izero",
        style="izero.panel",
        border_style="izero.panel.border",
    )
    console.print(panel)


# Override help
@app.command("help", hidden=True)
def help_cmd(
    ctx: typer.Context,
    rich: Annotated[bool, typer.Option("--rich", help="Show rich help")] = True,
):
    """Show help."""
    if rich:
        _print_rich_help()
    else:
        print(ctx.get_help())


def main() -> None:
    """Main entry point for setuptools console_scripts."""
    try:
        app()
    except KeyboardInterrupt:
        print(f"\n{glyph('hint')} Interrupted")
        sys.exit(130)
    except Exception as e:
        # Get context for json output
        from .core.context import get_context
        cli_ctx = get_context()
        if cli_ctx.json:
            import json
            print(json.dumps({"error": str(e), "type": type(e).__name__}))
        else:
            print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()