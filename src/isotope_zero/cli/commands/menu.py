"""isotope_zero CLI — Menu Command.

Interactive onboarding menu with arrow navigation and command search.
"""

from __future__ import annotations

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
from ..ui.glyphs import glyph, can_unicode


def register(app: typer.Typer) -> None:
    """Register menu command with the typer app."""

    @app.command("menu", hidden=True)
    def menu_cmd(
        ctx: typer.Context,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """Interactive onboarding menu (default when running `izero` with no args)."""
        # Only run menu if no subcommand was invoked
        if ctx.invoked_subcommand is not None:
            return

        cli_ctx = inject_context(ctx, db, json, verbose, no_color)
        renderer = get_renderer_from_context(cli_ctx)

        from isotope_zero.core.store import MemoryStore
        store = MemoryStore(cli_ctx.db_path)

        # Get store info for banner
        count = store.count()
        db_path = cli_ctx.db_path

        banner = f"Your memories live at: {db_path}\n{count} cards remembered so far.\nWhat would you like to do?"

        entries = [
            ("add a memory", "add"),
            ("recall something", "recall"),
            ("search the store", "search"),
            ("list memories", "list"),
            ("get a memory", "get"),
            ("forget a memory", "forget"),
            ("touch (refresh) a memory", "touch"),
            ("see stats", "stats"),
            ("see tags", "tags"),
            ("inspect the store", "inspect"),
            ("dry-run consolidation", "dry-run-consolidation"),
            ("open the live dashboard", "dashboard"),
            ("exit", "exit"),
        ]

        if cli_ctx.json:
            print(renderer.render_menu(banner, entries, 0))
            return

        # Interactive mode
        if cli_ctx.use_rich and can_unicode():
            _run_rich_menu(renderer, banner, entries, cli_ctx)
        else:
            _run_plain_menu(renderer, banner, entries, cli_ctx)


def _run_rich_menu(renderer, banner: str, entries: list[tuple[str, str]], cli_ctx: CLIContext) -> None:
    """Run menu with rich TUI (arrow keys, search)."""
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.prompt import Prompt
        from ..ui.theme import get_console

        console = get_console()
        selected = 0
        search_query = ""
        filtered_entries = entries

        def render_menu():
            # Apply search filter
            nonlocal filtered_entries
            if search_query:
                filtered_entries = [
                    (label, action) for label, action in entries
                    if search_query.lower() in label.lower()
                ]
            else:
                filtered_entries = entries

            # Clamp selection
            nonlocal selected
            if selected >= len(filtered_entries):
                selected = max(0, len(filtered_entries) - 1)

            return renderer.render_menu(banner, filtered_entries, selected)

        with Live(render_menu(), console=console, refresh_per_second=30, screen=True) as live:
            while True:
                live.update(render_menu())

                # Get keypress
                import sys
                if sys.platform == "win32":
                    import msvcrt
                    key = msvcrt.getwch()
                    if key == '\xe0':  # Arrow key prefix
                        key = msvcrt.getwch()
                        if key == 'H':  # Up
                            selected = max(0, selected - 1)
                            continue
                        elif key == 'P':  # Down
                            selected = min(len(filtered_entries) - 1, selected + 1)
                            continue
                    elif key == '\r':  # Enter
                        action = filtered_entries[selected][1]
                        break
                    elif key == 'q' or key == 'Q':
                        return
                    elif key == '/':
                        # Search mode
                        search_query = Prompt.ask("Search", console=console, default="")
                        selected = 0
                        continue
                    elif key == '\x08':  # Backspace
                        search_query = search_query[:-1]
                        selected = 0
                        continue
                else:
                    # Unix - use termios
                    import termios
                    import tty
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setraw(fd)
                        key = sys.stdin.read(1)
                        if key == '\x1b':  # Escape sequence
                            key2 = sys.stdin.read(1)
                            if key2 == '[':
                                key3 = sys.stdin.read(1)
                                if key3 == 'A':  # Up
                                    selected = max(0, selected - 1)
                                    continue
                                elif key3 == 'B':  # Down
                                    selected = min(len(filtered_entries) - 1, selected + 1)
                                    continue
                        elif key == '\r' or key == '\n':  # Enter
                            action = filtered_entries[selected][1]
                            break
                        elif key == 'q' or key == 'Q':
                            return
                        elif key == '/':
                            search_query = Prompt.ask("Search", console=console, default="")
                            selected = 0
                            continue
                        elif key in ('\x7f', '\x08'):  # Backspace
                            search_query = search_query[:-1]
                            selected = 0
                            continue
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        # Execute selected action
        _execute_action(action, cli_ctx)

    except ImportError:
        # Fallback to plain
        _run_plain_menu(renderer, banner, entries, cli_ctx)


def _run_plain_menu(renderer, banner: str, entries: list[tuple[str, str]], cli_ctx: CLIContext) -> None:
    """Run menu in plain mode (numbered list)."""
    print(renderer.render_menu(banner, entries, -1))

    while True:
        try:
            choice = input("\nEnter number (1-13) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return

            idx = int(choice) - 1
            if 0 <= idx < len(entries):
                action = entries[idx][1]
                break
            else:
                print(f"Invalid choice. Enter 1-{len(entries)}")
        except (ValueError, EOFError, KeyboardInterrupt):
            return

    _execute_action(action, cli_ctx)


def _execute_action(action: str, cli_ctx: CLIContext) -> None:
    """Execute the selected menu action by invoking the corresponding command."""
    import subprocess
    import sys

    if action == "exit":
        return

    # Build command
    cmd = [sys.executable, "-m", "isotope_zero.cli.main", action]

    # Add global flags
    if cli_ctx.db_path and cli_ctx.db_path != ":memory:":
        cmd.extend(["--db", cli_ctx.db_path])
    if cli_ctx.json:
        cmd.append("--json")
    if cli_ctx.verbose:
        cmd.append("-v")
    if cli_ctx.no_color:
        cmd.append("--no-color")

    # For commands that need arguments, prompt for them
    if action in ("add", "recall", "search", "get", "forget", "touch"):
        if action == "add":
            fact = input("Fact to remember: ").strip()
            if not fact:
                return
            cmd.append(fact)
            evidence = input("Evidence (optional): ").strip()
            if evidence:
                cmd.extend(["--evidence", evidence])
            tags = input("Tags (comma-separated, optional): ").strip()
            if tags:
                cmd.extend(["--tags", tags])
        elif action in ("recall", "search"):
            query = input("Search query: ").strip()
            if not query:
                return
            cmd.append(query)
        elif action in ("get", "forget", "touch"):
            card_id = input("Card ID: ").strip()
            if not card_id:
                return
            cmd.append(card_id)
            if action == "forget":
                cmd.append("-y")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")


__all__ = ["register"]