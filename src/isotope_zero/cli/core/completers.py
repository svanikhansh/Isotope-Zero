"""isotope_zero CLI — Tab Completion.

Provides shell completions for tags, scopes, card IDs, and commands.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

import typer
from click.shell_completion import CompletionItem


def complete_tags(ctx: typer.Context, incomplete: str) -> Iterator[CompletionItem]:
    """Complete tag names from the database."""
    from ...core.context import get_context

    cli_ctx = get_context()
    if not cli_ctx.db:
        # Try to load the store
        try:
            from isotope_zero import MemoryStore
            cli_ctx.db = MemoryStore(cli_ctx.db_path)
        except Exception:
            return

    try:
        # Get all tags from the store
        all_tags = cli_ctx.db.get_all_tags()
        for tag in all_tags:
            if tag.startswith(incomplete):
                yield CompletionItem(tag, help=f"Tag: {tag}")
    except Exception:
        pass


def complete_scopes(ctx: typer.Context, incomplete: str) -> Iterator[CompletionItem]:
    """Complete scope names from the database."""
    from ...core.context import get_context

    cli_ctx = get_context()
    if not cli_ctx.db:
        try:
            from isotope_zero import MemoryStore
            cli_ctx.db = MemoryStore(cli_ctx.db_path)
        except Exception:
            return

    try:
        all_scopes = cli_ctx.db.get_all_scopes()
        for scope in all_scopes:
            if scope.startswith(incomplete):
                yield CompletionItem(scope, help=f"Scope: {scope}")
    except Exception:
        pass


def complete_card_ids(ctx: typer.Context, incomplete: str) -> Iterator[CompletionItem]:
    """Complete card IDs (prefix match)."""
    from ...core.context import get_context

    cli_ctx = get_context()
    if not cli_ctx.db:
        try:
            from isotope_zero import MemoryStore
            cli_ctx.db = MemoryStore(cli_ctx.db_path)
        except Exception:
            return

    try:
        # Get recent card IDs
        cards = cli_ctx.db.list_cards(limit=100)
        for card in cards:
            cid = card.get("id", "")
            if cid.startswith(incomplete):
                fact = card.get("fact", "")[:50]
                yield CompletionItem(cid, help=fact)
    except Exception:
        pass


def complete_commands(ctx: typer.Context, incomplete: str) -> Iterator[CompletionItem]:
    """Complete subcommand names."""
    commands = [
        ("add", "Remember a fact"),
        ("recall", "Semantic retrieval"),
        ("search", "Hybrid search (vector+BM25+graph)"),
        ("list", "All cards, newest-first"),
        ("get", "Full card detail"),
        ("forget", "Delete a card"),
        ("touch", "Refresh access tracking"),
        ("tags", "Tag distribution"),
        ("stats", "Store overview"),
        ("inspect", "Diagnostic store report"),
        ("dry-run-consolidation", "Preview consolidation"),
        ("dashboard", "Live TUI overview"),
        ("hook", "Claude Code lifecycle hook"),
        ("plugin", "Editor plugin management"),
    ]

    for name, help_text in commands:
        if name.startswith(incomplete):
            yield CompletionItem(name, help=help_text)


# Export all completers
__all__ = [
    "complete_tags",
    "complete_scopes",
    "complete_card_ids",
    "complete_commands",
]