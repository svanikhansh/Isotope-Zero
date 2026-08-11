"""isotope_zero CLI — Memory Commands.

Core memory operations: add, recall, search, list, get, forget, touch.
All commands use the new renderer system for consistent output.
"""

from __future__ import annotations

import typer
from typing import Optional, Annotated

from ..core.context import (
    CLIContext,
    inject_context,
    db_option,
    json_option,
    verbose_option,
    no_color_option,
    force_option,
    dry_run_option,
)
from ..core.completers import complete_tags, complete_scopes, complete_card_ids
from ..render import get_renderer_from_context
from ..ui.glyphs import glyph


def register(app: typer.Typer) -> None:
    """Register memory commands with the typer app."""

    # --- add ---
    @app.command("add")
    def add_cmd(
        ctx: typer.Context,
        fact: Annotated[str, typer.Argument(help="The fact to remember")],
        evidence: Annotated[Optional[str], typer.Option("--evidence", "-e", help="Supporting evidence")] = None,
        tags: Annotated[Optional[str], typer.Option("--tags", "-t", help="Comma-separated tags")] = None,
        scope: Annotated[str, typer.Option("--scope", "-s", help="Memory scope")] = "default",
        id: Annotated[Optional[str], typer.Option("--id", help="Explicit card ID")] = None,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
        force: bool = force_option(),
    ):
        """Remember a fact."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color, False, force)
        renderer = get_renderer_from_context(cli_ctx)

        # Parse tags
        tag_list = [t.strip() for t in tags.split(",")] if tags else []

        # Use IsotopeZero client for proper embedding + storage
        from isotope_zero.client import IsotopeZero
        client = IsotopeZero(db_path=cli_ctx.db_path)

        # Add the memory
        card_id = client.remember(
            fact=fact,
            evidence=evidence or "",
            tags=tag_list,
            scope=scope,
            card_id=id,
        )

        # Output
        if cli_ctx.json:
            print(renderer.render_add_result(card_id, True))
        else:
            print(renderer.render_add_result(card_id, True))

    # --- recall ---
    @app.command("recall")
    def recall_cmd(
        ctx: typer.Context,
        query: Annotated[str, typer.Argument(help="Search query")],
        k: Annotated[int, typer.Option("--k", "-k", help="Number of results")] = 10,
        alpha: Annotated[float, typer.Option("--alpha", "-a", help="Vector weight (0-1)")] = 0.7,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """Semantic retrieval — find memories by meaning."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color)
        renderer = get_renderer_from_context(cli_ctx)

        from isotope_zero.client import IsotopeZero
        client = IsotopeZero(db_path=cli_ctx.db_path, spawn_daemon=False, use_mmap=False)

        results = client.recall(query, k=k, alpha=alpha)

        # Convert to MemoryRow
        rows = []
        for i, r in enumerate(results):
            from ..ui.components import MemoryRow
            rows.append(MemoryRow(
                rank=i + 1,
                id=r.get("id", ""),
                fact=r.get("fact", ""),
                score=r.get("score"),
                vitality=r.get("vitality"),
                tags=r.get("tags", []),
                age_days=None,
                evidence=r.get("evidence", ""),
                timestamp=r.get("timestamp"),
                access_count=0,
                last_access=None,
                source_tokens=0,
            ))

        if cli_ctx.json:
            print(renderer.render_memory_rows(rows, show_score=True, show_vitality=True))
        else:
            print(renderer.render_memory_rows(rows, show_score=cli_ctx.verbose, show_vitality=cli_ctx.verbose))
        client.close()

    # --- search ---
    @app.command("search")
    def search_cmd(
        ctx: typer.Context,
        query: Annotated[str, typer.Argument(help="Search query")],
        k: Annotated[int, typer.Option("--k", "-k", help="Number of results")] = 10,
        alpha: Annotated[float, typer.Option("--alpha", "-a", help="Vector weight (0-1)")] = 0.7,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """Hybrid search — vector + BM25 + graph."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color)
        renderer = get_renderer_from_context(cli_ctx)

        from isotope_zero.client import IsotopeZero
        client = IsotopeZero(db_path=cli_ctx.db_path, spawn_daemon=False, use_mmap=False)

        results = client.search(query, k=k, alpha=alpha)

        rows = []
        for i, r in enumerate(results):
            from ..ui.components import MemoryRow
            rows.append(MemoryRow(
                rank=i + 1,
                id=r.get("id", ""),
                fact=r.get("fact", ""),
                score=r.get("score"),
                vitality=r.get("vitality"),
                tags=r.get("tags", []),
                age_days=None,
                evidence=r.get("evidence", ""),
                timestamp=r.get("timestamp"),
                access_count=0,
                last_access=None,
                source_tokens=0,
            ))

        if cli_ctx.json:
            print(renderer.render_memory_rows(rows, show_score=True, show_vitality=True))
        else:
            print(renderer.render_memory_rows(rows, show_score=cli_ctx.verbose, show_vitality=cli_ctx.verbose))
        client.close()

    # --- list ---
    @app.command("list")
    def list_cmd(
        ctx: typer.Context,
        tags: Annotated[Optional[str], typer.Option("--tags", "-t", help="Filter by comma-separated tags", autocompletion=complete_tags)] = None,
        scope: Annotated[Optional[str], typer.Option("--scope", "-s", help="Filter by scope", autocompletion=complete_scopes)] = None,
        limit: Annotated[int, typer.Option("--limit", "-l", help="Maximum results")] = 50,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """List all memories, newest first."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color)
        renderer = get_renderer_from_context(cli_ctx)

        from isotope_zero.client import IsotopeZero
        client = IsotopeZero(db_path=cli_ctx.db_path, spawn_daemon=False, use_mmap=False)

        tag_list = [t.strip() for t in tags.split(",")] if tags else None

        cards = client.store.all()

        # Filter by tags and scope
        if tag_list:
            cards = [c for c in cards if any(t in c.tags for t in tag_list)]
        if scope:
            cards = [c for c in cards if c.scope == scope]

        # Limit and reverse (newest first)
        cards = cards[-limit:] if limit > 0 else cards
        cards.reverse()

        rows = []
        for i, c in enumerate(cards):
            from ..ui.components import MemoryRow
            rows.append(MemoryRow(
                rank=i + 1,
                id=c.id,
                fact=c.fact,
                tags=c.tags,
                score=None,
                vitality=c.stability,
                age_days=None,
                evidence=c.evidence,
                timestamp=c.timestamp,
                access_count=c.access_count,
                last_access=c.last_access,
                source_tokens=c.source_tokens,
            ))

        if cli_ctx.json:
            print(renderer.render_memory_rows(rows, show_vitality=True))
        else:
            print(renderer.render_memory_rows(rows, show_vitality=cli_ctx.verbose))
        client.close()

    # --- get ---
    @app.command("get")
    def get_cmd(
        ctx: typer.Context,
        card_id: Annotated[str, typer.Argument(help="Card ID", autocompletion=complete_card_ids)],
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """Get full card detail by ID."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color)
        renderer = get_renderer_from_context(cli_ctx)

        from isotope_zero.client import IsotopeZero
        client = IsotopeZero(db_path=cli_ctx.db_path, spawn_daemon=False, use_mmap=False)

        card = client.store.get(card_id)
        if not card:
            if cli_ctx.json:
                print(renderer.render_json({"error": f"Card not found: {card_id}"}))
            else:
                print(f"{glyph('cross')} Card not found: {card_id}")
            client.close()
            raise typer.Exit(1)

        from ..ui.components import MemoryRow
        row = MemoryRow(
            rank=1,
            id=card.id,
            fact=card.fact,
            tags=card.tags,
            scope=card.scope,
            score=None,
            vitality=card.stability,
            age_days=None,
            evidence=card.evidence,
            timestamp=card.timestamp,
            access_count=card.access_count,
            last_access=card.last_access,
            source_tokens=card.source_tokens,
        )

        if cli_ctx.json:
            print(renderer.render_memory_card(row, verbose=True))
        else:
            print(renderer.render_memory_card(row, verbose=cli_ctx.verbose))
        client.close()

    # --- forget ---
    @app.command("forget")
    def forget_cmd(
        ctx: typer.Context,
        card_id: Annotated[str, typer.Argument(help="Card ID to delete", autocompletion=complete_card_ids)],
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
        force: bool = force_option(),
    ):
        """Delete a memory card."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color, False, force)
        renderer = get_renderer_from_context(cli_ctx)

        # Confirm unless forced
        if not cli_ctx.force:
            from ..ui.glyphs import glyph
            confirm = typer.confirm(f"{glyph('warning')} Delete card {card_id[:12]}?")
            if not confirm:
                print("Cancelled")
                raise typer.Exit(0)

        from isotope_zero.client import IsotopeZero
        client = IsotopeZero(db_path=cli_ctx.db_path, spawn_daemon=False, use_mmap=False)

        # Check if card exists first
        card = client.store.get(card_id)
        if not card:
            if cli_ctx.json:
                print(renderer.render_json({"error": f"Card not found: {card_id}"}))
            else:
                print(f"{glyph('cross')} Card not found: {card_id}")
            client.close()
            raise typer.Exit(1)

        # Delete the card
        client.store.delete(card_id)

        if cli_ctx.json:
            print(renderer.render_forget_result(card_id))
        else:
            print(renderer.render_forget_result(card_id))
        client.close()

    # --- touch ---
    @app.command("touch")
    def touch_cmd(
        ctx: typer.Context,
        card_id: Annotated[str, typer.Argument(help="Card ID to refresh", autocompletion=complete_card_ids)],
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """Refresh access tracking for a card (bumps vitality)."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color)
        renderer = get_renderer_from_context(cli_ctx)

        from isotope_zero.client import IsotopeZero
        client = IsotopeZero(db_path=cli_ctx.db_path, spawn_daemon=False, use_mmap=False)

        success = client.touch(card_id)
        if not success:
            if cli_ctx.json:
                print(renderer.render_json({"error": f"Card not found: {card_id}"}))
            else:
                print(f"{glyph('cross')} Card not found: {card_id}")
            client.close()
            raise typer.Exit(1)

        if cli_ctx.json:
            print(renderer.render_touch_result(card_id))
        else:
            print(renderer.render_touch_result(card_id))
        client.close()


# Re-export for compatibility
__all__ = ["register"]