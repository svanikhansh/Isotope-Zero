"""isotope_zero CLI — Inspect Commands.

Analysis and diagnostic commands: inspect, dry-run-consolidation, stats, tags.
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
    dry_run_option,
)
from ..render import get_renderer_from_context
from ..ui.glyphs import glyph


def register(app: typer.Typer) -> None:
    """Register inspect commands with the typer app."""

    # --- stats ---
    @app.command("stats")
    def stats_cmd(
        ctx: typer.Context,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """Store overview: count, size, embedding mode, tokens."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color)
        renderer = get_renderer_from_context(cli_ctx)

        from isotope_zero.client import IsotopeZero
        from isotope_zero.tokens import estimate_tokens
        from isotope_zero.core.consolidation import Consolidator
        from isotope_zero.types import now_ts

        client = IsotopeZero(db_path=cli_ctx.db_path, spawn_daemon=False, use_mmap=False)

        cards = client.store.all()
        count = client.count()
        size_bytes = client.store.db_size_bytes()

        # Embedding mode
        emb = getattr(client.store, "embedder", None)
        if emb is None:
            mode = "none (no embedder attached)"
        elif getattr(emb, "is_real", False):
            mode = "REAL ONNX"
        else:
            mode = "FALLBACK"

        tokens = sum(estimate_tokens(c.fact) + estimate_tokens(c.evidence) for c in cards)

        # Tag distribution (count desc, tag asc tiebreak).
        tag_counts: dict[str, int] = {}
        for card in cards:
            for tag in card.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        tag_dist = dict(sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0])))

        # Vitality histogram: fresh / aging / decayed buckets.
        cons = Consolidator(client.store)
        now = now_ts()
        fresh = aging = decayed = 0
        for card in cards:
            v = cons.vitality(card, now=now)
            if v >= 0.66:
                fresh += 1
            elif v >= 0.33:
                aging += 1
            else:
                decayed += 1
        histogram = {"fresh": fresh, "aging": aging, "decayed": decayed}

        # Convert tag_dist to TagCount
        from ..ui.components import TagCount, VitalityHistogram
        tag_counts_list = [TagCount(tag=t, count=c) for t, c in tag_dist.items()]

        # Convert histogram
        vit_hist = VitalityHistogram(fresh=fresh, aging=aging, decayed=decayed)

        if cli_ctx.json:
            print(renderer.render_json({
                "count": count,
                "db_size_bytes": size_bytes,
                "embedding_mode": mode,
                "token_footprint": tokens,
                "tags": tag_dist,
                "tag_distribution": tag_dist,
                "vitality_histogram": histogram,
                "vitality_buckets": {"fresh": ">=0.66", "aging": "0.33-0.66", "decayed": "<0.33"},
            }))
        else:
            print(renderer.render_stats(
                count, size_bytes, mode, tokens,
                tag_counts_list, vit_hist, verbose=cli_ctx.verbose
            ))
        client.close()

    # --- tags ---
    @app.command("tags")
    def tags_cmd(
        ctx: typer.Context,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """Tag distribution."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color)
        renderer = get_renderer_from_context(cli_ctx)

        from isotope_zero.client import IsotopeZero
        client = IsotopeZero(db_path=cli_ctx.db_path, spawn_daemon=False, use_mmap=False)

        cards = client.store.all()

        # Tag distribution (count desc, tag asc tiebreak).
        tag_counts: dict[str, int] = {}
        for card in cards:
            for tag in card.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        tag_dist = dict(sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0])))

        from ..ui.components import TagCount
        tag_counts_list = [TagCount(tag=t, count=c) for t, c in tag_dist.items()]

        if cli_ctx.json:
            print(renderer.render_json({
                "tags": tag_dist,
                "tag_distribution": tag_dist,
            }))
        else:
            print(renderer.render_tags(tag_counts_list, verbose=cli_ctx.verbose))
        client.close()

    # --- inspect ---
    @app.command("inspect")
    def inspect_cmd(
        ctx: typer.Context,
        top: Annotated[int, typer.Option("--top", help="Number of decay candidates to show")] = 10,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
    ):
        """Diagnostic store report."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color)
        renderer = get_renderer_from_context(cli_ctx)

        from isotope_zero.client import IsotopeZero
        from isotope_zero.core.consolidation import Consolidator
        from isotope_zero.types import now_ts

        client = IsotopeZero(db_path=cli_ctx.db_path, spawn_daemon=False, use_mmap=False)

        cards = client.store.all()
        total = client.count()
        size_bytes = client.store.db_size_bytes()

        # Embedding mode
        emb = getattr(client.store, "embedder", None)
        if emb is None:
            embedding_mode = "none (no embedder attached)"
        elif getattr(emb, "is_real", False):
            embedding_mode = "REAL ONNX"
        else:
            embedding_mode = "FALLBACK"

        # Average embedding dim
        cards_with_emb = [c for c in cards if c.embedding is not None and len(c.embedding) > 0]
        avg_dim = sum(len(c.embedding) for c in cards_with_emb) / len(cards_with_emb) if cards_with_emb else None

        # Decay candidates
        cons = Consolidator(client.store)
        now = now_ts()
        decay_rows = []
        for card in cards:
            v = cons.vitality(card, now=now)
            decay_rows.append({
                "id": card.id,
                "fact": card.fact,
                "vitality": v,
                "access_count": card.access_count,
                "age_days": round(max(0.0, now - card.timestamp) / 86400, 1),
            })
        decay_rows.sort(key=lambda r: r["vitality"])
        decay_rows = decay_rows[:top]

        tokens = sum(1 for c in cards for _ in (c.fact.split() + c.evidence.split()))

        if cli_ctx.json:
            print(renderer.render_inspect(
                cli_ctx.db_path, total, size_bytes, embedding_mode,
                avg_dim, len(cards_with_emb), decay_rows, tokens
            ))
        else:
            print(renderer.render_inspect(
                cli_ctx.db_path, total, size_bytes, embedding_mode,
                avg_dim, len(cards_with_emb), decay_rows, tokens
            ))
        client.close()

    # --- dry-run-consolidation ---
    @app.command("dry-run-consolidation")
    def dry_run_consolidation_cmd(
        ctx: typer.Context,
        limit: Annotated[int, typer.Option("--limit", "-l", help="Limit number of cards to show")] = 0,
        db: str = db_option(),
        json: bool = json_option(),
        verbose: bool = verbose_option(),
        no_color: bool = no_color_option(),
        dry_run: bool = dry_run_option(),
    ):
        """Preview consolidation without making changes."""
        cli_ctx = inject_context(ctx, db, json, verbose, no_color, dry_run)
        renderer = get_renderer_from_context(cli_ctx)

        from isotope_zero.client import IsotopeZero
        from isotope_zero.core.consolidation import Consolidator

        client = IsotopeZero(db_path=cli_ctx.db_path, spawn_daemon=False, use_mmap=False)

        consolidator = Consolidator(client.store, embedder=client.engine)
        plan = consolidator.dry_run()

        if cli_ctx.json:
            print(renderer.render_dry_run(plan, limit=limit))
        else:
            print(renderer.render_dry_run(plan, limit=limit))
        client.close()


__all__ = ["register"]