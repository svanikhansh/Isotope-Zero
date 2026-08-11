"""isotope_zero CLI — JSON Renderer.

Machine-readable output for scripting and automation.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import BaseRenderer
from ..ui.components import (
    MemoryRow,
    StatRow,
    TagCount,
    VitalityHistogram,
)


class JsonRenderer(BaseRenderer):
    """JSON renderer — pure JSON output for all commands."""

    def render_memory_rows(
        self,
        rows: Sequence[MemoryRow],
        show_score: bool = False,
        show_vitality: bool = False,
        show_tags: bool = True,
    ) -> str:
        data = []
        for row in rows:
            item = {
                "id": row.id,
                "fact": row.fact,
                "tags": row.tags,
                "scope": row.scope,
            }
            if show_score and row.score is not None:
                item["score"] = row.score
            if show_vitality and row.vitality is not None:
                item["vitality"] = row.vitality
            if row.created_at:
                item["created_at"] = row.created_at
            if row.accessed_at:
                item["accessed_at"] = row.accessed_at
            data.append(item)
        return self.render_json(data)

    def render_memory_card(self, card: MemoryRow, verbose: bool = False) -> str:
        data = {
            "id": card.id,
            "fact": card.fact,
            "evidence": card.evidence,
            "tags": card.tags,
            "scope": card.scope,
        }
        if verbose:
            if card.score is not None:
                data["score"] = card.score
            if card.vitality is not None:
                data["vitality"] = card.vitality
            if card.created_at:
                data["created_at"] = card.created_at
            if card.accessed_at:
                data["accessed_at"] = card.accessed_at
            if card.access_count is not None:
                data["access_count"] = card.access_count
        return self.render_json(data)

    def render_stats(
        self,
        count: int,
        size_bytes: int,
        embedding_mode: str,
        tokens: int,
        tag_dist: Sequence[TagCount] = (),
        histogram: VitalityHistogram | None = None,
        verbose: bool = False,
    ) -> str:
        data = {
            "count": count,
            "size_bytes": size_bytes,
            "embedding_mode": embedding_mode,
            "tokens": tokens,
        }
        if verbose:
            data["tags"] = [
                {"tag": t.tag, "count": t.count} for t in tag_dist
            ]
            if histogram:
                data["vitality_histogram"] = {
                    "fresh": histogram.fresh,
                    "aging": histogram.aging,
                    "decayed": histogram.decayed,
                }
        return self.render_json(data)

    def render_tags(self, tag_counts: Sequence[TagCount], verbose: bool = False) -> str:
        data = [
            {"tag": t.tag, "count": t.count}
            for t in tag_counts
        ]
        return self.render_json(data)

    def render_inspect(
        self,
        db_path: str,
        total: int,
        size_bytes: int,
        embedding_mode: str,
        avg_dim: float | None,
        cards_with_emb: int,
        decay_rows: list[dict],
        tokens: int,
    ) -> str:
        data = {
            "db_path": db_path,
            "total_cards": total,
            "size_bytes": size_bytes,
            "embedding_mode": embedding_mode,
            "avg_dim": avg_dim,
            "cards_with_embeddings": cards_with_emb,
            "tokens": tokens,
            "decay_candidates": decay_rows,
        }
        return self.render_json(data)

    def render_dry_run(self, plan: dict[str, Any], limit: int = 0) -> str:
        if limit > 0:
            plan = {**plan, "cards": plan.get("cards", [])[:limit]}
        return self.render_json(plan)

    def render_dashboard(self, state: dict[str, Any], interval: float) -> str:
        # Dashboard JSON is just the state
        return self.render_json(state)

    def render_menu(self, banner: str, entries: list[tuple[str, Any]], selected_idx: int) -> str:
        data = {
            "banner": banner,
            "entries": [{"label": e[0], "action": e[1]} for e in entries],
            "selected_index": selected_idx,
        }
        return self.render_json(data)

    def render_add_result(self, card_id: str, created: bool) -> str:
        return self.render_json({"id": card_id, "created": created})

    def render_forget_result(self, card_id: str) -> str:
        return self.render_json({"id": card_id, "deleted": True})

    def render_touch_result(self, card_id: str) -> str:
        return self.render_json({"id": card_id, "touched": True})