"""isotope_zero CLI — Render protocol and base classes.

Defines the renderer interface that all output formats implement.
This ensures consistent contracts across JSON, plain, and rich output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, Sequence, runtime_checkable

from ..ui.components import (
    MemoryRow,
    StatRow,
    TagCount,
    VitalityHistogram,
)


@runtime_checkable
class Renderer(Protocol):
    """Protocol for all renderers. Each output format implements this."""

    # Memory commands
    def render_memory_rows(
        self,
        rows: Sequence[MemoryRow],
        show_score: bool = False,
        show_vitality: bool = False,
        show_tags: bool = True,
    ) -> str: ...

    def render_memory_card(self, card: MemoryRow, verbose: bool = False) -> str: ...

    # Stats/inspect
    def render_stats(
        self,
        count: int,
        size_bytes: int,
        embedding_mode: str,
        tokens: int,
        tag_dist: Sequence[TagCount] = (),
        histogram: VitalityHistogram | None = None,
        verbose: bool = False,
    ) -> str: ...

    def render_tags(self, tag_counts: Sequence[TagCount], verbose: bool = False) -> str: ...

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
    ) -> str: ...

    def render_dry_run(self, plan: dict[str, Any], limit: int = 0) -> str: ...

    # Dashboard
    def render_dashboard(self, state: dict[str, Any], interval: float) -> str: ...

    # Menu
    def render_menu(self, banner: str, entries: list[tuple[str, Any]], selected_idx: int) -> str: ...

    # Actions
    def render_add_result(self, card_id: str, created: bool) -> str: ...

    def render_forget_result(self, card_id: str) -> str: ...

    def render_touch_result(self, card_id: str) -> str: ...

    # JSON (passthrough)
    def render_json(self, obj: Any) -> str: ...


class BaseRenderer(ABC):
    """Base renderer with shared utilities."""

    def __init__(self, use_color: bool = True, verbose: bool = False):
        self.use_color = use_color
        self.verbose = verbose

    @abstractmethod
    def render_memory_rows(
        self,
        rows: Sequence[MemoryRow],
        show_score: bool = False,
        show_vitality: bool = False,
        show_tags: bool = True,
    ) -> str:
        pass

    @abstractmethod
    def render_memory_card(self, card: MemoryRow, verbose: bool = False) -> str:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def render_tags(self, tag_counts: Sequence[TagCount], verbose: bool = False) -> str:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def render_dry_run(self, plan: dict[str, Any], limit: int = 0) -> str:
        pass

    @abstractmethod
    def render_dashboard(self, state: dict[str, Any], interval: float) -> str:
        pass

    @abstractmethod
    def render_menu(self, banner: str, entries: list[tuple[str, Any]], selected_idx: int) -> str:
        pass

    @abstractmethod
    def render_add_result(self, card_id: str, created: bool) -> str:
        pass

    @abstractmethod
    def render_forget_result(self, card_id: str) -> str:
        pass

    @abstractmethod
    def render_touch_result(self, card_id: str) -> str:
        pass

    def render_json(self, obj: Any) -> str:
        """JSON rendering is the same for all renderers."""
        import json
        return json.dumps(obj, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Renderer Factory
# --------------------------------------------------------------------------- #
def create_renderer(
    format: str,
    use_color: bool = True,
    verbose: bool = False,
) -> Renderer:
    """Create a renderer by format name.

    Args:
        format: One of "json", "plain", "rich"
        use_color: Whether to use colors (ignored for json)
        verbose: Whether to show verbose output by default

    Returns:
        Renderer instance.
    """
    if format == "json":
        from .json_renderer import JsonRenderer
        return JsonRenderer(use_color=use_color, verbose=verbose)
    elif format == "plain":
        from .plain_renderer import PlainRenderer
        return PlainRenderer(use_color=use_color, verbose=verbose)
    elif format == "rich":
        from .rich_renderer import RichRenderer
        return RichRenderer(use_color=use_color, verbose=verbose)
    else:
        raise ValueError(f"Unknown renderer format: {format}")


def get_renderer_from_context(ctx: "CLIContext") -> Renderer:
    """Get the appropriate renderer from CLI context."""
    # Determine format from context
    if ctx.json:
        format = "json"
    elif ctx.no_color or not ctx.use_rich:
        format = "plain"
    else:
        format = "rich"

    return create_renderer(format, use_color=not ctx.no_color, verbose=ctx.verbose)