"""isotope_zero CLI — Shared Context.

Holds database connection, configuration, and runtime state for all commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import typer


@dataclass
class CLIContext:
    """Shared CLI context passed to all commands."""

    # Database
    db_path: Optional[str] = None
    db: Any = None  # MemoryStore instance (lazy-loaded)

    # Output
    json: bool = False
    verbose: bool = False
    no_color: bool = False
    use_rich: bool = True  # Auto-detected, can be overridden

    # Runtime
    dry_run: bool = False
    force: bool = False

    # Config
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Resolve defaults after initialization."""
        if self.db_path is None:
            self.db_path = self._default_db_path()

    def _default_db_path(self) -> str:
        """Get default database path."""
        # Check env var first
        import os
        env_path = os.environ.get("ISOTOPE_ZERO_DB")
        if env_path:
            return env_path

        # Default to ~/.isotope_zero/isotope_zero.db
        home = Path.home()
        db_dir = home / ".isotope_zero"
        db_dir.mkdir(parents=True, exist_ok=True)
        return str(db_dir / "isotope_zero.db")

    def get_renderer(self):
        """Get the appropriate renderer for this context."""
        from .render import get_renderer_from_context
        return get_renderer_from_context(self)


# Global context instance (set by main.py)
_context: Optional[CLIContext] = None


def get_context() -> CLIContext:
    """Get the global CLI context."""
    global _context
    if _context is None:
        _context = CLIContext()
    return _context


def set_context(ctx: CLIContext) -> None:
    """Set the global CLI context."""
    global _context
    _context = ctx


# Typer option factories for global flags
def db_option() -> Any:
    """--db option for all commands."""
    return typer.Option(
        None,
        "--db",
        help="Database path (default: ~/.isotope_zero/isotope_zero.db or :memory:)",
        envvar="ISOTOPE_ZERO_DB",
    )


def json_option() -> Any:
    """--json option for all commands."""
    return typer.Option(
        False,
        "--json",
        help="Output as JSON for scripting",
    )


def verbose_option() -> Any:
    """-v/--verbose option for all commands."""
    return typer.Option(
        False,
        "-v",
        "--verbose",
        help="Show technical detail (IDs, scores, timestamps)",
    )


def no_color_option() -> Any:
    """--no-color option for all commands."""
    return typer.Option(
        False,
        "--no-color",
        help="Disable colored output",
    )


def dry_run_option() -> Any:
    """--dry-run option for applicable commands."""
    return typer.Option(
        False,
        "--dry-run",
        help="Show what would happen without making changes",
    )


def force_option() -> Any:
    """-y/--force option for destructive commands."""
    return typer.Option(
        False,
        "-y",
        "--force",
        "--yes",
        help="Skip confirmation prompts",
    )


# Context injection for typer commands
def inject_context(
    ctx: typer.Context,
    db: str = db_option(),
    json: bool = json_option(),
    verbose: bool = verbose_option(),
    no_color: bool = no_color_option(),
    dry_run: bool = dry_run_option(),
    force: bool = force_option(),
) -> CLIContext:
    """Create and inject CLI context from global options."""
    cli_ctx = CLIContext(
        db_path=db,
        json=json,
        verbose=verbose,
        no_color=no_color,
        dry_run=dry_run,
        force=force,
    )
    set_context(cli_ctx)
    ctx.obj = cli_ctx
    return cli_ctx