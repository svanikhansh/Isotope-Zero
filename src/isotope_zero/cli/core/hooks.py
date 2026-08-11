"""isotope_zero CLI — Pre/Post Command Hooks.

Extensible hook system for cross-cutting concerns (logging, metrics, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .context import CLIContext


@dataclass
class HookContext:
    """Context passed to hooks."""
    cli_ctx: CLIContext
    command_name: str
    args: dict[str, Any]
    result: Any = None
    error: Optional[Exception] = None


class Hook(ABC):
    """Base class for CLI hooks."""

    @abstractmethod
    def pre_command(self, ctx: HookContext) -> None:
        """Called before command execution."""
        pass

    @abstractmethod
    def post_command(self, ctx: HookContext) -> None:
        """Called after command execution (success or error)."""
        pass


class HookRegistry:
    """Registry for CLI hooks."""

    def __init__(self):
        self._hooks: list[Hook] = []

    def register(self, hook: Hook) -> None:
        """Register a hook."""
        self._hooks.append(hook)

    def unregister(self, hook: Hook) -> None:
        """Unregister a hook."""
        self._hooks.remove(hook)

    def run_pre(self, ctx: HookContext) -> None:
        """Run all pre-command hooks."""
        for hook in self._hooks:
            try:
                hook.pre_command(ctx)
            except Exception:
                # Hooks must not break commands
                pass

    def run_post(self, ctx: HookContext) -> None:
        """Run all post-command hooks."""
        for hook in self._hooks:
            try:
                hook.post_command(ctx)
            except Exception:
                pass


# Global registry
_hook_registry = HookRegistry()


def get_hook_registry() -> HookRegistry:
    """Get the global hook registry."""
    return _hook_registry


def register_hook(hook: Hook) -> None:
    """Register a global hook."""
    _hook_registry.register(hook)


def unregister_hook(hook: Hook) -> None:
    """Unregister a global hook."""
    _hook_registry.unregister(hook)


# Built-in hooks
class LoggingHook(Hook):
    """Log command execution (for debug mode)."""

    def __init__(self, logger=None):
        self.logger = logger

    def pre_command(self, ctx: HookContext) -> None:
        if self.logger:
            self.logger.debug(f"CLI: {ctx.command_name} args={ctx.args}")

    def post_command(self, ctx: HookContext) -> None:
        if self.logger:
            if ctx.error:
                self.logger.debug(f"CLI: {ctx.command_name} error={ctx.error}")
            else:
                self.logger.debug(f"CLI: {ctx.command_name} ok")


class TimingHook(Hook):
    """Measure command execution time."""

    def __init__(self):
        self._start_times: dict[str, float] = {}

    def pre_command(self, ctx: HookContext) -> None:
        import time
        self._start_times[ctx.command_name] = time.perf_counter()

    def post_command(self, ctx: HookContext) -> None:
        import time
        start = self._start_times.pop(ctx.command_name, None)
        if start:
            elapsed = (time.perf_counter() - start) * 1000
            if ctx.cli_ctx.verbose:
                ctx.cli_ctx.get_renderer().render_json(
                    {"command": ctx.command_name, "elapsed_ms": round(elapsed, 2)}
                )


class MetricsHook(Hook):
    """Collect usage metrics (local only)."""

    def __init__(self, metrics_file: Optional[str] = None):
        self.metrics_file = metrics_file

    def pre_command(self, ctx: HookContext) -> None:
        pass

    def post_command(self, ctx: HookContext) -> None:
        if not self.metrics_file:
            return
        try:
            import json
            from pathlib import Path
            import time

            Path(self.metrics_file).parent.mkdir(parents=True, exist_ok=True)

            entry = {
                "timestamp": time.time(),
                "command": ctx.command_name,
                "success": ctx.error is None,
            }

            # Append to file
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


# Decorator for easy hook registration
def hook(pre: Optional[Callable[[HookContext], None]] = None,
         post: Optional[Callable[[HookContext], None]] = None) -> Hook:
    """Create a hook from callables."""
    class FuncHook(Hook):
        def pre_command(self, ctx: HookContext) -> None:
            if pre:
                pre(ctx)

        def post_command(self, ctx: HookContext) -> None:
            if post:
                post(ctx)
    return FuncHook()


__all__ = [
    "Hook",
    "HookContext",
    "HookRegistry",
    "get_hook_registry",
    "register_hook",
    "unregister_hook",
    "LoggingHook",
    "TimingHook",
    "MetricsHook",
    "hook",
]