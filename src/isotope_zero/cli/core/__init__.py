"""isotope_zero CLI — Core Package.

Shared infrastructure for all CLI commands.
"""

from .context import (
    CLIContext,
    get_context,
    set_context,
    inject_context,
    db_option,
    json_option,
    verbose_option,
    no_color_option,
    dry_run_option,
    force_option,
)

from .completers import (
    complete_tags,
    complete_scopes,
    complete_card_ids,
    complete_commands,
)

from .hooks import (
    Hook,
    HookContext,
    HookRegistry,
    get_hook_registry,
    register_hook,
    unregister_hook,
    LoggingHook,
    TimingHook,
    MetricsHook,
    hook,
)

__all__ = [
    # Context
    "CLIContext",
    "get_context",
    "set_context",
    "inject_context",
    "db_option",
    "json_option",
    "verbose_option",
    "no_color_option",
    "dry_run_option",
    "force_option",
    # Completers
    "complete_tags",
    "complete_scopes",
    "complete_card_ids",
    "complete_commands",
    # Hooks
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