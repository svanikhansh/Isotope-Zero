"""isotope_zero CLI — Commands Package.

All command implementations organized by domain.
"""

# Exported for command registration
__all__ = [
    "register_commands",
]


def register_commands(app):
    """Register all command modules with the typer app."""
    from . import memory, inspect, dashboard, serve, hook, plugin, menu

    # Memory commands
    memory.register(app)

    # Inspect/analysis commands
    inspect.register(app)

    # Advanced UX commands
    dashboard.register(app)
    serve.register(app)
    hook.register(app)
    plugin.register(app)
    menu.register(app)