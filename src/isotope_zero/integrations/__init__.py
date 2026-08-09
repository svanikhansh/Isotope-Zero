"""Local-first editor integration surface for ``isotope_zero``.

Mirrors mem0's *capture surface* (Claude Code lifecycle hooks, slash-command
skills, editor plugin manifests) but routes every capture pipe into the
**local** ``IsotopeZeroServer`` / SQLite store at ``~/.isotope_zero/`` instead
of ``mcp.mem0.ai``. No network, no API key, works offline — the headline
differentiator.

The linchpin is ONE shared entrypoint, ``izero hook <event>``: all bash hook
wrappers pipe stdin JSON to it, and all logic lives in the testable Python
module :mod:`isotope_zero.integrations.hooks`. This inverts mem0's anti-pattern
of a dozen separate python scripts each re-resolving identity and calling the
cloud API.

Public surface:
    - :func:`run_hook` — the stdin-driven entrypoint the CLI subcommand calls.
    - :func:`handle_hook` — the pure (payload, db_path) -> dict core every
      handler dispatches through; what the tests call directly.
"""
from __future__ import annotations

from .hooks import handle_hook, run_hook

__all__ = ["handle_hook", "run_hook"]
