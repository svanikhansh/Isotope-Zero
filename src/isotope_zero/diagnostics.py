"""Diagnostic logging configuration for isotope_zero.

Structured-ish logging controlled by the `ISOTOPE_ZERO_LOG_LEVEL` env var. When set
to `DEBUG` (case-insensitive), the core modules (`core.triage`, `core.store`,
`core.consolidation`) emit per-operation debug records: action decisions,
SQLite write sizes, consolidation plans. At the default (`INFO`/`WARNING`)
level only warnings surface, so normal runs stay quiet and fast.

Call `configure_logging()` once from any entrypoint (`isotope_zero` CLI, the MCP
`main()`, `python -m isotope_zero...` smoke tests) before the core modules are
exercised. It is idempotent: repeated calls with the same level are no-ops.
"""
from __future__ import annotations

import logging
import os
import sys

_LEVEL_FROM_ENV = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_CONFIGURED_LEVEL: str | None = None


def configure_logging() -> str:
    """Configure isotope_zero loggers from ``ISOTOPE_ZERO_LOG_LEVEL``.

    The shorter alias ``IZERO_LOG_LEVEL`` is accepted as a fallback. Returns
    the effective level name. Idempotent — re-calling with the same env value
    is a no-op; calling with a changed value reconfigures.
    """
    global _CONFIGURED_LEVEL
    desired = (
        os.environ.get("ISOTOPE_ZERO_LOG_LEVEL")
        or os.environ.get("IZERO_LOG_LEVEL")
        or "WARNING"
    ).strip().upper()
    if _CONFIGURED_LEVEL == desired:
        return desired
    level = _LEVEL_FROM_ENV.get(desired, logging.WARNING)

    # One handler on the root of the "isotope_zero" namespace, writing to stderr
    # so it never interferes with stdout (which MCP uses for stdio transport).
    logger = logging.getLogger("isotope_zero")
    logger.setLevel(level)

    # Replace any prior handlers so reconfiguration updates the level cleanly.
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    # Compact structured-ish format: time level module:message
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False  # avoid double-prints via the root logger.

    _CONFIGURED_LEVEL = desired
    return desired


def get_level() -> str:
    """Current configured level name (or the env default if not yet set)."""
    if _CONFIGURED_LEVEL:
        return _CONFIGURED_LEVEL
    return (
        os.environ.get("ISOTOPE_ZERO_LOG_LEVEL")
        or os.environ.get("IZERO_LOG_LEVEL")
        or "WARNING"
    ).upper()
