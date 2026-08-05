"""izero-cli: isolated read-only terminal inspection tool for Isotope Zero memory engines.

This package exposes a small, stable data layer (``izero_cli.db``) that opens
Isotope Zero SQLite memory-engine databases in **read-only URI mode** and
returns plain dict contracts that the UI layer (``izero_cli.ui``) renders.

Safety model
------------
Every SQLite connection is opened as ``file:<path>?mode=ro`` with ``uri=True``
and immediately guarded by ``PRAGMA query_only=ON`` (double safety: forbids
any write even if a stray PRAGMA would have one). The data layer NEVER writes
to the inspected database and NEVER creates sidecar files. ``izero-cli`` is an
inspection tool, not a store.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "main"]


def main() -> None:  # pragma: no cover - thin wrapper, owned by another agent
    """Console entry point.

    Implemented lazily inside ``izero_cli.main`` to avoid an import cycle
    between ``__init__`` and ``main``. This wrapper is what ``[project.scripts]
    izero = "izero_cli.main:main"`` resolves to.
    """
    from .main import main as _main  # lazy import to avoid cycles

    _main()
