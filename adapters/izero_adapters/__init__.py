"""Isotope Zero framework adapters — drop-in memory providers.

This package bridges Isotope Zero's low-latency SQLite + vector engine into the
standard ``VectorStore`` / ``Memory`` interfaces used by major Python AI agent
frameworks (LangChain, LlamaIndex, AutoGen, CrewAI), enabling two-line migration:

.. code-block:: python

    from izero_adapters.langchain import IsotopeZeroVectorStore
    vs = IsotopeZeroVectorStore(db_path="mem.db")   # that's it

Each framework adapter lives in its own submodule and imports its framework
lazily so that ``import izero_adapters`` never fails when an optional framework
package is absent. The shared engine plumbing — how every adapter talks to the
Isotope Zero storage layer — lives in :mod:`izero_adapters._engine`.

The adapter package itself never modifies core storage schemas or prototype
sources in ``prototypes/``; it only *uses* the engine via the stable
``MemoryStore`` / embedder contract documented in :mod:`._engine`.
"""
from __future__ import annotations

from ._engine import (
    Engine,
    EngineError,
    get_engine,
    DEFAULT_DIM,
    DEFAULT_MODEL,
)

__all__ = [
    "Engine",
    "EngineError",
    "get_engine",
    "DEFAULT_DIM",
    "DEFAULT_MODEL",
]

__version__ = "0.1.0"
