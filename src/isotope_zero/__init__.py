"""isotope_zero — efficiency-first agent memory.

Local ONNX embeddings, SQLite storage, hybrid query routing, and an MCP
surface — engineered to minimize tokens and latency per useful fact.
"""
from __future__ import annotations

from .types import ActionType, ActionResult, ConsolidationReport, MemoryCard, QueryHit, QueryResult, now_ts
from .tokens import estimate_tokens

__version__ = "1.1.0"

__all__ = [
    "ActionType",
    "ActionResult",
    "ConsolidationReport",
    "MemoryCard",
    "QueryHit",
    "QueryResult",
    "now_ts",
    "estimate_tokens",
    "__version__",
]
