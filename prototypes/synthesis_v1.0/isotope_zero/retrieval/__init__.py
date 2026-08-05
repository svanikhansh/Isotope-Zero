"""Multi-signal retrieval package.

Ports Mem0's multi-signal retrieval (``mem0/memory/main.py:1369`` ``search``
fusing vector + BM25 + entity boost at ``main.py:1763``) onto isotope_zero's
already-wired primitives — the ``memories_fts`` FTS5 index
(``core/store.py:513``), ``vector_search`` (``core/store.py:1398``), the
``card_edges`` graph (``core/graph.py`` via ``graph.relation_graph``), and the
Ebbinghaus decay engine (``core/decay.py``). No LLM, no network, no new
storage: the fusion is pure rank arithmetic over the store's live rows.

Public surface:
    - :class:`HybridSearcher` — Reciprocal Rank Fusion of vector + BM25 ranks,
      re-ranked by a graph-neighborhood entity boost and Ebbinghaus retention.
"""
from __future__ import annotations

from isotope_zero.retrieval.hybrid_search import HybridSearcher

__all__ = ["HybridSearcher"]
