"""Entity-relation graph traversal package.

Ports Mem0's entity-relation model (``mem0/memory/main.py`` entity store +
``mem0/utils/entity_extraction.py`` co-occurrence relations) onto isotope_zero's
existing ``core.graph`` ``card_edges`` table — no parallel storage, no LLM, no
network on the search hot path.

Public surface:
    - :class:`RelationGraph` — multi-hop BFS, density, and text-triplet
      extraction that delegates edge persistence to ``core.graph.insert_edge``.
"""
from __future__ import annotations

from isotope_zero.graph.relation_graph import RelationGraph

__all__ = ["RelationGraph"]
