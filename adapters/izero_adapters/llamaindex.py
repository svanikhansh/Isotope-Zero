"""LlamaIndex VectorStore adapter for Isotope Zero.

Provides :class:`IsotopeZeroVectorStore` — a LlamaIndex-compatible vector
store backed by Isotope Zero's ``MemoryStore`` + embedder.

Design
------
When ``llama_index`` is importable, :class:`IsotopeZeroVectorStore` inherits
from :class:`llama_index.core.vector_stores.types.BasePydanticVectorStore` and
returns real LlamaIndex ``TextNode`` / ``VectorStoreQueryResult`` objects, so
it drops into any LlamaIndex pipeline unchanged.

When ``llama_index`` is **not** installed, duck-typed compatibility shims take
their place (a plain-object ``_BasePydanticVectorStore`` plus small dataclasses
for the query/result/node types) so the class is still constructible and fully
testable with zero third-party deps. The public method surface and return
shapes are identical in both paths.

The module imports ``llama_index`` **lazily and guarded** so that
``import izero_adapters.llamaindex`` never fails when llama_index is absent.
The module flag :data:`_HAS_LLAMAINDEX` records availability at import time.

All storage work is delegated to :class:`izero_adapters._engine.Engine`, the
single shared seam over Isotope Zero's ``MemoryStore`` + embedder. This adapter
never touches SQLite, the schema, or the embedding model directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, Sequence

from izero_adapters._engine import DEFAULT_DIM, Engine, get_engine

# Detect llama_index at import time without importing it.
_HAS_LLAMAINDEX: bool = find_spec("llama_index") is not None


# --------------------------------------------------------------------------- #
# Type resolution: real llama_index types or duck-typed shims.
# --------------------------------------------------------------------------- #
if _HAS_LLAMAINDEX:  # pragma: no cover - exercised only when llama_index installed
    from llama_index.core.vector_stores.types import (  # type: ignore
        BasePydanticVectorStore as _BasePydanticVectorStore,
        VectorStoreQuery as _VectorStoreQuery,
        VectorStoreQueryResult as _VectorStoreQueryResult,
    )
    from llama_index.core.schema import TextNode as _TextNode  # type: ignore
else:
    class _BasePydanticVectorStore:  # type: ignore[no-redef]
        """Minimal stand-in for LlamaIndex's ``BasePydanticVectorStore``.

        No pydantic dependency — a plain object so the adapter is usable and
        testable without llama_index. Subclasses provide ``add``/``delete``/
        ``query``.
        """

        stores_text: bool = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    @dataclass
    class _VectorStoreQuery:  # type: ignore[no-redef]
        """Duck-typed stand-in for ``VectorStoreQuery``."""

        query_embedding: list[float] | None = None
        query_str: str | None = None
        similarity_top_k: int = 5
        doc_ids: list[str] | None = None
        filter: dict[str, Any] | None = None

    @dataclass
    class _VectorStoreQueryResult:  # type: ignore[no-redef]
        """Duck-typed stand-in for ``VectorStoreQueryResult``."""

        nodes: list[Any] = field(default_factory=list)
        similarities: list[float] = field(default_factory=list)
        ids: list[str] = field(default_factory=list)

    @dataclass
    class _TextNode:  # type: ignore[no-redef]
        """Duck-typed stand-in for ``TextNode`` carrying text + metadata."""

        text: str = ""
        metadata: dict[str, Any] = field(default_factory=dict)
        id_: str = ""
        embedding: list[float] | None = None

        def get_content(self) -> str:
            return self.text


def _make_text_node(result: dict[str, Any]) -> Any:
    """Build a TextNode (real or shim) from an Engine result dict."""
    meta = dict(result.get("metadata") or {})
    # Preserve plain tags as a list for callers that want them.
    meta["tags"] = result.get("tags") or []
    if _HAS_LLAMAINDEX:  # pragma: no cover
        return _TextNode(text=result["text"], metadata=meta, id_=result["id"])
    return _TextNode(text=result["text"], metadata=meta, id_=result["id"])


class IsotopeZeroVectorStore(_BasePydanticVectorStore):
    """LlamaIndex-compatible ``BasePydanticVectorStore`` backed by Isotope Zero.

    Parameters
    ----------
    db_path:
        SQLite path for the Isotope Zero store (``":memory:"`` for in-process).
    embedder / use_daemon / dim:
        Forwarded to :func:`get_engine` when ``engine`` is not supplied.
    engine:
        An existing :class:`Engine` to reuse (tests / shared stores).
    """

    stores_text: bool = True

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        embedder: Any = None,
        use_daemon: bool = False,
        dim: int = DEFAULT_DIM,
        engine: Engine | None = None,
        **kwargs: Any,
    ) -> None:
        # BasePydanticVectorStore is a pydantic model in real llama_index; the
        # shim accepts anything. Guard both paths.
        try:
            super().__init__(**kwargs)
        except TypeError:  # pragma: no cover - shim path takes no kwargs
            pass
        if engine is not None:
            self._engine: Engine = engine
            self.db_path = engine.db_path
            self.dim = engine.dim
            self.is_real = engine.is_real
        else:
            self._engine = get_engine(
                db_path=db_path, embedder=embedder, use_daemon=use_daemon, dim=dim
            )
            self.db_path = db_path
            self.dim = self._engine.dim
            self.is_real = self._engine.is_real

    # -- writes ------------------------------------------------------------ #
    def add(self, nodes: Sequence[Any], **kwargs: Any) -> list[str]:
        """Store LlamaIndex ``BaseNode`` objects; return the list of card ids.

        For each node, reads ``get_content()`` (or ``.text``), ``metadata``, and
        ``id_`` (or ``.id``). If the node already carries an ``embedding``, it is
        passed through so Isotope Zero stores the provided vector verbatim;
        otherwise the engine embeds the text.

        **Scoping**: a node ``metadata`` key of ``scope`` (or the legacy
        ``user_id`` / ``doc_id``) is forwarded to the engine, so LlamaIndex
        document collections are isolated in distinct isotope_zero scopes
        (metadata filtering → scoping parity).
        """
        ids: list[str] = []
        for node in nodes:
            text = self._node_text(node)
            metadata = getattr(node, "metadata", {}) or {}
            node_id = getattr(node, "id_", None) or getattr(node, "id", None)
            embedding = getattr(node, "embedding", None)
            card_id = self._engine.add_text(
                text,
                metadata=metadata,
                card_id=str(node_id) if node_id is not None else None,
                embedding=list(embedding) if embedding else None,
            )
            ids.append(card_id)
        return ids

    @staticmethod
    def _node_text(node: Any) -> str:
        # Real BaseNode exposes get_content(metadata_mode=...); fall back to a
        # .text attr for duck-typed test nodes. Try the no-arg form first
        # (works for our MockTextNode), then the 1-arg form (MetadataMode.NONE
        # == 0 in real llama_index), then .text.
        getter = getattr(node, "get_content", None)
        if callable(getter):
            try:
                return getter() or ""
            except TypeError:
                try:
                    return getter(0) or ""  # 0 == MetadataMode.NONE
                except Exception:
                    pass
            except Exception:
                pass
        return getattr(node, "text", "") or ""

    # -- reads ------------------------------------------------------------- #
    def query(self, query: Any, **kwargs: Any) -> Any:
        """Run a LlamaIndex ``VectorStoreQuery`` against the store.

        Uses ``query.query_embedding`` if provided, else embeds
        ``query.query_str``. Returns a ``VectorStoreQueryResult`` with
        ``.nodes`` (``TextNode``s), ``.similarities`` (scores), ``.ids``
        (card ids), sorted by similarity descending.

        **Scoping**: ``query.filter`` (a LlamaIndex ``MetadataFilters`` or
        plain dict) is inspected for a ``scope`` / ``user_id`` / ``doc_id`` key;
        if present, the search is confined to that isotope_zero scope. This is
        the LlamaIndex-parity bridge from metadata filtering → scoping.
        """
        top_k = getattr(query, "similarity_top_k", 5) or 5
        q_emb = getattr(query, "query_embedding", None)
        q_str = getattr(query, "query_str", None) or ""
        scope = self._scope_from_filter(getattr(query, "filter", None))

        if q_emb:
            results = self._engine.search(
                q_str, top_k=top_k, query_embedding=list(q_emb), scope=scope
            )
        else:
            results = self._engine.search(q_str, top_k=top_k, scope=scope)

        nodes = [_make_text_node(r) for r in results]
        similarities = [float(r.get("score") or 0.0) for r in results]
        ids = [r.get("id", "") for r in results]
        return _VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    @staticmethod
    def _scope_from_filter(filt: Any) -> str | None:
        """Extract a scope string from a LlamaIndex filter spec.

        LlamaIndex passes either a plain ``dict`` or a
        ``MetadataFilters(filters=[MetadataFilter(key=..., value=...)])``
        object. We look for ``scope``/``user_id``/``doc_id``; anything else is
        ignored (the store's tag index handles equality predicates at the SQL
        layer, so we don't reimplement generic metadata filtering here — we
        only bridge the isotope_zero-native scoping axis).
        """
        if filt is None:
            return None
        # Plain dict: {"scope": "x"} or {"user_id": "y"}.
        if isinstance(filt, dict):
            for key in ("scope", "user_id", "doc_id"):
                if key in filt and filt[key] is not None:
                    return str(filt[key])
            return None
        # MetadataFilters object: walk .filters for a matching key.
        filters = getattr(filt, "filters", None) or []
        for f in filters:
            key = getattr(f, "key", None)
            val = getattr(f, "value", None)
            if key in ("scope", "user_id", "doc_id") and val is not None:
                return str(val)
        return None

    # -- deletes ----------------------------------------------------------- #
    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete the card whose id matches ``ref_doc_id``. Best-effort."""
        self._engine.delete(ref_doc_id)

    # -- persistence ------------------------------------------------------- #
    def persist(self, path: str, **kwargs: Any) -> None:
        """No-op: Isotope Zero persists to SQLite continuously on every write."""
        return None
