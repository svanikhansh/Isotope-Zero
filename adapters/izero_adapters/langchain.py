"""LangChain VectorStore adapter for Isotope Zero.

Provides :class:`IsotopeZeroVectorStore` — a LangChain-compatible vector store
backed by Isotope Zero's ``MemoryStore`` + embedder.

Design
------
When ``langchain_core`` is importable, :class:`IsotopeZeroVectorStore`
inherits from :class:`langchain_core.vectorstores.VectorStore` and returns
real :class:`langchain_core.documents.Document` objects, so it drops into any
LangChain pipeline unchanged.

When ``langchain_core`` is **not** installed, a duck-typed compatibility shim
(:class:`_BaseVectorStore` + :func:`_Document`) takes its place so the class is
still constructible and fully testable with zero third-party deps. The public
method surface and return shapes are identical in both paths — the only
difference is whether the returned ``Document`` is the real langchain type or a
look-alike with the same attributes.

The module imports ``langchain_core`` **lazily and guarded** so that
``import izero_adapters.langchain`` never fails when langchain is absent. The
module flag :data:`_HAS_LANGCHAIN` records availability at import time.

All storage work is delegated to :class:`izero_adapters._engine.Engine`, the
single shared seam over Isotope Zero's ``MemoryStore`` + embedder. This adapter
never touches SQLite, the schema, or the embedding model directly. Isotope Zero
manages its own embeddings (via the engine's embedder), so the LangChain
``embedding`` argument is accepted for signature compatibility but not used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, Sequence

from izero_adapters._engine import DEFAULT_DIM, Engine, get_engine

# Detect langchain_core at import time without importing it.
_HAS_LANGCHAIN: bool = find_spec("langchain_core") is not None


# --------------------------------------------------------------------------- #
# Base-class + Document resolution.
# --------------------------------------------------------------------------- #
if _HAS_LANGCHAIN:  # pragma: no cover - exercised only when langchain installed
    from langchain_core.vectorstores import VectorStore as _BaseVectorStore
    from langchain_core.documents import Document as _Document
else:
    # Duck-typed shims so the adapter is usable/testable without langchain.
    class _BaseVectorStore:  # type: ignore[no-redef]
        """Minimal stand-in for ``langchain_core.vectorstores.VectorStore``.

        Implements none of the abstract storage methods — subclasses provide
        ``add_texts`` / ``similarity_search`` / etc. Exists only so the class
        hierarchy is consistent across the installed / not-installed paths.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    @dataclass
    class _Document:  # type: ignore[no-redef]
        """Duck-typed stand-in for ``langchain_core.documents.Document``."""

        page_content: str
        metadata: dict[str, Any] = field(default_factory=dict)
        id: str | None = None


def _make_document(page_content: str, metadata: dict[str, Any]) -> Any:
    """Build a Document (real or shim) with the given content + metadata."""
    if _HAS_LANGCHAIN:  # pragma: no cover
        return _Document(page_content=page_content, metadata=metadata)
    return _Document(page_content=page_content, metadata=metadata)


class IsotopeZeroVectorStore(_BaseVectorStore):
    """LangChain-compatible ``VectorStore`` backed by Isotope Zero.

    Parameters
    ----------
    db_path:
        SQLite path for the Isotope Zero store (``":memory:"`` for in-process).
    embedder / use_daemon / dim:
        Forwarded to :func:`get_engine` when ``engine`` is not supplied.
    engine:
        An existing :class:`Engine` to reuse (tests / shared stores).

    Notes
    -----
    LangChain's ``embedding`` argument is accepted by ``from_texts`` for
    signature compatibility but ignored — Isotope Zero manages embeddings via
    its own engine, which keeps vectors compatible across the ecosystem.
    """

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
        # The real VectorStore.__init__ is abstract/empty; call it for parity.
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
    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Store texts; return the list of Isotope Zero card ids (input order).

        ``metadatas`` round-trip through the store as ``key=value`` tag pairs.
        """
        return self._engine.add_texts(
            texts, metadatas=metadatas, card_ids=ids
        )

    def add_documents(
        self,
        documents: Sequence[Any],
        ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Store LangChain ``Document`` objects; returns card ids.

        Reads ``page_content`` and ``metadata`` from each document. If ``ids``
        is omitted, collects any per-document ``id`` attribute.
        """
        texts = [getattr(d, "page_content", None) or d.get("page_content", "") for d in documents]
        metadatas = [getattr(d, "metadata", {}) or {} for d in documents]
        if ids is None:
            collected: list[str] = []
            for d in documents:
                did = getattr(d, "id", None)
                collected.append(str(did) if did is not None else None)
            # Only pass ids through if every document had one.
            if any(i is not None for i in collected):
                ids = collected  # type: ignore[assignment]
        return self.add_texts(texts, metadatas=metadatas, ids=ids)

    # -- reads ------------------------------------------------------------- #
    def _result_to_doc(
        self, result: dict[str, Any], *, with_score: bool = False
    ) -> Any | tuple[Any, float]:
        meta = dict(result.get("metadata") or {})
        meta["id"] = result.get("id")
        meta["tags"] = result.get("tags") or []
        if with_score:
            meta["score"] = result.get("score")
        doc = _make_document(result["text"], meta)
        if with_score:
            return doc, float(result.get("score") or 0.0)
        return doc

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Return up to ``k`` ``Document`` objects most similar to ``query``.

        ``filter`` (a metadata dict) post-filters results to those whose
        metadata contains every key/value pair — the LangChain filter
        convention. Each ``Document.metadata`` carries the card ``id`` and
        ``tags``; use :meth:`similarity_search_with_score` to also get scores.
        """
        return self._search(query, k=k, filter=filter, with_score=False)  # type: ignore[return-value]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Any, float]]:
        """Return ``(Document, score)`` tuples, score in [0, 1], sorted desc."""
        return self._search(query, k=k, filter=filter, with_score=True)  # type: ignore[return-value]

    def _search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        *,
        with_score: bool = False,
    ) -> list[Any]:
        results = self._engine.search(query, top_k=k)
        if filter:
            results = [r for r in results if self._matches_filter(r, filter)]
        out: list[Any] = []
        for r in results:
            out.append(self._result_to_doc(r, with_score=with_score))
        return out

    @staticmethod
    def _matches_filter(result: dict[str, Any], filter: dict[str, Any]) -> bool:
        """True if result metadata contains every filter key/value pair."""
        meta = result.get("metadata") or {}
        return all(meta.get(k) == v for k, v in filter.items())

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Return ``Document`` objects most similar to a given embedding vector."""
        results = self._engine.search(
            "", top_k=k, query_embedding=list(embedding)
        )
        if filter:
            results = [r for r in results if self._matches_filter(r, filter)]
        return [self._result_to_doc(r, with_score=False) for r in results]  # type: ignore[return-value]

    # -- deletes ----------------------------------------------------------- #
    def delete(
        self,
        ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Delete cards by id. Best-effort: missing ids are silently ignored."""
        if not ids:
            return
        for card_id in ids:
            self._engine.delete(card_id)

    # -- classmethod constructor ------------------------------------------- #
    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        embedding: Any = None,  # noqa: ARG002 - ignored; Isotope Zero embeds itself
        metadatas: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> "IsotopeZeroVectorStore":
        """Construct a store and add ``texts`` in one call (LangChain convention).

        ``embedding`` is accepted for signature compatibility but ignored —
        Isotope Zero manages embeddings via its own engine. ``db_path`` and
        other engine options are passed through ``kwargs``.
        """
        db_path = kwargs.pop("db_path", ":memory:")
        store = cls(db_path=db_path, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store


# --------------------------------------------------------------------------- #
# Chat message history
# --------------------------------------------------------------------------- #
def _resolve_chat_message_types() -> tuple[Any, Any, Any]:
    """Resolve (BaseChatMessageHistory, HumanMessage, AIMessage) lazily.

    Returns the real ``langchain_core`` classes when importable; otherwise
    duck-typed stand-ins (simple base class + ``_Msg`` dataclass) so the
    history adapter is constructible + testable without langchain installed.
    Mirrors the ``_BaseVectorStore`` / ``_Document`` shim pattern at the top
    of this module.
    """
    if _HAS_LANGCHAIN:  # pragma: no cover - exercised only with langchain
        try:
            from langchain_core.chat_history import BaseChatMessageHistory
            from langchain_core.messages import HumanMessage, AIMessage

            return BaseChatMessageHistory, HumanMessage, AIMessage
        except Exception:
            pass  # fall through to the shims below
    return _BaseChatMessageHistory, _HumanMessage, _AIMessage


# --------------------------------------------------------------------------- #
# Message-history base-class + message-type resolution.
#
# Mirrors the ``_BaseVectorStore`` / ``_Document`` resolution at the top of this
# module: when ``langchain_core`` is importable we bind the real ABC + real
# ``HumanMessage`` / ``AIMessage`` so ``IsotopeChatMessageHistory`` is a genuine
# ``isinstance(..., BaseChatMessageHistory)`` subclass; otherwise duck-typed
# shims keep it constructible + testable with zero dependencies.
# --------------------------------------------------------------------------- #
if _HAS_LANGCHAIN:  # pragma: no cover - exercised only with langchain installed
    from langchain_core.chat_history import (
        BaseChatMessageHistory as _BaseChatMessageHistory,
    )
    from langchain_core.messages import HumanMessage as _HumanMessage
    from langchain_core.messages import AIMessage as _AIMessage
else:
    class _BaseChatMessageHistory:  # type: ignore[no-redef]
        """Minimal stand-in for ``langchain_core.chat_history.BaseChatMessageHistory``.

        The real base class is abstract over ``messages`` / ``add_messages`` /
        ``clear`` (plus their async ``a*`` counterparts). Subclasses provide the
        storage; this shim only fixes the hierarchy so the class is consistent
        whether or not langchain is installed.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    @dataclass
    class _HumanMessage:  # type: ignore[no-redef]
        """Duck-typed stand-in for ``langchain_core.messages.HumanMessage``."""

        content: str
        additional_kwargs: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class _AIMessage:  # type: ignore[no-redef]
        """Duck-typed stand-in for ``langchain_core.messages.AIMessage``."""

        content: str
        additional_kwargs: dict[str, Any] = field(default_factory=dict)


class IsotopeChatMessageHistory(_BaseChatMessageHistory):
    """Scoped chat-message history backed by Isotope Zero.

    Stores each chat turn as an Isotope Zero memory card tagged
    ``"chat_history"`` with ``session_id`` carried in metadata, so the full
    transcript round-trips through the store and is retrievable by session.

    Mirrors the ``langchain_core.chat_history.BaseChatMessageHistory``
    surface: ``add_messages`` / ``messages`` / ``clear`` and their async
    ``a*`` counterparts (``aadd_messages`` / ``agessages`` / ``aclear``).
    When ``langchain_core`` is importable the real ``HumanMessage`` /
    ``AIMessage`` types are used and the class is a real
    ``BaseChatMessageHistory`` subclass; otherwise duck-typed shims keep it
    constructible + testable with zero third-party deps — the public method
    surface and return shapes are identical in both paths.

    Parameters
    ----------
    session_id:
        Conversation/session key. When ``scope`` is not given it doubles as
        the Isotope Zero scope, so each session's history is isolated.
    db_path / engine:
        Either pass an existing :class:`Engine` to reuse, or let the adapter
        build one against ``db_path``.
    scope:
        Isotope Zero scope for the underlying store. Defaults to
        ``session_id``.
    """

    def __init__(
        self,
        session_id: str,
        db_path: str = ":memory:",
        *,
        engine: Engine | None = None,
        scope: str | None = None,
    ) -> None:
        try:
            super().__init__()
        except TypeError:  # pragma: no cover - shim path takes no kwargs
            pass
        self.session_id = session_id
        self.scope = scope if scope is not None else session_id
        if engine is not None:
            self._engine: Engine = engine
            self.db_path = engine.db_path
        else:
            self._engine = get_engine(db_path=db_path)
            self.db_path = db_path
        self._BaseChatMessageHistory, self._HumanMessage, self._AIMessage = (
            _resolve_chat_message_types()
        )

    # -- internals --------------------------------------------------------- #
    @staticmethod
    def _role_for(msg: Any) -> str:
        """Infer the ``role`` of a message (real langchain or shim)."""
        cls_name = type(msg).__name__.lower()
        if isinstance(msg, _HumanMessage) or "human" in cls_name:
            return "human"
        if isinstance(msg, _AIMessage) or "ai" in cls_name or "assistant" in cls_name:
            return "ai"
        return "unknown"

    def _to_message_obj(self, role: str, content: str) -> Any:
        """Build a HumanMessage/AIMessage (real or shim) for the given role."""
        if role == "ai":
            return self._AIMessage(content=content)
        return self._HumanMessage(content=content)

    def _session_messages(self) -> list[dict[str, Any]]:
        """All stored cards for this session, oldest-first.

        Uses :meth:`Engine.all` (a full scan) rather than semantic search so
        every turn is returned regardless of lexical/semantic similarity to
        a query. The store has no tag-indexed lookup, so a post-filter on
        ``session_id`` metadata is the correct retrieval path here.
        """
        kept = [r for r in self._engine.all() if self._matches_session(r)]
        kept.sort(key=lambda r: r.get("timestamp") or 0.0)
        return kept

    # -- sync API ---------------------------------------------------------- #
    @property
    def messages(self) -> list[Any]:
        """All stored messages for this session, oldest-first."""
        out: list[Any] = []
        for r in self._session_messages():
            role = (r.get("metadata") or {}).get("role", "human")
            out.append(self._to_message_obj(role, r["text"]))
        return out

    def add_messages(self, messages: Sequence[Any]) -> None:
        """Append messages to this session's history."""
        for m in messages:
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                content = m.get("content", "")
            role = self._role_for(m)
            self._engine.add_text(
                str(content),
                metadata={
                    "session_id": self.session_id,
                    "role": role,
                },
                tags=["chat_history"],
            )

    def clear(self) -> None:
        """Remove every message stored for this session."""
        for r in self._session_messages():
            self._engine.delete(r["id"])

    # -- async API --------------------------------------------------------- #
    async def aadd_messages(self, messages: Sequence[Any]) -> None:
        """Async append (delegates to the sync path; storage is sync)."""
        self.add_messages(messages)

    async def aget_messages(self) -> list[Any]:
        """Async read of :attr:`messages`."""
        return self.messages

    async def aclear(self) -> None:
        """Async clear (delegates to the sync path)."""
        self.clear()

    # -- helpers ----------------------------------------------------------- #
    def _matches_session(self, result: dict[str, Any]) -> bool:
        meta = result.get("metadata") or {}
        return meta.get("session_id") == self.session_id
