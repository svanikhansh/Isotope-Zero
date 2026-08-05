"""Tests for the LangChain VectorStore adapter.

All tests run WITHOUT ``langchain_core`` installed, exercising the duck-typed
compatibility path via :class:`_mocks.MockDocument`. A final
``importorskip`` test adds the real-langchain integration check that skips
cleanly when the framework is absent.
"""
from __future__ import annotations

import pytest

from izero_adapters.langchain import IsotopeZeroVectorStore
from tests._mocks import MockDocument, make_documents


# --------------------------------------------------------------------------- #
# add_texts / add_documents
# --------------------------------------------------------------------------- #
def test_add_texts_returns_ids(mem_engine):
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    ids = vs.add_texts(
        ["The user prefers dark mode.", "SQLite is fast.", "Dark mode reduces eye strain."],
        metadatas=[{"source": "chat"}, {"source": "doc"}, {"source": "chat"}],
    )
    assert len(ids) == 3
    assert all(isinstance(i, str) for i in ids)
    assert vs._engine.count() == 3


def test_add_texts_no_metadatas(mem_engine):
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    ids = vs.add_texts(["one", "two"])
    assert len(ids) == 2
    assert vs._engine.count() == 2


def test_add_documents(mem_engine):
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    docs = make_documents(3, prefix="doc")
    ids = vs.add_documents(docs)
    assert len(ids) == 3
    assert vs._engine.count() == 3


def test_add_documents_with_ids(mem_engine):
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    docs = [MockDocument(page_content="hello", metadata={"s": "a"}, id="fixed-1")]
    ids = vs.add_documents(docs)
    assert ids == ["fixed-1"]


# --------------------------------------------------------------------------- #
# similarity_search
# --------------------------------------------------------------------------- #
def _seed_corpus(engine):
    engine.add_texts(
        [
            "The user prefers dark mode for the terminal at night.",
            "SQLite is a fast embedded database for local vector search.",
            "Dark mode reduces eye strain during long coding sessions.",
        ],
        metadatas=[{"source": "chat"}, {"source": "doc"}, {"source": "chat"}],
        tags=[["ui"], ["db"], ["ui"]],
    )


def test_similarity_search_returns_documents(mem_engine):
    _seed_corpus(mem_engine)
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    docs = vs.similarity_search("dark mode", k=2)
    assert len(docs) <= 2
    assert all(hasattr(d, "page_content") for d in docs)
    assert all("id" in d.metadata for d in docs)
    assert all("tags" in d.metadata for d in docs)


def test_similarity_search_with_score(mem_engine):
    _seed_corpus(mem_engine)
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    pairs = vs.similarity_search_with_score("dark mode", k=3)
    assert len(pairs) == 3
    # (Document, float) tuples, scores in [0,1], sorted descending.
    scores = [s for _, s in pairs]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)
    # The most relevant doc must be a dark-mode doc, not the sqlite doc.
    top_doc, _ = pairs[0]
    assert "dark mode" in top_doc.page_content.lower()


def test_similarity_search_with_score_orders_relevance(mem_engine):
    _seed_corpus(mem_engine)
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    pairs = vs.similarity_search_with_score("dark mode", k=3)
    # Both dark-mode cards should outrank the sqlite card.
    sqlite_idx = next(
        i for i, (d, _) in enumerate(pairs) if "sqlite" in d.page_content.lower()
    )
    assert sqlite_idx == 2  # sqlite card is least relevant


# --------------------------------------------------------------------------- #
# metadata + filter
# --------------------------------------------------------------------------- #
def test_metadata_passthrough(mem_engine):
    mem_engine.add_text("a memory", metadata={"source": "chat", "user": "alice"})
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    docs = vs.similarity_search("memory", k=1)
    assert docs[0].metadata["source"] == "chat"
    assert docs[0].metadata["user"] == "alice"


def test_filter_post_filters(mem_engine):
    _seed_corpus(mem_engine)
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    docs = vs.similarity_search("mode", k=5, filter={"source": "chat"})
    assert len(docs) >= 1
    assert all(d.metadata["source"] == "chat" for d in docs)
    # The sqlite/doc card is excluded.
    assert all("sqlite" not in d.page_content.lower() for d in docs)


# --------------------------------------------------------------------------- #
# delete + from_texts + engine passthrough
# --------------------------------------------------------------------------- #
def test_delete(mem_engine):
    ids = mem_engine.add_texts(["keep", "drop"])
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    vs.delete([ids[1]])
    assert vs._engine.count() == 1


def test_delete_none_is_noop(mem_engine):
    mem_engine.add_text("x")
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    vs.delete(None)  # should not raise
    assert vs._engine.count() == 1


def test_from_texts_classmethod(tmp_db_path):
    vs = IsotopeZeroVectorStore.from_texts(
        ["alpha", "beta"],
        metadatas=[{"k": "1"}, {"k": "2"}],
        db_path=tmp_db_path,
    )
    assert vs._engine.count() == 2
    docs = vs.similarity_search("alpha", k=2)
    assert len(docs) == 2


def test_engine_passthrough(mem_engine):
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    assert vs._engine is mem_engine
    assert vs.is_real == mem_engine.is_real


def test_similarity_search_by_vector(mem_engine):
    ids = mem_engine.add_texts(["dark mode", "sqlite"])
    # Reuse the first card's own embedding as the query vector -> self-match.
    card = mem_engine.get(ids[0])
    vec = mem_engine.store.get(ids[0]).embedding
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    docs = vs.similarity_search_by_vector(vec, k=2)
    assert len(docs) == 2
    assert docs[0].page_content == "dark mode"


# --------------------------------------------------------------------------- #
# real-langchain integration (skips if langchain_core absent)
# --------------------------------------------------------------------------- #
def test_with_real_langchain(tmp_db_path):
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore

    vs = IsotopeZeroVectorStore(db_path=tmp_db_path)
    assert isinstance(vs, VectorStore)
    ids = vs.add_documents([Document(page_content="real doc", metadata={"x": 1})])
    assert len(ids) == 1
    docs = vs.similarity_search("real", k=1)
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "real doc"


# --------------------------------------------------------------------------- #
# Engine.hybrid_search
# --------------------------------------------------------------------------- #
def test_hybrid_search_returns_same_shape(mem_engine):
    _seed_corpus(mem_engine)
    results = mem_engine.hybrid_search("dark mode", top_k=3)
    assert len(results) <= 3
    # Same dict shape as Engine.search (id, text, score, metadata, tags, timestamp).
    for r in results:
        assert set(r.keys()) == {"id", "text", "score", "metadata", "tags", "timestamp"}
        assert isinstance(r["id"], str)
        assert isinstance(r["text"], str)


def test_hybrid_search_alpha_mapping(mem_engine):
    # fts_weight=0, vector_weight=1 -> alpha=1.0 (pure vector). Must not raise
    # and must return results for a known-relevant query.
    _seed_corpus(mem_engine)
    results = mem_engine.hybrid_search(
        "dark mode", top_k=2, fts_weight=0.0, vector_weight=1.0
    )
    assert len(results) >= 1
    assert "dark mode" in results[0]["text"].lower()


def test_hybrid_search_zero_weights_uses_default_alpha(mem_engine):
    # Both weights zero -> degenerate to store default alpha=0.70; still works.
    _seed_corpus(mem_engine)
    results = mem_engine.hybrid_search(
        "dark mode", top_k=2, fts_weight=0.0, vector_weight=0.0
    )
    assert len(results) >= 1


def test_hybrid_search_accepts_query_embedding(mem_engine):
    _seed_corpus(mem_engine)
    # Pass an explicit embedding (a self-vector) and an empty query string.
    vec = mem_engine.store.get(mem_engine.all()[0]["id"]).embedding
    results = mem_engine.hybrid_search(
        "", query_embedding=vec, top_k=2
    )
    assert len(results) >= 1


# --------------------------------------------------------------------------- #
# IsotopeChatMessageHistory
# --------------------------------------------------------------------------- #
def test_chat_history_scope_defaults_to_session_id(mem_engine):
    from izero_adapters.langchain import IsotopeChatMessageHistory

    h = IsotopeChatMessageHistory("sess-42", engine=mem_engine)
    assert h.session_id == "sess-42"
    assert h.scope == "sess-42"  # scope derives from session_id when not given


def test_chat_history_explicit_scope(mem_engine):
    from izero_adapters.langchain import IsotopeChatMessageHistory

    h = IsotopeChatMessageHistory("sess-42", engine=mem_engine, scope="custom")
    assert h.scope == "custom"


def test_chat_history_roundtrip(mem_engine):
    from izero_adapters.langchain import IsotopeChatMessageHistory, _HumanMessage, _AIMessage

    h = IsotopeChatMessageHistory("sess-1", engine=mem_engine)
    h.add_messages([_HumanMessage(content="hello"), _AIMessage(content="hi back")])
    msgs = h.messages
    assert len(msgs) == 2
    # Oldest-first: human turn, then ai turn.
    assert msgs[0].content == "hello"
    assert msgs[1].content == "hi back"
    # Each stored card carries the session_id in metadata.
    for r in h._session_messages():
        assert r["metadata"]["session_id"] == "sess-1"


def test_chat_history_isolation_between_sessions(mem_engine):
    from izero_adapters.langchain import IsotopeChatMessageHistory, _HumanMessage

    h1 = IsotopeChatMessageHistory("sess-a", engine=mem_engine)
    h2 = IsotopeChatMessageHistory("sess-b", engine=mem_engine)
    h1.add_messages([_HumanMessage(content="msg from a")])
    h2.add_messages([_HumanMessage(content="msg from b")])
    # Each session sees only its own messages.
    assert len(h1.messages) == 1
    assert h1.messages[0].content == "msg from a"
    assert len(h2.messages) == 1
    assert h2.messages[0].content == "msg from b"


def test_chat_history_clear(mem_engine):
    from izero_adapters.langchain import IsotopeChatMessageHistory, _HumanMessage

    h = IsotopeChatMessageHistory("sess-1", engine=mem_engine)
    h.add_messages([_HumanMessage(content="one"), _HumanMessage(content="two")])
    assert len(h.messages) == 2
    h.clear()
    assert len(h.messages) == 0


def test_chat_history_real_langchain(tmp_db_path):
    pytest.importorskip("langchain_core")
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage

    from izero_adapters.langchain import IsotopeChatMessageHistory

    h = IsotopeChatMessageHistory("sess-real", db_path=tmp_db_path)
    assert isinstance(h, BaseChatMessageHistory)
    h.add_messages([HumanMessage(content="hi"), AIMessage(content="hello")])
    msgs = h.messages
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "hi"

