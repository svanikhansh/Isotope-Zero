"""Tests for the LlamaIndex VectorStore adapter.

All tests run WITHOUT ``llama_index`` installed, exercising the duck-typed
compatibility path via :class:`_mocks.MockTextNode` / ``MockNodeWithEmbedding``.
A final ``importorskip`` test adds the real-llamaindex integration check that
skips cleanly when the framework is absent.
"""
from __future__ import annotations

import pytest

from izero_adapters.llamaindex import IsotopeZeroVectorStore, _VectorStoreQuery
from tests._mocks import MockNodeWithEmbedding, MockTextNode, make_text_nodes


# --------------------------------------------------------------------------- #
# add
# --------------------------------------------------------------------------- #
def test_add_nodes(mem_engine):
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    ids = vs.add(make_text_nodes(3, prefix="node"))
    assert len(ids) == 3
    assert vs._engine.count() == 3


def test_add_node_with_embedding(mem_engine):
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    emb = [1.0, 0.0, 0.0, 0.0]  # dim 4 stub space
    node = MockNodeWithEmbedding(text="explicit vector", embedding=emb, id_="n1")
    ids = vs.add([node])
    assert ids == ["n1"]
    card = vs._engine.store.get("n1")
    assert card is not None
    assert [round(x, 6) for x in card.embedding] == emb


def test_add_uses_engine_embedding_when_absent(mem_engine):
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    node = MockTextNode(text="auto embedded", id_="n2")
    vs.add([node])
    card = vs._engine.store.get("n2")
    assert card is not None
    assert card.embedding is not None
    assert len(card.embedding) == mem_engine.dim


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #
def _seed(engine):
    engine.add_texts(
        [
            "The user prefers dark mode for the terminal.",
            "SQLite is a fast embedded database.",
            "Dark mode reduces eye strain at night.",
        ],
        metadatas=[{"cat": "ui"}, {"cat": "db"}, {"cat": "ui"}],
    )


def test_query_returns_result(mem_engine):
    _seed(mem_engine)
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    q = _VectorStoreQuery(query_str="dark mode", similarity_top_k=3)
    res = vs.query(q)
    assert hasattr(res, "nodes")
    assert hasattr(res, "similarities")
    assert hasattr(res, "ids")
    assert len(res.nodes) == 3
    assert len(res.similarities) == 3
    assert len(res.ids) == 3
    assert all(0.0 <= s <= 1.0 for s in res.similarities)
    # similarities sorted descending
    assert res.similarities == sorted(res.similarities, reverse=True)


def test_query_ranks_relevant_first(mem_engine):
    _seed(mem_engine)
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    q = _VectorStoreQuery(query_str="dark mode", similarity_top_k=3)
    res = vs.query(q)
    assert "dark mode" in res.nodes[0].text.lower()
    # sqlite card is least relevant
    sqlite_idx = next(i for i, n in enumerate(res.nodes) if "sqlite" in n.text.lower())
    assert sqlite_idx == 2


def test_query_with_explicit_embedding(mem_engine):
    ids = mem_engine.add_texts(["dark mode", "sqlite"])
    vec = mem_engine.store.get(ids[0]).embedding
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    q = _VectorStoreQuery(query_embedding=vec, similarity_top_k=2)
    res = vs.query(q)
    assert len(res.nodes) == 2
    assert res.nodes[0].text == "dark mode"  # self-match


def test_query_default_top_k(mem_engine):
    _seed(mem_engine)
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    q = _VectorStoreQuery(query_str="database")  # similarity_top_k unset -> 5
    res = vs.query(q)
    assert len(res.nodes) == 3  # only 3 cards in the store


# --------------------------------------------------------------------------- #
# delete + metadata + engine passthrough + persist
# --------------------------------------------------------------------------- #
def test_delete(mem_engine):
    ids = mem_engine.add_texts(["keep", "drop"])
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    vs.delete(ids[1])
    assert vs._engine.count() == 1


def test_metadata_passthrough(mem_engine):
    node = MockTextNode(text="has meta", metadata={"category": "cat-0"}, id_="m1")
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    vs.add([node])
    res = vs.query(_VectorStoreQuery(query_str="meta", similarity_top_k=1))
    assert res.nodes[0].metadata["category"] == "cat-0"
    # tags list is preserved in metadata too
    assert "tags" in res.nodes[0].metadata


def test_engine_passthrough(mem_engine):
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    assert vs._engine is mem_engine
    assert vs.stores_text is True


def test_persist_is_noop(mem_engine):
    vs = IsotopeZeroVectorStore(engine=mem_engine)
    # Should not raise and should not write anything.
    vs.persist("/tmp/whatever_izero")
    assert vs._engine.count() == 0


# --------------------------------------------------------------------------- #
# real-llamaindex integration (skips if llama_index absent)
# --------------------------------------------------------------------------- #
def test_with_real_llamaindex(tmp_db_path):
    pytest.importorskip("llama_index")
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import BasePydanticVectorStore

    vs = IsotopeZeroVectorStore(db_path=tmp_db_path)
    assert isinstance(vs, BasePydanticVectorStore)
    ids = vs.add([TextNode(text="real node", metadata={"k": "v"})])
    assert len(ids) == 1


# --------------------------------------------------------------------------- #
# LlamaIndex parity: metadata filtering → scoping
# (audit item 4 — query.filter drives multi-tier scope isolation)
# --------------------------------------------------------------------------- #
def test_add_metadata_scope_isolates_cards(mem_engine):
    """A node carrying ``metadata={"scope": "A"}`` lands in scope A and is
    hidden from a default-scope query."""
    from izero_adapters.llamaindex import _VectorStoreQuery

    vs = IsotopeZeroVectorStore(engine=mem_engine)
    vs.add(make_text_nodes(2, prefix="pub"))  # default scope
    scoped = MockTextNode(text="scoped secret", id_="s1")
    scoped.metadata = {"scope": "tenant-a"}
    vs.add([scoped])
    # Default-scope query must NOT surface the tenant-a card.
    res = vs.query(_VectorStoreQuery(query_str="scoped secret", similarity_top_k=5))
    texts = [n.text for n in res.nodes]
    assert "scoped secret" not in texts


def test_query_filter_dict_selects_scope(mem_engine):
    """``query.filter={"scope": "tenant-a"}`` confines results to that scope."""
    from izero_adapters.llamaindex import _VectorStoreQuery

    vs = IsotopeZeroVectorStore(engine=mem_engine)
    # Two scopes, same text.
    a = MockTextNode(text="shared fact", id_="a1"); a.metadata = {"scope": "tenant-a"}
    b = MockTextNode(text="shared fact", id_="b1"); b.metadata = {"scope": "tenant-b"}
    vs.add([a, b])
    # Query with a dict filter selecting tenant-a → only the a1 card.
    q = _VectorStoreQuery(query_str="shared fact", similarity_top_k=5)
    q.filter = {"scope": "tenant-a"}
    res = vs.query(q)
    ids = res.ids
    assert ids == ["a1"], f"expected only tenant-a card, got {ids}"


def test_query_filter_user_id_alias_for_scope(mem_engine):
    """``user_id`` is accepted as a scope alias (mem0/legacy parity)."""
    from izero_adapters.llamaindex import _VectorStoreQuery

    vs = IsotopeZeroVectorStore(engine=mem_engine)
    a = MockTextNode(text="user fact", id_="u1"); a.metadata = {"scope": "user-42"}
    other = MockTextNode(text="user fact", id_="u2")  # default scope
    vs.add([a, other])
    q = _VectorStoreQuery(query_str="user fact", similarity_top_k=5)
    q.filter = {"user_id": "user-42"}
    res = vs.query(q)
    assert res.ids == ["u1"]
