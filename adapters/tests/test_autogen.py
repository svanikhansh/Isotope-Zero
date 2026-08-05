"""Tests for the AutoGen memory adapter.

All tests run WITHOUT ``pyautogen`` installed, exercising the standalone
``IsotopeZeroMemory`` class and its session-tag isolation. The
``attach_to_agent`` helper is verified to raise a clean error when pyautogen is
absent (no traceback leak).
"""
from __future__ import annotations

import pytest

from izero_adapters.autogen import IsotopeZeroMemory, _HAS_AUTOGEN


# --------------------------------------------------------------------------- #
# remember / recall basics
# --------------------------------------------------------------------------- #
def test_remember_returns_id(mem_engine):
    mem = IsotopeZeroMemory(engine=mem_engine, agent_id="a1")
    cid = mem.remember("The user likes Python.", metadata={"turn": 1})
    assert isinstance(cid, str)
    assert mem.count() == 1


def test_recall_returns_dicts(mem_engine):
    mem = IsotopeZeroMemory(engine=mem_engine, agent_id="a1")
    mem.remember("alpha fact about cats")
    mem.remember("beta fact about dogs")
    hits = mem.recall("cats", top_k=2)
    assert isinstance(hits, list)
    assert len(hits) <= 2
    assert all("text" in h and "score" in h and "tags" in h for h in hits)


# --------------------------------------------------------------------------- #
# session-tag isolation (the core multi-agent feature)
# --------------------------------------------------------------------------- #
def test_session_tag_isolation(tmp_db_path, stub_embedder):
    # Two agents sharing one DB file via separate engines on the same path.
    mem_a = IsotopeZeroMemory(
        db_path=tmp_db_path, agent_id="agent_a", embedder=stub_embedder
    )
    # Second adapter opens the same DB path; SQLite WAL allows concurrent
    # readers and the store writes through its own connection.
    mem_b = IsotopeZeroMemory(
        db_path=tmp_db_path, agent_id="agent_b", embedder=stub_embedder
    )
    mem_a.remember("alpha memory unique to agent a")
    mem_b.remember("beta memory unique to agent b")

    a_hits = mem_a.recall("alpha", top_k=5)
    b_hits = mem_b.recall("beta", top_k=5)

    # Agent a only sees its own memories.
    a_texts = [h["text"] for h in a_hits]
    assert any("alpha" in t for t in a_texts)
    assert not any("beta" in t for t in a_texts)

    # Agent b only sees its own memories.
    b_texts = [h["text"] for h in b_hits]
    assert any("beta" in t for t in b_texts)
    assert not any("alpha" in t for t in b_texts)


def test_recall_filter_session_false_is_global(tmp_db_path, stub_embedder):
    mem_a = IsotopeZeroMemory(
        db_path=tmp_db_path, agent_id="agent_a", embedder=stub_embedder
    )
    mem_b = IsotopeZeroMemory(
        db_path=tmp_db_path, agent_id="agent_b", embedder=stub_embedder
    )
    mem_a.remember("alpha global")
    mem_b.remember("beta global")
    # Global recall from agent a's handle should see both.
    hits = mem_a.recall("global", top_k=10, filter_session=False)
    texts = [h["text"] for h in hits]
    assert any("alpha" in t for t in texts)
    assert any("beta" in t for t in texts)


def test_no_agent_id_is_global(mem_engine):
    mem = IsotopeZeroMemory(engine=mem_engine)  # no agent_id
    assert mem.session_tag is None
    mem.remember("global memory")
    # recall with filter_session=True is a no-op filter (no tag) -> global.
    hits = mem.recall("global", top_k=5)
    assert len(hits) == 1


# --------------------------------------------------------------------------- #
# metadata + forget + clear_session + count
# --------------------------------------------------------------------------- #
def test_metadata_passthrough(mem_engine):
    mem = IsotopeZeroMemory(engine=mem_engine, agent_id="a1")
    mem.remember("a turn happened", metadata={"turn": 3, "role": "user"})
    hits = mem.recall("turn", top_k=1)
    md = hits[0]["metadata"]
    assert md["turn"] == "3"  # values stringify via key=value tag pairs
    assert md["role"] == "user"


def test_forget(mem_engine):
    mem = IsotopeZeroMemory(engine=mem_engine, agent_id="a1")
    cid = mem.remember("forget me")
    assert mem.count() == 1
    assert mem.forget(cid) is True
    assert mem.count() == 0


def test_clear_session(tmp_db_path, stub_embedder):
    mem_a = IsotopeZeroMemory(
        db_path=tmp_db_path, agent_id="agent_a", embedder=stub_embedder
    )
    mem_b = IsotopeZeroMemory(
        db_path=tmp_db_path, agent_id="agent_b", embedder=stub_embedder
    )
    for i in range(3):
        mem_a.remember(f"alpha {i}")
    for i in range(2):
        mem_b.remember(f"beta {i}")
    deleted = mem_a.clear_session()
    assert deleted == 3
    # Agent a is empty; agent b is untouched.
    assert mem_a.count() == 0
    assert mem_b.count() == 2


def test_clear_session_without_tag_is_noop(mem_engine):
    mem = IsotopeZeroMemory(engine=mem_engine)  # no session tag
    mem.remember("x")
    assert mem.clear_session() == 0  # refuse to wipe global store
    assert mem.count() == 1


def test_count_session_vs_global(tmp_db_path, stub_embedder):
    mem_a = IsotopeZeroMemory(
        db_path=tmp_db_path, agent_id="agent_a", embedder=stub_embedder
    )
    mem_b = IsotopeZeroMemory(
        db_path=tmp_db_path, agent_id="agent_b", embedder=stub_embedder
    )
    mem_a.remember("a1")
    mem_a.remember("a2")
    mem_b.remember("b1")
    assert mem_a.count() == 2          # session_only
    assert mem_a.count(session_only=False) == 3  # global


# --------------------------------------------------------------------------- #
# attach_to_agent (pyautogen absent -> clean error)
# --------------------------------------------------------------------------- #
def test_attach_to_agent_without_autogen_raises(mem_engine):
    mem = IsotopeZeroMemory(engine=mem_engine, agent_id="a1")
    if _HAS_AUTOGEN:
        pytest.skip("pyautogen is installed; skipping the absent-path test")
    with pytest.raises(RuntimeError, match="pyautogen"):
        mem.attach_to_agent(object())


def test_engine_passthrough(mem_engine):
    mem = IsotopeZeroMemory(engine=mem_engine, agent_id="a1")
    assert mem._engine is mem_engine
    assert mem.is_real == mem_engine.is_real
