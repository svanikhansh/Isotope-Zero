"""Tests for the CrewAI Isotope Zero memory adapter.

All tests run WITHOUT crewai installed — they exercise the standalone
:class:`IsotopeZeroMemory` core and the session-tag isolation design using the
shared ``stub_embedder`` / ``tmp_db_path`` fixtures from ``conftest.py``.
"""
from __future__ import annotations

import pytest

from izero_adapters.crewai import IsotopeZeroMemory, _HAS_CREWAI


# --------------------------------------------------------------------------- #
# Basic remember / recall contract.
# --------------------------------------------------------------------------- #
def test_remember_returns_id(mem_engine):
    """remember returns a card id and the store count increments."""
    mem = IsotopeZeroMemory(engine=mem_engine, crew_id="c", agent_id="a")
    before = mem.count()
    cid = mem.remember("Q3 revenue grew 12% YoY.", metadata={"task": "analysis"})
    assert isinstance(cid, str) and cid.startswith("iz-")
    assert mem.count() == before + 1


def test_recall_returns_dicts(mem_engine):
    """recall returns dicts carrying text / score / tags for each hit."""
    mem = IsotopeZeroMemory(engine=mem_engine, crew_id="c", agent_id="a")
    mem.remember("Revenue growth accelerated in Q3.")
    mem.remember("Margins compressed due to input costs.")
    hits = mem.recall("revenue growth", top_k=5)
    assert len(hits) >= 1
    for h in hits:
        assert {"id", "text", "score", "tags"} <= set(h)
        assert isinstance(h["score"], float)


# --------------------------------------------------------------------------- #
# Session-tag isolation.
# --------------------------------------------------------------------------- #
def test_session_tag_isolation_crew_agent(tmp_db_path, stub_embedder):
    """Same crew, two agents on a shared DB each recall only their own."""
    a = IsotopeZeroMemory(
        db_path=tmp_db_path, embedder=stub_embedder,
        crew_id="research", agent_id="analyst",
    )
    b = IsotopeZeroMemory(
        db_path=tmp_db_path, embedder=stub_embedder,
        crew_id="research", agent_id="writer",
    )
    assert a.session_tag == "crew:research:agent:analyst"
    assert b.session_tag == "crew:research:agent:writer"

    a.remember("Analyst note: revenue up 12%.")
    b.remember("Writer draft: Q3 summary report.")

    a_hits = a.recall("revenue", top_k=10)
    b_hits = b.recall("report", top_k=10)
    assert len(a_hits) == 1 and a_hits[0]["text"].startswith("Analyst note")
    assert len(b_hits) == 1 and b_hits[0]["text"].startswith("Writer draft")
    # Each agent must NOT see the other's memory.
    assert all("writer" not in h["tags"] for h in a_hits)
    assert all("analyst" not in h["tags"] for h in b_hits)


def test_crew_level_isolation(tmp_db_path, stub_embedder):
    """Two different crews on a shared DB are isolated at the crew level."""
    c1 = IsotopeZeroMemory(
        db_path=tmp_db_path, embedder=stub_embedder, crew_id="alpha",
    )
    c2 = IsotopeZeroMemory(
        db_path=tmp_db_path, embedder=stub_embedder, crew_id="bravo",
    )
    assert c1.session_tag == "crew:alpha"
    assert c2.session_tag == "crew:bravo"

    c1.remember("Alpha crew discovered a new signal.")
    c2.remember("Bravo crew shipped the feature.")

    hits1 = c1.recall("signal", top_k=10)
    hits2 = c2.recall("feature", top_k=10)
    assert len(hits1) == 1 and "Alpha" in hits1[0]["text"]
    assert len(hits2) == 1 and "Bravo" in hits2[0]["text"]
    # No cross-crew leakage.
    assert all("crew:bravo" not in h["tags"] for h in hits1)
    assert all("crew:alpha" not in h["tags"] for h in hits2)


def test_recall_filter_session_false_global(tmp_db_path, stub_embedder):
    """filter_session=False sees across sessions on a shared DB."""
    a = IsotopeZeroMemory(
        db_path=tmp_db_path, embedder=stub_embedder,
        crew_id="crew", agent_id="agent_a",
    )
    b = IsotopeZeroMemory(
        db_path=tmp_db_path, embedder=stub_embedder,
        crew_id="crew", agent_id="agent_b",
    )
    a.remember("Agent A wrote about quantum computing.")
    b.remember("Agent B wrote about neural networks.")

    # Global recall from A's handle should see both sessions.
    global_hits = a.recall("wrote", top_k=10, filter_session=False)
    texts = {h["text"] for h in global_hits}
    assert any("Agent A" in t for t in texts)
    assert any("Agent B" in t for t in texts)
    # Session-filtered recall from A sees only A.
    session_hits = a.recall("wrote", top_k=10, filter_session=True)
    assert all("Agent B" not in h["text"] for h in session_hits)


# --------------------------------------------------------------------------- #
# Metadata, forget, clear_session, count.
# --------------------------------------------------------------------------- #
def test_metadata_passthrough(mem_engine):
    """metadata dict round-trips in recall result metadata as key=value."""
    mem = IsotopeZeroMemory(engine=mem_engine, crew_id="c", agent_id="a")
    mem.remember(
        "Q3 revenue grew 12% YoY.",
        metadata={"task": "analysis", "step": 2},
    )
    hits = mem.recall("revenue", top_k=5)
    assert hits, "expected at least one hit"
    meta = hits[0]["metadata"]
    assert meta["task"] == "analysis"
    # Values are stringified by the tag-based store, but the key is preserved.
    assert meta["step"] == "2"


def test_forget(mem_engine):
    """forget(id) removes the card and decreases the session count."""
    mem = IsotopeZeroMemory(engine=mem_engine, crew_id="c", agent_id="a")
    cid = mem.remember("Temporary memory to be forgotten.")
    assert mem.count() == 1
    assert mem.forget(cid) is True
    assert mem.count() == 0
    assert mem.forget("iz-doesnotexist") is False


def test_clear_session(tmp_db_path, stub_embedder):
    """clear_session removes only this session's cards, not others'."""
    a = IsotopeZeroMemory(
        db_path=tmp_db_path, embedder=stub_embedder,
        crew_id="crew", agent_id="agent_a",
    )
    b = IsotopeZeroMemory(
        db_path=tmp_db_path, embedder=stub_embedder,
        crew_id="crew", agent_id="agent_b",
    )
    a.remember("A memory 1")
    a.remember("A memory 2")
    a.remember("A memory 3")
    b.remember("B memory 1")
    b.remember("B memory 2")
    assert a.count() == 3
    assert b.count() == 2

    removed = a.clear_session()
    assert removed == 3
    assert a.count() == 0
    # B's memories must survive A clearing its session.
    assert b.count() == 2


def test_count_session_vs_global(tmp_db_path, stub_embedder):
    """session_only count is a strict subset of the global count."""
    a = IsotopeZeroMemory(
        db_path=tmp_db_path, embedder=stub_embedder,
        crew_id="crew", agent_id="agent_a",
    )
    b = IsotopeZeroMemory(
        db_path=tmp_db_path, embedder=stub_embedder,
        crew_id="crew", agent_id="agent_b",
    )
    a.remember("A one")
    a.remember("A two")
    b.remember("B one")

    assert a.count(session_only=True) == 2
    assert b.count(session_only=True) == 1
    assert a.count(session_only=False) == 3  # both sessions visible globally
    assert a.count(session_only=True) < a.count(session_only=False)


# --------------------------------------------------------------------------- #
# Cross-agent recall + crewai-absence + global (no ids).
# --------------------------------------------------------------------------- #
def test_recall_for_agent(tmp_db_path, stub_embedder):
    """Within a crew, recall_for_agent finds another agent's memories."""
    a = IsotopeZeroMemory(
        db_path=tmp_db_path, embedder=stub_embedder,
        crew_id="research", agent_id="analyst",
    )
    # Writer remembers something; analyst reaches across to read it.
    IsotopeZeroMemory(
        db_path=tmp_db_path, embedder=stub_embedder,
        crew_id="research", agent_id="writer",
    ).remember("Writer drafted the Q3 revenue summary.")

    hits = a.recall_for_agent("writer", "revenue summary", top_k=10)
    assert len(hits) == 1
    assert "Writer" in hits[0]["text"]
    # The hit must carry the writer's session tag, not the analyst's.
    assert "crew:research:agent:writer" in hits[0]["tags"]


def test_attach_to_crew_without_crewai_raises(mem_engine):
    """attach_to_crew raises a clean RuntimeError when crewai is absent."""
    if _HAS_CREWAI:
        pytest.skip("crewai is installed; cannot assert the absent-path.")
    mem = IsotopeZeroMemory(engine=mem_engine, crew_id="c", agent_id="a")
    with pytest.raises(RuntimeError, match="crewai not installed"):
        mem.attach_to_crew(object())


def test_no_ids_is_global(mem_engine):
    """No crew_id / agent_id → no session tag → recall is global."""
    mem = IsotopeZeroMemory(engine=mem_engine)
    assert mem.session_tag is None
    mem.remember("Global memory one.")
    mem.remember("Global memory two.")
    # No session filter applied (no tag), so recall sees everything.
    hits = mem.recall("memory", top_k=10)
    assert len(hits) == 2
    # count() with no tag is global regardless of session_only.
    assert mem.count() == 2
    assert mem.count(session_only=False) == 2
