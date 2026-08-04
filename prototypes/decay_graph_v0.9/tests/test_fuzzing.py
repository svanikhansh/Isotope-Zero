"""Property-based fuzzing + adversarial edge-case tests for isotope_zero.

This suite uses `hypothesis` to hammer the public API (store, triage, tokens,
embeddings) with generated and hostile inputs, asserting only the invariants
the contract promises: round-trip identity, no-crash, monotonicity, and
parameterized-query safety. Semantic-correctness of the ONNX embedder is NOT
asserted — only structural invariants (dimensionality, determinism,
L2-norm) that hold in BOTH real and fallback modes.

Run:
    .venv/bin/python -m pytest tests/test_fuzzing.py -v
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from isotope_zero.core.store import MemoryStore
from isotope_zero.core.triage import classify_action, compress_to_card
from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.tokens import estimate_tokens
from isotope_zero.types import ActionType, MemoryCard, now_ts

# Float32 (de)serialization tolerance: a round-trip through array('f') loses
# at most ~1e-6 relative precision. 1e-5 abs gives slack for the cumulative
# error over one encode + decode.
F32_TOL = 1e-5


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _norm(v: list[float]) -> list[float]:
    """L2-normalize a vector; leave a zero vector untouched."""
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def _make_embedder() -> EmbeddingEngine:
    """Build the embedding engine once per process; construction may download."""
    return EmbeddingEngine()


# --------------------------------------------------------------------------- #
# Hypothesis strategies
# --------------------------------------------------------------------------- #
# General text: printable unicode, bounded length so individual examples stay
# fast even when fed to the (real) ONNX embedder.
text_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        whitelist_characters="\x00",
    ),
    min_size=0,
    max_size=200,
)

# Small alphabetic strings safe for use as a card id (sqlite TEXT PRIMARY KEY).
# Excludes null/control chars so the PK stays well-behaved; the `fact`/`evidence`
# strategies below exercise hostile text separately.
id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-."),
    min_size=1,
    max_size=32,
)

# Free-form fact / evidence text including whitespace, punctuation, emoji,
# and the null byte (to exercise SQLite TEXT round-trip fidelity).
fact_text_strategy = st.sampled_from(
    [
        "The user prefers Rust.",
        "I love 🦀 Rust and 日本語 🚀",
        "'; DROP TABLE memories; --",
        "') OR 1=1; --",
        "UNION SELECT * FROM memories",
        "a\x00b",
        "",
        "   ",
        "multi\nline\ttabbed",
    ]
) | st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        whitelist_characters="\x00\n\t ",
    ),
    min_size=0,
    max_size=200,
)

# A short list of simple tag strings.
tags_strategy = st.lists(
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-."),
        min_size=1,
        max_size=20,
    ),
    max_size=5,
    unique=True,
)

# Small float vectors used directly for embedding round-trip (NOT from the
# embedder — these test the array('f') encode/decode path cheaply).
small_vec_strategy = st.lists(
    st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
    min_size=0,
    max_size=16,
)

# A full MemoryCard built from generated fields. `embedding` is either None
# (NULL blob path) or a normalized small float vector (real blob path).
@st.composite
def card_strategy(draw) -> MemoryCard:
    cid = draw(id_strategy)
    fact = draw(fact_text_strategy)
    evidence = draw(fact_text_strategy)
    tags = draw(tags_strategy)
    tokens = draw(st.integers(min_value=0, max_value=10_000))
    ts = draw(st.floats(min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False))
    access = draw(st.integers(min_value=0, max_value=1000))
    last_access = draw(st.floats(min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False))
    use_emb = draw(st.booleans())
    emb = None
    if use_emb:
        raw = draw(small_vec_strategy)
        emb = _norm(raw) if raw else [0.0, 1.0, 0.0]
    return MemoryCard(
        id=cid,
        fact=fact,
        evidence=evidence,
        timestamp=ts,
        tags=tags,
        embedding=emb,
        source_tokens=tokens,
        access_count=access,
        last_access=last_access,
    )


# =========================================================================== #
# Property tests (invariants that must ALWAYS hold)
# =========================================================================== #
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(card=card_strategy())
def test_sqlite_roundtrip_identity(card: MemoryCard) -> None:
    """add(card) then get(card.id) returns a card equal on every persisted field.

    fact/evidence/tags/source_tokens/access_count/last_access are exact; timestamp
    is a float compare with tolerance; embedding round-trips through float32
    BLOB so elementwise compare with abs tolerance, and None stays None.
    """
    store = MemoryStore(":memory:")
    try:
        store.add(card)
        got = store.get(card.id)
        assert got is not None
        assert got.id == card.id
        assert got.fact == card.fact
        assert got.evidence == card.evidence
        assert got.tags == card.tags
        assert got.source_tokens == card.source_tokens
        assert got.access_count == card.access_count
        # Store contract: last_access=0.0 means "not yet set" and is substituted
        # with the card's timestamp at write time; a non-zero last_access echoes.
        expected_last = card.last_access if card.last_access else card.timestamp
        assert got.last_access == pytest.approx(expected_last, abs=1e-5)
        assert got.timestamp == pytest.approx(card.timestamp, rel=1e-6)
        if card.embedding is None:
            assert got.embedding is None
        else:
            assert got.embedding is not None
            assert len(got.embedding) == len(card.embedding)
            for a, b in zip(got.embedding, card.embedding):
                assert a == pytest.approx(b, abs=F32_TOL)
    finally:
        store.close()


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(cards=st.lists(card_strategy(), min_size=0, max_size=10, unique_by=lambda c: c.id))
def test_add_all_consistency(cards: list[MemoryCard]) -> None:
    """After adding N cards, store.all() ids == added ids and count() == N."""
    store = MemoryStore(":memory:")
    try:
        for c in cards:
            store.add(c)
        added_ids = {c.id for c in cards}
        got_ids = {c.id for c in store.all()}
        assert got_ids == added_ids
        assert store.count() == len(cards)
    finally:
        store.close()


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(card=card_strategy())
def test_delete_idempotency(card: MemoryCard) -> None:
    """delete(id) twice: first True (if it existed), second False; get(id) None after."""
    store = MemoryStore(":memory:")
    try:
        store.add(card)
        first = store.delete(card.id)
        second = store.delete(card.id)
        assert first is True
        assert second is False
        assert store.get(card.id) is None
    finally:
        store.close()


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(card=card_strategy())
def test_update_upsert(card: MemoryCard) -> None:
    """update(card) on a new id inserts; on an existing id overwrites fields
    with no duplicate row (count stays 1)."""
    store = MemoryStore(":memory:")
    try:
        # First update = insert.
        store.update(card)
        assert store.count() == 1
        got = store.get(card.id)
        assert got is not None
        assert got.fact == card.fact
        assert got.evidence == card.evidence

        # Second update with a different fact/evidence on the SAME id = overwrite.
        revised = MemoryCard(
            id=card.id,
            fact=card.fact + "_rev",
            evidence=card.evidence + "_rev",
            timestamp=card.timestamp + 1.0,
            tags=card.tags + ["rev"],
            embedding=card.embedding,
            source_tokens=card.source_tokens + 1,
            access_count=card.access_count,
            last_access=card.last_access,
        )
        store.update(revised)
        assert store.count() == 1  # no dup row
        got2 = store.get(card.id)
        assert got2 is not None
        assert got2.fact == revised.fact
        assert got2.evidence == revised.evidence
    finally:
        store.close()


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    query=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        min_size=0,
        max_size=8,
    )
)
def test_vector_search_never_raises(query: list[float]) -> None:
    """vector_search never raises on any NaN-free float vector, including
    empty and all-zeros. For degenerate queries it returns []."""
    store = MemoryStore(":memory:")
    try:
        # Empty store: any query -> [].
        assert store.vector_search(query, k=5) == []
        # Add a card with a real embedding and retry — still must not raise.
        c = MemoryCard(
            id="probe",
            fact="probe",
            evidence="",
            timestamp=now_ts(),
            tags=[],
            embedding=_norm([1.0, 0.0, 0.0, 0.0]),
            source_tokens=1,
        )
        store.add(c)
        hits = store.vector_search(query, k=5)
        assert isinstance(hits, list)
        # All-zoles / empty query -> [] even with data present.
        if not query or all(v == 0.0 for v in query):
            assert hits == []
    finally:
        store.close()


def test_vector_search_exact_match_returns_score_one() -> None:
    """For a store with a known card whose embedding == query (normalized),
    that card is in the top results with score ~1.0. For all-zeros query: [].
    For empty store: []."""
    store = MemoryStore(":memory:")
    try:
        # Empty store.
        assert store.vector_search(_norm([1.0, 0.0, 0.0]), k=5) == []
        emb = _norm([1.0, 0.0, 0.0, 0.0])
        c = MemoryCard(
            id="match",
            fact="exact match",
            evidence="",
            timestamp=now_ts(),
            tags=[],
            embedding=emb,
            source_tokens=2,
        )
        store.add(c)
        hits = store.vector_search(emb, k=5)
        assert len(hits) >= 1
        top_card, top_score = hits[0]
        assert top_card.id == "match"
        assert top_score == pytest.approx(1.0, abs=1e-5)
        # Degenerate queries still return [].
        assert store.vector_search([0.0, 0.0, 0.0, 0.0], k=5) == []
        assert store.vector_search([], k=5) == []
    finally:
        store.close()


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=text_strategy, context=text_strategy)
def test_triage_classify_never_crashes(text: str, context: str) -> None:
    """classify_action(any_text, context) returns an ActionResult with a valid
    ActionType and never raises, even for empty/giant/weird input."""
    result = classify_action(text, context)
    assert isinstance(result.action, ActionType)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.escalated, bool)
    assert isinstance(result.reasoning, str)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=text_strategy)
def test_triage_compress_never_crashes(text: str) -> None:
    """compress_to_card(any_text) returns a MemoryCard and never raises, even
    for empty/giant/weird input. The card may have an empty fact — that's fine."""
    card = compress_to_card(text)
    assert isinstance(card, MemoryCard)
    assert isinstance(card.fact, str)
    assert isinstance(card.evidence, str)
    assert isinstance(card.tags, list)
    assert isinstance(card.source_tokens, int)
    assert card.source_tokens >= 0


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(a=text_strategy, b=text_strategy.filter(lambda s: s != ""))
def test_estimate_tokens_monotonic(a: str, b: str) -> None:
    """estimate_tokens(a+b) >= estimate_tokens(a) for non-empty b:
    concatenation can't reduce the token count."""
    ta = estimate_tokens(a)
    tab = estimate_tokens(a + b)
    assert tab >= ta


# =========================================================================== #
# Embedding-engine structural invariants (run in BOTH real and fallback modes)
# =========================================================================== #
def test_embedder_structural_invariants() -> None:
    """The embedder — real ONNX or fallback — must satisfy structural
    invariants in either mode: returns a list of floats of the engine's dim,
    is deterministic, and is L2-normalized (or zero for empty input)."""
    eng = _make_embedder()
    a = eng.embed_text("the user prefers rust and tea")
    b = eng.embed_text("the user prefers rust and tea")
    z = eng.embed_text("")

    assert isinstance(a, list)
    assert len(a) == eng.dim
    # Determinism: identical inputs -> identical vectors.
    assert a == b
    # Empty input -> zero vector of the right length.
    assert len(z) == eng.dim
    assert all(v == 0.0 for v in z)
    # Non-empty output is L2-normalized (within float slack).
    norm = math.sqrt(sum(x * x for x in a))
    assert norm == pytest.approx(1.0, abs=1e-3)


def test_embedder_vector_search_self_score_one() -> None:
    """A card whose embedding is the embedder's own output for a text must
    score ~1.0 when that same text is re-embedded and used as the query.
    This holds in BOTH real and fallback modes (deterministic embedder).

    Structural invariant only — does NOT assert semantic neighborliness.
    """
    eng = _make_embedder()
    text = "the user prefers rust and tea"
    emb = eng.embed_text(text)
    store = MemoryStore(":memory:")
    try:
        c = MemoryCard(
            id="self",
            fact=text,
            evidence=text,
            timestamp=now_ts(),
            tags=[],
            embedding=emb,
            source_tokens=estimate_tokens(text),
        )
        store.add(c)
        # Query with the SAME text's embedding -> self-similarity ~1.0.
        query = eng.embed_text(text)
        hits = store.vector_search(query, k=5)
        assert len(hits) >= 1
        top_card, top_score = hits[0]
        assert top_card.id == "self"
        # Deterministic identical vectors -> cosine 1.0 (store clamps to [0,1]).
        assert top_score == pytest.approx(1.0, abs=1e-3)
    finally:
        store.close()


def test_embedder_semantic_mode_optional() -> None:
    """If a test STRICTLY needs real semantic behavior, it skips gracefully
    when only the fallback is available. This guard itself runs unconditionally
    and documents the skip contract."""
    eng = _make_embedder()
    if not eng.is_real:
        pytest.skip("Real ONNX embedder unavailable; semantic-dependent assertion skipped.")


# =========================================================================== #
# Edge-case tests (dedicated, assert specific behavior — no crash)
# =========================================================================== #
def test_extreme_length_fact_roundtrips() -> None:
    """A 100KB+ fact string add->get round-trips exactly; estimate_tokens
    returns a large int."""
    big = "A" * 100_003  # > 100KB
    assert len(big) > 100_000
    store = MemoryStore(":memory:")
    try:
        c = MemoryCard(
            id="big",
            fact=big,
            evidence="e",
            timestamp=now_ts(),
            tags=[],
            source_tokens=estimate_tokens(big),
        )
        store.add(c)
        got = store.get("big")
        assert got is not None
        assert got.fact == big
        assert len(got.fact) == len(big)
        assert got.source_tokens > 1000
    finally:
        store.close()


def test_empty_string_fact_evidence_preserved_not_none() -> None:
    """Empty string fact/evidence add+get preserves empty string (NOT None)."""
    store = MemoryStore(":memory:")
    try:
        c = MemoryCard(
            id="empty",
            fact="",
            evidence="",
            timestamp=now_ts(),
            tags=[],
            source_tokens=0,
        )
        store.add(c)
        got = store.get("empty")
        assert got is not None
        assert got.fact == ""
        assert got.fact is not None
        assert got.evidence == ""
        assert got.evidence is not None
    finally:
        store.close()


def test_null_bytes_roundtrip() -> None:
    """A fact containing \\x00 round-trips through SQLite TEXT unchanged.
    SQLite stores TEXT with embedded nulls fine; assert get returns it as-is."""
    store = MemoryStore(":memory:")
    try:
        payload = "a\x00b\x00c"
        c = MemoryCard(
            id="null",
            fact=payload,
            evidence="e\x00f",
            timestamp=now_ts(),
            tags=["t\x00g"],
            source_tokens=4,
        )
        store.add(c)
        got = store.get("null")
        assert got is not None
        assert got.fact == payload
        assert "\x00" in got.fact
        assert got.evidence == "e\x00f"
        assert got.tags == ["t\x00g"]
    finally:
        store.close()


def test_unicode_emoji_roundtrips() -> None:
    """fact = 'I love 🦀 Rust and 日本語 🚀' round-trips exactly through the store."""
    store = MemoryStore(":memory:")
    try:
        payload = "I love 🦀 Rust and 日本語 🚀"
        c = MemoryCard(
            id="uni",
            fact=payload,
            evidence=payload,
            timestamp=now_ts(),
            tags=["emoji", "日本語"],
            source_tokens=estimate_tokens(payload),
        )
        store.add(c)
        got = store.get("uni")
        assert got is not None
        assert got.fact == payload
        assert got.evidence == payload
        assert got.tags == ["emoji", "日本語"]
    finally:
        store.close()


@pytest.mark.parametrize(
    "injection",
    [
        "'; DROP TABLE memories; --",
        "') OR 1=1; --",
        "UNION SELECT * FROM memories",
        "Robert'); DROP TABLE students; --",
        "\" OR \"\"=\"",
        "1=1; DELETE FROM memories; --",
    ],
)
def test_sql_injection_strings_stored_as_literal(injection: str) -> None:
    """SQL-injection strings must be stored AS LITERAL TEXT via parameterized
    queries, retrieved unchanged, and MUST NOT drop the table (count() works
    before and after)."""
    store = MemoryStore(":memory:")
    try:
        # Seed with one innocent card so count() is meaningful.
        base = MemoryCard(
            id="base",
            fact="innocent",
            evidence="e",
            timestamp=now_ts(),
            tags=[],
            source_tokens=1,
        )
        store.add(base)
        assert store.count() == 1

        c = MemoryCard(
            id="inj",
            fact=injection,
            evidence=injection,
            timestamp=now_ts(),
            tags=[injection],
            source_tokens=estimate_tokens(injection),
        )
        store.add(c)
        got = store.get("inj")
        assert got is not None
        assert got.fact == injection
        assert got.evidence == injection
        assert got.tags == [injection]

        # The table is intact: count() works and reflects both rows.
        assert store.count() == 2
        # The 'memories' table was NOT dropped — schema introspection still works.
        all_cards = store.all()
        assert len(all_cards) == 2
        assert {c.id for c in all_cards} == {"base", "inj"}
    finally:
        store.close()


def test_bad_tags_json_recovers_to_empty() -> None:
    """Directly injecting malformed tags JSON into the DB must NOT crash
    _row_to_card; it catches JSONDecodeError and returns tags=[] instead."""
    store = MemoryStore(":memory:")
    try:
        c = MemoryCard(
            id="badjson",
            fact="f",
            evidence="e",
            timestamp=now_ts(),
            tags=["good"],
            source_tokens=1,
        )
        store.add(c)
        # Corrupt the tags column with a malformed JSON string directly.
        store._conn.execute(
            "UPDATE memories SET tags=? WHERE id=?",
            ("{not json", "badjson"),
        )
        # get() must go through _row_to_card, which must catch and recover.
        got = store.get("badjson")
        assert got is not None
        assert got.tags == []
        # The rest of the card is still intact.
        assert got.fact == "f"
        assert got.id == "badjson"
    finally:
        store.close()
