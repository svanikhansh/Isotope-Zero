"""Multi-tier scoping helpers for isotope_zero (mem0 port).

mem0 isolates memories behind a deterministic session scope built from up to
three actor identifiers (``user_id``, ``agent_id``, ``run_id``). The scope
string is the canonical, order-stable join of the provided identifiers in
``key=value`` form, sorted by key and joined by ``&`` — e.g.
``"agent_id=A1&run_id=R1&user_id=U1"``. This is ported verbatim from
mem0's ``_build_session_scope`` at ``mem0/memory/main.py:407`` (sorted
``["user_id","agent_id","run_id"]`` keys, ``f"{key}={val}"`` parts,
``"&".join(parts)``).

isotope_zero's ``MemoryStore`` already persists a ``scope`` column
(``store.py`` migration at ``:488`` wired into ``add`` at ``:735`` and
``vector_search`` at ``:1398``). This module is the OFF path-portable helper
layer: pure functions to parse/build/match scope strings WITHOUT touching the
store, so routers, extractors, and tests can reason about scoping without a
live DB connection. The store's own ``scope``-filter param on
``vector_search`` / ``sql_lookup`` is the on-disk enforcement; these helpers
are the in-memory contract the callers share.

Semantics (backward compatible):
    - The literal string ``"default"`` and the empty string ``""`` BOTH denote
      the GLOBAL scope (no identifiers). A global card is visible to EVERY
      query, including scoped ones — pre-scoping callers that never set a
      scope keep surfacing exactly as before.
    - A scoped query (e.g. ``user_id=U1``) matches a card iff EVERY key the
      query carries is present on the card with the SAME value. Keys the card
      carries but the query omits are irrelevant (the query is a refinement,
      not an equality, on the card's full identity). This mirrors mem0's
      "filter is a subset of metadata" lookup posture.
    - A global QUERY (``"default"`` / ``""`` / ``None``) matches EVERY card —
      cross-tenant global retrieval, the documented ``scope=None`` behavior.
"""
from __future__ import annotations

# The three mem0 session-identifier keys, in the canonical sort order mem0
# uses at ``mem0/memory/main.py:410`` (``sorted(["user_id","agent_id","run_id"])``
# is ``["agent_id","run_id","user_id"]`` — agent < run < user lexicographically).
# Kept as an explicit tuple (not a derived ``sorted(...)`` call) so the order
# is auditable in one place and the round-trip is obviously stable.
_SCOPE_KEYS: tuple[str, ...] = ("agent_id", "run_id", "user_id")

# The literal global-scope sentinel. Matches ``MemoryStore``'s column default
# (``store.py:489``: ``DEFAULT 'default'``) and ``MemoryCard.scope``'s dataclass
# default (``types.py:80``). Both the empty string and this sentinel parse to
# an empty (global) identifier dict.
GLOBAL_SCOPE: str = "default"


def parse_scope(scope_str: str | None) -> dict[str, str]:
    """Parse a scope string into a ``{key: value}`` dict.

    Accepts the canonical ``"agent_id=A1&run_id=R1&user_id=U1"`` form produced
    by :func:`build_scope`, and tolerates the messy real-world variants a caller
    might hand-build: a trailing ``&``, surrounding whitespace, ``None``, the
    empty string, and the ``GLOBAL_SCOPE`` sentinel (``"default"``).

    ``None`` / ``""`` / ``"default"`` (after ``strip()``) all parse to ``{}``
    (the global scope). Unknown keys are preserved verbatim — this function is
    a parser, not a validator; a future key (``project_id``) should round-trip
    without a code change. Empty VALUES (``"user_id="``) are dropped so a
    half-built scope string does not inject a ``"user_id": ""`` identity that
    would then fail to match a real ``user_id=U1`` card.

    Inverse of :func:`build_scope`: ``build_scope(parse_scope(s)) == s`` for
    every canonical (non-empty, sorted) ``s``.
    """
    if scope_str is None:
        return {}
    s = scope_str.strip()
    if not s or s == GLOBAL_SCOPE:
        return {}
    out: dict[str, str] = {}
    # Tolerate a trailing ampersand (``a=1&b=2&``) by filtering empties; split
    # on ``&`` and then on the FIRST ``=`` so a value containing ``=`` survives.
    for part in s.split("&"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            # A bare token with no ``=`` is not a ``key=value`` pair; drop it
            # rather than guessing, so a malformed scope stays global-ish.
            continue
        key, _, val = part.partition("=")
        key = key.strip()
        val = val.strip()
        if not key or not val:
            continue
        out[key] = val
    return out


def build_scope(d: dict[str, str] | None) -> str:
    """Build the canonical scope string from a ``{key: value}`` dict.

    Ports ``mem0/memory/main.py:407`` (``_build_session_scope``): keys are
    sorted, only non-empty values are emitted, parts are ``f"{key}={val}"``,
    joined by ``&``. The sort is over ALL keys (not just the three known
    mem0 keys) so a future identifier round-trips deterministically too.

    ``None`` / ``{}`` / an all-empty-values dict all produce ``GLOBAL_SCOPE``
    (``"default"``) — the global scope. This is the inverse of
    :func:`parse_scope` for canonical inputs.
    """
    if not d:
        return GLOBAL_SCOPE
    # Bind v in a nested comprehension so the ``if v`` filter sees it (the
    # flat ``for k in sorted(d) if v`` form leaves ``v`` unbound — Python only
    # names the loop target, not the mapped value).
    parts = [f"{k}={v}" for k in sorted(d) for v in [d[k]] if v]
    if not parts:
        return GLOBAL_SCOPE
    return "&".join(parts)


def match_scope(card_scope: str | None, query_scope: str | None) -> bool:
    """Return True iff ``card_scope`` is visible to ``query_scope``.

    Visibility rules (backward compatible — a pre-scoping caller's cards stay
    visible to every query):

    1. A GLOBAL query (``query_scope`` is ``None`` / ``""`` / ``"default"``)
       matches EVERY card — cross-tenant retrieval, the documented
       ``scope=None`` behavior of ``MemoryStore.vector_search``.
    2. A GLOBAL card (``card_scope`` is ``None`` / ``""`` / ``"default"``) is
       visible to EVERY query, including scoped ones. This is the backward-
       compatibility guarantee: cards written before scoping existed (which
       carry the column default ``"default"``) keep surfacing in scoped
       retrieval exactly as they did in the global-only world.
    3. Otherwise the query is a SUBSET filter: the card matches iff EVERY key
       the query carries is present on the card with the SAME value. Extra
       keys on the card that the query omits do NOT disqualify it — the query
       narrows the candidate set, it does not require an exact full-identity
       match. This mirrors mem0's "filter is a subset of metadata" lookup
       posture at ``mem0/memory/main.py:604`` (``search_filters`` are the
       ``user_id``/``agent_id``/``run_id`` keys extracted from the call).
    """
    q = parse_scope(query_scope)
    if not q:
        # Global query => sees everything.
        return True
    c = parse_scope(card_scope)
    if not c:
        # Global card => visible to every query (backward compat).
        return True
    # Subset filter: every query key must match the card's value for that key.
    return all(c.get(k) == v for k, v in q.items())


def _smoke() -> None:
    """Inline smoke test — run with ``python -m isotope_zero.core.scoping``."""
    # Round-trip.
    d = {"user_id": "U1", "agent_id": "A1", "run_id": "R1"}
    s = build_scope(d)
    assert s == "agent_id=A1&run_id=R1&user_id=U1", s
    assert parse_scope(s) == d, parse_scope(s)
    # Global variants.
    assert build_scope(None) == GLOBAL_SCOPE
    assert build_scope({}) == GLOBAL_SCOPE
    assert parse_scope(None) == {}
    assert parse_scope("") == {}
    assert parse_scope("default") == {}
    assert parse_scope("agent_id=A1&") == {"agent_id": "A1"}  # trailing &
    # Match: global query sees all.
    assert match_scope("user_id=U1", None) is True
    assert match_scope("user_id=U1", "") is True
    assert match_scope("user_id=U1", "default") is True
    # Match: global card visible to scoped query (backward compat).
    assert match_scope("default", "user_id=U1") is True
    assert match_scope("", "user_id=U1") is True
    assert match_scope(None, "user_id=U1") is True
    # Match: exact scoped match.
    assert match_scope("user_id=U1", "user_id=U1") is True
    # Match: query subset of card identity.
    assert match_scope("agent_id=A1&run_id=R1&user_id=U1", "user_id=U1") is True
    # Mismatch: query key absent on card.
    assert match_scope("agent_id=A1", "user_id=U1") is False
    # Mismatch: query key present but value differs.
    assert match_scope("user_id=U1", "user_id=U2") is False
    print("scoping._smoke OK")


if __name__ == "__main__":
    _smoke()
