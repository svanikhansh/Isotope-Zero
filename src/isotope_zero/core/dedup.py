"""Content-aware dedup + TTL for isotope_zero (mem0 port).

Ports mem0's content-hash dedup into isotope_zero's SQLite store. mem0 inlines
dedup into ONE file ``mem0/memory/main.py``; it hashes the memory TEXT with
``hashlib.md5(text.encode()).hexdigest()`` at mem0/memory/main.py:1010 (also
:1960 for ``_create_memory``, :2047). There the hash keys a ``seen_hashes`` set
so a second extraction of the SAME text within one batch is skipped before
insert, and the same hash is stored in ``metadata["hash"]`` so future writes
can detect the duplicate cross-batch.

We adapt that pattern to isotope_zero's ``MemoryCard`` model with two
deliberate differences:

1. **sha256, not md5.** mem0's md5 is inherited from its Qdrant-payload key
   convention; isotope_zero stores the hash as a plain indexed SQLite column
   (``idx_memories_fingerprint``) where collision resistance matters more than
   speed, so we upgrade to sha256.

2. **Content-aware, not text-only.** mem0 hashes ``text`` alone — but two
   cards with the same fact text yet different entity tags are semantically
   distinct (a ``"preference"`` vs ``"todo"`` tag on the same sentence is a
   different memory). So ``content_aware_fingerprint`` folds the SORTED entity
   list into the hash input: the fact joined with a double-pipe then the
   comma-joined sorted tags. Sorting makes the hash order-independent (a card
   tagged ``["b","a"]`` and one tagged ``["a","b"]`` hash identically, which
   is correct — they ARE the same memory).

Design rules (per repo constraints):
  - Pure stdlib: ``hashlib``. No external deps.
  - Does NOT touch the DB: the store owns the connection + the schema
    migration (additive ALTER ADD COLUMN for ``content_fingerprint``,
    ``ttl_seconds``, ``expiration_timestamp``). This module is a pure helper.
  - Cross-module import is from the store side (store imports this helper);
    this file has no upward imports, so it is import-safe even if sibling
    modules are absent.
  - Double-quoted strings, ``from __future__ import annotations``, typed
    signatures, docstrings cite mem0 source path:line.
"""

from __future__ import annotations

import hashlib


def content_aware_fingerprint(fact: str, entities: list[str]) -> str:
    """Content-aware SHA-256 fingerprint of a fact + its sorted entity tags.

    Mirrors mem0's per-memory hash (mem0/memory/main.py:1010 —
    ``hashlib.md5(text.encode()).hexdigest()``) but is **content-aware**: the
    hash input is ``fact + "||" + ",".join(sorted(entities))`` so two cards
    with the same fact text but different tags hash DIFFERENTLY, and two cards
    with the same fact + same tag set (in any order) hash IDENTICALLY. When
    ``entities`` is empty the hash is over the fact alone (preserving the
    mem0 text-only behaviour for untagged memories).

    Args:
        fact: the memory fact text.
        entities: the card's tags / entity list. Order is irrelevant — they
            are sorted before hashing so ``["b","a"]`` and ``["a","b"]`` yield
            the same fingerprint.

    Returns:
        The 64-char lowercase hex sha256 digest. Never empty.
    """
    if not fact:
        # An empty fact still needs a stable hash; hash the empty string so
        # the column is never NULL for a live card (NULL would break the
        # dedup SELECT's ``content_fingerprint = ?`` lookup).
        fact = ""
    if entities:
        payload = fact + "||" + ",".join(sorted(entities))
    else:
        payload = fact
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # Determinism: same fact + same tags (any order) -> same fingerprint.
    fp1 = content_aware_fingerprint("The user prefers dark mode.", ["ui", "pref"])
    fp2 = content_aware_fingerprint("The user prefers dark mode.", ["pref", "ui"])
    assert fp1 == fp2, "tag order must not change the fingerprint"
    print("same fact+tags (reordered) -> identical:", fp1 == fp2)

    # Different tags -> different fingerprint.
    fp3 = content_aware_fingerprint("The user prefers dark mode.", ["todo"])
    assert fp1 != fp3, "different tags must change the fingerprint"
    print("same fact, diff tags -> different:", fp1 != fp3)

    # No tags -> fact-only hash (mem0 text-only behaviour).
    fp4 = content_aware_fingerprint("A fact.", [])
    fp5 = content_aware_fingerprint("A fact.", [])
    assert fp4 == fp5
    print("empty-tags deterministic:", fp4 == fp5)

    # 64-char hex.
    assert len(fp1) == 64 and all(c in "0123456789abcdef" for c in fp1)
    print("fingerprint is 64-char lowercase hex:", len(fp1) == 64)

    print("smoke test OK")
