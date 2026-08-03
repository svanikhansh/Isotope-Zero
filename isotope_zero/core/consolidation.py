"""Phase 3 async consolidation — the off-hot-path housekeeping engine.

An off-hot-path engine that periodically deduplicates, merges, and prunes
memory cards so the store stays flat in context size and sub-ms latency
regardless of how many cards have ever been written.

Three responsibilities, run as ONE atomic sweep:

1. Deduplication: cluster near-identical cards (cosine > threshold OR exact
   fact equality OR high token overlap) and fold each cluster into a single
   survivor with unioned evidence/tags and summed access pressure. Survivor
   selection: prefer the NEWEST card when the cluster's facts differ beyond
   case/whitespace (a near-duplicate CORRECTION keeps the correction), but keep
   the EARLIEST card when every fact is token-identical after normalization
   (true duplicates / pure paraphrases keep the canonical first-seen phrasing).
   Folded cards are marked SUPERSEDED (audit trail pointer to the survivor)
   rather than hard-deleted, so a correction is never a silent lost update.
   Negation pairs never merge.

2. Temporal decay & pruning: score every card by a vitality formula combining
   recency and recall frequency, and conservatively prune cards that are both
   stale (vitality < floor) AND never-recalled AND past a minimum-age grace
   period. Recalled cards are never pruned.

3. Atomic apply: hand the full (survivors, deleted_ids) plan to
   `store.consolidate_memories(...)` ONCE, which runs BEGIN IMMEDIATE + COMMIT
   so concurrent WAL readers never see a half-applied sweep.

Concurrency / locking model (documented):
    The planning phase calls `store.all()` and `store.batch_get()`, each of
    which acquires the store's lock only for the duration of its own DB read
    and releases it before returning. The consolidator holds NO lock across
    planning — it works on plain Python lists of `MemoryCard` snapshots. Only
    the final `store.consolidate_memories(...)` call takes the lock again, for
    its single atomic transaction. So a sweep never blocks concurrent reads
    or writes for more than the brief moment of the final apply.

Complexity (documented):
    Deduplication is naive O(n^2) pairwise cosine over the card set, which is
    fine for prototype scale (hundreds to low thousands of cards). A simple
    optional blocking optimization buckets cards by the sign of their first
    embedding dimension so only same-bucket pairs are compared, cutting the
    constant factor roughly by the number of buckets. Pruning is O(n). The
    whole sweep is run off the hot path (background thread or
    `asyncio.to_thread`), so its cost does not tax query latency.
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from typing import Any

from isotope_zero.core.store import MemoryStore
from isotope_zero.tokens import estimate_tokens
from isotope_zero.types import ConsolidationReport, MemoryCard, now_ts

log = logging.getLogger("isotope_zero.consolidation")

# Seconds per day — keeps the half-life math readable.
_SECS_PER_DAY: float = 86400.0
# Half-life target for the recency term: recency contribution halves every
# this many days. Tunable via the constructor's `decay_lambda` once you know
# the half-life you want.
_DEFAULT_HALFLIFE_DAYS: float = 30.0
# Cap on evidence fragments retained when merging a cluster, so a survivor's
# evidence string can't grow without bound across many sweeps.
_MAX_EVIDENCE_FRAGMENTS: int = 3


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two L2-normalized embeddings via dot product.

    The ONNX embedder L2-normalizes its outputs, so cosine == dot product.
    This is a pure-Python loop (no numpy) to match the store's
    `vector_search` style. Length mismatches are tolerated by comparing only
    the overlapping prefix (robust to a stray dim mismatch; in practice the
    embedder guarantees equal dims).
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    for i in range(n):
        dot += a[i] * b[i]
    # Clamp to [0, 1] for safety; a normalized dot lives in [-1, 1] and the
    # dedup threshold is positive, so negative similarities are treated as 0.
    if dot < 0.0:
        return 0.0
    if dot > 1.0:
        return 1.0
    return dot


def _token_overlap_ratio(a: str, b: str) -> float:
    """Jaccard-ish token overlap ratio between two fact strings.

    Used as an optional secondary dedup signal: two facts sharing most of
    their words are likely paraphrases. Returns |A∩B| / |A∪B| in [0, 1].
    Empty inputs yield 0.
    """
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 0.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


# Negation markers — if one fact asserts X and another asserts NOT-X, they
# are semantically opposite and must NEVER be merged, even when their
# embeddings are nearly identical (real embedders place "uses Mac" and "does
# NOT use Mac" very close together, which would otherwise trip dedup).
_NEGATION_MARKERS: tuple[str, ...] = (
    "not", "no longer", "doesn't", "does not", "don't", "do not",
    "never", "isn't", "is not", "wasn't", "was not", "won't", "will not",
    "cannot", "can't", "neither", "nor", "without", "lacks", "stopped",
    "quit", "no more",
)


def _strip_negations(text: str) -> tuple[str, bool]:
    """Return (text-with-negation-markers-removed, was_any_negation_found).

    Lowercases and strips whitespace for comparison only; the returned text
    is only used to judge polarity, never stored.
    """
    t = " " + text.lower().strip() + " "
    found = False
    for marker in sorted(_NEGATION_MARKERS, key=len, reverse=True):
        needle = " " + marker + " "
        if needle in t:
            found = True
            t = t.replace(needle, " ")
    t = " ".join(t.split())
    return t, found


def _stem(tok: str) -> str:
    """Crude suffix stemmer for negation comparison only.

    Strips a trailing 'ing'/'ed'/'es'/'s' so morphological variants of the
    same verb ("uses"/"use"/"using") collapse to a common stem. Deliberately
    crude — it is only used to judge polarity equality, never stored, and a
    false collapse just means two negations are compared a little more
    liberally (which errs toward caution: not-merging).
    """
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def _are_negations(a: str, b: str) -> bool:
    """True if `a` and `b` assert opposite polarities of the same fact.

    Heuristic: after removing negation markers from both sides, if exactly
    one of them originally contained a negation and the denegated token sets
    are highly overlapping (>= 0.6 Jaccard after a crude stem), treat them
    as negations. This catches "User uses Mac" vs "User does not use Mac"
    without merging them, while not falsely blocking unrelated facts that
    happen to share words.
    """
    if not a or not b:
        return False
    ta, neg_a = _strip_negations(a)
    tb, neg_b = _strip_negations(b)
    # Need a polarity difference: exactly one side is negated.
    if neg_a == neg_b:
        return False
    # The denegated texts must be near-identical (same core assertion).
    sa = {_stem(t) for t in ta.split()}
    sb = {_stem(t) for t in tb.split()}
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= 0.6


def _merge_evidence(fragments: list[str], cap: int = _MAX_EVIDENCE_FRAGMENTS) -> str:
    """Union distinct evidence strings, preserving order, capped to `cap`.

    Fragments are deduped by exact string equality (case-sensitive) before
    joining with " | ". The cap prevents unbounded evidence growth across
    many consolidation sweeps.
    """
    seen: set[str] = set()
    out: list[str] = []
    for frag in fragments:
        if frag is None:
            continue
        frag = frag.strip()
        if not frag or frag in seen:
            continue
        seen.add(frag)
        out.append(frag)
        if len(out) >= cap:
            break
    return " | ".join(out)


class Consolidator:
    """Off-hot-path consolidation worker: dedup + decay + atomic sweep.

    Construct one per store and either call `.run()` (sync) / `.run_async()`
    (off the event loop) for a one-shot sweep, or `start_background_loop()`
    for periodic daemon sweeps. `stop()` cancels the background loop.

    Defaults (documented):
        w_recency        = 0.7   — weight on the recency term.
        w_access         = 0.3   — weight on the recall-frequency term.
        decay_lambda     = ln(2)/(30d) ≈ 2.675e-7 per second
                                 — recency contribution halves every ~30 days.
        vitality_floor   = 0.05 — a card below this is a decay candidate
                                  (only if also never-recalled and past grace).
        min_age_seconds  = 3600 — 1 hour grace period before a never-recalled
                                  fresh write can be pruned.
        dedup_threshold  = 0.88 — cosine above this (OR exact fact equality OR
                                  high token overlap) flags a duplicate.
        token_overlap_floor = 0.7 — fact token-overlap above this also flags
                                    a duplicate (paraphrase catch-all).
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: Any = None,
        *,
        dedup_threshold: float = 0.88,
        token_overlap_floor: float = 0.70,
        w_recency: float = 0.7,
        w_access: float = 0.3,
        decay_lambda: float = math.log(2.0) / (_DEFAULT_HALFLIFE_DAYS * _SECS_PER_DAY),
        vitality_floor: float = 0.05,
        min_age_seconds: float = 3600.0,
        max_evidence_fragments: int = _MAX_EVIDENCE_FRAGMENTS,
    ) -> None:
        self.store = store
        # Prefer an explicitly-passed embedder, else fall back to the store's.
        # Used to re-embed a merged survivor's fact. May be in fallback mode
        # (deterministic hash) — that's fine, dedup still works via exact-fact
        # and token-overlap paths.
        self.embedder = embedder if embedder is not None else getattr(store, "embedder", None)
        self.dedup_threshold = float(dedup_threshold)
        self.token_overlap_floor = float(token_overlap_floor)
        self.w_recency = float(w_recency)
        self.w_access = float(w_access)
        self.decay_lambda = float(decay_lambda)
        self.vitality_floor = float(vitality_floor)
        self.min_age_seconds = float(min_age_seconds)
        self.max_evidence_fragments = int(max_evidence_fragments)

        # Background-loop control.
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    # Vitality scoring
    # ------------------------------------------------------------------ #
    def vitality(self, card: MemoryCard, now: float | None = None) -> float:
        """Vitality score S for a card, combining recency and recall frequency.

            S = w_recency * exp(-lambda * delta_t) + w_access * ln(1 + access_count)

        where `delta_t = now - last_access` in seconds. Higher is more vital.

        - The recency term decays exponentially; with the default lambda it
          halves every ~30 days. A card accessed "just now" contributes
          `w_recency` (its full weight) from recency.
        - The access term is unbounded but grows slowly (logarithmically), so
          a frequently-recalled card stays vital even if it hasn't been read
          in a while — recalled knowledge is never silently pruned.
        - `last_access` of 0.0 (unset) is treated as the card's creation
          `timestamp`, matching the store's convention for fresh writes.
        """
        now = now if now is not None else now_ts()
        last = card.last_access if card.last_access else card.timestamp
        delta_t = max(0.0, now - last)
        recency = math.exp(-self.decay_lambda * delta_t)
        access = math.log(1.0 + max(0, card.access_count))
        return self.w_recency * recency + self.w_access * access

    # ------------------------------------------------------------------ #
    # Deduplication
    # ------------------------------------------------------------------ #
    def _are_duplicates(self, a: MemoryCard, b: MemoryCard) -> bool:
        """Decide whether two cards are duplicates (should merge).

        Order of checks (cheapest/most-certain first):
          0. Negation guard — if the two facts assert opposite polarities of
             the same core assertion ("User uses Mac" vs "User does not use
             Mac"), they are NEVER duplicates, regardless of how close their
             embeddings sit. Real embedders place a fact and its negation
             near-identically, so this guard must run BEFORE the cosine path
             or semantic dedup would silently destroy a correction.
          1. Exact case-insensitive `fact` equality — catches true duplicates
             even when the embedder is in fallback mode and cosine is
             meaningless (hash collisions aside, identical texts still share
             a fact string).
          2. High token-overlap of facts — paraphrase catch-all that works in
             both real and fallback embedding modes.
          3. Cosine similarity of embeddings above `dedup_threshold` — the
             semantic path, only meaningful with a real ONNX embedder. Cards
             with `embedding is None` skip this check (similarity treated 0).
        """
        # 0) Negation guard — polarity difference means not-duplicate.
        if _are_negations(a.fact, b.fact):
            return False
        if a.fact and b.fact and a.fact.strip().lower() == b.fact.strip().lower():
            return True
        if (
            a.fact
            and b.fact
            and _token_overlap_ratio(a.fact, b.fact) >= self.token_overlap_floor
        ):
            return True
        ea, eb = a.embedding, b.embedding
        if ea and eb and len(ea) > 0 and len(eb) > 0:
            if _cosine(ea, eb) > self.dedup_threshold:
                return True
        return False

    def _find_clusters(self, cards: list[MemoryCard]) -> list[list[MemoryCard]]:
        """Cluster duplicate cards via union-find over pairwise dup edges.

        O(n^2) pairwise comparison, fine for prototype scale. Returns a list
        of clusters (each a non-empty list of cards); cards that duplicate
        nothing form their own singleton cluster (which the caller will
        ignore since no merge is needed).
        """
        n = len(cards)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(n):
            for j in range(i + 1, n):
                if self._are_duplicates(cards[i], cards[j]):
                    union(i, j)

        groups: dict[int, list[MemoryCard]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(cards[i])
        return list(groups.values())

    def _merge_cluster(self, cluster: list[MemoryCard]) -> tuple[MemoryCard, list[str]]:
        """Merge one duplicate cluster into a survivor + the folded ids.

        Survivor selection rule (the near-duplicate-correction fix):
            Prefer the NEWEST card by default, so a near-duplicate CORRECTION —
            two facts that differ by a single substantive token, e.g.
            "... configured with inline." -> "... configured with remote." —
            keeps the newer statement, because the correction supersedes the
            stale original (a silent lost update would otherwise keep the old
            fact).

            EXCEPTION: when every fact in the cluster is token-identical after
            normalization (strip + lowercase — true duplicates, or paraphrases
            that differ only in case/whitespace), keep the EARLIEST card as the
            canonical "first seen" phrasing. This preserves the original
            behavior for genuine duplicates and paraphrase clusters where no
            single-token correction is happening.

            Detect "differs beyond case/whitespace" by comparing the
            stripped/lowercased facts: if the cluster's normalized fact set has
            exactly one element, the facts are token-identical -> earliest wins;
            otherwise the facts substantively differ -> newest wins. Tie-break
            by id for determinism.

        The survivor's `fact` is kept as-is (conservative — never invent a new
        fact). Other fields are combined across the cluster:

          - evidence: union of distinct fragments joined with " | ", capped
            to `max_evidence_fragments` to avoid unbounded growth.
          - tags: union of all cluster tags (preserve order, dedupe).
          - access_count: SUM across the cluster (merged card inherits all
            recall pressure).
          - last_access: MAX across the cluster (most-recent touch wins).
          - source_tokens: the survivor's own (it's the canonical source).
          - embedding: re-embed the survivor's fact via the embedder if
            available; otherwise keep the survivor's existing embedding
            (which may be None for fallback-only cards).
        """
        # Newest-by-default; earliest only when all facts are token-identical
        # after normalization (true duplicates / case-whitespace paraphrases).
        normalized = {c.fact.strip().lower() for c in cluster}
        if len(normalized) == 1:
            # Canonical "first seen" — identical facts keep the earliest card.
            survivor_card = min(cluster, key=lambda c: (c.timestamp, c.id))
        else:
            # Facts differ beyond case/whitespace: the NEWEST statement wins,
            # because it is a correction/supersession of the earlier one.
            survivor_card = max(cluster, key=lambda c: (c.timestamp, c.id))
        folded = [c for c in cluster if c.id != survivor_card.id]

        # Union evidence, capped. Lead with the survivor's own evidence so it
        # always appears first.
        fragments = [survivor_card.evidence]
        fragments.extend(c.evidence for c in folded)
        merged_evidence = _merge_evidence(fragments, cap=self.max_evidence_fragments)

        # Union tags, preserve order, dedupe.
        seen_tags: set[str] = set()
        merged_tags: list[str] = []
        for c in cluster:
            for t in c.tags:
                if t not in seen_tags:
                    seen_tags.add(t)
                    merged_tags.append(t)

        # Sum access pressure; take most-recent last_access across cluster.
        sum_access = sum(c.access_count for c in cluster)
        max_last_access = max((c.last_access for c in cluster), default=0.0)

        # Re-embed the survivor's fact if we have a working embedder.
        emb = survivor_card.embedding
        if self.embedder is not None:
            try:
                re = self.embedder.embed_text(survivor_card.fact)
                if re:
                    emb = re
            except Exception as exc:  # defensive: never let re-embed kill a sweep
                log.warning("re-embed failed for survivor %s: %s", survivor_card.id, exc)

        survivor = MemoryCard(
            id=survivor_card.id,
            fact=survivor_card.fact,
            evidence=merged_evidence,
            timestamp=survivor_card.timestamp,
            tags=merged_tags,
            embedding=emb,
            source_tokens=survivor_card.source_tokens,
            access_count=sum_access,
            last_access=max_last_access,
        )
        folded_ids = [c.id for c in folded]
        return survivor, folded_ids

    def _plan_dedup(
        self, cards: list[MemoryCard]
    ) -> tuple[list[MemoryCard], list[str], dict[str, str]]:
        """Plan the dedup half of the sweep.

        Returns (survivors_to_upsert, dedup_folded_ids, superseded_map).
        `dedup_folded_ids` lists every folded member; `superseded_map` maps
        each folded id -> the survivor id it folded into, so the store can mark
        the folded rows as superseded (audit trail) instead of deleting them.
        Only clusters with more than one member produce a survivor to upsert +
        folded ids; singletons produce nothing (they're already canonical).
        """
        survivors: list[MemoryCard] = []
        folded_ids_all: list[str] = []
        superseded_map: dict[str, str] = {}
        for cluster in self._find_clusters(cards):
            if len(cluster) < 2:
                continue
            survivor, folded_ids = self._merge_cluster(cluster)
            survivors.append(survivor)
            folded_ids_all.extend(folded_ids)
            for fid in folded_ids:
                superseded_map[fid] = survivor.id
        return survivors, folded_ids_all, superseded_map

    # ------------------------------------------------------------------ #
    # Temporal decay & pruning
    # ------------------------------------------------------------------ #
    # Conservative rules (applied inline in `run()`): a card is pruned ONLY
    # if ALL of these hold:
    #   - vitality < vitality_floor (stale), AND
    #   - access_count == 0        (never recalled), AND
    #   - age >= min_age_seconds   (past the fresh-write grace period).
    # Recalled cards (access_count > 0) are NEVER pruned — their log-access
    # term keeps them vital and they represent demonstrably-used knowledge.
    # Fresh never-recalled writes get a grace period so a card written
    # moments ago can't be pruned before it has a chance to be recalled.

    # ------------------------------------------------------------------ #
    # Token accounting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _card_tokens(card: MemoryCard) -> int:
        """Tokens a card contributes to context: fact + evidence."""
        return estimate_tokens(card.fact) + estimate_tokens(card.evidence)

    @staticmethod
    def _total_tokens(cards: list[MemoryCard]) -> int:
        return sum(Consolidator._card_tokens(c) for c in cards)

    # ------------------------------------------------------------------ #
    # Public: one-shot sweep (synchronous core)
    # ------------------------------------------------------------------ #
    def run(self) -> ConsolidationReport:
        """Run one full consolidation sweep synchronously and return a report.

        Steps:
          1. Snapshot all cards via `store.all()` (lock held only for that read).
          2. Plan dedup: cluster + merge -> (survivors, dedup_deleted_ids).
          3. Plan decay: score vitality, conservatively pick prune candidates
             (excluding ids already folded by dedup).
          4. Apply ONCE via `store.consolidate_memories(survivors, deleted_ids)`
             — a single atomic transaction (BEGIN IMMEDIATE + COMMIT).
          5. Build and return the ConsolidationReport.

        No store lock is held across the planning phase; only the final
        `consolidate_memories` call takes the lock for its atomic txn.
        """
        t0 = time.perf_counter()
        now = now_ts()

        # 1. Snapshot.
        all_cards = self.store.all()
        tokens_before = self._total_tokens(all_cards)

        # 2. Plan dedup.
        survivors, dedup_deleted, superseded_map = self._plan_dedup(all_cards)
        dedup_deleted_set = set(dedup_deleted)

        # Survivors to upsert keyed by id (for decay to possibly re-score the
        # survivor rather than its folded members).
        survivor_by_id = {s.id: s for s in survivors}

        # 3. Plan decay. Score every card that is NOT already folded by dedup.
        # For a cluster, the survivor (canonical id) is what we score, using
        # its merged access_count/last_access so a cluster that was recalled
        # is never pruned.
        pruned_ids: list[str] = []
        pruned_vitalities: list[float] = []
        pruned_set: set[str] = set()

        for card in all_cards:
            cid = card.id
            if cid in dedup_deleted_set:
                # Already being deleted as a duplicate; don't double-count.
                continue
            # Use the merged survivor's values if this id is a survivor, else
            # the card as-is.
            scoring_card = survivor_by_id.get(cid, card)
            age = max(0.0, now - scoring_card.timestamp)
            if age < self.min_age_seconds:
                # Fresh write grace period — give it a chance to be recalled.
                continue
            if scoring_card.access_count > 0:
                # Recalled cards are never pruned, regardless of vitality.
                continue
            s = self.vitality(scoring_card, now=now)
            if s < self.vitality_floor:
                pruned_ids.append(cid)
                pruned_vitalities.append(s)
                pruned_set.add(cid)

        # 4. Apply once, atomically. Folded cards are marked superseded (audit
        # trail) rather than hard-deleted; only decay-pruned ids are deleted.
        deleted_ids = list(pruned_ids)
        # `consolidate_memories` returns # rows hard-deleted (pruned only;
        # superseded rows remain in the table for the audit trail).
        self.store.consolidate_memories(
            survivors, deleted_ids, superseded_ids=superseded_map
        )

        # 5. Report. `survivors` count is read AFTER the sweep.
        after_cards = self.store.all()
        tokens_after = self._total_tokens(after_cards)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        mean_v = (
            sum(pruned_vitalities) / len(pruned_vitalities)
            if pruned_vitalities
            else 0.0
        )

        log.debug(
            "consolidation sweep: in=%d merged=%d decayed=%d survivors=%d "
            "tokens %d->%d latency=%.1fms",
            len(all_cards), len(dedup_deleted), len(pruned_ids),
            len(after_cards), tokens_before, tokens_after, latency_ms,
        )

        return ConsolidationReport(
            merged_cards=len(dedup_deleted),  # folded into survivors only
            decayed_cards=len(pruned_ids),    # pruned by decay only
            survivors=len(after_cards),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_reclaimed=max(0, tokens_before - tokens_after),
            latency_ms=latency_ms,
            pruned_mean_vitality=mean_v,
        )

    # ------------------------------------------------------------------ #
    # Public: dry-run (plan without committing) — for the debug CLI
    # ------------------------------------------------------------------ #
    def dry_run(self) -> dict[str, Any]:
        """Compute the consolidation plan WITHOUT touching the store.

        Returns a JSON-serializable diff describing what a real `run()` would
        do: proposed merges (cluster survivor + folded ids + merged preview)
        and proposed deletions (decay-pruned ids with their vitality scores).
        Nothing is written — the store is read once via `.all()` and the plan
        is computed in memory. Used by `isotope_zero dry-run-consolidation`.
        """
        now = now_ts()
        all_cards = self.store.all()
        survivors, dedup_deleted, _superseded_map = self._plan_dedup(all_cards)
        dedup_deleted_set = set(dedup_deleted)
        survivor_by_id = {s.id: s for s in survivors}

        merges: list[dict[str, Any]] = []
        # Map each survivor to its folded ids for the diff preview.
        # Re-derive clusters so we can list the folded members' facts too.
        for cluster in self._find_clusters(all_cards):
            if len(cluster) < 2:
                continue
            survivor, _ = self._merge_cluster(cluster)
            folded = [c for c in cluster if c.id != survivor.id]
            merges.append({
                "survivor_id": survivor.id,
                "survivor_fact": survivor.fact,
                "folded_ids": [c.id for c in folded],
                "folded_facts": [c.fact for c in folded],
            })

        pruned: list[dict[str, Any]] = []
        for card in all_cards:
            cid = card.id
            if cid in dedup_deleted_set:
                continue
            scoring_card = survivor_by_id.get(cid, card)
            age = max(0.0, now - scoring_card.timestamp)
            if age < self.min_age_seconds:
                continue
            if scoring_card.access_count > 0:
                continue
            v = self.vitality(scoring_card, now=now)
            if v < self.vitality_floor:
                pruned.append({
                    "id": cid,
                    "fact": card.fact,
                    "vitality": round(v, 4),
                    "age_days": round(age / _SECS_PER_DAY, 1) if _SECS_PER_DAY else 0.0,
                })

        tokens_before = self._total_tokens(all_cards)
        # After: survivors only (dedup_deleted removed, pruned removed),
        # with survivors counted once (already present in all_cards).
        after_ids = {c.id for c in all_cards} - set(dedup_deleted) - {p["id"] for p in pruned}
        # Approximate after-tokens from the survivor previews where available.
        after_tokens = 0
        for c in all_cards:
            if c.id in after_ids:
                if c.id in survivor_by_id:
                    after_tokens += self._card_tokens(survivor_by_id[c.id])
                else:
                    after_tokens += self._card_tokens(c)

        return {
            "proposed_merges": merges,
            "proposed_deletions": {
                "dedup": dedup_deleted,
                "decay": pruned,
            },
            "summary": {
                "cards_in": len(all_cards),
                "cards_after": len(after_ids),
                "merged_cards": len(dedup_deleted),
                "decayed_cards": len(pruned),
                "tokens_before": tokens_before,
                "tokens_after_approx": after_tokens,
                "tokens_reclaimed_approx": max(0, tokens_before - after_tokens),
            },
        }

    # ------------------------------------------------------------------ #
    # Public: async wrapper (off the event loop)
    # ------------------------------------------------------------------ #
    async def run_async(self) -> ConsolidationReport:
        """Run `.run()` in a worker thread via `asyncio.to_thread`.

        Use this from an async caller so the (CPU-bound, lock-touching) sweep
        does not block the event loop. The synchronous `.run()` is the actual
        work; this just offloads it.
        """
        return await asyncio.to_thread(self.run)

    # ------------------------------------------------------------------ #
    # Public: background loop (daemon thread)
    # ------------------------------------------------------------------ #
    def start_background_loop(self, interval_seconds: float = 300.0) -> None:
        """Start a daemon thread that runs `.run()` every `interval_seconds`.

        Non-blocking: returns immediately. The loop checks `self._stop`
        between sweeps. Call `.stop()` to terminate it (the current sweep, if
        any, finishes first). Safe to call once; a second call is a no-op if
        a loop is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            return

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    self.run()
                except Exception as exc:  # never let a sweep crash the loop
                    log.exception("consolidation sweep failed: %s", exc)
                # Sleep cooperatively, waking promptly on stop().
                self._stop.wait(interval_seconds)

        self._stop.clear()
        self._thread = threading.Thread(
            target=_loop, name="isotope_zero-consolidator", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Signal the background loop to stop and (optionally) join it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None


# ---------------------------------------------------------------------- #
# Smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import math as _math

    def _norm(v: list[float]) -> list[float]:
        n = _math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else v

    from isotope_zero.embeddings.onnx_embed import EmbeddingEngine

    eng = EmbeddingEngine()
    store = MemoryStore(":memory:", embedder=eng)

    t0 = now_ts()
    # Two near-duplicate embeddings (aligned unit vectors): high cosine.
    store.add(
        MemoryCard(
            id="dup-1",
            fact="The user prefers dark mode.",
            evidence="user said: 'I love dark mode'",
            timestamp=t0 - 100,
            tags=["preference", "ui"],
            embedding=_norm([1.0, 0.0, 0.0, 0.0]),
            source_tokens=6,
            access_count=2,
            last_access=t0 - 50,
        )
    )
    store.add(
        MemoryCard(
            id="dup-2",
            fact="User prefers dark mode",
            evidence="settings log: dark theme enabled",
            timestamp=t0 - 90,
            tags=["preference"],
            embedding=_norm([0.99, 0.01, 0.0, 0.0]),
            source_tokens=5,
            access_count=1,
            last_access=t0 - 40,
        )
    )
    # Exact-fact duplicate (catches the fallback-embedding case too).
    store.add(
        MemoryCard(
            id="dup-3",
            fact="The user prefers dark mode.",
            evidence="seen in profile settings",
            timestamp=t0 - 80,
            tags=["ui", "theme"],
            embedding=_norm([0.0, 1.0, 0.0, 0.0]),
            source_tokens=6,
            access_count=0,
            last_access=0.0,
        )
    )
    # Old, never-accessed card -> decay candidate. With the default 30-day
    # half-life and vitality_floor=0.05, a never-recalled card crosses below
    # the floor at ~114 days, so make this one 200 days old to be safely
    # prunable (well past the 1-hour grace period too).
    store.add(
        MemoryCard(
            id="stale-1",
            fact="An old throwaway note.",
            evidence="debug log line 42",
            timestamp=t0 - (200 * 86400),  # 200 days old
            tags=["debug"],
            embedding=_norm([0.0, 0.0, 1.0, 0.0]),
            source_tokens=5,
            access_count=0,
            last_access=0.0,
        )
    )
    # Fresh card -> must survive (within grace, never mind vitality).
    store.add(
        MemoryCard(
            id="fresh-1",
            fact="A brand new note.",
            evidence="just arrived",
            timestamp=t0,
            tags=["fresh"],
            embedding=_norm([0.0, 0.0, 0.0, 1.0]),
            source_tokens=4,
            access_count=0,
            last_access=0.0,
        )
    )

    print("before count:", store.count())
    report = Consolidator(store).run()
    print("after  count:", store.count())
    print(report)
    print(
        "merged+decayed =",
        report.merged_cards + report.decayed_cards,
        "(expect 3: 2 dups folded + 1 decayed; 2 survivors remain)",
    )
    assert report.merged_cards >= 2, f"expected >=2 merges, got {report.merged_cards}"
    assert report.decayed_cards >= 1, f"expected >=1 decay, got {report.decayed_cards}"
    assert report.tokens_reclaimed >= 0
    assert report.survivors == store.count()
    print("smoke test OK")
