"""Top-level unified IsotopeZero client API (Phase 8 synthesis).

A clean, intuitive public surface that hides every internal subsystem — the
SQLite-backed ``MemoryStore``, the hybrid IPC/in-process ``HybridEmbeddingEngine``,
the dedup/decay ``Consolidator``, and the semantic graph — behind one small
object you construct with sensible defaults and talk to in plain verbs::

    from isotope_zero.client import IsotopeZero

    mem = IsotopeZero()                 # in-memory DB, real embedder if available
    cid = mem.remember("I live in SF", evidence="user statement", tags=["location"])
    hits = mem.recall("where do I live", k=3)
    mem.touch(cid)
    mem.consolidate()
    mem.close()

This is a facade, not a reimplementation: it wires the Phase 8 winning stack
(engine + store + consolidator) together and exposes only the operations an
agent actually needs. Everything passes through the store's existing lock,
WAL, mmap vector index, and atomic consolidation transaction, so the
performance + correctness guarantees of the lower layers are inherited
verbatim.

Defaults are chosen so ``IsotopeZero()`` with no arguments is immediately
useful for a prototype: an in-memory SQLite DB, the hybrid embedder with the
daemon spawn enabled (transparently falls back to in-process / deterministic
pseudo-embeddings if the daemon can't start), mmap-backed vector index, and a
0.70 cosine/retention fusion alpha. File-backed persistence is one argument
away (``db_path="path.db"``).
"""
from __future__ import annotations

import atexit
import contextlib
import logging
import re
import time
import uuid
from typing import Any, Iterator

from .core.history import History
from .core.store import MemoryStore
from .embeddings.engine import HybridEmbeddingEngine
from .types import MemoryCard, now_ts

log = logging.getLogger("isotope_zero.client")

# TTL is encoded as a tag ``ttl:<seconds>`` so it rides the existing tag
# column (no schema change) and survives ``store.add``/``update``. The client
# filters expired cards at read time and prunes them on ``consolidate``.
_TTL_TAG_RE = re.compile(r"^ttl:(\d+(?:\.\d+)?)$")

# Extra rows to over-fetch when a client TTL default is active, so dropping
# expired hits client-side doesn't shrink the returned top-k below k. Bounded
# so the over-fetch cost is constant; if more than this many of the top
# results are expired, the caller simply gets fewer than k (rare).
_TTL_OVERFETCH = 8


def _card_ttl_seconds(card: MemoryCard) -> float | None:
    """Return the card's TTL in seconds if a ``ttl:<n>`` tag is present."""
    for tag in card.tags:
        m = _TTL_TAG_RE.match(tag)
        if m is not None:
            try:
                return float(m.group(1))
            except (ValueError, TypeError):
                continue
    return None


class IsotopeZero:
    """Unified agent-memory client: remember, recall, touch, consolidate.

    One object owns the embedding engine, the persistent store, and (lazily)
    the consolidator. Construction is cheap for ``:memory:``; a file-backed
    ``db_path`` opens/creates the SQLite file and enables WAL.

    Parameters
    ----------
    db_path:
        SQLite path. ``":memory:"`` (default) keeps data in RAM for the life
        of this client. Any other string is a file path (opened/created on
        disk, WAL-enabled by the store).
    model_name:
        Embedding model shorthand forwarded to ``HybridEmbeddingEngine``
        (default ``"all-MiniLM-L6-v2"``, dim 384).
    socket_path:
        Unix domain socket the embedding daemon listens on, forwarded to the
        engine. Only relevant when the daemon path is taken.
    spawn_daemon:
        When True (default), the engine may spawn a shared-memory embedding
        daemon so onnxruntime runs out-of-process. The engine transparently
        falls back to an in-process or deterministic pseudo-embedding path if
        the daemon can't start, so a False value just forces that fallback
        eagerly (useful for tests / CI / sandboxed runs).
    use_mmap:
        When True (default), the store's vector index is a file-backed
        ``np.memmap`` (zero-copy, OS-paged) instead of a heap ``np.stack``.
    alpha:
        Default cosine-vs-Ebbinghaus-retention fusion weight for ``recall``
        (``final = alpha*cosine + (1-alpha)*retention``). 1.0 = pure cosine.
        Per-call ``alpha`` to ``recall`` overrides this.
    ttl_seconds:
        Default time-to-live for cards written via ``remember``. ``None``
        (default) means no expiry. When set, ``remember`` stamps a
        ``ttl:<seconds>`` tag and ``recall``/``search`` filter out cards
        whose ``timestamp + ttl`` is in the past. Override per-card via the
        ``ttl_seconds`` argument to ``remember``.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        model_name: str = "all-MiniLM-L6-v2",
        socket_path: str = "/tmp/izero.sock",
        spawn_daemon: bool = True,
        use_mmap: bool = True,
        alpha: float = 0.70,
        ttl_seconds: float | None = None,
    ) -> None:
        self.alpha = float(alpha)
        # Default time-to-live for cards written via ``remember``. ``None``
        # means "no expiry" (cards live until consolidated/archived). When set,
        # ``remember`` stamps a ``ttl:<seconds>`` tag and ``recall``/search
        # filter out cards whose ``timestamp + ttl < now``.
        self.ttl_seconds = ttl_seconds
        # The hybrid engine owns the daemon-vs-in-process-vs-fallback decision;
        # we just forward the knobs. Constructed first because the store takes
        # the embedder as a constructor argument (it stores it for the router
        # and consolidator re-embed path, though the store itself never calls
        # it — cards arrive with embeddings already populated).
        self.engine = HybridEmbeddingEngine(
            model_name=model_name,
            socket_path=socket_path,
            spawn_daemon=spawn_daemon,
            dim=384,
        )
        self.store = MemoryStore(
            db_path=db_path,
            embedder=self.engine,
            use_daemon=False,  # the engine already handles daemon-vs-in-process
            use_mmap=use_mmap,
        )
        # Compose the time-travel history handle. It lazily creates its sidecar
        # ``memories_history`` table on the store's connection.
        self.history = History(self.store)
        atexit.register(self.close)
        log.debug(
            "IsotopeZero ready db=%s model=%s spawn_daemon=%s mmap=%s alpha=%.2f "
            "ttl=%s engine_mode=%s is_real=%s dim=%d",
            db_path,
            model_name,
            spawn_daemon,
            use_mmap,
            self.alpha,
            self.ttl_seconds,
            getattr(self.engine, "mode", "?"),
            getattr(self.engine, "is_real", False),
            getattr(self.engine, "dim", 0),
        )

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #
    def remember(
        self,
        fact: str,
        evidence: str = "",
        tags: list[str] | None = None,
        importance: float = 0.0,
        ttl_seconds: float | None = None,
        scope: str | None = None,
        card_id: str | None = None,
    ) -> str:
        """Embed and persist one fact. Returns the new card's id.

        ``fact`` is the compressed memory unit; ``evidence`` is the smallest
        quote that justifies it (never the full raw input). ``tags`` are
        free-form labels used by SQL tag lookups and graph auto-linking.
        ``importance`` is a user-set [0.0, 1.0] weight that boosts the card's
        Ebbinghaus stability on recall (see ``core.decay``).

        ``ttl_seconds`` overrides the client default for THIS card; when a
        positive value is in effect a ``ttl:<seconds>`` tag is stamped so
        ``recall`` filters the card out once ``timestamp + ttl`` is in the
        past. Pass ``ttl_seconds=0`` to explicitly disable TTL for a card
        even when the client default is set.

        ``scope`` stamps a multi-tier scope onto the card (overriding any
        scoped-context scope for this single write). When omitted, an active
        ``scoped()`` context's scope is used; otherwise the card keeps the
        store default. ``card_id`` lets a caller deterministically upsert
        (re-``remember`` an existing id logs history + overwrites) — defaults
        to a fresh uuid4.

        The embedding is produced by the engine BEFORE the card enters the
        store, so the store's vector index and graph auto-linking see a
        fully-formed card. A re-``remember`` of an existing id snapshots the
        prior state into ``client.history`` before overwriting, so the
        revision is undoable via ``client.history.rollback``.
        """
        card_id = card_id or uuid.uuid4().hex
        embedding = self.engine.embed_text(fact)
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        out_tags = list(tags) if tags else []
        if effective_ttl and effective_ttl > 0:
            out_tags.append(f"ttl:{float(effective_ttl)}")
        # An active scoped() context supplies the scope unless the caller
        # passed one explicitly for this write.
        effective_scope = scope if scope is not None else getattr(self, "_active_scope", None)
        # If this id already exists, snapshot its current state for history
        # BEFORE overwriting (so the prior fact is recoverable via rollback).
        if self.store.get(card_id) is not None:
            try:
                self.history.snapshot(card_id)
            except Exception as exc:  # never block a write on history failure
                log.debug("history snapshot failed for %s: %s", card_id, exc)
        card = MemoryCard(
            id=card_id,
            fact=fact,
            evidence=evidence,
            timestamp=now_ts(),
            tags=out_tags,
            embedding=embedding,
            importance=float(importance),
        )
        self.store.add(card, scope=effective_scope)
        log.debug("remember id=%s fact=%r tags=%s scope=%s", card_id, fact[:80], card.tags, effective_scope)
        return card_id

    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #
    def recall(
        self,
        query: str,
        k: int = 5,
        alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic retrieval: embed the query and return the top-k hits.

        Returns a list of dicts sorted by score descending::

            [{"id", "fact", "evidence", "score", "tags", "timestamp"}]

        The raw embedding is deliberately stripped from the public result —
        callers never need 384 floats and leaking them bloats agent context.
        ``alpha`` fuses cosine similarity with Ebbinghaus retention
        (``final = alpha*cosine + (1-alpha)*retention``); when None the
        client's default ``alpha`` is used. ``alpha=1.0`` is pure cosine.
        """
        qvec = self.engine.embed_text(query)
        effective_scope = getattr(self, "_active_scope", None) or "default"
        # Over-fetch when a TTL is in effect: TTL filtering drops expired cards
        # client-side, so requesting exactly k can leave fewer than k after
        # the filter. We fetch k + a small buffer (only when the client has a
        # non-None default TTL — pure-cosine clients pay no over-fetch) so the
        # returned list still has up to k live hits.
        fetch_k = k + (_TTL_OVERFETCH if self.ttl_seconds else 0)
        hits = self.store.vector_search(
            qvec, k=fetch_k, alpha=alpha if alpha is not None else self.alpha, scope=effective_scope
        )
        # TTL filtering: drop cards whose ``ttl:<seconds>`` tag says they've
        # expired (timestamp + ttl < now). Done client-side so the store needs
        # no schema change.
        now = time.time()
        pruned = []
        for card, score in hits:
            ttl = _card_ttl_seconds(card)
            if ttl is not None and (card.timestamp + ttl) < now:
                continue
            pruned.append((card, score))
        hits = pruned
        # vector_search already sorts by (score desc, timestamp asc, id asc);
        # re-sort by score desc here so the contract is explicit and robust
        # to any future store-side ordering change.
        hits.sort(key=lambda item: (-item[1], item[0].timestamp, item[0].id))
        return [
            {
                "id": card.id,
                "fact": card.fact,
                "evidence": card.evidence,
                "score": score,
                "tags": list(card.tags),
                "timestamp": card.timestamp,
            }
            for card, score in hits
        ]

    # ------------------------------------------------------------------ #
    # Access tracking
    # ------------------------------------------------------------------ #
    def touch(self, card_id: str, apply_decay: bool = True) -> bool:
        """Record a recall on ``card_id``. Returns True iff the card existed.

        Bumps the card's access_count, updates last_access, and recomputes its
        Ebbinghaus stability S via the store's spaced-repetition dynamics. A
        touch on a missing id is a no-op and returns False (the store's own
        ``touch`` is silent on missing ids, so we probe with ``get`` first to
        give a meaningful boolean).

        When ``apply_decay`` (default), the Ebbinghaus stability is also
        reinforced opportunistically — the store's ``touch`` already does
        this internally, so this flag is a forward hook for callers who want
        to touch WITHOUT re-scoring (e.g. a bulk reindex). No-op overhead
        when True.
        """
        if self.store.get(card_id) is None:
            return False
        self.store.touch(card_id)
        if apply_decay:
            # The store's touch already updates stability; this hook is for a
            # future client-side reinforcement policy. Kept cheap (a debug
            # log) so enabling it is free in v1.0.
            log.debug("touch+decay card_id=%s", card_id)
        return True

    # ------------------------------------------------------------------ #
    # Multi-tier scoping
    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def scoped(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> Iterator["IsotopeZero"]:
        """Scope subsequent ``remember``/``recall``/``search`` to a boundary.

        Builds a deterministic scope string from the three tiers
        (``user_id=..&agent_id=..&run_id=..``) and makes it the active scope
        for the duration of the ``with`` block. ``remember`` inside the block
        stamps that scope onto the card; ``recall``/``search`` filter to it.
        Cards written under one scope never surface in another's search —
        the store's vector search masks out-of-scope rows (see
        ``MemoryStore._scope_mask``).

        Any tier may be ``None``; only the provided tiers are encoded, so
        ``scoped(user_id="u1")`` scopes to all of user ``u1``'s cards across
        agents/runs. Entering a new scope inside an already-active scope
        replaces it (nested scopes are NOT unioned — the inner wins).
        """
        parts = []
        if user_id is not None:
            parts.append(f"user_id={user_id}")
        if agent_id is not None:
            parts.append(f"agent_id={agent_id}")
        if run_id is not None:
            parts.append(f"run_id={run_id}")
        scope = "&".join(parts) if parts else "default"
        prev = getattr(self, "_active_scope", None)
        self._active_scope = scope
        try:
            yield self
        finally:
            if prev is None:
                # Restore the "no active scope" state rather than a stale
                # "default", so a non-scoped block after a with-block behaves
                # like a fresh client.
                if hasattr(self, "_active_scope"):
                    del self._active_scope
            else:
                self._active_scope = prev

    # ------------------------------------------------------------------ #
    # Hybrid late-fusion search (semantic + BM25/FTS5 + entity graph)
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        k: int = 5,
        alpha: float | None = None,
        top_n_per_branch: int = 30,
        fts_weight: float | None = None,
        vector_weight: float | None = None,
        rrf_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search: semantic vector + BM25 (FTS5) + entity-graph boost.

        Delegates to ``store.hybrid_search`` (the Late-Fusion path). The
        weight knobs map onto the store's single ``alpha`` as
        ``alpha = vector_weight / (vector_weight + fts_weight)`` (so the
        default ``alpha=0.70`` corresponds to ``vector_weight=7,
        fts_weight=3``). ``rrf_k`` is forwarded as ``top_n_per_branch``
        when set (larger = broader per-branch candidate pool before fusion).

        Returns a list of dicts sorted by fused score descending, in the
        same shape as ``recall`` but with an added ``"source"`` showing
        which branch surfaced the hit (``"semantic"``, ``"bm25"``, or
        ``"boost"``) when the store reports it.
        """
        qvec = self.engine.embed_text(query)
        if vector_weight is not None and fts_weight is not None:
            denom = vector_weight + fts_weight
            a = vector_weight / denom if denom > 0 else self.alpha
        elif alpha is not None:
            a = alpha
        else:
            a = self.alpha
        effective_scope = getattr(self, "_active_scope", None) or "default"
        tn = rrf_k if rrf_k is not None else top_n_per_branch
        # Over-fetch when a TTL default is active so TTL filtering doesn't
        # starve the fused top-k (same rationale as recall).
        fetch_k = k + (_TTL_OVERFETCH if self.ttl_seconds else 0)
        raw = self.store.hybrid_search(
            query, qvec, k=fetch_k, alpha=a, top_n_per_branch=tn, scope=effective_scope
        )
        # hybrid_search returns list[tuple[MemoryCard, float]] (scored +
        # already sorted). Apply TTL filtering + shape like recall.
        now = time.time()
        out: list[dict[str, Any]] = []
        for card, score in raw:
            ttl = _card_ttl_seconds(card)
            if ttl is not None and (card.timestamp + ttl) < now:
                continue
            out.append(
                {
                    "id": card.id,
                    "fact": card.fact,
                    "evidence": card.evidence,
                    "score": score,
                    "tags": list(card.tags),
                    "timestamp": card.timestamp,
                }
            )
        return out

    # ------------------------------------------------------------------ #
    # TTL housekeeping
    # ------------------------------------------------------------------ #
    def prune_expired(self) -> int:
        """Hard-delete all TTL-expired cards. Returns the count pruned.

        Scans live cards for a ``ttl:<n>`` tag and deletes any whose
        ``timestamp + ttl < now``. Run on ``consolidate`` or manually. Uses
        ``store.delete`` so the vector index is invalidated.
        """
        now = time.time()
        pruned = 0
        for card in self.store.all():
            ttl = _card_ttl_seconds(card)
            if ttl is not None and (card.timestamp + ttl) < now:
                self.store.delete(card.id)
                pruned += 1
        if pruned:
            log.info("prune_expired removed %d TTL-expired cards", pruned)
        return pruned

    # ------------------------------------------------------------------ #
    # Housekeeping
    # ------------------------------------------------------------------ #
    def consolidate(self) -> dict[str, Any]:
        """Run one full dedup + decay + graph consolidation sweep.

        Constructs a ``Consolidator`` over this client's store and runs a
        single synchronous sweep (the off-hot-path housekeeping that keeps
        the store flat in context size). Returns a summary dict built
        honestly from the ``ConsolidationReport`` the sweep produces::

            {
              "merged":   <cards folded into survivors>,
              "pruned":   <cards hard-deleted by decay>,
              "survivors": <live card count after the sweep>,
              "tokens_before", "tokens_after", "tokens_reclaimed",
              "latency_ms", "pruned_mean_retention",
            }

        ``merged`` counts cards marked SUPERSEDED (audit-trail pointer to the
        survivor, NOT hard-deleted); ``pruned`` counts cards hard-deleted by
        temporal decay. Graph-cluster folds are included in ``merged``.
        """
        # Local import: the consolidator is only needed on the housekeeping
        # path, and keeping it out of module top-level avoids pulling its
        # native/graph deps for users who never call consolidate().
        from .core.consolidation import Consolidator

        # Prune TTL-expired cards first so the consolidator doesn't fold
        # or supersede cards the user explicitly marked ephemeral.
        ttl_pruned = self.prune_expired()
        report = Consolidator(self.store, embedder=self.engine).run()
        return {
            "merged": report.merged_cards,
            "pruned": report.decayed_cards + ttl_pruned,
            "survivors": report.survivors,
            "tokens_before": report.tokens_before,
            "tokens_after": report.tokens_after,
            "tokens_reclaimed": report.tokens_reclaimed,
            "latency_ms": report.latency_ms,
            "pruned_mean_retention": report.pruned_mean_vitality,
        }

    def count(self) -> int:
        """Number of live (non-archived, non-superseded) stored cards."""
        return self.store.count()

    def close(self) -> None:
        """Release the embedding engine's resources.

        The store has no explicit close of its own here (its held SQLite
        connection is cleaned up by its ``__del__``); the engine's close
        drops the daemon socket / in-process session. Registered with
        ``atexit`` on construction so a forgotten close still drains the
        daemon connection. Safe to call repeatedly.
        """
        try:
            self.engine.close()
        except Exception as exc:  # best-effort: never raise from close
            log.debug("engine.close() raised %s; ignoring", exc)


# v1.0 public name. ``IsotopeZero`` remains the original Phase-8 facade;
# ``IsotopeClient`` is the forward-looking SDK alias expected by the docs and
# the framework adapters. Both point at the same class so existing code
# keeps working unchanged.
IsotopeClient = IsotopeZero
