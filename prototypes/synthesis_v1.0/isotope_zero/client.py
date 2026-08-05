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
import logging
import uuid
from typing import Any

from isotope_zero.core.store import MemoryStore
from isotope_zero.embeddings.engine import HybridEmbeddingEngine
from isotope_zero.types import MemoryCard, now_ts

log = logging.getLogger("isotope_zero.client")


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
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        model_name: str = "all-MiniLM-L6-v2",
        socket_path: str = "/tmp/izero.sock",
        spawn_daemon: bool = True,
        use_mmap: bool = True,
        alpha: float = 0.70,
    ) -> None:
        self.alpha = float(alpha)
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
        atexit.register(self.close)
        log.debug(
            "IsotopeZero ready db=%s model=%s spawn_daemon=%s mmap=%s alpha=%.2f "
            "engine_mode=%s is_real=%s dim=%d",
            db_path,
            model_name,
            spawn_daemon,
            use_mmap,
            self.alpha,
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
    ) -> str:
        """Embed and persist one fact. Returns the new card's id.

        ``fact`` is the compressed memory unit; ``evidence`` is the smallest
        quote that justifies it (never the full raw input). ``tags`` are
        free-form labels used by SQL tag lookups and graph auto-linking.
        ``importance`` is a user-set [0.0, 1.0] weight that boosts the card's
        Ebbinghaus stability on recall (see ``core.decay``).

        The embedding is produced by the engine BEFORE the card enters the
        store, so the store's vector index and graph auto-linking see a
        fully-formed card. The id is a fresh uuid4 hex.
        """
        card_id = uuid.uuid4().hex
        embedding = self.engine.embed_text(fact)
        card = MemoryCard(
            id=card_id,
            fact=fact,
            evidence=evidence,
            timestamp=now_ts(),
            tags=list(tags) if tags else [],
            embedding=embedding,
            importance=float(importance),
        )
        self.store.add(card)
        log.debug("remember id=%s fact=%r tags=%s", card_id, fact[:80], card.tags)
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
        hits = self.store.vector_search(qvec, k=k, alpha=alpha if alpha is not None else self.alpha)
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
    def touch(self, card_id: str) -> bool:
        """Record a recall on ``card_id``. Returns True iff the card existed.

        Bumps the card's access_count, updates last_access, and recomputes its
        Ebbinghaus stability S via the store's spaced-repetition dynamics. A
        touch on a missing id is a no-op and returns False (the store's own
        ``touch`` is silent on missing ids, so we probe with ``get`` first to
        give a meaningful boolean).
        """
        if self.store.get(card_id) is None:
            return False
        self.store.touch(card_id)
        return True

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
        from isotope_zero.core.consolidation import Consolidator

        report = Consolidator(self.store, embedder=self.engine).run()
        return {
            "merged": report.merged_cards,
            "pruned": report.decayed_cards,
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
