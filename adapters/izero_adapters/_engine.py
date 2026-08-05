"""Shared engine plumbing for Isotope Zero framework adapters.

Every framework adapter (langchain / llamaindex / autogen / crewai) talks to the
Isotope Zero storage layer through this module — it is the single integration
seam. The adapter submodules never touch SQLite, the schema, or the embedding
model directly; they call :class:`Engine` methods that return plain
framework-agnostic results.

Design
------
The Isotope Zero engine lives in ``prototypes/daemon_v0.7/isotope_zero`` and is
NOT pip-installed in this repo. Rather than reinvent storage (which would
diverge from the real schema and break vector compatibility), this module
imports the engine by path with a graceful fallback. The public surface is the
real ``MemoryStore`` + an embedder chosen with this priority:

1. ``embedder=`` passed explicitly by the caller (highest priority — lets tests
   inject deterministic stubs).
2. ``use_daemon=True`` → ``DaemonClient`` (onnxruntime in a separate process,
   the production path). Only chosen if a daemon is actually reachable.
3. A local ONNX ``EmbeddingEngine`` if ``onnxruntime`` + ``tokenizers`` import.
4. A deterministic, L2-normalized feature-hash stub (always available) so the
   adapters remain fully runnable and testable with zero third-party deps.

Vectors from every embedder path are L2-normalized, so cosine similarity ==
dot product, matching ``MemoryStore.vector_search``'s contract.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import time
import uuid
from typing import Any, Sequence

# --------------------------------------------------------------------------- #
# Engine discovery: locate prototypes/synthesis_v1.0/isotope_zero by path.
# --------------------------------------------------------------------------- #
# The repo layout is:  <repo>/adapters/  (this package)
#                     <repo>/prototypes/synthesis_v1.0/isotope_zero/  (engine)
# synthesis_v1.0 is the canonical, feature-complete prototype: multi-tier
# scoping, late-fusion hybrid search, TTL, content dedup, and change history
# all live here. The earlier daemon_v0.7 baseline predates every one of those
# surfaces (its MemoryCard has no `scope`, its store has no `hybrid_search`),
# so adapters that expose scope/hybrid MUST resolve synthesis_v1.0. Resolve the
# engine root relative to this file so the adapter works regardless of the
# caller's CWD. ``IZERO_ENGINE_PATH`` env var overrides for flexibility.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_DEFAULT_ENGINE_PATH = os.path.join(_REPO_ROOT, "prototypes", "synthesis_v1.0")
_ENGINE_PATH = os.environ.get("IZERO_ENGINE_PATH", _DEFAULT_ENGINE_PATH)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIM = 384  # all-MiniLM-L6-v2 output dimension

# Module-level cached imports (populated lazily by _import_engine).
_MemoryStore: Any = None
_MemoryCard: Any = None
_DaemonClient: Any = None
_engine_import_attempted = False
_engine_import_error: Exception | None = None


class EngineError(RuntimeError):
    """Raised when the Isotope Zero engine cannot be located or imported."""


def _import_engine() -> None:
    """Import MemoryStore / MemoryCard / DaemonClient from the engine by path.

    Idempotent; caches the classes at module level. Raises :class:`EngineError`
    with an actionable message if the engine cannot be imported — adapters
    surface this as a clean error rather than an ImportError traceback.
    """
    global _MemoryStore, _MemoryCard, _DaemonClient
    global _engine_import_attempted, _engine_import_error
    if _engine_import_attempted:
        if _engine_import_error is not None:
            raise EngineError(
                f"Isotope Zero engine not importable from {_ENGINE_PATH!r}: "
                f"{_engine_import_error}\n"
                "Set IZERO_ENGINE_PATH to the directory containing the "
                "`isotope_zero` package (e.g. prototypes/daemon_v0.7)."
            )
        return
    _engine_import_attempted = True
    try:
        if _ENGINE_PATH not in sys.path:
            sys.path.insert(0, _ENGINE_PATH)
        from isotope_zero.core.store import MemoryStore  # type: ignore
        from isotope_zero.types import MemoryCard  # type: ignore
        from isotope_zero.daemon.client import DaemonClient  # type: ignore

        _MemoryStore = MemoryStore
        _MemoryCard = MemoryCard
        _DaemonClient = DaemonClient
    except Exception as exc:  # pragma: no cover - path/env dependent
        _engine_import_error = exc
        raise EngineError(
            f"Isotope Zero engine not importable from {_ENGINE_PATH!r}: {exc}\n"
            "Set IZERO_ENGINE_PATH to the directory containing the "
            "`isotope_zero` package (e.g. prototypes/daemon_v0.7)."
        ) from exc


# --------------------------------------------------------------------------- #
# Embedder selection — graceful degradation.
# --------------------------------------------------------------------------- #
class _StubEmbedder:
    """Deterministic L2-normalized feature-hash embedder (zero deps).

    Used when neither the daemon nor onnxruntime is available. Identical texts
    produce identical vectors (so self-similarity is 1.0); unrelated texts are
    near-orthogonal. This keeps the adapters fully runnable for tests and local
    dev without forcing a 360MB onnxruntime install.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def is_real(self) -> bool:
        return False

    def _hash_vec(self, text: str) -> list[float]:
        # Unsigned feature hashing over word tokens AND character trigrams.
        # Unsigned (no sign flipping) is essential: shared tokens/strings always
        # add to the same bucket, so texts with lexical overlap land with
        # POSITIVE cosine similarity and identical texts score exactly 1.0.
        # Trigrams give partial-word overlap ("dark mode" vs "dark mode
        # preference") a positive signal, not just exact-token matches. The
        # L2-normalized output keeps cosine == dot product for vector_search.
        vec = [0.0] * self._dim
        norm_text = text.lower()
        features: list[str] = list(norm_text.split())
        # Pad so short texts still yield trigrams; collapse whitespace.
        padded = " " + " ".join(norm_text.split()) + " "
        for i in range(len(padded) - 2):
            features.append(padded[i : i + 3])  # char trigram
        for feat in features:
            h = hashlib.sha256(feat.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "little") % self._dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # Degenerate (empty/no-token text): use a deterministic non-zero
            # vector derived from the whole-string hash so it is still searchable.
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec[0] = 1.0
            for i in range(min(self._dim, 8)):
                vec[i] += (int.from_bytes(h[i : i + 1], "little") % 7) - 3
            norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    def embed_text(self, text: str) -> list[float]:
        return self._hash_vec(text)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._hash_vec(t) for t in texts]


def _build_embedder(
    embedder: Any = None,
    use_daemon: bool = False,
    dim: int = DEFAULT_DIM,
) -> tuple[Any, bool]:
    """Choose an embedder with graceful degradation.

    Returns ``(embedder, is_real)``. Priority: explicit > daemon (if reachable)
    > local ONNX (if importable) > stub. ``is_real`` tells callers whether
    semantic vectors are "real" (ONNX/daemon) or the deterministic fallback.
    """
    if embedder is not None:
        real = getattr(embedder, "is_real", lambda: True)
        return embedder, bool(real() if callable(real) else real)

    if use_daemon:
        _import_engine()
        assert _DaemonClient is not None
        try:
            client = _DaemonClient()
            if client.ping():  # daemon reachable?
                return client, client.is_real()
        except Exception:
            pass  # fall through to local onnx / stub

    # Try a local ONNX engine (avoids the daemon spawn if the libs are present).
    try:
        _import_engine()
        from isotope_zero.embeddings.onnx_embed import EmbeddingEngine  # type: ignore

        eng = EmbeddingEngine()
        if eng.is_real():
            return eng, True
    except Exception:
        pass

    return _StubEmbedder(dim=dim), False


# --------------------------------------------------------------------------- #
# Engine facade — the API every adapter calls.
# --------------------------------------------------------------------------- #
class Engine:
    """Framework-agnostic facade over ``MemoryStore`` + an embedder.

    Adapters construct one of these (or call :func:`get_engine`) and use its
    methods; they never import ``MemoryStore`` / ``MemoryCard`` themselves.
    All search results are returned as plain dicts so adapters can map them
    into framework objects without engine coupling.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        embedder: Any = None,
        use_daemon: bool = False,
        dim: int = DEFAULT_DIM,
    ) -> None:
        _import_engine()
        assert _MemoryStore is not None and _MemoryCard is not None
        self._MemoryCard = _MemoryCard
        self._embedder, self._is_real = _build_embedder(
            embedder=embedder, use_daemon=use_daemon, dim=dim
        )
        self._store = _MemoryStore(db_path, embedder=self._embedder)
        self.db_path = db_path
        self.dim = getattr(self._embedder, "dim", dim)

    # -- properties -------------------------------------------------------- #
    @property
    def is_real(self) -> bool:
        """True when using ONNX/daemon embeddings (not the stub fallback)."""
        return self._is_real

    @property
    def store(self) -> Any:
        """Direct access to the underlying ``MemoryStore`` (escape hatch)."""
        return self._store

    @property
    def embedder(self) -> Any:
        return self._embedder

    # -- writes ------------------------------------------------------------ #
    def add_text(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        card_id: str | None = None,
        evidence: str | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Store one text as a memory card; returns the card id.

        ``metadata`` is a free-form dict. Non-reserved keys are folded into the
        card's JSON ``tags`` as ``"key=value"`` strings so they round-trip
        through the tag-based store and survive export; reserved keys
        (``evidence``, ``source_tokens``, ``scope``) map onto native card
        fields — ``scope`` drives multi-tier isolation (LlamaIndex parity:
        metadata filtering → scoping).
        """
        metadata = dict(metadata or {})
        tag_list: list[str] = list(tags or [])
        for k, v in metadata.items():
            if k in ("evidence", "source_tokens", "scope"):
                continue
            tag_list.append(f"{k}={v}")
        if card_id is None:
            card_id = f"iz-{uuid.uuid4().hex[:12]}"
        if embedding is None:
            embedding = self._embedder.embed_text(text)
        card_kwargs: dict[str, Any] = dict(
            id=card_id,
            fact=text,
            evidence=evidence or metadata.get("evidence", ""),
            timestamp=metadata.get("timestamp", time.time()),
            tags=tag_list,
            embedding=embedding,
            source_tokens=int(metadata.get("source_tokens", len(text.split()))),
        )
        # Multi-tier scoping: forward `scope` only when the resolved MemoryCard
        # dataclass supports it (synthesis_v1.0+). The daemon_v0.7 baseline —
        # the engine's default resolved path — predates the scope field, so we
        # introspect the dataclass fields once to stay backward-compatible.
        scope = metadata.get("scope", "default")
        if scope and scope != "default":
            try:
                _fields = {f.name for f in self._MemoryCard.__dataclass_fields__.values()}
            except Exception:
                _fields = set()
            if "scope" in _fields:
                card_kwargs["scope"] = str(scope)
        card = self._MemoryCard(**card_kwargs)
        self._store.add(card)
        return card_id

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Sequence[dict[str, Any]] | None = None,
        tags: Sequence[Sequence[str]] | None = None,
        card_ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Batch add; returns the list of card ids in input order."""
        if metadatas is None:
            metadatas = [None] * len(texts)
        if tags is None:
            tags = [None] * len(texts)
        if card_ids is None:
            card_ids = [None] * len(texts)
        ids: list[str] = []
        for text, meta, tg, cid in zip(texts, metadatas, tags, card_ids):
            ids.append(
                self.add_text(text, metadata=meta, tags=tg, card_id=cid)
            )
        return ids

    # -- reads ------------------------------------------------------------- #
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        query_embedding: list[float] | None = None,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search; returns up to ``top_k`` result dicts.

        Each dict: ``{id, text, score, metadata, tags}``. ``score`` is cosine
        similarity in [0, 1] (higher = better). Results are sorted by score
        descending.

        ``scope`` confines results to one isotope_zero memory scope (LlamaIndex
        parity: metadata filtering → scoping). ``None`` uses the store default.
        Forwarded only when the backing store's ``vector_search`` accepts it;
        older prototype stores predate the param and get the unscoped path.
        """
        if query_embedding is None:
            query_embedding = self._embedder.embed_text(query)
        try:
            hits = self._store.vector_search(
                query_embedding, k=top_k,
                scope=scope if scope is not None else "default",
            )
        except TypeError:
            # Older store (daemon_v0.7 baseline) predates the `scope` kwarg.
            hits = self._store.vector_search(query_embedding, k=top_k)
        results: list[dict[str, Any]] = []
        for card, score in hits:
            results.append(self._card_to_dict(card, score))
        return results

    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
        rrf_k: int = 60,  # noqa: ARG002 - reserved for a future store API
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid (vector + FTS5 BM25) search via late-fusion RRF.

        Combines the store's semantic cosine branch and lexical BM25 branch
        using Reciprocal Rank Fusion, then re-ranks the fused candidates.
        ``fts_weight`` / ``vector_weight`` are mapped onto the store's single
        ``alpha`` knob (the vector branch's share) as
        ``alpha = vector_weight / (fts_weight + vector_weight)``. If both
        weights are zero the search degenerates to the store's default
        ``alpha=0.70``.

        Returns the same list-of-dicts shape as :meth:`search`
        (``{id, text, score, metadata, tags, timestamp}``), sorted by fused
        score descending. Note the fused RRF scores are NOT clamped to [0, 1]
        (they live in ~[0, 1/rrf_k] + boost); callers comparing against
        cosine similarity should prefer :meth:`search`.

        ``rrf_k`` is accepted for API completeness but is currently ignored:
        the store hardcodes the RRF smoothing constant at 60 internally. It
        is reserved for a future store API that exposes it as a parameter.

        Fallback posture: if the backing store does not expose
        ``hybrid_search`` (older store revisions such as the daemon_v0.7
        baseline that only implement ``vector_search``), this method
        transparently degrades to a pure-vector search. The method never
        raises on store capability — callers get the best available fusion.
        """
        total = fts_weight + vector_weight
        if total <= 0:
            alpha = 0.70  # degenerate: both weights zero -> store default
        else:
            alpha = vector_weight / total
        if query_embedding is None:
            query_embedding = self._embedder.embed_text(query)
        # rrf_k is NOT forwarded: the store's hybrid_search signature exposes
        # (query, query_vec, k, alpha, top_n_per_branch, scope) but hardcodes
        # the RRF smoothing constant at 60 internally. Pass alpha + scope; a
        # future store revision that adds an `rrf_k` parameter can thread it
        # through here without breaking this adapter's public signature.
        store_hybrid = getattr(self._store, "hybrid_search", None)
        if callable(store_hybrid):
            hits = store_hybrid(
                query,
                query_embedding,
                k=top_k,
                alpha=alpha,
                scope=scope if scope is not None else "default",
            )
        else:
            # Store lacks hybrid_search (e.g. daemon_v0.7 baseline). Degrade
            # to pure vector search so the adapter stays functional across
            # store revisions. The lexical branch is simply absent.
            hits = self._store.vector_search(query_embedding, k=top_k)
        results: list[dict[str, Any]] = []
        for card, score in hits:
            results.append(self._card_to_dict(card, score))
        return results

    def get(self, card_id: str) -> dict[str, Any] | None:
        """Fetch a single card by id, or None if absent."""
        card = self._store.get(card_id)
        if card is None:
            return None
        return self._card_to_dict(card, None)

    def count(self) -> int:
        return self._store.count()

    def all(self) -> list[dict[str, Any]]:
        return [self._card_to_dict(c, None) for c in self._store.all()]

    # -- deletes ----------------------------------------------------------- #
    def delete(self, card_id: str) -> bool:
        """Delete one card; returns True if a row was removed."""
        return bool(self._store.delete(card_id))

    # -- internals --------------------------------------------------------- #
    @staticmethod
    def _parse_meta_tags(tags: Sequence[str]) -> tuple[list[str], dict[str, str]]:
        """Split card tags into free-form tags + ``key=value`` metadata pairs."""
        plain: list[str] = []
        meta: dict[str, str] = {}
        for t in tags:
            if "=" in t:
                k, _, v = t.partition("=")
                meta[k] = v
            else:
                plain.append(t)
        return plain, meta

    def _card_to_dict(
        self, card: Any, score: float | None
    ) -> dict[str, Any]:
        plain_tags, meta = self._parse_meta_tags(card.tags or [])
        return {
            "id": card.id,
            "text": card.fact,
            "score": score,
            "metadata": {**meta, "evidence": card.evidence},
            "tags": plain_tags,
            "timestamp": card.timestamp,
        }


def get_engine(
    db_path: str = ":memory:",
    *,
    embedder: Any = None,
    use_daemon: bool = False,
    dim: int = DEFAULT_DIM,
) -> Engine:
    """Convenience factory — construct an :class:`Engine` with defaults."""
    return Engine(db_path=db_path, embedder=embedder, use_daemon=use_daemon, dim=dim)
