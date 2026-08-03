"""Shared type definitions — the single integration contract for isotope_zero.

Every subsystem (embeddings, triage, store, router, mcp, eval) imports from
here so that parallel-built modules slot together without reconciliation.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any


class ActionType(str, enum.Enum):
    """Write-path triage decision."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


@dataclass(frozen=True)
class ActionResult:
    """Output of the write-path classifier.

    `confidence` is in [0.0, 1.0]. `escalated` is True when the fast heuristic
    gave low confidence and the decision was deferred to the (mocked) LLM path.
    """

    action: ActionType
    confidence: float
    escalated: bool = False
    target_id: str | None = None  # memory id this action targets (UPDATE/DELETE)
    reasoning: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


@dataclass
class MemoryCard:
    """Compressed memory unit: a fact + minimal evidence.

    Designed to be cheap to store and retrieve. `evidence` is the smallest
    quote that justifies `fact` — never the full raw input.

    Access tracking (`access_count`, `last_access`) feeds the temporal-decay
    vitality score in Phase 3 consolidation. Both default to "fresh / never
    explicitly accessed" so existing call sites that construct cards without
    them remain valid.
    """

    id: str
    fact: str
    evidence: str
    timestamp: float
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None  # set by embeddings engine at write time
    source_tokens: int = 0  # token count of the raw input this card was made from
    # Phase 3: access tracking for temporal decay. `last_access` defaults to
    # the creation timestamp when unset (a freshly written card has just been
    # "touched" by the act of writing it).
    access_count: int = 0
    last_access: float = 0.0  # 0.0 => not yet set; store treats 0.0 as "use timestamp"
    # Phase 3 (supersession audit trail): when this card is folded into a
    # newer/other survivor during consolidation, this holds the survivor's id
    # instead of the card being hard-deleted. `None` (default) means "live".
    # Superseded cards remain retrievable by id (`store.get`) for the audit
    # trail but are excluded from `all()` / `sql_lookup` / `vector_search` /
    # `count()` so they never surface in retrieval, and the consolidator never
    # picks one as a future survivor.
    superseded_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fact": self.fact,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
            "tags": list(self.tags),
            "source_tokens": self.source_tokens,
            "access_count": self.access_count,
            "last_access": self.last_access,
            "superseded_by": self.superseded_by,
        }


@dataclass(frozen=True)
class QueryHit:
    """One retrieval result from the router."""

    card: MemoryCard
    score: float  # similarity/relevance score in [0,1] (higher = better)
    route: str  # "sql" | "vector" — which path surfaced this hit
    token_cost: int  # approximate tokens this hit contributes to context


@dataclass(frozen=True)
class QueryResult:
    """Final budget-aware retrieval result."""

    hits: list[QueryHit]
    route_used: str  # dominant route
    tokens_used: int
    tokens_saved_vs_raw: int  # vs. replaying the full raw history
    latency_ms: float
    budget_exhausted: bool


@dataclass(frozen=True)
class ConsolidationReport:
    """Outcome of one consolidation sweep.

    All the numbers the `run_consolidation` MCP tool and the scaling benchmark
    need to report: how many redundant cards were merged into survivors, how
    many decayed cards were pruned, and the net tokens reclaimed from context.
    """

    merged_cards: int  # number of cards folded into survivors
    decayed_cards: int  # number of cards pruned by temporal decay
    survivors: int  # card count after consolidation
    tokens_before: int  # total fact+evidence tokens before
    tokens_after: int  # total fact+evidence tokens after
    tokens_reclaimed: int  # tokens_before - tokens_after (>= 0)
    latency_ms: float
    # Vitality-score diagnostics for the pruned set (mean), for observability.
    pruned_mean_vitality: float = 0.0


def now_ts() -> float:
    """Wall-clock timestamp. Wrapped so tests and the benchmark can monkeypatch
    determinism without touching `time.time` globally."""
    return time.time()
