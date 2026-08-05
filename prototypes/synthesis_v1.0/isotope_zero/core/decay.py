"""Ebbinghaus temporal decay engine for isotope_zero.

Phase 7C — Ebbinghaus Forgetting Curve. Adds recency-aware scoring so that
semantic similarity and temporal proximity are fused into a single retrieval
score. A fact learned 3 seconds ago outranks a semantically similar fact learned
30 days ago, all else being equal.

Core primitives:
- calculate_retention: R(t) = exp(-Δt / S), the Ebbinghaus retention.
- update_stability: increase S after retrieval/consolidation (spaced repetition).
- hybrid_score: α · cosine + (1-α) · retention, fusing semantic and temporal.
"""
from __future__ import annotations

import math
import time

# Default half-life in hours — after 24h an unreinforced memory drops to
# ~exp(-24/24) ≈ 0.37.
_DEFAULT_HALF_LIFE_HOURS: float = 24.0
_DEFAULT_ALPHA: float = 0.70  # cosine weight in hybrid score


def calculate_retention(
    last_accessed_ts: float,
    stability: float = 1.0,
    current_ts: float | None = None,
    half_life_hours: float = _DEFAULT_HALF_LIFE_HOURS,
) -> float:
    """Ebbinghaus retention R(t) = exp(-Δt / S), clamped to [0.0, 1.0].

    Args:
        last_accessed_ts: Unix timestamp of last retrieval/access.
        stability: Memory stability factor S (higher = slower decay).
            Defaults to 1.0 for a fresh, unreinforced memory.
        current_ts: Current timestamp (defaults to time.time() if None).
        half_life_hours: Base time constant in hours.

    Returns:
        Retention value in [0.0, 1.0]. 1.0 = fully retained (just accessed),
        0.0 = completely forgotten.

    The time delta is converted from seconds to hours. A stability of S=2.0
    doubles the effective half-life; S=0.5 halves it.
    """
    if current_ts is None:
        current_ts = time.time()

    # A timestamp of zero or negative means the memory was never accessed;
    # treat it as freshly encoded.
    if last_accessed_ts <= 0.0:
        return 1.0

    delta_hours = (current_ts - last_accessed_ts) / 3600.0
    if delta_hours <= 0.0:
        return 1.0

    effective_half_life = stability * half_life_hours
    if effective_half_life <= 0.0:
        return 0.0

    retention = math.exp(-delta_hours / effective_half_life)
    # Clamp: retention is always positive but floating error can push it
    # slightly above 1.0 or below 0.0 for extreme deltas.
    if retention > 1.0:
        return 1.0
    if retention < 0.0:
        return 0.0
    return retention


def update_stability(
    current_stability: float,
    access_count: int,
    explicit_importance: float = 0.0,
) -> float:
    """Update stability S after a retrieval or consolidation event.

    Args:
        current_stability: The current stability value.
        access_count: Number of times the card has been accessed/retrieved.
            Higher counts produce diminishing returns.
        explicit_importance: User-set importance flag (0.0 to 1.0).

    Returns:
        New stability value. Never decreases below 1.0.

    Stability increases non-linearly with access count:
        boost = 1.0 + 0.5 * log1p(access_count) + 0.3 * explicit_importance
    The boost is multiplicative on current_stability.
    """
    boost = 1.0 + 0.5 * math.log1p(access_count) + 0.3 * explicit_importance
    new_s = current_stability * boost

    # Floor at 1.0 (memory never decays faster than baseline).
    if new_s < 1.0:
        return 1.0
    # Cap at 10.0 (10× half-life is sufficient; beyond this the curve is
    # essentially flat for practical retention windows).
    if new_s > 10.0:
        return 10.0
    return new_s


def hybrid_score(
    cosine_similarity: float,
    retention: float,
    alpha: float = _DEFAULT_ALPHA,
) -> float:
    """Fuse cosine similarity with Ebbinghaus retention.

    Score_final = α · cos⁺ + (1-α) · retention

    Args:
        cosine_similarity: Cosine/dot-product score from vector search.
            Negative values are clamped to 0 before weighting.
        retention: R(t) from calculate_retention.
        alpha: Weight for cosine (0.0 = pure decay, 1.0 = pure similarity).
            Default 0.70 gives semantic 70%, recency 30%.

    Returns:
        Hybrid score in [0.0, 1.0].
    """
    cos = cosine_similarity
    if cos < -1.0:
        cos = -1.0
    if cos > 1.0:
        cos = 1.0
    cos_positive = max(0.0, cos)
    score = alpha * cos_positive + (1.0 - alpha) * retention
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


# -- inline smoke test --------------------------------------------------------


def _run_smoke() -> None:
    """Inline sanity checks for the decay module."""
    now = time.time()
    passed = 0
    failed = 0

    def check(label: str, actual: float, expected_min: float, expected_max: float) -> None:
        nonlocal passed, failed
        if expected_min - 0.001 <= actual <= expected_max + 0.001:
            print(f"  PASS  {label}: {actual:.6f} in [{expected_min:.4f}, {expected_max:.4f}]")
            passed += 1
        else:
            print(f"  FAIL  {label}: {actual:.6f} NOT in [{expected_min:.4f}, {expected_max:.4f}]")
            failed += 1

    print("--- sanity: calculate_retention ---")

    # Just-created retention ~ 1.0
    r1 = calculate_retention(now, stability=1.0, current_ts=now)
    check("just-now S=1.0", r1, 0.999, 1.001)

    # 24h-old retention with S=1.0 -> exp(-1) ~ 0.3679
    r2 = calculate_retention(now - 86400, stability=1.0, current_ts=now)
    check("24h-old S=1.0", r2, 0.35, 0.38)

    # 24h-old retention with S=5.0 -> exp(-24/(5*24)) = exp(-0.2) ~ 0.8187
    r3 = calculate_retention(now - 86400, stability=5.0, current_ts=now)
    check("24h-old S=5.0", r3, 0.80, 0.83)

    # Unaccessed (ts=0) -> 1.0
    r4 = calculate_retention(0.0, stability=1.0, current_ts=now)
    check("never-accessed ts=0", r4, 0.999, 1.001)

    # Negative delta (future timestamp) -> 1.0
    r5 = calculate_retention(now + 3600, stability=1.0, current_ts=now)
    check("future ts", r5, 0.999, 1.001)

    # Very old memory -> near 0
    r6 = calculate_retention(now - 86400 * 100, stability=1.0, current_ts=now)
    check("100d-old S=1.0", r6, -0.001, 0.001)

    print("--- sanity: update_stability ---")

    s1 = update_stability(1.0, 10, 0.5)
    check("S=1.0, 10 accesses, imp=0.5", s1, 2.3, 2.5)

    s2 = update_stability(1.0, 0, 0.0)
    check("S=1.0, 0 accesses, imp=0.0", s2, 0.99, 1.01)  # boost=1.0 -> S stays 1.0

    s3 = update_stability(1.0, 100, 0.0)
    check("S=1.0, 100 accesses, imp=0.0", s3, 3.2, 3.4)  # 1 + 0.5*log1p(100) ~ 3.31

    s4 = update_stability(5.0, 0, 0.0)
    check("S=5.0, 0 accesses, imp=0.0", s4, 4.99, 5.01)

    # Cap at 10.0 — use an access count large enough to exceed the cap
    s5 = update_stability(1.0, 10**15, 1.0)
    check("cap at 10.0", s5, 9.99, 10.01)

    # Floor at 1.0
    s6 = update_stability(0.1, 0, 0.0)
    check("floor at 1.0", s6, 0.99, 1.01)

    print("--- sanity: hybrid_score ---")

    hs1 = hybrid_score(0.9, 0.4, alpha=0.7)
    check("cos=0.9, ret=0.4, alpha=0.7", hs1, 0.74, 0.76)  # 0.7*0.9 + 0.3*0.4 = 0.75

    hs2 = hybrid_score(1.0, 1.0, alpha=0.7)
    check("cos=1.0, ret=1.0, alpha=0.7", hs2, 0.99, 1.01)

    hs3 = hybrid_score(0.0, 0.0, alpha=0.7)
    check("cos=0.0, ret=0.0, alpha=0.7", hs3, -0.001, 0.001)

    hs4 = hybrid_score(-0.5, 0.5, alpha=0.7)
    check("cos=-0.5, ret=0.5, alpha=0.7", hs4, 0.14, 0.16)  # 0.7*0.0 + 0.3*0.5 = 0.15

    print(f"\n{passed} passed, {failed} failed")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_smoke()
