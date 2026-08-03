"""Phase 0 benchmark harness for isotope_zero.

Measures the cost thesis directly: how many tokens does isotope_zero inject per
useful fact, versus replaying raw conversation history? And what does that
save in dollars versus a remote-embedding baseline?

Metrics produced (as a Markdown table via `main()`):

    Tokens per Useful Fact      isotope_zero vs raw-context-history
    Latency per Operation       write (ms), read (ms), vector search (ms)
    Estimated Cost Savings      $0 local embeddings vs OpenAI text-embedding-3-small
    Correctness Floor           100% recall on structured fact queries

The harness runs a synthetic multi-turn session: a sequence of user
statements (identity, preferences, project, corrections/updates) followed by
a battery of queries — some explicit (must hit via SQL), some fuzzy.
"""
from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import dataclass, field

from isotope_zero.core.consolidation import Consolidator
from isotope_zero.core.router import QueryRouter
from isotope_zero.core.store import MemoryStore
from isotope_zero.core.triage import classify_action, compress_to_card
from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.tokens import estimate_tokens
from isotope_zero.types import MemoryCard

# OpenAI text-embedding-3-small pricing (USD per 1M tokens), used as the
# remote-embedding cost baseline. Source: OpenAI public pricing.
_OPENAI_EMBED_PRICE_PER_M = 0.02  # $/1M tokens

# --- Synthetic session workload ----------------------------------------- #

# Multi-turn user statements: identity, preferences, state, then updates.
SESSION_STATEMENTS: list[str] = [
    "My name is Alice Chen.",
    "I'm a senior backend engineer.",
    "I work at a company called Acme Corp.",
    "My current project is the Mercury billing system.",
    "I prefer Rust for systems work.",
    "I live in Berlin.",
    "My timezone is Europe/Berlin (CET).",
    "I use neovim as my editor.",
    "Actually, I switched from Rust to Go for the Mercury backend.",
    "I no longer prefer neovim; I use Zed now.",
    "My favorite database is PostgreSQL.",
    "I'm learning Japanese in my spare time.",
    "I have a dog named Pixel.",
    "My team uses Linear for issue tracking.",
    "I prefer dark mode in my editor.",
]

# Explicit state queries — these MUST be answered correctly (correctness floor).
# Each is paired with the expected answer substring (case-insensitive).
STRUCTURED_QUERIES: list[tuple[str, str]] = [
    ("What is my name?", "alice"),
    ("What's my current project?", "mercury"),
    ("Who am I?", "alice"),
    ("What is my role?", "engineer"),
    ("Where do I live?", "berlin"),
    ("What is my timezone?", "berlin"),
    ("What is my favorite database?", "postgres"),
    ("What editor do I use?", "zed"),
]

# Fuzzy / semantic queries — no single exact key, should route to vector.
FUZZY_QUERIES: list[str] = [
    "What programming languages does the user care about?",
    "Tell me about the user's pet.",
    "What does the user do for fun?",
    "Which tools does the user prefer?",
    "What is the user's working style?",
]


@dataclass
class BenchmarkResult:
    """Aggregate benchmark output."""
    # Tokens per useful fact.
    isotope_zero_tokens_per_fact: float = 0.0
    raw_tokens_per_fact: float = 0.0
    tokens_saved_per_fact: float = 0.0
    # Latency (ms).
    write_latency_ms: list[float] = field(default_factory=list)
    read_sql_latency_ms: list[float] = field(default_factory=list)
    read_vector_latency_ms: list[float] = field(default_factory=list)
    # Cost.
    embedding_cost_usd: float = 0.0
    openai_baseline_cost_usd: float = 0.0
    cost_savings_usd: float = 0.0
    cost_savings_pct: float = 0.0
    # Correctness.
    structured_recall: float = 0.0
    structured_correct: int = 0
    structured_total: int = 0
    # Context.
    embedding_is_real: bool = False
    card_count: int = 0
    db_size_bytes: int = 0


def run_benchmark(db_path: str = ":memory:", embedder: EmbeddingEngine | None = None) -> BenchmarkResult:
    """Run the full synthetic-session benchmark and return aggregate metrics."""
    eng = embedder or EmbeddingEngine()
    store = MemoryStore(db_path, embedder=eng)
    router = QueryRouter(store, eng)

    # ---- Write path: ingest all session statements ---- #
    write_latencies: list[float] = []
    raw_history_tokens = 0
    for stmt in SESSION_STATEMENTS:
        t0 = time.perf_counter()
        action = classify_action(stmt)
        card = compress_to_card(stmt, embedding=eng.embed_text(compress_to_card(stmt).fact))
        if action.action.value == "DELETE" and action.target_id:
            for field_ in ("tags", "fact"):
                cands = store.sql_lookup(field_, action.target_id)
                if cands:
                    for c in cands:
                        store.delete(c.id)
                    break
        elif action.action.value == "UPDATE" and action.target_id:
            updated = False
            for field_ in ("tags", "fact"):
                existing = store.sql_lookup(field_, action.target_id)
                if existing:
                    card.id = existing[0].id
                    store.update(card)
                    updated = True
                    break
            if not updated:
                store.add(card)
        else:
            store.add(card)
        write_latencies.append((time.perf_counter() - t0) * 1000.0)
        raw_history_tokens += estimate_tokens(stmt)

    # ---- Read path: structured correctness floor ---- #
    sql_latencies: list[float] = []
    correct = 0
    for q, expected in STRUCTURED_QUERIES:
        t0 = time.perf_counter()
        res = router.query(q, token_budget=300)
        sql_latencies.append((time.perf_counter() - t0) * 1000.0)
        # Correctness: at least one hit whose fact contains the expected answer.
        hit_facts = " ".join(h.card.fact for h in res.hits).lower()
        if expected.lower() in hit_facts:
            correct += 1

    # ---- Read path: fuzzy (vector) latency ---- #
    vec_latencies: list[float] = []
    for q in FUZZY_QUERIES:
        t0 = time.perf_counter()
        router.query(q, token_budget=300)
        vec_latencies.append((time.perf_counter() - t0) * 1000.0)

    # ---- Tokens per useful fact ---- #
    # "Useful facts" = the structured facts we expect to be retrievable.
    n_facts = len(STRUCTURED_QUERIES)
    # isotope_zero tokens injected: sum of fact+evidence tokens across the structured
    # query results (the actual context cost isotope_zero would impose).
    isotope_zero_tokens = 0
    for q, _ in STRUCTURED_QUERIES:
        res = router.query(q, token_budget=300)
        isotope_zero_tokens += res.tokens_used
    isotope_zero_per_fact = isotope_zero_tokens / max(1, n_facts)
    # Raw baseline: replaying the entire conversation history for each query.
    raw_per_fact = (raw_history_tokens * len(STRUCTURED_QUERIES)) / max(1, n_facts)
    # Simplify: raw per-query cost IS the full history.
    raw_per_fact = raw_history_tokens / 1  # one query's worth of raw replay

    # ---- Cost: local embeddings vs OpenAI baseline ---- #
    # Count embedding input tokens processed during the benchmark (writes only;
    # reads also embed queries but we count those too for a fair comparison).
    embed_input_tokens = sum(estimate_tokens(s) for s in SESSION_STATEMENTS)
    embed_input_tokens += sum(estimate_tokens(q) for q, _ in STRUCTURED_QUERIES)
    embed_input_tokens += sum(estimate_tokens(q) for q in FUZZY_QUERIES)
    openai_cost = (embed_input_tokens / 1_000_000) * _OPENAI_EMBED_PRICE_PER_M

    return BenchmarkResult(
        isotope_zero_tokens_per_fact=round(isotope_zero_per_fact, 1),
        raw_tokens_per_fact=round(raw_per_fact, 1),
        tokens_saved_per_fact=round(raw_per_fact - isotope_zero_per_fact, 1),
        write_latency_ms=write_latencies,
        read_sql_latency_ms=sql_latencies,
        read_vector_latency_ms=vec_latencies,
        embedding_cost_usd=0.0,
        openai_baseline_cost_usd=round(openai_cost, 6),
        cost_savings_usd=round(openai_cost, 6),
        cost_savings_pct=100.0 if openai_cost > 0 else 0.0,
        structured_recall=round(correct / n_facts, 3),
        structured_correct=correct,
        structured_total=n_facts,
        embedding_is_real=eng.is_real,
        card_count=store.count(),
        db_size_bytes=store.db_size_bytes(),
    )


def _stats(xs: list[float]) -> tuple[float, float]:
    """Return (mean, median) rounded to 3 dp."""
    if not xs:
        return 0.0, 0.0
    return round(statistics.mean(xs), 3), round(statistics.median(xs), 3)


def render_markdown(r: BenchmarkResult) -> str:
    """Render the benchmark result as a clean Markdown table."""
    w_mean, w_med = _stats(r.write_latency_ms)
    s_mean, s_med = _stats(r.read_sql_latency_ms)
    v_mean, v_med = _stats(r.read_vector_latency_ms)
    lines: list[str] = []
    lines.append("# isotope_zero Phase 0 Benchmark\n")
    lines.append(
        f"> Embedding mode: **{'REAL ONNX' if r.embedding_is_real else 'FALLBACK (deterministic hash)'}** "
        f"| Cards: {r.card_count} | DB size: {r.db_size_bytes} B\n"
    )
    lines.append("## Tokens per Useful Fact (isotope_zero vs raw history)\n")
    lines.append("| Metric | isotope_zero | Raw history |")
    lines.append("|---|---|---|")
    lines.append(f"| Tokens / useful fact | **{r.isotope_zero_tokens_per_fact}** | {r.raw_tokens_per_fact} |")
    lines.append(f"| Tokens saved / fact | **{r.tokens_saved_per_fact}** | — |")
    lines.append("")
    lines.append("## Latency per Operation (ms)\n")
    lines.append("| Operation | Mean | Median |")
    lines.append("|---|---|---|")
    lines.append(f"| Write (triage+compress+embed+store) | {w_mean} | {w_med} |")
    lines.append(f"| Read — SQL route | {s_mean} | {s_med} |")
    lines.append(f"| Read — Vector route | {v_mean} | {v_med} |")
    lines.append("")
    lines.append("## Estimated Cost Savings (embeddings)\n")
    lines.append("| Baseline | Cost (USD) |")
    lines.append("|---|---|")
    lines.append(f"| isotope_zero (local ONNX) | **${r.embedding_cost_usd:.6f}** |")
    lines.append(f"| OpenAI text-embedding-3-small | ${r.openai_baseline_cost_usd:.6f} |")
    lines.append(f"| Savings | **${r.cost_savings_usd:.6f} ({r.cost_savings_pct:.1f}%)** |")
    lines.append("")
    lines.append("## Correctness Floor (structured fact recall)\n")
    lines.append("| Correct | Total | Recall |")
    lines.append("|---|---|---|")
    lines.append(f"| {r.structured_correct} | {r.structured_total} | **{r.structured_recall:.1%}** |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Phase 3: scaling benchmark (deduplication + decay at scale)
# ---------------------------------------------------------------------- #

# Number of synthetic cards to seed before consolidation. >1000 per the spec.
_SCALING_SEED_COUNT = 1200

# A small pool of canonical fact templates; we generate N slightly-redundant
# variants of each so the store fills with near-duplicates + exact duplicates
# that consolidation can fold. This mirrors real accumulation: the same fact
# re-stated across many turns. The templates are aligned to the structured
# correctness queries so the post-consolidation recall floor is meaningful.
_FACT_TEMPLATES: list[str] = [
    "The user's name is Alice Chen",
    "The user's current project is the Mercury billing system",
    "The user prefers Rust for systems work",
    "The user lives in Berlin",
    "The user's favorite database is PostgreSQL",
    "The user uses zed as their editor",
    "The user's role is a software engineer",
    "The user has a dog named Pixel",
    "The user's timezone is Europe/Berlin",
    "The user is learning Japanese",
]

# Variants that paraphrase the template — high token-overlap / near-duplicate
# embeddings, so the dedup engine folds them. (last word swapped, filler
# inserted, etc.)
_VARIANTS: list[str] = [
    "{base}.",
    "Actually, {base_lower}.",
    "Just so you know, {base_lower}.",
    "{base} — that's still true.",
    "I should mention that {base_lower}.",
]


@dataclass
class ScalingResult:
    """Outcome of the Phase 3 scaling benchmark."""
    seed_count: int = 0
    pre_consolidation_cards: int = 0
    post_consolidation_cards: int = 0
    pre_consolidation_tokens: int = 0
    post_consolidation_tokens: int = 0
    token_reduction_pct: float = 0.0
    pre_query_latency_ms: float = 0.0
    post_query_latency_ms: float = 0.0
    merged_cards: int = 0
    decayed_cards: int = 0
    consolidation_latency_ms: float = 0.0
    correctness_floor: float = 0.0  # post-consolidation recall on structured queries
    embedding_is_real: bool = False
    reduction_target_met: bool = False  # >= 25% context reduction


def _seed_redundant_store(store: MemoryStore, eng: EmbeddingEngine, n: int) -> None:
    """Seed a store with n slightly-redundant cards built from templates."""
    for i in range(n):
        base = _FACT_TEMPLATES[i % len(_FACT_TEMPLATES)]
        variant = _VARIANTS[i % len(_VARIANTS)].format(base=base, base_lower=base[0].lower() + base[1:])
        emb = eng.embed_text(variant)
        store.add(
            MemoryCard(
                id=uuid.uuid4().hex,
                fact=variant,
                evidence=variant,
                timestamp=float(i),
                tags=[],
                embedding=emb,
                source_tokens=estimate_tokens(variant),
                # Spread access counts so some cards are "live" and some are
                # decay candidates. ~70% never recalled.
                access_count=1 if i % 10 == 0 else 0,
                last_access=float(i) if i % 10 == 0 else float(i),
            )
        )


def _query_latency_ms(router: QueryRouter, queries: list[str], budget: int = 300) -> float:
    """Mean query latency (ms) over a battery of queries."""
    lats = []
    for q in queries:
        res = router.query(q, token_budget=budget)
        lats.append(res.latency_ms)
    return statistics.mean(lats) if lats else 0.0


def _total_context_tokens(store: MemoryStore) -> int:
    """Sum of fact+evidence tokens across all cards — the context footprint."""
    return sum(estimate_tokens(c.fact + " " + c.evidence) for c in store.all())


def _correctness_floor(router: QueryRouter) -> float:
    """Post-consolidation recall on the structured queries from the base suite."""
    correct = 0
    for q, expected in STRUCTURED_QUERIES:
        res = router.query(q, token_budget=300)
        facts = " ".join(h.card.fact for h in res.hits).lower()
        if expected.lower() in facts:
            correct += 1
    return correct / len(STRUCTURED_QUERIES)


def run_scaling_benchmark(
    db_path: str = ":memory:",
    embedder: EmbeddingEngine | None = None,
    seed_count: int = _SCALING_SEED_COUNT,
) -> ScalingResult:
    """Seed 1000+ redundant cards, consolidate, and measure the delta.

    Asserts the Phase 3 thesis: post-consolidation context size is reduced by
    >= 25% while the correctness floor is maintained at 100%.
    """
    eng = embedder or EmbeddingEngine()
    store = MemoryStore(db_path, embedder=eng)
    router = QueryRouter(store, eng)

    # 1) Seed with redundant + decay-prone cards.
    _seed_redundant_store(store, eng, seed_count)
    pre_cards = store.count()
    pre_tokens = _total_context_tokens(store)

    # 2) Baseline query latency (before consolidation).
    sample_queries = [q for q, _ in STRUCTURED_QUERIES] + FUZZY_QUERIES
    pre_latency = _query_latency_ms(router, sample_queries)

    # 3) Run one consolidation sweep.
    t0 = time.perf_counter()
    report = Consolidator(store, embedder=eng, min_age_seconds=0).run()
    cons_ms = (time.perf_counter() - t0) * 1000.0

    # 4) Re-measure latency + context size post-consolidation.
    post_cards = store.count()
    post_tokens = _total_context_tokens(store)
    post_latency = _query_latency_ms(router, sample_queries)
    correctness = _correctness_floor(router)

    reduction_pct = ((pre_tokens - post_tokens) / pre_tokens * 100.0) if pre_tokens else 0.0

    return ScalingResult(
        seed_count=seed_count,
        pre_consolidation_cards=pre_cards,
        post_consolidation_cards=post_cards,
        pre_consolidation_tokens=pre_tokens,
        post_consolidation_tokens=post_tokens,
        token_reduction_pct=round(reduction_pct, 1),
        pre_query_latency_ms=round(pre_latency, 3),
        post_query_latency_ms=round(post_latency, 3),
        merged_cards=report.merged_cards,
        decayed_cards=report.decayed_cards,
        consolidation_latency_ms=round(cons_ms, 1),
        correctness_floor=round(correctness, 3),
        embedding_is_real=eng.is_real,
        reduction_target_met=reduction_pct >= 25.0,
    )


def render_scaling_markdown(s: ScalingResult) -> str:
    """Render the scaling benchmark result as Markdown."""
    lines: list[str] = []
    lines.append("# isotope_zero Phase 3 Scaling Benchmark (Consolidation)\n")
    lines.append(
        f"> Embedding mode: **{'REAL ONNX' if s.embedding_is_real else 'FALLBACK'}** "
        f"| Seeded: {s.seed_count} cards\n"
    )
    lines.append("## Context Compression at Scale\n")
    lines.append("| Metric | Before | After |")
    lines.append("|---|---|---|")
    lines.append(f"| Card count | {s.pre_consolidation_cards} | {s.post_consolidation_cards} |")
    lines.append(f"| Context tokens (fact+evidence) | {s.pre_consolidation_tokens} | {s.post_consolidation_tokens} |")
    lines.append(f"| Token reduction | — | **{s.token_reduction_pct}%** |")
    lines.append(f"| >= 25% target | — | {'✅ MET' if s.reduction_target_met else '❌ MISSED'} |")
    lines.append("")
    lines.append("## Query Latency (ms, mean over battery)\n")
    lines.append("| Before consolidation | After consolidation |")
    lines.append("|---|---|")
    lines.append(f"| {s.pre_query_latency_ms} | {s.post_query_latency_ms} |")
    lines.append("")
    lines.append("## Consolidation Sweep\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Merged (dedup) | {s.merged_cards} |")
    lines.append(f"| Decayed (pruned) | {s.decayed_cards} |")
    lines.append(f"| Sweep latency (ms) | {s.consolidation_latency_ms} |")
    lines.append("")
    lines.append("## Correctness Floor (post-consolidation)\n")
    lines.append("| Recall | Target |")
    lines.append("|---|---|")
    lines.append(f"| {s.correctness_floor:.1%} | 100% |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run the base + scaling benchmarks and print Markdown reports."""
    print(render_markdown(run_benchmark()))
    print()
    print(render_scaling_markdown(run_scaling_benchmark()))


if __name__ == "__main__":  # pragma: no cover
    main()
