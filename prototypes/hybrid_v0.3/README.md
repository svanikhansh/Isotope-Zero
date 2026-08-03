# Hybrid v0.3 Prototype — BM25 + Vector Re-Ranking (Method 1)

This folder is the **v0.3.0a1 prototype** of Isotope Zero implementing
**Method 1: Hybrid Pre-Filtering**. It was scaffolded from the pure-Python
`prototypes/python_v0.1/` baseline (same 129-test green suite) and adds a
two-pass retrieval pipeline designed to bypass the full-matrix vector math
bottleneck and target **sub-0.05 ms** query latency at 10,000+ cards.

## Thesis

Instead of computing floating-point cosine similarity across all 10,000
vectors (𝒪(N·d)), Hybrid Pre-Filtering uses SQLite's C-native **FTS5** engine
to perform high-speed **BM25** lexical candidate extraction in <0.04 ms
(≤ 50 cards), followed by NumPy vector re-ranking on only the candidate set
(𝒪(k·d), where k = 50):

1. **Pass 1 — Lexical filter (FTS5/BM25):** a single SQL query against the
   `cards_fts` virtual table (`tokenize='porter unicode61'`) returns up to
   `candidate_limit` card_ids ranked by BM25. This runs in SQLite's C engine,
   never entering Python per-row.
2. **Fallback:** if FTS5 returns fewer than `top_k` candidates (strict keyword
   mismatch — the query shares no tokens with any card), the search falls back
   to the full-matrix vector scan so recall never silently drops to zero.
3. **Pass 2 — Vector re-rank (NumPy):** fetch embeddings only for the candidate
   set, compute cosine (dot) vs the query, and combine
   `Score = α·Norm(BM25) + (1−α)·Cosine` (α = 0.3). Return the top `top_k`.

The architectural bet: a C-native lexical filter plus a tiny 𝒪(k·d) re-rank is
both faster *and* cheaper than a 10k×384 float32 matmul — *if* the lexical
filter preserves recall. The honest open question this prototype answers:
**does BM25 pre-filtering preserve 100% needle recall when the queries are
semantic phrasings that share no rare tokens with the target?**

## What changed from v0.1

- `isotope_zero/core/store.py`:
  - New FTS5 virtual table `cards_fts(card_id UNINDEXED, content,
    tokenize='porter unicode61')`, created in `_setup_schema` with a one-time
    backfill from `memories` for pre-existing DBs.
  - Automatic sync: every `add` / `update` / `delete` mirrors content into
    `cards_fts` (delete-then-insert upsert on the write-path cursor;
    superseded/folded cards are removed and never searchable).
  - New `hybrid_vector_search(query_text, query_vec, top_k=5, candidate_limit=50,
    alpha=0.3)` implementing the two-pass pipeline. The original `vector_search`
    full-matrix scan is **untouched** and serves as the fallback.
- `isotope_zero/eval/adversarial.py`: the stress harness now measures hybrid
  p50/p95/p99 latency at 10k cards alongside the full-matrix scan, and a
  parallel hybrid needle-recall run that isolates `hybrid_vector_search` from
  the router.

## Run it

```bash
cd prototypes/hybrid_v0.3
# editable install must point HERE (re-point after switching prototypes):
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m pip install -e .

# baseline compatibility (unchanged from v0.1):
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m pytest tests/ -q   # 129 passed / 5 skipped

# the Method 1 benchmark (10k scale + 25-proc concurrency, a few minutes):
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m isotope_zero.eval.adversarial
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m isotope_zero.eval.adversarial --json
```

Always use the **absolute** venv path after `cd`-ing in — a relative
`.venv/bin/python` resolves against the new cwd and fails.

## Measured results (captured at freeze)

Adversarial suite, 10,000 cards, 25-worker concurrency. Claims recorded
verbatim beside the measured reality; a claim that does not hold is FAIL
(per the project's no-fabrication rule).

| metric | measured | claim | verdict |
|---|---|---|---|
| Hybrid search p99 @10k | 0.87–0.93 ms | < 0.05 ms | **FAIL** (~17× over) |
| Needle recall via hybrid | 40.0% | 100% | **FAIL** (−60 pts) |
| Full-matrix vector p99 @10k | 0.33–0.52 ms | < 2.0 ms | PASS (baseline) |
| Needle recall via router | 100.0% | 100% | PASS (baseline) |
| SQL lookup p99 @10k | 0.66–0.97 ms | < 0.80 ms | borderline |
| Negation incorrect merges | 0 / 100 | 0 | PASS |
| Concurrency (25 proc, 25k ops) | 0 err / 0 corrupt | 0 / 0 | PASS |
| RSS @10k | ~400 MB | < 200 MB | FAIL (structural) |

### Honest verdict — Method 1 does not meet its targets

**The hybrid two-pass pipeline is *slower* than the full-matrix scan it was
meant to beat** (hybrid p99 0.87–0.93 ms vs full-matrix vector p99 0.33–0.52 ms
at 10k cards). The reason: at this scale the baseline `vector_search` is
already a single BLAS-accelerated `matrix @ q` matmul (~0.3 ms), so the FTS5
MATCH parse + candidate retrieval + second re-rank adds overhead that
*exceeds* the full scan it replaces — the lexical pre-filter never pays off.

**Needle recall collapsed to 40%** (vs the router's 100%). The `NEEDLE_QUERIES`
are semantic phrasings ("What port is the SSH key on?") that contain no
literal rare token ("2204"), and the distractor set is dense with "SSH
port"/"port N" text — so BM25 lexical matching surfaces distractors over the
needle. The router recovers 100% via its lexical/numeric boost and
truncation logic that the bare hybrid search lacks. This is the semantic-only
query failure mode the brief anticipated, now measured rather than assumed.

Both Method 1 claims FAIL, by large margins. The 0.05 ms target was set against
a full-scan 𝒪(N·d) baseline that is in practice already ~0.3 ms, so even an
ideal pre-filter would clear it — but FTS5's own per-query overhead (parse +
BM25 over a 10k-row index) plus the re-rank lands well above it. There is a
real, measured tension between the latency target and the recall target, not
a free win.

The implementation itself is correct: FTS5 sync is verified on add/update/
delete/supersede, superseded cards are never searchable, and the two-pass
pipeline produces valid ranked results — it simply does not achieve the
sub-0.05 ms / 100%-recall target the thesis claimed.

## Siblings

- `prototypes/python_v0.1/` — the pure-Python v0.1.0 baseline this was cut from.
- `prototypes/rust_v0.2/` — the Phase 6 Smart Bridge (NumPy/BLAS vector +
  Rust PyO3 negation), a different acceleration axis.
