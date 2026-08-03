# Quantization v0.4 Prototype — Int8 Scalar Quantization (Method 2)

This folder is the **v0.4.0a1 prototype** of Isotope Zero implementing
**Method 2: Int8 Scalar Quantization (SQ8)**. It was scaffolded from the
pure-Python `prototypes/python_v0.1/` baseline and converts 384-dimensional
`float32` embeddings to `int8` + a per-vector scale, evaluating the RAM and
latency/accuracy trade-offs.

## Thesis

Each 384-dim embedding shrinks from `float32` (1,536 bytes) to `int8`
(384 bytes) + a scalar `scale` (4 bytes) — a **4× / 75% reduction** in the
embedding matrix footprint:

$$v_{\text{int8}} = \text{clip}\left(\left\lfloor \frac{v_{\text{f32}}}{\text{scale}} \right\rceil, -128, 127\right), \quad \text{scale} = \frac{\max(|v|)}{127}$$

At 10,000 cards the float32 matrix is 15.36 MB; the int8 matrix is 3.84 MB.
The dequantized dot-product is recovered as:

$$\text{Score}_i = \text{scale}_i \cdot \text{scale}_{\text{query}} \cdot \sum_{j=1}^{d} q_{i,j} \cdot q_{\text{query},j}$$

The architectural bet: 75% RAM savings with >98% rank correlation to the raw
float32 cosine (semantic order preserved), and a measurement of whether int8
dot-product latency beats or trails float32 BLAS.

## The honest open question — int8 latency vs BLAS

This is the subtlety the scope's `q_matrix @ q_query` hides: `np.matmul`/`@`
on int8 arrays is **not BLAS-accelerated** on most platforms (BLAS kernels
are float32/float64 only). Numpy upcasts int8 @ int8 to an **int32 or int64
accumulator** and runs a generic integer loop — which can be *slower* than the
float32 BLAS matmul, the same "premature optimization backfires" trap Method 1
hit (FTS5 pre-filter was slower than the BLAS scan it replaced). The RAM
savings (3.84 MB) and the recall/rank-correlation question (>98%) are the
real deliverables; the latency comparison is reported honestly either way.

## What changed from v0.1

- `isotope_zero/core/quantization.py` (new): pure quantization math —
  `quantize_vector`, `dequantize_vector`, `int8_dot_product` (int32
  accumulator to avoid int8 overflow; rescaled by per-vector scales).
- `isotope_zero/core/store.py`: adds `q_embedding BLOB` + `q_scale REAL`
  columns (idempotent migration via `PRAGMA table_info` guard, mirroring the
  `access_count`/`superseded_by` pattern). `add`/`update` store the int8
  bytes + scale alongside the original float32 BLOB. A parallel
  `_ensure_qvec_cache` builds a cached `(n, d)` int8 matrix + `(n,)` scales.
  New `vector_search_int8(query_vec, k=5)` does the int8 dot-product search
  (same ordering/return type as `vector_search`). New
  `quantized_matrix_nbytes()` / `float32_matrix_nbytes()` for footprint
  reporting. The original float32 `vector_search` and all write/read paths
  are **untouched** — `card.embedding` stays `list[float]`.
- `isotope_zero/eval/adversarial.py`: the stress harness now measures int8
  p99 latency, the float32 vs int8 matrix footprint, and the Spearman rank
  correlation of int8 vs float32 cosine scores at 10k cards.

## Run it

```bash
cd prototypes/quantization_v0.4
# editable install must point HERE:
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m pip install -e .
# quantization self-check:
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python isotope_zero/core/quantization.py
# baseline compatibility:
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m pytest tests/ -q   # 129 passed / 5 skipped
# the Method 2 benchmark (10k scale + 25-proc concurrency):
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m isotope_zero.eval.adversarial
/Users/svanikhansh/Documents/isotope_zero/.venv/bin/python -m isotope_zero.eval.adversarial --json
```

Always use the **absolute** venv path after `cd`-ing in — a relative
`.venv/bin/python` resolves against the new cwd and fails.

## Measured results (captured at freeze)

Adversarial suite, 10,000 cards, 25-worker concurrency. Claims recorded
verbatim beside measured reality; FAIL for any that don't hold.

| dimension | float32 (BLAS) | int8 SQ8 | claim | verdict |
|---|---|---|---|---|
| matrix footprint @10k | 14.65 MB | **3.66 MB** | < 4.0 MB | **PASS** (~4× smaller) |
| rank corr vs f32 cosine | 1.0000 (self) | **0.99997–1.0000** | > 0.98 | **PASS** |
| vector p99 latency | 0.40–0.47 ms | 2.55–3.21 ms | (no claim) | int8 5–8× slower |
| vector_search p99 (baseline) | 0.47 ms | — | < 2.0 ms | PASS |
| sql_lookup p99 | 0.95 ms | — | < 0.80 ms | FAIL (pre-existing) |
| RSS @10k | 745 MB | — | < 200 MB | FAIL (structural) |
| negation / concurrency | 0 / 0 | — | 0 / 0 | PASS |

### Honest verdict — Method 2 passes its two real targets, with one expected negative

**(1) Footprint — PASS.** int8 achieved **3.66 MB** at 10k cards vs float32's
14.65 MB — a clean 4× / 75% reduction, beating the <4.0 MB target. Verified
independently at micro-scale: int8 cache is exactly 0.25× the float32 cache
(3840 vs 15360 bytes).

**(2) Rank correlation — PASS.** int8 cosine ranking preserved the float32
order at **0.99997–1.0000 Spearman** correlation, comfortably above the >0.98
target. Quantization noise does not reorder the ranked list — the top-k from
int8 search matches the float32 top-k (same card order, scores within ~0.001).

**(3) Latency — the honest negative, exactly as the thesis anticipated.**
int8 is **5–8× slower**, not faster: int8 p99 measured 2.55–3.21 ms vs
float32's 0.40–0.47 ms. The cause is exactly the subtlety flagged up front:
numpy's `@` on int8 arrays is **not BLAS-accelerated** (BLAS kernels are
float32/float64 only), so numpy upcasts to an int32 accumulator and runs a
generic integer loop. The thesis's "integer SIMD dot-product latency" framing
does not materialize through numpy on this platform — genuine int8 SIMD
would require a custom kernel (AVX2 `_mm256_dpbusd` / NEON `sdot`), not a
numpy matmul. The scope made no int8 latency claim, so this is reported as
INFO, not a FAIL.

**The trade-off is clear and measured:** int8 buys a 4× memory reduction and
perfect rank preservation at the cost of 5–8× search latency. Whether that
trade is worth it depends on whether the system is RAM-bound or latency-bound
— at 10k cards the float32 matrix is only 14.65 MB (trivial vs the ~745 MB
process RSS), so the RAM savings don't relieve a real bottleneck, and the
latency cost is pure loss *unless* the deployment path adds a real int8 SIMD
kernel. As a pure-Python/numpy prototype, Method 2 is a partial win: the
accuracy/memory claims hold, the latency win does not.

## Siblings

- `prototypes/python_v0.1/` — the pure-Python v0.1.0 baseline this was cut from.
- `prototypes/rust_v0.2/` — the Phase 6 Smart Bridge (NumPy/BLAS vector +
  Rust negation).
- `prototypes/hybrid_v0.3/` — Method 1 (BM25 pre-filter), a committed
  negative result (FTS5 overhead exceeded BLAS; recall dropped to 40%).
