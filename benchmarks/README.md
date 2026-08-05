# isotope_zero benchmarks

Two scripts measure the memory engine's memory footprint and read-path
latency/recall. Run both with the test venv interpreter at
`/tmp/iz_test_venv/bin/python3.14`:

```bash
# RSS floor (cold-start, at 10k cards) + tracemalloc zero-leak over 100k
# add-then-delete cycles. Exits 0 on pass, non-zero on leak/RSS regression.
/tmp/iz_test_venv/bin/python3.14 \
  /Users/svanikhansh/Documents/isotope_zero/benchmarks/profile_rss_and_allocs.py

# vector_search + hybrid_search p50/p95/p99 latency at N=100/1000/10000
# (dim=384) and recall@5 vs an exact numpy cosine baseline. Exits 0 on pass.
/tmp/iz_test_venv/bin/python3.14 \
  /Users/svanikhansh/Documents/isotope_zero/benchmarks/benchmark_latency_recall.py
```

`profile_rss_and_allocs.py` auto-installs `psutil` into the venv on first run
if it is missing. Both scripts add `prototypes/synthesis_v1.0` to `sys.path`
so they run from any cwd. Seeds use direct bulk SQL inserts (bypassing the
O(n^2) graph auto-linker in `store.add()`) so the 10k-card runs stay under
the 60s budget; the store's `invalidate_vector_cache()` is called after
seeding so the lazy numpy matrix rebuilds on the first search.
