---
name: stats
description: Show the local isotope_zero store size, card count, embedding mode, and cumulative tokens saved versus replaying raw history. Use to check memory health or how much context the store is reclaiming.
---

# izero:stats

Report the health of the **local** isotope_zero store. Everything stays in
`~/.isotope_zero/isotope_zero.db` — no network, no API key, fully offline.

## Execution

### Step 1: Read metrics

Call the `get_metrics` MCP tool (takes no arguments).

### Step 2: Report

`get_metrics` returns:
- `db_size_bytes` and `db_size_human` — on-disk store size
- `card_count` — number of stored memory cards
- `embedding_is_real` — `true` if a real embedding model (e.g. ONNX) is in
  use, `false` if the deterministic pseudo-embedding fallback is active
- `cumulative_tokens_saved_vs_raw` — tokens reclaimed from context versus
  replaying raw conversation history
- `cumulative_raw_history_tokens` — the raw-history token baseline

Present as a compact summary:

```
## izero stats

Store:      ~/.isotope_zero/isotope_zero.db
DB size:    <db_size_human> (<db_size_bytes> bytes)
Cards:      <card_count>
Embedding:  <real | pseudo fallback>   (embedding_is_real=<...>)
Tokens saved vs raw history: <cumulative_tokens_saved_vs_raw>
Raw history baseline:       <cumulative_raw_history_tokens>
```

### Step 3: Interpret

- If `embedding_is_real` is `false`, mention that the store is running in the
  offline deterministic fallback (no onnxruntime) — fine for recall, but
  semantic scores are approximate.
- If `cumulative_tokens_saved_vs_raw` is low relative to
  `cumulative_raw_history_tokens`, suggest `/izero:decay-review` to reclaim
  context tokens via a consolidation sweep.
