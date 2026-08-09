---
name: decay-review
description: Review decayed memory cards and run a consolidation sweep against the local isotope_zero store. Use periodically to merge near-duplicate cards, prune zero-recall decayed entries, and reclaim context tokens.
---

# izero:decay-review

Review decay and run one consolidation sweep over the **local** isotope_zero
store. Everything stays in `~/.isotope_zero/isotope_zero.db` — no network, no
API key, fully offline.

## Execution

### Step 1: Read metrics

Call the `get_metrics` MCP tool (no arguments). Report the pre-sweep state:

```
## izero decay-review (pre-sweep)

Store:      <db_size_human> (<db_size_bytes> bytes)
Cards:      <card_count>
Tokens saved vs raw: <cumulative_tokens_saved_vs_raw>
```

This is the baseline; the consolidation report in Step 3 will be compared
against it. If `card_count` is small (say, under ~20 cards), note that there
may be little to consolidate — proceed anyway, the sweep is cheap and safe.

### Step 2: Show decay candidates

From the pre-sweep metrics, flag the decay candidates the consolidation pass
will act on. The store tracks per-card vitality (recall frequency recency);
cards at zero recall since the last sweep are the prune candidates. State the
intent clearly before mutating anything:

```
Decay candidates: zero-recall cards flagged for pruning.
Near-duplicate cards flagged for evidence-merging (no fact loss).
```

If you cannot enumerate specific card ids from `get_metrics` alone, that is
expected — the sweep decides internally. Do not guess card ids.

### Step 3: Run consolidation

Call the `run_consolidation` MCP tool (no arguments). This deduplicates
near-identical cards (merging evidence without losing facts), prunes decayed
zero-recall cards, and reclaims context tokens. It runs off the hot path in a
single atomic transaction (WAL mode), so it is safe alongside active
reads/writes.

### Step 4: Report the sweep

`run_consolidation` returns:
- `merged_cards` — near-duplicates merged (evidence folded in)
- `decayed_cards` — zero-recall cards pruned
- `survivors` — cards remaining after the sweep
- `tokens_before` / `tokens_after` — context-token footprint
- `tokens_reclaimed` — net tokens reclaimed (`tokens_before - tokens_after`)
- `latency_ms` — sweep wall time
- `pruned_mean_vitality` — mean vitality of pruned cards (sanity check)

Present the outcome:

```
## izero decay-review (post-sweep)

Merged:          <merged_cards> near-duplicate cards
Decayed/pruned:  <decayed_cards> zero-recall cards
Survivors:       <survivors> cards
Tokens:          <tokens_before> -> <tokens_after>  (reclaimed <tokens_reclaimed>)
Sweep latency:   <latency_ms> ms
Pruned mean vitality: <pruned_mean_vitality>
```

### Step 5: Suggest follow-up

- If `decayed_cards` is high and `pruned_mean_vitality` is near zero, the
  sweep behaved as expected. Suggest running `/izero:stats` to confirm the
  reclaimed token delta.
- If `merged_cards` is high, suggest `/izero:recall` on a likely-affected
  topic to confirm the merged evidence is retrievable.
- If nothing changed (`merged_cards` == 0 and `decayed_cards` == 0), say so —
  the store was already compact.
