---
name: recall
description: Ground an answer in prior work — find what was previously decided, tried, or preferred in the local isotope_zero memory. Use before answering a question that prior sessions may have already settled.
---

# izero:recall

Search the **local** isotope_zero memory and present hits fact-first. Everything
stays in `~/.isotope_zero/isotope_zero.db` — no network, no API key, fully
offline.

## Execution

### Step 1: Parse the query

The user provides a search query: `/izero:recall <query>`

If no query is provided, ask: "What should I recall?"

Rephrase a question ("how do we do auth?") into its noun form ("authentication
middleware") — the router matches on keywords and semantics, not phrasing.

### Step 2: Search

Call the `query_memory` MCP tool with:
- `query="<the query>"`
- `token_budget=300` (the default)

The store routes the query SQL-first for explicit state (tag/fact/evidence
lookups) and falls back to vector similarity for fuzzy/semantic matches, so a
single call covers both paths — no need to issue parallel queries.

### Step 3: Present hits

`query_memory` returns `hits` (each with `id`, `fact`, `evidence`, `tags`,
`score`, `route`, `tokens`), plus `route_used`, `tokens_used`, and
`tokens_saved_vs_raw`. Present hits **fact-first**:

```
## izero recall: "<query>" (<N> hits, route: <route_used>)

1. <fact> [tags: <tags>] (score: <score>) [id: <id>]
2. <fact> [tags: <tags>] (score: <score>) [id: <id>]
```

Format: the bare fact leads, then its tags, then the score and id for
follow-up. Truncate each fact to ~90 chars. Do not dump `evidence` inline —
mention it is available if the user wants detail.

### Step 4: Tie back

If the hits answer the user's question, lead with the fact in your reply and
cite the memory id (`[izero:<id>]`) so the user can verify or `/izero:forget`
it. If no hits returned, say so explicitly rather than guessing — do not fall
back to an unsupported answer when recall was the point.
