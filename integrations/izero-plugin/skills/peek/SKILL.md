---
name: peek
description: Quick, compact memory id or fact lookup against the local isotope_zero store. Use for resolving an [izero:id] citation, checking whether a decision was already recorded, or browsing hits without full recall detail.
---

# izero:peek

A lighter, compact search of the **local** isotope_zero memory. Everything
stays in `~/.isotope_zero/isotope_zero.db` — no network, no API key, fully
offline.

## Execution

### Step 1: Parse the query

The user provides a lookup: `/izero:peek <query or id>`

If no input is provided, ask: "What should I peek at?"

**Id detection:** if the argument matches `^[a-f0-9-]+$` (a bare hex id or
UUID), treat it as a direct memory-id lookup rather than a semantic search.

### Step 2: Search (compact)

Call the `query_memory` MCP tool with:
- `query="<the query or id>"`
- `token_budget=100`  (compact — peek is for quick confirmation, not deep
  grounding)

A single call covers both the SQL (explicit id/tag/fact) and vector (semantic)
routes. The smaller budget caps how many hits come back, which keeps peek fast
and narrow.

### Step 3: Display

`query_memory` returns `hits` (each with `id`, `fact`, `evidence`, `tags`,
`score`) plus `route_used` and `tokens_used`. Show compact one-liners:

```
## izero peek: "<query>" (<N> hits, route: <route_used>)

1. <fact, ~80 chars> [izero:<id>] (score: <score>)
2. <fact, ~80 chars> [izero:<id>] (score: <score>)
```

Format: fact first, then the id citation (so the user can `/izero:forget` or
`/izero:recall` it next), then the score. Do not show `evidence` inline —
that is what `/izero:recall` is for.

If no hits returned:

```
No memories matching "<query>".
```

If an id was given and matched, the single hit's `fact` is the answer; if it
did not match, say so rather than leaving the citation unresolved.
