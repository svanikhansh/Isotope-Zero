---
name: remember
description: Store a fact, decision, preference, or learning into the local isotope_zero memory. Use when the user says "remember this", "save this", "note that", or asks to record a decision/preference/learning/convention.
---

# izero:remember

Persist a memory into the **local** isotope_zero store. Everything stays in
`~/.isotope_zero/isotope_zero.db` — no network, no API key, fully offline.

## Execution

### Step 1: Extract the content

The user provides the content as an argument: `/izero:remember <text>`

If no text was provided, ask: "What should I remember?"

If the user's phrasing is a directive ("remember that we use JWT"), capture
the underlying fact as a concise standalone statement ("We use JWT for
authentication") rather than verbatim instruction prose.

### Step 2: Classify a type tag

Pick **one** `type:*` tag based on signal words in the content:

| Signal in content | Tag |
|---|---|
| "we decided", "always use", "never", "the rule is" | `type:decision` |
| "doesn't work because", "don't try", "fails when", "caused by" | `type:anti_pattern` |
| "I prefer", "use X instead of Y", "I like" | `type:user_preference` |

If none of the signal words match, omit the type tag — the store's own
auto-extraction will still tag the card. Do not invent a tag that does not
match the content.

### Step 3: Store

Call the `add_memory` MCP tool with:
- `content="<the concise fact statement>"`
- `tags=["type:<classified_type>", "source:remember"]`

`source:remember` stamps provenance so the recall skills and capture hooks can
trace the card back to this slash command. The store merges these onto its
own auto-extracted tags (order-preserving, deduped) — caller tags never drop
auto-extracted ones.

### Step 4: Confirm

`add_memory` returns `memory_id`, `action` (`ADD` or `UPDATE`), `confidence`,
`fact`, and `tags`. Report the outcome:

```
Remembered (action: <action>, confidence: <confidence>):
<fact, first ~80 chars>...
Memory ID: <memory_id>
```

If `action` was `UPDATE`, note that an existing card was overwritten in place
rather than a new card created.
