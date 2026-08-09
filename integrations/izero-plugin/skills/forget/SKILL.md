---
name: forget
description: Remove a specific memory from the local isotope_zero store by its memory id. Use when a recorded fact is outdated, wrong, or should no longer influence future sessions.
---

# izero:forget

Delete one memory card from the **local** isotope_zero store by id. Everything
stays in `~/.isotope_zero/isotope_zero.db` — no network, no API key, fully
offline.

## Execution

### Step 1: Parse the id

The user provides a memory id: `/izero:forget <memory_id>`

If no id is provided, ask: "Which memory should I forget? Give me the memory id
(run `/izero:recall <query>` first if you need to find it)."

If the user gives a **search query** instead of an id, do not delete by guess.
Tell them to run `/izero:recall <query>` to find the exact id, then re-invoke
`/izero:forget <id>`. This skill only deletes by explicit id — see Step 3.

### Step 2: Confirm

This is destructive. Before deleting, echo the target back and ask for
confirmation:

```
Delete memory <memory_id>? [y/N]
```

If you do not already know the card's fact (e.g. from a prior recall), you may
call `query_memory` with the id as the query to show what will be removed.
**Never delete without confirmation.**

### Step 3: Delete

On confirmation, call the `delete_memory` MCP tool with:
- `memory_id="<the id>"`

### Step 4: Report

`delete_memory` returns `{"memory_id": "<id>", "deleted": <bool>}`. Report:

```
Forgot memory <memory_id>.
```

If `deleted` is `false`, the id did not match a stored card — say so rather
than claiming success:

```
No memory with id <memory_id> found (nothing deleted).
```
