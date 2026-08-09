# izero — local cognitive memory for this agent

`izero` is a **local-first** memory store. Every memory you write or read stays in
`~/.isotope_zero/isotope_zero.db` on this machine. There is no network call, no API key,
and no cloud — recall, storage, decay, and consolidation all run against a local SQLite
store with offline ONNX embeddings. Nothing you remember ever leaves the user's computer.

## When to use izero

Use the `/izero:*` slash commands to ground your answers in prior work and to persist
decisions so future sessions (yours or a teammate's) don't re-derive them:

- `/izero:remember <fact>` — store a decision, preference, anti-pattern, or learning.
  Use it when the user says "remember this", "save this", "note that", or when you've
  reached a non-obvious conclusion worth keeping (an architecture decision, a workaround
  that worked, a constraint that bit you).
- `/izero:recall <query>` — semantic search over prior memories. Use it before answering
  a question that prior work might already cover ("did we decide how to do X?", "what
  did we find about Y?"), so you build on what's stored rather than re-discovering it.
- `/izero:peek <query|id>` — compact id/fact lookup (small token budget).
- `/izero:forget <id>` — delete a specific memory by id.
- `/izero:stats` — store size, card count, embedding mode, tokens saved.
- `/izero:decay-review` — review decayed cards and run a consolidation sweep (merge
  near-duplicates, prune low-vitality ones).

## Automatic capture (lifecycle hooks)

Beyond the explicit slash commands, izero captures context automatically through editor
lifecycle hooks — you don't need to do anything for these to run:

- **File read** (`PreToolUse:Read`) — when you open a file you've worked on before, izero
  injects a prior-work timeline so you see what was previously decided/learned about that
  file before you touch it again.
- **Terminal errors** (`PostToolUse:Bash`) — when a bash command fails with a traceback,
  izero captures it as an `anti_pattern` memory and, on recurrence, surfaces the prior fix.
  This is gated (skips git operations, dedups repeated errors, caps at 3 captures per
  signature) so it never spams the store.
- **Session summary** (`Stop` / `PreCompact`) — at session end and before context compaction,
  izero runs a *local* heuristic extractor over the transcript (no LLM, no network) and
  persists decisions/preferences it finds, tagged with the session id and any derivable
  file/branch provenance. This is what keeps context alive across compaction boundaries.

All of the above are local: the hooks call the `izero` CLI / MCP server, which reads and
writes the local SQLite store only. Verify any time with `izero stats`.

## Privacy contract

- No outbound network traffic from any hook or command. Memory never leaves `~/.isotope_zero/`.
- No API key is required or read.
- Works fully offline.
