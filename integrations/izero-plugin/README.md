# izero editor plugin

Local-first cognitive memory for Claude Code, Cursor, and Codex.

**Memory never leaves your machine.** `izero` stores everything in
`~/.isotope_zero/isotope_zero.db` — a local SQLite database with offline ONNX
embeddings, Ebbinghaus-style decay, consolidation, and an entity graph. There is no
cloud endpoint, no API key, and no network traffic from any hook or command. It works
fully offline.

This is the capture surface: a local-stdio MCP server (`izero-mcp`) plus lifecycle hooks
(file-context injection, terminal-error capture, session-summary preservation) and
slash-command skills — wired into your editor so memory accumulates **automatically**,
not just on explicit recall.

## Install

```bash
# Claude Code (default)
izero plugin install

# Cursor / Codex
izero plugin install --editor cursor
izero plugin install --editor codex

# Preview what it would do, without writing
izero plugin install --dry-run
```

`izero plugin install` copies this bundle into your editor's plugin directory
(`~/.claude/plugins/izero`, `~/.cursor/plugins/izero`, or `~/.codex/plugins/izero`) and
rewrites the bundled `.mcp.json` so the MCP `command` resolves correctly on your machine
(`izero-mcp` if it's on PATH, otherwise `python -m isotope_zero.mcp.server` using your
interpreter). No download, no marketplace — the bundle ships inside the `isotope-zero`
package.

If you don't have `izero` installed yet:

```bash
pip install isotope-zero      # or: uv pip install isotope-zero
```

## Smoke test

In an editor session after install:

```
/izero:remember decided to use uv-first python resolution
/izero:recall resolution
```

The first command stores the decision in `~/.isotope_zero/`; the second retrieves it via
local semantic search. Confirm the store grew with `izero stats`, and confirm **no**
network activity (e.g. `lsof` / Activity Monitor) — there should be none.

The lifecycle hooks also run automatically: open a file you've worked on before and izero
injects a prior-work timeline; run a failing bash command and izero captures the error as
an anti-pattern for future recall.

## What's in the bundle

```
plugin.json          # editor manifest (id/name/version/contextFileName=AGENTS.md)
.mcp.json            # LOCAL-stdio MCP config → izero-mcp (no url, no key)
hooks.json           # lifecycle event → bash-wrapper command map
hooks/*.sh           # trivial wrappers → `izero hook <event>` (testable python core)
skills/*/SKILL.md    # /izero:remember | recall | forget | peek | stats | decay-review
AGENTS.md            # context file the agent reads on session start
```

All hook logic lives in one Python module (`isotope_zero.integrations.hooks`) behind the
`izero hook <event>` subcommand, so the bash wrappers stay trivial and the logic is fully
unit-testable without a live editor.

## Privacy, explicitly

- No `url`, no `Authorization`, no `api_key` in `.mcp.json` or any manifest.
- No `requests` / `urllib` / `httpx` / `openai` imports anywhere in the integration tree.
- Memory persists at `~/.isotope_zero/` and is never transmitted.

Verify it yourself:

```bash
grep -rniE 'http|requests\.|urllib|httpx|api_key|openai' integrations/izero-plugin/ src/isotope_zero/integrations/
# expected: only comments/docstrings explaining the absence of these
```
