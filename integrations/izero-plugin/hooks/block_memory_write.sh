#!/usr/bin/env bash
# izero lifecycle hook — PreToolUse:Write|Edit|MultiEdit (write guard).
# Blocks direct writes into the izero store directory, redirecting to the
# `add_memory` MCP tool.
#
# UNLIKE the other hooks, this one PROPAGATES the python exit code: the python
# `block_memory_write` handler exits 2 to DENY the write (Claude Code's block
# signal) or 0 to allow. Swallowing exit 2 (e.g. an `if cmd; then exit 0; else
# fallback` pattern) would turn a deny into a non-blocking error and let the
# write through — so this wrapper MUST forward the exact code.
#
# Entry-point resolution mirrors the other wrappers (izero console script, else
# python3 -m), but the chosen entry's exit code is forwarded verbatim. A
# *missing* entrypoint (PATH has neither `izero` nor `python3` with the module)
# defaults to ALLOW (exit 0) so a broken izero install never wedges the editor
# — the write-guard is defense-in-depth, not a hard security boundary, and a
# hard failure here would block all edits until izero is fixed.
set -uo pipefail
INPUT=$(cat)
if command -v izero >/dev/null 2>&1; then
  printf '%s' "$INPUT" | izero hook block_memory_write 2>/dev/null
  exit $?
elif command -v python3 >/dev/null 2>&1; then
  printf '%s' "$INPUT" | python3 -m isotope_zero.cli.debug hook block_memory_write 2>/dev/null
  exit $?
fi
# No entrypoint available — fail open (allow) rather than wedge the editor.
exit 0
