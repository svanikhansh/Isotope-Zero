#!/usr/bin/env bash
# izero lifecycle hook — PreToolUse:Read.
# Injects a prior-work timeline for the file being read (cards tagged
# file:<relpath> + semantic recall). Trivial wrapper → `izero hook file_read`.
# Non-blocking (PreToolUse:Read always allows; the hook only injects context).
# The install helper / CI is responsible for chmod +x on this file.
set -uo pipefail
INPUT=$(cat)
if command -v izero >/dev/null 2>&1; then
  printf '%s' "$INPUT" | izero hook file_read 2>/dev/null || true
elif command -v python3 >/dev/null 2>&1; then
  printf '%s' "$INPUT" | python3 -m isotope_zero.cli.debug hook file_read 2>/dev/null || true
fi
exit 0
