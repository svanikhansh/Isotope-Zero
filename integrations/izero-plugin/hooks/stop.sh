#!/usr/bin/env bash
# izero lifecycle hook — Stop.
# Captures a local-heuristic session summary (decisions/preferences/anti-
# patterns) to the store on session end. Trivial wrapper → `izero hook stop`.
# Non-blocking: always exits 0 so a hook failure never breaks the editor stream.
# The install helper / CI is responsible for chmod +x on this file.
set -uo pipefail
INPUT=$(cat)
if command -v izero >/dev/null 2>&1; then
  printf '%s' "$INPUT" | izero hook stop 2>/dev/null || true
elif command -v python3 >/dev/null 2>&1; then
  printf '%s' "$INPUT" | python3 -m isotope_zero.cli.debug hook stop 2>/dev/null || true
fi
exit 0
