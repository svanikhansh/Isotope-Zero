#!/usr/bin/env bash
# izero lifecycle hook — PostToolUse:Bash.
# Scans tool_output for tracebacks; on a real error, auto-captures an
# anti_pattern card (deduped by error signature) and injects a "prior fix"
# rubric. Trivial wrapper → `izero hook bash_output`.
# Non-blocking: always exits 0 so a hook failure never breaks the editor stream.
# The install helper / CI is responsible for chmod +x on this file.
set -uo pipefail
INPUT=$(cat)
if command -v izero >/dev/null 2>&1; then
  printf '%s' "$INPUT" | izero hook bash_output 2>/dev/null || true
elif command -v python3 >/dev/null 2>&1; then
  printf '%s' "$INPUT" | python3 -m isotope_zero.cli.debug hook bash_output 2>/dev/null || true
fi
exit 0
