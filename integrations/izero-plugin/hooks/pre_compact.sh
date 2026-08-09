#!/usr/bin/env bash
# izero lifecycle hook — PreCompact.
# Preserves context about to be lost to compaction: extracts decisions from
# the compacted transcript and writes them as cards before the context window
# shrinks. Trivial wrapper → `izero hook pre_compact`.
# Non-blocking: always exits 0 so a hook failure never breaks the editor stream.
# The install helper / CI is responsible for chmod +x on this file.
set -uo pipefail
INPUT=$(cat)
if command -v izero >/dev/null 2>&1; then
  printf '%s' "$INPUT" | izero hook pre_compact 2>/dev/null || true
elif command -v python3 >/dev/null 2>&1; then
  printf '%s' "$INPUT" | python3 -m isotope_zero.cli.debug hook pre_compact 2>/dev/null || true
fi
exit 0
