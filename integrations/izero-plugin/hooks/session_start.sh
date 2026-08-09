#!/usr/bin/env bash
# izero lifecycle hook — SessionStart.
# Trivial wrapper: read the hook payload from stdin, pipe it to the testable
# Python entrypoint `izero hook session_start`, falling back to
# `python3 -m isotope_zero.cli.debug` if the `izero` console script isn't on
# PATH (pip-only install). All logic lives in the Python module; this file is
# only here because Claude Code hooks require a `command:` entry.
# Non-blocking: always exits 0 so a hook failure never breaks the editor stream.
# The install helper / CI is responsible for chmod +x on this file.
set -uo pipefail
INPUT=$(cat)
if command -v izero >/dev/null 2>&1; then
  printf '%s' "$INPUT" | izero hook session_start 2>/dev/null || true
elif command -v python3 >/dev/null 2>&1; then
  printf '%s' "$INPUT" | python3 -m isotope_zero.cli.debug hook session_start 2>/dev/null || true
fi
exit 0
