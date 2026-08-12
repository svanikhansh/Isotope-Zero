#!/usr/bin/env node
/**
 * izero.js — npm bin entry for `isotope-zero`.
 *
 * Compiled output lands in `dist/` (`npm run build`); this thin shim imports the
 * dispatcher from there. All routing lives in `src/cli.ts`:
 *   - no args / `tui`            → React/Ink TUI
 *   - `--version` / `-V`/version → package version
 *   - `--help` / `-h`/help       → usage
 *   - anything else              → forwarded to the Python CLI (hybrid bridge)
 */
import { main } from '../dist/cli.js';

main()
  .then((code) => {
    process.exitCode = code;
  })
  .catch((err) => {
    console.error('izero: fatal error:', err?.message ?? err);
    process.exit(1);
  });
