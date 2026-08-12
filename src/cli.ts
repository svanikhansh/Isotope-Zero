/**
 * cli.ts — Node argument dispatcher for the `izero` npm package.
 *
 * Compiled to `dist/cli.js` and imported by `bin/izero.js`. Routing:
 *   - no args / `tui`        → launch the React/Ink TUI
 *   - `--version` / `-V`     → print package version
 *   - `--help` / `-h` / help → print usage
 *   - anything else          → forward to the Python CLI (hybrid bridge)
 */

import { runTUI } from './tui/App.js';
import { runPythonCommand } from './pybridge.js';

const HELP = `isotope-zero — sub-millisecond, local-first cognitive memory layer

Usage: izero <command> [options]

TUI (default):
  izero                       Launch the terminal UI
  izero tui                   Launch the terminal UI

Help & version:
  izero help                  Show this help
  izero --version / -V        Print the installed version

Python engine commands (delegated to the \`izero\` Python CLI):
  add, recall, search, list, get, forget, touch, stats, tags, inspect,
  consolidate, dashboard, serve, menu, doctor, hook, plugin, config

Options:
  --theme <name>    TUI color theme (ocean-blue, forest-green, sunset-orange, monochrome)
  --json            Render engine command output as JSON
`;

export async function main(): Promise<number> {
  const argv = process.argv.slice(2);
  const first = argv[0];

  if (first === undefined || first === 'tui') {
    await runTUI();
    return 0;
  }

  if (first === '--version' || first === '-V' || first === 'version') {
    console.log(await packageVersion());
    return 0;
  }

  if (first === '--help' || first === '-h' || first === 'help') {
    console.log(HELP);
    return 0;
  }

  // Any other command → Python CLI bridge (hybrid semantics).
  return runPythonCommand(argv);
}

async function packageVersion(): Promise<string> {
  try {
    const { createRequire } = await import('node:module');
    const req = createRequire(import.meta.url);
    const pkg = req('../package.json') as { version?: string };
    return pkg.version ?? '0.0.0';
  } catch {
    return '0.0.0';
  }
}

export default main;