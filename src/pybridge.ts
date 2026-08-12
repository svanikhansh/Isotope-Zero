/**
 * pybridge.ts — Hybrid bridge from the Node `izero` bin to the Python engine.
 *
 * The npm package's primary surface is the React/Ink TUI, but the non-TUI
 * subcommands (add, search, list, serve, consolidate, config, …) are owned by
 * the installed Python CLI (`izero` console script == `isotope_zero.cli.main`).
 * This module spawns that CLI with the Node argv forwarded verbatim, inherits
 * stdio, and forwards the exit code. If Python / the module is absent it fails
 * gracefully with a clear message instead of crashing.
 */

import { spawn } from 'node:child_process';

/**
 * Try to launch a command and forward its exit code. Resolves `null` when the
 * command could not be spawned at all (e.g. ENOENT — interpreter not present),
 * and `number` when it ran (its exit code). Never rejects.
 */
function trySpawn(cmd: string, args: string[]): Promise<number | null> {
  return new Promise<number | null>((resolve) => {
    let launched = false;
    const child = spawn(cmd, args, { stdio: 'inherit' });

    child.on('error', (err: NodeJS.ErrnoException) => {
      // Command failed to launch (ENOENT). Treat as "candidate not available".
      void err;
      resolve(null);
    });

    child.on('spawn', () => {
      launched = true;
    });

    child.on('exit', (code) => {
      if (launched) resolve(code ?? 0);
    });

    child.on('close', (code) => {
      if (launched) resolve(code ?? 0);
    });
  });
}

/** Candidate launches for "the izero Python CLI", in preference order. */
function candidates(args: string[]): Array<[string, string[]]> {
  return [
    // 1. A global `izero` on PATH (the pip-installed console script).
    ['izero', args],
    // 2. python3 -m fallback (covers editable/pip -e installs without a shim).
    ['python3', ['-m', 'isotope_zero.cli.main', ...args]],
    // 3. `python` alias on some systems where python3 is absent.
    ['python', ['-m', 'isotope_zero.cli.main', ...args]],
  ];
}

/**
 * Run a subcommand against the Python CLI. Returns the forwarded exit code.
 * When no Python CLI is reachable, prints a graceful diagnostic and returns a
 * non-zero code (never throws, never dumps a stack trace).
 */
export async function runPythonCommand(args: string[]): Promise<number> {
  for (const [cmd, cmdArgs] of candidates(args)) {
    const code = await trySpawn(cmd, cmdArgs);
    if (code !== null) return code;
  }

  console.error(
    [
      'izero: the requested command needs the Python CLI, but it is not installed.',
      '  Install it with:  pip install isotope-zero',
      '  (or run inside a venv where the Python `izero` CLI is available.)',
      `  Tried: ${candidates(args).map(([c]) => `\`${c}\``).join(', ')}`,
    ].join('\n'),
  );
  return 1;
}
