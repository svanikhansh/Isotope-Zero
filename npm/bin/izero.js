#!/usr/bin/env node
// bin/izero.js — global launcher for isotope-zero.
//
// Guarantees a Python env that imports isotope-zero (uv venv by default,
// system-python fallback), then delegates argv verbatim to the Python `izero`
// CLI (`python -m isotope_zero.cli.debug`). Stdio is inherited so help,
// progress, and exit codes flow straight through — identical UX to the Python
// console script.
//
// Launcher-private flags are prefixed `--izero-` so they never collide with the
// Python CLI's own arguments. Everything else is passed through untouched.

import { spawn } from "node:child_process";
import { resolvePython, pkgVersion, IZERO_HOME, VENV_DIR } from "../lib/ensure-env.js";

const ARGS = process.argv.slice(2);

// --- Launcher-private commands (prefixed --izero- to avoid CLI collisions) ---
if (ARGS[0] === "--izero-version" || ARGS[0] === "-V") {
  console.log(pkgVersion());
  process.exit(0);
}

if (ARGS[0] === "doctor") {
  await doctor();
  process.exit(0);
}

if (ARGS.includes("--izero-self-test")) {
  await selfTest();
  process.exit(0);
}

// --- Delegate everything else to the Python izero CLI ---
let resolved;
try {
  resolved = resolvePython(pkgVersion());
} catch (e) {
  console.error(`\n✗ ${e.message}\n`);
  process.exit(1);
}

const child = spawn(
  resolved.python,
  ["-m", "isotope_zero.cli.debug", ...ARGS],
  { stdio: "inherit" },
);
child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  process.exit(code ?? 1);
});

// --- doctor: print resolved env + version, run a self-test import ---
async function doctor() {
  console.log("isotope-zero (npm launcher) diagnostics");
  console.log("  npm package version :", pkgVersion());
  console.log("  IZERO_HOME          :", IZERO_HOME);
  console.log("  venv dir            :", VENV_DIR);
  let resolved;
  try {
    resolved = resolvePython(pkgVersion());
  } catch (e) {
    console.log("  resolved python     : NONE");
    console.log("  status              : ✗ " + e.message);
    process.exit(1);
  }
  console.log("  resolved python     :", resolved.python, `(${resolved.source})`);
  await selfTest(resolved.python);
}

async function selfTest(pythonBin) {
  const py = pythonBin || (await (async () => {
    try { return resolvePython(pkgVersion()).python; }
    catch { return null; }
  })());
  if (!py) {
    console.log("  status              : ✗ no python env could be resolved");
    process.exit(1);
  }
  const { spawnSync } = await import("node:child_process");
  const r = spawnSync(py, ["-c", "import isotope_zero; print(isotope_zero.__version__)"], {
    stdio: ["ignore", "pipe", "pipe"], encoding: "utf8",
  });
  if (r.status !== 0) {
    console.log("  status              : ✗ import isotope_zero failed");
    process.exit(1);
  }
  const v = r.stdout.trim();
  const ok = v === pkgVersion();
  console.log("  isotope-zero version:", v, ok ? "✓ matches npm pin" : "✗ MISMATCH with npm pin " + pkgVersion());
  process.exit(ok ? 0 : 1);
}
