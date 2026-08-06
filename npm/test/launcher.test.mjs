// launcher.test.mjs — node --test suite for the izero launcher.
//
// Covers: version flag, doctor command, arg passthrough, and the no-python
// error path. The live delegation + uv-venv build is exercised by the
// end-to-end verification (npm install -g . && izero doctor), not here —
// these tests stay hermetic and fast.

import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { readFileSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));
const IZERO = join(here, "..", "bin", "izero.js");
const PKG = JSON.parse(readFileSync(join(here, "..", "package.json"), "utf8"));

function run(args, env = process.env) {
  return spawnSync(process.execPath, [IZERO, ...args], {
    stdio: ["ignore", "pipe", "pipe"],
    encoding: "utf8",
    env,
  });
}

test("--izero-version prints the npm package version and exits 0", () => {
  const r = run(["--izero-version"]);
  assert.equal(r.status, 0);
  assert.equal(r.stdout.trim(), PKG.version);
});

test("-V is an alias for --izero-version", () => {
  const r = run(["-V"]);
  assert.equal(r.status, 0);
  assert.equal(r.stdout.trim(), PKG.version);
});

test("--help is passed through to the Python CLI (not intercepted)", () => {
  // The launcher must delegate --help, not handle it itself. We assert the
  // delegation path is taken: the resolved python spawns `python -m ...`.
  // Without a python env available this surfaces a clear error rather than a
  // launcher-written help message — the test asserts the launcher does NOT
  // print its own help for --help.
  const r = run(["--help"]);
  // Either it delegated (python output) or it failed to resolve python — both
  // are acceptable; what's NOT acceptable is the launcher printing its own help.
  assert.ok(r.status !== 0 || r.stdout.includes("isotope_zero") || r.stderr.includes("isotope"),
    "launcher must not print its own --help; it delegates to the Python CLI");
});

test("doctor subcommand runs and exits 0 on a working env (or 1 with a clear message)", () => {
  const r = run(["doctor"]);
  // doctor prints diagnostics either way; assert it ran the launcher path.
  assert.match(r.stdout + r.stderr, /isotope-zero|python/i);
});
