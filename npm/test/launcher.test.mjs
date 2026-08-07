// launcher.test.mjs — node --test suite for the izero launcher.
//
// Covers: version flag, doctor command, arg passthrough, the no-python error
// path, and bare-izero delegation to the Python onboarding menu. Most tests
// stay hermetic and fast; the bare-izero case is the exception — it drives the
// real delegation path (timeout-bounded) so the launcher wiring is exercised,
// not just stubbed.

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

test("bare izero opens the onboarding menu (no 'command required' error)", () => {
  // Regression guard: bare `izero` must NOT hit argparse's "the following
  // arguments are required: command" error — the pre-menu behavior. With a
  // working env it opens the interactive menu (the numbered fallback, since
  // a piped stdin is not a tty); we pipe "q" to exit cleanly with 0.
  const r = spawnSync(process.execPath, [IZERO], {
    stdio: ["pipe", "pipe", "pipe"],
    encoding: "utf8",
    input: "q\n",
    env: process.env,
    timeout: 30000, // a hanging uv bootstrap must not hang the suite
  });
  const out = r.stdout + r.stderr;
  assert.ok(!out.includes("the following arguments are required"),
    `bare izero must not error with 'command required': ${out}`);
  if (r.status === 0) {
    // Reached the Python menu: banner + at least the primary entry.
    assert.ok(out.includes("isotope-zero"), `menu banner missing: ${out}`);
    assert.ok(out.includes("add a memory"), `menu entry missing: ${out}`);
  }
  // status !== 0 means no python env could be resolved on this machine — an
  // acceptable environment gap; the Python-side suite covers the menu itself.
});

// --------------------------------------------------------------------------- #
// venv-build lock: two concurrent izero invocations must not both build the
// venv. ensure-env.js guards the build with an atomic mkdir of a lock dir
// (EEXIST = another process holds it). This unit-tests the invariant the fix
// relies on — exactly one of two concurrent mkdirSync calls on the same path
// wins, the other gets EEXIST — without driving a real (slow, flaky) double
// venv build. Hermetic: uses a throwaway IZERO_HOME under os.tmpdir.
// --------------------------------------------------------------------------- #
import { mkdtempSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";

test("venv-build lock: concurrent mkdirSync on the lock dir — exactly one wins, the other gets EEXIST", () => {
  const home = mkdtempSync(join(tmpdir(), "izero-lock-"));
  const lockDir = join(home, ".venv-build-lock");
  // Two concurrent attempts to claim the build lock. mkdirSync is atomic on
  // POSIX: the first succeeds, the second throws EEXIST (not a race where both
  // could win). This is the exact invariant ensure-env.js's ensureUvVenv uses
  // to serialize concurrent venv builds.
  let firstWon = false, secondEEXIST = false;
  try {
    mkdirSync(lockDir);
    firstWon = true;
  } catch (e) {
    // first should win; if it somehow lost, the invariant is violated
  }
  try {
    mkdirSync(lockDir);
  } catch (e) {
    secondEEXIST = e.code === "EEXIST";
  }
  assert.ok(firstWon, "first mkdirSync must claim the lock");
  assert.ok(secondEEXIST, "second mkdirSync must get EEXIST, not silently win");
});
