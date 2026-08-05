#!/usr/bin/env node
// =============================================================================
// izero-cli — npm bin proxy launcher
// -----------------------------------------------------------------------------
// Spawns the real `izero` Python console script living in the venv that
// scripts/postinstall.js provisions next to this package, and forwards argv,
// stdio, exit code, and signals through. This file has zero Node deps so the
// shim works on a bare Node >=16 install with no node_modules.
//
// If the venv has not been provisioned yet (e.g. postinstall was skipped via
// --ignore-scripts, or ran in an env without python), we attempt to provision
// it lazily on first run so `npx izero` still works after a manual `npm i
// --ignore-scripts`. If that also fails, we print actionable guidance.
// =============================================================================
"use strict";

const fs = require("fs");
const path = require("path");
const { spawn, spawnSync } = require("child_process");

const PKG_DIR = path.resolve(__dirname, ".."); // .../tools/izero_cli/npm
const VENV_DIR = path.join(PKG_DIR, ".venv");
const IZERO_BIN = path.join(VENV_DIR, "bin", "izero");
const POSTINSTALL = path.join(PKG_DIR, "scripts", "postinstall.js");

function fail(msg) {
  process.stderr.write(`izero: ${msg}\n`);
  process.exit(127); // 127 = "command not found" convention
}

function venvReady() {
  try {
    return fs.existsSync(IZERO_BIN) && fs.accessSync(IZERO_BIN, fs.constants.X_OK) === undefined;
  } catch (_) {
    return false;
  }
}

// Lazily provision the venv on first use if postinstall was skipped.
function ensureVenv() {
  if (venvReady()) return true;
  if (!fs.existsSync(POSTINSTALL)) return false;
  process.stderr.write("izero: venv not provisioned yet (postinstall was likely skipped); provisioning now…\n");
  const r = spawnSync(process.execPath, [POSTINSTALL], { stdio: "inherit" });
  return r.status === 0 && venvReady();
}

if (!ensureVenv()) {
  fail(
    `could not find a provisioned izero-cli venv at ${VENV_DIR}\n` +
      `Run the postinstall manually:\n` +
      `    node ${POSTINSTALL}\n` +
      `or set IZERO_PY_SRC=/path/to/izero-cli-source and retry.`
  );
}

// Spawn the real CLI, forwarding argv and inheriting stdio.
const child = spawn(IZERO_BIN, process.argv.slice(2), {
  stdio: "inherit",
  windowsHide: true,
});

// Propagate signals we might receive to the child so Ctrl-C / SIGTERM behave
// correctly. The child's own exit becomes ours.
function forward(sig) {
  if (child.pid) {
    try {
      child.kill(sig);
    } catch (_) {
      /* child may have already exited */
    }
  }
}
["SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT"].forEach((sig) => process.on(sig, () => forward(sig)));

child.on("error", (err) => {
  fail(`failed to launch izero (${err.message})`);
});

child.on("exit", (code, signal) => {
  if (signal) {
    // Re-emit the same signal so the caller sees the child was killed.
    process.kill(process.pid, signal);
  } else {
    process.exit(code == null ? 1 : code);
  }
});
