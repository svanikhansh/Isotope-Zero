#!/usr/bin/env node
// =============================================================================
// izero-cli npm postinstall
// -----------------------------------------------------------------------------
// Provisions a private Python venv inside this package and installs the real
// izero-cli (the Python package) into it. The `bin/izero.js` proxy then spawns
// the venv's `izero` console script, passing argv/stdio/exit-code through.
//
// Where the Python source comes from (first match wins):
//   1. IZERO_PY_SRC  — explicit absolute path to the izero-cli source dir
//                      (the dir containing pyproject.toml).
//   2. Bundled peer  — ../  relative to this npm package (i.e. this package is
//                      shipped alongside tools/izero_cli in a monorepo/git clone).
//   3. IZERO_GIT_URL — a git URL (https/ssh) to install from via pip.
//   4. PyPI          — fall back to the published `izero-cli` package.
//
// This script only writes inside this package's own directory (the venv lives
// at <pkg>/.venv). It never touches $HOME, core prototype source, or any other
// project files. It is idempotent: re-running upgrades.
//
// Environment overrides:
//   IZERO_PY_SRC     path to izero-cli source dir (has pyproject.toml)
//   IZERO_GIT_URL    git URL to pip-install from
//   IZERO_PY_EXTRAS  optional extras, e.g. "onnx"  (default: "")
//   IZERO_NO_VENV    "1" to skip provisioning (assume venv already exists)
//   npm_config_izero_skip_postinstall  "1" to skip entirely (CI opt-out)
// =============================================================================
"use strict";

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

// --- config ------------------------------------------------------------------
const PKG_DIR = path.resolve(__dirname, ".."); // .../tools/izero_cli/npm
const VENV_DIR = path.join(PKG_DIR, ".venv");
const PY_EXTRAS = process.env.IZERO_PY_EXTRAS || "";
const GIT_URL = process.env.IZERO_GIT_URL || "";
const EXPLICIT_SRC = process.env.IZERO_PY_SRC || "";
const NO_VENV = process.env.IZERO_NO_VENV === "1";
const SKIP = process.env.npm_config_izero_skip_postinstall === "1";

// Console helpers (no deps; works on Node 16+).
const isTTY = process.stdout.isTTY;
const c = {
  dim: isTTY ? "\x1b[2m" : "",
  green: isTTY ? "\x1b[32m" : "",
  yellow: isTTY ? "\x1b[33m" : "",
  red: isTTY ? "\x1b[31m" : "",
  reset: isTTY ? "\x1b[0m" : "",
};
const log = (m) => console.log(`[izero-cli] ${m}`);
const warn = (m) => console.warn(`${c.yellow}[izero-cli] ⚠  ${m}${c.reset}`);
const die = (m) => {
  console.error(`${c.red}[izero-cli] ✖  ${m}${c.reset}`);
  process.exit(1);
};

// Strip embedded credentials (user:pass@ or token@) from a URL for display.
// Leaves non-URL strings (paths, package names) unchanged.
function redactUrl(s) {
  if (typeof s !== "string") return s;
  const i = s.indexOf("://");
  if (i < 0) return s;
  const rest = s.slice(i + 3);
  const at = rest.indexOf("@");
  if (at < 0) return s;
  return s.slice(0, i + 3) + rest.slice(at + 1);
}

if (SKIP) {
  log("postinstall skipped via npm_config_izero_skip_postinstall=1 (bin will lazily provision on first run)");
  process.exit(0);
}

// --- locate a Python >= 3.10 ------------------------------------------------
const NEED = { major: 3, minor: 10 };

function pyVersion(python) {
  // Returns [major, minor, patch] or null if the interpreter is unusable.
  const r = spawnSync(python, ["-c", "import sys; print('%d %d %d' % sys.version_info[:3])"], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (r.status !== 0 || r.error) return null;
  const parts = (r.stdout || "").trim().split(/\s+/).map((n) => parseInt(n, 10));
  if (parts.length < 2 || parts.some((n) => Number.isNaN(n))) return null;
  return parts; // [major, minor, patch?]
}

function findPython() {
  const candidates = [
    ...filterEnv(process.env.IZERO_PYTHON ? [process.env.IZERO_PYTHON] : []),
    "python3",
    "python",
  ].filter(Boolean);
  for (const c of candidates) {
    const v = pyVersion(c);
    if (!v) continue;
    if (v[0] > NEED.major || (v[0] === NEED.major && v[1] >= NEED.minor)) {
      return { python: c, version: v };
    }
  }
  return null;
}

function filterEnv(arr) {
  return arr.filter((p) => p && p.length > 0);
}

function hasVenvModule(python) {
  const r = spawnSync(python, ["-c", "import venv"], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  return r.status === 0 && !r.error;
}

const found = findPython();
if (!found) {
  die(
    `python >= ${NEED.major}.${NEED.minor} not found.\n` +
      `Install Python 3.10+ or set IZERO_PYTHON=/path/to/python3, then reinstall.\n` +
      `If npm ran in an environment without python on PATH, install python and run:\n` +
      `  npm rebuild izero-cli   (or:  IZERO_PY_SRC=/path node scripts/postinstall.js)`
  );
}
const { python, version } = found;
log(`Using Python: ${python} (${version.join(".")})`);

if (!hasVenvModule(python)) {
  die(
    `the 'venv' stdlib module is missing for ${python}.\n` +
      `On Debian/Ubuntu:  sudo apt install python3-venv`
  );
}

// --- decide the install source ----------------------------------------------
function resolveSource() {
  // 1. explicit env path
  if (EXPLICIT_SRC) {
    if (!fs.existsSync(path.join(EXPLICIT_SRC, "pyproject.toml"))) {
      die(`IZERO_PY_SRC=${EXPLICIT_SRC} has no pyproject.toml (not izero-cli source).`);
    }
    return { spec: EXPLICIT_SRC, kind: "dir" };
  }
  // 2. bundled peer: this npm pkg is at tools/izero_cli/npm, source is ../
  const peer = path.resolve(PKG_DIR, "..");
  if (fs.existsSync(path.join(peer, "pyproject.toml"))) {
    // Confirm it's actually izero-cli, not some other package.
    try {
      const pp = fs.readFileSync(path.join(peer, "pyproject.toml"), "utf8");
      if (/name\s*=\s*"izero-cli"/.test(pp)) return { spec: peer, kind: "dir" };
    } catch (_) {
      /* ignore */
    }
  }
  // 3. git URL
  if (GIT_URL) return { spec: GIT_URL, kind: "git" };
  // 4. PyPI
  return { spec: "izero-cli", kind: "pypi" };
}

// --- (re)create or reuse the venv -------------------------------------------
const venvPython = path.join(VENV_DIR, "bin", "python");

function venvExists() {
  try {
    return fs.existsSync(venvPython) && fs.accessSync(venvPython, fs.constants.X_OK) === undefined;
  } catch (_) {
    return false;
  }
}

if (NO_VENV) {
  if (!venvExists()) die("IZERO_NO_VENV=1 but no venv found at " + VENV_DIR);
  log(`Reusing existing venv (IZERO_NO_VENV=1): ${VENV_DIR}`);
} else if (venvExists()) {
  log(`Reusing existing venv: ${VENV_DIR}`);
} else {
  log(`Creating venv: ${VENV_DIR}`);
  const r = spawnSync(python, ["-m", "venv", VENV_DIR], { stdio: "inherit" });
  if (r.status !== 0 || r.error) die("venv creation failed: " + (r.error ? r.error.message : "exit " + r.status));
}

// Upgrade pip (best-effort).
{
  const r = spawnSync(venvPython, ["-m", "pip", "install", "--upgrade", "pip"], {
    stdio: "ignore",
  });
  if (r.status !== 0) warn("pip self-upgrade failed (continuing with bundled pip).");
}

// --- install izero-cli -------------------------------------------------------
// Build the pip install spec. A git URL needs the `git+` VCS prefix (a bare
// https URL downloads the HTML page). Extras on a git URL use PEP 508
// direct-URL syntax `izero-cli[onnx] @ git+<url>` — `<url>[onnx]` is rejected.
// `displaySpec` is the credentials-redacted twin for logging/die so an embedded
// token in IZERO_GIT_URL never reaches stdout/logs.
const { spec, kind } = resolveSource();
let installSpec;
let displaySpec;
if (kind === "git") {
  if (PY_EXTRAS) {
    installSpec = `izero-cli[${PY_EXTRAS}] @ git+${spec}`;
    displaySpec = `izero-cli[${PY_EXTRAS}] @ git+${redactUrl(spec)}`;
  } else {
    installSpec = `git+${spec}`;
    displaySpec = `git+${redactUrl(spec)}`;
  }
} else {
  installSpec = PY_EXTRAS ? `${spec}[${PY_EXTRAS}]` : spec;
  displaySpec = installSpec;
}
log(`Installing izero-cli from ${kind}: ${displaySpec}`);

// For non-dir sources (git, pypi) a re-install where the resolved version
// equals the installed one is a silent no-op — `--no-cache-dir` alone does not
// force a refresh. `--force-reinstall` makes `npm rebuild` actually pull a new
// commit on a moving branch. Local-dir installs always reinstall, so they're
// unaffected; we skip --force-reinstall there to keep the common path cheap.
const pipArgs = ["-m", "pip", "install", "--no-cache-dir"];
if (kind !== "dir") pipArgs.push("--force-reinstall", "--no-deps");
pipArgs.push(installSpec);

const r = spawnSync(venvPython, pipArgs, { stdio: "inherit" });
if (r.status !== 0 || r.error) {
  die("pip install failed for: " + displaySpec + (r.error ? " (" + r.error.message + ")" : ""));
}

// --- verify the console script exists ---------------------------------------
const izeroBin = path.join(VENV_DIR, "bin", "izero");
if (!fs.existsSync(izeroBin)) {
  die(`install succeeded but '${izeroBin}' is missing. The proxy will not work.`);
}

// --- smoke ------------------------------------------------------------------
const smoke = spawnSync(izeroBin, ["--help"], { stdio: "ignore" });
if (smoke.status !== 0) {
  warn("izero --help returned non-zero; the install may be incomplete.");
} else {
  log(`Smoke test: izero --help ${c.green}✓${c.reset}`);
}

log(`${c.green}✓ izero-cli ready.${c.reset} venv at ${VENV_DIR}`);
log(`Run: ${c.dim}npx izero --help${c.reset}  (or  ${c.dim}izero --help${c.reset} if installed globally)`);
