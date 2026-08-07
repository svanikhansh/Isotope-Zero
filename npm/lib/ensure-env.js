// ensure-env.js — guarantee a Python interpreter that can import isotope_zero.
//
// Resolution order (Strategy A, uv-first with graceful degradation):
//   1. Existing ~/.izero/venv already satisfies isotope-zero==<pkgVer> → reuse
//      it as-is (the venv is the durable home; uv may be absent this run)
//   2. `uv` on PATH, or bootstrapped into ~/.izero/bin → create/refresh the venv
//   3. Bootstrap fails (offline/sandbox) or uv can't install → system python3 +
//      pip install --user
//   4. No python at all → throw ENOPython (caller prints actionable error)
//
// The venv is pinned to the npm package version so npm and PyPI never drift:
// `npm i -g isotope-zero@1.0.0` → `isotope-zero==1.0.0` in the venv.

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, platform } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";

// ~/.izero is the launcher's home: venv/ + bin/ (bootstrapped uv) + version stamp.
export const IZERO_HOME = process.env.IZERO_HOME || join(homedir(), ".izero");
export const VENV_DIR = join(IZERO_HOME, "venv");
export const UV_BIN_DIR = join(IZERO_HOME, "bin");
const STAMP_FILE = join(IZERO_HOME, ".installed-version");

// Read our own package.json#version to pin the PyPI isotope-zero release.
export function pkgVersion() {
  // bin/izero.js → ../package.json
  const p = new URL("../package.json", import.meta.url);
  return JSON.parse(readFileSync(p, "utf8")).version;
}

/** Locate a binary on PATH (cross-platform, no shell). */
function which(name) {
  const isWin = platform() === "win32";
  const cmd = isWin ? "where" : "which";
  const r = spawnSync(cmd, [name], { stdio: ["ignore", "pipe", "ignore"], encoding: "utf8" });
  if (r.status !== 0) return null;
  const first = r.stdout.trim().split(/\r?\n/)[0];
  return first && existsSync(first) ? first : null;
}

/** Is `isotope-zero` importable in `pythonBin`? */
function hasIsotopeZero(pythonBin, pkgVer) {
  const r = spawnSync(pythonBin, ["-c", `import isotope_zero; assert isotope_zero.__version__ == "${pkgVer}"`], {
    stdio: ["ignore", "pipe", "pipe"],
    encoding: "utf8",
  });
  return r.status === 0;
}

/** Bootstrap the uv single-binary into ~/.izero/bin. Returns path or null. */
function bootstrapUv() {
  mkdirSync(UV_BIN_DIR, { recursive: true });
  const isWin = platform() === "win32";
  const uvBin = join(UV_BIN_DIR, isWin ? "uv.exe" : "uv");
  if (existsSync(uvBin)) return uvBin;

  // Astral publishes per-target static binaries. The standalone downloader is
  // the documented bootstrap path; it selects the right asset per (os, arch).
  const r = spawnSync(process.execPath, ["--input-type=module", "-e", UV_BOOTSTRAP], {
    stdio: ["ignore", "pipe", "pipe"],
    encoding: "utf8",
    env: { ...process.env, UV_INSTALL_DIR: UV_BIN_DIR },
  });
  return existsSync(uvBin) ? uvBin : null;
}

// Inline uv bootstrap script. uv's own `uv-self-update`/standalone installer:
// fetches https://astral.sh/uv/install.sh | sh (unix) or the .ps1 (win). We
// exec it via the shell uv documents, scoped to UV_INSTALL_DIR.
const UV_BOOTSTRAP = `
import { spawnSync } from "node:child_process";
import { platform } from "node:os";
const isWin = platform() === "win32";
if (isWin) {
  spawnSync("powershell", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
    "irm https://astral.sh/uv/install.ps1 | iex"], { stdio: "inherit", env: process.env });
} else {
  spawnSync("sh", ["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], {
    stdio: "inherit", env: process.env,
  });
}
`;

/** Create/refresh the uv venv at VENV_DIR with isotope-zero pinned to pkgVer. */
function ensureUvVenv(uvBin, pkgVer) {
  // Fast path: a previous run already built + stamped the right version.
  if (readStamp() === pkgVer && hasIsotopeZero(venvPython(), pkgVer)) return true;

  const needsBuild = !existsSync(join(VENV_DIR, "pyvenv.cfg"));
  if (needsBuild) {
    mkdirSync(IZERO_HOME, { recursive: true });
    // Atomic build lock: mkdir is exclusive on POSIX, so two concurrent izero
    // invocations can't both run `uv venv` into the same dir. The loser polls
    // for the winner's venv to appear (≤60s), then reuses it. This matters for
    // agent harnesses that launch parallel `izero` subprocesses: without it, the
    // losing process could fall through to `pip install --user` into the system
    // Python. A stale lock self-heals — the stamp/hasIsotopeZero probe re-
    // verifies, and the next clean build removes it.
    const lockDir = join(IZERO_HOME, ".venv-build-lock");
    let gotLock = false;
    try {
      mkdirSync(lockDir); // throws EEXIST if another process holds it
      gotLock = true;
    } catch (e) {
      if (e.code !== "EEXIST") throw e;
    }
    if (gotLock) {
      try {
        const r = spawnSync(uvBin, ["venv", VENV_DIR], { stdio: "pipe", encoding: "utf8" });
        if (r.status !== 0) return false;
      } finally {
        try { rmSync(lockDir, { recursive: true, force: true }); } catch {}
      }
    } else {
      // Wait for the holder's build, then fall through to the probe/install
      // below (which reuses the venv once pyvenv.cfg exists).
      for (let i = 0; i < 120 && !existsSync(join(VENV_DIR, "pyvenv.cfg")); i++) {
        spawnSync(process.execPath, ["-e", "require('timers').setTimeout(()=>0,500)"], { stdio: "ignore" });
      }
    }
  }
  // (Re)install the pinned version if not already present. Idempotent: uv skips
  // if isotope-zero==pkgVer is already satisfied. The sentinel file lets us
  // short-circuit subsequent runs without even probing Python.
  if (readStamp() === pkgVer && hasIsotopeZero(venvPython(), pkgVer)) return true;
  const r = spawnSync(uvBin, ["pip", "install", "--python", venvPython(), `isotope-zero==${pkgVer}`], {
    stdio: "pipe",
    encoding: "utf8",
  });
  if (r.status !== 0) return false;
  writeStamp(pkgVer);
  return hasIsotopeZero(venvPython(), pkgVer);
}

function readStamp() {
  try {
    return readFileSync(STAMP_FILE, "utf8").trim();
  } catch {
    return "";
  }
}
function writeStamp(v) {
  try {
    mkdirSync(IZERO_HOME, { recursive: true });
    writeFileSync(STAMP_FILE, v, "utf8");
  } catch {
    /* non-fatal: the probe on next run re-verifies */
  }
}

/** Path to the venv's python (the interpreter the launcher delegates to). */
export function venvPython() {
  const isWin = platform() === "win32";
  return join(VENV_DIR, isWin ? "Scripts/python.exe" : "bin/python");
}

/**
 * Resolve a Python interpreter that imports isotope-zero==pkgVer.
 * Returns { python: string, source: string } or throws { code, message }.
 */
export function resolvePython(pkgVer) {
  // 1. The venv from a previous run already satisfies pkgVer → reuse it as-is.
  //    The venv is the durable home: even when uv isn't on PATH this run (and
  //    bootstrap would need the network), a pre-built env is used with no
  //    re-install. Probe FIRST so a healthy venv never pays for uv bootstrap.
  if (hasIsotopeZero(venvPython(), pkgVer)) {
    return { python: venvPython(), source: "uv-venv" };
  }

  // 2. uv on PATH (or bootstrapped) → build/refresh the dedicated venv.
  let uvBin = which("uv") || (process.env.UV_BIN && existsSync(process.env.UV_BIN) ? process.env.UV_BIN : null);
  if (!uvBin) uvBin = bootstrapUv();
  if (uvBin) {
    if (ensureUvVenv(uvBin, pkgVer)) {
      return { python: venvPython(), source: "uv-venv" };
    }
    // uv venv build failed (e.g. pypi unreachable) → fall through.
  }

  // 3. System python3 with pip --user fallback.
  const py = which("python3") || which("python");
  if (!py) {
    const e = new Error(
      "isotope-zero requires Python 3.10+. Install it from https://python.org " +
      "or run `brew install python@3.12` (macOS) / `apt install python3` (Linux), " +
      "then re-run `izero`.",
    );
    e.code = "ENOPython";
    throw e;
  }
  if (hasIsotopeZero(py, pkgVer)) return { python: py, source: "system" };

  // Try `pip install --user` (may fail under PEP668 externally-managed envs).
  const r = spawnSync(py, ["-m", "pip", "install", "--user", `isotope-zero==${pkgVer}`], {
    stdio: "pipe",
    encoding: "utf8",
  });
  if (r.status === 0 && hasIsotopeZero(py, pkgVer)) {
    return { python: py, source: "system-user" };
  }
  const e = new Error(
    `isotope-zero ${pkgVer} could not be installed into ${py}.\n` +
    "If this is a Homebrew/system Python (PEP 668 'externally-managed'), install " +
    "uv (https://astral.sh/uv) and re-run `izero` — the launcher will build an " +
    "isolated env automatically. Original pip error:\n" + (r.stderr || r.stdout || "").trim(),
  );
  e.code = "EInstallFailed";
  throw e;
}
