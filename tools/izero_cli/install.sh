#!/bin/sh
# shellcheck shell=dash
# =============================================================================
# izero-cli — universal installer
# -----------------------------------------------------------------------------
# Sets up an isolated izero-cli runtime in ~/.izero (a private venv) and
# symlinks the `izero` executable to ~/.local/bin/izero so it is on $PATH.
#
# One-liner usage:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/isotope_zero/main/tools/izero_cli/install.sh | sh
#   # or, from a source checkout:
#   sh tools/izero_cli/install.sh
#
# This script only WRITES inside the install root (default ~/.izero) and the bin
# dir (default ~/.local/bin). It never modifies core prototype source or any
# other project files. It is idempotent: re-running upgrades the package.
#
# Environment overrides (useful for testing without touching $HOME):
#   PYTHON         python interpreter to use           (default: python3)
#   IZERO_ROOT     install root (venv lives here)      (default: ~/.izero)
#   IZERO_VENV     explicit venv path                  (default: $IZERO_ROOT/venv)
#   BIN_DIR        where the `izero` symlink goes      (default: ~/.local/bin)
#   PY_SRC         source dir to `pip install` from    (default: dir of this script)
#   PY_EXTRAS      optional extras, e.g. "onnx"        (default: "")
#   GIT_URL        git URL to install from instead of  (default: "")
#                 a local source dir; overrides PY_SRC.
#   NO_SYMLINK     "1" to skip creating the symlink     (default: "")
#   DRY_RUN        "1" to print the plan and exit 0     (default: "")
# =============================================================================
set -eu

# --- helpers -----------------------------------------------------------------
log()  { printf '%s\n' "$*"; }
warn() { printf '⚠  %s\n' "$*" >&2; }
die()  { printf '✖  %s\n' "$*" >&2; exit 1; }

# Strip embedded credentials (user:pass@ or token@) from a URL for display.
# Leaves non-URL strings (paths, package names) unchanged. Uses _ru* vars to
# avoid clobbering the _rest/_minor locals used by find_python (sh has no local).
redact_url() {
    _ru="$1"
    case "$_ru" in
        *://*@*)
            _ru_scheme=${_ru%%://*}
            _ru_rest=${_ru#*://}
            _ru_host=${_ru_rest#*@}
            printf '%s://%s' "$_ru_scheme" "$_ru_host"
            ;;
        *) printf '%s' "$_ru" ;;
    esac
}

# Reject stray positional arguments. All configuration is via environment
# variables; `sh install.sh FOO=bar` passes FOO=bar as $1 (NOT as an env var),
# which silently does nothing and can surprise users (e.g. they think they
# scoped an install to a temp dir when they did not). Fail loudly instead.
if [ "$#" -gt 0 ]; then
    die "this script takes no arguments; configure it via environment variables.
Example:  IZERO_ROOT=/tmp/izero BIN_DIR=/tmp/bin sh install.sh
See the header comment for the full list of overrides (PYTHON, IZERO_VENV,
PY_SRC, PY_EXTRAS, GIT_URL, NO_SYMLINK, DRY_RUN)."
fi

# portable "$HOME" with a fallback (HOME is not guaranteed set under some CI)
: "${HOME:=}"
[ -n "$HOME" ] || die "HOME is not set; cannot determine an install location."

# --- configuration from env --------------------------------------------------
PYTHON="${PYTHON:-python3}"
IZERO_ROOT="${IZERO_ROOT:-$HOME/.izero}"
IZERO_VENV="${IZERO_VENV:-$IZERO_ROOT/venv}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
PY_EXTRAS="${PY_EXTRAS:-}"
GIT_URL="${GIT_URL:-}"
NO_SYMLINK="${NO_SYMLINK:-}"
DRY_RUN="${DRY_RUN:-}"

# Resolve the bundled source dir relative to this script. ONLY meaningful when
# the installer is invoked as a real file (a checkout: `sh path/to/install.sh`,
# or `sh install.sh` from inside the source dir). When piped to `sh` (the
# curl|sh one-liner), $0 is just the shell name ("sh") with no slash and is NOT
# a file in the CWD, so we leave SCRIPT_DIR EMPTY: there is no bundled source
# to locate, and the installer correctly falls through to the PyPI package.
# (A bare SCRIPT_DIR="." would probe the caller's CWD and risk matching an
# unrelated pyproject.toml there — the F6 defect.)
SCRIPT_DIR=""
if [ -n "${0:-}" ]; then
    case "$0" in
        */*) SCRIPT_DIR="${0%/*}" ;;   # path with a slash: real file invocation
        *)
            # No slash: either piped ($0="sh") or a bare relative filename
            # (`sh install.sh` from inside the source dir). Only the latter is
            # a real source location — a piped shell's $0 is never a CWD file.
            if [ -f "$0" ] && [ -r "$0" ]; then
                SCRIPT_DIR="."
            else
                SCRIPT_DIR=""   # piped to sh (e.g. curl|sh): no local source
            fi
            ;;
    esac
fi
PY_SRC="${PY_SRC:-$SCRIPT_DIR}"

# When pip-installing from a local dir, it must actually be the izero-cli
# source (contain pyproject.toml). If GIT_URL is set, the local dir is unused.
install_target=""
if [ -n "$GIT_URL" ]; then
    install_target="$GIT_URL"
elif [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    install_target="$SCRIPT_DIR"
else
    # No local source + no GIT_URL: fall back to the published PyPI package.
    install_target="izero-cli"
fi

# --- requirement: the package needs Python >= 3.10 ---------------------------
need_py_minor=10

find_python() {
    # Try PYTHON, then python3, python in that order; pick the first that
    # exists and meets the minimum version. Echoes the chosen interpreter.
    _candidates="$PYTHON python3 python"
    for _c in $_candidates; do
        _path=$(command -v "$_c" 2>/dev/null || true)
        [ -n "$_path" ] || continue
        # Parse "Python 3.<minor>.<patch>" (or "Python X.Y" w/o patch).
        _ver=$("$_path" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
        case "$_ver" in
            3.*)
                _maj=${_ver%%.*}
                _rest=${_ver#*.}
                _minor=${_rest%%.*}
                if [ "$_maj" -ge 3 ] 2>/dev/null && [ "$_minor" -ge "$need_py_minor" ] 2>/dev/null; then
                    printf '%s' "$_path"
                    return 0
                fi
                ;;
        esac
    done
    return 1
}

# --- dry-run: print the plan and bail ----------------------------------------
if [ "$DRY_RUN" = "1" ]; then
    log "izero-cli install — DRY RUN (no changes will be made)"
    log "  python         : $PYTHON"
    log "  install root   : $IZERO_ROOT"
    log "  venv           : $IZERO_VENV"
    log "  bin dir        : $BIN_DIR"
    log "  source         : $PY_SRC"
    log "  install target : $(redact_url "$install_target")"
    log "  extras         : ${PY_EXTRAS:-(none)}"
    if [ -n "$GIT_URL" ]; then
        log "  git url        : $(redact_url "$GIT_URL")"
    else
        log "  git url        : (none)"
    fi
    log "  symlink        : ${NO_SYMLINK:+skipped}${NO_SYMLINK:-yes}"
    # Still validate python so the dry run is meaningful.
    if _py=$(find_python); then
        log "  python found   : $_py"
    else
        die "no python >= 3.$need_py_minor found on PATH (set PYTHON= to override)"
    fi
    exit 0
fi

# --- validate python ---------------------------------------------------------
if ! _py=$(find_python); then
    die "python >= 3.$need_py_minor not found.
Install Python 3.10+ or set PYTHON=/path/to/python3, then re-run."
fi
PYTHON="$_py"
log "Using Python: $PYTHON ($("$PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'))"

# venv module must be present (standard on CPython; absent on some stripped builds).
if ! "$PYTHON" -c 'import venv' 2>/dev/null; then
    die "the 'venv' stdlib module is missing for $PYTHON.
On Debian/Ubuntu:  sudo apt install python3-venv
On other systems:  install the full python3 standard library."
fi

# --- create directories ------------------------------------------------------
mkdir -p "$IZERO_ROOT" || die "could not create install root: $IZERO_ROOT"
if [ "$NO_SYMLINK" != "1" ]; then
    mkdir -p "$BIN_DIR" || die "could not create bin dir: $BIN_DIR"
fi

# --- (re)create or reuse the venv --------------------------------------------
# Reuse an existing compatible venv to make re-runs cheap (idempotent upgrades).
if [ -x "$IZERO_VENV/bin/python" ]; then
    log "Reusing existing venv: $IZERO_VENV"
else
    log "Creating venv: $IZERO_VENV"
    "$PYTHON" -m venv "$IZERO_VENV" || die "venv creation failed."
fi

# Upgrade pip so dependency resolution is robust across platforms.
log "Ensuring pip is up to date …"
# shellcheck disable=SC2086
"$IZERO_VENV/bin/python" -m pip install --upgrade pip >/dev/null 2>&1 || \
    warn "pip self-upgrade failed (continuing with bundled pip)."

# --- install izero-cli -------------------------------------------------------
# Build the pip install spec. A git URL needs the `git+` VCS prefix (a bare
# https URL would download the HTML page, not clone). Extras on a git URL must
# use PEP 508 direct-URL syntax `izero-cli[onnx] @ git+<url>` — the bracket
# form `<url>[onnx]` is parsed as a literal path segment and rejected by pip.
# `display_spec` is the credentials-redacted twin used for logging/die so an
# embedded token in GIT_URL never reaches stdout/logs.
if [ -n "$GIT_URL" ]; then
    if [ -n "$PY_EXTRAS" ]; then
        install_spec="izero-cli[${PY_EXTRAS}] @ git+${GIT_URL}"
        display_spec="izero-cli[${PY_EXTRAS}] @ git+$(redact_url "$GIT_URL")"
    else
        install_spec="git+${GIT_URL}"
        display_spec="git+$(redact_url "$GIT_URL")"
    fi
else
    # Local path or PyPI name: extras attach directly as name[onnx] / path[onnx].
    if [ -n "$PY_EXTRAS" ]; then
        install_spec="${install_target}[${PY_EXTRAS}]"
    else
        install_spec="$install_target"
    fi
    display_spec="$install_spec"
fi

log "Installing izero-cli from: $display_spec"
# --no-cache-dir keeps the install footprint small and avoids stale wheels.
# install_spec is quoted so a path/extras value containing spaces stays one arg.
"$IZERO_VENV/bin/python" -m pip install --no-cache-dir "$install_spec" \
    || die "pip install failed for: $display_spec"

# --- locate the console script inside the venv -------------------------------
izero_bin="$IZERO_VENV/bin/izero"
[ -x "$izero_bin" ] || die "install succeeded but '$izero_bin' is missing/not executable."

# --- symlink into BIN_DIR ----------------------------------------------------
if [ "$NO_SYMLINK" != "1" ]; then
    # `ln -sfn` only unlinks a non-directory destination. If $BIN_DIR/izero is a
    # real directory (prior mistake / unrelated package), ln would place the
    # symlink INSIDE it as $BIN_DIR/izero/izero and exit 0 — a silent false
    # success. Guard: refuse a directory target; refresh any file/symlink.
    if [ -e "$BIN_DIR/izero" ] && [ ! -L "$BIN_DIR/izero" ]; then
        die "'$BIN_DIR/izero' exists and is not a symlink; refusing to clobber it. \
Remove it manually, then re-run."
    fi
    ln -sfn "$izero_bin" "$BIN_DIR/izero" || die "could not symlink '$izero_bin' -> '$BIN_DIR/izero'."
    log "Linked: $BIN_DIR/izero -> $izero_bin"
fi

# --- smoke test --------------------------------------------------------------
if "$izero_bin" --help >/dev/null 2>&1; then
    log "Smoke test: izero --help ✓"
else
    warn "izero --help returned non-zero; the install may be incomplete."
fi

# --- PATH check + success message --------------------------------------------
PATH_OK=no
case ":${PATH:-}:" in
    *":$BIN_DIR:"*) PATH_OK=yes ;;
esac

log ""
log "✓ izero-cli installed."
log "  venv : $IZERO_VENV"
if [ "$NO_SYMLINK" != "1" ]; then
    log "  bin  : $BIN_DIR/izero"
else
    # No PATH symlink: tell the user where the executable actually lives so
    # they can invoke it directly or wire up their own wrapper.
    log "  bin  : $izero_bin  (symlink skipped, NO_SYMLINK=1)"
fi
if [ "$NO_SYMLINK" != "1" ]; then
    if [ "$PATH_OK" = "yes" ]; then
        log ""
        log "Run:  izero --help"
    else
        log ""
        warn "$BIN_DIR is not in your \$PATH."
        log "Add it by running one of (for your shell), then start a new shell:"
        log ""
        log "  sh/bash :  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.profile"
        log "  zsh     :  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshenv"
        log "  fish    :  fish_add_path $BIN_DIR"
        log ""
        log "Or run izero directly: $BIN_DIR/izero --help"
    fi
fi
log ""
log "To upgrade later, re-run this script."
log "To uninstall: rm -rf \"$IZERO_ROOT\" && rm -f \"$BIN_DIR/izero\""
