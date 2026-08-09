"""``izero plugin install`` — local-path editor plugin installer.

This is the install helper behind the ``izero plugin install`` subcommand. It
copies the bundled editor plugin (``integrations/izero-plugin/``) into an
editor's config directory and rewrites the bundle's ``.mcp.json`` so its
``mcpServers.izero`` block points at a Python interpreter that can actually
launch ``isotope_zero.mcp.server``.

Local-first design (the headline differentiator from mem0):
    mem0's editor integration downloads a marketplace plugin and points at
    ``api.mem0.ai``. We do the opposite — there is **no marketplace, no
    download, no URL, no API key**. The bundle ships *inside* the
    ``isotope_zero`` distribution (or the repo checkout); install is a
    ``shutil.copytree`` + a JSON rewrite, both purely local. The MCP server
    command is resolved to ``izero-mcp`` when the console script is on PATH,
    else to ``sys.executable`` (or ``python3``) with ``-m isotope_zero.mcp.server``
    — mirroring ``ensure-env.js``'s graceful degradation, never a remote call.

    # No-network guarantee: this module imports only stdlib (os, sys, json,
    # shutil, pathlib, importlib.resources). It must NEVER import requests /
    # urllib.request / httpx / aiohttp / openai — the verification step greps
    # the integrations tree for those and expects zero hits.

Editor defaults:
    claude -> ~/.claude/plugins/izero
    cursor -> ~/.cursor/plugins/izero
    codex  -> ~/.codex/plugins/izero

Marketplace / remote discovery is deferred to v1.3.0 (see roadmap); this
helper is strictly a local-path installer.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# The editor-bundle directory name, shipped at the repo root (dev) and via the
# installed package's parents (PyPI). Same name in both locations so the
# resolver can collapse to one ``Path / "integrations" / "izero-plugin"``.
_BUNDLE_NAME = "izero-plugin"

# Per-editor default install roots. Explicit ``--target`` always wins; these
# are the conventional config dirs each editor scans for plugins. ``expanduser``
# is applied at use time so a literal ``~`` in an env-var override resolves too.
_EDITOR_DEFAULTS: dict[str, str] = {
    "claude": "~/.claude/plugins/izero",
    "cursor": "~/.cursor/plugins/izero",
    "codex": "~/.codex/plugins/izero",
}

# The manifest filenames every plugin bundle carries. In dry-run we validate
# that whichever of these *exist* in the bundle parse as JSON (a missing file
# is not an error — the bundle may ship incrementally; only a present-but-
# malformed manifest is). The install path rewrites the MCP configs in place.
_MANIFESTS: tuple[str, ...] = ("plugin.json", ".mcp.json", "hooks.json")

# Every MCP config shipped in the bundle — the Claude (``.mcp.json``), Cursor
# (``.cursor-mcp.json``), and Codex (``.codex-mcp.json``) variants. All use the
# same ``mcpServers.izero`` shape, so one rewrite routine handles all three.
# The install path rewrites EACH present config so the ``python -m`` fallback
# (when ``izero-mcp`` isn't on PATH) reaches Cursor/Codex too — leaving these
# untouched would strand those editors pointing at a nonexistent console script.
_MCP_CONFIGS: tuple[str, ...] = (".mcp.json", ".cursor-mcp.json", ".codex-mcp.json")

# The persistent store path stamped into the installed ``.mcp.json`` env so
# the MCP server + the hook entrypoint share one store. Matches the CLI's
# ``_DEFAULT_DB`` and the hook engine's ``_default_db_path``.
_DEFAULT_DB_ENV = "~/.isotope_zero/isotope_zero.db"

# The MCP server entrypoint used by the python-fallback command. Kept as a
# list so it composes cleanly into the ``args`` array. Never a URL.
_MCP_MODULE_ARGS: list[str] = ["-m", "isotope_zero.mcp.server"]


def plugin_install(
    target: str | None,
    editor: str,
    dry_run: bool,
) -> int:
    """Install (or preview) the izero editor plugin into an editor config dir.

    Resolution + behavior:

    1. **Bundle path** — dev repo path ``integrations/izero-plugin/`` if it
       exists, else the installed-package path via
       :func:`importlib.resources.files`. If neither resolves, print an error
       and return 1.
    2. **Target dir** — explicit ``target`` if given, else the per-editor
       default under ``~`` (see :data:`_EDITOR_DEFAULTS`). ``expanduser``'d.
    3. **MCP command** — ``izero-mcp`` if :func:`shutil.which` finds it on
       PATH (the cleanest path, no args); else ``sys.executable`` (or
       ``python3`` as a last resort) with ``-m isotope_zero.mcp.server``.
       Mirrors ``ensure-env.js``'s graceful degradation. NEVER a URL / key.
    4. ``dry_run`` — print the resolved bundle / target / command as JSON,
       validate the bundle's present manifests parse, and return 0 *without*
       writing anything.
    5. non-``dry_run`` — :func:`shutil.copytree` (``dirs_exist_ok=True``) into
       the target, then rewrite the target's ``.mcp.json`` so
       ``mcpServers.izero.{command,args,env}`` match the resolved command and
       ``env.ISOTOPE_ZERO_DB`` points at the persistent store. Print what was
       written. Return 0.
    6. All file ops are wrapped in try/except; on any failure a clear error is
       printed to stderr and 1 is returned. This function never raises.

    Returns the process exit code (0 success, 1 failure). Never raises: a
    top-level guard renders any unexpected failure as a clean stderr message +
    exit 1, so a broken install never aborts the surrounding CLI with a
    traceback (mirrors :func:`isotope_zero.integrations.hooks.run_hook`'s
    "never break the editor stream" contract).
    """
    try:
        bundle = _resolve_bundle_path()
        target_dir = _resolve_target(target, editor)
        command, args = _resolve_mcp_command()

        if dry_run:
            return _dry_run(bundle, target_dir, command, args)
        return _install(bundle, target_dir, command, args)
    except _ResolveError as exc:
        # Precise, actionable message for the expected "bundle not found" path.
        print(f"izero plugin install: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — never raise from the CLI helper
        print(
            f"izero plugin install: unexpected error: {exc}", file=sys.stderr
        )
        return 1


# --------------------------------------------------------------------------- #
# Resolution helpers
# --------------------------------------------------------------------------- #
class _ResolveError(RuntimeError):
    """Raised when the plugin bundle directory cannot be located.

    Carried to the top-level try/except, which renders it as a clear stderr
    message + exit 1. Kept as a distinct type (not a bare ``RuntimeError``)
    so the catch is precise and never masks an unrelated OS error.
    """


def _resolve_bundle_path() -> Path:
    """Locate the editor-bundle directory (dev repo path, then installed pkg).

    Order:
      (a) Dev mode — ``<repo>/integrations/izero-plugin/`` if it exists. The
          repo root is resolved relative to THIS file (parents[3]) so the
          helper works regardless of the caller's CWD (same trick
          :mod:`izero_adapters._engine` uses for engine discovery).
      (b) PyPI install — walk the installed package's parent directories via
          :func:`importlib.resources.files` until a sibling
          ``integrations/izero-plugin/`` appears (a wheel may ship it next
          to the package, not inside it).

    Raises :class:`_ResolveError` if neither resolves — the caller surfaces it
    as a clean stderr message rather than an ImportError traceback.
    """
    # (a) Dev repo path. __file__ is .../src/isotope_zero/integrations/cli.py,
    # so parents[3] is the repo root containing the top-level integrations/ dir.
    here = Path(__file__).resolve()
    dev_bundle = here.parents[3] / "integrations" / _BUNDLE_NAME
    if dev_bundle.is_dir():
        return dev_bundle

    # (b) Installed-package path. importlib.resources.files gives the package
    # root; walk up looking for a sibling integrations/izero-plugin/ — covers
    # wheels/sdists that place the bundle alongside the package rather than
    # inside it (so we don't have to declare it as package_data).
    try:
        from importlib.resources import files
    except ImportError:  # pragma: no cover — py>=3.10 always has this
        raise _ResolveError(
            "importlib.resources.files is unavailable (Python < 3.9?)"
        )
    try:
        root = Path(str(files("isotope_zero"))).resolve()
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        raise _ResolveError(
            f"could not locate the installed isotope_zero package: {exc}"
        )
    for parent in [root, *root.parents]:
        candidate = parent / "integrations" / _BUNDLE_NAME
        if candidate.is_dir():
            return candidate

    raise _ResolveError(
        f"plugin bundle not found: looked for {_BUNDLE_NAME}/ under the repo "
        f"root ({dev_bundle}) and alongside the installed package "
        f"({root}). Reinstall isotope-zero or run from a repo checkout."
    )


def _resolve_target(target: str | None, editor: str) -> Path:
    """Pick the install target dir: explicit override, else per-editor default.

    An unknown editor (shouldn't happen — argparse restricts to the known set,
    but this is also unit-callable) falls back to the claude default so the
    function is total and never raises on a novel editor name.
    """
    if target:
        return Path(os.path.expanduser(target))
    default = _EDITOR_DEFAULTS.get(editor, _EDITOR_DEFAULTS["claude"])
    return Path(os.path.expanduser(default))


def _resolve_mcp_command() -> tuple[str, list[str]]:
    """Resolve the MCP server launch command — never a URL or API key.

    Prefers ``izero-mcp`` when the console script is on PATH (the clean path:
    no args, the script entrypoint handles :mod:`isotope_zero.mcp.server`).
    Falls back to the current interpreter (``sys.executable``) with
    ``-m isotope_zero.mcp.server``; if the interpreter is empty/dummy (e.g. a
    frozen build without ``sys.executable``), ``python3`` is the last resort.
    This mirrors ``ensure-env.js``'s graceful-degradation order.
    """
    if shutil.which("izero-mcp"):
        return "izero-mcp", []
    py = sys.executable or "python3"
    return py, list(_MCP_MODULE_ARGS)


# --------------------------------------------------------------------------- #
# dry-run: validate, print plan, write nothing
# --------------------------------------------------------------------------- #
def _dry_run(
    bundle: Path, target_dir: Path, command: str, args: list[str]
) -> int:
    """Validate the bundle's manifests and print the install plan as JSON.

    Returns 0 without touching the filesystem. A present-but-malformed manifest
    is reported on stderr (return 1); a *missing* manifest is not an error —
    the bundle ships incrementally and a later phase may add ``.mcp.json``.
    """
    plan: dict[str, Any] = {
        "dry_run": True,
        "bundle": str(bundle),
        "target": str(target_dir),
        "mcp": {"command": command, "args": list(args)},
        "manifests": {},
        "rewrite": [],
    }
    bad: list[str] = []
    for name in _MANIFESTS:
        path = bundle / name
        if not path.is_file():
            plan["manifests"][name] = None  # absent (incremental bundle)
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            plan["manifests"][name] = f"INVALID: {exc}"
            bad.append(f"{name}: {exc}")
        else:
            plan["manifests"][name] = "ok"

    # Report which MCP configs will be rewritten on install (the editor
    # variants must be in this list, or a fallback install strands them).
    for cfg_name in _MCP_CONFIGS:
        if (bundle / cfg_name).is_file():
            plan["rewrite"].append(cfg_name)

    print(json.dumps(plan, indent=2))
    if bad:
        print(
            f"izero plugin install: malformed manifest(s):\n  "
            + "\n  ".join(bad),
            file=sys.stderr,
        )
        return 1
    return 0


# --------------------------------------------------------------------------- #
# install: copytree + rewrite .mcp.json
# --------------------------------------------------------------------------- #
def _install(
    bundle: Path, target_dir: Path, command: str, args: list[str]
) -> int:
    """Copy the bundle into ``target_dir`` and rewrite its ``.mcp.json``.

    ``copytree(dirs_exist_ok=True)`` makes the install idempotent — re-running
    over an existing install refreshes the bundle in place. After the copy,
    the target's ``.mcp.json`` (if present) is rewritten so
    ``mcpServers.izero.{command,args,env}`` match the resolved command and
    ``env.ISOTOPE_ZERO_DB`` points at the persistent store. Returns 0 on
    success; any failure is caught, reported to stderr, and returns 1.
    """
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle, target_dir, dirs_exist_ok=True)
    except OSError as exc:
        print(
            f"izero plugin install: failed to copy {bundle} -> {target_dir}: "
            f"{exc}",
            file=sys.stderr,
        )
        return 1

    written: list[str] = [str(target_dir)]
    # Rewrite EVERY MCP config the bundle ships — Claude (.mcp.json), Cursor
    # (.cursor-mcp.json), Codex (.codex-mcp.json). All three use the same
    # mcpServers.izero shape, so _rewrite_mcp_json handles each. Skipping the
    # editor variants would leave them pointing at `izero-mcp` when this
    # install fell back to `python -m`, stranding Cursor/Codex.
    for cfg_name in _MCP_CONFIGS:
        mcp_path = target_dir / cfg_name
        if mcp_path.is_file():
            rc = _rewrite_mcp_json(mcp_path, command, args)
            if rc != 0:
                return rc
            written.append(str(mcp_path))

    print(
        json.dumps(
            {"installed": True, "written": written,
             "mcp": {"command": command, "args": list(args)}},
            indent=2,
        )
    )
    return 0


def _rewrite_mcp_json(mcp_path: Path, command: str, args: list[str]) -> int:
    """Rewrite ``mcpServers.izero`` in ``mcp_path`` to the resolved command.

    Preserves any sibling servers / keys the bundle declares; only the
    ``izero`` block's ``command``, ``args``, and ``env.ISOTOPE_ZERO_DB`` are
    touched. ``env`` is created if absent. The file is written atomically via
    a temp write + replace so a partial write can't leave a corrupt manifest.

    Returns 0 on success, 1 on a parse/write error (reported to stderr).
    """
    try:
        data = json.loads(mcp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"izero plugin install: {mcp_path} is not valid JSON ({exc}); "
            f"leaving the copied bundle's manifest untouched.",
            file=sys.stderr,
        )
        return 1

    if not isinstance(data, dict):
        print(
            f"izero plugin install: {mcp_path} top-level is not a JSON object; "
            f"leaving it untouched.",
            file=sys.stderr,
        )
        return 1

    mcp_servers = data.setdefault("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        print(
            f"izero plugin install: {mcp_path} mcpServers is not an object; "
            f"leaving it untouched.",
            file=sys.stderr,
        )
        return 1

    izero = mcp_servers.setdefault("izero", {})
    if not isinstance(izero, dict):
        # An object with the wrong shape clobbered to a well-formed block.
        mcp_servers["izero"] = izero = {}
    izero["command"] = command
    izero["args"] = list(args)
    env = izero.setdefault("env", {})
    if not isinstance(env, dict):
        env = izero["env"] = {}
    env["ISOTOPE_ZERO_DB"] = os.path.expanduser(_DEFAULT_DB_ENV)

    try:
        _atomic_write_json(mcp_path, data)
    except OSError as exc:
        print(
            f"izero plugin install: failed to rewrite {mcp_path}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write ``data`` as JSON to ``path`` via temp-file + ``os.replace``.

    A direct write truncated mid-flush would leave a corrupt manifest that the
    editor can't parse; the temp + atomic replace means a crash mid-write
    preserves the previous (copied) file intact.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
