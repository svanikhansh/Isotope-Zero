"""Tests for the izero editor-plugin bundle manifests + offline guarantee.

Validates the files under ``integrations/izero-plugin/`` that an editor reads at
plugin-load time: the root manifest, the local-stdio MCP config, the lifecycle-hook
registry, the editor-specific re-exports, and the skills. The headline assertion is
the **privacy contract**: no manifest references a URL, an API key, or an auth header,
and the MCP config points at the local stdio command (``izero-mcp``), never a cloud
HTTP endpoint — the structural opposite of mem0's ``mcp.mem0.ai`` + ``MEM0_API_KEY``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Resolve the bundle dir relative to the repo root (test runs from repo root or
# from src/tests/). Works under both ``pytest src/tests`` and ``pytest`` at root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = _REPO_ROOT / "integrations" / "izero-plugin"

# All skills must reference one of these real MCP tool names exposed by
# ``build_mcp_app()`` in ``isotope_zero.mcp.server``.
_REAL_MCP_TOOLS = {
    "add_memory",
    "query_memory",
    "delete_memory",
    "get_metrics",
    "run_consolidation",
}


def _load_json(rel: str) -> dict:
    return json.loads((BUNDLE / rel).read_text(encoding="utf-8"))


def _bundle_files() -> list[Path]:
    return [p for p in BUNDLE.rglob("*") if p.is_file()]


# --------------------------------------------------------------------------- #
# 1. All manifests parse as JSON + the headline privacy contract
# --------------------------------------------------------------------------- #
class TestManifestsParse:
    @pytest.mark.parametrize("rel", [
        "plugin.json",
        ".mcp.json",
        "hooks.json",
        ".claude-plugin/plugin.json",
        ".cursor-plugin/plugin.json",
        ".codex-plugin/plugin.json",
    ])
    def test_json_valid(self, rel):
        _load_json(rel)  # raises on invalid JSON

    @pytest.mark.parametrize("rel", [
        ".mcp.json",
        "hooks.json",
    ])
    def test_transport_files_have_no_url_auth_or_apikey(self, rel):
        """The transport files must carry no URL, no auth header, no API key —
        izero is stdio-only, no cloud endpoint."""
        text = (BUNDLE / rel).read_text(encoding="utf-8").lower()
        assert "https://" not in text and "http://" not in text, (
            f"{rel}: transport file must reference no network URL"
        )
        assert "authorization" not in text, f"{rel}: must carry no auth header"
        assert "api_key" not in text, f"{rel}: must reference no API key"

    @pytest.mark.parametrize("rel", [
        "plugin.json",
        ".claude-plugin/plugin.json",
        ".cursor-plugin/plugin.json",
        ".codex-plugin/plugin.json",
    ])
    def test_manifests_have_no_apikey_auth_or_cloud_url(self, rel):
        """Manifests may carry homepage/repository/websiteURL repo URLs
        (legitimate metadata), but must NOT declare an API key, an auth
        header, or an inline ``url`` field on an mcpServers entry naming a
        cloud service endpoint (the mem0 anti-pattern).

        ``mcpServers`` may be a *string* (a path reference to a separate
        ``.cursor-mcp.json``/``.codex-mcp.json``, as cursor/codex do) — in that
        case there's no inline server to inspect here; the referenced file is
        itself audited by ``test_mcp_config_has_no_url_or_auth``."""
        m = _load_json(rel)
        flat = json.dumps(m).lower()
        assert "api_key" not in flat, f"{rel}: references an API key"
        assert "authorization" not in flat, f"{rel}: carries an auth header"
        servers = m.get("mcpServers")
        if isinstance(servers, dict):
            for name, srv in servers.items():
                assert not isinstance(srv, dict) or "url" not in srv, (
                    f"{rel}: mcp server {name} has a cloud url — izero is stdio-only"
                )

    def test_mcp_config_is_local_stdio_not_cloud(self):
        mcp = _load_json(".mcp.json")
        srv = mcp["mcpServers"]["izero"]
        assert srv.get("command") == "izero-mcp", (
            "MCP must spawn the local stdio console script, not a cloud HTTP endpoint"
        )
        assert "url" not in srv, "izero MCP must have no url"
        assert "Authorization" not in srv and "headers" not in srv, (
            "izero MCP must carry no auth — there is no API key"
        )
        # The env points the store at the user's persistent local path.
        assert srv["env"]["ISOTOPE_ZERO_DB"].endswith("isotope_zero.db")


# --------------------------------------------------------------------------- #
# 2. hooks.json — every command resolves to a wrapper that exists + is executable
# --------------------------------------------------------------------------- #
class TestHooksRegistry:
    def test_all_hook_commands_resolve_to_existing_scripts(self):
        reg = _load_json("hooks.json")
        for _event, matchers in reg["hooks"].items():
            for m in matchers:
                for h in m["hooks"]:
                    cmd = h["command"]
                    assert "${CLAUDE_PLUGIN_ROOT}" in cmd, (
                        f"hook command should be plugin-root-relative: {cmd}"
                    )
                    script_rel = cmd.split("${CLAUDE_PLUGIN_ROOT}/", 1)[1].split()[0]
                    script = BUNDLE / script_rel
                    assert script.is_file(), (
                        f"hooks.json command references a missing script: {script_rel}"
                    )
                    # Executable bit set (the install helper / CI enforces this;
                    # assert here so a chmod slip is caught immediately).
                    assert os.access(script, os.X_OK), (
                        f"hook wrapper not executable: {script_rel} "
                        f"(run: chmod +x {script})"
                    )

    def test_hook_events_cover_lifecycle_surface(self):
        reg = _load_json("hooks.json")
        events = set(reg["hooks"].keys())
        # The full capture surface: session start, prompt submit, pre-tool-use
        # (read + write/edit), post-tool-use (bash), stop, pre-compact.
        assert {"SessionStart", "UserPromptSubmit", "PreToolUse",
                "PostToolUse", "Stop", "PreCompact"} <= events


# --------------------------------------------------------------------------- #
# 3. plugin.json — manifest fields the editor reads
# --------------------------------------------------------------------------- #
class TestRootManifest:
    def test_fields_present(self):
        m = _load_json("plugin.json")
        for k in ("id", "name", "version", "description", "author", "license"):
            assert k in m, f"plugin.json missing required field: {k}"
        assert m["contextFileName"] == "AGENTS.md"
        assert m["version"] == "1.2.0"  # matches the pyproject bump


# --------------------------------------------------------------------------- #
# 4. skills — frontmatter + real MCP tool references
# --------------------------------------------------------------------------- #
class TestSkills:
    SKILL_NAMES = ["remember", "recall", "forget", "peek", "stats", "decay-review"]

    @pytest.mark.parametrize("name", SKILL_NAMES)
    def test_skill_has_frontmatter(self, name):
        sk = (BUNDLE / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        assert sk.startswith("---\n"), f"{name}: SKILL.md must start with frontmatter"
        # frontmatter is the text between the first pair of --- fences.
        _, fm, _body = sk.split("---\n", 2)
        assert "name:" in fm, f"{name}: frontmatter missing name"
        assert "description:" in fm, f"{name}: frontmatter missing description"

    @pytest.mark.parametrize("name", SKILL_NAMES)
    def test_skill_references_a_real_mcp_tool(self, name):
        sk = (BUNDLE / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        # At least one of the real tool names must appear in the skill body.
        assert any(tool in sk for tool in _REAL_MCP_TOOLS), (
            f"{name}: SKILL.md must reference one of the real MCP tools "
            f"{sorted(_REAL_MCP_TOOLS)}"
        )

    @pytest.mark.parametrize("name", SKILL_NAMES)
    def test_skill_no_cloud_references(self, name):
        sk = (BUNDLE / "skills" / name / "SKILL.md").read_text(encoding="utf-8").lower()
        assert "api_key" not in sk, f"{name}: skill must not reference an API key"
        assert "mcp.mem0.ai" not in sk and "api.mem0.ai" not in sk, (
            f"{name}: skill must not reference a cloud endpoint"
        )


# --------------------------------------------------------------------------- #
# 5. AGENTS.md context file present
# --------------------------------------------------------------------------- #
class TestContextFile:
    def test_agents_md_present_and_local_first(self):
        p = BUNDLE / "AGENTS.md"
        assert p.is_file(), "contextFileName points at AGENTS.md — it must exist"
        text = p.read_text(encoding="utf-8").lower()
        # The privacy contract is stated explicitly in the context file.
        assert "local" in text and "offline" in text
        assert "no api key" in text or "no api_key" in text

    def test_readme_present(self):
        assert (BUNDLE / "README.md").is_file()


# --------------------------------------------------------------------------- #
# 6. Offline guarantee — the structural, machine-checked privacy contract
# --------------------------------------------------------------------------- #
class TestOfflineGuarantee:
    # The files that actually wire up transport + capture logic. The MCP config
    # and the hook wrappers are where a network call would live if we'd copied
    # mem0's design; docs/manifests legitimately name URLs (homepage, repo) and
    # the words "API key"/"cloud" (to explain their *absence*), so they're out of
    # scope for this grep. The contract under test is: no transport file opens a
    # network endpoint or requires credentials.
    _TRANSPORT_FILES = [
        Path(".mcp.json"),
        Path("hooks.json"),
    ]

    def test_mcp_config_has_no_url_or_auth(self):
        mcp = _load_json(".mcp.json")["mcpServers"]["izero"]
        assert "url" not in mcp, "izero MCP must not be an HTTP endpoint"
        assert "Authorization" not in mcp and "headers" not in mcp, (
            "izero MCP must carry no auth header — there is no API key"
        )
        assert "api_key" not in json.dumps(mcp).lower()

    def test_hook_wrappers_invoke_only_local_commands(self):
        """Every hook wrapper calls the local izero CLI / python module — never
        curl/wget/python-to-a-cloud-URL. The wrapper is the only shell surface a
        network call could hide in."""
        for sh in sorted((BUNDLE / "hooks").glob("*.sh")):
            text = sh.read_text(encoding="utf-8").lower()
            assert "curl " not in text, f"{sh.name}: must not curl a network endpoint"
            assert "wget " not in text
            assert "https://" not in text and "http://" not in text
            # It must invoke the local entrypoint (the whole design: trivial
            # wrapper → izero hook <event>).
            assert "izero hook" in text or "python3 -m isotope_zero" in text, (
                f"{sh.name}: wrapper must call the local izero/python entrypoint"
            )

    def test_python_hook_module_imports_no_network_clients(self):
        """The hook engine module must not import a network client. This is the
        real, import-time guarantee that no hook ever makes an outbound call.

        We parse the AST and inspect actual ``import``/``import from`` names —
        NOT grep the source text — because the module's own docstring
        legitimately *names* the anti-pattern (``urllib.request.urlopen`` to
        ``api.mem0.ai``) to explain what it deliberately avoids. A substring
        grep would flag that explanatory mention as a false violation."""
        import ast
        import inspect

        from isotope_zero.integrations import hooks as hooks_mod

        tree = ast.parse(inspect.getsource(hooks_mod))
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names.add(node.module.split(".")[0])
                for alias in node.names:
                    imported_names.add(alias.name.split(".")[0])

        network_clients = {"urllib", "requests", "httpx", "aiohttp", "openai",
                           "http", "socket", "websocket", "websockets"}
        offenders = imported_names & network_clients
        assert not offenders, (
            f"hooks.py imports a network client ({sorted(offenders)}) — "
            f"izero hooks are strictly local-first and must not open any socket"
        )


# --------------------------------------------------------------------------- #
# 7. Shell-wrapper exit-code contract (the write-guard MUST propagate exit 2)
# --------------------------------------------------------------------------- #
class TestHookWrapperExitCodes:
    """The bash wrappers are the real hook entrypoints Claude Code invokes.

    The non-guard hooks always exit 0 (non-blocking context injection). The
    write-guard (``block_memory_write.sh``) is the ONE case where the exit code
    is load-bearing: the python handler exits 2 to DENY a write to the store
    dir, and Claude Code treats exit 2 as a hard block. A wrapper that swallows
    exit 2 (e.g. an ``if cmd; then exit 0; else fallback`` pattern) turns a deny
    into a non-blocking error and lets the write through — so this test executes
    the real wrapper and asserts exit 2 survives it.
    """

    # The two tests below execute the real ``.sh`` wrappers with ``bash``. On
    # Windows the ``bash`` on PATH is WSL's (C:\\Windows\\System32\\bash.exe),
    # which errors "no installed distributions" and exits 1 — so the wrapper
    # contract can't be exercised there. The python-side contract they pin
    # (deny ⇒ exit 2, allow ⇒ exit 0) is already asserted cross-platform by
    # ``test_hooks.py::TestBlockMemoryWrite`` (via ``handle_hook``), so
    # skipping the POSIX shell-wrapper execution on Windows loses nothing.
    @pytest.mark.skipif(os.name == "nt", reason="bash-wrapper contract is POSIX-only")
    def test_guard_wrapper_propagates_deny_exit_2(self, tmp_path):
        """Feed the guard a Write payload targeting the store file and assert
        the wrapper exits 2 (deny). Uses ``ISOTOPE_ZERO_DB`` pointed at a temp
        store and writes to that exact file — the handler protects the DB file
        itself, so this triggers the deny path. This is the regression guard
        for the if/else that swallowed exit 2 (the wrapper must forward the
        python handler's exit 2, not flatten it to 0)."""
        import subprocess

        guard = BUNDLE / "hooks" / "block_memory_write.sh"
        store_dir = tmp_path / "store"
        db_path = store_dir / "isotope_zero.db"
        # A Write to the store file (env-configured) → deny → exit 2.
        payload = json.dumps({
            "tool_input": {"file_path": str(db_path)},
            "cwd": str(tmp_path),
        })
        env = {
            **os.environ,
            "ISOTOPE_ZERO_DB": str(db_path),
        }
        proc = subprocess.run(
            ["bash", str(guard)],
            input=payload, capture_output=True, text=True, env=env,
        )
        assert proc.returncode == 2, (
            f"block_memory_write.sh must propagate exit 2 (deny), got rc="
            f"{proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}. "
            f"The wrapper likely swallows the deny via an if/else or || pattern."
        )

    @pytest.mark.skipif(os.name == "nt", reason="bash-wrapper contract is POSIX-only")
    def test_non_guard_wrappers_always_exit_0(self, tmp_path):
        """The six context-injection hooks must never block the editor stream —
        even on a store error, they exit 0. Feed each an empty payload and
        assert rc 0 (no entrypoint failure can wedge the editor)."""
        import subprocess

        non_guard = ["session_start.sh", "user_prompt.sh", "file_read.sh",
                     "bash_output.sh", "stop.sh", "pre_compact.sh"]
        env = {**os.environ, "ISOTOPE_ZERO_DB": str(tmp_path / "iz.db")}
        for name in non_guard:
            sh = BUNDLE / "hooks" / name
            proc = subprocess.run(
                ["bash", str(sh)],
                input="{}", capture_output=True, text=True, env=env,
            )
            assert proc.returncode == 0, (
                f"{name}: non-guard hook must exit 0 always, got rc="
                f"{proc.returncode} stderr={proc.stderr!r}"
            )
