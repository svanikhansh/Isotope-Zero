"""Tests for the ``izero plugin install`` helper — ``isotope_zero.integrations.cli``.

The installer copies the bundled editor plugin and rewrites the MCP config(s)
so ``mcpServers.izero`` points at a Python interpreter that can launch the
local stdio MCP server. These tests lock in two contract guarantees:

- **No URL, no API key** (the headline local-first differentiator): the
  resolved MCP command is a local binary / ``python -m`` invocation, never an
  HTTP endpoint, and the rewritten configs carry no auth header.
- **Every shipped MCP config is rewritten** (regression guard): the bundle
  ships three — ``.mcp.json`` (Claude), ``.cursor-mcp.json`` (Cursor),
  ``.codex-mcp.json`` (Codex). The installer must rewrite *all three*, because
  skipping the editor variants leaves them pointing at ``izero-mcp`` when this
  install fell back to ``python -m`` — stranding Cursor/Codex with a command
  that doesn't resolve on their machine.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from isotope_zero.integrations import cli as plugin_cli
from isotope_zero.integrations.cli import (
    _MCP_CONFIGS,
    plugin_install,
)


BUNDLE = Path(__file__).resolve().parents[2] / "integrations" / "izero-plugin"


# --------------------------------------------------------------------------- #
# 1. the fallback path rewrites EVERY shipped MCP config (the regression guard)
# --------------------------------------------------------------------------- #
class TestRewritesAllMcpConfigs:
    """Finding #3: ``_install`` previously rewrote only ``.mcp.json``, leaving
    ``.cursor-mcp.json`` / ``.codex-mcp.json`` hardcoding ``izero-mcp``. On a
    machine where the install fell back to ``python -m`` (``izero-mcp`` not on
    PATH), Cursor and Codex would receive a command that fails to resolve.
    """

    @staticmethod
    def _force_python_module_fallback(monkeypatch):
        """Make ``_resolve_mcp_command`` return the ``python -m`` path even if
        ``izero-mcp`` IS on PATH (CI / dev machines), so the rewrite under
        test is the fallback rewrite, not a no-op."""

        def _fake_resolve():
            return sys.executable, list(plugin_cli._MCP_MODULE_ARGS)

        monkeypatch.setattr(plugin_cli, "_resolve_mcp_command", _fake_resolve)

    def test_dry_run_lists_all_three_mcp_configs_for_rewrite(self, capsys):
        """``--dry-run`` must report all three MCP configs in ``rewrite`` —
        if an editor variant is missing from that list, the install will skip
        it and strand that editor."""
        rc = plugin_install(target=None, editor="claude", dry_run=True)
        assert rc == 0
        plan = json.loads(capsys.readouterr().out)
        rewrite = set(plan.get("rewrite", []))
        for cfg in _MCP_CONFIGS:
            assert cfg in rewrite, (
                f"dry-run plan must list {cfg} for rewrite; got {sorted(rewrite)}. "
                f"Missing it means the install path would strand that editor."
            )
        # And the resolved command is local (no URL).
        cmd = plan["mcp"]["command"]
        assert not cmd.startswith("http"), (
            f"MCP command must be a local binary/path, not a URL: {cmd!r}"
        )

    def test_install_rewrites_all_three_configs_on_fallback(self, tmp_path, monkeypatch):
        """A real install on the ``python -m`` fallback path must rewrite all
        three MCP configs' ``mcpServers.izero.command`` to the resolved
        interpreter + args — not leave the editor variants at ``izero-mcp``."""
        self._force_python_module_fallback(monkeypatch)
        target = tmp_path / "plugins" / "izero"

        rc = plugin_install(target=str(target), editor="claude", dry_run=False)
        assert rc == 0, f"install failed (see stderr)"

        for cfg_name in _MCP_CONFIGS:
            cfg = target / cfg_name
            assert cfg.is_file(), f"{cfg_name}: should be installed"
            data = json.loads(cfg.read_text(encoding="utf-8"))
            izero = data["mcpServers"]["izero"]
            # The command must be the resolved interpreter (NOT the unchanged
            # `izero-mcp` console script that would strand a fallback install).
            assert izero["command"] == sys.executable, (
                f"{cfg_name}: command should be the resolved interpreter "
                f"{sys.executable!r} on the fallback path, got "
                f"{izero['command']!r}"
            )
            assert izero["args"] == list(plugin_cli._MCP_MODULE_ARGS)
            assert "ISOTOPE_ZERO_DB" in izero.get("env", {}), (
                f"{cfg_name}: rewritten config must stamp the store path"
            )
            # No network surface on ANY rewritten config.
            assert "url" not in izero
            assert "Authorization" not in izero
            assert "headers" not in izero

    def test_bundle_actually_ships_all_three_mcp_configs(self):
        """If this fails, the bundle lost an editor variant and the rewrite
        loop has nothing to rewrite for that editor — a silent coverage gap."""
        for cfg_name in _MCP_CONFIGS:
            assert (BUNDLE / cfg_name).is_file(), (
                f"bundle must ship {cfg_name} for the rewrite to cover it"
            )


# --------------------------------------------------------------------------- #
# 2. no-network surface (the local-first headline)
# --------------------------------------------------------------------------- #
class TestInstallerIsLocalOnly:
    def test_resolved_command_is_never_a_url(self, monkeypatch):
        """Both the console-script path and the fallback path resolve to a
        local command — never ``http(s)://``."""
        # console-script path (if izero-mcp is on PATH) — still not a URL.
        cmd, _ = plugin_cli._resolve_mcp_command()
        assert not cmd.startswith("http")

        # forced fallback path — also not a URL.
        def _fake():
            return sys.executable, list(plugin_cli._MCP_MODULE_ARGS)

        monkeypatch.setattr(plugin_cli, "_resolve_mcp_command", _fake)
        cmd, _ = plugin_cli._resolve_mcp_command()
        assert not cmd.startswith("http")
