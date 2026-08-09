"""Tests for the izero plugin skills — ``integrations/izero-plugin/skills/``.

Each skill is a thin ``SKILL.md`` that instructs the agent to call a real MCP
tool exposed by ``isotope_zero.mcp.server``. These tests assert the skill ↔
MCP-tool contract at the documentation layer: every skill is well-formed
frontmatter, its declared ``name`` matches its directory, its body names a real
MCP tool, and the argument shape the skill describes matches the tool's real
signature (``add_memory(tags=...)``, ``query_memory(token_budget=...)``, and the
``get_metrics`` → ``run_consolidation`` sweep pair).

YAML frontmatter is parsed without a dependency (split on ``---``; simple
``key: value`` lines) because the skills only use flat scalar frontmatter —
pulling in PyYAML for two keys would violate the zero-core-dep house style.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Bundle lives at repo-root/integrations/izero-plugin/skills. parents[2] climbs
# from src/tests/ up to the repo root, so the path resolves identically whether
# pytest runs from the repo root or from src/tests/.
SKILLS_DIR = Path(__file__).resolve().parents[2] / "integrations" / "izero-plugin" / "skills"

# The real MCP tool names exposed by isotope_zero.mcp.server. A skill is
# contract-correct only if its body references at least one of these — otherwise
# it instructs the agent to call a tool that doesn't exist.
_MCP_TOOLS = {
    "add_memory",
    "query_memory",
    "delete_memory",
    "get_metrics",
    "run_consolidation",
}

# The six shipped skills. Hardcoded (not discovered) so a missing or renamed
# skill fails loudly per-case instead of silently collecting zero parametrized
# cases — a stale SKILL_NAMES list is itself a signal to update this file.
SKILL_NAMES = ["remember", "recall", "forget", "peek", "stats", "decay-review"]


def _split_skill(text: str) -> tuple[dict[str, str], str]:
    """Split a ``SKILL.md`` into ``(frontmatter, body)``.

    Frontmatter is the leading ``---``-fenced block of flat ``key: value``
    pairs. We parse it without PyYAML: the skills only use scalar values, so a
    line-by-line split on the first colon is exact. Returns ``({}, text)`` if
    no well-formed frontmatter fence is present so callers can still inspect a
    malformed file's body rather than hard-crashing.
    """
    if not text.startswith("---"):
        return {}, text
    # maxsplit=2 → [before_open, frontmatter, body]. A later ``---`` in the body
    # (e.g. a markdown horizontal rule) lands untouched in the body slice.
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return {}, text
    frontmatter: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        frontmatter[key.strip()] = value.strip()
    return frontmatter, parts[2]


def _read_skill(name: str) -> tuple[dict[str, str], str]:
    """Return ``(frontmatter, body)`` for ``skills/<name>/SKILL.md``."""
    text = (SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")
    return _split_skill(text)


# --------------------------------------------------------------------------- #
# Frontmatter well-formedness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", SKILL_NAMES)
def test_each_skill_has_name_and_description(name):
    """A skill with no name can't be invoked; no description can't be indexed."""
    frontmatter, _ = _read_skill(name)
    assert frontmatter.get("name"), f"{name}: frontmatter `name` is missing or empty"
    assert frontmatter.get("description"), (
        f"{name}: frontmatter `description` is missing or empty"
    )


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_skill_names_match_directory(name):
    """The frontmatter ``name`` must equal the directory name so the slash
    command (``/izero:<name>``) and the on-disk skill resolve to the same thing."""
    frontmatter, _ = _read_skill(name)
    assert frontmatter.get("name") == name, (
        f"{name}: frontmatter name={frontmatter.get('name')!r} != directory {name!r}"
    )


# --------------------------------------------------------------------------- #
# Skill ↔ MCP tool contract: every skill names a real tool
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", SKILL_NAMES)
def test_skills_reference_valid_tools(name):
    """Each skill body must mention at least one real MCP tool name — a skill
    that names a non-existent tool would instruct the agent to call into a void."""
    _, body = _read_skill(name)
    referenced = [tool for tool in _MCP_TOOLS if tool in body]
    assert referenced, (
        f"{name}: SKILL.md body references none of the real MCP tools "
        f"{sorted(_MCP_TOOLS)}"
    )


# --------------------------------------------------------------------------- #
# Argument-shape spot checks: the skills' documented args match tool signatures
# --------------------------------------------------------------------------- #
def test_remember_skill_documents_tags_arg():
    """``remember`` passes ``tags=[...]`` to ``add_memory``; the skill must
    document that ``tags`` kwarg so callers and the capture hooks share one
    argument shape with the server's real ``add_memory(content, tags=None)``."""
    _, body = _read_skill("remember")
    assert "tags" in body, (
        "remember: SKILL.md must document the `tags` arg of add_memory"
    )


def test_recall_skill_documents_token_budget():
    """``recall`` passes ``token_budget=300`` to ``query_memory``; the skill
    must document it so the recall budget is discoverable alongside the tool's
    real ``query_memory(query, token_budget=300)`` signature."""
    _, body = _read_skill("recall")
    assert "token_budget" in body, (
        "recall: SKILL.md must document the `token_budget` arg of query_memory"
    )


def test_decay_review_calls_both_metrics_and_consolidation():
    """``decay-review`` is a two-step sweep: ``get_metrics`` (pre-sweep
    baseline) then ``run_consolidation`` (the sweep). The skill must name both
    — naming only one would instruct a half-run (report without sweeping, or
    sweep without a baseline to compare against)."""
    _, body = _read_skill("decay-review")
    assert "get_metrics" in body, (
        "decay-review: SKILL.md must call get_metrics for the pre-sweep baseline"
    )
    assert "run_consolidation" in body, (
        "decay-review: SKILL.md must call run_consolidation for the sweep"
    )
