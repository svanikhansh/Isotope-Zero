"""Tests for the Claude Code lifecycle hook engine — ``isotope_zero.integrations.hooks``.

The hook engine is the testable core behind the ``izero hook <event>`` subcommand and
the ``integrations/izero-plugin/hooks/*.sh`` bash wrappers. Every handler is a pure
function of ``(payload, db_path)`` that returns the Claude Code hook-output JSON shape
and calls the existing ``IsotopeZeroServer`` methods. **No network anywhere** — these
tests assert the local-first guarantee directly.

These tests feed canned hook JSON (a ``Read`` tool_input with a file_path; a ``Bash``
tool_response with a traceback; a ``Write`` against the store dir; a ``Stop`` payload
with an inline transcript) and assert the emitted JSON shape + the store side effects
against a temp DB — never needing a live Claude Code.
"""
from __future__ import annotations

import json
import os
import re

import pytest

from isotope_zero.integrations import hooks
from isotope_zero.integrations.hooks import handle_hook


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _seed_card(db_path: str, fact: str, tags: list[str] | None = None) -> str:
    """Add a card to a temp store via the real server seam, return its id.

    Exercises the same ``_server`` + ``add_memory(tags=...)`` path the hooks use,
    so the tag-lookup + dedup assertions hit a store that mirrors production.
    """
    server = hooks._server(db_path)
    try:
        res = server.add_memory(fact, tags=tags or [])
        # add_memory returns {"memory_id": ..., "action": ...}
        return res.get("memory_id", "")
    finally:
        hooks._close_quietly(server)


def _query(db_path: str, query: str) -> list[dict]:
    """Run a real query_memory against a temp store, return the hits list."""
    server = hooks._server(db_path)
    try:
        res = server.query_memory(query, token_budget=200)
        return res.get("hits", []) if isinstance(res, dict) else []
    finally:
        hooks._close_quietly(server)


def _make_file(tmp_path, name: str, size_bytes: int = 2000) -> str:
    """A real on-disk file big enough to clear the file-read gate (1500 B)."""
    p = tmp_path / name
    p.write_text("x" * size_bytes)
    return str(p)


def _ctx(result):
    """Pull the hookSpecificOutput dict out of a handler result."""
    return (result or {}).get("hookSpecificOutput", {})


# --------------------------------------------------------------------------- #
# 1. file_read — prior-work timeline injection (the flagship hook)
# --------------------------------------------------------------------------- #
class TestFileRead:
    def test_returns_timeline_for_file_with_tagged_prior_work(self, tmp_path):
        db = str(tmp_path / "iz.db")
        # Seed a card tagged with file:<relpath> for the file we're about to read.
        fpath = _make_file(tmp_path, "auth.py")
        rel = os.path.relpath(fpath, str(tmp_path))
        _seed_card(db, "We decided JWT for auth in this file.", tags=[f"file:{rel}"])

        payload = {"tool_input": {"file_path": fpath}, "cwd": str(tmp_path)}
        rc = handle_hook("file_read", payload, db)
        assert rc == 0  # PreToolUse:Read never blocks (permissionDecision allow)

        # The engine emits via stdout only on the CLI path (_emit=True); the pure
        # handle_hook path returns the dict in the result and emits nothing. Re-run
        # the handler directly to inspect the returned JSON.
        result = hooks._on_file_read(payload, db)
        ctx = _ctx(result)
        assert ctx.get("hookEventName") == "PreToolUse"
        assert ctx.get("permissionDecision") == "allow"
        timeline = ctx.get("additionalContext", "")
        assert "prior work" in timeline
        assert "JWT" in timeline  # the seeded fact surfaced

    def test_no_context_for_tiny_file_under_gate(self, tmp_path):
        db = str(tmp_path / "iz.db")
        fpath = _make_file(tmp_path, "tiny.txt", size_bytes=200)  # < 1500 gate
        payload = {"tool_input": {"file_path": fpath}, "cwd": str(tmp_path)}
        result = hooks._on_file_read(payload, db)
        assert result is None  # below the gate → no round-trip, no context

    def test_no_context_for_missing_file(self, tmp_path):
        db = str(tmp_path / "iz.db")
        payload = {"tool_input": {"file_path": "/nonexistent/path/x.py"}, "cwd": str(tmp_path)}
        assert hooks._on_file_read(payload, db) is None

    def test_never_blocks_read_even_with_store_populated(self, tmp_path):
        db = str(tmp_path / "iz.db")
        fpath = _make_file(tmp_path, "svc.py")
        _seed_card(db, "a prior fact about svc.py", tags=[f"file:{os.path.relpath(fpath, str(tmp_path))}"])
        payload = {"tool_input": {"file_path": fpath}, "cwd": str(tmp_path)}
        assert handle_hook("file_read", payload, db) == 0  # allow, never deny


# --------------------------------------------------------------------------- #
# 2. bash_output — gated auto-capture of anti-pattern errors
# --------------------------------------------------------------------------- #
class TestBashOutput:
    TRACEBACK_PAYLOAD = {
        "tool_input": {"command": "python -m foo.run"},
        "tool_response": (
            "Traceback (most recent call last):\n"
            '  File "src/foo.py", line 42, in <module>\n'
            "    raise ValueError('bad input')\n"
            "ValueError: bad input\n"
        ),
        "cwd": "/tmp",
    }

    def test_captures_traceback_as_anti_pattern_card(self, tmp_path):
        db = str(tmp_path / "iz.db")
        # FIRST capture of this error signature → writes a NEW anti_pattern card
        # and emits the "captured error" context. Call the handler once only;
        # a second call would be a recurrence (touch → "boosted" message).
        result = hooks._on_bash_output(self.TRACEBACK_PAYLOAD, db)
        ctx = _ctx(result)
        assert ctx.get("hookEventName") == "PostToolUse"
        assert "captured error" in ctx.get("additionalContext", "").lower()
        # The card actually landed in the store, tagged anti_pattern + error sig.
        hits = _query(db, "ValueError bad input")
        assert hits, "captured error should be recallable"
        server = hooks._server(db)
        try:
            tagged = server.store.sql_lookup("tags", "type:anti_pattern")
        finally:
            hooks._close_quietly(server)
        assert tagged, "captured card should carry the type:anti_pattern tag"

    def test_dedup_touches_existing_error_under_cap(self, tmp_path):
        db = str(tmp_path / "iz.db")
        # First capture: writes a new card.
        hooks._on_bash_output(self.TRACEBACK_PAYLOAD, db)
        server = hooks._server(db)
        try:
            after_first = len(server.store.sql_lookup("tags", "type:anti_pattern"))
        finally:
            hooks._close_quietly(server)
        # Second capture of the SAME error signature → touch, not a duplicate add.
        hooks._on_bash_output(self.TRACEBACK_PAYLOAD, db)
        server = hooks._server(db)
        try:
            after_second = len(server.store.sql_lookup("tags", "type:anti_pattern"))
        finally:
            hooks._close_quietly(server)
        # Under the cap (3), a recurrence is a touch — count should not grow
        # beyond the single first-occurrence card for this signature.
        assert after_second == after_first == 1

    def test_count_gate_stops_spam_at_cap_limit(self, tmp_path):
        db = str(tmp_path / "iz.db")
        # Hammer the same error signature well past any cap.
        for _ in range(hooks._BASH_ERROR_CAP_LIMIT + 3):
            hooks._on_bash_output(self.TRACEBACK_PAYLOAD, db)
        server = hooks._server(db)
        try:
            cards = server.store.sql_lookup("tags", "type:anti_pattern")
        finally:
            hooks._close_quietly(server)
        # Dedup guarantee: exactly ONE card per error signature, no matter how many
        # times it recurs — a crashloop can't spam the store. Recurrences touch the
        # existing card (vitality boost), they don't add duplicates.
        assert len(cards) == 1, (
            f"expected exactly 1 anti_pattern card for one error signature, "
            f"got {len(cards)} — dedup is leaking duplicates"
        )

    def test_skips_git_commands(self, tmp_path):
        db = str(tmp_path / "iz.db")
        payload = {
            "tool_input": {"command": "git merge main"},
            "tool_response": "CONFLICT (content): Merge conflict in foo.py\n" + "x" * 60,
            "cwd": str(tmp_path),
        }
        result = hooks._on_bash_output(payload, db)
        assert result is None  # git noise is skipped, no capture
        server = hooks._server(db)
        try:
            assert server.store.sql_lookup("tags", "type:anti_pattern") == []
        finally:
            hooks._close_quietly(server)

    def test_no_capture_for_clean_output(self, tmp_path):
        db = str(tmp_path / "iz.db")
        payload = {
            "tool_input": {"command": "echo hello"},
            "tool_response": "hello\n" + "x" * 60,
            "cwd": str(tmp_path),
        }
        assert hooks._on_bash_output(payload, db) is None


# --------------------------------------------------------------------------- #
# 3. block_memory_write — the one hook that may exit 2 (deny)
# --------------------------------------------------------------------------- #
class TestBlockMemoryWrite:
    def test_blocks_write_into_isotope_zero_store_dir(self, tmp_path):
        db = str(tmp_path / "iz.db")
        payload = {"tool_input": {"file_path": os.path.expanduser("~/.isotope_zero/anything.db")}}
        result = hooks._block_memory_write(payload, db)
        ctx = _ctx(result)
        assert ctx.get("permissionDecision") == "deny"
        assert "add_memory" in ctx.get("permissionDecisionReason", "").lower()
        # And handle_hook surfaces this as exit 2 (the Claude Code block signal).
        assert handle_hook("block_memory_write", payload, db) == 2

    def test_allows_write_outside_store_dir(self, tmp_path):
        db = str(tmp_path / "iz.db")
        payload = {"tool_input": {"file_path": str(tmp_path / "normal.txt")}}
        result = hooks._block_memory_write(payload, db)
        assert _ctx(result).get("permissionDecision") == "allow"
        assert handle_hook("block_memory_write", payload, db) == 0

    def test_missing_file_path_allows(self, tmp_path):
        db = str(tmp_path / "iz.db")
        result = hooks._block_memory_write({"tool_input": {}}, db)
        assert _ctx(result).get("permissionDecision") == "allow"

    def test_guards_env_relocated_store_file(self, tmp_path):
        """A user who points ``ISOTOPE_ZERO_DB`` at a custom location must get
        THAT store file guarded — not just the conventional ``~/.isotope_zero/``.
        The handler derives the protected paths from ``db_path`` (the file +
        its SQLite sidecars), so a relocated store can't be sneak-corrupted by
        a direct write to the (custom-located) DB file. (Regression guard: an
        earlier version hardcoded only ``~/.isotope_zero/`` and would have
        allowed this write to a relocated store.)"""
        store_dir = tmp_path / "custom_store"
        db = str(store_dir / "isotope_zero.db")  # ISOTOPE_ZERO_DB would resolve here
        payload = {"tool_input": {"file_path": db}}  # write to the store file itself
        result = hooks._block_memory_write(payload, db)
        assert _ctx(result).get("permissionDecision") == "deny"
        assert handle_hook("block_memory_write", payload, db) == 2

    def test_guards_env_relocated_store_sidecar(self, tmp_path):
        """The SQLite write-ahead-log sidecar (``<stem>-wal``) is part of the
        live store — corrupting it corrupts the store. A relocated store's
        sidecar must be guarded just like the main file."""
        store_dir = tmp_path / "custom_store"
        db = str(store_dir / "isotope_zero.db")
        payload = {"tool_input": {"file_path": str(store_dir / "isotope_zero-wal")}}
        result = hooks._block_memory_write(payload, db)
        assert _ctx(result).get("permissionDecision") == "deny"


# --------------------------------------------------------------------------- #
# 4. user_prompt — memory-grounded prompts
# --------------------------------------------------------------------------- #
class TestUserPrompt:
    def test_injects_relevant_memories_when_store_has_hits(self, tmp_path):
        db = str(tmp_path / "iz.db")
        _seed_card(db, "We decided to use uv for python env resolution.")
        payload = {"prompt": "how should we manage the python environment for this project?"}
        result = hooks._on_user_prompt(payload, db)
        ctx = _ctx(result)
        assert ctx.get("hookEventName") == "UserPromptSubmit"
        assert "relevant memories" in ctx.get("additionalContext", "")
        # The seeded decision surfaced.
        assert "uv" in ctx.get("additionalContext", "")

    def test_skips_short_prompts(self, tmp_path):
        db = str(tmp_path / "iz.db")
        _seed_card(db, "some fact that should not be injected for a one-word reply")
        result = hooks._on_user_prompt({"prompt": "ok"}, db)
        assert result is None  # < 20 chars → no round-trip

    def test_no_context_when_store_empty(self, tmp_path):
        db = str(tmp_path / "iz.db")
        payload = {"prompt": "what do we know about the auth architecture here?"}
        result = hooks._on_user_prompt(payload, db)
        assert result is None  # empty store → nothing to inject


# --------------------------------------------------------------------------- #
# 5. session_start — one-line store summary
# --------------------------------------------------------------------------- #
class TestSessionStart:
    def test_injects_summary_when_store_populated(self, tmp_path):
        db = str(tmp_path / "iz.db")
        _seed_card(db, "a durable decision worth surfacing at session start")
        result = hooks._on_session_start({"cwd": str(tmp_path)}, db)
        ctx = _ctx(result)
        assert ctx.get("hookEventName") == "SessionStart"
        summary = ctx.get("additionalContext", "")
        assert "1 memories" in summary  # the one seeded card
        assert "/izero:recall" in summary  # the affordance hint

    def test_no_context_when_store_empty(self, tmp_path):
        db = str(tmp_path / "iz.db")
        result = hooks._on_session_start({"cwd": str(tmp_path)}, db)
        assert result is None  # empty store → nothing to recall


# --------------------------------------------------------------------------- #
# 6. stop / pre_compact — local-heuristic session summary (no LLM)
# --------------------------------------------------------------------------- #
class TestStopAndPreCompact:
    TRANSCRIPT = (
        "User: let's set up the build\n"
        "Assistant: we decided to use uv for the python environment resolution. "
        "It's faster than pip and handles venvs natively.\n"
        "User: great\n"
        "Assistant: the convention is to always pin the python version in "
        "pyproject.toml so CI is reproducible.\n"
    )

    def test_stop_captures_decision_signals_from_transcript(self, tmp_path):
        db = str(tmp_path / "iz.db")
        payload = {"transcript": self.TRANSCRIPT, "cwd": str(tmp_path)}
        result = hooks._on_stop(payload, db)
        ctx = _ctx(result)
        assert ctx.get("hookEventName") == "Stop"
        assert "captured session summary" in ctx.get("additionalContext", "").lower()
        # The decision signals actually persisted as cards, tagged session.
        server = hooks._server(db)
        try:
            tagged = server.store.sql_lookup("tags", "session")
        finally:
            hooks._close_quietly(server)
        assert tagged, "session-summary cards should be persisted"
        # The 'uv' decision should be recallable.
        assert any("uv" in (h.get("fact", "") or "").lower() for h in _query(db, "uv python"))

    def test_pre_compact_tags_pre_compaction_distinctly(self, tmp_path):
        db = str(tmp_path / "iz.db")
        payload = {"transcript": self.TRANSCRIPT, "cwd": str(tmp_path)}
        result = hooks._on_pre_compact(payload, db)
        ctx = _ctx(result)
        assert ctx.get("hookEventName") == "PreCompact"
        assert "pre-compaction" in ctx.get("additionalContext", "").lower()
        server = hooks._server(db)
        try:
            tagged = server.store.sql_lookup("tags", "pre-compaction")
        finally:
            hooks._close_quietly(server)
        assert tagged, "pre-compaction cards should carry the pre-compaction tag"

    def test_thin_transcript_is_a_noop(self, tmp_path):
        db = str(tmp_path / "iz.db")
        payload = {"transcript": "short", "cwd": str(tmp_path)}
        assert hooks._on_stop(payload, db) is None  # < 100 chars → no capture


# --------------------------------------------------------------------------- #
# 7. dispatch + robustness — unknown events, malformed stdin, never-block
# --------------------------------------------------------------------------- #
class TestDispatchRobustness:
    def test_unknown_event_is_noop_exit_0(self, tmp_path):
        db = str(tmp_path / "iz.db")
        assert handle_hook("totally_made_up_event", {"x": 1}, db) == 0

    def test_handler_exception_never_breaks_stream(self, tmp_path, monkeypatch):
        db = str(tmp_path / "iz.db")
        # Force the server seam to raise — a store failure must not propagate.
        def _boom(_db_path):
            raise RuntimeError("store exploded")
        monkeypatch.setattr(hooks, "_server", _boom)
        # handle_hook swallows the exception and returns 0 (never breaks the stream).
        assert handle_hook("session_start", {"cwd": str(tmp_path)}, db) == 0

    def test_supported_events_dispatch_table_complete(self):
        # Every event in SUPPORTED_EVENTS has a registered handler and vice-versa.
        assert set(hooks.SUPPORTED_EVENTS) == set(hooks._HANDLERS)

    def test_handle_hook_emit_true_writes_valid_json_to_stdout(self, tmp_path, capsys):
        db = str(tmp_path / "iz.db")
        _seed_card(db, "a fact for session start")
        handle_hook("session_start", {"cwd": str(tmp_path)}, db, _emit=True)
        captured = capsys.readouterr()
        assert captured.out.strip(), "emit path should print JSON to stdout"
        obj = json.loads(captured.out.strip())
        assert "hookSpecificOutput" in obj


# --------------------------------------------------------------------------- #
# 8. fresh-install persistence — captures survive a brand-new install
# --------------------------------------------------------------------------- #
class TestFreshInstallPersistence:
    """On a machine that has NEVER created a store, a capture hook must still
    write to a persistent on-disk path — NOT to a discarded ``:memory:`` DB.

    The failure mode this guards: hook ``command:`` subprocesses don't inherit
    the ``.mcp.json`` ``env`` block (Claude Code applies that only to the MCP
    stdio server, not to hooks), so when ``ISOTOPE_ZERO_DB`` is unset the hook
    engine falls back to ``_default_db_path()``. If that retreats to
    ``:memory:`` (the old behavior, when the store file didn't exist yet), a
    capture on a fresh install silently vanishes the moment the subprocess
    exits. This is the single most damaging failure for a "memory" product —
    the user adds a fact, it's confirmed, and it's gone.
    """

    def test_default_db_path_is_file_backed_even_when_store_missing(self):
        """``_default_db_path()`` returns a real file path, not ``:memory:``,
        so a fresh install's captures land on disk where they persist."""
        path = hooks._default_db_path()
        assert path != ":memory:", (
            "_default_db_path() must return a file-backed path even when the "
            "store doesn't exist yet — :memory: would discard fresh-install "
            "captures"
        )
        assert path.endswith(".db"), (
            f"_default_db_path() should resolve to a .db file, got {path!r}"
        )

    def test_capture_with_no_env_no_store_persists_to_disk(self, tmp_path, monkeypatch):
        """End-to-end: with no ``ISOTOPE_ZERO_DB`` and no existing store, a
        ``bash_output`` capture must still land in a real file we can reopen.

        Points ``HOME`` at a temp dir so ``_default_db_path()``'s
        ``~/.isotope_zero/`` resolves inside the sandbox (no touching the real
        user home), runs the capture, then reopens the resolved path in a fresh
        process to prove the card survived the subprocess boundary."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("ISOTOPE_ZERO_DB", raising=False)
        # The default path the hook engine will resolve, under the temp HOME.
        expected_path = str(tmp_path / ".isotope_zero" / "isotope_zero.db")
        assert not os.path.exists(expected_path)  # genuinely fresh

        payload = {
            "tool_input": {"command": "python -c 'raise RuntimeError(\"boom\")'"},
            "tool_response": (
                "Traceback (most recent call last):\n"
                "  File \"<string>\", line 1, in <module>\n"
                "RuntimeError: boom\n"
            ),
            "cwd": str(tmp_path),
        }
        # handle_hook with no db_path → _default_db_path() → the temp-HOME file.
        rc = handle_hook("bash_output", payload, db_path=None)
        assert rc == 0
        # The capture must have landed in the file-backed store (created on
        # first write), not a discarded :memory: DB.
        assert os.path.exists(expected_path), (
            f"capture should have created {expected_path} on disk; if it "
            f"wrote to :memory: instead, fresh-install captures are lost"
        )
        # Reopen a fresh server against that path and prove the card is there.
        server = hooks._server(expected_path)
        try:
            hits = server.query_memory("RuntimeError boom", token_budget=200)
        finally:
            hooks._close_quietly(server)
        assert hits.get("hits"), (
            "the captured error card must be reopenable from the persisted file"
        )


# --------------------------------------------------------------------------- #
# 9. offline guarantee — the headline differentiator vs mem0 (no network)
# --------------------------------------------------------------------------- #
class TestNoNetwork:
    """The integrations module must NEVER import a network client.

    mem0's hook scripts each carry their own ``urllib.request.urlopen`` call to
    ``api.mem0.ai``. Our headline guarantee is the inversion: every hook reads
    and writes the local SQLite store, full stop. The verification step greps
    the integrations tree for ``urllib``/``httpx``/``requests``/``openai`` and
    expects zero hits; this test asserts the same property at the source-text
    level so a future PR that sneaks in an ``import requests`` fails CI here,
    not in a downstream grep.
    """

    # Module names whose presence in an import statement would mean a network
    # client crept in. ``urllib`` covers both ``urllib`` and ``urllib.request``;
    # the others are the cloud-LLM / HTTP-client libraries mem0 depends on.
    _FORBIDDEN = ("urllib", "httpx", "requests", "openai")

    @staticmethod
    def _hooks_source() -> str:
        """Read the hook engine source from disk (not via import-time introspection).

        We read the file the module was loaded from so the assertion reflects
        the actual bytes on disk — the same surface the grep step scans —
        rather than the post-import module namespace, which would miss a
        conditionally-imported client that the import guard hid.
        """
        src_path = os.path.normpath(hooks.__file__)
        with open(src_path, "r", encoding="utf-8") as fh:
            return fh.read()

    def test_no_network_imports_in_hooks_source(self):
        """No ``import``/``from`` line in ``hooks.py`` names a network client.

        Scans each import statement for a forbidden module token. We restrict
        to import lines (rather than banning the bare substring everywhere)
        because the module docstring deliberately *discusses*
        ``urllib.request.urlopen`` as the anti-pattern being inverted — a naive
        whole-file substring ban would false-positive on that prose. The
        import statement is the only place a network client could actually
        bind a name, so it is the correct surface to scan.
        """
        source = self._hooks_source()
        offenders: list[str] = []
        for line in source.splitlines():
            stripped = line.strip()
            if not (
                stripped.startswith("import ") or stripped.startswith("from ")
            ):
                continue
            # Tokenize on whitespace/dot/paren/comma so ``urllib.request`` and
            # ``from urllib.request import ...`` both surface the ``urllib``
            # token, while ``requests_foo`` (a longer, unrelated name) does not.
            tokens = re.split(r"[\s.():,]+", stripped)
            for forbidden in self._FORBIDDEN:
                if any(token == forbidden for token in tokens):
                    offenders.append(stripped)
                    break
        assert not offenders, (
            f"hooks.py must not import a network client (the mem0-inversion "
            f"guarantee), but found these import lines: {offenders!r}"
        )
