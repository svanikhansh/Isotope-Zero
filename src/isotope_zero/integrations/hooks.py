"""Claude Code lifecycle hook engine — the testable, local-only core.

All bash hook wrappers (``integrations/izero-plugin/hooks/<event>.sh``) pipe
their stdin JSON to ONE CLI subcommand, ``izero hook <event>``, which calls
:func:`run_hook` here. Every event handler is a pure function of
``(payload: dict, db_path: str) -> dict`` that returns the Claude Code
hook-output JSON shape and calls the existing :class:`IsotopeZeroServer`
methods (add/query/delete/metrics/consolidation). **No network calls anywhere.**

Hook output contract (Claude Code):

    {"hookSpecificOutput": {
        "hookEventName": "PreToolUse" | "PostToolUse" | ...,
        "additionalContext": "...",          # text injected into the agent
        "permissionDecision": "allow"|"deny" # for PreToolUse guards
    }}

Design (inverts mem0's anti-pattern):
    mem0 has ~15 separate Python scripts in ``scripts/``, each with its own
    ``sys.path.insert``, identity resolution, and ``urllib.request.urlopen`` to
    ``api.mem0.ai``. We collapse ALL of these into one entrypoint with a
    dispatch table; the bash wrappers are 3-line scripts. This makes the hooks
    trivially testable without a live Claude Code (feed canned JSON, assert the
    emitted JSON) and keeps memory on the local store.

Local-heuristic guarantee:
    mem0's ``Stop``/``PreCompact`` call a cloud LLM summarizer. Ours cannot —
    so session summary uses :mod:`isotope_zero.extraction.fact_extractor`
    (local ADD/UPDATE/DELETE triage + compression, no LLM) on the transcript
    payload, falling back to cards touched in the session time-window.
"""
from __future__ import annotations

import json
import os
import re
import sys
import logging
from typing import Any, Callable

log = logging.getLogger("isotope_zero.integrations.hooks")

# Events the dispatch table recognizes. Bash wrappers pass one of these as the
# ``<event>`` arg to ``izero hook <event>``. Unknown events are a no-op (exit
# 0, never block) so a forward-compatible editor payload never breaks a hook.
SUPPORTED_EVENTS: tuple[str, ...] = (
    "session_start",
    "user_prompt",
    "file_read",
    "block_memory_write",
    "bash_output",
    "stop",
    "pre_compact",
)

# Minimum file size (bytes) before the file-context hook injects prior work.
# Mirrors mem0's ``FILE_READ_GATE_MIN_BYTES``: tiny config/readme files don't
# carry enough context to be worth a store round-trip.
_FILE_READ_MIN_BYTES = 1500

# Bash-error capture gate: stop auto-capturing after this many occurrences of
# the same error signature, switching to touch-only (vitality boost). Bounded
# by consolidation (run_consolidation) deduping near-identical cards anyway.
_BASH_ERROR_CAP_LIMIT = 3

# Traceback / fatal-error patterns scanned for in Bash tool_response. A hit
# triggers the gated auto-capture path. Compiled once at import.
_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\)|panic:|FATAL:|error\[[A-Z]+\d*\]|"
    r"(?:^|\n)\s*(?:Error|Exception|RuntimeError|ValueError|TypeError|"
    r"AttributeError|KeyError|ImportError|SyntaxError|ZeroDivisionError):",
    re.MULTILINE,
)

# Git commands whose output often contains stderr that looks like errors but
# isn't actionable (merge conflicts, rebase state) — skip capture for these.
_GIT_NOISE_RE = re.compile(r"\bgit\s+(commit|merge|rebase|cherry-pick|stash)\b")


# --------------------------------------------------------------------------- #
# Public entrypoints
# --------------------------------------------------------------------------- #
def run_hook(event: str, db_path: str | None = None) -> int:
    """Stdin-driven CLI entrypoint: read hook JSON, dispatch, print, exit 0.

    Never raises for a bad payload or an unknown event — a hook failure must
    never break the editor's tool stream. Errors are logged to stderr and the
    function returns 0 (except the write-guard, which returns 2 to block, per
    the Claude Code PreToolUse ``exit 2 == block`` convention).

    ``db_path`` defaults to ``$ISOTOPE_ZERO_DB`` (set by the plugin's
    ``.mcp.json`` env) so the hook reads/writes the user's persistent store.
    """
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception as exc:  # noqa: BLE001 — never block on malformed input
        log.debug("hook %s: unparseable stdin (%s); no-op", event, exc)
        return 0

    if db_path is None:
        db_path = os.environ.get("ISOTOPE_ZERO_DB") or _default_db_path()

    rc = handle_hook(event, payload, db_path, _emit=True)
    return rc if rc is not None else 0


def handle_hook(
    event: str,
    payload: dict[str, Any],
    db_path: str | None = None,
    *,
    _emit: bool = False,
) -> int:
    """Pure dispatch core: ``(event, payload, db_path) -> exit_code``.

    Tests call this directly (``_emit=False``) and assert via the return value
    / the store, never needing a live Claude Code. When ``_emit=True`` (the CLI
    path), the handler's JSON is printed to stdout in the Claude Code shape.

    ``db_path`` defaults to ``None``, which resolves the same way ``run_hook``
    does (``$ISOTOPE_ZERO_DB`` then ``_default_db_path()``) — so a test that
    omits it exercises the real default-resolution flow (the fresh-install
    path), rather than a contrived explicit path.

    Returns the process exit code: 0 for allow/no-op, 2 for the write-guard
    block. Unknown events return 0.
    """
    if db_path is None:
        db_path = os.environ.get("ISOTOPE_ZERO_DB") or _default_db_path()
    handler = _HANDLERS.get(event)
    if handler is None:
        log.debug("hook %s: unknown event; no-op", event)
        return 0
    try:
        result = handler(payload, db_path)
    except Exception as exc:  # noqa: BLE001 — never break the editor stream
        log.warning("hook %s failed (%s); no-op", event, exc)
        return 0
    if _emit and result is not None:
        sys.stdout.write(json.dumps(result))
        sys.stdout.write("\n")
    # The write-guard signals a block via a sentinel; all others allow.
    return _exit_code_for(result)


def _exit_code_for(result: dict[str, Any] | None) -> int:
    """Claude Code exit-code convention: PreToolUse deny => exit 2 (block)."""
    if result is None:
        return 0
    hso = result.get("hookSpecificOutput") or {}
    if hso.get("permissionDecision") == "deny":
        return 2
    return 0


def _default_db_path() -> str:
    """The persistent local store path, mirroring the CLI default.

    Returns ``~/.isotope_zero/isotope_zero.db`` unconditionally — even when the
    file doesn't exist yet. A fresh install (no store ever created) MUST still
    hit a file-backed path so captured cards persist: hook subprocesses don't
    inherit the ``.mcp.json`` env block (Claude Code applies that only to the
    MCP stdio server subprocess, not to hook ``command:`` subprocesses), so a
    retreat to ``:memory:`` here would silently discard every capture on a
    brand-new machine. ``IsotopeZeroServer.__init__`` makedirs the parent on
    first use, so returning a not-yet-existent file path is safe.
    """
    return os.path.expanduser("~/.isotope_zero/isotope_zero.db")


# --------------------------------------------------------------------------- #
# Hook output builders (the Claude Code JSON shape)
# --------------------------------------------------------------------------- #
def _context(event_name: str, additional_context: str) -> dict[str, Any]:
    """Build an ``additionalContext`` hook output (the common case)."""
    return {
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": additional_context,
            "permissionDecision": "allow",
        }
    }


def _allow(event_name: str) -> dict[str, Any]:
    """PreToolUse allow (no additional context)."""
    return {"hookSpecificOutput": {"hookEventName": event_name, "permissionDecision": "allow"}}


def _deny(event_name: str, reason: str) -> dict[str, Any]:
    """PreToolUse deny — blocks the tool call and surfaces the reason."""
    return {
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


# --------------------------------------------------------------------------- #
# Server construction (the testable seam)
# --------------------------------------------------------------------------- #
def _server(db_path: str):
    """Construct an ``IsotopeZeroServer`` for the hook's store.

    Imported lazily so :mod:`isotope_zero.integrations` stays cheap to import
    (no onnxruntime at import time) and so tests can monkeypatch this seam.
    The server defaults its embedder to the in-process ``EmbeddingEngine``,
    which itself falls back to deterministic pseudo-embeddings when onnxruntime
    is absent — so hooks work with zero optional deps, degraded semantic
    quality only.
    """
    from ..mcp.server import IsotopeZeroServer

    return IsotopeZeroServer(db_path=db_path)


# --------------------------------------------------------------------------- #
# Helpers — paths, tags, error signatures
# --------------------------------------------------------------------------- #
def _relative_path(file_path: str, cwd: str) -> str:
    """Repo-relative path for ``file:<rel>`` tags (falls back to basename)."""
    try:
        rel = os.path.relpath(file_path, cwd)
    except (ValueError, TypeError):
        rel = os.path.basename(file_path) or file_path
    return rel.replace(os.sep, "/")


def _git_branch(cwd: str) -> str | None:
    """Best-effort current git branch (None if not a repo / detached)."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            name = out.stdout.strip()
            if name and name != "HEAD":
                return name
    except Exception:  # noqa: BLE001
        pass
    return None


def _error_signature(error_line: str, command: str) -> str:
    """A stable dedup key for a captured error (hash of class + first 120c)."""
    import hashlib

    key = f"{command[:80]}::{error_line[:120]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _extract_error_line(output: str) -> str | None:
    """First salient error line from a traceback'd output (None if none)."""
    for line in output.splitlines():
        stripped = line.strip()
        if re.match(r"^(Error|Exception|RuntimeError|ValueError|TypeError|"
                    r"AttributeError|KeyError|ImportError|SyntaxError|"
                    r"ZeroDivisionError|AssertionError|panic:|FATAL:)", stripped):
            return stripped[:200]
    if "Traceback (most recent call last)" in output:
        return "Traceback (most recent call last)"
    return None


def _extract_trace_files(output: str) -> list[str]:
    """File paths named in a traceback (best-effort, for card evidence)."""
    return list(dict.fromkeys(re.findall(r'File "([^"]+\.py)", line', output)))[:5]


# --------------------------------------------------------------------------- #
# Event handlers (event -> (payload, db_path) -> hook-output JSON)
# --------------------------------------------------------------------------- #
def _on_file_read(payload: dict[str, Any], db_path: str) -> dict[str, Any] | None:
    """PreToolUse:Read — inject a prior-work timeline scoped to this file.

    Path-scoped recall via the existing ``store.sql_lookup("tags", "file:<rel>")``
    (exact single-tag membership — no schema change) PLUS a semantic fallback
    via ``query_memory`` for cards that mention the path in their fact. Gated
    on file size (``_FILE_READ_MIN_BYTES``) so tiny files skip the round-trip.
    """
    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path") or tool_input.get("path") or ""
    if not file_path:
        return None
    cwd = payload.get("cwd") or os.getcwd()

    # Gate: skip tiny / missing files (mem0's gate, same rationale).
    try:
        if not os.path.isfile(file_path) or os.path.getsize(file_path) < _FILE_READ_MIN_BYTES:
            return None
    except OSError:
        return None

    rel = _relative_path(file_path, cwd)
    server = _server(db_path)
    try:
        # 1. Exact tag lookup — cards explicitly tagged with this file's
        #    provenance (prior captures from bash_output / session summary).
        tagged = server.store.sql_lookup("tags", f"file:{rel}")
        # 2. Semantic fallback — cards whose fact mentions the path.
        sem = server.query_memory(rel, token_budget=150)
        semantic_hits = sem.get("hits", []) if isinstance(sem, dict) else []
    finally:
        _close_quietly(server)

    timeline = _format_prior_work(tagged, semantic_hits, rel)
    if not timeline:
        return None
    return _context("PreToolUse", timeline)


def _format_prior_work(tagged_cards, semantic_hits, rel: str) -> str:
    """Build the prior-work timeline injected on file read.

    Tagged (exact) hits are the high-confidence "we've worked on this exact
    file" line; semantic hits are a lower-confidence supplement. De-duped by
    card id so a card present in both isn't double-listed.
    """
    seen: set[str] = set()
    lines: list[str] = []
    for card in tagged_cards:
        if card.id in seen:
            continue
        seen.add(card.id)
        lines.append(f"• {card.fact}")
    for h in semantic_hits:
        hid = h.get("id")
        if not hid or hid in seen:
            continue
        seen.add(hid)
        lines.append(f"• {h.get('fact', '')}")
    if not lines:
        return ""
    header = f"izero: prior work on {rel}"
    return header + "\n" + "\n".join(lines[:8])


def _on_bash_output(payload: dict[str, Any], db_path: str) -> dict[str, Any] | None:
    """PostToolUse:Bash — scan for tracebacks, gated auto-capture (Phase 2).

    mem0 only injects a "check memory" rubric; our differentiator is that we
    ALSO capture the error as a typed ``anti_pattern`` card so the cognitive
    store learns from failures (and decay/consolidation manages the volume).
    Gated: skip git noise; dedup via ``store.touch`` after ``_BASH_ERROR_CAP_LIMIT``
    occurrences so a crashloop can't spam the store.
    """
    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command") or ""
    tool_response = payload.get("tool_response") or ""
    if isinstance(tool_response, dict):
        # Claude Code may pass the response as {stdout, stderr, ...}.
        tool_response = tool_response.get("stdout", "") + "\n" + tool_response.get("stderr", "")
    if not isinstance(tool_response, str) or len(tool_response) < 50:
        return None
    if _GIT_NOISE_RE.search(command):
        return None

    error_line = _extract_error_line(tool_response)
    if not error_line:
        return None

    cwd = payload.get("cwd") or os.getcwd()
    sig = _error_signature(error_line, command)
    tag_key = f"error:{sig}"
    files = _extract_trace_files(tool_response)
    rel_files = [_relative_path(f, cwd) for f in files]

    server = _server(db_path)
    try:
        existing = server.store.sql_lookup("tags", tag_key)
        if existing:
            # A card with this error signature already exists → NEVER duplicate it.
            # Touch the first match (boost vitality so the recurring error surfaces
            # higher in recall), regardless of how many prior captures exist. Only a
            # genuinely new signature (existing == []) writes a new card. This is the
            # dedup guarantee: a crashloop can't spam the store with near-identical
            # anti_pattern cards — at most one card per signature.
            server.store.touch(existing[0].id)
            return _context("PostToolUse",
                            f"izero: prior occurrence of this error boosted "
                            f"(seen {len(existing)}×). "
                            f"/izero:recall \"{error_line[:60]}\" for prior fixes.")
        # Auto-capture the error as an anti_pattern card with provenance tags.
        body = f"Error in `{command[:80]}`: {error_line}"
        if rel_files:
            body += f". Files: {', '.join(rel_files)}"
        tags = ["type:anti_pattern", tag_key, "source:bash_output"]
        for rf in rel_files:
            tags.append(f"file:{rf}")
        branch = _git_branch(cwd)
        if branch:
            tags.append(f"branch:{branch}")
        server.add_memory(body, tags=tags)
    finally:
        _close_quietly(server)

    return _context("PostToolUse",
                    f"izero: captured error to memory. "
                    f"/izero:recall \"{error_line[:60]}\" for prior fixes.")


def _on_user_prompt(payload: dict[str, Any], db_path: str) -> dict[str, Any] | None:
    """UserPromptSubmit — inject top relevant memories for the prompt (Phase 2)."""
    prompt = (payload.get("prompt") or payload.get("user_prompt") or "").strip()
    if len(prompt) < 20:  # skip acknowledgements / one-word replies
        return None
    server = _server(db_path)
    try:
        result = server.query_memory(prompt, token_budget=200)
    finally:
        _close_quietly(server)
    hits = result.get("hits", []) if isinstance(result, dict) else []
    if not hits:
        return None
    lines = [f"izero: {len(hits)} relevant memories"]
    lines += ["• " + h.get("fact", "") for h in hits[:5]]
    return _context("UserPromptSubmit", "\n".join(lines))


def _block_memory_write(payload: dict[str, Any], db_path: str) -> dict[str, Any] | None:
    """PreToolUse:Write|Edit|MultiEdit — guard the store file (Phase 2).

    Blocks direct writes to the store file (and its SQLite sidecars) so an
    agent can't corrupt it; writes must go through the ``add_memory`` MCP tool.
    Pure permission gate — no store call. (Also kept in bash for a fast path,
    but the Python handler is the testable, spec-faithful source of truth.)

    Protected paths, derived from ``db_path`` (which ``handle_hook`` resolves
    from ``ISOTOPE_ZERO_DB`` or ``_default_db_path()``) — NOT hardcoded to
    ``~/.isotope_zero/`` alone, so a user who relocates the store gets the new
    path guarded too:

    - the store DB file itself (``db_path``), and its SQLite write-ahead-log /
      shared-memory / rollback sidecars (``<stem>-wal``/``-shm``/``-journal``
      and the ``.wal``/``.shm`` variants) — anything sqlite3 would open as part
      of the live store — matched **exactly** so a legitimate sibling file
      next to a relocated store stays writable;
    - the conventional ``~/.isotope_zero/`` directory root, **prefix-matched**,
      so an agent can't sidestep a relocated store by writing into the default
      path, and any future sidecar/aux file under the conventional dir is
      covered too.

    We do NOT prefix-match the DB's whole parent directory — that would wrongly
    block legitimate sibling files the user keeps next to a relocated store.
    """
    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path") or tool_input.get("path") or ""
    if not file_path:
        return _allow("PreToolUse")
    # Canonicalize to the same form _protected_store_paths stores: normcase
    # lowercases the drive + normalizes separators on Windows (no-op on POSIX)
    # so a ~/.isotope_zero root that expanduser returns with forward slashes
    # still prefix-matches a target that abspath normalized to backslashes.
    abs_target = os.path.normcase(os.path.abspath(os.path.expanduser(file_path)))
    files, dir_roots = _protected_store_paths(db_path)
    # Exact-match the store file + sidecars (a relocated store's sibling
    # files must remain writable); prefix-match the directory roots.
    if abs_target in files or any(
        abs_target == root or abs_target.startswith(root + os.sep)
        for root in dir_roots
    ):
        return _deny(
            "PreToolUse",
            f"Direct writes to {file_path} are blocked — this is the izero "
            f"memory store. Use the `add_memory` MCP tool instead.",
        )
    return _allow("PreToolUse")


def _protected_store_paths(db_path: str) -> tuple[set[str], set[str]]:
    """``(protected_files, protected_dir_roots)`` for the write-guard.

    - ``protected_files``: the store file + its SQLite sidecar variants
      (``-wal``/``-shm``/``-journal``, ``.wal``/``.shm``/``.journal``) — matched
      exactly so siblings of a relocated store stay writable.
    - ``protected_dir_roots``: the conventional ``~/.isotope_zero/`` dir, plus
      the DB's parent dir **when that parent looks store-dedicated** (its
      basename contains ``isotope_zero`` or ``.izero``) — prefix-matched so the
      whole dedicated dir is guarded, while an incidental scratch dir (e.g. a
      unit-test ``tmp_path``) where the DB happens to live is NOT.

    Returns ``({}, {})`` for a ``:memory:`` store (nothing on disk to protect).
    """
    if not db_path or db_path == ":memory:":
        return set(), set()
    files: set[str] = set()
    # Canonical form: normcase (drive lowercase + separator normalization on
    # Windows, no-op on POSIX) + normpath so every file/root matches the
    # normcase'd abs_target _block_memory_write compares against. On Windows,
    # os.path.expanduser("~/.isotope_zero") keeps the forward slash from the
    # literal while abspath normalizes the target to backslashes — without
    # normalizing here, the prefix match would fail and the write-guard would
    # silently allow a write into the store dir (regression guard:
    # test_blocks_write_into_isotope_zero_store_dir on win).
    store_file = os.path.normcase(os.path.abspath(os.path.expanduser(db_path)))
    files.add(store_file)
    stem, _ = os.path.splitext(store_file)
    # SQLite sidecars: the modern hyphenated names + the legacy dotted ones.
    for suffix in ("-wal", "-shm", "-journal", ".wal", ".shm", ".journal"):
        files.add(stem + suffix)
    dir_roots: set[str] = set()
    # The conventional store dir root — guards the default location even when
    # db_path points elsewhere (so an agent can't sidestep a relocated store by
    # writing into the default ~/.isotope_zero/).
    dir_roots.add(
        os.path.normcase(os.path.normpath(os.path.expanduser("~/.isotope_zero")))
    )
    parent = os.path.dirname(store_file)
    parent_base = os.path.basename(parent).lower()
    if "isotope_zero" in parent_base or ".izero" in parent_base:
        dir_roots.add(parent)
    return files, dir_roots


def _on_session_start(payload: dict[str, Any], db_path: str) -> dict[str, Any] | None:
    """SessionStart — inject a one-line store summary (Phase 2)."""
    server = _server(db_path)
    try:
        metrics = server.get_metrics()
    finally:
        _close_quietly(server)
    count = metrics.get("card_count", 0)
    if count == 0:
        return None  # empty store → no context (the agent has nothing to recall)
    size = metrics.get("db_size_human", "?")
    real = "onnx" if metrics.get("embedding_is_real") else "local"
    branch = _git_branch(payload.get("cwd") or os.getcwd())
    hint = f" (branch {branch})" if branch else ""
    return _context(
        "SessionStart",
        f"izero: {count} memories · {size} · {real} embeddings{hint}. "
        f"Use /izero:recall to ground answers in prior work.",
    )


def _on_stop(payload: dict[str, Any], db_path: str) -> dict[str, Any] | None:
    """Stop — local-heuristic session summary capture (Phase 3).

    No LLM: extract decision/preference signals from the session transcript
    via :mod:`fact_extractor` and persist them as cards. Falls back to a
    no-card summary when the transcript is too thin for heuristics.
    """
    summary = _summarize_session_locally(payload, db_path, tag_prefix="session")
    if not summary:
        return None
    return _context("Stop", f"izero: captured session summary to memory.")


def _on_pre_compact(payload: dict[str, Any], db_path: str) -> dict[str, Any] | None:
    """PreCompact — preserve pre-compaction context before it's lost (Phase 3).

    Same local-heuristic path as Stop, tagged ``pre-compaction`` so the
    preserved context is distinguishable from a normal session end.
    """
    summary = _summarize_session_locally(payload, db_path, tag_prefix="pre-compaction")
    if not summary:
        return None
    return _context("PreCompact", "izero: preserved pre-compaction context to memory.")


# --------------------------------------------------------------------------- #
# Session summary — local heuristic, no LLM (Stop + PreCompact share this)
# --------------------------------------------------------------------------- #
def _summarize_session_locally(
    payload: dict[str, Any], db_path: str, *, tag_prefix: str
) -> str | None:
    """Extract a session summary from the transcript via local heuristics.

    Uses :mod:`isotope_zero.extraction.fact_extractor`'s heuristic classifier
    (ADD/UPDATE/DELETE, no LLM) + ``compress_to_card`` to pull decision /
    preference / anti-pattern signals from the transcript text. Tags the
    resulting card ``session:<iso>`` / ``pre-compaction`` + any derivable
    ``file:``/``branch:`` provenance.

    If the transcript is absent or too thin, returns None (no card) — the
    time-window fallback (cards touched in [session_start, now]) is a future
    enhancement gated on the Stop/PreCompact payload spec.
    """
    transcript = _extract_transcript_text(payload)
    if not transcript or len(transcript) < 100:
        return None

    cwd = payload.get("cwd") or os.getcwd()
    branch = _git_branch(cwd)
    # Heuristic: pull user-message + assistant-decision lines that look like
    # durable knowledge (not every transcript line — only signal-bearing ones).
    signals = _extract_decision_signals(transcript)
    if not signals:
        return None

    from ..extraction.fact_extractor import compress_to_card

    server = _server(db_path)
    try:
        for signal in signals[:5]:  # bound the write volume per session
            card = compress_to_card(signal)
            card.tags = list(card.tags)
            card.tags.append(tag_prefix)
            card.tags.append(f"source:{tag_prefix}")
            if branch:
                card.tags.append(f"branch:{branch}")
            emb = server.embedder.embed_text(card.fact)
            card.embedding = emb
            server.store.add(card)
    finally:
        _close_quietly(server)
    return tag_prefix


def _extract_transcript_text(payload: dict[str, Any]) -> str:
    """Pull raw text from the session transcript (best-effort).

    Claude Code passes ``transcript_path`` (a JSONL file) on Stop/PreCompact.
    We read it and concatenate user/assistant message text. If the payload
    already carries the transcript inline, use that. Never raises.
    """
    # Inline transcript (some editor variants pass it directly).
    inline = payload.get("transcript") or payload.get("session")
    if isinstance(inline, str) and inline.strip():
        return inline
    path = payload.get("transcript_path")
    if not path or not isinstance(path, str) or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _extract_decision_signals(transcript: str) -> list[str]:
    """Local-heuristic: pull durable-knowledge lines from transcript text.

    Looks for decision / preference / anti-pattern signal phrases the same way
    :mod:`fact_extractor`'s triage does, but over the whole transcript,
    returning the matching lines as candidate "facts" to store. No LLM.
    """
    signals: list[str] = []
    decision_re = re.compile(
        r"\b(decided|decision|always use|never use|prefer|preference|"
        r"doesn't work|don't try|rule of thumb|convention is|standard is|"
        r"agreed|conclusion)\b",
        re.IGNORECASE,
    )
    for line in transcript.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 15 or len(stripped) > 300:
            continue
        if decision_re.search(stripped):
            # Avoid re-capturing the same line.
            if stripped not in signals:
                signals.append(stripped)
    return signals


# --------------------------------------------------------------------------- #
# Dispatch table + lifecycle
# --------------------------------------------------------------------------- #
_HANDLERS: dict[str, Callable[[dict[str, Any], str], dict[str, Any] | None]] = {
    "session_start": _on_session_start,
    "user_prompt": _on_user_prompt,
    "file_read": _on_file_read,
    "block_memory_write": _block_memory_write,
    "bash_output": _on_bash_output,
    "stop": _on_stop,
    "pre_compact": _on_pre_compact,
}


def _close_quietly(server: Any) -> None:
    """Best-effort close of the hook's short-lived server's store connection.

    The MCP server object holds a ``MemoryStore`` with an open SQLite handle;
    hooks are one-shot processes, so we close it on the way out to flush WAL.
    Never raises — a close failure must not break the hook's emitted JSON.
    """
    try:
        store = getattr(server, "store", None)
        if store is not None:
            close = getattr(store, "close", None)
            if callable(close):
                close()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Offline-guarantee self-check (import-time, no network)
# --------------------------------------------------------------------------- #
# This module must NEVER import a network client. The verification step greps
# the integrations tree for http/requests/urllib/httpx/api_key/openai and
# expects zero hits. Keep it that way: the local-first guarantee is the
# headline differentiator from mem0.
