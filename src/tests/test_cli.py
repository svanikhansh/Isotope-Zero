"""End-to-end tests for the isotope_zero debug CLI.

These tests drive the public ``main(argv=[...])`` entry point exported from
``isotope_zero.cli.debug`` and assert against the CLI SPEC — they do NOT poke
at ``debug.py`` internals. Each subcommand is exercised through its argv
surface (``--db``, ``--json``, ``--yes`` ...) against a per-test temp-file DB
(``tmp_path``) or an empty ``:memory:`` store.

CRITICAL: ``recall``/``search`` MUST work with no ONNX runtime installed. The
``HybridEmbeddingEngine`` falls back to deterministic feature-hash
pseudo-embeddings (``is_real=False``), so recall on a store that contains an
added fact returns at least one ranked hit. These tests therefore never
import ``onnxruntime`` and never mark themselves ``onnx``.
"""
from __future__ import annotations

import json
import sys
import time

import pytest

from isotope_zero.cli.debug import main

# These tests intentionally avoid importing onnxruntime or any embedding
# backend that requires it — the fallback embedder path is the whole point of
# the recall/search coverage below.


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run(main_fn, argv, capsys):
    """Invoke ``main`` once, then drain capsys exactly once.

    ``capsys.readouterr()`` drains the captured streams, so we call it a
    single time after ``main`` returns to grab both stdout and stderr.
    Returns ``(exit_code, stdout, stderr)``.
    """
    rc = main_fn(argv)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def db_arg(db_path: str) -> list[str]:
    """Build the ``--db`` argument list; elides it for ``:memory:`` default.

    The SPEC's default DB path is ``:memory:`` when no override is given, so
    callers that want an in-memory store can simply omit ``--db``. We keep the
    helper explicit so each test documents which store it targets.
    """
    return ["--db", str(db_path)]


def _parse_json_lines_or_object(out: str):
    """Parse JSON output that may be a single object/array or a JSON stream.

    The CLI's ``--json`` flag emits one JSON document per invocation. Some
    commands may additionally print a trailing newline or surrounding
    whitespace; ``json.loads`` tolerates that. If the output is a sequence of
    independent JSON values (one per line), we parse the first non-blank line.
    """
    stripped = out.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Fall back to the first non-blank line for stream-style output.
        for line in out.splitlines():
            line = line.strip()
            if line:
                return json.loads(line)
        raise


def _add_fact(
    main_fn, capsys, db_path: str, fact: str, *extra: str
) -> tuple[int, str, str]:
    """Run ``add <fact>`` against ``db_path`` with arbitrary extra args."""
    argv = ["add", fact, *db_arg(db_path), *extra]
    return run(main_fn, argv, capsys)


def _add_json(main_fn, capsys, db_path: str, fact: str, *extra: str) -> dict:
    """``add --json`` and return the parsed JSON object (must carry ``id``)."""
    rc, out, err = _add_fact(main_fn, capsys, db_path, fact, "--json", *extra)
    assert rc == 0, f"add --json failed rc={rc} err={err!r} out={out!r}"
    obj = _parse_json_lines_or_object(out)
    assert isinstance(obj, dict), f"add --json must emit an object, got {out!r}"
    assert "id" in obj, f"add --json object must contain 'id', got {obj!r}"
    return obj


def _id_from_plain(out: str) -> str:
    """Extract the id printed by ``add`` WITHOUT ``--json`` (just the id)."""
    candidate = out.strip()
    # The plain path prints just the id on stdout. Be tolerant of a trailing
    # newline or a single leading label line, but prefer the last non-blank
    # line so a future "id: <id>" label still parses.
    if not candidate:
        raise AssertionError(f"add without --json printed nothing on stdout")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    # If the last line looks like "<label>: <id>" or "<label> <id>", take the
    # final whitespace-separated token; otherwise the whole last line is the id.
    last = lines[-1]
    if " " in last:
        return last.split()[-1]
    return last


# --------------------------------------------------------------------------- #
# 1. add / get round-trip
# --------------------------------------------------------------------------- #
def test_add_and_get(tmp_path, capsys):
    """``add --json`` emits an object carrying the new id+fact; ``get --json``
    returns the same fact for that id."""
    db = tmp_path / "cli_add_get.db"

    added = _add_json(main, capsys, str(db), "The user's name is Alice.")
    new_id = added["id"]
    # The fact should round-trip verbatim through the JSON object.
    assert added.get("fact") == "The user's name is Alice.", added

    rc, out, err = run(main, ["get", new_id, *db_arg(str(db)), "--json"], capsys)
    assert rc == 0, f"get --json failed rc={rc} err={err!r} out={out!r}"
    got = _parse_json_lines_or_object(out)
    assert isinstance(got, dict), f"get --json must emit an object, got {out!r}"
    assert got.get("id") == new_id, got
    assert got.get("fact") == "The user's name is Alice.", got


# --------------------------------------------------------------------------- #
# 2. add prints id without --json
# --------------------------------------------------------------------------- #
def test_add_prints_id_without_json(tmp_path, capsys):
    """Without ``--json``, ``add`` prints just the id (the return of
    ``remember``) on stdout — no JSON, no decoration."""
    db = tmp_path / "cli_add_plain.db"

    rc, out, err = _add_fact(main, capsys, str(db), "A plain fact with no JSON.")
    assert rc == 0, f"add failed rc={rc} err={err!r} out={out!r}"
    # stdout must be a non-empty id and NOT valid JSON (a bare id string is not
    # a JSON object/array — json.loads on a bare word raises).
    plain = out.strip()
    assert plain, "add without --json must print the id on stdout"
    with pytest.raises(json.JSONDecodeError):
        json.loads(plain)

    # The printed id must resolve to a real card via get --json.
    new_id = _id_from_plain(out)
    rc, out2, err2 = run(main, ["get", new_id, *db_arg(str(db)), "--json"], capsys)
    assert rc == 0, f"get for plain-add id failed rc={rc} err={err2!r}"
    got = _parse_json_lines_or_object(out2)
    assert got.get("fact") == "A plain fact with no JSON.", got


# --------------------------------------------------------------------------- #
# 3. recall returns ranked hits (fallback embedder, no ONNX)
# --------------------------------------------------------------------------- #
def test_recall_returns_ranked_hits(tmp_path, capsys):
    """With the deterministic fallback embedder (``is_real=False``), recall on
    a store holding similar facts returns at least one hit, and ``--json``
    emits a list sorted by score descending."""
    db = tmp_path / "cli_recall.db"

    _add_json(main, capsys, str(db), "The user prefers dark mode for UI.")
    _add_json(main, capsys, str(db), "Python is the user's favorite language.")

    # Plain path: exit 0 and at least one hit (stdout must be non-empty).
    rc, out, err = run(
        main, ["recall", "dark mode preference", *db_arg(str(db))], capsys
    )
    assert rc == 0, f"recall failed rc={rc} err={err!r} out={out!r}"
    assert out.strip(), "recall must print at least one hit on stdout"

    # JSON path: a list sorted by score desc, with at least one entry.
    rc, out, err = run(
        main,
        ["recall", "dark mode preference", *db_arg(str(db)), "--json"],
        capsys,
    )
    assert rc == 0, f"recall --json failed rc={rc} err={err!r} out={out!r}"
    hits = _parse_json_lines_or_object(out)
    assert isinstance(hits, list), f"recall --json must emit a list, got {out!r}"
    assert len(hits) >= 1, f"recall --json must return >=1 hit, got {hits!r}"

    # Every hit carries a numeric score; the list is sorted desc by score.
    scores = [h.get("score") for h in hits]
    assert all(isinstance(s, (int, float)) for s in scores), hits
    assert scores == sorted(scores, reverse=True), f"scores not desc: {scores}"


# --------------------------------------------------------------------------- #
# 4. search (hybrid) returns hits
# --------------------------------------------------------------------------- #
def test_search_returns_hits(tmp_path, capsys):
    """``search`` (hybrid) mirrors ``recall``: ranked hits, JSON list sorted by
    score desc, at least one hit for a similar query."""
    db = tmp_path / "cli_search.db"

    _add_json(main, capsys, str(db), "The user prefers dark mode for UI.")
    _add_json(main, capsys, str(db), "Python is the user's favorite language.")

    rc, out, err = run(
        main, ["search", "dark mode preference", *db_arg(str(db))], capsys
    )
    assert rc == 0, f"search failed rc={rc} err={err!r} out={out!r}"
    assert out.strip(), "search must print at least one hit on stdout"

    rc, out, err = run(
        main,
        ["search", "dark mode preference", *db_arg(str(db)), "--json"],
        capsys,
    )
    assert rc == 0, f"search --json failed rc={rc} err={err!r} out={out!r}"
    hits = _parse_json_lines_or_object(out)
    assert isinstance(hits, list), f"search --json must emit a list, got {out!r}"
    assert len(hits) >= 1, f"search --json must return >=1 hit, got {hits!r}"

    scores = [h.get("score") for h in hits]
    assert all(isinstance(s, (int, float)) for s in scores), hits
    assert scores == sorted(scores, reverse=True), f"scores not desc: {scores}"


# --------------------------------------------------------------------------- #
# 5. active orders by vitality desc (touch boosts a card)
# --------------------------------------------------------------------------- #
def test_active_orders_by_vitality_desc(tmp_path, capsys):
    """``active --json`` returns cards ranked by vitality DESC. Touching one
    card (incrementing its access count / last_access) must lift it, so the
    result list is vitality-sorted and the touched card ranks highest."""
    db = tmp_path / "cli_active.db"

    a = _add_json(main, capsys, str(db), "Fact alpha about routing.")
    b = _add_json(main, capsys, str(db), "Fact bravo about storage.")
    c = _add_json(main, capsys, str(db), "Fact charlie about retrieval.")
    touched_id = b["id"]

    # Touch card b to boost its vitality (access_count++ / last_access=now).
    rc, out, err = run(main, ["touch", touched_id, *db_arg(str(db))], capsys)
    assert rc == 0, f"touch failed rc={rc} err={err!r} out={out!r}"

    rc, out, err = run(main, ["active", *db_arg(str(db)), "--json"], capsys)
    assert rc == 0, f"active --json failed rc={rc} err={err!r} out={out!r}"
    rows = _parse_json_lines_or_object(out)
    assert isinstance(rows, list), f"active --json must emit a list, got {out!r}"
    assert len(rows) == 3, f"active should list all 3 cards, got {rows!r}"

    # Each row carries a numeric vitality; the list is sorted desc by vitality.
    vitalities = [r.get("vitality") for r in rows]
    assert all(
        isinstance(v, (int, float)) for v in vitalities
    ), f"vitality must be numeric, got {rows!r}"
    assert vitalities == sorted(
        vitalities, reverse=True
    ), f"active not sorted by vitality desc: {vitalities}"

    # The touched card must rank highest (touch boosts vitality strictly).
    ids = [r.get("id") for r in rows]
    assert ids[0] == touched_id, (
        f"touched card {touched_id!r} should rank first, got order {ids!r}"
    )


# --------------------------------------------------------------------------- #
# 6. list newest-first
# --------------------------------------------------------------------------- #
def test_list_newest_first(tmp_path, capsys):
    """``list --json`` returns all cards sorted by timestamp DESC (newest
    first). We rely on add order + a small time gap to guarantee ordering."""
    db = tmp_path / "cli_list.db"

    first = _add_json(main, capsys, str(db), "First fact written earlier.")
    # Small wall-clock gap so the two timestamps are not identical.
    time.sleep(0.02)
    second = _add_json(main, capsys, str(db), "Second fact written later.")

    rc, out, err = run(main, ["list", *db_arg(str(db)), "--json"], capsys)
    assert rc == 0, f"list --json failed rc={rc} err={err!r} out={out!r}"
    rows = _parse_json_lines_or_object(out)
    assert isinstance(rows, list), f"list --json must emit a list, got {out!r}"
    assert len(rows) == 2, f"list should show both cards, got {rows!r}"

    timestamps = [r.get("timestamp") for r in rows]
    assert all(isinstance(t, (int, float)) for t in timestamps), rows
    # Newest first => timestamps non-increasing.
    assert timestamps == sorted(
        timestamps, reverse=True
    ), f"list not newest-first: {timestamps}"

    # The second-added (later) card must come before the first-added card.
    ids = [r.get("id") for r in rows]
    assert ids == [second["id"], first["id"]], (
        f"expected newest (second) then oldest (first), got {ids!r}"
    )


# --------------------------------------------------------------------------- #
# 7. get not found -> exit 1, stderr mentions the id
# --------------------------------------------------------------------------- #
def test_get_not_found(tmp_path, capsys):
    """``get`` for a missing id exits 1 and prints the id on stderr."""
    db = tmp_path / "cli_get_missing.db"
    missing_id = "nonexistent-card-id"

    rc, out, err = run(main, ["get", missing_id, *db_arg(str(db))], capsys)
    assert rc == 1, f"get missing must exit 1, got rc={rc} out={out!r} err={err!r}"
    assert missing_id in err, (
        f"stderr must mention the missing id {missing_id!r}, got err={err!r}"
    )


# --------------------------------------------------------------------------- #
# 8. forget --yes removes the card
# --------------------------------------------------------------------------- #
def test_forget_yes_removes(tmp_path, capsys):
    """``forget --yes`` deletes the card; a subsequent ``get`` exits 1."""
    db = tmp_path / "cli_forget_yes.db"

    added = _add_json(main, capsys, str(db), "A fact to be forgotten.")
    new_id = added["id"]

    rc, out, err = run(
        main, ["forget", new_id, *db_arg(str(db)), "--yes"], capsys
    )
    assert rc == 0, f"forget --yes failed rc={rc} err={err!r} out={out!r}"

    # The card must now be gone: get exits 1 and stderr mentions the id.
    rc, out, err = run(main, ["get", new_id, *db_arg(str(db))], capsys)
    assert rc == 1, (
        f"get after forget must exit 1, got rc={rc} out={out!r} err={err!r}"
    )
    assert new_id in err, (
        f"get after forget must mention the id on stderr, got err={err!r}"
    )


# --------------------------------------------------------------------------- #
# 9. forget --yes path + forget nonexistent prints not-found
# --------------------------------------------------------------------------- #
def test_forget_nonexistent_reports_not_found(tmp_path, capsys):
    """``forget --yes`` on a missing id does NOT crash and reports not-found
    (exit non-zero, stderr mentions the id). We exercise the ``--yes`` path
    explicitly to avoid any interactive prompt."""
    db = tmp_path / "cli_forget_missing.db"
    missing_id = "ghost-id-not-present"

    rc, out, err = run(
        main, ["forget", missing_id, *db_arg(str(db)), "--yes"], capsys
    )
    # A missing target must surface as a non-zero exit with the id on stderr
    # (mirrors the get-not-found contract).
    assert rc != 0, (
        f"forget missing must exit non-zero, got rc={rc} out={out!r} err={err!r}"
    )
    assert missing_id in err, (
        f"forget missing must mention id on stderr, got err={err!r}"
    )


def test_forget_confirms_without_yes_on_non_tty(tmp_path, capsys, monkeypatch):
    """Without ``--yes`` on a non-tty stdin, ``forget`` auto-confirms (the
    SPEC's non-interactive path) and still removes the card. We force the
    non-interactive path by stubbing ``sys.stdin.isatty`` to False."""
    db = tmp_path / "cli_forget_confirm.db"

    added = _add_json(main, capsys, str(db), "A fact to forget non-interactively.")
    new_id = added["id"]

    # Force the non-tty / non-interactive path: no human can answer a prompt
    # during a test run, so the CLI must auto-confirm when stdin is not a tty.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    rc, out, err = run(main, ["forget", new_id, *db_arg(str(db))], capsys)
    assert rc == 0, (
        f"forget without --yes on non-tty must auto-confirm and succeed, "
        f"got rc={rc} out={out!r} err={err!r}"
    )

    # Confirm removal via get.
    rc, out, err = run(main, ["get", new_id, *db_arg(str(db))], capsys)
    assert rc == 1, (
        f"get after non-tty forget must exit 1, got rc={rc} err={err!r}"
    )


# --------------------------------------------------------------------------- #
# 10. tags aggregation
# --------------------------------------------------------------------------- #
def test_tags(tmp_path, capsys):
    """``tags --json`` returns a tag->count dict. Adding cards tagged
    ``a,b`` and ``b,c`` yields ``{a:1, b:2, c:1}`` (compared as a dict, order
    may vary)."""
    db = tmp_path / "cli_tags.db"

    _add_json(main, capsys, str(db), "Tagged ab fact.", "--tags", "a,b")
    _add_json(main, capsys, str(db), "Tagged bc fact.", "--tags", "b,c")

    rc, out, err = run(main, ["tags", *db_arg(str(db)), "--json"], capsys)
    assert rc == 0, f"tags --json failed rc={rc} err={err!r} out={out!r}"
    tags = _parse_json_lines_or_object(out)
    assert isinstance(tags, dict), f"tags --json must emit an object, got {out!r}"
    assert tags == {"a": 1, "b": 2, "c": 1}, f"tag counts mismatch: {tags!r}"


# --------------------------------------------------------------------------- #
# 11. stats summary
# --------------------------------------------------------------------------- #
def test_stats(tmp_path, capsys):
    """``stats --json`` returns a summary object with the expected key set and
    a ``count`` that matches the number of added cards."""
    db = tmp_path / "cli_stats.db"

    _add_json(main, capsys, str(db), "Stats fact one.", "--tags", "x")
    _add_json(main, capsys, str(db), "Stats fact two.", "--tags", "y")

    rc, out, err = run(main, ["stats", *db_arg(str(db)), "--json"], capsys)
    assert rc == 0, f"stats --json failed rc={rc} err={err!r} out={out!r}"
    stats = _parse_json_lines_or_object(out)
    assert isinstance(stats, dict), f"stats --json must emit an object, got {out!r}"

    # The SPEC names these summary keys (some impls use near-synonyms; we
    # require at least the count + embedding mode + token footprint + tags
    # + a vitality histogram, allowing snake_case variants).
    required = {"count", "embedding_mode", "token_footprint", "tags"}
    missing = required - set(stats.keys())
    assert not missing, (
        f"stats --json missing keys {missing!r}; got keys {sorted(stats)!r}"
    )
    # A vitality histogram is expected (named vitality_histogram or similar).
    has_vitality_hist = any(
        "vitality" in k and ("hist" in k or "bucket" in k or "dist" in k)
        for k in stats
    )
    assert has_vitality_hist, (
        f"stats --json must include a vitality histogram, got keys {sorted(stats)!r}"
    )
    # db_size may be named db_size / db_size_bytes — accept either presence.
    has_db_size = any(k.startswith("db_size") for k in stats)
    assert has_db_size, (
        f"stats --json must include db_size, got keys {sorted(stats)!r}"
    )

    assert stats["count"] == 2, f"stats count must equal added cards: {stats!r}"


# --------------------------------------------------------------------------- #
# 12. nonexistent --db errors (inspect guard)
# --------------------------------------------------------------------------- #
def test_nonexistent_db_errors(capsys):
    """An explicit ``--db`` path that does not exist must exit 1 and mention
    the path on stderr. ``inspect`` is the existing command; this confirms the
    missing-DB guard is unchanged by the new subcommands."""
    bad_path = "/no/such/path/does/not/exist.db"
    rc, out, err = run(main, ["inspect", *db_arg(bad_path)], capsys)
    assert rc == 1, (
        f"inspect on missing --db must exit 1, got rc={rc} out={out!r} err={err!r}"
    )
    assert bad_path in err, (
        f"stderr must mention the missing db path, got err={err!r}"
    )


# --------------------------------------------------------------------------- #
# 13. inspect unchanged (backward compat on empty :memory:)
# --------------------------------------------------------------------------- #
def test_inspect_unchanged_empty_memory(capsys):
    """``inspect --json`` on an empty ``:memory:`` store still works (backward
    compat): exit 0 and valid JSON with the existing payload shape."""
    rc, out, err = run(main, ["inspect", *db_arg(":memory:"), "--json"], capsys)
    assert rc == 0, f"inspect --json :memory: failed rc={rc} err={err!r}"
    payload = _parse_json_lines_or_object(out)
    assert isinstance(payload, dict), f"inspect --json must emit an object, got {out!r}"
    # The existing inspect payload nests a 'store' object with total_cards.
    store = payload.get("store")
    assert isinstance(store, dict), (
        f"inspect --json must nest a 'store' object, got {payload!r}"
    )
    assert store.get("total_cards") == 0, (
        f"empty :memory: store must report 0 cards, got {store!r}"
    )


# --------------------------------------------------------------------------- #
# 14. --json output is valid JSON for every command that supports it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "argv_factory",
    [
        pytest.param(
            lambda db: ["inspect", "--db", db, "--json"],
            id="inspect",
        ),
        pytest.param(
            lambda db: ["stats", "--db", db, "--json"],
            id="stats",
        ),
        pytest.param(
            lambda db: ["tags", "--db", db, "--json"],
            id="tags",
        ),
        pytest.param(
            lambda db: ["list", "--db", db, "--json"],
            id="list",
        ),
        pytest.param(
            lambda db: ["active", "--db", db, "--json"],
            id="active",
        ),
    ],
)
def test_json_valid_for_read_commands(tmp_path, capsys, argv_factory):
    """For every read command that supports ``--json``, the stdout parses as
    JSON. We seed one card so list/active/tags/stats have non-trivial data."""
    db = tmp_path / "cli_json_valid.db"
    db_path = str(db)
    _add_json(main, capsys, db_path, "JSON validity seed fact.", "--tags", "z")

    argv = argv_factory(db_path)
    rc, out, err = run(main, argv, capsys)
    assert rc == 0, f"{' '.join(argv)} failed rc={rc} err={err!r} out={out!r}"
    parsed = _parse_json_lines_or_object(out)
    # Must parse to a JSON object or array (not None / not a bare scalar).
    assert parsed is not None, f"{' '.join(argv)} emitted no JSON: {out!r}"
    assert isinstance(parsed, (dict, list)), (
        f"{' '.join(argv)} must emit JSON object/array, got {out!r}"
    )


def test_json_valid_for_add_get_recall_search(tmp_path, capsys):
    """``add``, ``get``, ``recall``, and ``search`` all emit valid JSON under
    ``--json``. (``forget --yes`` is exercised separately; its ``--json`` path,
    if any, is not required by the SPEC.)"""
    db = tmp_path / "cli_json_io.db"
    db_path = str(db)

    # add --json -> object
    added = _add_json(main, capsys, db_path, "JSON IO seed fact.")
    new_id = added["id"]

    # get --json -> object
    rc, out, err = run(main, ["get", new_id, "--db", db_path, "--json"], capsys)
    assert rc == 0, f"get --json failed rc={rc} err={err!r}"
    got = _parse_json_lines_or_object(out)
    assert isinstance(got, dict), f"get --json must emit an object, got {out!r}"

    # recall --json -> list
    rc, out, err = run(
        main, ["recall", "seed", "--db", db_path, "--json"], capsys
    )
    assert rc == 0, f"recall --json failed rc={rc} err={err!r}"
    rec = _parse_json_lines_or_object(out)
    assert isinstance(rec, list), f"recall --json must emit a list, got {out!r}"

    # search --json -> list
    rc, out, err = run(
        main, ["search", "seed", "--db", db_path, "--json"], capsys
    )
    assert rc == 0, f"search --json failed rc={rc} err={err!r}"
    srch = _parse_json_lines_or_object(out)
    assert isinstance(srch, list), f"search --json must emit a list, got {out!r}"
