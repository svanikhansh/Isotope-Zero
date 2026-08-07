"""Tests for the interactive ``izero`` menu (bare ``izero`` with no subcommand).

These exercise the public surface in ``isotope_zero.cli.menu`` and the no-command
guard in ``isotope_zero.cli.debug.main``:

- bare ``main([])`` launches the menu (banner + command list), exit 0;
- the static ``once=True`` path is the deterministic snapshot the assertions use;
- direct subcommands still work after dropping ``required=True`` (regression guard);
- an unknown subcommand still errors (the menu triggers only on *no* command);
- prompt-collection runners dispatch to the right ``_cmd_*`` helper with prompted
  args (driven via a fake stdin);
- the stdlib numbered fallback works when ``rich`` is force-absent;
- exit / q / Ctrl-C return 0.

The recall/search runners use the deterministic fallback embedder (``is_real=False``)
so no ONNX runtime is needed — same contract as ``test_cli.py``.
"""
from __future__ import annotations

import builtins

import pytest

from isotope_zero.cli.debug import main
from isotope_zero.cli import menu as menu_mod


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run_main(argv, capsys):
    rc = main(argv)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def fake_stdin(monkeypatch, lines):
    """Make ``sys.stdin.readline()`` yield the given lines, then EOF."""
    it = iter(lines)

    def _readline(*a, **k):
        try:
            return next(it) + "\n"
        except StopIteration:
            return ""  # EOF → _prompt returns ""

    monkeypatch.setattr("sys.stdin.readline", _readline)
    # _confirm checks isatty(); force False so it doesn't try to prompt interactively
    # in a way that bypasses our readline. The menu's _prompt uses readline directly.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)


# --------------------------------------------------------------------------- #
# 1. bare `izero` launches the menu (static path)
# --------------------------------------------------------------------------- #
def test_bare_izero_shows_menu_and_exits_zero(capsys, monkeypatch):
    # Bare main([]) reaches run_menu; force the static once path by monkeypatching
    # run_menu with a fake that calls the real static path (once=True) — so we
    # assert the banner/list text without starting a live loop in CI.
    real_run_menu = menu_mod.run_menu

    def fake_run_menu(db_path, once=False):
        return real_run_menu(db_path, once=True)

    monkeypatch.setattr("isotope_zero.cli.menu.run_menu", fake_run_menu)

    rc, out, err = run_main([], capsys)
    assert rc == 0
    assert "isotope-zero" in out
    # The command list must name the primary actions the user selected.
    for label in (
        "add a memory",
        "recall something",
        "search the store",
        "open the live dashboard",
        "exit",
    ):
        assert label in out, f"menu missing entry {label!r}\nout={out!r}"


def test_first_run_banner_shows_quickstart(capsys):
    # :memory: → first-run banner (no on-disk DB).
    rc = menu_mod.run_menu(":memory:", once=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "first run" in out
    assert "Welcome to isotope-zero" in out


def test_welcome_back_banner_shows_count(tmp_path, capsys):
    db = str(tmp_path / "mem.db")
    # Seed one card via the real CLI so the DB exists + has count=1.
    rc = main(["add", "a fact to remember", "--db", db])
    assert rc == 0
    capsys.readouterr()  # drain

    rc = menu_mod.run_menu(db, once=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "welcome back" in out
    assert "1 card" in out  # singular form


# --------------------------------------------------------------------------- #
# 2. direct subcommands still work (regression guard for dropping required=True)
# --------------------------------------------------------------------------- #
def test_direct_add_still_works(tmp_path, capsys):
    db = str(tmp_path / "mem.db")
    # add takes a positional `fact` first, then `--db PATH` (see test_cli.db_arg).
    rc, out, err = run_main(["add", "direct add still works", "--db", db], capsys)
    assert rc == 0, f"direct add failed: rc={rc} err={err!r}"
    assert "remembered" in out


def test_unknown_subcommand_still_errors(capsys):
    # The menu must trigger ONLY on no subcommand — a bad command still errors.
    with pytest.raises(SystemExit) as exc:
        main(["this-is-not-a-command"])
    # argparse exits 2 on an unknown subcommand.
    assert exc.value.code == 2


# --------------------------------------------------------------------------- #
# 3. prompt-collection runners dispatch correctly (fake stdin)
# --------------------------------------------------------------------------- #
def test_add_runner_prompts_and_calls_cmd_add(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "mem.db")
    client = menu_mod._open_client(db, create=True)
    try:
        runner = menu_mod._runner_add()
        fake_stdin(monkeypatch, [
            "a fact from the menu",        # fact
            "smallest quote",              # evidence
            "ui, preference",              # tags
            "",                            # scope (empty → None path)
        ])
        rc = runner(client)
    finally:
        client.close()
    out = capsys.readouterr().out
    assert rc == 0
    assert "remembered" in out
    # Confirm the tags were parsed + persisted.
    cid = out.strip().split()[-1]
    client2 = menu_mod._open_client(db, create=False)
    try:
        card = client2.store.get(cid)
        assert card is not None
        assert {"ui", "preference"} <= set(card.tags)
    finally:
        client2.close()


def test_add_runner_eof_cancels_not_loops(monkeypatch, capsys):
    # Regression: the fact prompt is required, but EOF must CANCEL — not spin
    # forever re-prompting. _prompt returns None on EOF; _runner_add treats it
    # as cancel (exit 1), matching the recall/search/get runners.
    client = menu_mod._open_client(":memory:", create=True)
    try:
        runner = menu_mod._runner_add()
        fake_stdin(monkeypatch, [])  # immediate EOF at the fact prompt
        rc = runner(client)
    finally:
        client.close()
    out = capsys.readouterr().out
    assert rc == 1
    assert "(cancelled)" in out


def test_add_runner_ctrl_c_cancels(monkeypatch, capsys):
    # Ctrl-C at the fact prompt (KeyboardInterrupt from readline) → cancel too.
    client = menu_mod._open_client(":memory:", create=True)
    try:
        runner = menu_mod._runner_add()

        def _readline(*a, **k):
            raise KeyboardInterrupt

        monkeypatch.setattr("sys.stdin.readline", _readline)
        rc = runner(client)
    finally:
        client.close()
    out = capsys.readouterr().out
    assert rc == 1
    assert "(cancelled)" in out


def test_recall_runner_empty_query_cancels(monkeypatch):
    client = menu_mod._open_client(":memory:", create=True)
    try:
        runner = menu_mod._runner_recall()
        fake_stdin(monkeypatch, [""])  # empty query → cancel
        rc = runner(client)
    finally:
        client.close()
    assert rc == 1  # cancelled


# --------------------------------------------------------------------------- #
# 4. stdlib numbered fallback works when rich is absent
# --------------------------------------------------------------------------- #
def test_numbered_fallback_dispatches(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "mem.db")
    # Force the rich import inside _run_rich to fail → it degrades to _run_plain.
    real_import = builtins.__import__

    def blocking_import(name, *a, **k):
        if name == "rich.console" or name.startswith("rich."):
            raise ImportError(f"blocked {name}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    # Seed a card so 'list' has something to show; then drive the numbered menu:
    # pick "4) list memories" (index 4 in _entries ordering) then "0) exit".
    rc_seed = main(["add", "listed via menu", "--db", db])
    assert rc_seed == 0
    capsys.readouterr()

    # _entries order: 1 add,2 recall,3 search,4 list,5 get,6 forget,7 touch,
    # 8 stats,9 tags,10 inspect,11 dry-run,12 dashboard,13 exit. "list" = 4.
    # Stdin sequence: "4" (menu→list) → "" (tags filter) → "" (limit) → "0" (exit).
    fake_stdin(monkeypatch, ["4", "", "", "0"])
    rc = menu_mod.run_menu(db)
    out = capsys.readouterr().out
    assert rc == 0
    assert "listed via menu" in out  # the list command printed our seeded card


# --------------------------------------------------------------------------- #
# 5. exit / q / Ctrl-C return 0 (numbered fallback)
# --------------------------------------------------------------------------- #
def test_numbered_fallback_q_exits_zero(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "mem.db")
    real_import = builtins.__import__

    def blocking_import(name, *a, **k):
        if name.startswith("rich."):
            raise ImportError("blocked")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    fake_stdin(monkeypatch, ["q"])
    rc = menu_mod.run_menu(db)
    assert rc == 0
    capsys.readouterr()


def test_numbered_fallback_out_of_range_is_handled(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "mem.db")
    real_import = builtins.__import__

    def blocking_import(name, *a, **k):
        if name.startswith("rich."):
            raise ImportError("blocked")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    # "99" is out of range → "(no command 99)" → then "0" exits.
    fake_stdin(monkeypatch, ["99", "0"])
    rc = menu_mod.run_menu(db)
    out = capsys.readouterr().out
    assert rc == 0
    assert "no command 99" in out


# --------------------------------------------------------------------------- #
# 6. _banner / _numbered_menu are pure
# --------------------------------------------------------------------------- #
def test_numbered_menu_lists_all_entries_with_exit_zero():
    entries = [("a", None), ("b", None), ("c", None)]
    text = menu_mod._numbered_menu(entries)
    assert "1) a" in text
    assert "2) b" in text
    assert "3) c" in text
    assert "0) exit" in text


def test_banner_first_run_vs_welcome_back():
    first = menu_mod._banner(":memory:", count=None)
    assert "first run" in first
    back = menu_mod._banner("/x.db", count=7)
    assert "welcome back" in back
    assert "7 cards" in back
