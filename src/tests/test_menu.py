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

    def _no_fileno():
        raise OSError(9, "Bad file descriptor")

    monkeypatch.setattr("sys.stdin.readline", _readline)
    # _confirm checks isatty(); force False so it doesn't try to prompt interactively
    # in a way that bypasses our readline. The menu's _prompt uses readline directly.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    # _is_real_tty probes os.isatty(fileno()) to decide whether _run_plain should
    # promote to the raw-mode arrow-key transport. Make fileno() raise so the
    # numbered path is exercised even when pytest runs on a real TTY — otherwise
    # _run_plain_interactive would setraw() the terminal and block on real input
    # that fake_stdin's readline never provides.
    monkeypatch.setattr("sys.stdin.fileno", _no_fileno)


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


def test_dashboard_runner_uses_live_tui(monkeypatch):
    """Menu item 12 delegates to ``run_live`` (the polished multi-panel TUI
    ``izero dashboard`` renders) with the menu client's store — NOT the legacy
    single-panel ``dashboard.run_dashboard``."""
    captured = {}

    def fake_run_live(store, db_path, interval=2.0, once=False):
        captured["store"] = store
        captured["db_path"] = db_path
        captured["interval"] = interval
        captured["once"] = once
        return 7  # any non-trivial rc the dashboard returns on exit

    monkeypatch.setattr(menu_mod, "run_live", fake_run_live)
    client = menu_mod._open_client(":memory:", create=True)
    try:
        runner = menu_mod._runner_dashboard("/some/db/path.db")
        rc = runner(client)
    finally:
        client.close()
    assert rc == 7
    assert captured["store"] is client.store  # reuses the menu's open store
    assert captured["db_path"] == "/some/db/path.db"
    assert captured["interval"] == 2.0
    assert captured.get("once") is False


def test_prompt_oserror_treated_as_eof_not_crash(monkeypatch):
    """Backing out of a full-screen surface can leave stdin read raising
    ``OSError: [Errno 5] Input/output error`` — ``_prompt`` must treat that like
    EOF (return None → caller cancels/quits) instead of dumping a traceback."""
    def _readline(*a, **k):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr("sys.stdin.readline", _readline)
    assert menu_mod._prompt("anything") is None


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


# --------------------------------------------------------------------------- #
# 7. Interactive-list helpers: escape tails, key decode, viewport scroll
# --------------------------------------------------------------------------- #
def _drive_escape_tail(monkeypatch, byte_stream, ready=True):
    """Drive ``_escape_tail`` with a synthetic byte stream. Stubs ``os.read``
    (the raw unbuffered reads the key loop uses — ``_escape_tail`` reads the fd,
    not the buffered ``sys.stdin``) and ``select`` so every probe reports ready
    (or not-ready when ``ready=False``)."""
    it = iter(byte_stream)

    def fake_read(fd, n=1):
        return next(it).encode("latin-1")

    monkeypatch.setattr(menu_mod.os, "read", fake_read)
    if ready:
        monkeypatch.setattr(
            menu_mod.select, "select", lambda *a, **k: ([a[0][0]], [], [])
        )
    else:
        monkeypatch.setattr(
            menu_mod.select, "select", lambda *a, **k: ([], [], [])
        )


def test_escape_tail_reads_csi_arrow(monkeypatch):
    _drive_escape_tail(monkeypatch, ["[", "A"])
    assert menu_mod._escape_tail(0) == "[A"


def test_escape_tail_reads_ss3_arrow(monkeypatch):
    # Application-cursor-mode terminals send ESC O B for Down — the reader must
    # not assume the '[' CSI form (the bug that froze the highlight on iTerm2 /
    # tmux and made the menu appear to "can't scroll up or down").
    _drive_escape_tail(monkeypatch, ["O", "B"])
    assert menu_mod._escape_tail(0) == "OB"


def test_escape_tail_reads_csi_page_down(monkeypatch):
    _drive_escape_tail(monkeypatch, ["[", "6", "~"])
    assert menu_mod._escape_tail(0) == "[6~"


def test_escape_tail_lone_escape_returns_empty(monkeypatch):
    # No tail bytes → select reports not-ready → "" (a lone Esc is ignored).
    _drive_escape_tail(monkeypatch, [], ready=False)
    assert menu_mod._escape_tail(0) == ""


def test_apply_key_maps_csi_and_ss3_arrows():
    n = 5
    assert menu_mod._apply_key("[B", 0, n, 10) == 1    # Down (CSI)
    assert menu_mod._apply_key("OB", 0, n, 10) == 1    # Down (SS3)
    assert menu_mod._apply_key("[A", 0, n, 10) == 4    # Up wraps to last
    assert menu_mod._apply_key("[H", 3, n, 10) == 0    # Home
    assert menu_mod._apply_key("[F", 1, n, 10) == 4    # End
    assert menu_mod._apply_key("[6~", 0, n, 10) == 4   # PgDn
    assert menu_mod._apply_key("[5~", 4, n, 10) == 0   # PgUp
    assert menu_mod._apply_key("junk", 2, n, 10) == 2  # unknown → unchanged


def test_scroll_window_keeps_selection_visible():
    # Short list (fits) → no scrolling.
    assert menu_mod._scroll_window(sel=3, top=0, height=13, n=13) == 0
    # sel inside window → top unchanged.
    assert menu_mod._scroll_window(sel=5, top=4, height=4, n=13) == 4
    # sel above window → scroll up to reveal it.
    assert menu_mod._scroll_window(sel=2, top=4, height=4, n=13) == 2
    # sel below window → scroll down to reveal it.
    assert menu_mod._scroll_window(sel=9, top=4, height=4, n=13) == 6
