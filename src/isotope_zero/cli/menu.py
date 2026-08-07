"""isotope_zero CLI — interactive onboarding + command menu (bare ``izero``).

Running ``izero`` with **no subcommand** opens this menu instead of erroring with
"the following arguments are required: command". It shows a welcome banner (where
your memories live) and a navigable list of commands; pick one → it prompts for that
command's inputs → runs it → returns to the menu. Direct ``izero <cmd> ...`` is
unchanged and remains the path for power users / scripts.

Two navigation transports, chosen at runtime — exactly the graceful-degradation
pattern the dashboard already uses (``cli/dashboard.py``):

    rich  (optional ``[dashboard]`` extra)
        Arrow-key navigation (↑↓ highlight, enter to run, q/Ctrl-C to quit) via
        ``rich.live.Live`` with the alternate screen buffer so the terminal is
        restored on exit. Imported lazily inside ``_run_rich`` so the module
        imports with zero optional deps.

    plain (stdlib only — the fallback when ``rich`` is absent)
        A numbered list (``1) add a memory`` … ``0) exit``); type the number +
        enter to run. Works in every terminal, no dependencies.

The menu **invents no store logic** — every entry calls an existing ``_cmd_*``
helper from ``debug.py`` (or ``run_dashboard``) with prompted arguments matching the
exact shapes ``main()`` already passes. It opens one long-lived ``IsotopeZero``
client across read/write actions (``add`` … ``stats``) and reuses the existing
per-call open/close path for ``inspect`` / ``dry-run-consolidation`` (which take a
raw ``MemoryStore``) and yields to ``run_dashboard`` (which owns its own client).

Calling :func:`run_menu` with ``once=True`` prints the banner + static menu text and
returns 0 — the deterministic path the tests exercise (no live loop, no ``input``).
It is an internal/test-only surface, not a ``--menu-once`` CLI flag.
"""

from __future__ import annotations

import os
import select
import sys
from typing import Any, Callable

from .debug import (
    IsotopeZero,
    _cmd_add,
    _cmd_dry_run,
    _cmd_forget,
    _cmd_get,
    _cmd_inspect,
    _cmd_list,
    _cmd_recall,
    _cmd_search,
    _cmd_stats,
    _cmd_tags,
    _cmd_touch,
    _confirm,
    _open_client,
    _open_store,
    _parse_tags,
    _resolve_db_path,
)
from .dashboard import run_dashboard


def _ensure_parent_dir(db_path: str) -> None:
    """Create the DB's parent directory so the store can be created on first write.

    SQLite does not create missing parent directories, so a first run against a
    brand-new default path (e.g. ``~/.isotope_zero/isotope_zero.db``) would fail
    with ``unable to open database file``. Mirrors ``mcp/server.py``'s guard;
    skips ``:memory:`` (no on-disk file). No-op when the dir already exists.
    """
    if db_path == ":memory:":
        return
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# --------------------------------------------------------------------------- #
# Menu entries — order is the on-screen order. Each runner is (client) -> int
# and may prompt on stdin; it returns the subcommand's exit code. The two
# read-only store commands (inspect / dry-run) take a MemoryStore, not a
# client, so they open their own store per call and close it immediately.
# --------------------------------------------------------------------------- #
def _prompt(label: str, default: str | None = None) -> str | None:
    """Read one line from stdin.

    Returns ``""`` for an empty Enter, the ``default`` (if given) for an empty
    Enter too, and ``None`` for EOF/Ctrl-D (caller treats None as cancel — a
    required prompt must never spin forever when the user can't type).
    """
    suffix = f" [{default}]" if default is not None else ""
    print(f"{label}{suffix}: ", end="", flush=True)
    try:
        line = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        return None
    if line == "":  # EOF
        return None
    line = line.strip()
    if not line and default is not None:
        return default
    return line


def _runner_add() -> Callable[[IsotopeZero], int]:
    def go(client: IsotopeZero) -> int:
        fact = _prompt("fact")
        if fact is None:  # EOF/Ctrl-D → cancel, don't loop forever
            print("(cancelled)")
            return 1
        while not fact:  # empty Enter → re-prompt
            print("(fact cannot be empty — Ctrl-D to cancel)")
            fact = _prompt("fact")
            if fact is None:
                print("(cancelled)")
                return 1
        evidence = _prompt("evidence (optional)") or ""
        tags_csv = _prompt("tags (comma-separated, optional)") or ""
        scope = _prompt("scope (optional)") or None
        return _cmd_add(
            client, fact, evidence, _parse_tags(tags_csv), scope, None, False
        )

    return go


def _runner_recall() -> Callable[[IsotopeZero], int]:
    def go(client: IsotopeZero) -> int:
        query = _prompt("query")
        if not query:
            print("(cancelled)")
            return 1
        k = _prompt("k (top results)", "5") or "5"  # None (EOF) → default too
        try:
            k_int = int(k)
        except ValueError:
            print(f"(using default 5 — {k!r} isn't a number)")
            k_int = 5
        alpha_s = _prompt("alpha (blend, blank=auto)")
        alpha = None if not alpha_s else _parse_float(alpha_s)
        return _cmd_recall(client, query, k_int, alpha, False)

    return go


def _runner_search() -> Callable[[IsotopeZero], int]:
    def go(client: IsotopeZero) -> int:
        query = _prompt("query")
        if not query:
            print("(cancelled)")
            return 1
        k = _prompt("k (top results)", "5") or "5"  # None (EOF) → default too
        try:
            k_int = int(k)
        except ValueError:
            print(f"(using default 5 — {k!r} isn't a number)")
            k_int = 5
        alpha_s = _prompt("alpha (blend, blank=auto)")
        alpha = None if not alpha_s else _parse_float(alpha_s)
        return _cmd_search(client, query, k_int, alpha, False)

    return go


def _runner_list() -> Callable[[IsotopeZero], int]:
    def go(client: IsotopeZero) -> int:
        tags_csv = _prompt("tags filter (comma-separated, optional)") or ""
        limit_s = _prompt("limit (blank=all)", "")
        try:
            limit = int(limit_s) if limit_s else 0
        except ValueError:
            limit = 0
        return _cmd_list(client, _parse_tags(tags_csv), limit, False)

    return go


def _runner_get() -> Callable[[IsotopeZero], int]:
    def go(client: IsotopeZero) -> int:
        card_id = _prompt("card id")
        if not card_id:
            print("(cancelled)")
            return 1
        return _cmd_get(client, card_id, False)

    return go


def _runner_forget() -> Callable[[IsotopeZero], int]:
    def go(client: IsotopeZero) -> int:
        card_id = _prompt("card id")
        if not card_id:
            print("(cancelled)")
            return 1
        # Reuse the shared confirmer; in the menu stdin is a tty so it prompts.
        return _cmd_forget(client, card_id, False)

    return go


def _runner_touch() -> Callable[[IsotopeZero], int]:
    def go(client: IsotopeZero) -> int:
        card_id = _prompt("card id")
        if not card_id:
            print("(cancelled)")
            return 1
        return _cmd_touch(client, card_id)

    return go


def _runner_stats() -> Callable[[IsotopeZero], int]:
    return lambda client: _cmd_stats(client, False)


def _runner_tags() -> Callable[[IsotopeZero], int]:
    return lambda client: _cmd_tags(client, False)


def _runner_inspect(db_path: str) -> Callable[[IsotopeZero], int]:
    # inspect takes a raw MemoryStore; open/close one per call.
    def go(_client: IsotopeZero) -> int:
        store = _open_store(db_path)
        if store is None:
            print(f"DB path does not exist: {db_path}", file=sys.stderr)
            return 1
        try:
            return _cmd_inspect(store, 5, False)
        finally:
            store.close()

    return go


def _runner_dry_run(db_path: str) -> Callable[[IsotopeZero], int]:
    def go(_client: IsotopeZero) -> int:
        store = _open_store(db_path)
        if store is None:
            print(f"DB path does not exist: {db_path}", file=sys.stderr)
            return 1
        try:
            return _cmd_dry_run(store, 0)
        finally:
            store.close()

    return go


def _runner_dashboard(db_path: str) -> Callable[[IsotopeZero], int]:
    # run_dashboard owns its own client + terminal; yield to it.
    return lambda _client: run_dashboard(db_path, 2.0, False)


def _parse_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


# An entry: (label, runner-factory). The factory takes db_path (only needed by
# the store/dashboard runners; the client runners ignore it) and returns the
# (client) -> int closure. Ordering here is the on-screen order.
def _entries(db_path: str) -> list[tuple[str, Callable[[IsotopeZero], int]]]:
    return [
        ("add a memory", _runner_add()),
        ("recall something", _runner_recall()),
        ("search the store", _runner_search()),
        ("list memories", _runner_list()),
        ("get a memory", _runner_get()),
        ("forget a memory", _runner_forget()),
        ("touch (refresh) a memory", _runner_touch()),
        ("see stats", _runner_stats()),
        ("see tags", _runner_tags()),
        ("inspect the store", _runner_inspect(db_path)),
        ("dry-run consolidation", _runner_dry_run(db_path)),
        ("open the live dashboard", _runner_dashboard(db_path)),
        ("exit", None),  # sentinel — runner is None
    ]


# --------------------------------------------------------------------------- #
# Banner — pure, testable (no terminal I/O)
# --------------------------------------------------------------------------- #
def _banner(db_path: str, count: int | None = None) -> str:
    """Welcome panel. ``count`` is the live card count when the DB exists; None
    means first run (no on-disk DB yet). Box-drawing matches the dashboard.
    """
    if count is None:
        head = "isotope-zero · first run"
        line1 = "Welcome to isotope-zero — a local-first cognitive"
        line2 = "memory layer for AI agents."
        line3 = f"Your memories will live at: {db_path}"
        line4 = "Pick an action below to get started."
    else:
        head = "isotope-zero · welcome back"
        line1 = f"Your memories live at: {db_path}"
        line2 = f"{count} card{'s' if count != 1 else ''} remembered so far."
        line3 = "What would you like to do?"
        line4 = ""
    lines = [line1, line2, line3]
    if line4:
        lines.append(line4)
    inner = max(len(head), *(len(l) for l in lines))
    top = "┌ " + head + " " * (inner - len(head)) + " ┐"
    bot = "└" + "─" * (inner + 2) + "┘"
    body = ["│ " + l + " " * (inner - len(l)) + " │" for l in lines]
    return "\n".join([top, *body, bot])


def _numbered_menu(entries: list[tuple[str, Any]]) -> str:
    """The plain-transport list: `1) label` … `0) exit`."""
    lines = []
    for i, (label, _) in enumerate(entries, start=1):
        lines.append(f"  {i}) {label}")
    lines.append("  0) exit")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #
def _run_plain(db_path: str, entries: list[tuple[str, Callable[[IsotopeZero], int]]],
              banner: str) -> int:
    """Stdlib numbered-menu loop. Type a number + enter to run; 0/q to quit."""
    # create=True mirrors _run_client's default: a missing DB is treated as an
    # empty store (so reads say "no cards", and `add` creates the file on first
    # write). NOTE: this also creates an empty DB file the moment the menu opens
    # against a brand-new default path — so a first-run user who only browses and
    # quits leaves an empty isotope_zero.db behind, and the next run shows
    # "welcome back — 0 cards". Consistent with the rest of the CLI (which has
    # the same create-or-empty contract); a lazy create-on-write-only model would
    # be a stricter improvement but is a cross-transport lifecycle change.
    client = _open_client(db_path, create=True)
    try:
        while True:
            print("\n" + banner)
            print(_numbered_menu(entries))
            sel = _prompt("\n❯")
            # None = EOF (exits); a bare Enter is NOT an exit — it falls through
            # to the int() parse and re-prompts, so one accidental Enter can't
            # end the whole session.
            if sel is None or sel in ("q", "Q", "0", "quit", "exit"):
                return 0
            try:
                idx = int(sel)
            except ValueError:
                print("(enter a number)")
                continue
            if idx == 0:
                return 0
            if idx < 1 or idx > len(entries):
                print(f"(no command {idx})")
                continue
            label, runner = entries[idx - 1]
            if runner is None:  # the explicit "exit" entry
                return 0
            try:
                runner(client)
            except KeyboardInterrupt:
                # Ctrl-C mid-action: back to menu, not exit.
                print("\n(interrupted — back to menu)")
            print()  # blank line between the command's output and the next menu
    except KeyboardInterrupt:
        return 0
    finally:
        client.close()


def _escape_tail() -> str:
    """Read the ``[A..D`` tail of an arrow-key escape sequence, non-blocking.

    In raw mode a lone Esc delivers just ``\\x1b``; blocking on ``read(2)`` would
    freeze the menu until the user types more keys. Arrow keys deliver their tail
    as a burst, so a short ``select`` poll distinguishes them and a lone Esc is
    simply ignored.
    """
    r, _, _ = select.select([sys.stdin], [], [], 0.1)
    if not r:
        return ""
    return sys.stdin.read(2)


def _run_rich(db_path: str, entries: list[tuple[str, Callable[[IsotopeZero], int]]],
              banner: str) -> int:
    """Arrow-key navigation via rich.live (lazy import; degrades to _run_plain)."""
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.text import Text
    except ImportError:
        return _run_plain(db_path, entries, banner)

    client = _open_client(db_path, create=True)
    sel = 0
    n = len(entries)

    def _frame() -> Text:
        body = banner + "\n\nWhat would you like to do?\n"
        for i, (label, _) in enumerate(entries):
            marker = "❯" if i == sel else " "
            body += f"\n  {marker} {label}"
        body += "\n\n↑↓ navigate · enter to run · q to quit"
        return Text(body)

    # Only key-reading/transport failures trigger the plain fallback; errors a
    # runner raises must surface (mirroring _run_plain), not be swallowed into
    # a fresh menu loop. `fallback` defers the degradation until the rich client
    # is closed so _run_plain never runs with two clients open.
    running_action = False
    fallback = False
    try:
        console = Console()
        # auto_refresh=False: the menu is static, so rich renders only when we
        # call update(refresh=True) — once per keypress. The default 30fps
        # auto-refresh thread would re-render the frame constantly for no reason
        # and, worse, hold the render lock while the action runs.
        with Live(_frame(), console=console, screen=True, auto_refresh=False) as live:
            # Raw-mode key reading via termios/tty (POSIX-only; any failure here
            # falls back to the numbered plain transport). Arrow keys navigate;
            # q / Ctrl-C / Ctrl-D quit. No numbers are handled — the on-screen
            # hint only advertises the arrows.
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while True:
                    live.update(_frame(), refresh=True)
                    ch = sys.stdin.read(1)
                    if ch == "\r" or ch == "\n":
                        label, runner = entries[sel]
                        # Leave cooked mode + the alternate screen before running
                        # the action: rich's 30fps Live refresh would otherwise
                        # re-render the frame over the action's output and wipe
                        # the user's typed prompt input. Resume after.
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)
                        live.stop()
                        if runner is not None:
                            running_action = True
                            try:
                                runner(client)
                            except KeyboardInterrupt:
                                pass
                            running_action = False
                            print()
                        else:
                            return 0
                        tty.setraw(fd)
                        live.start()
                    elif ch == "\x1b":  # escape sequence (arrow keys)
                        # Read the rest of the arrow sequence: [A/B/C/D
                        seq = _escape_tail()
                        if seq == "[A":  # up
                            sel = (sel - 1) % n
                        elif seq == "[B":  # down
                            sel = (sel + 1) % n
                    elif ch in ("q", "Q", "\x03", "\x04"):  # q / Ctrl-C / Ctrl-D
                        return 0
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except (KeyboardInterrupt, Exception):
        # Transport failure → degrade to the robust plain menu. A runner
        # exception (running_action still True) is real work, not a key-reading
        # failure — propagate it instead of silently restarting the menu.
        if running_action:
            raise
        fallback = True
    finally:
        client.close()

    if fallback:
        return _run_plain(db_path, entries, banner)
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_menu(db_path: str, once: bool = False) -> int:
    """Open the interactive command menu for the store at ``db_path``.

    ``once`` prints the banner + a static menu listing and exits 0 (scriptable;
    the deterministic path the tests exercise — no live loop, no ``input``).
    Otherwise runs the rich arrow-key menu when ``rich`` is importable, else the
    stdlib numbered menu. Returns 0 on clean exit / Ctrl-C.
    """
    # The store is created on first write (_open_client(create=True)); make sure
    # its parent directory exists so a brand-new default path actually works.
    _ensure_parent_dir(db_path)

    # First-run detection: if the resolved path is a real on-disk file, show the
    # card count; otherwise (e.g. ":memory:" or a not-yet-created DB) it's a
    # first-run welcome. This mirrors _resolve_db_path's create-or-empty contract.
    if db_path != ":memory:" and os.path.exists(db_path):
        try:
            client = _open_client(db_path, create=False)
        except Exception:
            # A corrupt/partial DB (e.g. an interrupted prior run left a
            # schema-less file) → treat as first run rather than crashing the
            # whole menu. _open_client(create=True) below will reinitialize it.
            client = None
        if client is not None:
            try:
                count = len(client.store.all())
            except Exception:
                count = None  # unreadable store → first-run banner
            finally:
                client.close()
        else:
            count = None
    else:
        count = None

    banner = _banner(db_path, count=count)
    entries = _entries(db_path)

    if once:
        sys.stdout.write(banner + "\n\nWhat would you like to do?\n")
        sys.stdout.write(_numbered_menu(entries) + "\n")
        sys.stdout.flush()
        return 0

    return _run_rich(db_path, entries, banner)
