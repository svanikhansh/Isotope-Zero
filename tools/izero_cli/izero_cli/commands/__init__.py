"""Subcommand handlers for izero-cli.

Each command lives in its own module so the subcommand surface can grow without
one giant file. Modules export a ``cmd(args: argparse.Namespace) -> int`` entry
point (the dispatcher contract used by ``main.py``) plus, where useful, a data
function returning a plain dict contract and a render function.

Layout::

    commands/
        __init__.py      # this file — registry of the 8 new subcommands
        _dbutil.py       # shared read-only + write-access SQLite helpers
        _uiutil.py       # shared rich palette / badge / table helpers (re-export)
        watch.py         # izero watch <db> [--interval]
        doctor.py        # izero doctor <db>
        diff.py          # izero diff <db1> <db2> [--since]
        export.py        # izero export <db> --out <file> [--format] [--tag]
        import_cmd.py    # izero import <db> <file> [--format]
        vacuum.py        # izero vacuum <db>
        benchmark.py     # izero benchmark <db> [--queries]
        stats.py         # izero stats <db>

The four original commands (inspect/search/card/daemon-status) stay wired
directly in ``main.py``; the eight new ones are dispatched through this package.
"""
from __future__ import annotations

# Dispatcher registry: maps the CLI subcommand name -> the handler callable.
# main.py imports this lazily so `izero --help` stays fast and decoupled.
# Each handler has signature (args: argparse.Namespace) -> int (exit code).
COMMANDS: dict[str, str] = {
    "watch": "izero_cli.commands.watch:cmd",
    "doctor": "izero_cli.commands.doctor:cmd",
    "diff": "izero_cli.commands.diff:cmd",
    "export": "izero_cli.commands.export:cmd",
    "import": "izero_cli.commands.import_cmd:cmd",
    "vacuum": "izero_cli.commands.vacuum:cmd",
    "benchmark": "izero_cli.commands.benchmark:cmd",
    "stats": "izero_cli.commands.stats:cmd",
}
