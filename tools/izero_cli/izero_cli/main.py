"""izero-cli: Isotope Zero memory engine inspector.

The CLI entrypoint. Wires argparse subcommands -> db functions -> ui renderers.
Pure plumbing: no SQL, no schema knowledge, no business logic lives here.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

# Default top-k for search. Kept here as a CLI-level default; the db layer
# may also default internally, but the user-facing flag lives here.
_DEFAULT_TOP_K = 5


# --------------------------------------------------------------------------- #
# Help / usage guide (rich-rendered, bypasses argparse default formatting)
# --------------------------------------------------------------------------- #
def _print_help_guide() -> None:
    """Render a rich-styled command guide for `izero --help`.

    Printed directly to stdout via a rich Console. Kept self-contained so the
    help path is fast (no db/ui import required) and visually consistent with
    the lavender/cyan/dim aesthetic used across izero_cli.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    title = Text("⚡ izero — Isotope Zero memory inspector", style="bold magenta")
    subtitle = Text(
        "A read-only terminal inspection tool for Isotope Zero memory engines.",
        style="dim italic",
    )

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("Command", style="bold magenta", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Usage", style="dim")

    table.add_row(
        "🧠 inspect",
        "Summarize a memory engine DB (cards, schema, size).",
        "izero inspect <db_path>",
    )
    table.add_row(
        "🔍 search",
        "Semantic search over a memory engine.",
        "izero search <db_path> \"<query>\" [--top-k N]",
    )
    table.add_row(
        "📄 card",
        "Fetch a single memory card by id.",
        "izero card <db_path> <card_id>",
    )
    table.add_row(
        "🟢 daemon-status",
        "Report the Isotope Zero daemon health.",
        "izero daemon-status",
    )
    table.add_row(
        "👀 watch",
        "Live-tail new/superseded cards (read-only).",
        'izero watch <db_path> [--interval 1.0]',
    )
    table.add_row(
        "🏥 doctor",
        "Health & integrity scorecard (read-only).",
        "izero doctor <db_path>",
    )
    table.add_row(
        "↔️ diff",
        "Compare two memory DBs (read-only).",
        "izero diff <db1> <db2> [--since TS]",
    )
    table.add_row(
        "📦 export",
        "Dump cards to jsonl/csv/md (read-only).",
        "izero export <db_path> --out <file> [--format] [--tag]",
    )
    table.add_row(
        "📥 import",
        "Seed cards from jsonl (writes to DB).",
        "izero import <db_path> <file> [--format jsonl]",
    )
    table.add_row(
        "🧹 vacuum",
        "WAL checkpoint + VACUUM (writes to DB).",
        "izero vacuum <db_path>",
    )
    table.add_row(
        "⚡ benchmark",
        "Search latency p50/p90/p99 + QPS (read-only).",
        "izero benchmark <db_path> [--queries 100]",
    )
    table.add_row(
        "📊 stats",
        "Tag/age/turnover analytics (read-only).",
        "izero stats <db_path>",
    )

    panel = Panel.fit(
        table,
        title=title,
        subtitle=subtitle,
        border_style="magenta",
        padding=(1, 2),
    )
    console.print(panel)
    console.print(
        Text(
            "Pass a command and its arguments to run it. "
            "Most commands are read-only (mode=ro + query_only=ON). "
            "`import` and `vacuum` require write access. "
            "Exit codes: 0 success, 1 error, 2 usage fault.",
            style="dim",
        )
    )


# --------------------------------------------------------------------------- #
# Subcommand handlers (lazy-import db + ui so --help stays fast & decoupled)
# --------------------------------------------------------------------------- #
def _cmd_inspect(args: argparse.Namespace) -> int:
    """`izero inspect <db_path>`."""
    from .db import inspect_db
    from .ui import render_inspect

    data = inspect_db(args.db_path)
    # Pass contract dict straight through; ui renders red panel on error.
    render_inspect(data)
    return 1 if data.get("error") and not data.get("exists", True) else 0


def _cmd_search(args: argparse.Namespace) -> int:
    """`izero search <db_path> <query> [--top-k N]`."""
    from .db import search_db
    from .ui import render_search

    data = search_db(args.db_path, args.query, top_k=args.top_k)
    render_search(data)
    return 1 if data.get("error") and not data.get("exists", True) else 0


def _cmd_card(args: argparse.Namespace) -> int:
    """`izero card <db_path> <card_id>`."""
    from .db import get_card
    from .ui import render_card

    data = get_card(args.db_path, args.card_id)
    render_card(data)
    return 1 if data.get("error") and not data.get("exists", True) else 0


def _cmd_daemon_status(args: argparse.Namespace) -> int:
    """`izero daemon-status` (no db_path)."""
    from .db import daemon_status
    from .ui import render_daemon_status

    data = daemon_status()
    render_daemon_status(data)
    return 1 if data.get("error") else 0


# --------------------------------------------------------------------------- #
# New subcommand handlers (the 8-command suite). Each is a thin shim that
# lazy-imports its command module from .commands and delegates to its `cmd`
# entry point. Keeping the shim here means main.py stays the single dispatch
# surface; the command modules own all SQL + rendering logic.
# --------------------------------------------------------------------------- #
def _cmd_watch(args: argparse.Namespace) -> int:
    """`izero watch <db_path> [--interval N]` — live WAL tailer."""
    from .commands.watch import cmd

    return int(cmd(args) or 0)


def _cmd_doctor(args: argparse.Namespace) -> int:
    """`izero doctor <db_path>` — health & integrity scorecard."""
    from .commands.doctor import cmd

    return int(cmd(args) or 0)


def _cmd_diff(args: argparse.Namespace) -> int:
    """`izero diff <db1> <db2> [--since TS]` — session comparison."""
    from .commands.diff import cmd

    return int(cmd(args) or 0)


def _cmd_export(args: argparse.Namespace) -> int:
    """`izero export <db_path> --out <file> [--format] [--tag]`."""
    from .commands.export import cmd

    return int(cmd(args) or 0)


def _cmd_import(args: argparse.Namespace) -> int:
    """`izero import <db_path> <file> [--format jsonl]` (mutating)."""
    from .commands.import_cmd import cmd

    return int(cmd(args) or 0)


def _cmd_vacuum(args: argparse.Namespace) -> int:
    """`izero vacuum <db_path>` — WAL checkpoint + VACUUM (mutating)."""
    from .commands.vacuum import cmd

    return int(cmd(args) or 0)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """`izero benchmark <db_path> [--queries N]` — latency harness."""
    from .commands.benchmark import cmd

    return int(cmd(args) or 0)


def _cmd_stats(args: argparse.Namespace) -> int:
    """`izero stats <db_path>` — memory distribution analytics."""
    from .commands.stats import cmd

    return int(cmd(args) or 0)


# --------------------------------------------------------------------------- #
# Parser construction
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with the four subcommands.

    A thin custom formatter is attached so that when argparse DOES print usage
    (e.g. on a missing required arg), the description style stays consistent.
    The full rich guide is rendered separately in `main()` for the --help path.
    """
    parser = argparse.ArgumentParser(
        prog="izero",
        description="Isotope Zero memory engine inspector (read-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="izero 0.1.0",
    )

    sub = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        required=True,
    )

    # inspect -------------------------------------------------------------- #
    p_inspect = sub.add_parser(
        "inspect",
        help="Summarize a memory engine DB.",
        description="Summarize a memory engine DB (cards, schema, size).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_inspect.add_argument("db_path", help="Path to the memory engine SQLite DB.")
    p_inspect.set_defaults(func=_cmd_inspect)

    # search --------------------------------------------------------------- #
    p_search = sub.add_parser(
        "search",
        help="Semantic search over a memory engine.",
        description="Semantic search over a memory engine DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_search.add_argument("db_path", help="Path to the memory engine SQLite DB.")
    p_search.add_argument("query", help="Natural-language query string.")
    p_search.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help=f"Maximum number of results to return (default: {_DEFAULT_TOP_K}).",
    )
    p_search.set_defaults(func=_cmd_search)

    # card ----------------------------------------------------------------- #
    p_card = sub.add_parser(
        "card",
        help="Fetch a single memory card by id.",
        description="Fetch a single memory card by id.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_card.add_argument("db_path", help="Path to the memory engine SQLite DB.")
    p_card.add_argument("card_id", help="The card id to retrieve.")
    p_card.set_defaults(func=_cmd_card)

    # daemon-status -------------------------------------------------------- #
    p_daemon = sub.add_parser(
        "daemon-status",
        help="Report the Isotope Zero daemon health.",
        description="Report the Isotope Zero daemon health (no db_path needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_daemon.set_defaults(func=_cmd_daemon_status)

    # --- the 8-command expansion suite ------------------------------------ #
    # watch ---------------------------------------------------------------- #
    p_watch = sub.add_parser(
        "watch",
        help="Live-tail new/superseded memory cards.",
        description="Polling WAL tailer: streams newly created or superseded cards as they are written.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_watch.add_argument("db_path", help="Path to the memory engine SQLite DB.")
    p_watch.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Poll interval in seconds (default: 1.0).",
    )
    p_watch.set_defaults(func=_cmd_watch)

    # doctor --------------------------------------------------------------- #
    p_doctor = sub.add_parser(
        "doctor",
        help="Full health & integrity diagnostic.",
        description="Scans vector integrity, WAL fragmentation, FTS5, integrity, and daemon IPC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_doctor.add_argument("db_path", help="Path to the memory engine SQLite DB.")
    p_doctor.set_defaults(func=_cmd_doctor)

    # diff ----------------------------------------------------------------- #
    p_diff = sub.add_parser(
        "diff",
        help="Compare two memory DBs (session delta).",
        description="Compares two memory states and reports added/superseded/deleted cards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_diff.add_argument("db1", help="Baseline memory engine SQLite DB.")
    p_diff.add_argument("db2", help="Comparison (after-session) memory engine SQLite DB.")
    p_diff.add_argument(
        "--since",
        type=float,
        default=None,
        help="Only compare cards with timestamp >= this epoch (session window).",
    )
    p_diff.set_defaults(func=_cmd_diff)

    # export --------------------------------------------------------------- #
    p_export = sub.add_parser(
        "export",
        help="Dump memory cards to jsonl/csv/md.",
        description="Exports stored memory cards into a portable data format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_export.add_argument("db_path", help="Path to the memory engine SQLite DB.")
    p_export.add_argument("--out", required=True, help="Output file path.")
    p_export.add_argument(
        "--format",
        choices=["jsonl", "csv", "md"],
        default="jsonl",
        help="Output format (default: jsonl).",
    )
    p_export.add_argument(
        "--tag",
        default=None,
        help="Only export cards carrying this metadata tag.",
    )
    p_export.set_defaults(func=_cmd_export)

    # import --------------------------------------------------------------- #
    p_import = sub.add_parser(
        "import",
        help="Seed memory cards from a file (writes to DB).",
        description="Imports memory cards from a jsonl file into an existing or fresh DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_import.add_argument("db_path", help="Path to the memory engine SQLite DB (created if missing).")
    p_import.add_argument("file", help="Input jsonl file to import.")
    p_import.add_argument(
        "--format",
        choices=["jsonl"],
        default="jsonl",
        help="Input format (default: jsonl).",
    )
    p_import.set_defaults(func=_cmd_import)

    # vacuum --------------------------------------------------------------- #
    p_vacuum = sub.add_parser(
        "vacuum",
        help="WAL checkpoint + VACUUM (writes to DB).",
        description="Flushes the WAL and reclaims disk space from purged cards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_vacuum.add_argument("db_path", help="Path to the memory engine SQLite DB.")
    p_vacuum.set_defaults(func=_cmd_vacuum)

    # benchmark ------------------------------------------------------------ #
    p_bench = sub.add_parser(
        "benchmark",
        help="Local search latency harness.",
        description="Runs N sample searches and reports p50/p90/p99 + cold/warm QPS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_bench.add_argument("db_path", help="Path to the memory engine SQLite DB.")
    p_bench.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of sample queries to run (default: 100).",
    )
    p_bench.set_defaults(func=_cmd_benchmark)

    # stats ---------------------------------------------------------------- #
    p_stats = sub.add_parser(
        "stats",
        help="Memory distribution analytics.",
        description="Tag distribution, age histogram, and turnover frequency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_stats.add_argument("db_path", help="Path to the memory engine SQLite DB.")
    p_stats.set_defaults(func=_cmd_stats)

    return parser


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns an exit code (0 ok, 1 error, 2 usage fault).

    If no args or --help/-h is requested, render the rich command guide and
    exit 0. Otherwise hand off to argparse for real parsing & dispatch.
    """
    raw = list(sys.argv[1:]) if argv is None else list(argv)

    # Help path: no args, or explicit help flag anywhere in the argv list.
    wants_help = (len(raw) == 0) or ("-h" in raw) or ("--help" in raw)

    if wants_help:
        _print_help_guide()
        return 0

    parser = _build_parser()
    # parse_args will SystemExit(2) on usage faults (missing required arg,
    # unknown subcommand) — that is the intended behavior per spec.
    args = parser.parse_args(raw)
    # `required=True` on subparsers guarantees `func` is set here.
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
