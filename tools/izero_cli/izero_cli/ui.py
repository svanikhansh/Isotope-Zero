"""Rendering layer for izero-cli.

All rich output lives here. The data layer (``db.py``) produces plain dict
contracts; these functions turn them into the sleek, dark-mode-friendly
terminal dashboard that the aesthetic guidelines call for.

Aesthetic system (enforced throughout)
-------------------------------------
- Headers & titles:   soft lavender / violet  (``#af87ff``)
- Highlights & badges: electric teal / cyan    (``#00d7ff`` / ``cyan``)
- Metrics & numbers:  bold white / gold       (``bold white`` / ``gold1``)
- Borders & muted:    subtle grey             (``dim``)
- Status badges:       emerald / amber / coral (``green3`` / ``gold3`` / ``red3``)
- Glyphs:              ⚡ latency  🧠 cards  🟢/🔴 daemon  📊 RAM  🔍 search  📄 card  💾 db

Every public render function takes the contract dict produced by ``db.py`` and
prints to the console. They never raise — error/missing cases render a red
panel so the user always sees something useful.
"""
from __future__ import annotations

from rich.box import ROUNDED
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Module-level console so renderers share a single output target. Callers may
# pass their own Console (e.g. a stderr-only one) via the `console` kwarg.
CONSOLE = Console()

# --- palette (single source of truth) --------------------------------------- #
_LAVENDER = "#af87ff"
_CYAN = "#00d7ff"
_GOLD = "bold gold1"
_WHITE = "bold white"
_DIM = "dim"
_GREEN = "green3"
_AMBER = "gold3"
_CORAL = "red3"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _badge(text: str, color: str) -> Text:
    """A small styled chip like ``[ text ]``."""
    return Text(f"[ {text} ]", style=f"bold {color}")


def _trunc(s: str, n: int) -> str:
    """Truncate to ``n`` chars with an ellipsis. None-safe."""
    if not s:
        return ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _human_age(seconds: float) -> str:
    """Compact relative age: 3.2s / 5.1m / 2.3h / 4d."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if s < 0:
        return "—"
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    if s < 86400:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d"


def _iso(ts: float) -> str:
    """Readable timestamp from an epoch float."""
    try:
        import datetime as _dt

        return _dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "—"


def _score_color(score: float) -> str:
    """Traffic-light color for a cosine / similarity score in [0,1]."""
    if score >= 0.75:
        return _GREEN
    if score >= 0.40:
        return _AMBER
    return _CORAL


def _title(text: str) -> Text:
    """A lavender bold title with a leading glyph slot already in `text`."""
    return Text(text, style=f"bold {_LAVENDER}")


def _error_panel(message: str, subtitle: str = "error") -> Panel:
    return Panel(
        Text(message, style=_CORAL),
        title=f"[bold {_CORAL}]✗ {subtitle}[/]",
        border_style=_CORAL,
        box=ROUNDED,
    )


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #
def render_inspect(data: dict, console: Console | None = None) -> None:
    c = console or CONSOLE
    if not data.get("exists"):
        c.print(_error_panel(data.get("error") or "database not found", "inspect"))
        return

    # --- summary panel (top of dashboard) ----------------------------------- #
    total = data["total_cards"]
    sup = data["superseded_count"]
    wal = data["wal"]
    q = data["quantization"]
    vr = data["vector_ram"]

    qbadge_color = _GREEN if q["status"] in ("int8_sq8", "mixed") else _AMBER
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style=_DIM)
    summary.add_column(style=_WHITE)
    summary.add_row("🧠 live cards", str(total))
    summary.add_row("🗂  superseded", str(sup))
    summary.add_row(
        "▦  quantization",
        Text.assemble(_badge(q["status"], qbadge_color), f"  ({q['cards_float32']} f32 · {q['cards_int8_sq8']} i8 · {q['cards_no_embedding']} none", style=_DIM),
    )
    summary.add_row("📕 journal", f"{wal['journal_mode']}")
    summary.add_row("📦 db size", Text(data["db_size_human"], style=_GOLD))
    wal_txt = Text.assemble(
        Text(wal["wal_size_human"], style=_GOLD),
        Text(f"   shm {wal['shm_size_human']}", style=_DIM),
    )
    summary.add_row("🗒  WAL", wal_txt)

    c.print(
        Panel(
            summary,
            title=_title("🧠 izero inspect — memory engine summary"),
            subtitle=f"[{_DIM}]{data['db_path']}[/]",
            border_style=_DIM,
            box=ROUNDED,
            padding=(1, 2),
        )
    )

    # --- vector matrix RAM footprint ----------------------------------------- #
    dim = vr["dim"]
    dim_str = str(dim) if dim is not None else "—"
    ram_table = Table.grid(padding=(0, 2))
    ram_table.add_column(style=_DIM)
    ram_table.add_column(style=_WHITE)
    ram_table.add_row("dimension", dim_str)
    ram_table.add_row("cards w/ embeddings", str(vr["cards_with_embeddings"]))
    ram_table.add_row("float32 bytes", str(vr["float32_bytes"]))
    ram_table.add_row("int8 bytes", str(vr["int8_bytes"]))
    ram_table.add_row("estimated RAM", Text(vr["ram_human"], style=_GOLD))
    c.print(
        Panel(
            ram_table,
            title=_title("📊 vector matrix RAM footprint"),
            border_style=_DIM,
            box=ROUNDED,
            padding=(1, 2),
        )
    )

    # --- access activity: recent + most-accessed, side by side --------------- #
    access = data["access"]
    recent_tbl = Table(
        title="⏱  most recent",
        header_style=f"bold {_CYAN}", border_style=_DIM, box=ROUNDED, padding=(0, 1),
    )
    recent_tbl.add_column("id", style=_CYAN, no_wrap=True, max_width=14)
    recent_tbl.add_column("fact", style="white", max_width=42)
    recent_tbl.add_column("age", style=_GOLD, no_wrap=True)
    for r in access["most_recent"]:
        recent_tbl.add_row(_trunc(str(r["id"]), 14), _trunc(r["fact"], 42), _human_age(r.get("age_seconds")))

    top_tbl = Table(
        title="👆 top accessed",
        header_style=f"bold {_CYAN}", border_style=_DIM, box=ROUNDED, padding=(0, 1),
    )
    top_tbl.add_column("id", style=_CYAN, no_wrap=True, max_width=14)
    top_tbl.add_column("fact", style="white", max_width=34)
    top_tbl.add_column("hits", style=_GOLD, no_wrap=True)
    for r in access["top_accessed"]:
        top_tbl.add_row(_trunc(str(r["id"]), 14), _trunc(r["fact"], 34), str(r["access_count"]))

    c.print(Panel(Group(recent_tbl, top_tbl), title=_title("📈 access activity"), border_style=_DIM, box=ROUNDED, padding=(1, 2)))

    # --- top tags / categories ---------------------------------------------- #
    tags = data["top_tags"]
    if tags:
        tag_tbl = Table(title="🏷  top tags",
                        header_style=f"bold {_CYAN}", border_style=_DIM, box=ROUNDED, padding=(0, 1))
        tag_tbl.add_column("tag", style=_LAVENDER)
        tag_tbl.add_column("count", style=_GOLD, no_wrap=True)
        for t in tags:
            # bar length proportional to count, capped
            bar = "█" * min(int(t["count"]) * 2, 24)
            tag_tbl.add_row(str(t["tag"]), f"{t['count']}  {bar}")
        c.print(Panel(tag_tbl, title=_title("🏷  categories & tags"), border_style=_DIM, box=ROUNDED, padding=(1, 2)))
    else:
        c.print(Panel(Text("no tags found", style=_DIM), title=_title("🏷  categories & tags"), border_style=_DIM, box=ROUNDED))


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def render_search(data: dict, console: Console | None = None) -> None:
    c = console or CONSOLE
    if not data.get("exists"):
        c.print(_error_panel(data.get("error") or "database not found", "search"))
        return

    mode = data.get("mode", "lexical")
    mode_color = _CYAN if mode == "semantic" else _AMBER
    latency = data.get("latency_ms", 0.0)

    header = Text.assemble(
        Text("🔍 search  ", style=f"bold {_LAVENDER}"),
        _badge(mode, mode_color),
        Text(f"   ⚡ {latency:.2f} ms   ·  top-k {data.get('top_k', 5)}", style=_GOLD),
    )

    results = data.get("results", [])
    if not results:
        c.print(
            Panel(
                Text(f'no matches for: "{data.get("query", "")}"', style=_DIM),
                title=header,
                border_style=_DIM,
                box=ROUNDED,
            )
        )
        return

    tbl = Table(
        header_style=f"bold {_CYAN}", border_style=_DIM, box=ROUNDED, padding=(0, 1),
    )
    tbl.add_column("#", style=_DIM, no_wrap=True, justify="right")
    tbl.add_column("score", no_wrap=True)
    tbl.add_column("card id", style=_CYAN, no_wrap=True, max_width=16)
    tbl.add_column("content preview", style="white", max_width=48)
    tbl.add_column("tags", style=_LAVENDER, max_width=20)
    for r in results:
        score = float(r.get("score", 0.0))
        tags = ", ".join(r.get("tags", []) or [])
        tbl.add_row(
            str(r["rank"]),
            Text(f"{score:.3f}", style=_score_color(score)),
            _trunc(str(r["card_id"]), 16),
            _trunc(r.get("fact", ""), 48),
            _trunc(tags, 20),
        )

    c.print(Panel(tbl, title=header, subtitle=f'[{_DIM}]query: "{data.get("query", "")}" · {data["db_path"]}[/]', border_style=_DIM, box=ROUNDED, padding=(1, 2)))


# --------------------------------------------------------------------------- #
# card
# --------------------------------------------------------------------------- #
def render_card(data: dict, console: Console | None = None) -> None:
    c = console or CONSOLE
    if not data.get("exists"):
        c.print(_error_panel(data.get("error") or "database not found", "card"))
        return
    if not data.get("found"):
        c.print(
            Panel(
                Text(f'no card with id "{data.get("card_id")}"', style=_AMBER),
                title=f"[bold {_AMBER}]📄 card not found[/]",
                border_style=_AMBER,
                box=ROUNDED,
            )
        )
        return

    card = data["card"]
    vec = data.get("vector")

    # superseded badge, if this is an audit-trail folded card
    sup_by = card.get("superseded_by")
    sup_line: Text | None = None
    if sup_by:
        sup_line = Text.assemble(
            _badge("SUPERSEDED", _CORAL),
            Text(f"  →  {sup_by}", style=_CORAL),
        )

    # --- fact (prominent) + evidence (muted) -------------------------------- #
    fact = Text(card.get("fact", ""), style=f"bold {_WHITE}")
    evidence = Text(card.get("evidence", "") or "—", style=_DIM)

    # --- metadata grid ------------------------------------------------------- #
    tags = card.get("tags") or []
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style=_DIM)
    meta.add_column(style=_WHITE)
    meta.add_row("id", Text(str(card.get("id", "")), style=_CYAN))
    meta.add_row("created", f"{_iso(card.get('timestamp'))}")
    meta.add_row("access count", Text(str(card.get("access_count", 0)), style=_GOLD))
    meta.add_row("last access", _iso(card.get("last_access")))
    meta.add_row("source tokens", str(card.get("source_tokens", 0)))
    meta.add_row("tags", Text(", ".join(tags) if tags else "—", style=_LAVENDER))

    body = Group(
        Text("fact", style=_DIM),
        fact,
        Text(""),
        Text("evidence", style=_DIM),
        evidence,
        Text(""),
        meta,
    )
    if sup_line:
        body = Group(sup_line, Text(""), body)

    c.print(
        Panel(
            body,
            title=_title("📄 izero card — deep inspection"),
            subtitle=f"[{_DIM}]{data['db_path']}[/]",
            border_style=_CORAL if sup_by else _DIM,
            box=ROUNDED,
            padding=(1, 2),
        )
    )

    # --- vector sub-panel ----------------------------------------------------- #
    if vec:
        dtype = vec.get("dtype")
        if dtype == "int8_sq8":
            dtype_color = _AMBER
        elif dtype == "float32":
            dtype_color = _CYAN
        else:
            dtype_color = _DIM
        norm = vec.get("norm")
        norm_str = f"{norm:.4f}" if norm is not None else "—"
        is_norm = vec.get("is_normalized")
        if is_norm is True:
            nrm_badge = _badge("normalized", _GREEN)
        elif is_norm is False:
            nrm_badge = _badge("not normalized", _AMBER)
        else:
            nrm_badge = Text("—", style=_DIM)

        vt = Table.grid(padding=(0, 2))
        vt.add_column(style=_DIM)
        vt.add_column(style=_WHITE)
        vt.add_row("dtype", _badge(str(dtype), dtype_color))
        vt.add_row("dimension", str(vec.get("dim") if vec.get("dim") is not None else "—"))
        vt.add_row("L2 norm", Text(norm_str, style=_GOLD))
        vt.add_row("unit?", nrm_badge)
        if dtype == "int8_sq8" and vec.get("q_scale") is not None:
            vt.add_row("q_scale", Text(f"{vec['q_scale']:.6f}", style=_GOLD))

        c.print(
            Panel(
                vt,
                title=_title("🧬 embedding vector"),
                border_style=_DIM,
                box=ROUNDED,
                padding=(1, 2),
            )
        )
    else:
        c.print(
            Panel(
                Text("no embedding stored (NULL)", style=_DIM),
                title=_title("🧬 embedding vector"),
                border_style=_DIM,
                box=ROUNDED,
            )
        )


# --------------------------------------------------------------------------- #
# daemon-status
# --------------------------------------------------------------------------- #
def render_daemon_status(data: dict, console: Console | None = None) -> None:
    c = console or CONSOLE

    active = data.get("daemon_active", False)
    status_badge = _badge("🟢 ACTIVE", _GREEN) if active else _badge("🔴 INACTIVE", _CORAL)

    # socket row
    sock_exists = data.get("socket_exists", False)
    sock_conn = data.get("socket_connected", False)
    if sock_conn:
        sock_badge = _badge("connected", _GREEN)
    elif sock_exists:
        sock_badge = _badge("no listener", _AMBER)
    else:
        sock_badge = _badge("not found", _CORAL)
    sock_err = data.get("socket_error")
    sock_err_txt = Text(f"  ({sock_err})", style=_DIM) if sock_err and not sock_conn else Text("")

    shm_exists = data.get("shm_exists", False)
    shm_badge = _badge("present" if shm_exists else "absent", _GREEN if shm_exists else _DIM)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=_DIM)
    grid.add_column(style=_WHITE)
    grid.add_row("socket", Text.assemble(Text(data.get("socket_path", ""), style=_CYAN), Text("  "), sock_badge, sock_err_txt))
    grid.add_row("shm", Text.assemble(Text(data.get("shm_path", ""), style=_CYAN), Text("  "), shm_badge))

    procs = data.get("processes") or []
    children = [grid, Text("")]
    if procs:
        pt = Table(
            "pid", "rss", "name", "cmd",
            header_style=f"bold {_CYAN}", border_style=_DIM, box=ROUNDED, padding=(0, 1),
        )
        pt.add_column("pid", style=_CYAN, no_wrap=True)
        pt.add_column("rss", style=_GOLD, no_wrap=True)
        pt.add_column("name", style="white", max_width=24)
        pt.add_column("cmd", style=_DIM, max_width=48)
        for p in procs:
            rss = p.get("rss_mb")
            rss_str = f"{rss:.1f} MB" if isinstance(rss, (int, float)) else "—"
            pt.add_row(str(p.get("pid", "")), rss_str, _trunc(str(p.get("name", "")), 24), _trunc(str(p.get("cmd", "")), 48))
        children.append(pt)
    else:
        children.append(Text("no isotope_zero / izero processes detected", style=_DIM))

    c.print(
        Panel(
            Group(*children),
            title=_title("🟢 izero daemon-status"),
            subtitle=Text.assemble(Text("daemon: ", style=_DIM), status_badge),
            border_style=_GREEN if active else _CORAL,
            box=ROUNDED,
            padding=(1, 2),
        )
    )
