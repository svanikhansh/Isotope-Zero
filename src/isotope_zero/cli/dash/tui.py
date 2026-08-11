"""isotope_zero CLI — Terminal TUI dashboard surface.

Renders the shared dashboard state (``dash.data.collect_state``) as a
polished multi-panel live terminal dashboard using the ``rich`` library and
the black/beige/white theme.

Layout (top→bottom, when rich is available):
    header       — brand + db path + embedding mode
    KPI row      — cards · size · tokens · reclaimable
    vitality     — stacked bucket bar + fresh/aging/decayed counts + pct
    tags strip   — top tags as an inline distribution (when present)
    recent       — newest memories, fact + tags + age + vitality bar
    decay        — decay candidates, fact + vitality + age

Transport strategy mirrors the legacy dashboard:
    rich   — ``rich.live.Live`` with an alternate screen buffer (smooth
             in-place refresh, terminal restored on exit). Keyboard: any key
             quits (Esc/Ctrl-C/q).
    plain  — stdlib-only clear-and-reprint loop, when rich is absent. Renders
             a compact box via ``ui.glyphs.build_box``.
    once   — ``--once`` prints one static frame and exits (scriptable/pipable,
             the deterministic path tests exercise).

Read-only: never writes to the store, never spawns the daemon.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable

from .data import collect_state


_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"  # U+2581..U+2588, low→high


def _sparkline(values: list[float], width: int = 14, style: str = "izero.dim") -> Text:
    """Render a mini sparkline from a time series (rich15 dropped Sparkline).

    Draws ``width`` glyphs from the last ``width`` samples, normalized to the
    sample range; a flat/empty series renders as a muted baseline. Returns a
    rich Text so the caller can style/append it.
    """
    from rich.text import Text

    if not values:
        return Text("▁" * width, style="izero.bar.empty")
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    text = Text()
    for v in vals:
        idx = int(round((v - lo) / span * (len(_SPARK_BLOCKS) - 1)))
        text.append(_SPARK_BLOCKS[idx], style=style)
    return text


def _rich_layout(state: dict[str, Any], interval: float, history: list[dict[str, Any]] | None = None):
    """Build the rich Layout grid for the dashboard frame."""
    import time as _time

    from rich.console import Group, Text
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from ..ui.glyphs import build_vitality_bar, glyph
    from ..ui.theme import vitality_style

    history = history or []
    count = state["count"]
    hist = state["histogram"]
    f_, a_, d_ = hist["fresh"], hist["aging"], hist["decayed"]
    pct = state["pct"]

    # --- Header: identity left, live clock right (the "alive" moment) ---
    now = _time.localtime()
    clock = Text.assemble(
        (_time.strftime("%H", now), "izero.number"),
        (":", "izero.dim"),
        (_time.strftime("%M", now), "izero.number"),
        (":", "izero.dim"),
        (_time.strftime("%S", now), "izero.number"),
    )
    from rich.table import Table as _Table

    header_grid = _Table.grid(expand=True)
    header_grid.add_column(ratio=1)
    header_grid.add_column(justify="right")
    header_grid.add_row(
        Text.assemble(
            ("isotope_zero", "izero.hero"),
            ("  ", "izero.body"),
            ("local-first cognitive memory", "izero.subtitle"),
            ("\n", ""),
            ("db: ", "izero.dim"),
            (state["db_path"], "izero.accent"),
            ("   ·   ", "izero.dim"),
            ("mode: ", "izero.dim"),
            (state["mode"], "izero.info"),
        ),
        clock,
    )
    header_panel = Panel(
        header_grid,
        border_style="izero.panel.border",
        style="izero.panel",
        padding=(0, 1),
    )

    # --- KPI row ---
    def kpi(label: str, value: str, accent: str = "izero.number") -> Panel:
        return Panel(
            Text.assemble((value + "\n", accent), (label, "izero.dim")),
            border_style="izero.panel.border",
            style="izero.panel",
            padding=(0, 1),
        )

    reclaim = state["reclaimable_tokens"]
    reclaim_text = f"~{reclaim}" if reclaim > 0 else "0"
    kpis = [
        kpi("cards", str(count)),
        kpi("size", state["size_human"], "izero.accent"),
        kpi("tokens", f"~{state['tokens_total']}"),
        kpi("reclaimable", reclaim_text, "izero.warning" if reclaim > 0 else "izero.dim"),
    ]

    # --- Vitality bar + buckets ---
    bar_segments = build_vitality_bar(f_, a_, d_, width=26, use_rich=True)
    bar_text = Text()
    for seg, style in bar_segments:
        bar_text.append(seg, style=style)
    bar_text.append(f"  {count} total", "izero.dim")

    # Two-line bucket summary (fits narrow panels; wrapping is the fallback).
    bucket_row = Group(
        Text.assemble(
            ("● ", "izero.bar.fresh"),
            (f"fresh {f_}", "izero.body"),
            (f"  ({pct['fresh']:.0f}%)", "izero.dim"),
            ("    ", ""),
            ("● ", "izero.bar.aging"),
            (f"aging {a_}", "izero.body"),
            (f"  ({pct['aging']:.0f}%)", "izero.dim"),
        ),
        Text.assemble(
            ("● ", "izero.bar.decay"),
            (f"decayed {d_}", "izero.body"),
            (f"  ({pct['decayed']:.0f}%)", "izero.dim"),
        ),
    )
    # Activity sparkline (cards per tick) — the "still alive" rhythm.
    counts = [h.get("count", 0) for h in history]
    activity_row = Text.assemble(
        ("cards ", "izero.dim"),
        _sparkline(counts, width=14, style="izero.bar.fresh"),
    ) if history else Text()

    vitality_panel = Panel(
        Group(bar_text, Text(""), bucket_row, activity_row),
        title="[izero.panel.title]vitality[/]",
        border_style="izero.panel.border",
        style="izero.panel",
        padding=(0, 1),
    )

    # --- Tags strip ---
    panels = [header_panel, *kpis, vitality_panel]
    top_tags = state.get("top_tags") or []
    if top_tags:
        tag_row = Text()
        for i, t in enumerate(top_tags):
            if i > 0:
                tag_row.append("  ·  ", "izero.dim")
            tag_row.append(f"{t['tag']}", "izero.tag")
            tag_row.append(f" {t['count']}", "izero.number")
        panels.append(
            Panel(
                tag_row,
                title="[izero.panel.title]tags[/]",
                border_style="izero.panel.border",
                style="izero.panel",
                padding=(0, 1),
            )
        )

    # --- Recent ---
    recent = state["recent"]
    if recent:
        rtable = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
        rtable.add_column("fact", style="izero.body", ratio=1, overflow="ellipsis", no_wrap=True)
        rtable.add_column("tags", style="izero.tag", width=16, overflow="ellipsis", no_wrap=True)
        rtable.add_column("vitality", width=8, no_wrap=True)
        rtable.add_column("age", style="izero.timestamp", justify="right", width=6, no_wrap=True)
        for r in recent:
            tags = ", ".join(r["tags"][:3]) if r["tags"] else ""
            v = r["vitality"]
            rbar = Text()
            segs = build_vitality_bar(
                1 if r["bucket"] == "fresh" else 0,
                1 if r["bucket"] == "aging" else 0,
                1 if r["bucket"] == "decayed" else 0,
                width=8, use_rich=True,
            )
            for seg, style in segs:
                rbar.append(seg, style=style)
            rtable.add_row(r["fact"], tags, rbar, f"{r['age_days']}d")
        panels.append(
            Panel(
                rtable,
                title=f"[izero.panel.title]recent[/]",
                border_style="izero.panel.border",
                style="izero.panel",
                padding=(0, 1),
            )
        )

    # --- Decay ---
    decay = state["decay"]
    if decay or state["decay_total"]:
        dtable = Table(show_header=True, box=None, padding=(0, 1), pad_edge=False)
        dtable.add_column("fact", style="izero.body", ratio=1, overflow="ellipsis", no_wrap=True)
        dtable.add_column("vitality", style="izero.number", justify="right", width=8, no_wrap=True)
        dtable.add_column("age", style="izero.timestamp", justify="right", width=6, no_wrap=True)
        for d in decay:
            v = d["vitality"]
            dtable.add_row(
                d["fact"],
                f"{v:.3f}",
                f"{d['age_days']}d",
            )
        reclaim_note = f"  ·  ~{state['reclaimable_tokens']} tokens reclaimable" if state["reclaimable_tokens"] > 0 else ""
        panels.append(
            Panel(
                dtable,
                title=f"[izero.panel.title]decay candidates ({state['decay_total']})[/]{reclaim_note}",
                border_style="izero.panel.border",
                style="izero.panel",
                padding=(0, 1),
            )
        )

    # --- Footer ---
    footer = Text.assemble(
        ("refresh ", "izero.hint"),
        (f"{interval:g}s", "izero.number"),
        ("   ·   ", "izero.hint"),
        ("esc/q/Ctrl-C to exit", "izero.hint"),
        ("   ·   ", "izero.hint"),
        ("read-only", "izero.hint"),
    )
    footer_panel = Panel(
        footer,
        border_style="izero.panel.border",
        style="izero.panel",
        padding=(0, 1),
    )

    # Glue panels into a Layout: KPI row split horizontally, then a nested
    # 2-column main region (vitality + tags on the left, recent + decay on the
    # right) so tall tables don't starve the compact panels at small terminal
    # heights. Header + footer are fixed-height frames.
    from rich.layout import Layout

    # panels = [header, kpi0, kpi1, kpi2, kpi3, *body...]
    header, k0, k1, k2, k3, *body = panels

    kpi_row = Layout(name="kpi_row", size=4)
    kpi_row.split_row(
        Layout(k0, name="kpi0"),
        Layout(k1, name="kpi1"),
        Layout(k2, name="kpi2"),
        Layout(k3, name="kpi3"),
    )

    # Split the body across two columns (left = vitality + tags, right =
    # recent + decay); each column splits its own stack. An empty store renders
    # only vitality+tags → single-column left.
    main = Layout(name="main")
    if len(body) <= 2:
        # Empty/small store: stack the body panels full-width.
        main.split_column(*[Layout(p, name=f"body_{i}") for i, p in enumerate(body)])
    else:
        # Two-column body: left = vitality + tags, right = recent (+ decay).
        left = Layout(name="left", ratio=3)
        left.split_column(Layout(body[0], name="left_0"), Layout(body[1], name="left_1"))
        right = Layout(name="right", ratio=5, minimum_size=40)
        if len(body) >= 4:
            right.split_column(Layout(body[2], name="right_0"), Layout(body[3], name="right_1"))
        else:
            right.split_column(Layout(body[2], name="right_0"))
        main.split_row(left, right)

    layout = Layout()
    layout.split_column(
        Layout(header, name="header", size=5),
        kpi_row,
        main,
        Layout(footer_panel, name="footer", size=3),
    )
    return layout


def render_frame(
    state: dict[str, Any],
    interval: float,
    use_rich: bool = True,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Render one dashboard frame as a string (rich markup when use_rich).

    ``history`` is an optional ring of prior snapshots (for the activity
    sparkline); renderers that don't use it (plain, tests) ignore it.
    """
    if use_rich:
        from ..ui.theme import get_console

        console = get_console()
        with console.capture() as capture:
            console.print(_rich_layout(state, interval, history))
        return capture.get()
    return _plain_frame(state, interval)


# --------------------------------------------------------------------------- #
# Plain fallback (stdlib-only, no rich) — mirrors legacy _render_frame
# --------------------------------------------------------------------------- #
def _plain_frame(state: dict[str, Any], interval: float) -> str:
    from ..ui.glyphs import build_vitality_bar, glyph

    hist = state["histogram"]
    f_, a_, d_ = hist["fresh"], hist["aging"], hist["decayed"]
    bar = build_vitality_bar(f_, a_, d_, width=20) if state["count"] else glyph("bar_empty") * 20

    title_src = state["db_path"] or ":memory:"
    title = f"isotope_zero · {title_src}"

    lines: list[str] = []
    lines.append(f"cards: {state['count']:<6} size: {state['size_human']:<9} mode: {state['mode']}")
    lines.append(f"vitality: {bar}  fresh {f_} / aging {a_} / decay {d_}")
    tokens_line = f"tokens: {state['tokens_total']} total"
    if state["reclaimable_tokens"] > 0:
        tokens_line += f"  ·  ~{state['reclaimable_tokens']} reclaimable"
    lines.append(tokens_line)

    if state["recent"]:
        lines.append("recent:")
        for r in state["recent"]:
            lines.append(f"  · {r['fact'][:54]}  ({r['age_days']}d)")
    else:
        lines.append("recent: (none)")

    if state["decay_total"]:
        lines.append(f"decay candidates ({state['decay_total']}):")
        for d in state["decay"]:
            lines.append(f"  · {d['fact'][:50]}  v={d['vitality']:.3f} age={d['age_days']}d")
    else:
        lines.append("decay candidates: none")

    footer = f"refresh {interval:g}s   ·   [esc/Ctrl-C to exit]"

    from ..ui.glyphs import build_box

    return build_box(title, lines, footer, heavy=True)


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #
def _sample_history(history: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Append ``state`` to the ring buffer (keeps only what the sparkline needs)."""
    history.append({"count": state["count"]})
    if len(history) > 40:
        del history[: len(history) - 40]
    return history


def _run_plain(supplier: Callable[[], dict[str, Any]], interval: float) -> int:
    """Stdlib-only refresh loop: clear screen, print frame, sleep. Any key exits."""
    history: list[dict[str, Any]] = []
    try:
        while True:
            state = supplier()
            history = _sample_history(history, state)
            frame = _plain_frame(state, interval)
            sys.stdout.write("\033[2J\033[H" + frame + "\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        return 0


def _run_rich(supplier: Callable[[], dict[str, Any]], interval: float) -> int:
    """Smooth in-place refresh via rich.live (lazy import; degrades to plain)."""
    try:
        from rich.live import Live
        from rich.console import Console
        from ..ui.theme import get_console
    except ImportError:
        return _run_plain(supplier, interval)

    console = get_console()
    rps = max(1, int(1.0 / max(interval, 0.1)))
    history: list[dict[str, Any]] = []
    try:
        state = supplier()
        history = _sample_history(history, state)
        with Live(
            render_frame(state, interval, use_rich=True, history=history),
            console=console,
            refresh_per_second=rps,
            screen=True,  # alternate screen buffer → terminal restored on exit
        ) as live:
            while True:
                time.sleep(interval)
                state = supplier()
                history = _sample_history(history, state)
                live.update(render_frame(state, interval, use_rich=True, history=history))
    except KeyboardInterrupt:
        return 0


def run_live(
    store,
    db_path: str,
    interval: float = 2.0,
    once: bool = False,
    force_plain: bool = False,
) -> int:
    """Run the dashboard for ``store`` at ``db_path``.

    Returns 0 on clean exit; the missing-DB guard is the caller's job.
    ``force_plain`` forces the stdlib transport (e.g. when ``--no-color`` or
    non-TTY — a piped stdout must print a clean frame, not control codes).
    """
    if interval <= 0:
        print("interval must be > 0 seconds", file=sys.stderr)
        return 1

    def supplier() -> dict[str, Any]:
        return collect_state(store, db_path)

    if once:
        sys.stdout.write(render_frame(supplier(), interval, use_rich=not force_plain) + "\n")
        sys.stdout.flush()
        return 0

    if force_plain or not sys.stdout.isatty():
        return _run_plain(supplier, interval)

    try:
        return _run_rich(supplier, interval)
    except KeyboardInterrupt:
        return 0


__all__ = ["render_frame", "run_live", "_run_plain", "_run_rich", "_plain_frame"]
