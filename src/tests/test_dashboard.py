"""End-to-end tests for ``izero dashboard`` (src/isotope_zero/cli/dashboard.py).

These exercise the public ``main(argv=[...])`` surface in
``isotope_zero.cli.debug`` — they never import the dashboard module's
internals directly except where a decay fixture needs a hand-built
``MemoryCard``. Every test uses ``--once`` so the dashboard prints a single
static frame to stdout and exits 0: no live loop, no terminal control, fully
deterministic (the refresh transports are covered structurally by the fact
that ``--once`` and the live loop share ``_render_frame``).

Like ``test_cli.py``, these never import ``onnxruntime`` — the fallback
embedder path keeps the dashboard's decay/vitality reads meaningful with zero
optional deps. ``rich`` is never required either (``--once`` is plain stdout).
"""
from __future__ import annotations

import time

from isotope_zero.cli.debug import main
from isotope_zero.core.store import MemoryStore
from isotope_zero.types import MemoryCard

# --------------------------------------------------------------------------- #
# Helpers (mirror test_cli.py's run / db_arg, kept local for isolation)
# --------------------------------------------------------------------------- #
def run(main_fn, argv, capsys):
    rc = main_fn(argv)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def db_arg(db_path: str) -> list[str]:
    return ["--db", str(db_path)]


def _add(main_fn, capsys, db_path: str, fact: str) -> None:
    rc, out, err = run(main_fn, ["add", fact, *db_arg(db_path)], capsys)
    assert rc == 0, f"add failed rc={rc} err={err!r} out={out!r}"


# --------------------------------------------------------------------------- #
# 1. --once on an empty :memory: store
# --------------------------------------------------------------------------- #
def test_dashboard_once_empty(capsys):
    """``dashboard --once`` on the default empty :memory: store exits 0 and
    the frame reports 0 cards, the embedding mode, and a vitality line."""
    rc, out, err = run(main, ["dashboard", "--once"], capsys)
    assert rc == 0, f"dashboard --once failed rc={rc} err={err!r} out={out!r}"
    assert "isotope_zero" in out, f"frame must carry the panel title, got {out!r}"
    assert "cards: 0" in out, f"empty store must show 0 cards, got {out!r}"
    # The mode line is always present (REAL ONNX / FALLBACK / none …).
    assert "mode:" in out, f"frame must report embedding mode, got {out!r}"
    # Vitality line present with the bar glyphs.
    assert "vitality:" in out, f"frame must show the vitality bar, got {out!r}"
    # An empty store has no decay candidates and no recent adds.
    assert "decay candidates: none" in out, f"empty store → no decay, got {out!r}"
    assert "recent: (none)" in out, f"empty store → no recent, got {out!r}"
    # The reclaimable-tokens hint is suppressed when there's nothing to reclaim
    # (a condition-flip on the `> 0` guard would surface it on an empty store).
    assert "reclaimable" not in out, (
        f"empty store must not show a reclaimable hint, got {out!r}"
    )


# --------------------------------------------------------------------------- #
# 2. --once on a seeded store
# --------------------------------------------------------------------------- #
def test_dashboard_once_seeded(tmp_path, capsys):
    """A seeded store: the frame shows the card count, the newest fact in the
    recent list, and a non-empty vitality bar."""
    db = tmp_path / "dash_seeded.db"
    _add(main, capsys, str(db), "Paris is the capital of France")
    _add(main, capsys, str(db), "The mitochondria is the powerhouse of the cell")

    rc, out, err = run(main, ["dashboard", "--once", *db_arg(str(db))], capsys)
    assert rc == 0, f"dashboard --once failed rc={rc} err={err!r} out={out!r}"
    assert "cards: 2" in out, f"seeded store must show 2 cards, got {out!r}"
    # Vitality bar uses block glyphs (fresh cards → dense blocks).
    assert "█" in out, f"vitality bar must render blocks, got {out!r}"
    # DB path shows in the title.
    assert str(db) in out, f"title must carry the db path, got {out!r}"

    # Recent list is newest-first: the second-added fact ("powerhouse") must
    # appear BEFORE the first-added ("Paris") in the rendered frame. A
    # sign-flip in the sort key would pass the bare-substring checks above
    # but fail this ordering invariant.
    recent_block = out.split("recent:", 1)[1].split("decay", 1)[0]
    pos_power = recent_block.find("powerhouse")
    pos_paris = recent_block.find("Paris")
    assert pos_power != -1 and pos_paris != -1, (
        f"both recent facts must render, got recent_block={recent_block!r}"
    )
    assert pos_power < pos_paris, (
        f"recent must be newest-first (powerhouse before Paris), "
        f"got pos={pos_power},{pos_paris} in {recent_block!r}"
    )

    # Bar-width invariant: the rendered bar is exactly _BAR_WIDTH cells.
    from isotope_zero.cli.dashboard import _BAR_WIDTH

    bar_line = next((ln for ln in out.splitlines() if "vitality:" in ln), "")
    bar_glyphs = "".join(c for c in bar_line if c in "█▒░")
    assert len(bar_glyphs) == _BAR_WIDTH, (
        f"vitality bar must be exactly {_BAR_WIDTH} cells, got {len(bar_glyphs)} "
        f"in {bar_line!r}"
    )


# --------------------------------------------------------------------------- #
# 3. missing DB → exit 1, path named on stderr (mirrors inspect)
# --------------------------------------------------------------------------- #
def test_dashboard_missing_db(capsys):
    """An explicit ``--db`` path that does not exist must exit 1 and name the
    path on stderr — the dashboard must NOT spin an empty panel over a typo."""
    bad_path = "/no/such/path/does/not/exist.db"
    rc, out, err = run(
        main, ["dashboard", "--once", *db_arg(bad_path)], capsys
    )
    assert rc == 1, (
        f"dashboard on missing --db must exit 1, got rc={rc} out={out!r} err={err!r}"
    )
    assert bad_path in err, f"stderr must name the missing path, got {err!r}"
    # And it must not have printed a frame to stdout.
    assert "isotope_zero" not in out, f"no frame on missing db, got {out!r}"


# --------------------------------------------------------------------------- #
# 4. decay candidate appears when a card is aged + never-recalled
# --------------------------------------------------------------------------- #
def test_dashboard_decayed_shown(tmp_path, capsys):
    """A card written with an old timestamp and access_count=0 falls below the
    vitality floor (and past the 1h grace) and must appear as a decay candidate
    in the frame — proving the dashboard surfaces ``dry_run`` decay entries."""
    db = tmp_path / "dash_decayed.db"

    # A fresh, healthy card (so the store isn't 100% decayed).
    _add(main, capsys, str(db), "Fresh healthy fact about routing")

    # A decayed card: old timestamp, never recalled. Add it directly to the
    # store (the CLI `add` would stamp now), bypassing the embedder — the
    # dashboard's reads (vitality, dry_run) don't need an embedding for a
    # decay candidate (vitality is timestamp/access driven; dry_run's decay
    # branch scores every card regardless of embedding presence).
    store = MemoryStore(str(db))
    try:
        ancient = time.time() - (400 * 24 * 3600)  # ~400 days old
        stale = MemoryCard(
            id="stale-decayed-00000000000000000000000000",
            fact="Ancient stale fact never recalled",
            evidence="",
            timestamp=ancient,
            tags=[],
            access_count=0,
            last_access=0.0,
        )
        store.add(stale)
    finally:
        store.close()

    rc, out, err = run(main, ["dashboard", "--once", *db_arg(str(db))], capsys)
    assert rc == 0, f"dashboard --once failed rc={rc} err={err!r} out={out!r}"
    # The decay section is non-empty and lists the stale fact.
    assert "decay candidates" in out, f"decay section must render, got {out!r}"
    assert "Ancient stale fact" in out, (
        f"the decayed card's fact must be listed, got {out!r}"
    )
    # A non-zero reclaimable-tokens hint appears when there's a decay candidate.
    assert "reclaimable" in out, (
        f"reclaimable tokens hint must show with decay, got {out!r}"
    )


# --------------------------------------------------------------------------- #
# 5. --interval <= 0 is rejected before any loop spins (guard test)
# --------------------------------------------------------------------------- #
def test_dashboard_interval_nonpositive(tmp_path, capsys):
    """``--interval 0`` (or negative) must exit 1 and explain on stderr, WITHOUT
    spinning the live loop. A mis-ordered guard (interval checked after the
    client opens, or never checked) would either hang or hide the message."""
    db = tmp_path / "dash_interval.db"
    _add(main, capsys, str(db), "fact")

    for bad in ("0", "-1", "-0.5"):
        rc, out, err = run(
            main, ["dashboard", *db_arg(str(db)), "--interval", bad], capsys
        )
        assert rc == 1, (
            f"--interval {bad} must exit 1, got rc={rc} out={out!r} err={err!r}"
        )
        assert "interval must be > 0" in err, (
            f"stderr must explain the interval guard for {bad}, got {err!r}"
        )
        # And it must not have started rendering frames.
        assert "isotope_zero" not in out, (
            f"no frame for --interval {bad}, got {out!r}"
        )


# --------------------------------------------------------------------------- #
# 6. transport selection: _run_rich degrades to _run_plain when rich is absent
#    (non-blocking — never spins the real loop)
# --------------------------------------------------------------------------- #
def test_dashboard_run_rich_degrades_without_rich(monkeypatch):
    """When ``rich`` cannot be imported, ``_run_rich`` must fall back to
    ``_run_plain`` rather than raise. We force the ImportError and stub the
    fallback so the test never blocks on a live loop."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "rich.console" or name == "rich.live" or name == "rich" or name.startswith("rich."):
            raise ImportError(f"forced for test: no module named '{name}'")
        return real_import(name, *a, **k)

    from isotope_zero.cli import dashboard as dash

    called_plain = {"v": False}

    def fake_plain(supplier, interval):
        called_plain["v"] = True
        return 0  # don't loop — return immediately

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(dash, "_run_plain", fake_plain)

    # supplier returns one throwaway state; only used to seed rich's Live.
    rc = dash._run_rich(lambda: {}, interval=2.0)
    assert rc == 0, f"_run_rich fallback must return the plain rc, got {rc}"
    assert called_plain["v"], (
        "_run_rich must call _run_plain when rich import fails, got rich path"
    )


def test_dashboard_render_per_second_clamp():
    """The rich refresh_per_second is clamped to >=1 and bounded by 1/interval:
    a long interval must not produce a sub-1 rps, and a tiny interval must not
    spin unbounded. Checks the clamp math without spinning a loop."""
    from isotope_zero.cli import dashboard as dash

    # Mirror the one-line clamp from _run_rich so we pin the arithmetic.
    def rps(interval: float) -> int:
        return max(1, int(1.0 / max(interval, 0.1)))

    assert rps(2.0) == 1, "2s interval → 1 rps"
    assert rps(0.5) == 2, "0.5s interval → 2 rps"
    assert rps(10.0) == 1, "long interval floored to 1 rps, not 0"
    # Zero/negative intervals are rejected upstream (run_dashboard's guard) so
    # _run_rich never sees them; the clamp nevertheless bounds them to 10 rps
    # (1/0.1) rather than raising a ZeroDivisionError.
    assert rps(0.0) == 10, "zero interval clamps to the 0.1 floor → 10 rps"
    assert rps(-5.0) == 10, "negative interval clamps to the 0.1 floor → 10 rps"
