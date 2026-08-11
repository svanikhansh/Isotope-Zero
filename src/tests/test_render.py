"""Tests for the clean (Mem0-style) CLI output path — ``isotope_zero.cli.render``.

The render module owns the *human-readable* branch of every ``_cmd_*``; ``--json`` is
the machine contract and is covered by ``test_cli.py``. These tests assert the output
contract directly against the pure formatters (no I/O, no DB, no ONNX):

- **fact is the hero**: recall/search/list/active print the fact text first, with a
  rank + a short id suffix; no uuid column, no score in the default view.
- **verbose restores technical columns**: ``verbose=True`` brings back the aligned
  id/fact/score/age (or id/fact/tags/age, or vitality) tables — a faithful restore of
  the pre-redesign behavior for power users.
- **action confirmations**: add → ``✓ remembered`` + a trailing ``id: <uuid>`` line
  (parseable by ``test_cli._id_from_plain``); forget → ``✓ forgot``; touch →
  ``✓ refreshed``.
- **empty states**: friendly prose (``No memories found.`` / ``No memories yet.`` /
  ``No tags yet.``), not ``(no cards)``.
- **stats prose summary** + verbose histogram; **tags** as a wrapped ``tag (n)`` line.
- **encoding fallback**: ``_glyph`` degrades Unicode marks to ASCII when stdout can't
  encode them (Windows cp1252 / ``LC_ALL=C`` pipes), so the clean output never raises.
"""
from __future__ import annotations

import io

from isotope_zero.cli.render import legacy as render


# --------------------------------------------------------------------------- #
# Shared fixtures — hit/row/card shapes match what debug.py passes in.
# --------------------------------------------------------------------------- #
_NOW = 1_000_000.0
_HITS = [
    {
        "id": "a5d44054522c44f69a833233dabc46eb",
        "fact": "The user prefers dark mode for UI.",
        "score": 0.8123,
        "timestamp": _NOW - 3600,  # ~0.04 days old
    },
    {
        "id": "d5bcb1474bad4b2a954a3d58ab3e9592",
        "fact": "Python is the user's favorite language.",
        "score": 0.7100,
        "timestamp": _NOW - 86400,  # 1 day old
    },
]
_ROWS = [
    {
        "id": "fabf751320e64bc2955b35ec5174b1f7",
        "fact": "a third fact for verbose testing",
        "tags": [],
        "timestamp": _NOW - 3600,
        "age_days": 0.0,
    },
    {
        "id": "d5bcb1474bad4b2a954a3d58ab3e9592",
        "fact": "user prefers clean Mem0-style CLI output",
        "tags": ["ui", "preference"],
        "timestamp": _NOW - 86400,
        "age_days": 1.0,
    },
]


# --------------------------------------------------------------------------- #
# 1. format_hits — fact-first, verbose restores columns
# --------------------------------------------------------------------------- #
def test_format_hits_fact_first_no_score_default():
    out = render.format_hits(_HITS, _NOW)
    # The fact text is the hero of each line.
    assert "The user prefers dark mode for UI." in out
    assert "Python is the user's favorite language." in out
    # Ranked (1., 2.).
    assert "1." in out and "2." in out
    # No raw full uuid, no score column, no === banner in the default view.
    assert "a5d44054522c44f69a833233dabc46eb" not in out  # full id suppressed
    assert "0.8123" not in out  # score suppressed
    assert "===" not in out
    # A short id suffix IS shown (for copy-back into `get`).
    assert "a5d440" in out


def test_format_hits_verbose_restores_technical_columns():
    out = render.format_hits(_HITS, _NOW, verbose=True)
    # Verbose brings back the aligned header + score + age + full-ish id.
    assert "score" in out and "age_d" in out and "#" in out
    assert "0.8123" in out  # score restored
    assert "a5d44054522c" in out  # id column restored (truncated to 12)


def test_format_hits_empty_is_friendly_prose():
    assert render.format_hits([], _NOW) == "No memories found."


# --------------------------------------------------------------------------- #
# 2. format_list_rows — tags inline, verbose restores columns
# --------------------------------------------------------------------------- #
def test_format_list_rows_tags_inline_when_present():
    out = render.format_list_rows(_ROWS, _NOW)
    assert "user prefers clean Mem0-style CLI output" in out
    # Tags shown inline only for the row that has them.
    assert "ui" in out and "preference" in out
    assert "===" not in out


def test_format_list_rows_verbose_restores_columns():
    out = render.format_list_rows(_ROWS, _NOW, verbose=True)
    assert "tags" in out and "age_d" in out and "id" in out


def test_format_list_rows_empty_is_friendly_prose():
    # Truly empty (no filter applied) → "yet", implying a fresh/empty store.
    assert render.format_list_rows([], _NOW) == "No memories yet."


def test_format_list_rows_filtered_empty_says_found_not_yet():
    # A tag filter was applied but matched nothing → "found", not "yet" (the
    # store isn't empty; the filter just narrowed to zero hits).
    assert render.format_list_rows([], _NOW, filtered=True) == "No memories found."


# --------------------------------------------------------------------------- #
# 3. format_card — fact headline + tags + id; verbose appends technical fields
# --------------------------------------------------------------------------- #
def test_format_card_fact_headline():
    out = render.format_card(
        "fabf751320e64bc2955b35ec5174b1f7",
        "a third fact for verbose testing",
        "the supporting quote",
        [],
        _NOW - 3600, _NOW - 3600, 0, 0.7, 0, _NOW,
    )
    # Fact is the first line (headline); evidence is rendered (content, not chrome).
    assert out.splitlines()[0] == "a third fact for verbose testing"
    assert "evidence: the supporting quote" in out
    assert "tags:     (none)" in out
    assert "id:       fabf751320e64bc2955b35ec5174b1f7" in out
    # No technical fields by default.
    assert "vitality" not in out
    assert "source_tokens" not in out


def test_format_card_evidence_none_when_empty():
    out = render.format_card("x", "a fact", "", [], 0, 0, 0, 0.5, 0, 1)
    assert "evidence: (none)" in out


def test_format_card_verbose_appends_technical_fields():
    out = render.format_card(
        "fabf751320e64bc2955b35ec5174b1f7",
        "a third fact for verbose testing",
        "",
        [],
        _NOW - 3600, _NOW - 3600, 0, 0.7, 0, _NOW,
        verbose=True,
    )
    assert "vitality:" in out
    assert "access_count:" in out
    assert "source_tokens:" in out
    assert "age_days:" in out


# --------------------------------------------------------------------------- #
# 4. Action confirmations — add emits ✓ + a parseable id: line
# --------------------------------------------------------------------------- #
def test_format_add_remembered_plus_parseable_id_line():
    out = render.format_add("a5d44054522c44f69a833233dabc46eb", created=True)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    # Hero line is the friendly confirmation, NOT a bare uuid.
    assert "remembered" in lines[0]
    # Trailing `id: <uuid>` line — parseable by test_cli._id_from_plain (last line,
    # last whitespace-separated token).
    assert lines[-1].startswith("id:")
    assert lines[-1].split()[-1] == "a5d44054522c44f69a833233dabc46eb"


def test_format_add_updated_when_not_created():
    out = render.format_add("a5d440...", created=False)
    assert "updated" in out


def test_format_forget_and_touch_are_friendly():
    assert "forgot" in render.format_forget("anyid")
    assert "refreshed" in render.format_touch("anyid")


# --------------------------------------------------------------------------- #
# 5. Aggregates — tags / stats
# --------------------------------------------------------------------------- #
def test_format_tags_wrapped_line_desc_by_count():
    out = render.format_tags({"release": 3, "ui": 2, "preference": 1})
    # All tags + counts present, joined by the bullet.
    assert "release (3)" in out
    assert "ui (2)" in out
    assert "preference (1)" in out


def test_format_tags_verbose_restores_aligned_table():
    out = render.format_tags({"release": 3, "ui": 2, "preference": 1}, verbose=True)
    # Header row + aligned tag/count columns, count-desc (insertion order is
    # count-desc because _cmd_tags sorts before passing the dict in).
    assert "tag" in out and "count" in out
    assert "release" in out and "ui" in out and "preference" in out
    assert "3" in out and "2" in out and "1" in out
    # Verbose is a multi-line table (one row per tag), not the single wrapped line.
    assert out.count("\n") >= 3


def test_format_tags_empty():
    assert render.format_tags({}) == "No tags yet."


def test_format_stats_prose_summary_default():
    out = render.format_stats(
        12, 4300, "FALLBACK", 280, {"ui": 3}, {"fresh": 10, "aging": 1, "decayed": 1},
    )
    # One prose line; no histogram/table by default.
    assert "12 memories" in out
    assert "fallback embeddings" in out  # mode lowercased
    assert "~280 tokens" in out
    assert "histogram" not in out
    assert "tag distribution" not in out


def test_format_stats_verbose_appends_histogram_and_tags():
    out = render.format_stats(
        12, 4300, "FALLBACK", 280, {"ui": 3}, {"fresh": 10, "aging": 1, "decayed": 1},
        verbose=True,
    )
    assert "vitality histogram" in out
    assert "tag distribution" in out
    assert "fresh" in out and "decayed" in out


def test_format_stats_singular_noun_for_one():
    out = render.format_stats(1, 100, "FALLBACK", 5, {}, {"fresh": 1, "aging": 0, "decayed": 0})
    assert "1 memory" in out  # singular
    assert "1 memories" not in out


# --------------------------------------------------------------------------- #
# 6. _glyph encoding fallback — Unicode degrades to ASCII on non-UTF-8 stdout
# --------------------------------------------------------------------------- #
def test_glyph_falls_back_to_ascii_when_stdout_cannot_encode_unicode(monkeypatch):
    # Simulate a Windows cp1252 / LC_ALL=C pipe: stdout claims an encoding that
    # can't represent ✓. _glyph must return the ASCII fallback, not raise.
    fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding="ascii", write_through=True)
    monkeypatch.setattr(render.sys, "stdout", fake_stdout)
    assert render._glyph("ok") == "ok"  # not "✓"
    assert render._glyph("bullet") == "-"
    assert render._glyph("arrow") == "->"


def test_glyph_unicode_when_stdout_is_utf8(monkeypatch):
    fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", write_through=True)
    monkeypatch.setattr(render.sys, "stdout", fake_stdout)
    assert render._glyph("ok") == "✓"


# --------------------------------------------------------------------------- #
# 7. _wrap_fact truncates long facts without blowing out one-line-per-memory
# --------------------------------------------------------------------------- #
def test_wrap_fact_truncates_long_facts():
    long = "x" * 200
    out = render._wrap_fact(long, width=70)
    assert len(out) <= 70
    assert out.endswith("…")


def test_wrap_fact_collapses_newlines():
    out = render._wrap_fact("line one\nline two", width=70)
    assert "\n" not in out
    assert "line one line two" == out
