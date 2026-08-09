#!/usr/bin/env python3
"""Negation correctness audit — systematic test of ``are_negations()``.

Run::

    .venv/bin/python scripts/verify_negation.py

Covers POSITIVE (negation pairs), NEGATIVE (unrelated pairs), and EDGE cases
(identical, short, unicode, empty). Reports precision / recall / verdict.
"""

from __future__ import annotations

import json
import sys
from typing import NamedTuple


class Case(NamedTuple):
    a: str
    b: str
    expected: bool  # True → are_negations should say "yes, these are negations"


# ---------------------------------------------------------------------------
# 1. POSITIVE cases — should return True
# ---------------------------------------------------------------------------
POSITIVE: list[Case] = [
    # Basic "not" insertion
    Case("I like cats", "I don't like cats", True),
    Case("the sky is blue", "the sky is not blue", True),
    Case("Rust is fast", "Rust is not fast", True),
    # Multi-word negation marker ("no longer") — post-v1.3.0 correction case
    Case("this project uses Rust", "this project no longer uses Rust", True),
    # "does not" / "doesn't"
    Case("the code compiles", "the code does not compile", True),
    Case("the server responds", "the server doesn't respond", True),
    # "do not" / "don't"
    Case("I understand this", "I don't understand this", True),
    Case("they agree", "they do not agree", True),
    # "never"
    Case("the API returns errors", "the API never returns errors", True),
    # "was not" / "wasn't"
    Case("the test was reliable", "the test was not reliable", True),
    Case("the file was found", "the file wasn't found", True),
    # "will not" / "won't"
    Case("the CI will pass", "the CI will not pass", True),
    Case("the build will succeed", "the build won't succeed", True),
    # "cannot" / "can't"
    Case("we can deploy", "we cannot deploy", True),
    Case("you can enter", "you can't enter", True),
    # "stopped"
    Case("the service runs", "the service stopped running", True),
    # "without"
    Case("the app launches with the flag", "the app launches without the flag", True),
]

# ---------------------------------------------------------------------------
# 2. NEGATIVE cases — should return False
# ---------------------------------------------------------------------------
NEGATIVE: list[Case] = [
    # Different subjects — not a negation, just different facts
    Case("I like cats", "I like dogs", False),
    Case("the sky is blue", "the sky is red", False),
    Case("Paris is in France", "London is in England", False),
    Case("SQLite is fast", "PostgreSQL is reliable", False),
    # Both negated, same polarity → not opposites
    Case("I don't like cats", "I don't like dogs", False),
    Case("the sky is not blue", "the sky is not green", False),
    # One negated but low Jaccard after stripping (different core fact)
    Case("I like cats", "I don't like rain", False),
    Case("Rust is fast", "Python is not fast", False),
    Case("the server is running", "the database is not responding", False),
    # Negation marker embedded in a different sentence entirely
    Case("the build passed", "I do not have time to review this", False),
    # "never" used non-negationally or in different context
    Case("I saw a comet", "I never visited Paris", False),
    # Negation of a completely different predicate
    Case("the API returns JSON", "the API does not have authentication", False),
]

# ---------------------------------------------------------------------------
# 3. EDGE cases
# ---------------------------------------------------------------------------
EDGE: list[Case] = [
    # --- Identical strings (should not be negations) ---
    Case("cats are mammals", "cats are mammals", False),
    Case("I don't like spinach", "I don't like spinach", False),
    # Both negated, identical after stripping → same polarity, not opposites
    Case("the sky is not blue", "the sky is not blue", False),

    # --- Short strings (< 3 words) ---
    Case("not", "yes", False),
    Case("yes", "no", False),
    Case("go", "stop", False),
    Case("it works", "it doesn't", False),
    # Single-word negation — after stripping, one side becomes empty set
    Case("cats", "no cats", False),

    # --- Unicode / emoji / non-ASCII ---
    Case("I ❤️ cats", "I don't ❤️ cats", True),
    Case("café is open", "café is not open", True),
    Case("naïve assumption", "not naive assumption", False),
    Case("résumé", "no résumé", False),

    # --- One empty string ---
    Case("I like cats", "", False),
    Case("", "I like cats", False),

    # --- Both empty strings ---
    Case("", "", False),

    # --- Case-insensitive negation markers ---
    Case("It IS blue", "It is NOT blue", True),
    Case("She WAS happy", "She WAS not happy", True),

    # --- Negation marker mid-sentence in a compound clause ---
    # "the sun rises in the east, not the west" vs "the sun rises in the east"
    # After stripping "not", "the sun rises in the east, the west" vs "the sun rises in the east"
    # Jaccard might be high enough → this is a real negation pair
    Case(
        "the sun rises in the east",
        "the sun rises in the east, not the west",
        True,
    ),
]


def run_all() -> tuple[list[Case], int, int, int, int]:
    """Execute every case; return (failed_cases, tp, tn, fp, fn)."""
    from isotope_zero.core.native import are_negations

    tp = tn = fp = fn = 0
    failed: list[Case] = []

    for case in POSITIVE + NEGATIVE + EDGE:
        result = are_negations(case.a, case.b)
        if case.expected:
            if result:
                tp += 1
            else:
                fn += 1
                failed.append(case)
        else:
            if not result:
                tn += 1
            else:
                fp += 1
                failed.append(case)

    return failed, tp, tn, fp, fn


def main() -> None:
    print("=" * 72)
    print("  isotope_zero — negation correctness audit")
    print("=" * 72)

    try:
        from isotope_zero.core.native import are_negations
    except ImportError as exc:
        print(f"FATAL: cannot import are_negations — {exc}")
        sys.exit(1)

    failed, tp, tn, fp, fn = run_all()
    total_pos = tp + fn
    total_neg = tn + fp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # ------------------------------------------------------------------
    # Dump failures for diagnosis
    # ------------------------------------------------------------------
    if failed:
        print(f"\n{len(failed)} FAILURE(S):\n")
        for c in failed:
            actual = are_negations(c.a, c.b)
            tag = "EXPECTED TRUE" if c.expected else "EXPECTED FALSE"
            print(f"  [{tag}]")
            print(f"    A: {c.a!r}")
            print(f"    B: {c.b!r}")
            print(f"    → are_negations returned {actual}")
            print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("-" * 72)
    print("SUMMARY")
    print("-" * 72)
    print(f"  POSITIVE cases (expected True):  {total_pos:3d}  (tp={tp}, fn={fn})")
    print(f"  NEGATIVE cases (expected False): {total_neg:3d}  (tn={tn}, fp={fp})")
    print(f"  Total cases tested:              {total_pos + total_neg:3d}")
    print()
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0):.4f}")
    print()

    if failed:
        print(f"  VERDICT: FAIL — {len(failed)} case(s) did not match expected labels")
        verdict = "FAIL"
    else:
        print("  VERDICT: PASS — all cases match expected labels")
        verdict = "PASS"

    print("=" * 72)

    # Machine-readable JSON line
    metrics = {
        "truePos": tp,
        "trueNeg": tn,
        "falsePos": fp,
        "falseNeg": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0, 4),
        "total": total_pos + total_neg,
        "verdict": verdict,
    }
    print("\n[METRICS]", json.dumps(metrics))
    print()


if __name__ == "__main__":
    main()
