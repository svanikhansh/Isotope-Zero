//! Bit-for-bit port of `isotope_zero/core/consolidation.py::_are_negations`.
//!
//! The C-suite correctness budget allows 0 incorrect merges from this
//! heuristic, so the Rust version mirrors the Python logic exactly:
//! lowercased, whitespace-padded marker removal (longest markers first,
//! stable tie order matching the Python tuple), whitespace collapse, a
//! crude `ing/ed/es/s` stemmer, and a >= 0.6 Jaccard gate.

use std::collections::HashSet;

/// Each negation marker from `consolidation.py::_NEGATION_MARKERS`, wrapped in
/// the single spaces Python pads with (`" " + marker + " "`) and precomputed at
/// compile time so the hot path never allocates a `format!`'d needle per marker
/// per call. Order mirrors Python's `sorted(_NEGATION_MARKERS, key=len,
/// reverse=True)` (length descending, stable within equal length, preserving
/// the original tuple order) — longest markers first, so a longer marker always
/// wins over a marker that is a substring of it (e.g. "does not" before "not").
const NEGATION_NEEDLES_SORTED: &[&str] = &[
    " no longer ", " does not ", " will not ", " doesn't ", " was not ",
    " neither ", " without ", " stopped ", " no more ", " do not ",
    " is not ", " wasn't ", " cannot ", " don't ", " never ", " isn't ",
    " won't ", " can't ", " lacks ", " quit ", " not ", " nor ",
];

/// Mirror of `_strip_negations`: returns the denegated text and whether any
/// negation marker was found. Uses the precomputed padded needles so the hot
/// path does no per-call allocation beyond the single lowercase buffer.
fn strip_negations(text: &str) -> (String, bool) {
    // `" " + text.lower().strip() + " "`
    let mut t = format!(" {} ", text.trim().to_lowercase());
    let mut found = false;
    for needle in NEGATION_NEEDLES_SORTED {
        if t.contains(needle) {
            found = true;
            // Replace all occurrences of this padded marker with a single
            // space — exactly Python's `t.replace(needle, " ")`.
            t = t.replace(needle, " ");
        }
    }
    // `" ".join(t.split())` — collapse any whitespace runs.
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    (t, found)
}

/// Mirror of `_stem`: strip one trailing `ing`/`ed`/`es`/`s` when
/// `len(tok) - len(suf) >= 3` (Python `len` counts code points, so we count
/// chars; the suffixes are ASCII so byte-slicing the same amount is exact).
fn stem(tok: &str) -> &str {
    let tok_chars = tok.chars().count();
    for suf in ["ing", "ed", "es", "s"] {
        if tok.ends_with(suf) && tok_chars >= suf.chars().count() + 3 {
            // suf is ASCII, so its byte length == its char count; slicing the
            // same number of bytes off the end is exact.
            return &tok[..tok.len() - suf.len()];
        }
    }
    tok
}

/// Mirror of `_are_negations`. Exact equivalence to the Python float Jaccard
/// `>= 0.6` via integer math: `inter/union >= 3/5` <=> `5*inter >= 3*union`,
/// which is bit-for-bit identical for the small token-set sizes in practice.
pub fn are_negations(a: &str, b: &str) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    let (ta, neg_a) = strip_negations(a);
    let (tb, neg_b) = strip_negations(b);
    // Need a polarity difference: exactly one side is negated.
    if neg_a == neg_b {
        return false;
    }
    let sa: HashSet<&str> = ta.split_whitespace().map(stem).collect();
    let sb: HashSet<&str> = tb.split_whitespace().map(stem).collect();
    if sa.is_empty() || sb.is_empty() {
        return false;
    }
    let inter = sa.intersection(&sb).count();
    let union = sa.union(&sb).count();
    inter * 5 >= union * 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_removal_order() {
        // Longer marker must win over a substring of it. Note: strip_negations
        // lowercases the whole input (mirrors Python's `text.lower().strip()`),
        // so "Mac" -> "mac".
        let (t, found) = strip_negations("the user does not use a Mac");
        assert!(found);
        assert_eq!(t, "the user use a mac");

        // "not" must not fire inside "notebook" (it is not space-padded there).
        let (t2, found2) = strip_negations("the service uses notebook");
        assert!(!found2);
        assert_eq!(t2, "the service uses notebook");
    }

    #[test]
    fn stem_examples() {
        // Mirrors consolidation.py::_stem exactly: first matching suffix in
        // ("ing","ed","es","s") whose len(tok)-len(suf) >= 3 wins.
        // "uses" (len 4): "es" guard 4-2=2 <3 fails -> "s" 4-1=3 >=3 -> "use".
        assert_eq!(stem("uses"), "use");
        assert_eq!(stem("use"), "use"); // len 3: no suffix guard passes
        // "using" (len 5): "ing" 5-3=2 <3 fails -> "ed"/"es"/"s" all fail -> unchanged
        assert_eq!(stem("using"), "using");
        assert_eq!(stem("go"), "go"); // len 2 - 1 = 1 < 3
        assert_eq!(stem("us"), "us"); // len 2 - 1 = 1 < 3
        // Extra parity anchors vs the Python reference:
        assert_eq!(stem("likes"), "lik"); // 5-2=3 >=3 -> strip "es"
        assert_eq!(stem("passes"), "pass"); // 6-2=4 >=3 -> strip "es"
        assert_eq!(stem("running"), "runn"); // 7-3=4 >=3 -> strip "ing"
        assert_eq!(stem("stopped"), "stopp"); // 7-2=5 >=3 -> strip "ed"
        assert_eq!(stem("buses"), "bus"); // 5-2=3 >=3 -> strip "es"
    }

    #[test]
    fn basic_pairs() {
        assert!(are_negations(
            "the service uses a Mac",
            "the service does not use a Mac"
        ));
        assert!(!are_negations(
            "the service uses a Mac",
            "the service uses a Mac"
        ));
        assert!(!are_negations(
            "the service uses a Mac",
            "the user prefers linux"
        ));
    }
}
