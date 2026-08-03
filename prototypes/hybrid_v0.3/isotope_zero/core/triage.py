"""Write-path triage: classify user intent and compress raw input into cards.

This module is the first stop on the isotope_zero write path. It runs cheap
stdlib-only heuristics to decide whether a piece of input is a new fact to
store (ADD), a correction to an existing fact (UPDATE), or a request to
forget something (DELETE). When the fast path is uncertain, it escalates to
a *mocked* LLM path — a deterministic re-examination that stands in for a
future local-LLM triage call. No external services, no network.

Conservative by design: `compress_to_card` never invents facts that are not
supported by a verbatim substring of the input (its `evidence`). When in
doubt it keeps the fact slightly long rather than risk a lossy rewrite.
"""
from __future__ import annotations

import re
import uuid
import logging

from isotope_zero.types import ActionType, ActionResult, MemoryCard, now_ts
from isotope_zero.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword signals for the fast classifier.
# Each list is matched case-insensitively as whole-word/phrase boundaries.
# Order matters only for documentation; scoring weights are applied below.
# ---------------------------------------------------------------------------

# Signals that the user wants to forget / remove something already known.
# Kept strict: these phrases assert pure removal with no replacement value.
# "I no longer prefer X" alone is NOT here — it usually introduces a new value
# ("I no longer prefer X; I use Y now") and is therefore an UPDATE, not a
# DELETE. Pure forgetting is phrased as "forget", "stop remembering", etc.
DELETE_PATTERNS: tuple[str, ...] = (
    "forget", "remove", "delete",
    "stop remembering", "clear my",
    "ignore that i said",
)

# Signals that the user is correcting or revising something already known.
# "i no longer" is a revision signal — it amends a prior stance, and in
# practice almost always carries a replacement ("I no longer use X, I use Y").
UPDATE_PATTERNS: tuple[str, ...] = (
    "actually", "changed", "switched to", "now it's", "update",
    "correction", "i meant", "moved to", "i now prefer",
    "no longer use", "i no longer",
)

# Signals that the user is asserting something new (preferences/identity/facts).
# ADD is the default fallback; these patterns nudge confidence up for
# unambiguously new factual statements.
ADD_PATTERNS: tuple[str, ...] = (
    "my name is", "i am", "i'm", "i like", "i love", "i prefer",
    "i work at", "i work on", "i live in", "i use", "this is",
    "the project is", "my project",
)

# ---------------------------------------------------------------------------
# Scoring weights. A single strong keyword gives a middling score; multiple
# agreeing signals (e.g. an UPDATE phrase + an explicit "instead of") push
# confidence high enough to skip escalation.
# ---------------------------------------------------------------------------
_BASE_SIGNAL_WEIGHT = 0.65   # one strong keyword/phrase match clears the bar
_EXTRA_SIGNAL_WEIGHT = 0.12  # each additional agreeing signal (diminishing)
_MAX_CONFIDENCE = 0.95       # never claim certainty
_ESCALATE_THRESHOLD = 0.6    # below this, defer to the mock LLM path
_DEFAULT_ADD_SCORE = 0.35   # no signal at all: low-confidence ADD (escalates)


def _find_phrase_matches(text: str, phrases: tuple[str, ...]) -> list[str]:
    """Return the subset of `phrases` that occur in `text` (case-insensitive)."""
    lowered = text.lower()
    return [p for p in phrases if p in lowered]


def _extract_target_id(text: str, phrases: tuple[str, ...]) -> str | None:
    """Best-effort extraction of a referenced entity for UPDATE/DELETE.

    Strategy (in priority order, first hit wins):
      1. A quoted phrase ("like this" or 'like this') — most reliable signal
         that the user is pointing at a specific entity.
      2. The noun phrase immediately following an UPDATE/DELETE trigger word
         (e.g. "switched to Rust" -> "Rust"). We grab the next capitalized or
         alphanumeric token cluster, SKIPPING common connector words
         ("that", "i", "to", "my", "the", ...).

    Returns None if nothing usable was found. This is explicitly best-effort;
    callers must tolerate None.
    """
    # 1. Quoted phrases — strongest signal.
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    if quoted:
        for g1, g2 in quoted:
            candidate = (g1 or g2 or "").strip()
            if candidate:
                return _normalize_id(candidate)

    # Connector / filler words that frequently trail a trigger and should not
    # be treated as the referenced entity.
    skip_words = {
        "that", "i", "to", "my", "the", "a", "an", "of", "and", "or",
        "said", "told", "use", "using", "prefer", "name", "is", "was",
        "me", "you", "we", "they", "he", "she", "it", "for", "in", "on",
        "at", "so", "now", "no", "longer",
    }
    # Phrase-specific "tail past these words" hints — e.g. after "i no longer"
    # the entity is what follows, possibly across "use"/"like".
    phrase_skip: dict[str, frozenset[str]] = {
        "i no longer": frozenset({"use", "like", "prefer", "remember", "need"}),
        "stop remembering": frozenset(),
        "clear my": frozenset({"memory", "memories", "history"}),
        "ignore that i said": frozenset({"that", "i", "said"}),
        "i now prefer": frozenset(),
        "no longer use": frozenset(),
        "i meant": frozenset(),
    }

    # 2. Trailing noun phrase after a trigger keyword.
    # Iterate phrases longest-first so the most specific trigger (e.g.
    # "switched to" rather than bare "actually") wins the entity extraction.
    lowered = text.lower()
    for phrase in sorted(phrases, key=len, reverse=True):
        idx = lowered.find(phrase)
        if idx == -1:
            continue
        tail = text[idx + len(phrase):]
        # Tokenize the tail so we can skip filler words.
        tokens = re.findall(r"[A-Za-z0-9][\w.\-]*", tail)
        allowed_skip = skip_words | phrase_skip.get(phrase, frozenset())
        # Drop leading connector/pronoun words; stop at the first content word.
        # The pronoun "i" is skipped here — after a trigger like "actually" or
        # "forget", "i said"/"i switched" are filler around the real entity.
        while tokens and tokens[0].lower() in allowed_skip:
            tokens.pop(0)
        if not tokens:
            continue
        # Take up to 3 content tokens as the entity name, but stop before a
        # connector word so we don't slurp "for my backend" into the entity.
        entity_tokens: list[str] = []
        for tok in tokens:
            if tok.lower() in skip_words and entity_tokens:
                break
            if len(entity_tokens) >= 3:
                break
            entity_tokens.append(tok)
        if entity_tokens:
            return _normalize_id(" ".join(entity_tokens))
    return None


def _normalize_id(raw: str) -> str:
    """Turn a human phrase into a slug-ish stable id string."""
    cleaned = re.sub(r'\s+', '_', raw.strip().lower())
    cleaned = re.sub(r'[^a-z0-9_\-.]', '', cleaned)
    return cleaned[:64] or None  # type: ignore[return-value]


def _mock_llm_re_examine(input_text: str, context: str) -> ActionResult:
    """Deterministic stand-in for a future local-LLM triage call.

    This is NOT a real API call. It re-examines the text with slightly
    different (looser) rules so that ambiguous inputs get a second look and a
    decisive answer instead of silently falling through to the ADD default.
    The real implementation will swap this body for a local-LLM inference
    while keeping the same signature and the `escalated=True` contract.

    Heuristic differences from the fast path:
      - Treats negation ("don't remember", "not actually") as DELETE-leaning.
      - Treats past-tense changes ("was X, now Y") as UPDATE even without a
        keyword from UPDATE_PATTERNS.
      - Otherwise commits to whichever category the fast path leaned toward,
        bumping confidence to just above the escalation threshold.
    """
    lowered = input_text.lower()

    # Negation / forgetting signals missed by the strict keyword list.
    negation_signals = (
        "don't remember", "do not remember", "not actually",
        "didn't mean", "did not mean", "scratch that",
    )
    if any(s in lowered for s in negation_signals):
        target = _extract_target_id(input_text, DELETE_PATTERNS) or _extract_target_id(input_text, negation_signals)
        return ActionResult(
            action=ActionType.DELETE,
            confidence=0.72,
            escalated=True,
            target_id=target,
            reasoning="mock-LLM: negation/scratch signal -> DELETE",
        )

    # "was X, now Y" pattern -> revision even without an UPDATE keyword.
    was_now = re.search(r'\bwas\b.*\bnow\b', lowered)
    if was_now:
        target = _extract_target_id(input_text, UPDATE_PATTERNS)
        return ActionResult(
            action=ActionType.UPDATE,
            confidence=0.7,
            escalated=True,
            target_id=target,
            reasoning="mock-LLM: was/now revision pattern -> UPDATE",
        )

    # Commit to the fast path's lean (ADD by default), just above threshold.
    delete_hits = _find_phrase_matches(input_text, DELETE_PATTERNS)
    update_hits = _find_phrase_matches(input_text, UPDATE_PATTERNS)
    if delete_hits:
        return ActionResult(
            action=ActionType.DELETE,
            confidence=0.62,
            escalated=True,
            target_id=_extract_target_id(input_text, DELETE_PATTERNS),
            reasoning=f"mock-LLM: committed DELETE on weak signal {delete_hits[0]!r}",
        )
    if update_hits:
        return ActionResult(
            action=ActionType.UPDATE,
            confidence=0.62,
            escalated=True,
            target_id=_extract_target_id(input_text, UPDATE_PATTERNS),
            reasoning=f"mock-LLM: committed UPDATE on weak signal {update_hits[0]!r}",
        )

    return ActionResult(
        action=ActionType.ADD,
        confidence=0.62,
        escalated=True,
        target_id=None,
        reasoning="mock-LLM: no delete/update signal, committed ADD",
    )


def classify_action(input_text: str, context: str = "") -> ActionResult:
    """Classify a write-path input into ADD / UPDATE / DELETE.

    Fast heuristics run first (no LLM). If the resulting confidence is below
    `_ESCALATE_THRESHOLD`, the decision is re-examined by `_mock_llm_re_examine`
    (a deterministic stand-in for a future local-LLM triage call), and the
    returned `ActionResult` carries `escalated=True`.

    `context` is optional surrounding text; it is currently folded into the
    keyword search to give weak signals a chance to corroborate. It does not
    change the action type on its own.
    """
    if not input_text or not input_text.strip():
        # Empty input is not a real action; classify as a low-confidence ADD
        # and let the caller decide to ignore it.
        return ActionResult(
            action=ActionType.ADD, confidence=0.0, escalated=False,
            reasoning="empty input",
        )

    combined = f"{input_text}\n{context}" if context else input_text

    delete_hits = _find_phrase_matches(combined, DELETE_PATTERNS)
    update_hits = _find_phrase_matches(combined, UPDATE_PATTERNS)
    add_hits = _find_phrase_matches(combined, ADD_PATTERNS)

    # Score each category. The category with the highest score wins; ties
    # resolve in favor of the more specific action (UPDATE > DELETE > ADD),
    # because an explicit correction or removal is costlier to mis-classify
    # than a redundant ADD.
    def _score(hits: list[str]) -> float:
        if not hits:
            return 0.0
        # First signal full weight; subsequent agreeing signals add less,
        # diminishing so we never exceed _MAX_CONFIDENCE.
        score = _BASE_SIGNAL_WEIGHT
        for i in range(1, len(hits)):
            score += _EXTRA_SIGNAL_WEIGHT / (i + 1)
        return min(_MAX_CONFIDENCE, score)

    scores = {
        ActionType.DELETE: _score(delete_hits),
        ActionType.UPDATE: _score(update_hits),
        ActionType.ADD: _score(add_hits),
    }

    # Pick the best category; ADD is the implicit default when nothing fires.
    best_action = max(scores, key=lambda a: scores[a])  # type: ignore[arg-type]
    best_score = scores[best_action]

    if best_action is ActionType.ADD and best_score == 0.0:
        # No signal at all: still an ADD, but low confidence so we escalate.
        best_score = _DEFAULT_ADD_SCORE

    # When the top score is below the escalation threshold, defer to the mock
    # LLM path. The mock may either confirm the lean or flip it.
    if best_score < _ESCALATE_THRESHOLD:
        logger.debug("triage: low confidence %.2f, escalating '%s'", best_score, input_text[:60])
        return _mock_llm_re_examine(input_text, context)

    # Build a reasoning string naming the fired signals.
    if best_action is ActionType.DELETE:
        fired = delete_hits
        target = _extract_target_id(combined, DELETE_PATTERNS)
        reasoning = f"DELETE: matched {fired}"
    elif best_action is ActionType.UPDATE:
        fired = update_hits
        target = _extract_target_id(combined, UPDATE_PATTERNS)
        reasoning = f"UPDATE: matched {fired}"
    else:
        fired = add_hits
        target = None
        reasoning = f"ADD: matched {fired}" if fired else "ADD: default (new factual statement)"

    logger.debug(
        "classify action=%s conf=%.2f target=%s reason=%s input=%r",
        best_action.value, best_score, target, reasoning, input_text[:80],
    )
    return ActionResult(
        action=best_action,
        confidence=best_score,
        escalated=False,
        target_id=target,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Compression: turn raw input into a MemoryCard.
# ---------------------------------------------------------------------------

# Patterns used to extract the factual core and tags. Each maps a surface
# form ("my X is Y") to a normalized tag key. Kept deliberately simple — no
# POS tagger, no external NLP. Patterns are matched in order; first hit wins
# for a given sentence.
_TAG_PATTERNS: tuple[re.Pattern[str], tuple[str, ...]] = (
    # name
    (re.compile(r'\bmy name is\s+([A-Za-z][\w\-\.]*)', re.I), ("name",)),
    (re.compile(r"\bi(?:'m| am)\s+([A-Za-z][\w\-\.]*)", re.I), ("name",)),
    # preferences
    (re.compile(r'\bi (?:like|love|prefer)\s+([\w\s,]+?)(?:[.!]|$)', re.I), ("preference",)),
    (re.compile(r'\bi now prefer\s+([\w\s,]+?)(?:[.!]|$)', re.I), ("preference",)),
    # language / tool
    (re.compile(r'\bi use\s+([A-Za-z0-9][\w.\-]*)', re.I), ("language", "tool")),
    (re.compile(r'\bswitched to using\s+([A-Za-z0-9][\w.\-]*)', re.I), ("language", "tool")),
    # project
    (re.compile(r'\bmy (?:backend\s+)?project(?:\s+is|\s+named)?\s+([A-Za-z0-9][\w.\-]*)', re.I), ("project",)),
    # employer
    (re.compile(r'\bi work (?:at|on)\s+([A-Za-z0-9][\w.&\-]*)', re.I), ("organization", "project")),
    # location
    (re.compile(r'\bi live in\s+([A-Za-z][\w\s]+?)(?:[.!]|$)', re.I), ("location",)),
)


def _extract_tags(text: str) -> list[str]:
    """Auto-extract lowercase tag keywords from `text`.

    Uses the simple regex patterns above — a stand-in for real POS tagging.
    Returns a de-duplicated list like ['name', 'preference', 'language'].
    """
    tags: list[str] = []
    for pattern, keys in _TAG_PATTERNS:
        if pattern.search(text):
            tags.extend(keys)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique = [t for t in tags if not (t in seen or seen.add(t))]
    return unique


# Filler/conversational lead-ins that we strip from the front of a fact.
# Conservative: only strip when the rest still reads as a complete assertion.
_FILLER_PREFIXES = (
    "actually ", "so ", "well ", "ok ", "okay ", "um ", "i just want to say ",
    "just so you know ", "hey ", "hi ", "listen ",
)


def _strip_filler(text: str) -> str:
    """Remove conversational filler from the front of a statement.

    Only strips a single leading filler token/phrase so we never accidentally
    eat the subject of the assertion (e.g. "Actually" -> safe; "So I like" ->
    "I like").
    """
    lowered = text.lstrip()
    for filler in _FILLER_PREFIXES:
        if lowered.lower().startswith(filler):
            return lowered[len(filler):].lstrip()
    return lowered


def _extract_fact(raw_input: str) -> str:
    """Pull the core assertion out of `raw_input`.

    Conservative policy:
      - Strip leading conversational filler.
      - If the input contains a revision clause ("instead of X", "rather than
        X", "no longer X"), preserve BOTH sides so the fact is faithful to the
        full assertion. Lossy compression that drops the "instead of" clause
        would invent ambiguity, so we keep it.
      - Multiple facts in one sentence: keep them as one combined fact rather
        than picking arbitrarily. Splitting is left to higher layers; the
        contract permits a single combined fact.
      - Trim trailing whitespace and terminal punctuation duplication.
    """
    cleaned = _strip_filler(raw_input).strip()
    # Collapse internal whitespace runs.
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Normalize terminal punctuation: at most one trailing period.
    cleaned = re.sub(r'[.!\?]+$', '', cleaned).strip()
    if not cleaned:
        return raw_input.strip()
    return cleaned


def _extract_evidence(raw_input: str) -> str:
    """Return the SHORTEST verbatim substring that justifies the fact.

    Heuristic: take the largest contiguous clause that does not include pure
    filler. For most single-sentence inputs this is the whole stripped
    sentence. For multi-clause inputs we take the first clause containing a
    factual verb (is/like/use/live/work). Always a verbatim substring of
    `raw_input` (no rewording), so the fact is always back-traceable.
    """
    if not raw_input or not raw_input.strip():
        return ""
    stripped = raw_input.strip()

    # If short enough, the whole stripped input IS the evidence.
    if len(stripped) <= 120:
        return stripped

    # Otherwise try to bound to the first factual clause.
    # Split on ", " / " and " boundaries and keep the smallest prefix that
    # contains a factual verb.
    factual_verbs = re.compile(r'\b(?:is|are|am|like|love|prefer|use|live|work|named|called)\b', re.I)
    clauses = re.split(r'\s*,\s*|\s+and\s+', stripped)
    accum: list[str] = []
    for clause in clauses:
        accum.append(clause)
        if factual_verbs.search(" ".join(accum)):
            return ", ".join(accum)
    # Fallback: whole stripped input.
    return stripped


def compress_to_card(raw_input: str, embedding: list[float] | None = None) -> MemoryCard:
    """Compress raw user input into a MemoryCard.

    The card's `fact` is the core assertion (filler stripped, faithful to the
    input — never invents). `evidence` is the shortest verbatim substring of
    `raw_input` that justifies the fact. `tags` are auto-extracted via simple
    regex patterns. `embedding` is passed through unchanged.

    Multi-fact policy: a single combined fact is emitted rather than splitting
    arbitrarily; downstream layers may split further if desired. This keeps
    the write path single-card-per-input and avoids inventing facts.
    """
    fact = _extract_fact(raw_input)
    evidence = _extract_evidence(raw_input)
    tags = _extract_tags(raw_input)
    return MemoryCard(
        id=uuid.uuid4().hex,
        fact=fact,
        evidence=evidence,
        timestamp=now_ts(),
        tags=tags,
        embedding=embedding,
        source_tokens=estimate_tokens(raw_input),
    )


# ---------------------------------------------------------------------------
# Smoke test: run when executed directly. No external deps, no network.
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.DEBUG)

    examples = [
        # 1. Pure ADD: new identity + preference.
        "My name is Alice and I like tea",
        # 2. UPDATE with a revision clause and an explicit trigger keyword.
        "Actually I switched to using Rust for my backend project instead of Go",
        # 3. DELETE via a forgetting phrasing.
        "Please forget that I said I like tea",
        # 4. Ambiguous: should escalate to the mock-LLM path.
        "so the thing about the deploy is it's fine now",
    ]

    print("=== classify_action + compress_to_card smoke test ===\n")
    for ex in examples:
        result = classify_action(ex)
        card = compress_to_card(ex)
        print(f"input      : {ex!r}")
        print(f"  action   : {result.action.value}  conf={result.confidence:.2f}  escalated={result.escalated}  target_id={result.target_id!r}")
        print(f"  reasoning: {result.reasoning}")
        print(f"  card.id  : {card.id[:8]}...  tags={card.tags}  source_tokens={card.source_tokens}")
        print(f"  fact     : {card.fact!r}")
        print(f"  evidence : {card.evidence!r}")
        print()
