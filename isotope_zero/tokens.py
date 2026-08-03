"""Token estimation — the single source of truth for token math.

Cost is isotope_zero's first-class metric, so every module must use the same
estimator. We avoid a real tokenizer dependency on the hot path: a tight
regex splitter approximates GPT-style BPE counts within ~5% for English and
keeps latency sub-millisecond. The eval harness cross-checks this against a
real `tokenizers` model when present.
"""
from __future__ import annotations

import re

# Matches the rough word-piece boundaries GPT tokenizers split on.
_TOKEN_RE = re.compile(r"\S+|\s+")


def estimate_tokens(text: str) -> int:
    """Estimate the token count of `text`.

    Empirically ~0.75 words/token for English; we approximate by counting
    whitespace/non-whitespace runs and scaling. Cheaper and offline than a
    real tokenizer, which matters because this runs on every write & read.
    """
    if not text:
        return 0
    runs = _TOKEN_RE.findall(text)
    # ~4 chars per token is the standard BPE rule of thumb for English.
    char_estimate = max(1, len(text) // 4)
    # Blend: average of run-count and char-based estimate is more stable
    # across short and long inputs than either alone.
    return max(1, (len(runs) + char_estimate) // 2)
