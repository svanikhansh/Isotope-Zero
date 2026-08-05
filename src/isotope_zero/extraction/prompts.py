"""LLM triage prompts — *reserved, not on the shipped path*.

The v1.0 write path (``fact_extractor.classify_action`` /
``compress_to_card``) is fully **deterministic**: stdlib regex keyword
signals + a mocked re-examination stand-in for a future local-LLM triage
call. No external services, no network. This module reserves the
``prompts`` namespace for the planned local-LLM escalation interface
(Ollama / llama.cpp) so the extraction package layout matches the documented
distribution tree. The shipped heuristics remain the default; an LLM path
would be an opt-in escalation tier on top of them.
"""
from __future__ import annotations
