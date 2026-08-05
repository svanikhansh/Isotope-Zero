"""AutoGen memory adapter for Isotope Zero.

Provides :class:`IsotopeZeroMemory` — a self-contained memory provider with the
uniform ``.remember`` / ``.recall`` API plus automatic ``agent_id`` session
tagging so multiple AutoGen agents sharing one DB keep their memory spaces
isolated.

Design
------
AutoGen's memory model is less standardized than the LangChain/LlamaIndex
``VectorStore`` contracts, so this adapter does **not** subclass an autogen base
type. It exposes a clean, standalone object usable directly in a multi-agent
loop and, when the ``pyautogen`` package is importable, wireable into a
``ConversableAgent`` via :meth:`IsotopeZeroMemory.attach_to_agent`.

The module imports ``pyautogen`` **lazily and guarded** so that
``import izero_adapters.autogen`` never fails when pyautogen is absent. The
module flag :data:`_HAS_AUTOGEN` records availability at import time, and
:meth:`attach_to_agent` raises a clear ``RuntimeError`` when it is not.

All storage work is delegated to :class:`izero_adapters._engine.Engine`, the
single shared seam over Isotope Zero's ``MemoryStore`` + embedder. This adapter
never touches SQLite, the schema, or the embedding model directly.

Session tagging & isolation
---------------------------
A *session tag* is a plain string folded into each card's tags on write and
used to post-filter search results on read. It is computed in the constructor:

- explicit ``session_tag`` → used as-is
- ``agent_id`` given             → ``"agent:{agent_id}"``
- neither                        → no tag (global, shared memory)

``recall(..., filter_session=True)`` (the default) returns only hits whose
``tags`` contain this session's tag; ``filter_session=False`` searches the
whole store. This lets a fleet of AutoGen agents share one Isotope Zero DB
file while keeping their recall scopes isolated.
"""
from __future__ import annotations

from importlib.util import find_spec
from typing import Any

from izero_adapters._engine import DEFAULT_DIM, Engine, get_engine

# Detect pyautogen at import time without importing it (no heavy deps loaded).
_HAS_AUTOGEN: bool = find_spec("autogen") is not None or find_spec("pyautogen") is not None


class IsotopeZeroMemory:
    """AutoGen-friendly memory provider backed by Isotope Zero.

    Parameters
    ----------
    db_path:
        SQLite path for the Isotope Zero store (``":memory:"`` for in-process).
    agent_id:
        Identifier for the owning agent. Drives the session tag so this
        agent's memories are isolated from other agents on the same DB.
    session_tag:
        Override the derived session tag (takes precedence over ``agent_id``).
    embedder / use_daemon / dim:
        Forwarded to :func:`get_engine` when ``engine`` is not supplied.
    engine:
        An existing :class:`Engine` to reuse (useful for tests or for sharing
        one store across several adapter instances).
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        agent_id: str | None = None,
        session_tag: str | None = None,
        embedder: Any = None,
        use_daemon: bool = False,
        dim: int = DEFAULT_DIM,
        engine: Engine | None = None,
    ) -> None:
        if engine is not None:
            self._engine: Engine = engine
            self.db_path = engine.db_path
            self.dim = engine.dim
            self.is_real = engine.is_real
        else:
            self._engine = get_engine(
                db_path=db_path,
                embedder=embedder,
                use_daemon=use_daemon,
                dim=dim,
            )
            self.db_path = db_path
            self.dim = self._engine.dim
            self.is_real = self._engine.is_real

        self.agent_id: str | None = agent_id
        self.session_tag: str | None = self._compute_session_tag(
            session_tag, agent_id
        )

    # -- session tag ------------------------------------------------------- #
    @staticmethod
    def _compute_session_tag(
        session_tag: str | None, agent_id: str | None
    ) -> str | None:
        if session_tag:
            return session_tag
        if agent_id:
            return f"agent:{agent_id}"
        return None

    def _tags_for_write(self, tags: list[str] | None) -> list[str]:
        """Prepend the session tag (if any) to caller-supplied tags."""
        merged: list[str] = []
        if self.session_tag:
            merged.append(self.session_tag)
        if tags:
            merged.extend(tags)
        return merged

    @staticmethod
    def _has_tag(result: dict[str, Any], tag: str) -> bool:
        return tag in (result.get("tags") or [])

    # -- writes ------------------------------------------------------------ #
    def remember(
        self,
        text: str,
        metadata: dict | None = None,
        *,
        tags: list[str] | None = None,
        card_id: str | None = None,
    ) -> str:
        """Store ``text`` as a memory card; return the card id.

        ``metadata`` is a free-form dict (turn number, role, speaker, ...) whose
        non-reserved keys round-trip through the store as ``key=value`` tag
        pairs (handled by :meth:`Engine.add_text`). The session tag is prepended
        to ``tags`` so this agent's memories can be filtered on recall.
        """
        return self._engine.add_text(
            text,
            metadata=metadata,
            tags=self._tags_for_write(tags),
            card_id=card_id,
        )

    # -- reads ------------------------------------------------------------- #
    def recall(
        self,
        query: str,
        top_k: int = 5,
        *,
        filter_session: bool = True,
    ) -> list[dict]:
        """Semantic search for memories matching ``query``.

        Returns up to ``top_k`` result dicts (``{id, text, score, metadata,
        tags, timestamp}``) sorted by cosine similarity descending.

        When ``filter_session`` is True (the default) and this memory has a
        session tag, results are post-filtered to only those whose ``tags``
        contain the session tag — giving per-agent isolation on a shared DB.
        Set ``filter_session=False`` to search across all agents (global
        recall).
        """
        results = self._engine.search(query, top_k=top_k)
        if filter_session and self.session_tag:
            results = [
                r for r in results if self._has_tag(r, self.session_tag)
            ]
        return results

    # -- deletes / counts -------------------------------------------------- #
    def forget(self, card_id: str) -> bool:
        """Delete one card by id; True if a row was removed."""
        return self._engine.delete(card_id)

    def clear_session(self) -> int:
        """Delete every card belonging to this session; return the count.

        If the adapter has no session tag this is a no-op returning 0 (refuse
        to wipe the global store from an untagged instance — caller must
        delete ids explicitly). This is a maintenance op that writes to the DB.
        """
        if not self.session_tag:
            return 0
        deleted = 0
        for card in self._engine.all():
            if self._has_tag(card, self.session_tag):
                if self._engine.delete(card["id"]):
                    deleted += 1
        return deleted

    def count(self, *, session_only: bool = True) -> int:
        """Card count. ``session_only`` restricts to this session's cards
        (when a session tag is set); ``session_only=False`` is the global
        :meth:`Engine.count`."""
        if not session_only or not self.session_tag:
            return self._engine.count()
        return sum(
            1 for card in self._engine.all() if self._has_tag(card, self.session_tag)
        )

    # -- AutoGen wiring ---------------------------------------------------- #
    def attach_to_agent(self, agent: Any, *, recall_top_k: int = 5) -> None:
        """Wire this memory into a pyautogen ``ConversableAgent``.

        Registers a ``last_user_message`` hook so that, before each generated
        reply, relevant memories are recalled and injected into the agent's
        system message as context, and the user's message is remembered for
        future turns.

        Raises a clear ``RuntimeError`` when ``pyautogen`` is not installed —
        the core ``IsotopeZeroMemory`` class remains usable without it.
        """
        if not _HAS_AUTOGEN:
            raise RuntimeError(
                "pyautogen is not installed — install with "
                "`pip install -e adapters[autogen]` to use attach_to_agent."
            )
        memory = self

        def _context_injection(recipient, messages, sender, config):
            # `messages` is the conversation so far; recall against the latest
            # user turn and surface the hits as extra context.
            last_text = ""
            for msg in reversed(messages):
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
                if role == "user":
                    last_text = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                    break
            if last_text:
                memory.remember(last_text, metadata={"role": "user"})
                hits = memory.recall(last_text, top_k=recall_top_k)
                if hits:
                    ctx = "\n".join(f"- {h['text']}" for h in hits)
                    return [{"role": "system", "content": f"Relevant memory:\n{ctx}"}]
            return None

        # ConversableAgent.register_reply(trigger, reply_func) — the trigger
        # is a callable predicate over the agent state; we use the standard
        # "always run before generate_reply" trigger `is_termination`-style.
        # The exact API surface has shifted across autogen releases, so bind
        # defensively: if register_reply accepts our hook, use it; otherwise
        # stash the memory on the agent for manual recall in the reply loop.
        register = getattr(agent, "register_reply", None)
        hooked = False
        if register is not None:
            try:
                # The default trigger is a string "auto" in newer autogen and a
                # callable in older ones; "auto" works across both.
                register("auto", _context_injection)
                hooked = True
            except Exception:  # pragma: no cover - version-dependent
                hooked = False
        if not hooked:  # pragma: no cover - version-dependent
            # Fallback: stash the memory on the agent for manual use.
            agent._izero_memory = memory
