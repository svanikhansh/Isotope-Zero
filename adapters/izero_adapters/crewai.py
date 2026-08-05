"""CrewAI memory adapter for Isotope Zero.

Provides :class:`IsotopeZeroMemory` — a self-contained, framework-agnostic
memory provider with a uniform ``.remember`` / ``.recall`` API plus
CrewAI-flavored crew/agent session tagging for isolation across crews and the
agents within them.

Design
------
CrewAI's memory layer is not a strict base-class contract (its public surface
has shifted across releases), so this adapter does **not** subclass any crewai
type. Instead it exposes a clean, standalone object that can be used directly
and, when the ``crewai`` package is importable, wired into a ``Crew`` via the
:meth:`IsotopeZeroMemory.attach_to_crew` helper.

The module imports ``crewai`` **lazily and guarded** so that
``import izero_adapters.crewai`` never fails when crewai is absent. The module
flag :data:`_HAS_CREWAI` records whether crewai was found at import time, and
:meth:`attach_to_crew` raises a clear ``RuntimeError`` when it is not.

All storage work is delegated to :class:`izero_adapters._engine.Engine`, the
single shared seam over Isotope Zero's ``MemoryStore`` + embedder. This adapter
never touches SQLite, the schema, or the embedding model directly.

Session tagging & isolation
---------------------------
A *session tag* is a plain string folded into each card's tags on write and
used to post-filter search results on read. It is computed in the constructor:

- explicit ``session_tag`` → used as-is
- both ``crew_id`` and ``agent_id`` → ``"crew:{crew_id}:agent:{agent_id}"``
- only ``crew_id``                 → ``"crew:{crew_id}"``
- only ``agent_id``                → ``"agent:{agent_id}"``
- neither                          → no tag (global memory)

``recall(..., filter_session=True)`` (the default) returns only hits whose
``tags`` contain this session's tag; ``filter_session=False`` searches the
whole store. :meth:`recall_for_agent` reaches across agents within the same
crew by building that agent's session tag on the fly.
"""
from __future__ import annotations

from typing import Any

from ._engine import DEFAULT_DIM, Engine, get_engine

# --------------------------------------------------------------------------- #
# Optional crewai import — guarded so the module loads without crewai.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import availability is environment dependent
    import crewai  # type: ignore  # noqa: F401

    _HAS_CREWAI: bool = True
except Exception:  # pragma: no cover - the normal case in CI / bare envs
    _HAS_CREWAI = False


class IsotopeZeroMemory:
    """CrewAI-friendly memory provider with crew/agent session isolation.

    Mirrors the uniform adapter API (``remember`` / ``recall`` / ``forget`` /
    ``clear_session`` / ``count``) shared across the izero-adapters, and adds
    CrewAI-flavored helpers (``attach_to_crew``, ``recall_for_agent``).

    Parameters
    ----------
    db_path:
        SQLite path for the underlying ``MemoryStore`` (``":memory:"`` for RAM).
    crew_id, agent_id, session_tag:
        Identity inputs used to compute :attr:`session_tag`. See the module
        docstring for the precedence rules.
    embedder, use_daemon, dim:
        Forwarded to :class:`Engine` for embedder selection.
    engine:
        An existing :class:`Engine` to wrap (e.g. shared across several memory
        objects on one DB). When given, ``db_path``/``embedder``/``use_daemon``
        /``dim`` are taken from it and the constructor args ignored.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        crew_id: str | None = None,
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

        # Identity (kept verbatim for recall_for_agent / cross-agent reach).
        self.crew_id: str | None = crew_id
        self.agent_id: str | None = agent_id
        # Compute the session tag with the documented precedence.
        self.session_tag: str | None = self._compute_session_tag(
            session_tag, crew_id, agent_id
        )

    # -- session tag ------------------------------------------------------- #
    @staticmethod
    def _compute_session_tag(
        session_tag: str | None,
        crew_id: str | None,
        agent_id: str | None,
    ) -> str | None:
        if session_tag:
            return session_tag
        if crew_id and agent_id:
            return f"crew:{crew_id}:agent:{agent_id}"
        if crew_id:
            return f"crew:{crew_id}"
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

        ``metadata`` is a free-form dict whose non-reserved keys round-trip
        through the store as ``key=value`` tag pairs (handled by
        :meth:`Engine.add_text`). The session tag is prepended to ``tags`` so
        this session's memories can be filtered on recall.
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
        contain the session tag — giving per-crew / per-agent isolation on a
        shared DB. Set ``filter_session=False`` to search across all sessions
        (global recall).
        """
        results = self._engine.search(query, top_k=top_k)
        if filter_session and self.session_tag:
            results = [
                r for r in results if self._has_tag(r, self.session_tag)
            ]
        return results

    def recall_for_agent(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Cross-agent recall within the same crew.

        Builds the target agent's session tag
        ``"crew:{self.crew_id}:agent:{agent_id}"`` (requires ``self.crew_id``
        to be set) and post-filters ``engine.search`` results by it. Useful
        for an agent to pull a teammate's memories within the same crew
        without exposing memories from other crews.

        If ``self.crew_id`` is unset, falls back to the bare
        ``"agent:{agent_id}"`` tag so the call still works in agent-only
        (no-crew) configurations.
        """
        if self.crew_id:
            target = f"crew:{self.crew_id}:agent:{agent_id}"
        else:
            target = f"agent:{agent_id}"
        results = self._engine.search(query, top_k=top_k)
        return [r for r in results if self._has_tag(r, target)]

    # -- deletes / counts -------------------------------------------------- #
    def forget(self, card_id: str) -> bool:
        """Delete one card by id; True if a row was removed."""
        return self._engine.delete(card_id)

    def clear_session(self) -> int:
        """Delete every card belonging to this session; return the count.

        If there is no session tag this is a no-op returning 0 (clearing the
        global store is intentionally not exposed here — use ``engine`` +
        per-id ``delete`` if you really need that).
        """
        if not self.session_tag:
            return 0
        removed = 0
        for card in self._engine.all():
            if self._has_tag(card, self.session_tag):
                if self._engine.delete(card["id"]):
                    removed += 1
        return removed

    def count(self, *, session_only: bool = True) -> int:
        """Number of stored cards.

        ``session_only=True`` (default) counts only this session's cards;
        ``session_only=False`` returns the global store count. With no session
        tag both modes return the global count.
        """
        if not self.session_tag or not session_only:
            return self._engine.count()
        return sum(
            1
            for card in self._engine.all()
            if self._has_tag(card, self.session_tag)
        )

    # -- CrewAI wiring ----------------------------------------------------- #
    def attach_to_crew(self, crew: Any, *, recall_top_k: int = 5) -> None:
        """Wire this memory into a CrewAI ``Crew`` (requires crewai installed).

        Best-effort glue across crewai's evolving memory surface: it first
        tries to install ``self`` as ``crew.memory`` (the modern path), then
        falls back to registering before/after-task hooks that recall relevant
        memories into the task context and remember task outputs back. The
        core :class:`IsotopeZeroMemory` works standalone without ever calling
        this — it is purely optional integration sugar.

        Raises a clean :class:`RuntimeError` when crewai is not importable, so
        callers can branch on presence rather than catching an ImportError.
        """
        if not _HAS_CREWAI:
            raise RuntimeError(
                "crewai not installed — install with "
                "`pip install -e \"adapters[crewai]\"` to use attach_to_crew."
            )
        # Preferred modern path: expose this memory object on the crew.
        try:
            crew.memory = self  # type: ignore[attr-defined]
            return
        except Exception:
            pass  # fall through to hook-based wiring

        # Fallback: register task hooks. crewai's hook registration surface
        # varies by version; we attempt the common attribute and tolerate the
        # case where it is unavailable rather than hard-failing.
        mem = self

        def _before_task(task: Any) -> None:
            query = getattr(task, "description", "") or ""
            if not query:
                return
            hits = mem.recall(query, top_k=recall_top_k)
            if hits:
                context = getattr(task, "context", None)
                summary = "\n".join(h["text"] for h in hits)
                if context is None:
                    task.context = summary  # type: ignore[attr-defined]
                else:
                    task.context = f"{context}\n{summary}"  # type: ignore[attr-defined]

        def _after_task(task: Any) -> None:
            output = getattr(task, "output", None)
            text = getattr(output, "raw", None) or str(output)
            if text and text.lower() not in ("none", ""):
                mem.remember(text, metadata={"task": getattr(task, "description", "")})

        hooks = getattr(crew, "task_hooks", None)
        if hooks is None:
            crew.task_hooks = {"before": [_before_task], "after": [_after_task]}  # type: ignore[attr-defined]
        else:  # pragma: no cover - depends on crewai internals
            hooks.setdefault("before", []).append(_before_task)
            hooks.setdefault("after", []).append(_after_task)

    # -- introspection ----------------------------------------------------- #
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        tag = self.session_tag or "global"
        return (
            f"IsotopeZeroMemory(db_path={self.db_path!r}, "
            f"session={tag!r}, real={self.is_real})"
        )
