"""MCP (Model Context Protocol) server for isotope_zero.

Exposes isotope_zero's memory store over a standard MCP tool interface so any
MCP-compatible agent client (Claude Desktop, etc.) can read/write the local
SQLite-backed memory without knowing anything about the internals.

Tools:
    add_memory(content)              -> {memory_id, action, confidence, ...}
    query_memory(query, token_budget) -> {hits, route, tokens_used, saved}
    delete_memory(memory_id)          -> bool
    get_metrics()                     -> {db_size, count, tokens_saved, ...}

The server is intentionally thin: it wires the MCP tool layer onto the
`MemoryStore` + `QueryRouter` + triage pipeline. All the cost-saving logic
lives below this layer.
"""
from __future__ import annotations

import os
from typing import Any

from isotope_zero.core.router import QueryRouter
from isotope_zero.core.store import MemoryStore
from isotope_zero.core.triage import classify_action, compress_to_card
from isotope_zero.embeddings.onnx_embed import EmbeddingEngine
from isotope_zero.tokens import estimate_tokens

# Track cumulative savings across the server's lifetime for get_metrics.
# (Per-query savings are also in each QueryResult; this is the running total.)
_tokens_saved_total = 0.0
_raw_history_total = 0.0


class IsotopeZeroServer:
    """The isotope_zero service object backing the MCP tools.

    Encapsulating the store/embedder/router in one object makes the tools
    trivially testable without spinning up a real MCP transport: tests can
    instantiate `IsotopeZeroServer(db_path=":memory:")` and call the methods
    directly, then the MCP layer below is a thin adapter over the same calls.
    """

    def __init__(
        self,
        db_path: str | None = None,
        embedder: EmbeddingEngine | None = None,
        use_daemon: bool = False,
    ) -> None:
        if use_daemon and embedder is None:
            from isotope_zero.daemon.client import DaemonClient

            embedder = DaemonClient()
        self.embedder = embedder or EmbeddingEngine()
        # Default to a file-backed DB in the user's cache dir so memory
        # persists across server restarts; ":memory:" for tests.
        if db_path is None:
            db_path = os.environ.get(
                "ISOTOPE_ZERO_DB", os.path.join(".isotope_zero_cache", "isotope_zero.sqlite")
            )
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.store = MemoryStore(db_path, embedder=self.embedder)
        self.router = QueryRouter(self.store, self.embedder)

    # ------------------------------------------------------------------ #
    # Tool implementations
    # ------------------------------------------------------------------ #

    def add_memory(self, content: str) -> dict[str, Any]:
        """Triage + compress + store a raw memory input.

        Runs the local heuristic classifier (ADD/UPDATE/DELETE) and the
        compressor, embeds the resulting fact, and persists a MemoryCard.
        """
        global _tokens_saved_total, _raw_history_total
        if not content or not content.strip():
            return {"error": "content must be non-empty"}

        action = classify_action(content)
        # Embed the compressed FACT (not the raw input) — cheaper and the
        # vector then represents the canonical assertion.
        card = compress_to_card(content, embedding=None)
        emb = self.embedder.embed_text(card.fact)
        card.embedding = emb

        if action.action.value == "DELETE":
            deleted = False
            if action.target_id:
                # Best-effort: try a tag/fact lookup for the target.
                for field in ("tags", "fact"):
                    candidates = self.store.sql_lookup(field, action.target_id)
                    if candidates:
                        for c in candidates:
                            self.store.delete(c.id)
                            deleted = True
                        break
            return {
                "action": action.action.value,
                "confidence": round(action.confidence, 3),
                "escalated": action.escalated,
                "target_id": action.target_id,
                "deleted": deleted,
                "reasoning": action.reasoning,
            }

        # ADD or UPDATE — persist the card. For UPDATE we upsert; if a
        # matching card exists (by tag target), we update it in place.
        if action.action.value == "UPDATE" and action.target_id:
            for field in ("tags", "fact"):
                existing = self.store.sql_lookup(field, action.target_id)
                if existing:
                    # Reuse the first match's id so this overwrites it.
                    card.id = existing[0].id
                    self.store.update(card)
                    break
            else:
                self.store.add(card)
        else:
            self.store.add(card)

        _raw_history_total += estimate_tokens(content)
        return {
            "memory_id": card.id,
            "action": action.action.value,
            "confidence": round(action.confidence, 3),
            "escalated": action.escalated,
            "fact": card.fact,
            "evidence": card.evidence,
            "tags": card.tags,
            "source_tokens": card.source_tokens,
        }

    def query_memory(self, query: str, token_budget: int = 300) -> dict[str, Any]:
        """Route the query (SQL-first, vector-second) within a token budget."""
        global _tokens_saved_total
        if not query or not query.strip():
            return {"error": "query must be non-empty"}
        result = self.router.query(query, token_budget=token_budget)
        _tokens_saved_total += result.tokens_saved_vs_raw
        return {
            "hits": [
                {
                    "id": h.card.id,
                    "fact": h.card.fact,
                    "evidence": h.card.evidence,
                    "tags": h.card.tags,
                    "score": round(h.score, 3),
                    "route": h.route,
                    "tokens": h.token_cost,
                }
                for h in result.hits
            ],
            "route_used": result.route_used,
            "tokens_used": result.tokens_used,
            "tokens_saved_vs_raw": result.tokens_saved_vs_raw,
            "latency_ms": round(result.latency_ms, 3),
            "budget_exhausted": result.budget_exhausted,
        }

    def delete_memory(self, memory_id: str) -> dict[str, Any]:
        """Delete a memory by id."""
        deleted = self.store.delete(memory_id)
        return {"memory_id": memory_id, "deleted": deleted}

    def get_metrics(self) -> dict[str, Any]:
        """Return DB size, card count, and cumulative tokens saved vs raw."""
        return {
            "db_size_bytes": self.store.db_size_bytes(),
            "db_size_human": _human_size(self.store.db_size_bytes()),
            "card_count": self.store.count(),
            "embedding_is_real": self.embedder.is_real,
            "cumulative_tokens_saved_vs_raw": int(_tokens_saved_total),
            "cumulative_raw_history_tokens": int(_raw_history_total),
        }

    def run_consolidation(self) -> dict[str, Any]:
        """Trigger one async-consolidation sweep and report the outcome.

        Deduplicates near-identical cards (merging evidence), prunes
        decayed zero-recall cards, and reports the net tokens reclaimed from
        context. Runs off the request hot path; safe alongside active
        reads/writes (WAL mode + a single atomic transaction per sweep).
        """
        from isotope_zero.core.consolidation import Consolidator

        report = Consolidator(self.store, embedder=self.embedder).run()
        return {
            "merged_cards": report.merged_cards,
            "decayed_cards": report.decayed_cards,
            "survivors": report.survivors,
            "tokens_before": report.tokens_before,
            "tokens_after": report.tokens_after,
            "tokens_reclaimed": report.tokens_reclaimed,
            "latency_ms": round(report.latency_ms, 3),
            "pruned_mean_vitality": round(report.pruned_mean_vitality, 4),
        }


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n} {unit}" if unit == "B" else f"{n/1024:.1f} {unit}"
        n = int(n)
    return f"{n} B"


# ---------------------------------------------------------------------- #
# MCP adapter
# ---------------------------------------------------------------------- #

def _import_mcp_server_class():
    """Locate the high-level MCP server class across mcp SDK versions.

    The SDK renamed `FastMCP` to `MCPServer` in the 2.0 line. We support both:
      - mcp >= 2.0: `from mcp.server.mcpserver import MCPServer`
      - mcp 1.x:   `from mcp.server.fastmcp import FastMCP`
    Both expose a `@tool()` decorator, `add_tool`, `list_tools`, `call_tool`,
    and `run(transport='stdio')`, so the rest of the adapter is identical.
    """
    # mcp 2.x
    try:
        from mcp.server.mcpserver import MCPServer  # type: ignore
        return MCPServer
    except Exception:
        pass
    # mcp 1.x
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
        return FastMCP
    except Exception:
        pass
    raise ImportError(
        "The `mcp` package is required to run the MCP server. "
        "Install it with: pip install mcp"
    )


def build_mcp_app(
    server: IsotopeZeroServer | None = None, use_daemon: bool = False
):
    """Build and return an MCP server app exposing isotope_zero tools.

    Kept lazy/optional: if the `mcp` SDK isn't installed (e.g. in a minimal
    test env), this raises a clear error only when actually called, not at
    import time. The `IsotopeZeroServer` above works without MCP.
    """
    ServerCls = _import_mcp_server_class()
    server = server or IsotopeZeroServer(use_daemon=use_daemon)
    app = ServerCls("isotope_zero")

    @app.tool()
    def add_memory(content: str) -> dict:
        """Store a memory. Triage classifies ADD/UPDATE/DELETE locally; the
        input is compressed into a minimal Memory Card (fact + evidence) and
        persisted. Returns the action taken and the card id."""
        return server.add_memory(content)

    @app.tool()
    def query_memory(query: str, token_budget: int = 300) -> dict:
        """Query the memory within a token budget. Routes to fast SQL lookup
        for explicit state queries, vector similarity for fuzzy/semantic ones.
        Returns ranked hits and tokens saved vs replaying raw history."""
        return server.query_memory(query, token_budget)

    @app.tool()
    def delete_memory(memory_id: str) -> dict:
        """Delete a memory card by its id."""
        return server.delete_memory(memory_id)

    @app.tool()
    def get_metrics() -> dict:
        """Return DB size, card count, embedding mode, and cumulative tokens
        saved versus replaying raw conversation history."""
        return server.get_metrics()

    @app.tool()
    def run_consolidation() -> dict:
        """Run one async-consolidation sweep: deduplicate near-identical memory
        cards (merging evidence without losing facts), prune decayed zero-recall
        cards, and reclaim context tokens. Returns counts of merged/decayed
        cards, survivors, and net tokens reclaimed."""
        return server.run_consolidation()

    return app


def main() -> None:
    """Run the isotope_zero MCP server over stdio (default MCP transport)."""
    app = build_mcp_app()
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
