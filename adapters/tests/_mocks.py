"""Lightweight mock framework objects for adapter tests.

The adapter tests exercise the *mapping logic* between Isotope Zero cards and
each framework's object model. Rather than depend on (and pin to) heavy
framework packages, these duck-typed mocks expose exactly the attributes the
adapters read/write. They are intentionally minimal — just enough surface area
to verify the adapter round-trips data correctly.

Each mock mirrors the real class's public shape:

- ``MockDocument``  ~ ``langchain_core.documents.Document``
- ``MockTextNode``  ~ ``llama_index.core.schema.TextNode``
- ``MockNodeWithEmbedding`` ~ a TextNode carrying its own embedding

If a real framework is installed, the adapters work against the real classes
identically (the attribute names match); these mocks only exist so the test
suite runs green in a bare environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockDocument:
    """~ langchain_core.documents.Document"""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class MockTextNode:
    """~ llama_index.core.schema.TextNode (text + metadata, no embedding)."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id_: str | None = None
    embedding: list[float] | None = None

    def get_content(self) -> str:
        return self.text

    def get_metadata_str(self) -> str:
        return " | ".join(f"{k}: {v}" for k, v in self.metadata.items())


@dataclass
class MockNodeWithEmbedding(MockTextNode):
    """A TextNode that already carries an explicit embedding vector."""

    pass


def make_documents(n: int, prefix: str = "doc") -> list[MockDocument]:
    """Build N distinct MockDocuments for batch-add tests."""
    return [
        MockDocument(
            page_content=f"{prefix} content number {i} about topic {i % 3}",
            metadata={"source": "test", "index": i},
            id=f"{prefix}-{i}",
        )
        for i in range(n)
    ]


def make_text_nodes(n: int, prefix: str = "node") -> list[MockTextNode]:
    """Build N distinct MockTextNodes for LlamaIndex add() tests."""
    return [
        MockTextNode(
            text=f"{prefix} text number {i} about subject {i % 4}",
            metadata={"category": f"cat-{i % 2}", "seq": i},
            id_=f"{prefix}-{i}",
        )
        for i in range(n)
    ]
