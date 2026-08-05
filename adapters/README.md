# izero-adapters

Drop-in **Isotope Zero** memory providers for the major Python AI agent frameworks:
**LangChain**, **LlamaIndex**, **AutoGen**, and **CrewAI**.

`izero-adapters` bridges Isotope Zero's low-latency SQLite + vector engine into
the standard `VectorStore` / `Memory` interfaces those frameworks expect, so a
project can adopt Isotope Zero memory in two lines of Python.

## Design

Every adapter talks to the Isotope Zero storage layer through a single shared
seam — `izero_adapters._engine.Engine`, a facade over the real `MemoryStore` +
an embedder. The adapters never touch SQLite, the schema, or the embedding
model directly, which keeps vectors compatible (same L2-normalized model) and
avoids schema duplication.

The engine is located by path (`prototypes/daemon_v0.7/isotope_zero`) and is
**not** pip-installed. Embedder selection degrades gracefully:

1. an explicit `embedder=` passed by the caller (tests inject a deterministic
   stub), then
2. the shared-memory daemon (`use_daemon=True`, if reachable), then
3. a local ONNX `EmbeddingEngine` (if `onnxruntime` + `tokenizers` import), then
4. a deterministic, L2-normalized feature-hash stub (always available) so the
   whole package is runnable with **zero** third-party deps.

Each framework package is imported **lazily**, so `import izero_adapters` never
fails when a framework isn't installed — the adapter class is still
constructible and testable via duck-typed compatibility shims.

## Install

```bash
pip install -e adapters                       # core only, no frameworks
pip install -e "adapters[langchain]"          # + LangChain
pip install -e "adapters[llamaindex]"         # + LlamaIndex
pip install -e "adapters[autogen]"            # + AutoGen
pip install -e "adapters[crewai]"             # + CrewAI
pip install -e "adapters[langchain,llamaindex,autogen,crewai,onnx,dev]"  # everything
```

## Quick start

### LangChain

```python
from izero_adapters.langchain import IsotopeZeroVectorStore

vs = IsotopeZeroVectorStore(db_path="mem.db")
ids = vs.add_texts(["The user prefers dark mode.", "SQLite is fast."],
                   metadatas=[{"source": "chat"}, {"source": "doc"}])
docs = vs.similarity_search("dark mode", k=5)
docs_and_scores = vs.similarity_search_with_score("dark mode", k=5)
```

### LlamaIndex

```python
from llama_index.core.schema import TextNode
from izero_adapters.llamaindex import IsotopeZeroVectorStore

vs = IsotopeZeroVectorStore(db_path="mem.db")
vs.add([TextNode(text="Q3 revenue grew 12%.", metadata={"crew": "fin"})])
result = vs.query(query_str="revenue", similarity_top_k=5)  # via VectorStoreQuery
```

### AutoGen

```python
from izero_adapters.autogen import IsotopeZeroMemory

mem = IsotopeZeroMemory(db_path="agent.db", agent_id="researcher_1")
mem.remember("User asked about quantum computing.", metadata={"turn": 3})
hits = mem.recall("quantum", top_k=5)          # only this agent's memories
mem.attach_to_agent(agent)                      # wire into a ConversableAgent
```

### CrewAI

```python
from izero_adapters.crewai import IsotopeZeroMemory

mem = IsotopeZeroMemory(db_path="crew.db", crew_id="research_crew", agent_id="analyst")
mem.remember("Q3 revenue grew 12% YoY.", metadata={"task": "analysis", "step": 2})
hits = mem.recall("revenue growth", top_k=5)   # isolated to this crew+agent
hits = mem.recall_for_agent("writer", "revenue")  # cross-agent within a crew
```

## The `Engine` seam (advanced)

Adapters that need an escape hatch can use the shared facade directly:

```python
from izero_adapters import get_engine

e = get_engine(db_path="mem.db", use_daemon=True)  # production: daemon embeddings
e.add_text("a fact", metadata={"k": "v"}, tags=["t"])
for hit in e.search("a fact", top_k=5):
    print(hit["id"], hit["score"], hit["text"], hit["metadata"], hit["tags"])
e.delete("iz-...")
print(e.count())
```

`Engine.is_real` reports whether real ONNX/daemon embeddings are in use
(`False` = the deterministic stub fallback). `Engine.store` exposes the
underlying `MemoryStore` for advanced use.

## Testing

```bash
.venv/bin/python -m pytest adapters/tests -q
```

The full suite runs green with **no** frameworks and **no** ONNX installed —
tests use a deterministic stub embedder and duck-typed mock framework objects
(`tests/_mocks.py`). `pytest.importorskip` guards the optional real-framework
integration tests so they skip cleanly when a framework is absent.

## Scope & safety

`izero-adapters` is an **integration library**: it *uses* the Isotope Zero
engine via the stable `MemoryStore` / embedder contract. It never modifies core
storage schemas or the prototype sources under `prototypes/`. Running the test
suite leaves `prototypes/` completely untouched.
