# Framework Adapters

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://pypi.org/project/isotope-zero/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)
[![Tests: 134 passed](https://img.shields.io/badge/tests-134%20passed-brightgreen)](../prototypes/synthesis_v1.0/tests)

`izero-adapters` is the zero-friction drop-in that wires Isotope Zero's low-latency
SQLite + float32 BLAS vector engine into the standard `VectorStore` / `Memory`
interfaces expected by **LangChain**, **LlamaIndex**, **AutoGen**, and **CrewAI**.
A project can adopt Isotope Zero memory in two lines of Python — no remote
embedding API, no schema migration, no per-framework storage rewrite.

```python
from izero_adapters.langchain import IsotopeZeroVectorStore

vs = IsotopeZeroVectorStore(db_path="mem.db")   # that's it
```

The engine that powers every adapter is the unified client shipped in
[`prototypes/synthesis_v1.0`](../prototypes/synthesis_v1.0) — the Grand Synthesis
that unifies the winning stack (float32 BLAS, Ebbinghaus decay, semantic graph,
shared-memory embedding daemon). See [`architecture.md`](architecture.md) for the
8-phase research lineage and the root [`README.md`](../README.md) for the full
`IsotopeZero` client API. The adapters themselves live in [`adapters/`](../adapters) and are
documented at length in [`adapters/README.md`](../adapters/README.md).

> **Production default.** Construct the underlying engine with `use_mmap=False`
> (the concurrency-safe heap BLAS path). The `mmap` tier is an *experimental*
> opt-in that, at prototype scale, measured ~7% slower than heap, added ~40 MB
> RSS, and `SIGILL`-crashed under 10-thread concurrency — it is not advertised
> as a benefit and is not the default. See [`architecture.md`](architecture.md).

---

## The Engine seam

Every adapter talks to Isotope Zero through one shared module:
[`izero_adapters._engine.Engine`](../adapters/izero_adapters/_engine.py) — a
framework-agnostic facade over the real `MemoryStore` + an embedder. The adapter
submodules (`langchain`, `llamaindex`, `autogen`, `crewai`) never touch SQLite,
the schema, or the embedding model directly. They call `Engine` methods that
return plain dicts, then map those dicts into framework-native objects
(`Document`, `TextNode`, memory-result lists). This keeps vectors compatible
across the ecosystem — every path uses the same L2-normalized model, so cosine
similarity == dot product, matching `MemoryStore.vector_search`'s contract.

### Engine located by path, not pip-installed

The engine lives in `prototypes/daemon_v0.7/isotope_zero` and is **not**
pip-installed as a dependency of the adapters. Rather than reinvent storage
(which would diverge from the real schema and break vector compatibility),
`_engine.py` resolves the engine root relative to its own file and inserts it on
`sys.path`:

```
<repo>/adapters/izero_adapters/_engine.py
<repo>/prototypes/daemon_v0.7/isotope_zero/   ← engine, imported by path
```

The `IZERO_ENGINE_PATH` environment variable overrides this for flexibility. If
the engine cannot be imported, the adapters raise a clean `EngineError` with an
actionable message instead of an `ImportError` traceback. This means the adapter
package is an **integration library**: it *uses* the engine via the stable
`MemoryStore` / embedder contract and never modifies core storage schemas or the
prototype sources under `prototypes/`.

### Four-tier graceful embedder degradation

`_build_embedder` selects an embedder with a strict priority cascade, so the
package is fully runnable with **zero** third-party deps:

| Tier | Source | When chosen | `is_real` |
|---|---|---|---|
| 1 | `embedder=` passed explicitly by the caller | Always, if given | per-embedder |
| 2 | `DaemonClient` (`use_daemon=True`) | Only if a daemon answers `ping()` | `True` |
| 3 | Local ONNX `EmbeddingEngine` | If `onnxruntime` + `tokenizers` import and `is_real()` | `True` |
| 4 | `_StubEmbedder` (feature-hash) | Always available — the final fallback | `False` |

The tier-4 stub is a deterministic, L2-normalized feature-hashing embedder over
word tokens and character trigrams (unsigned bucketing via SHA-256). Identical
texts score exactly 1.0; lexically overlapping texts land with positive cosine;
unrelated texts are near-orthogonal. It keeps the adapters — and their entire
test suite — runnable on a bare machine with no ONNX, no daemon, and no
framework packages installed. `Engine.is_real` reports whether real
ONNX/daemon embeddings are in use (`False` = the stub fallback).

Tier 2 is the production embedding path: the daemon centralizes the ~360 MB
`onnxruntime` footprint in a single process (default socket `/tmp/izero.sock`)
so that client workers stay small. `HybridEmbeddingEngine` is daemon-FIRST with
a *silent* in-process ONNX fallback and never raises on transport failure.

### Lazy framework imports

Each framework package is imported **lazily and guarded**. `langchain.py` uses
`importlib.util.find_spec("langchain_core")`; `llamaindex.py` uses
`find_spec("llama_index")`; `autogen.py` checks `autogen`/`pyautogen`; `crewai.py`
wraps `import crewai` in a `try/except`. Module flags (`_HAS_LANGCHAIN`,
`_HAS_LLAMAINDEX`, `_HAS_AUTOGEN`, `_HAS_CREWAI`) record availability at import
time. When a framework is absent, a **duck-typed shim** (a minimal
`_BaseVectorStore` + `_Document` for LangChain; `_BasePydanticVectorStore` +
`_VectorStoreQuery`/`_VectorStoreQueryResult`/`_TextNode` for LlamaIndex) takes
its place so the adapter class is still constructible and testable. The public
method surface and return shapes are identical in both paths — only the
returned object's concrete type differs.

This is why `izero-adapters` has **zero hard dependencies** (its `pyproject.toml`
`dependencies = []`): the package imports cleanly everywhere and degrades to the
stub embedder + framework shims when nothing else is installed.

---

## Install

The adapters require Python 3.9+ (the engine itself requires 3.10+). The engine
must be importable by path — either run from the repo checkout, or set
`IZERO_ENGINE_PATH` to the directory containing the `isotope_zero` package
(e.g. `prototypes/daemon_v0.7`).

```bash
# Core only — no frameworks, no ONNX. Runs on the stub embedder + shims.
pip install -e adapters

# One framework at a time:
pip install -e "adapters[langchain]"      # + langchain-core
pip install -e "adapters[llamaindex]"     # + llama-index-core
pip install -e "adapters[autogen]"        # + pyautogen
pip install -e "adapters[crewai]"         # + crewai

# Real embeddings via local ONNX (otherwise the stub is used for tests/dev):
pip install -e "adapters[onnx]"           # + onnxruntime, tokenizers, numpy

# Everything — all four frameworks, real ONNX, and dev/test tooling:
pip install -e "adapters[langchain,llamaindex,autogen,crewai,onnx,dev]"
```

The `onnx` extra pulls `onnxruntime>=1.16.0`, `tokenizers>=0.15.0`, and
`numpy>=1.24.0`. The `dev` extra pulls `pytest>=7.0.0` and `pytest-cov>=4.0.0`.

---

## LangChain

`izero_adapters.langchain.IsotopeZeroVectorStore` — a LangChain-compatible
`VectorStore`. When `langchain_core` is importable it subclasses
`langchain_core.vectorstores.VectorStore` and returns real
`langchain_core.documents.Document` objects, so it drops into any LangChain
pipeline unchanged. When `langchain_core` is absent, a duck-typed
`_BaseVectorStore` + `_Document` shim takes its place.

LangChain's `embedding` argument is accepted by `from_texts` for signature
compatibility but **ignored** — Isotope Zero manages embeddings via its own
engine, which keeps vectors compatible across the ecosystem.

```python
from izero_adapters.langchain import IsotopeZeroVectorStore

# Construct (db_path defaults to ":memory:"; pass a file path to persist).
vs = IsotopeZeroVectorStore(db_path="mem.db")

# Write — returns Isotope Zero card ids in input order.
# metadatas round-trip through the store as "key=value" tag pairs.
ids = vs.add_texts(
    ["The user prefers dark mode.", "SQLite is fast."],
    metadatas=[{"source": "chat"}, {"source": "doc"}],
)

# Read — up to k Document objects, sorted by cosine similarity descending.
docs = vs.similarity_search("dark mode", k=5)

# Read with scores — (Document, score) tuples, score in [0, 1].
docs_and_scores = vs.similarity_search_with_score("dark mode", k=5)

# Metadata filter (LangChain convention): post-filter to results whose
# metadata contains every key/value pair.
docs = vs.similarity_search("dark mode", k=5, filter={"source": "chat"})

# Delete by id (best-effort; missing ids are silently ignored).
vs.delete(ids)

# One-shot constructor (LangChain convention). embedding= is accepted
# for compatibility but ignored.
vs = IsotopeZeroVectorStore.from_texts(
    ["A fact."], embedding=None, metadatas=[{"source": "seed"}], db_path="mem.db"
)
```

Each returned `Document.metadata` carries the card `id` and `tags` (a list);
`similarity_search_with_score` additionally sets `metadata["score"]`.

---

## LlamaIndex

`izero_adapters.llamaindex.IsotopeZeroVectorStore` — a LlamaIndex-compatible
`BasePydanticVectorStore`. When `llama_index` is importable it subclasses the
real `BasePydanticVectorStore` and returns real `TextNode` /
`VectorStoreQueryResult` objects. When `llama_index` is absent, duck-typed shims
take their place. `stores_text` is `True`.

```python
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from izero_adapters.llamaindex import IsotopeZeroVectorStore

vs = IsotopeZeroVectorStore(db_path="mem.db")

# Write — add([BaseNode, ...]) returns the list of card ids.
# For each node it reads get_content() (or .text), .metadata, and .id_ (or .id).
# If a node already carries an embedding, it is stored verbatim; otherwise the
# engine embeds the text.
ids = vs.add([TextNode(text="Q3 revenue grew 12%.", metadata={"crew": "fin"})])

# Read — query() takes a VectorStoreQuery and returns a VectorStoreQueryResult
# with .nodes (TextNode), .similarities (scores), .ids (card ids), sorted desc.
result = vs.query(
    VectorStoreQuery(query_str="revenue", similarity_top_k=5)
)
for node, score, cid in zip(result.nodes, result.similarities, result.ids):
    print(cid, score, node.get_content())

# Delete by id (best-effort).
vs.delete(ref_doc_id=ids[0])

# persist() is a no-op: Isotope Zero persists to SQLite continuously on write.
```

If `query.query_embedding` is provided it is used directly; otherwise the engine
embeds `query.query_str`.

---

## AutoGen

`izero_adapters.autogen.IsotopeZeroMemory` — a self-contained memory provider
with the uniform `.remember` / `.recall` API plus automatic `agent_id` session
tagging. AutoGen's memory model is less standardized than the
LangChain/LlamaIndex `VectorStore` contracts, so this adapter does **not**
subclass an autogen base type. It is a clean, standalone object usable directly
in a multi-agent loop and, when `pyautogen` is importable, wireable into a
`ConversableAgent` via `attach_to_agent` (which raises a clear `RuntimeError`
when `pyautogen` is absent).

```python
from izero_adapters.autogen import IsotopeZeroMemory

# agent_id drives the session tag so this agent's memories are isolated.
mem = IsotopeZeroMemory(db_path="agent.db", agent_id="researcher_1")

# Write — metadata is free-form; non-reserved keys round-trip as "key=value" tags.
# The session tag ("agent:researcher_1") is prepended automatically.
cid = mem.remember("User asked about quantum computing.", metadata={"turn": 3})

# Read — only this agent's memories (filter_session=True is the default).
hits = mem.recall("quantum", top_k=5)

# Cross-agent / global recall:
hits = mem.recall("quantum", top_k=5, filter_session=False)

# Wire into a ConversableAgent (requires pyautogen installed).
# Registers a hook that recalls relevant memories into the system message and
# remembers each user turn for future runs.
mem.attach_to_agent(agent, recall_top_k=5)

# Maintenance:
mem.forget(cid)            # delete one card by id
mem.clear_session()        # delete every card in this session (no-op if global)
mem.count(session_only=True)   # this session only
mem.count(session_only=False)  # global store count
```

Each `recall` hit is a dict `{id, text, score, metadata, tags, timestamp}`.

---

## CrewAI

`izero_adapters.crewai.IsotopeZeroMemory` — a self-contained memory provider
with CrewAI-flavored **crew + agent** session tagging for isolation across
crews and the agents within them. Like the AutoGen adapter it does not subclass
any crewai type; it is standalone and, when `crewai` is importable, wireable
into a `Crew` via `attach_to_crew` (which raises a clear `RuntimeError` when
`crewai` is absent).

```python
from izero_adapters.crewai import IsotopeZeroMemory

# crew_id + agent_id together drive the session tag.
mem = IsotopeZeroMemory(
    db_path="crew.db", crew_id="research_crew", agent_id="analyst",
)

# Write — the session tag ("crew:research_crew:agent:analyst") is prepended.
cid = mem.remember(
    "Q3 revenue grew 12% YoY.", metadata={"task": "analysis", "step": 2},
)

# Read — isolated to this crew + agent (filter_session=True is the default).
hits = mem.recall("revenue growth", top_k=5)

# Cross-agent recall within the SAME crew (no exposure to other crews):
hits = mem.recall_for_agent("writer", "revenue", top_k=5)

# Wire into a Crew (requires crewai installed).
# Preferred: installs self as crew.memory; falls back to before/after task hooks.
mem.attach_to_crew(crew, recall_top_k=5)

# Maintenance:
mem.forget(cid)
mem.clear_session()
mem.count(session_only=True)
```

---

## Multi-agent session tagging

A *session tag* is a plain string folded into each card's `tags` on write and
used to post-filter `recall` results on read. It is computed in each adapter's
constructor, giving a fleet of agents (or crews) sharing one Isotope Zero DB
file isolated memory scopes with no extra infrastructure.

**AutoGen** derives the tag from `agent_id`:

| Inputs | Session tag |
|---|---|
| explicit `session_tag` | used as-is |
| `agent_id` given | `agent:{agent_id}` |
| neither | *(none — global, shared memory)* |

**CrewAI** derives the tag from `crew_id` and `agent_id` together:

| Inputs | Session tag |
|---|---|
| explicit `session_tag` | used as-is |
| both `crew_id` and `agent_id` | `crew:{crew_id}:agent:{agent_id}` |
| only `crew_id` | `crew:{crew_id}` |
| only `agent_id` | `agent:{agent_id}` |
| neither | *(none — global)* |

`recall(..., filter_session=True)` (the default) returns only hits whose `tags`
contain this session's tag; `filter_session=False` searches the whole store.
CrewAI's `recall_for_agent(other_agent, query)` builds the target agent's tag on
the fly and post-filters by it — letting an agent pull a teammate's memories
within the same crew without exposing memories from other crews.

### Two agents, one DB

```python
from izero_adapters.autogen import IsotopeZeroMemory

# Two AutoGen agents sharing one SQLite file. Each sees only its own memories
# by default; both can be wired into their respective ConversableAgents.
researcher = IsotopeZeroMemory(db_path="shared.db", agent_id="researcher")
writer     = IsotopeZeroMemory(db_path="shared.db", agent_id="writer")

researcher.remember("Found that Q3 revenue grew 12% YoY.", metadata={"turn": 1})
writer.remember("Drafted the Q3 summary paragraph.", metadata={"turn": 2})

# Isolated by default:
assert all("agent:researcher" in h["tags"] for h in researcher.recall("Q3"))
assert all("agent:writer"     in h["tags"] for h in writer.recall("summary"))

# Global recall across both agents:
all_hits = researcher.recall("Q3", filter_session=False)
```

The same pattern applies to CrewAI with `crew_id` + `agent_id` for two-level
isolation (across crews *and* across agents within a crew).

---

## The Engine escape hatch

Adapters that need direct access can use the shared facade itself. This is the
single integration seam every adapter calls — useful for sharing one store
across several adapter instances, for custom retrieval logic, or for
framework-agnostic scripts.

```python
from izero_adapters import get_engine

# Production: daemon embeddings (centralizes ~360 MB ONNX in one process).
# For the concurrency-safe heap path use use_daemon=False (no mmap).
e = get_engine(db_path="mem.db", use_daemon=True)

# Writes ---------------------------------------------------------------
id1 = e.add_text("a fact", metadata={"k": "v"}, tags=["t"])
ids  = e.add_texts(["one", "two"], metadatas=[{"a": 1}, {"a": 2}])

# Reads ----------------------------------------------------------------
for hit in e.search("a fact", top_k=5):
    print(hit["id"], hit["score"], hit["text"], hit["metadata"], hit["tags"])
card = e.get(id1)          # single card by id, or None
print(e.count())           # total cards
print(e.is_real)           # True = ONNX/daemon; False = stub fallback

# Deletes --------------------------------------------------------------
e.delete(id1)              # -> True if a row was removed

# Escape hatch ---------------------------------------------------------
store = e.store            # the underlying MemoryStore, for advanced use
```

`Engine` public surface:

| Member | Kind | Returns |
|---|---|---|
| `Engine(db_path, *, embedder, use_daemon, dim)` | constructor | `Engine` |
| `get_engine(db_path, *, embedder, use_daemon, dim)` | factory | `Engine` |
| `add_text(text, *, metadata, tags, card_id, evidence, embedding)` | method | card id `str` |
| `add_texts(texts, *, metadatas, tags, card_ids)` | method | `list[str]` |
| `search(query, *, top_k, query_embedding)` | method | `list[dict]` (`{id,text,score,metadata,tags,timestamp}`) |
| `get(card_id)` | method | `dict \| None` |
| `count()` | method | `int` |
| `all()` | method | `list[dict]` |
| `delete(card_id)` | method | `bool` |
| `is_real` | property | `bool` |
| `store` | property | underlying `MemoryStore` |
| `embedder` | property | the active embedder |

---

## Compatibility & testing

Because frameworks import lazily and the embedder degrades to the stub,
`izero_adapters` is **always importable** — `import izero_adapters` and
`import izero_adapters.<framework>` never fail when a framework or ONNX is
absent. The adapter classes remain constructible and testable via the duck-typed
shims.

The test suite reflects this. Every framework adapter test
(`adapters/tests/test_{langchain,llamaindex,autogen,crewai}.py`) runs green with
**no** frameworks and **no** ONNX installed: a deterministic 64-dim
`_StubEmbedder` (via the `stub_embedder` / `engine` / `mem_engine` fixtures in
`conftest.py`) backs a temp-SQLite `Engine`, and the framework-interface tests
exercise the adapter's mapping logic against duck-typed mock objects
(`adapters/tests/_mocks.py`) rather than pinning to a real framework version.

Each test file adds **one** real-framework integration check that skips cleanly
when the framework is absent:

| Test file | Skip mechanism | Skips when |
|---|---|---|
| `test_langchain.py` | `pytest.importorskip("langchain_core")` | `langchain_core` not installed |
| `test_llamaindex.py` | `pytest.importorskip("llama_index")` | `llama_index` not installed |
| `test_autogen.py` | `if _HAS_AUTOGEN: pytest.skip(...)` | `pyautogen` installed (skips the *absent-path* assertion) |
| `test_crewai.py` | `if _HAS_CREWAI: pytest.skip(...)` | `crewai` installed (skips the *absent-path* assertion) |

So on a bare machine, the LangChain and LlamaIndex real-framework tests are
*skipped* (expected), and the AutoGen/CrewAI absent-path assertions *run* (and
pass); when the frameworks are installed the inverse happens. The adapter
mapping tests always run and always pass.

```bash
.venv/bin/python -m pytest adapters/tests -q
```

The adapters are an integration library: they *use* the Isotope Zero engine via
the stable `MemoryStore` / embedder contract and never modify core storage
schemas or the prototype sources under `prototypes/`. Running the test suite
leaves `prototypes/` completely untouched.

---

## See also

- [`adapters/README.md`](../adapters/README.md) — the adapter package's own
  README (quick start, design, scope & safety).
- [Root `README.md`](../README.md) — project overview, key benchmarks, and the
  MCP / CLI entry points.
- [`architecture.md`](architecture.md) — the 8-phase research evolution
  (float32 BLAS baseline, refuted int8/mmap/1-bit variants, decay + graph,
  shared-memory daemon) and the durable RSS-wall conclusion.
- [`README.md`](../README.md) — the unified `IsotopeZero` client API
  (`remember` / `recall` / `touch` / `consolidate` / `count` / `close`) and quick-start.
