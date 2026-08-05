"""Shared pytest fixtures for the Isotope Zero adapter tests.

Every framework adapter test imports from here to get a deterministic,
in-process :class:`Engine` backed by a temp SQLite DB + the stub embedder —
so the full suite runs with ZERO third-party deps (no onnxruntime, no daemon,
no framework packages required for the core path). Framework-interface tests
use the mock objects in :mod:`._mocks` rather than importing the real
frameworks, so they exercise the adapter's mapping logic without pinning to a
framework version.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Make ``izero_adapters`` importable when running tests from a source checkout
# that hasn't been pip-installed (e.g. `pytest adapters/tests` from repo root).
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ADAPTERS_PKG_ROOT = os.path.dirname(_TESTS_DIR)  # .../adapters
if _ADAPTERS_PKG_ROOT not in sys.path:
    sys.path.insert(0, _ADAPTERS_PKG_ROOT)


@pytest.fixture()
def tmp_db_path(tmp_path):
    """A fresh file path for an Isotope Zero SQLite DB (not yet created)."""
    return str(tmp_path / "izero_adapter_test.db")


@pytest.fixture()
def stub_embedder():
    """A deterministic stub embedder (64-dim) for fast, stable tests.

    64 dims gives clear cosine margins: identical→1.0, related→~0.6,
    unrelated→~0.1. Small enough to keep vector math trivial in tests, large
    enough that related/unrelated scores don't collapse together.
    """
    from izero_adapters._engine import _StubEmbedder

    return _StubEmbedder(dim=64)


@pytest.fixture()
def engine(tmp_db_path, stub_embedder):
    """A clean Engine per test: temp DB + stub embedder (no onnx/daemon)."""
    from izero_adapters import get_engine

    return get_engine(db_path=tmp_db_path, embedder=stub_embedder)


@pytest.fixture()
def mem_engine(stub_embedder):
    """An in-memory Engine (fastest; no file IO) for pure-logic tests."""
    from izero_adapters import get_engine

    return get_engine(db_path=":memory:", embedder=stub_embedder)
