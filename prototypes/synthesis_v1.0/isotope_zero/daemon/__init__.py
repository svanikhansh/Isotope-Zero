"""Shared-memory embedding daemon for isotope_zero (Phase 7A).

The daemon (`isotope_zero.daemon.server`) loads the ONNX embedding model once
and serves embedding requests over a Unix domain socket, returning vectors via
POSIX shared memory. Client processes use `isotope_zero.daemon.client.
DaemonClient`, a duck-typed drop-in for `EmbeddingEngine` that never imports
onnxruntime / tokenizers / numpy — so a client's RSS never loads the ~360MB
onnxruntime shared library.

This package `__init__` intentionally imports NOTHING: importing
``isotope_zero.daemon`` must be a no-op so the client's RSS story is preserved
even if a stray ``import isotope_zero.daemon`` is added.
"""
