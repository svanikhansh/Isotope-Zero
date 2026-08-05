"""Isotope Zero adapter test suite.

Marked as a package so test modules can import shared fixtures/mocks via
``from tests._mocks import ...`` when ``adapters/`` is on ``sys.path`` (the
``conftest.py`` here inserts it). With this ``__init__.py`` in place, pytest
collects the suite cleanly from the repo root:

    .venv/bin/python -m pytest adapters/tests/ -q
"""
