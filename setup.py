"""Setup script — emits the editor-plugin bundle as wheel ``data_files``.

The build backend is setuptools (see ``pyproject.toml`` ``[build-system]``);
this file exists for ONE job the declarative ``[tool.setuptools.data-files]``
table cannot do: ship the repo-root ``integrations/izero-plugin/`` tree at the
wheel root with its subdirectory structure intact (``hooks/``, ``skills/<name>/``,
``.claude-plugin/``, etc.). The pyproject ``data-files`` glob
``integrations/izero-plugin/**/*`` also matches the *directories* ``hooks`` and
``skills`` as if they were files, so the build fails with ``can't copy
'.../hooks': doesn't exist or not a regular file``. Walking the tree here and
emitting one ``(dest_dir, [files])`` entry per subdirectory is the standard
setuptools idiom for shipping a whole data tree at a chosen wheel location.

The installer (``src/isotope_zero/integrations/cli.py::_resolve_bundle_path``)
walks ``importlib.resources.files("isotope_zero").parents`` looking for a
sibling ``integrations/izero-plugin/``; shipping the bundle at the wheel root
as ``integrations/izero-plugin/`` is exactly what makes that resolve after a
plain ``pip install isotope-zero`` with no repo checkout.

No new dependencies, no network, no compiled code — pure data files.
"""
from __future__ import annotations

import os
from setuptools import setup

_BUNDLE_ROOT = "integrations/izero-plugin"


def _bundle_data_files() -> list[tuple[str, list[str]]]:
    """Walk ``integrations/izero-plugin/`` and emit one ``(dest, files)`` entry
    per subdirectory, preserving the tree at the wheel root. ``data_files``
    paths are interpreted relative to the install prefix, so
    ``("integrations/izero-plugin/hooks", [...])`` lands at
    ``<prefix>/integrations/izero-plugin/hooks/`` — exactly the sibling tree
    the installer expects. Returns ``[]`` (no-op) when the bundle is absent
    (e.g. a clean sdist built without the MANIFEST.in include)."""
    if not os.path.isdir(_BUNDLE_ROOT):
        return []
    entries: list[tuple[str, list[str]]] = []
    for dirpath, _dirnames, filenames in os.walk(_BUNDLE_ROOT):
        # Skip empty dirs; setuptools needs at least one file per entry.
        if not filenames:
            continue
        # dest is the repo-relative dirpath, i.e. the wheel-root-relative path.
        rel_files = [os.path.join(dirpath, name) for name in filenames]
        entries.append((dirpath, rel_files))
    return entries


setup(data_files=_bundle_data_files())
