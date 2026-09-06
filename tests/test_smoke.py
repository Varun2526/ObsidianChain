"""Structural smoke tests: the package and its dependency stack import."""

from __future__ import annotations

import importlib

import pytest

import obsidianchain

SUBPACKAGES = [
    "obsidianchain.io",
    "obsidianchain.cluster",
    "obsidianchain.network",
    "obsidianchain.eval",
    "obsidianchain.cli",
]

# The pinned stack. If any of these fail to import, the vendored wheel
# set is wrong for this image and every later result is suspect.
DEPENDENCIES = [
    "numpy",
    "pandas",
    "pyarrow",
    "duckdb",
    "igraph",
    "sklearn",
    "lightgbm",
    "typer",
]


def test_version() -> None:
    assert obsidianchain.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("name", DEPENDENCIES)
def test_dependency_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_io_subpackage_does_not_shadow_stdlib() -> None:
    """`obsidianchain.io` must not break absolute imports of stdlib io."""
    import io as stdlib_io

    assert hasattr(stdlib_io, "StringIO")
