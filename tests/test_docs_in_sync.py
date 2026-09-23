"""Generated documentation must match the code it documents."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_feature_catalog_doc_is_current() -> None:
    spec = importlib.util.spec_from_file_location("render", ROOT / "scripts" / "render_feature_catalog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (ROOT / "docs" / "feature_catalog.md").read_text(encoding="utf-8") == mod.render(), \
        "docs/feature_catalog.md is stale: run python scripts/render_feature_catalog.py"
