"""The console package may not reach a recomputation or evaluation path.

The same discipline ``tests/test_api_boundary.py`` applies to
``obsidianchain.api``, applied to the application layer, for the same reason
and by the same mechanism: the console is mounted on the production
application, so a handler one call away from ``build_oracle`` could produce a
number no manifest describes.

There is one extra property here that the analytical layer does not need.
The console WRITES, and what it writes must never be an analytical quantity.
It stores identifiers - ``alert_id``, ``run_fingerprint``, ``evidence_id`` -
so a risk score continues to exist in exactly one place: the parquet the
pipeline wrote.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from obsidianchain.api import boundary

CONSOLE_DIR = Path(__file__).resolve().parents[1] / "src" / "obsidianchain" / "console"

#: Same patterns the API boundary test scans for, kept on the test side for
#: the same reason: a module holding these strings would be indistinguishable
#: from one that reads truth.
TRUTH_ACCESS_PATTERNS = (
    "FOR_EVALUATION_ONLY",
    "network_truth",
    "worlds_truth",
    # The synthetic world's quarantined truth: entity identity, origin
    # identity and the behaviour that produced each transaction.
    "world_truth",
    "true_entity_id",
    "true_origin_id",
)

#: Columns that carry an analytical quantity. The console may reference an
#: alert by id; it may not persist what the model said about it, because a
#: second copy of a score is a second thing that can disagree with the
#: artifact.
ANALYTICAL_COLUMNS = (
    "risk_score",
    "severity",
    "shap",
    "contribution",
    "base_value",
    "calibrated",
)


def console_modules() -> list[Path]:
    return sorted(CONSOLE_DIR.rglob("*.py"))


def code_only(source: str) -> str:
    path = Path(__file__).resolve().parent / "test_truth_isolation.py"
    spec = importlib.util.spec_from_file_location("_ti_for_console", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.code_only(source)


def imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_the_console_package_is_not_empty() -> None:
    assert console_modules(), f"no modules found under {CONSOLE_DIR}"


@pytest.mark.parametrize("path", console_modules(), ids=lambda p: p.name)
def test_no_module_imports_a_recomputation_path(path: Path) -> None:
    offenders = sorted(
        name
        for name in imported_names(path)
        for forbidden in boundary.FORBIDDEN_RECOMPUTATION
        if name == forbidden or name.startswith(forbidden + ".")
    )
    assert not offenders, (
        f"{path.name} imports {offenders}. The console records investigator "
        f"state; it must not be able to produce an analytical quantity."
    )


@pytest.mark.parametrize("path", console_modules(), ids=lambda p: p.name)
def test_no_module_imports_an_evaluation_module(path: Path) -> None:
    offenders = sorted(
        name for name in imported_names(path)
        if name.startswith("obsidianchain.eval")
    )
    assert not offenders, f"{path.name} imports evaluation code: {offenders}"


@pytest.mark.parametrize("path", console_modules(), ids=lambda p: p.name)
def test_no_module_references_a_truth_loader(path: Path) -> None:
    executable = code_only(path.read_text(encoding="utf-8"))
    offenders = [m for m in TRUTH_ACCESS_PATTERNS if m in executable]
    assert not offenders, (
        f"{path.name} references truth access: {offenders}."
    )


def test_the_schema_stores_no_analytical_quantity() -> None:
    """The DDL may hold identifiers, never a score.

    This is the structural guarantee that IMMUTABLE ANALYTICAL TRUTH and
    MUTABLE INVESTIGATOR STATE stay separated: if the schema cannot hold a
    risk score, no handler can persist one, and the artifact remains the only
    place a number lives.
    """
    from obsidianchain.console import db

    ddl = " ".join(script for _version, script in db.MIGRATIONS).lower()
    for column in ANALYTICAL_COLUMNS:
        assert column not in ddl, (
            f"the application schema declares {column!r}. Analytical "
            f"quantities live in the pipeline's artifacts and are referenced "
            f"by id; a second copy is a second thing that can disagree."
        )
    # And the identifiers it DOES hold, so this test cannot pass vacuously.
    for identifier in ("alert_id", "run_fingerprint", "sha256"):
        assert identifier in ddl
