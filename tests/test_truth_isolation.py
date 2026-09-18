"""Assert ground truth is read ONLY in evaluation modules, never in inference.

This is a source-level test, deliberately. Runtime tests can only catch the
paths they happen to exercise; scanning the source catches a truth read on a
branch nobody thought to test. The failure mode is somebody debugging
inference, joining the truth table to see what is happening, watching the
numbers improve, and never tracing the improvement back to the join.
"""

from __future__ import annotations

import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "obsidianchain"

#: Modules explicitly excused from the truth ban, each for a stated reason.
#: Everything else is checked by ENUMERATION, not by a list - see
#: ``inference_modules()``. That inversion is the point: before it, a module
#: nobody remembered to add was silently exempt, and the Phase 4 audit (§6.5)
#: found that a new package would default to unchecked. Now a new package is
#: inference-by-default and has to be excused deliberately.
TRUTH_ACCESSOR_MODULES = {
    # Defines load_ground_truth_FOR_EVALUATION_ONLY. Somebody has to.
    "network/boundary.py",
}

GENERATOR_MODULES = {
    # These CREATE truth - they invent the answer, then write it to a
    # quarantined directory. They cannot avoid naming it.
    "network/synthetic.py",
    "network/worlds.py",
    "network/reach_stress.py",
    "demo/scenarios.py",
    # The coherent synthetic world. It invents the entity, the origin and the
    # behaviour behind every transaction, then writes them to world_truth/.
    # Like network/synthetic.py it cannot avoid naming what it creates; what
    # matters is that no INFERENCE module reads that directory, which the
    # rest of this file checks.
    "world/generate.py",
}

DISPATCH_MODULES = {
    # The CLI wires up the evaluation commands, so it names their accessors.
    "cli.py",
}

#: Anything under this prefix may read truth: scoring happens after inference
#: has finished. Derived from the path rather than listed, so a new evaluation
#: module does not need a test edit - but a new module ANYWHERE ELSE does.
EVALUATION_PREFIX = "eval/"

#: Evaluation harnesses that live outside eval/ by historical accident.
EVALUATION_MODULES = {
    "network/audit.py",
}

#: Everything excused, for whatever reason.
EXCUSED = (
    TRUTH_ACCESSOR_MODULES
    | GENERATOR_MODULES
    | DISPATCH_MODULES
    | EVALUATION_MODULES
)


def all_modules() -> list[str]:
    """Every module in the package, as a path relative to the package root."""
    return sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if path.name != "__init__.py"
    )


def inference_modules() -> list[str]:
    """Every module that must be truth-free, computed rather than listed.

    A module is inference until proven otherwise. That is the whole
    mechanism: ``api/`` and any future package are covered the moment they
    exist, with no test edit and no chance of being forgotten.
    """
    return [
        module
        for module in all_modules()
        if not module.startswith(EVALUATION_PREFIX) and module not in EXCUSED
    ]


TRUTH_MARKERS = (
    "load_ground_truth_FOR_EVALUATION_ONLY",
    "network_truth",
    "worlds_truth",
    "true_origin_id",
    "true_entity_id",
    "ground_truth",
)


def read(relative: str) -> str:
    """Source of one module. Raises rather than skipping: under enumeration
    every path comes from the filesystem, so a missing file is a bug here."""
    return (SRC / relative).read_text(encoding="utf-8")


def code_only(source: str) -> str:
    """Strip comments and docstring-ish lines.

    A module may legitimately *discuss* ground truth in prose - separation.py
    explains at length that it must not read it. Only executable references
    matter.
    """
    lines = []
    in_doc = False
    for raw in source.split("\n"):
        stripped = raw.strip()
        if stripped.count('"""') == 1:
            in_doc = not in_doc
            continue
        if in_doc or stripped.startswith("#") or stripped.startswith('"""'):
            continue
        lines.append(raw.split("#", 1)[0])
    return "\n".join(lines)


@pytest.mark.parametrize("module", inference_modules())
def test_inference_module_never_reads_ground_truth(module: str) -> None:
    executable = code_only(read(module))
    offenders = [marker for marker in TRUTH_MARKERS if marker in executable]
    assert not offenders, (
        f"{module} references ground truth in executable code: {offenders}. "
        f"An inference result computed with truth in scope is invalid. If "
        f"this module is genuinely an evaluation harness, add it to "
        f"EVALUATION_MODULES with a reason - do not widen TRUTH_MARKERS."
    )


@pytest.mark.parametrize("module", inference_modules())
def test_inference_module_never_imports_the_truth_accessor(module: str) -> None:
    assert "FOR_EVALUATION_ONLY" not in code_only(read(module)), module


def test_the_enumeration_covers_every_module() -> None:
    """No module may fall outside both the checked set and the excused set.

    This is what makes the inversion airtight. Without it, a path-matching
    slip could quietly drop a module from both sides.
    """
    checked = set(inference_modules())
    evaluation = {m for m in all_modules() if m.startswith(EVALUATION_PREFIX)}
    accounted = checked | evaluation | EXCUSED
    missing = set(all_modules()) - accounted
    assert not missing, f"{sorted(missing)} are neither checked nor excused"


def test_every_excused_module_actually_exists() -> None:
    """A stale excuse is a hole: the module it named may have been renamed."""
    for module in sorted(EXCUSED):
        assert (SRC / module).is_file(), (
            f"{module} is excused from the truth ban but does not exist; "
            f"remove the excuse rather than leaving it to cover a future file"
        )


def test_the_excuse_list_is_not_load_bearing_for_ordinary_modules() -> None:
    """Sanity: the checked set is the majority, not a rump.

    If an excuse ever grew to cover most of the package the test would pass
    while checking almost nothing.
    """
    checked = inference_modules()
    assert len(checked) > len(EXCUSED), (
        f"only {len(checked)} modules are checked against {len(EXCUSED)} "
        f"excused; the ban has stopped meaning anything"
    )


def test_a_new_package_is_checked_by_default() -> None:
    """The property the inversion exists to guarantee.

    Simulated rather than asserted about today's tree: a module at a path
    nobody has listed must land in the checked set.
    """
    hypothetical = "api/routes.py"
    assert hypothetical not in EXCUSED
    assert not hypothetical.startswith(EVALUATION_PREFIX)
    would_be_checked = (
        not hypothetical.startswith(EVALUATION_PREFIX)
        and hypothetical not in EXCUSED
    )
    assert would_be_checked, "a new package would escape the truth ban"


def test_the_truth_accessor_exists_and_is_conspicuously_named() -> None:
    from obsidianchain.network import boundary

    assert hasattr(boundary, "load_ground_truth_FOR_EVALUATION_ONLY")


#: Every truth read must go through an accessor whose name shouts, so that a
#: call from an inference path is obvious in review and in a grep. There are
#: now two truth artifacts - the origin map and the address-to-entity map -
#: so the guarantee is enforced on the naming convention rather than on one
#: function name.
TRUTH_ACCESSOR_MARKER = "FOR_EVALUATION_ONLY"


def test_phase33_reads_truth_only_through_a_shouted_accessor() -> None:
    """Truth must never be read by a bare file open in the scorer.

    Regression: the decision-level rewrite briefly loaded
    entity_assignment.csv with a plain pd.read_csv, bypassing the convention.
    """
    source = read("eval/phase33.py")
    executable = code_only(source)
    assert TRUTH_ACCESSOR_MARKER in executable

    scoring_start = source.index("def _score_against_truth")
    scorer = source[scoring_start:]
    assert TRUTH_ACCESSOR_MARKER in code_only(scorer), (
        "the scorer must obtain truth through a shouted accessor"
    )
    assert "worlds_truth" not in code_only(scorer), (
        "the scorer must not build a truth path itself"
    )


def test_phase33_does_not_read_truth_before_inference() -> None:
    source = read("eval/phase33.py")
    scoring_start = source.index("def _score_against_truth")
    run_body = source[source.index("def run_regime"):scoring_start]
    executable = code_only(run_body)
    assert TRUTH_ACCESSOR_MARKER not in executable, (
        "run_regime must not read truth itself"
    )
    assert "worlds_truth" not in executable


def test_boundary_is_the_only_truth_path() -> None:
    """No module may open a truth file directly, bypassing the accessor.

    Checked over EVERY module now, not over two hand-kept lists.
    """
    for module in all_modules():
        source = code_only(read(module))
        for bad in ('read_csv("/data/processed/network_truth',
                    "read_csv('/data/processed/network_truth"):
            assert bad not in source, f"{module} bypasses boundary"


def test_the_split_is_real_not_vacuous() -> None:
    """Somebody does read truth, or the ban is checking an empty set."""
    readers = [
        module for module in all_modules()
        if any(marker in code_only(read(module)) for marker in TRUTH_MARKERS)
    ]
    assert readers, "nothing reads truth; the isolation test is vacuous"
    # And every one of them must be excused or under eval/ - which is the
    # same claim the enumeration makes, verified from the other direction.
    for module in readers:
        assert module in EXCUSED or module.startswith(EVALUATION_PREFIX), (
            f"{module} reads truth but is in the checked set"
        )
