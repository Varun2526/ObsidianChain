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

#: Modules that participate in producing a constraint, a similarity, a merge
#: decision or an arrival vector. None may reference ground truth.
INFERENCE_MODULES = [
    "network/separation.py",
    "network/arrivals.py",
    "cluster/constrained.py",
    "cluster/pipeline.py",
    "cluster/unionfind.py",
    "cluster/replay.py",
    "cluster/index.py",
    "io/elliptic.py",
]

#: Modules allowed to read truth: the evaluation harnesses, plus boundary
#: (which defines the accessor) and the generators (which create it).
EVALUATION_MODULES = [
    "eval/phase33.py",
    "eval/world_diagnostics.py",
    "eval/purity.py",
    "eval/compare.py",
    "eval/entity_resolution.py",
    "eval/evolution.py",
    "network/audit.py",
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
    path = SRC / relative
    return path.read_text(encoding="utf-8") if path.is_file() else ""


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


@pytest.mark.parametrize("module", INFERENCE_MODULES)
def test_inference_module_never_reads_ground_truth(module: str) -> None:
    source = read(module)
    if not source:
        pytest.skip(f"{module} not present")
    executable = code_only(source)
    offenders = [marker for marker in TRUTH_MARKERS if marker in executable]
    assert not offenders, (
        f"{module} references ground truth in executable code: {offenders}. "
        f"An inference result computed with truth in scope is invalid."
    )


@pytest.mark.parametrize("module", INFERENCE_MODULES)
def test_inference_module_never_imports_the_truth_accessor(module: str) -> None:
    source = read(module)
    if not source:
        pytest.skip(f"{module} not present")
    assert "FOR_EVALUATION_ONLY" not in code_only(source), module


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
    """No module may open a truth file directly, bypassing the accessor."""
    for module in INFERENCE_MODULES + EVALUATION_MODULES:
        source = code_only(read(module))
        if not source:
            continue
        for bad in ('read_csv("/data/processed/network_truth',
                    "read_csv('/data/processed/network_truth"):
            assert bad not in source, f"{module} bypasses boundary"


def test_evaluation_modules_may_read_truth() -> None:
    """Sanity: the split is real, not vacuous - somebody does read truth."""
    readers = [
        module for module in EVALUATION_MODULES
        if any(marker in code_only(read(module)) for marker in TRUTH_MARKERS)
    ]
    assert readers, "no evaluation module reads truth; the test is vacuous"
