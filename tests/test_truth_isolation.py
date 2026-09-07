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


def test_phase33_reads_truth_only_after_inference() -> None:
    """The scoring function must be separate from the inference call.

    Structural check: run_regime builds the oracle and clusters before
    _score_against_truth is reachable, and the truth accessor appears only
    inside the scoring function.
    """
    source = read("eval/phase33.py")
    executable = code_only(source)
    assert "load_ground_truth_FOR_EVALUATION_ONLY" in executable
    scoring_start = source.index("def _score_against_truth")
    accessor_at = source.index(
        "load_ground_truth_FOR_EVALUATION_ONLY", scoring_start
    )
    assert accessor_at > scoring_start, "truth must be read inside scoring"

    run_body = source[source.index("def run_regime"):scoring_start]
    assert "load_ground_truth" not in code_only(run_body), (
        "run_regime must not read truth itself"
    )


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
