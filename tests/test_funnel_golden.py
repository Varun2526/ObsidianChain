"""Section 14: the thirteen pre-existing funnel columns must never move.

Why this file exists rather than the publish gate alone
-------------------------------------------------------
The gate in ``cli._verify_funnel_artifact`` compares a fresh artifact
against the published one and refuses on any change to the thirteen. That
check is real and it runs on every regeneration - but ``os.replace`` then
overwrites the very artifact it compared against, so it consumes its own
baseline. After the Phase 5.3-B publish, "the thirteen never moved" rested
entirely on a check whose evidence no longer existed and which no test
exercised.

``tests/golden/evidence_funnel_probe_columns.json`` is that evidence, made
durable. It was captured from the artifact as it stood BEFORE the schema /2
regeneration, and it survives every regeneration that follows.

What the digests prove
----------------------
Each column is hashed over its exact bytes in row order, so a digest is an
element-wise assertion over all 253,429 values of that column - one flipped
value changes it. Row order is additionally pinned by its own digest over
``edge_index``, so a permutation with identical contents fails, and fails
with a message that says which of the two went wrong.

One column, ``p_value``, is compared at twelve significant digits instead of
bit-exactly. That is not a softened check, it is a portable one: the reason
is measured and recorded in ``funnel_golden.TOLERANT_COLUMNS``, and the
tests below prove the tolerance still catches a real change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain.api import artifacts
from obsidianchain.eval import evidence_funnel as funnel
from obsidianchain.eval import funnel_golden as golden

DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))
BASELINE = Path(__file__).resolve().parent / "golden" / (
    "evidence_funnel_probe_columns.json"
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return golden.read_baseline(BASELINE)


@pytest.fixture(scope="module")
def published() -> pd.DataFrame:
    """The published artifact's thirteen columns.

    A FAILURE, never a skip, when the artifact is absent. This is the only
    durable check that the thirteen columns survived schema /2; a suite that
    reports green without having run it is reporting on nothing.
    """
    path = DATA_ROOT / artifacts.EVIDENCE_FUNNEL
    assert path.is_file(), (
        f"{path} is absent, so the thirteen pre-existing columns have not "
        f"been compared against the golden baseline. This must not skip: "
        f"generate the artifact with 'make run ARGS=\"evidence-funnel\"'."
    )
    return pd.read_parquet(path, columns=list(golden.PROBE_COLUMNS))


# ---- the baseline itself -----------------------------------------------


def test_the_baseline_is_present_and_readable(baseline) -> None:
    assert baseline["schema"] == golden.GOLDEN_SCHEMA
    assert baseline["n_rows"] == 253_429
    assert len(baseline["columns"]) == 13


def test_the_baseline_covers_exactly_the_pre_existing_columns(baseline) -> None:
    """"The thirteen" must not be able to drift as a definition.

    ``RECORD_COLUMNS`` is the funnel's own list of the columns that predate
    schema /2, so the baseline is checked against it rather than against a
    second hand-maintained copy.
    """
    assert baseline["column_order"] == list(funnel.RECORD_COLUMNS)
    assert list(golden.PROBE_COLUMNS) == list(funnel.RECORD_COLUMNS)


def test_the_production_columns_are_not_in_the_baseline(baseline) -> None:
    """The seven added columns are expected to change; they are not pinned."""
    from obsidianchain import evidence_contract as contract

    for column in contract.PRODUCTION_COLUMNS:
        assert column not in baseline["columns"]


def test_only_p_value_is_compared_tolerantly(baseline) -> None:
    """A tolerance that spread to other columns would weaken the check."""
    tolerant = [
        name for name, block in baseline["columns"].items()
        if block["comparison"] != "bit-exact"
    ]
    assert tolerant == ["p_value"]
    assert golden.TOLERANT_COLUMNS == ("p_value",)


# ---- the published artifact against the baseline -----------------------


def test_the_published_thirteen_columns_match_the_baseline(
    published, baseline
) -> None:
    """The headline assertion: 253,429 x 13 values and the row order."""
    problems = golden.compare(published, baseline)
    assert not problems, "the pre-existing columns moved:\n" + "\n".join(problems)


def test_the_published_row_order_matches_the_baseline(
    published, baseline
) -> None:
    """Stated separately because it fails differently from a value change."""
    assert golden.column_digest(published["edge_index"], "int64") == (
        baseline["row_order_digest"]
    )


def test_the_published_artifact_has_the_frozen_row_count(published) -> None:
    assert len(published) == 253_429


# ---- the comparison is not vacuous ------------------------------------


def test_a_single_changed_value_is_detected(published, baseline) -> None:
    """One value in a quarter of a million rows.

    Without this, every comparison above could be passing because the
    digests are insensitive rather than because the values are intact.
    """
    tampered = published.copy()
    tampered.loc[tampered.index[123_456], "pooled_a"] = (
        int(tampered.loc[tampered.index[123_456], "pooled_a"]) + 1
    )
    problems = golden.compare(tampered, baseline)
    assert any("pooled_a" in p for p in problems), problems


def test_a_permuted_row_order_is_detected(published, baseline) -> None:
    """Identical multiset, different order. Must not pass."""
    permuted = published.iloc[::-1].reset_index(drop=True)
    problems = golden.compare(permuted, baseline)
    assert any("ROW ORDER changed" in p for p in problems), problems


def test_a_swapped_pair_of_rows_is_detected(published, baseline) -> None:
    """The subtler permutation: two rows exchanged, everything else in place."""
    swapped = published.copy()
    first, second = swapped.index[10], swapped.index[11]
    swapped.loc[[first, second]] = swapped.loc[[second, first]].to_numpy()
    problems = golden.compare(swapped, baseline)
    assert problems, "a two-row swap passed the comparison"


def test_a_dropped_row_is_detected(published, baseline) -> None:
    problems = golden.compare(published.iloc[:-1], baseline)
    assert any("row count" in p for p in problems), problems


def test_a_nan_turned_into_a_zero_is_detected(published, baseline) -> None:
    """"Not computed" becoming "computed, and the answer was zero".

    251,914 rows carry NaN in chi2. Substituting zero is the single most
    plausible corruption of this artifact and the most misleading, because
    a chi-square of zero reads as perfect agreement.
    """
    tampered = published.copy()
    tampered["chi2"] = tampered["chi2"].fillna(0.0)
    problems = golden.compare(tampered, baseline)
    assert any("chi2" in p for p in problems), problems


def test_the_tolerant_column_still_catches_a_real_change(
    published, baseline
) -> None:
    """Twelve significant digits is a tolerance, not an exemption.

    A p-value moved by 1e-9 relative - still far smaller than any change
    that could alter a verdict - must fail. If it did not, the tolerance
    would be wide enough to hide a methodology change.
    """
    tampered = published.copy()
    values = tampered["p_value"].to_numpy("float64").copy()
    finite = np.flatnonzero(~np.isnan(values))
    assert finite.size, "no finite p_value to perturb; the test is vacuous"
    values[finite[0]] *= 1.0 + 1e-9
    tampered["p_value"] = values
    problems = golden.compare(tampered, baseline)
    assert any("p_value" in p for p in problems), problems


def test_the_tolerant_column_absorbs_one_ulp_of_libm_noise(
    published, baseline
) -> None:
    """And the measured cross-architecture noise must NOT fail.

    Re-deriving on arm64 moves 608 of the 1,515 finite p-values by up to
    5.55e-16 absolute, because ``scipy.stats.chi2.sf`` follows the platform
    libm. A check that fired on that would be asserting the architecture,
    and would be deleted by the third engineer who hit it.
    """
    nudged = published.copy()
    original = published["p_value"].to_numpy("float64")
    values = original.copy()
    # Nonzero, because a p-value that UNDERFLOWED to exactly 0.0 is not
    # subject to this noise - zero is zero on every platform, and
    # nextafter(0.0) is a denormal, a relative change of infinity rather
    # than of one ULP. Perturbing it would test something that cannot
    # happen and would misrepresent the tolerance as looser than it is.
    movable = np.flatnonzero(~np.isnan(values) & (values != 0.0))
    assert movable.size, "no perturbable p_value; the test is vacuous"
    values[movable] = np.nextafter(values[movable], np.inf)
    nudged["p_value"] = values

    # The bytes really did change, on every one of them...
    assert not np.array_equal(values[movable], original[movable])
    # ...and the comparison correctly does not care.
    assert not golden.compare(nudged, baseline)


def test_a_p_value_that_underflowed_to_zero_is_held_exactly(
    published, baseline
) -> None:
    """Zero is not noise-prone, so it is not given any slack.

    Some p-values underflow to exactly 0.0 against an enormous chi-square.
    That is deterministic across platforms, and turning one into a denormal
    - or into anything else - must fail.
    """
    values = published["p_value"].to_numpy("float64")
    zeros = np.flatnonzero(values == 0.0)
    assert zeros.size, "no zero p_value in the artifact; nothing to hold"

    tampered = published.copy()
    nudged = values.copy()
    nudged[zeros[0]] = np.nextafter(0.0, np.inf)
    tampered["p_value"] = nudged
    assert any("p_value" in p for p in golden.compare(tampered, baseline))


# ---- element-wise comparison of two frames in hand --------------------


def test_compare_values_reports_identical_frames_as_identical(published) -> None:
    assert golden.compare_values(published, published) == []


def test_compare_values_counts_the_differing_rows(published) -> None:
    tampered = published.copy()
    tampered.loc[tampered.index[:5], "node_a"] = -1
    problems = golden.compare_values(tampered, published)
    assert any("node_a" in p and "5 rows differ" in p for p in problems), problems


def test_compare_values_detects_a_moved_nan(published) -> None:
    """NaN position is information: which rows were never computed."""
    tampered = published.copy()
    column = tampered["chi2"].to_numpy("float64").copy()
    nans = np.flatnonzero(np.isnan(column))
    finite = np.flatnonzero(~np.isnan(column))
    assert nans.size and finite.size
    column[nans[0]], column[finite[0]] = column[finite[0]], np.nan
    tampered["chi2"] = column
    problems = golden.compare_values(tampered, published)
    assert any("NaN positions differ" in p for p in problems), problems


# ---- the baseline file is committed in a readable form ----------------


def test_the_baseline_is_plain_sorted_json() -> None:
    """A reviewer has to be able to read a diff of it."""
    text = BASELINE.read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert text == json.dumps(parsed, indent=2, sort_keys=True) + "\n"
