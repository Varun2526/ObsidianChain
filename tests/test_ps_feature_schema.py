"""The PS feature schema must not contain fabricated or redundant columns.

The v1 schema carried nine features that measured nothing, all traceable to
one line in the dataset builder that synthesised per-input and per-output
amounts by dividing the total evenly. The source data has no per-output
values - only the min/max/mean/total summary - so the fix is not a better
fabrication. It is to compute only what the summary genuinely supports and
to drop the rest.

These tests pin the corrected schema. They are deliberately written against
the SCHEMA rather than a generated file, so they fail the moment a
fabricated column is reintroduced, before any dataset is rebuilt.
"""

from __future__ import annotations

import pytest

from obsidianchain.ml import diagnostics
from obsidianchain.pipeline import features_ps


#: Columns whose values could only ever come from fabricated per-output
#: amounts. The source summary cannot support any of them.
FABRICATED = (
    "input_amount_std",
    "output_amount_std",
    "input_amount_max",
    "output_amount_max",
    "equal_output_count",
    "output_entropy",
)

#: Columns that restate another column under a second name.
REDUNDANT = ("in_degree_asof_t", "out_degree_asof_t")


def test_no_fabricated_column_is_in_the_schema() -> None:
    present = [c for c in FABRICATED if c in features_ps.CORE_PS_FEATURE_COLUMNS]
    assert not present, (
        f"{present} can only be computed from synthesised per-output amounts; "
        f"the source data carries a min/max/mean summary and nothing finer"
    )


def test_no_redundant_degree_column_is_in_the_schema() -> None:
    present = [c for c in REDUNDANT if c in features_ps.CORE_PS_FEATURE_COLUMNS]
    assert not present, (
        f"{present} are identical to n_recv_asof_t / n_sent_asof_t; a "
        f"duplicate splits its own importance with its twin"
    )


def test_the_observable_spread_features_replace_the_fabricated_ones() -> None:
    """What the summary DOES support: how uniform the values are.

    ``(max - min) / mean`` is computable from the real columns and is the
    signal the fabricated std was pretending to carry.
    """
    for column in ("output_spread", "input_spread"):
        assert column in features_ps.CORE_PS_FEATURE_COLUMNS


def test_the_peeling_flag_is_capable_of_firing() -> None:
    """v1's flag required out_max >= 0.8*total_out with two equal outputs,
    i.e. 0.5 >= 0.8. It was constant zero across all 262,433 rows."""
    assert features_ps.is_peeling_shape(
        n_in=1, n_out=2, out_min=0.05, out_max=4.9, out_mean=2.475
    ) is True


def test_the_peeling_flag_rejects_an_even_split() -> None:
    assert features_ps.is_peeling_shape(
        n_in=1, n_out=2, out_min=2.0, out_max=2.0, out_mean=2.0
    ) is False


def test_the_mixing_flag_is_not_a_bare_cardinality_rule() -> None:
    """v1's flag reduced to input_count>=3 AND output_count>=3 and fired on
    34% of holdout addresses. Cardinality alone must not be enough."""
    assert features_ps.is_mixing_shape(
        n_in=8, n_out=8, out_min=1.0, out_max=9.0, out_mean=5.0,
        in_min=1.0, in_max=9.0, in_mean=5.0,
    ) is False


def test_the_mixing_flag_fires_on_the_real_structure() -> None:
    """Uniform outputs, varied inputs, enough participants."""
    assert features_ps.is_mixing_shape(
        n_in=8, n_out=8, out_min=0.1, out_max=0.1, out_mean=0.1,
        in_min=0.31, in_max=3.05, in_mean=1.28,
    ) is True


def test_the_mixing_flag_rejects_a_uniform_payout() -> None:
    """Equal outputs but one payer: a benign shape v1 could not separate."""
    assert features_ps.is_mixing_shape(
        n_in=1, n_out=6, out_min=0.25, out_max=0.25, out_mean=0.25,
        in_min=1.5, in_max=1.5, in_mean=1.5,
    ) is False


def test_the_schema_version_records_the_break() -> None:
    """v1 datasets and v2 datasets are not comparable and must not share a
    version string."""
    assert features_ps.PS_FEATURE_SCHEMA_VERSION == "ps_native_features/4"


#: Removed in v3 as restatements of a sibling (Spearman >= 0.995 on the
#: development data, research/autoresearch_2026_09_23/17_dataset_metrics_audit.md).
V3_REDUNDANT = (
    "total_output_amount",
    "tx_velocity_per_hour",
    "btc_sent_total_asof_t",
    "mean_fee_ratio_asof_t",
)


def test_the_v3_redundant_columns_stay_removed() -> None:
    present = [c for c in V3_REDUNDANT if c in features_ps.CORE_PS_FEATURE_COLUMNS]
    assert not present


def test_the_role_group_is_part_of_the_core_schema() -> None:
    assert set(features_ps.GROUP_F_ROLE) <= set(features_ps.CORE_PS_FEATURE_COLUMNS)


def test_the_upstream_group_is_part_of_the_core_schema() -> None:
    """Admitted in /4 after exp20 on causally ordered data."""
    assert set(features_ps.GROUP_G_UPSTREAM) <= set(features_ps.CORE_PS_FEATURE_COLUMNS)


#: The one v1 degenerate column that v2 KEEPS rather than drops. It was
#: degenerate because its definition was unsatisfiable, not because the
#: quantity is unmeasurable - so it is redefined instead, and
#: test_the_peeling_flag_is_capable_of_firing is what guarantees the fix.
REDEFINED_NOT_REMOVED = {"is_peeling_candidate"}


def test_every_v1_degeneracy_is_removed_or_redefined() -> None:
    """The schema diff, stated exactly.

    Eight of the nine flagged columns are gone. The ninth is kept under a
    definition that can actually fire, which the dedicated test above pins -
    dropping it would have lost a real capability along with the defect.
    """
    still_present = {
        c for c in diagnostics.KNOWN_DEGENERATE_PS_NATIVE
        if c in features_ps.CORE_PS_FEATURE_COLUMNS
    }
    assert still_present == REDEFINED_NOT_REMOVED, (
        f"unexpected v1 degenerate columns survive into v2: "
        f"{still_present - REDEFINED_NOT_REMOVED}"
    )


def test_the_redefined_flag_no_longer_matches_its_v1_definition() -> None:
    """v1 and v2 must disagree on at least one shape, or nothing changed."""
    even_split = dict(n_in=1, n_out=2, out_min=2.0, out_max=2.0, out_mean=2.0)
    dominant = dict(n_in=1, n_out=2, out_min=0.05, out_max=4.9, out_mean=2.475)
    assert features_ps.is_peeling_shape(**even_split) is False
    assert features_ps.is_peeling_shape(**dominant) is True
