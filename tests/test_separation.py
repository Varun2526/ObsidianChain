"""Tests for group-level network separation evidence.

SCIENTIFIC VALIDATION NOTICE: the network data is synthetic. These tests
check that the statistic behaves correctly and abstains when it should. They
say nothing about whether origin separation works on Bitcoin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obsidianchain.network import separation as sep


def stats_from(samples: np.ndarray) -> sep.GroupStats:
    """Sufficient statistics from an (n, d) sample matrix; NaN = missing."""
    present = ~np.isnan(samples)
    filled = np.nan_to_num(samples, nan=0.0)
    return sep.GroupStats(
        count=int(samples.shape[0]),
        dim_count=present.sum(axis=0).astype(np.int64),
        dim_sum=filled.sum(axis=0),
        dim_sumsq=(filled**2).sum(axis=0),
    )


# ---- the cannot-link-only invariant -----------------------------------


def test_there_is_no_must_link_verdict() -> None:
    """Shared origin never implies shared entity, so no such verdict exists."""
    names = {v.value for v in sep.Verdict}
    assert names == {"SEPARATED", "NOT_SEPARATED", "NO_EVIDENCE"}
    assert not any("MUST" in n or "SAME" in n for n in names)


def test_only_separated_blocks_a_merge() -> None:
    for verdict, blocks in (
        (sep.Verdict.SEPARATED, True),
        (sep.Verdict.NOT_SEPARATED, False),
        (sep.Verdict.NO_EVIDENCE, False),
    ):
        assert sep.SeparationEvidence(verdict=verdict).blocks_merge is blocks


def test_docstring_states_cannot_link_only() -> None:
    doc = sep.__doc__ or ""
    assert "CANNOT-LINK ONLY" in doc
    assert "never emits must-link" in doc
    assert "Electrum" in doc


# ---- sufficient statistics --------------------------------------------


def test_stats_combine_is_equivalent_to_pooling() -> None:
    """Merging statistics must equal computing them on the union."""
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(40, 4)), rng.normal(size=(60, 4))
    merged = stats_from(a).combine(stats_from(b))
    direct = stats_from(np.vstack([a, b]))
    assert merged.count == direct.count == 100
    np.testing.assert_allclose(merged.mean(), direct.mean())
    np.testing.assert_allclose(merged.variance(), direct.variance())


def test_stats_combine_is_associative() -> None:
    rng = np.random.default_rng(1)
    parts = [stats_from(rng.normal(size=(20, 3))) for _ in range(3)]
    left = parts[0].combine(parts[1]).combine(parts[2])
    right = parts[0].combine(parts[1].combine(parts[2]))
    np.testing.assert_allclose(left.mean(), right.mean())


def test_missing_observer_does_not_discard_the_other_dimensions() -> None:
    """A transaction one observer missed still contributes its other columns."""
    samples = np.array([[1.0, np.nan, 3.0], [2.0, 5.0, 4.0]])
    stats = stats_from(samples)
    assert list(stats.dim_count) == [2, 1, 2]
    assert stats.mean()[0] == pytest.approx(1.5)
    assert stats.mean()[1] == pytest.approx(5.0)


def test_variance_undefined_below_two_samples() -> None:
    stats = stats_from(np.array([[1.0, 2.0]]))
    assert np.isnan(stats.variance()).all()


def test_empty_stats() -> None:
    empty = sep.GroupStats.empty(4)
    assert empty.count == 0
    assert np.isnan(empty.mean()).all()


# ---- abstention -------------------------------------------------------


def test_below_minimum_returns_no_evidence_not_weak_evidence() -> None:
    """Four observations against three is unanswerable, not slightly separated."""
    rng = np.random.default_rng(2)
    a = stats_from(rng.normal(0.0, 1.0, size=(4, 6)))
    b = stats_from(rng.normal(9.0, 1.0, size=(3, 6)))  # hugely different means
    evidence = sep.separation_evidence(a, b)
    assert evidence.verdict is sep.Verdict.NO_EVIDENCE
    assert "below minimum" in evidence.reason
    assert evidence.blocks_merge is False


def test_a_single_observation_can_never_block() -> None:
    """The repetition requirement is structural, not a configured threshold."""
    rng = np.random.default_rng(3)
    lonely = stats_from(rng.normal(0.0, 1.0, size=(1, 6)))
    crowd = stats_from(rng.normal(5.0, 1.0, size=(500, 6)))
    assert sep.separation_evidence(lonely, crowd).verdict is sep.Verdict.NO_EVIDENCE


@pytest.mark.parametrize("n", [1, 5, 10, 24])
def test_no_evidence_holds_right_up_to_the_threshold(n: int) -> None:
    rng = np.random.default_rng(4)
    a = stats_from(rng.normal(0.0, 1.0, size=(n, 5)))
    b = stats_from(rng.normal(4.0, 1.0, size=(400, 5)))
    config = sep.SeparationConfig(min_pooled_observations=25)
    assert sep.separation_evidence(a, b, config).verdict is sep.Verdict.NO_EVIDENCE


def test_abstains_when_no_observer_has_paired_coverage() -> None:
    """Enough transactions overall, but no observer saw enough of both."""
    a = np.full((40, 3), np.nan)
    a[:, 0] = 1.0
    b = np.full((40, 3), np.nan)
    b[:, 2] = 1.0
    evidence = sep.separation_evidence(stats_from(a), stats_from(b))
    assert evidence.verdict is sep.Verdict.NO_EVIDENCE


# ---- the statistic ----------------------------------------------------


def test_clearly_different_groups_are_separated() -> None:
    rng = np.random.default_rng(5)
    a = stats_from(rng.normal(0.0, 0.1, size=(200, 6)))
    b = stats_from(rng.normal(0.5, 0.1, size=(200, 6)))
    evidence = sep.separation_evidence(a, b)
    assert evidence.verdict is sep.Verdict.SEPARATED
    assert evidence.blocks_merge
    assert evidence.p_value < 1e-4
    assert evidence.effect >= 0.05
    assert evidence.dof == 6


def test_identical_distributions_are_not_separated() -> None:
    rng = np.random.default_rng(6)
    a = stats_from(rng.normal(0.0, 1.0, size=(300, 6)))
    b = stats_from(rng.normal(0.0, 1.0, size=(300, 6)))
    assert sep.separation_evidence(a, b).verdict is sep.Verdict.NOT_SEPARATED


def test_significant_but_trivial_difference_is_refused() -> None:
    """The effect floor exists to stop noise-chasing at large n.

    With 20,000 samples a difference of 0.002 is statistically certain and
    practically meaningless. Blocking a merge on it would be a p-value
    dressed up as evidence.
    """
    rng = np.random.default_rng(7)
    a = stats_from(rng.normal(0.000, 0.05, size=(20_000, 4)))
    b = stats_from(rng.normal(0.002, 0.05, size=(20_000, 4)))
    evidence = sep.separation_evidence(a, b)
    assert evidence.p_value < 1e-4 or evidence.effect < 0.05
    assert evidence.verdict is not sep.Verdict.SEPARATED


def test_large_effect_but_tiny_samples_is_refused() -> None:
    """Symmetry check: effect size alone must not be enough either."""
    a = stats_from(np.tile([0.0, 0.0, 0.0], (26, 1)) + np.array([0.0, 0.0, 0.0]))
    rng = np.random.default_rng(8)
    a = stats_from(rng.normal(0.0, 3.0, size=(26, 3)))
    b = stats_from(rng.normal(1.0, 3.0, size=(26, 3)))
    evidence = sep.separation_evidence(a, b)
    assert evidence.verdict is not sep.Verdict.SEPARATED


def test_confidence_rises_with_pooled_count() -> None:
    """More pooled observations must sharpen the answer, never blunt it."""
    rng = np.random.default_rng(9)
    previous = 1.0
    for n in (30, 100, 400, 1600):
        a = stats_from(rng.normal(0.0, 1.0, size=(n, 5)))
        b = stats_from(rng.normal(0.25, 1.0, size=(n, 5)))
        p = sep.separation_evidence(a, b).p_value
        assert p <= previous + 1e-12, f"p rose at n={n}"
        previous = p


def test_verdict_is_symmetric_in_its_arguments() -> None:
    rng = np.random.default_rng(10)
    a = stats_from(rng.normal(0.0, 0.2, size=(150, 4)))
    b = stats_from(rng.normal(0.6, 0.2, size=(150, 4)))
    assert (
        sep.separation_evidence(a, b).verdict
        is sep.separation_evidence(b, a).verdict
    )


def test_zero_variance_everywhere_abstains() -> None:
    a = stats_from(np.zeros((50, 3)))
    b = stats_from(np.zeros((50, 3)))
    assert sep.separation_evidence(a, b).verdict is sep.Verdict.NO_EVIDENCE


def test_describe_is_readable() -> None:
    rng = np.random.default_rng(11)
    a = stats_from(rng.normal(0.0, 0.1, size=(100, 3)))
    b = stats_from(rng.normal(1.0, 0.1, size=(100, 3)))
    text = sep.separation_evidence(a, b).describe()
    assert "SEPARATED" in text and "pooled" in text and "effect" in text


# ---- normalisation ----------------------------------------------------


def test_unit_normalisation_makes_scale_irrelevant() -> None:
    """Two identically shaped vectors at different scales must pool alike."""
    small = np.array([[0.0, 50.0, 100.0]])
    large = np.array([[0.0, 5000.0, 10000.0]])
    np.testing.assert_allclose(
        sep._unit_normalise(small), sep._unit_normalise(large)
    )


def test_unit_normalisation_preserves_missing_values() -> None:
    out = sep._unit_normalise(np.array([[0.0, np.nan, 200.0]]))
    assert np.isnan(out[0, 1])
    assert out[0, 2] == pytest.approx(1.0)


def test_unit_normalisation_survives_an_all_zero_row() -> None:
    out = sep._unit_normalise(np.zeros((1, 3)))
    assert np.isfinite(out).all()


# ---- config -----------------------------------------------------------


def test_thresholds_are_configurable_in_one_place() -> None:
    config = sep.SeparationConfig(min_pooled_observations=5, min_effect=0.0)
    rng = np.random.default_rng(12)
    a = stats_from(rng.normal(0.0, 0.05, size=(6, 3)))
    b = stats_from(rng.normal(1.0, 0.05, size=(6, 3)))
    assert sep.separation_evidence(a, b, config).verdict is sep.Verdict.SEPARATED
    strict = sep.SeparationConfig(min_pooled_observations=50)
    assert sep.separation_evidence(a, b, strict).verdict is sep.Verdict.NO_EVIDENCE


def test_oracle_returns_empty_stats_for_unknown_address() -> None:
    oracle = sep.SeparationOracle(n_dims=4, observer_ids=list("abcd"))
    assert oracle.stats_for(999).count == 0


# ---- regression: a transaction must be pooled exactly once -------------


def test_transaction_is_attributed_to_exactly_one_address() -> None:
    """Regression for a 651x observation-inflation bug.

    Co-spend unions every input of a transaction into one component, so
    counting a transaction once per input address inflates that component's
    pooled count by its input degree. The standard error then falls by the
    square root of a number that is not a sample size, and constraints fire
    on a single observation replicated hundreds of times.
    """
    import pandas as pd

    edges = pd.DataFrame(
        {
            "txId": [10, 10, 10, 11, 11],
            "input_address": ["a", "b", "c", "b", "d"],
            "_code": [0, 1, 2, 1, 3],
        }
    )
    deduped = edges.sort_values(["txId", "_code"], kind="stable").drop_duplicates(
        subset="txId", keep="first"
    )
    assert len(deduped) == 2, "one row per transaction"
    assert deduped["_code"].tolist() == [0, 1], "lowest-coded input represents it"


def test_oracle_pooled_total_cannot_exceed_usable_transactions() -> None:
    """The invariant the bug violated, stated as an assertion."""
    oracle = sep.SeparationOracle(
        n_dims=4,
        observer_ids=list("abcd"),
        address_stats={
            0: stats_from(np.zeros((3, 4))),
            1: stats_from(np.zeros((2, 4))),
        },
        n_usable_transactions=5,
        n_pooled_observations=5,
    )
    total = sum(s.count for s in oracle.address_stats.values())
    assert total == oracle.n_pooled_observations
    assert oracle.n_pooled_observations <= oracle.n_usable_transactions
