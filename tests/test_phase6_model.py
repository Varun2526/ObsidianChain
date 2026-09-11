"""Phase 6 modelling: severity selection, metrics, cluster aggregation.

The severity procedure gets the most attention here because it is the part a
reader is most likely to assume is simple. It is not: precision is not
monotone in the threshold, so the obvious implementation - "the first
threshold where validation precision crosses the target" - picks a noisy
local maximum that will not survive on new data. SPEC 6.3.1 requires the
monotone-suffix form instead, and the tests below fail the obvious version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obsidianchain.ml import experiment, metrics, model


# ---- severity thresholds, SPEC 6.3.1 -----------------------------------


def test_a_band_takes_the_lowest_threshold_meeting_the_target() -> None:
    """Step 4: minimum feasible threshold, hence maximum recall.

    With 100 positives ranked first, the band does NOT stop at 100. It keeps
    descending while precision stays at or above the target, reaching 111
    (100/111 = 0.9009). Stopping at the pure-positive prefix would be a
    higher-precision band with less recall, which is the opposite of what
    step 4 asks for.
    """
    scores = np.linspace(1.0, 0.0, 200)
    y = (np.arange(200) < 100).astype(int)  # top 100 are all positive
    band = model._select_one("TEST", 0.90, scores, y)
    assert band.populated
    assert band.support == 111
    assert band.precision >= 0.90
    # One more row would break the target, which is what makes it the minimum.
    assert 100 / 112 < 0.90


def test_the_monotone_suffix_rule_rejects_a_noisy_crossing() -> None:
    """The heart of the procedure.

    Precision here dips below the target part-way down and recovers. A "first
    crossing" rule would accept the recovered point; the suffix rule refuses
    it, because a threshold is only usable if EVERY higher threshold also
    meets the target.
    """
    # ranks: 40 positives, then 40 negatives, then 40 positives
    y = np.array([1] * 40 + [0] * 40 + [1] * 40)
    scores = np.linspace(1.0, 0.0, len(y))
    band = model._select_one("TEST", 0.90, scores, y)
    # The suffix rule can only accept the first 40; adding any of the 40
    # negatives drops running precision below 0.90 permanently.
    assert band.support <= 40


def test_a_band_with_too_little_support_is_unpopulated() -> None:
    """Step 3: a band defined on a handful of samples is not a band."""
    y = np.array([1] * 10 + [0] * 190)
    scores = np.linspace(1.0, 0.0, len(y))
    band = model._select_one("TEST", 0.90, scores, y)
    assert not band.populated
    assert band.threshold is model.UNPOPULATED


def test_an_unreachable_target_leaves_the_band_unpopulated() -> None:
    """Step 5: never filled by lowering the target."""
    y = np.zeros(300, dtype=int)
    y[::10] = 1  # 10% positives everywhere, so 0.90 is unreachable
    scores = np.linspace(1.0, 0.0, len(y))
    band = model._select_one("TEST", 0.90, scores, y)
    assert not band.populated


def test_bands_are_ordered_critical_high_medium() -> None:
    rng = np.random.default_rng(3)
    scores = rng.uniform(size=4000)
    y = (rng.uniform(size=4000) < scores**3).astype(int)
    bands = model.select_bands(scores, y)
    model.assert_band_order(bands)
    populated = [b for b in bands if b.populated]
    thresholds = [b.threshold for b in populated]
    assert thresholds == sorted(thresholds, reverse=True)


def test_inverted_bands_fail_loudly() -> None:
    """A guarantee nobody checks is a hope."""
    bands = [
        model.Band("CRITICAL", 0.9, 0.10, 100, 0.95),
        model.Band("HIGH", 0.75, 0.80, 100, 0.80),
    ]
    with pytest.raises(RuntimeError, match="inverted"):
        model.assert_band_order(bands)


def test_severity_assignment_is_top_down_and_total() -> None:
    trained = model.TrainedModel(
        booster=None, calibrator=None, features=[],
        bands=[
            model.Band("CRITICAL", 0.9, 0.8, 100, 0.95),
            model.Band("HIGH", 0.75, 0.5, 200, 0.80),
            model.Band("MEDIUM", 0.5, 0.2, 400, 0.55),
        ],
    )
    assigned = trained.severity(np.array([0.9, 0.6, 0.3, 0.1]))
    assert list(assigned) == ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def test_an_unpopulated_band_is_skipped_not_filled() -> None:
    trained = model.TrainedModel(
        booster=None, calibrator=None, features=[],
        bands=[
            model.Band("CRITICAL", 0.9, None, 0, float("nan")),
            model.Band("HIGH", 0.75, 0.5, 200, 0.80),
            model.Band("MEDIUM", 0.5, 0.2, 400, 0.55),
        ],
    )
    assigned = trained.severity(np.array([0.99, 0.3]))
    assert list(assigned) == ["HIGH", "MEDIUM"], (
        "an unpopulated CRITICAL must not be filled by the next band's name"
    )


# ---- metrics -----------------------------------------------------------


def test_normalised_ap_is_zero_for_a_random_ranking() -> None:
    """no-skill = 0 in every period, which is the point (SPEC 6.5 P3)."""
    rng = np.random.default_rng(11)
    y = (rng.uniform(size=20000) < 0.05).astype(int)
    n_ap, prevalence = metrics.normalised_average_precision(y, rng.uniform(size=20000))
    assert prevalence == pytest.approx(0.05, abs=0.01)
    assert abs(n_ap) < 0.02


def test_normalised_ap_is_one_for_a_perfect_ranking() -> None:
    y = np.array([1] * 50 + [0] * 950)
    scores = np.linspace(1.0, 0.0, 1000)
    n_ap, _ = metrics.normalised_average_precision(y, scores)
    assert n_ap == pytest.approx(1.0, abs=1e-9)


def test_normalised_ap_is_comparable_across_a_prevalence_shift() -> None:
    """Why raw PR-AUC was rejected for P3.

    The same perfect ranking at two prevalences gives very different raw AP
    baselines but identical nAP.
    """
    high = np.array([1] * 200 + [0] * 800)
    low = np.array([1] * 20 + [0] * 980)
    scores = np.linspace(1.0, 0.0, 1000)
    nap_high, prev_high = metrics.normalised_average_precision(high, scores)
    nap_low, prev_low = metrics.normalised_average_precision(low, scores)
    assert prev_high != prev_low
    assert nap_high == pytest.approx(nap_low, abs=1e-9)


def test_recall_at_k_penalises_a_conservative_ranking() -> None:
    """Why cluster Recall@K was added beside Precision@K (SPEC 6.2)."""
    y = np.array([1] * 100 + [0] * 900)
    scores = np.linspace(1.0, 0.0, 1000)
    assert metrics.precision_at_k(y, scores, 10) == pytest.approx(1.0)
    assert metrics.recall_at_k(y, scores, 10) == pytest.approx(0.10)


def test_precision_at_k_handles_k_larger_than_the_set() -> None:
    y = np.array([1, 0, 1])
    assert metrics.precision_at_k(y, [0.9, 0.1, 0.8], 100) == pytest.approx(2 / 3)


def test_the_report_carries_its_own_baseline() -> None:
    """A PR-AUC without its no-skill baseline lets a prevalence shift read
    as a performance change."""
    y = np.array([1] * 20 + [0] * 180)
    report = metrics.evaluate("t", y, np.linspace(1, 0, 200))
    assert report.pr_auc_baseline == pytest.approx(0.10)
    assert "pr_auc_baseline" in report.as_dict()


def test_calibration_curve_bins_are_populated_only_where_data_exists() -> None:
    y = np.array([0, 1, 0, 1])
    curve = metrics.calibration_curve(y, np.array([0.05, 0.95, 0.05, 0.95]), bins=10)
    assert set(curve["bin"]) == {0, 9}
    assert curve["n"].sum() == 4


# ---- cluster aggregation ------------------------------------------------


def scored(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["address", "y", "risk", "n_txs_asof_t"])


def test_aggregations_differ_on_a_single_high_outlier() -> None:
    """The 11,001-member super-cluster case, in miniature (SPEC 0.6).

    One high-risk member in a large otherwise-quiet cluster: max is dominated
    by it, top-k and the quantile are not.
    """
    members = [(f"a{i}", 0, 0.01, 1.0) for i in range(20)]
    members[0] = ("a0", 0, 0.99, 1.0)
    index = pd.DataFrame({"address": [m[0] for m in members], "cluster_id": 1})
    out = experiment.aggregate_clusters(scored(members), index)
    row = out.iloc[0]
    assert row["AGG-MAX"] == pytest.approx(0.99)
    assert row["AGG-TOPK"] < row["AGG-MAX"]
    assert row["AGG-QUANT"] < row["AGG-MAX"]


def test_the_primary_label_excludes_a_trace_contaminated_cluster() -> None:
    """SPEC 2.1: 6 illicit in 11,001 is a clustering artefact, not an entity.

    Scaled down: 1 illicit in 40 is a share of 0.025, below the 0.10
    threshold, so the cluster is a NEGATIVE under the primary label and a
    positive under the registered sensitivity label.
    """
    members = [(f"a{i}", 0, 0.1, 1.0) for i in range(40)]
    members[0] = ("a0", 1, 0.9, 1.0)
    index = pd.DataFrame({"address": [m[0] for m in members], "cluster_id": 7})
    out = experiment.aggregate_clusters(scored(members), index)
    assert out.iloc[0]["illicit_share"] == pytest.approx(0.025)
    assert out.iloc[0]["label_primary"] == 0
    assert out.iloc[0]["label_sensitivity"] == 1


def test_a_majority_illicit_cluster_is_positive_under_both_labels() -> None:
    members = [(f"b{i}", 1, 0.9, 1.0) for i in range(8)]
    members += [(f"c{i}", 0, 0.1, 1.0) for i in range(2)]
    index = pd.DataFrame({"address": [m[0] for m in members], "cluster_id": 3})
    out = experiment.aggregate_clusters(scored(members), index)
    assert out.iloc[0]["label_primary"] == 1
    assert out.iloc[0]["label_sensitivity"] == 1


def test_cluster_evaluation_reports_recall_beside_precision() -> None:
    members = [(f"d{i}", int(i < 5), 1.0 - i / 100, 1.0) for i in range(60)]
    index = pd.DataFrame({
        "address": [m[0] for m in members],
        "cluster_id": [i // 10 for i in range(60)],
    })
    clusters = experiment.aggregate_clusters(scored(members), index)
    table = experiment.evaluate_clusters(clusters)
    assert {"precision_at_10", "recall_at_10"} <= set(table.columns)
    assert set(table["aggregation"]) == set(experiment.AGGREGATIONS)
    assert set(table["cluster_label"]) == {
        "primary_share>=0.10", "sensitivity_>=1_illicit"
    }


def test_the_stage_list_is_cumulative_and_ordered() -> None:
    names = [name for name, _ in experiment.STAGES]
    assert names == ["M0", "M0+M1", "M0+M1+M2", "M0+M1+M2+M3"]
    groups = [g for _, g in experiment.STAGES]
    for earlier, later in zip(groups, groups[1:]):
        assert set(earlier) < set(later), "each stage must ADD to the previous"
