"""Tests for cluster purity measurement against the Elliptic++ labels.

Hand-built fixture, 9 addresses in 4 clusters:

    cluster 0: [illicit, licit, unknown]   CONTAMINATED, coverage 2/3
    cluster 3: [illicit, illicit]          pure illicit, coverage 1.0
    cluster 5: [unknown]                   no labels at all, coverage 0
    cluster 6: [illicit, illicit, licit]   CONTAMINATED, coverage 1.0
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain.eval import purity as pu

ROOTS = np.array([0, 0, 0, 3, 3, 5, 6, 6, 6], dtype=np.int32)
CLASSES = np.array(
    [pu.ILLICIT, pu.LICIT, pu.UNKNOWN,
     pu.ILLICIT, pu.ILLICIT,
     pu.UNKNOWN,
     pu.ILLICIT, pu.ILLICIT, pu.LICIT],
    dtype=np.int8,
)


@pytest.fixture()
def table() -> pd.DataFrame:
    return pu.build_cluster_table(ROOTS, CLASSES)


def row_for(table: pd.DataFrame, cluster_id: int) -> pd.Series:
    return table[table["cluster_id"] == cluster_id].iloc[0]


# ---- aggregation ------------------------------------------------------


def test_one_row_per_cluster(table: pd.DataFrame) -> None:
    assert len(table) == 4
    assert sorted(table["cluster_id"].tolist()) == [0, 3, 5, 6]


def test_sizes_sum_to_address_count(table: pd.DataFrame) -> None:
    assert int(table["size"].sum()) == len(ROOTS)


def test_sorted_by_size_descending(table: pd.DataFrame) -> None:
    assert table["size"].tolist() == [3, 3, 2, 1]
    # ties broken by cluster id ascending, so output is deterministic
    assert table["cluster_id"].tolist() == [0, 6, 3, 5]


def test_class_counts(table: pd.DataFrame) -> None:
    row = row_for(table, 0)
    assert (int(row["n_illicit"]), int(row["n_licit"]), int(row["n_unknown"])) == (1, 1, 1)
    row = row_for(table, 6)
    assert (int(row["n_illicit"]), int(row["n_licit"]), int(row["n_unknown"])) == (2, 1, 0)


# ---- the unknown rule -------------------------------------------------


def test_unknown_is_never_counted_as_licit(table: pd.DataFrame) -> None:
    """Cluster 3 is illicit+unknown only; it must not read as contaminated."""
    row = row_for(table, 3)
    assert int(row["n_licit"]) == 0
    assert bool(row["contaminated"]) is False


def test_unknown_excluded_from_labelled(table: pd.DataFrame) -> None:
    row = row_for(table, 0)
    assert int(row["n_labelled"]) == 2, "the unknown address must not count"
    assert row["label_coverage"] == pytest.approx(2 / 3)


def test_cluster_with_no_labels_has_undefined_purity(table: pd.DataFrame) -> None:
    row = row_for(table, 5)
    assert int(row["n_labelled"]) == 0
    assert row["label_coverage"] == 0.0
    assert math.isnan(row["purity"])
    assert math.isnan(row["entropy"])
    assert bool(row["contaminated"]) is False


def test_all_unknown_cluster_is_not_evidence_of_anything() -> None:
    roots = np.array([0, 0, 0], dtype=np.int32)
    classes = np.full(3, pu.UNKNOWN, dtype=np.int8)
    result = pu.build_cluster_table(roots, classes)
    assert int(result.iloc[0]["n_labelled"]) == 0
    assert bool(result.iloc[0]["contaminated"]) is False


# ---- contamination ----------------------------------------------------


def test_contamination_flags(table: pd.DataFrame) -> None:
    contaminated = set(table[table["contaminated"]]["cluster_id"].tolist())
    assert contaminated == {0, 6}


def test_contamination_requires_both_classes() -> None:
    for classes, expected in [
        ([pu.ILLICIT, pu.ILLICIT], False),
        ([pu.LICIT, pu.LICIT], False),
        ([pu.ILLICIT, pu.LICIT], True),
        ([pu.ILLICIT, pu.UNKNOWN], False),
        ([pu.LICIT, pu.UNKNOWN], False),
        ([pu.UNKNOWN, pu.UNKNOWN], False),
    ]:
        result = pu.build_cluster_table(
            np.zeros(2, dtype=np.int32), np.array(classes, dtype=np.int8)
        )
        assert bool(result.iloc[0]["contaminated"]) is expected, classes


# ---- purity and entropy ----------------------------------------------


def test_purity_values(table: pd.DataFrame) -> None:
    assert row_for(table, 0)["purity"] == pytest.approx(0.5)    # 1 vs 1
    assert row_for(table, 3)["purity"] == pytest.approx(1.0)    # 2 vs 0
    assert row_for(table, 6)["purity"] == pytest.approx(2 / 3)  # 2 vs 1


def test_entropy_values(table: pd.DataFrame) -> None:
    assert row_for(table, 0)["entropy"] == pytest.approx(1.0)   # even split
    assert row_for(table, 3)["entropy"] == pytest.approx(0.0)   # pure
    expected = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
    assert row_for(table, 6)["entropy"] == pytest.approx(expected)


def test_entropy_is_zero_exactly_when_pure() -> None:
    result = pu.build_cluster_table(ROOTS, CLASSES)
    labelled = result[result["n_labelled"] > 0]
    pure = labelled["purity"] == 1.0
    assert ((labelled["entropy"] == 0.0) == pure).all()


def test_majority_class(table: pd.DataFrame) -> None:
    assert row_for(table, 0)["majority_class"] == "tie"
    assert row_for(table, 3)["majority_class"] == "illicit"
    assert row_for(table, 6)["majority_class"] == "illicit"


def test_purity_never_below_half(table: pd.DataFrame) -> None:
    labelled = table[table["n_labelled"] > 0]
    assert (labelled["purity"] >= 0.5).all()


# ---- the random baseline ---------------------------------------------


def test_log_choose_matches_exact_binomial() -> None:
    for n in (5, 40, 500):
        k = np.array([0, 1, 2, min(3, n), n], dtype=np.float64)
        got = np.exp(pu._log_choose(float(n), k))
        want = np.array([math.comb(n, int(x)) for x in k], dtype=np.float64)
        assert np.allclose(got, want, rtol=1e-9)


def test_single_labelled_address_cannot_be_contaminated() -> None:
    """k=1 must give exactly zero, not a small positive number."""
    clusters, addresses = pu.expected_random_contamination(
        size=np.array([10]), n_labelled=np.array([1]),
        total_illicit=50, total_licit=50,
    )
    assert clusters == 0.0
    assert addresses == 0.0


def test_zero_labelled_addresses_cannot_be_contaminated() -> None:
    clusters, _ = pu.expected_random_contamination(
        size=np.array([10]), n_labelled=np.array([0]),
        total_illicit=50, total_licit=50,
    )
    assert clusters == 0.0


def test_expected_contamination_two_draws_even_pool() -> None:
    """Two draws from 50/50 without replacement: P(mixed) = 50/99."""
    clusters, addresses = pu.expected_random_contamination(
        size=np.array([2]), n_labelled=np.array([2]),
        total_illicit=50, total_licit=50,
    )
    assert clusters == pytest.approx(50 / 99)
    assert addresses == pytest.approx(2 * 50 / 99)


def test_closed_form_matches_monte_carlo() -> None:
    """The hypergeometric formula must agree with sampling the null model."""
    rng = np.random.default_rng(7)
    n_labelled = rng.integers(0, 12, size=400)
    size = n_labelled + rng.integers(0, 5, size=400)
    # The label pool must be exactly the labels distributed across clusters,
    # which is the invariant the real data satisfies by construction.
    total = int(n_labelled.sum())
    total_illicit = int(total * 0.3)
    total_licit = total - total_illicit

    analytic, _ = pu.expected_random_contamination(
        size, n_labelled, total_illicit, total_licit
    )
    sampled = pu.monte_carlo_contamination(
        n_labelled, total_illicit, total_licit, trials=400, seed=11
    )
    assert analytic == pytest.approx(sampled, rel=0.06)


def test_expected_contamination_rises_with_cluster_size() -> None:
    sizes = np.array([2, 5, 20, 100])
    clusters, _ = pu.expected_random_contamination(
        sizes, sizes, total_illicit=500, total_licit=500
    )
    per_cluster = []
    for k in sizes:
        one, _ = pu.expected_random_contamination(
            np.array([k]), np.array([k]), 500, 500
        )
        per_cluster.append(one)
    assert per_cluster == sorted(per_cluster), "bigger clusters mix more by chance"
    assert clusters == pytest.approx(sum(per_cluster))


def test_no_labels_at_all_gives_zero_baseline() -> None:
    clusters, addresses = pu.expected_random_contamination(
        np.array([5]), np.array([0]), total_illicit=0, total_licit=0
    )
    assert (clusters, addresses) == (0.0, 0.0)


# ---- label loading ----------------------------------------------------


def test_class_values_are_validated(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "wallets_classes.csv").write_text("address,class\nA1,7\n")

    class FakeGraph:
        universe_codes = np.array([0], dtype=np.int32)
        n_addresses = 1

    with pytest.raises(ValueError, match="unexpected class values"):
        pu.load_classes_by_code(FakeGraph(), tmp_path)


def test_row_count_mismatch_is_caught(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "wallets_classes.csv").write_text("address,class\nA1,1\nA2,2\n")

    class FakeGraph:
        universe_codes = np.array([0], dtype=np.int32)
        n_addresses = 1

    with pytest.raises(ValueError, match="changed underneath us"):
        pu.load_classes_by_code(FakeGraph(), tmp_path)


# ---- report and CSV ---------------------------------------------------


@pytest.fixture()
def report(table: pd.DataFrame) -> pu.PurityReport:
    return pu.PurityReport(
        clusters=table,
        n_addresses=len(ROOTS),
        n_illicit=int((CLASSES == pu.ILLICIT).sum()),
        n_licit=int((CLASSES == pu.LICIT).sum()),
        n_unknown=int((CLASSES == pu.UNKNOWN).sum()),
        expected_contaminated_clusters=1.0,
        expected_contaminated_addresses=3.0,
    )


def test_report_counts(report: pu.PurityReport) -> None:
    assert report.n_clusters == 4
    assert report.n_contaminated == 2
    assert report.n_labelled == 7  # 5 illicit + 2 licit
    assert report.observed_over_expected == pytest.approx(2.0)


def test_csv_holds_only_contaminated_clusters(
    report: pu.PurityReport, tmp_path: Path
) -> None:
    out = tmp_path / "processed" / "contaminated.csv"
    rows = pu.write_contaminated_csv(report, out)
    assert rows == 2
    assert out.is_file()
    written = pd.read_csv(out)
    assert sorted(written["cluster_id"].tolist()) == [0, 6]
    assert list(written.columns) == pu.CSV_COLUMNS


def test_csv_parent_directory_is_created(
    report: pu.PurityReport, tmp_path: Path
) -> None:
    out = tmp_path / "a" / "b" / "c.csv"
    pu.write_contaminated_csv(report, out)
    assert out.is_file()


def test_summary_renders_and_names_the_key_findings(
    report: pu.PurityReport,
) -> None:
    text = pu.format_summary(report, top=5)
    for expected in (
        "label universe",
        "CONTAMINATED",
        "random baseline",
        "observed / expected",
        "purity distribution",
        "entropy",
        "label coverage",
    ):
        assert expected in text, expected
