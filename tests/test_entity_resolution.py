"""Tests for entity-resolution scoring.

Primary fixture, 6 labelled addresses hand-worked end to end:

    addr  entity  cluster
    a1    E1      C1
    a2    E1      C1
    a3    E1      C2      E1 is split across C1 and C2
    a4    E2      C1      E2 is merged into C1 alongside E1
    a5    E2      C3
    a6    E3      C4      singleton entity: negatives only

    all pairs      C(6,2) = 15
    positives      C(3,2) + C(2,2) + 0 = 3 + 1 + 0 = 4
    predicted same C(3,2) over C1 = 3

    TP = (a1,a2)                    = 1
    FP = (a1,a4), (a2,a4)           = 2   false merges
    FN = (a1,a3), (a2,a3), (a4,a5)  = 3   false splits
    TN = 15 - 1 - 2 - 3             = 9

    precision 1/3, recall 1/4, F1 = 2/7
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain.cluster.pipeline import ClusterRun
from obsidianchain.eval import entity_resolution as er
from obsidianchain.io.elliptic import CoSpendGraph

ENTITIES = ["E1", "E1", "E1", "E2", "E2", "E3"]
CLUSTERS = [0, 0, 1, 0, 2, 3]
FULL_SIZES = [3, 1, 1, 1]  # C1 holds 3 addresses; C2, C3, C4 are singletons


def make_table(entities, clusters):
    e_codes, e_uniq = pd.factorize(pd.Series(entities))
    c_codes, c_uniq = pd.factorize(pd.Series(clusters))
    return er.contingency(e_codes, c_codes, len(e_uniq), len(c_uniq))


@pytest.fixture()
def table() -> np.ndarray:
    return make_table(ENTITIES, CLUSTERS)


# ---- helpers ----------------------------------------------------------


def test_choose2() -> None:
    assert list(er._choose2(np.array([0, 1, 2, 3, 10]))) == [0, 0, 1, 3, 45]


def test_f1_edge_cases() -> None:
    assert er._f1(0.0, 0.0) == 0.0
    assert er._f1(1.0, 0.0) == 0.0
    assert er._f1(0.5, 0.5) == pytest.approx(0.5)
    assert er._f1(float("nan"), 0.5) == 0.0


# ---- contingency and micro counts --------------------------------------


def test_contingency_shape_and_totals(table: np.ndarray) -> None:
    assert table.shape == (3, 4)
    assert int(table.sum()) == 6
    assert list(table.sum(axis=1)) == [3, 2, 1]  # E1, E2, E3


def test_micro_confusion_matrix(table: np.ndarray) -> None:
    tp, fp, fn, tn = er.micro_from_contingency(table, 6)
    assert (tp, fp, fn, tn) == (1, 2, 3, 9)


def test_micro_counts_sum_to_all_pairs(table: np.ndarray) -> None:
    tp, fp, fn, tn = er.micro_from_contingency(table, 6)
    assert tp + fp + fn + tn == 15


def test_micro_positive_and_predicted_totals(table: np.ndarray) -> None:
    tp, fp, fn, _ = er.micro_from_contingency(table, 6)
    assert tp + fn == 4, "same-entity pairs"
    assert tp + fp == 3, "same-cluster pairs"


def test_perfect_clustering_scores_one() -> None:
    tp, fp, fn, tn = er.micro_from_contingency(
        make_table(["A", "A", "B", "B"], [0, 0, 1, 1]), 4
    )
    assert (fp, fn) == (0, 0)
    assert tp == 2


def test_everything_in_one_cluster_is_all_false_merges() -> None:
    tp, fp, fn, tn = er.micro_from_contingency(
        make_table(["A", "A", "B", "B"], [0, 0, 0, 0]), 4
    )
    assert (tp, fn) == (2, 0)
    assert fp == 4, "every cross-entity pair is a false merge"


def test_all_singleton_clusters_is_all_false_splits() -> None:
    tp, fp, fn, _ = er.micro_from_contingency(
        make_table(["A", "A", "B", "B"], [0, 1, 2, 3]), 4
    )
    assert (tp, fp) == (0, 0)
    assert fn == 2


# ---- per-entity metrics ------------------------------------------------


def test_per_entity_values(table: np.ndarray) -> None:
    s = er.per_entity_metrics(table)
    # E1: 2 of its 3 addresses share C1 -> 1 of 3 possible pairs recovered
    assert s["tp_e"][0] == 1 and s["possible"][0] == 3
    assert s["recall"][0] == pytest.approx(1 / 3)
    # entity-anchored precision: 1 within-pair vs 2 pairs reaching outside E1
    assert s["cross_e"][0] == 2
    assert s["precision"][0] == pytest.approx(1 / 3)
    assert s["f1"][0] == pytest.approx(1 / 3)
    # E2: never co-clustered with itself
    assert s["recall"][1] == pytest.approx(0.0)
    assert s["precision"][1] == pytest.approx(0.0)
    assert s["f1"][1] == 0.0


def test_singleton_entity_recall_is_undefined(table: np.ndarray) -> None:
    """E3 has one address: no positive pair exists, so recall is nan."""
    s = er.per_entity_metrics(table)
    assert s["possible"][2] == 0
    assert np.isnan(s["recall"][2])


def test_macro_averages(table: np.ndarray) -> None:
    macro = er.macro_from_contingency(table)
    assert macro.n_entities == 2, "only E1 and E2 have >=2 addresses"
    assert macro.macro_recall == pytest.approx(1 / 6)
    assert macro.macro_precision == pytest.approx(1 / 6)
    assert macro.macro_f1 == pytest.approx(1 / 6)


def test_precision_undefined_when_entity_has_no_predicted_pair() -> None:
    """An entity entirely in singleton clusters anchors no predicted pair."""
    t = make_table(["A", "A", "B", "B"], [0, 1, 2, 3])
    s = er.per_entity_metrics(t)
    assert np.isnan(s["precision"]).all()
    macro = er.macro_from_contingency(t)
    assert macro.n_precision_undefined == 2
    assert macro.macro_f1 == 0.0


def test_macro_diverges_from_micro_under_concentration() -> None:
    """One large well-clustered entity must not carry the headline.

    E_big has 10 addresses in one cluster (45 positive pairs, all recovered).
    Four two-address entities are each split. Micro recall is 45/49 = 0.92;
    macro recall is 1/5 = 0.20 because every entity counts once.
    """
    entities = ["big"] * 10
    clusters = [0] * 10
    for i in range(4):
        entities += [f"s{i}", f"s{i}"]
        clusters += [10 + 2 * i, 11 + 2 * i]
    t = make_table(entities, clusters)

    tp, fp, fn, _ = er.micro_from_contingency(t, len(entities))
    assert tp == 45 and fn == 4 and fp == 0
    assert tp / (tp + fn) == pytest.approx(45 / 49, abs=1e-6)

    macro = er.macro_from_contingency(t)
    assert macro.n_entities == 5
    assert macro.macro_recall == pytest.approx(0.2)
    assert macro.macro_recall < tp / (tp + fn) / 4, "macro must resist the big entity"


# ---- baseline ----------------------------------------------------------


def test_baseline_is_deterministic_for_a_seed(table: np.ndarray) -> None:
    e_codes, e_uniq = pd.factorize(pd.Series(ENTITIES))
    c_codes, c_uniq = pd.factorize(pd.Series(CLUSTERS))
    args = (e_codes, c_codes, len(e_uniq), len(c_uniq), 0.3, 0.3)
    first = er.permutation_baseline(*args, n_permutations=50, seed=7)
    second = er.permutation_baseline(*args, n_permutations=50, seed=7)
    assert first.micro_f1_mean == second.micro_f1_mean
    assert first.macro_p_value == second.macro_p_value


def test_different_seeds_give_different_draws(table: np.ndarray) -> None:
    e_codes, e_uniq = pd.factorize(pd.Series(ENTITIES))
    c_codes, c_uniq = pd.factorize(pd.Series(CLUSTERS))
    args = (e_codes, c_codes, len(e_uniq), len(c_uniq), 0.3, 0.3)
    a = er.permutation_baseline(*args, n_permutations=50, seed=1)
    b = er.permutation_baseline(*args, n_permutations=50, seed=2)
    assert a.micro_f1_mean != b.micro_f1_mean


def test_baseline_preserves_entity_sizes(table: np.ndarray) -> None:
    """Permuting the label vector cannot change how many addresses an entity has."""
    e_codes, _ = pd.factorize(pd.Series(ENTITIES))
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(e_codes)
    assert sorted(np.bincount(shuffled)) == sorted(np.bincount(e_codes))


def test_p_value_is_bounded(table: np.ndarray) -> None:
    e_codes, e_uniq = pd.factorize(pd.Series(ENTITIES))
    c_codes, c_uniq = pd.factorize(pd.Series(CLUSTERS))
    b = er.permutation_baseline(
        e_codes, c_codes, len(e_uniq), len(c_uniq), 1.0, 1.0,
        n_permutations=20, seed=3,
    )
    assert 0 < b.micro_p_value <= 1
    assert b.n_permutations == 20 and b.seed == 3


# ---- end to end --------------------------------------------------------


def build_fixture(tmp_path: Path):
    addresses = np.array([f"addr{i}" for i in range(6)], dtype=object)
    graph = CoSpendGraph(
        n_addresses=6,
        edges=np.empty((0, 2), dtype=np.int32),
        edge_tx_ids=np.empty(0, dtype=np.int64),
        universe_codes=np.arange(6, dtype=np.int32),
        n_transactions=0,
        n_input_rows=0,
        n_input_pairs=0,
        n_universe_addresses=6,
        n_input_only_addresses=0,
        addresses=addresses,
    )
    roots = np.array([0, 0, 2, 0, 4, 5], dtype=np.int32)
    sizes = np.array([3, 1, 1, 1], dtype=np.int64)
    run = ClusterRun(
        label="multi-input", roots=roots, sizes=sizes, n_addresses=6,
        cospend_edges=0, cospend_merges=0,
    )
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "address": addresses,
            "entity_norm": [e.lower() for e in ENTITIES],
            "entity_display": ENTITIES,
            "category": ["X"] * 6,
            "source": ["S"] * 6,
            "in_elliptic": [True] * 6,
        }
    ).to_csv(processed / er.LABELS_FILE, index=False)
    return graph, run


def test_evaluate_end_to_end(tmp_path: Path) -> None:
    graph, run = build_fixture(tmp_path)
    result = er.evaluate(graph, run, tmp_path, n_permutations=20, seed=0)

    assert (result.micro.tp, result.micro.fp, result.micro.fn, result.micro.tn) == (
        1, 2, 3, 9
    )
    assert result.micro.n_pairs == 15
    assert result.micro.n_positive == 4
    assert result.micro.n_negative == 11
    assert result.micro.precision == pytest.approx(1 / 3)
    assert result.micro.recall_raw == pytest.approx(0.25)
    assert result.micro.f1 == pytest.approx(2 / 7)
    assert result.micro.false_merges == 2
    assert result.micro.false_splits == 3


def test_evaluate_recall_ceiling(tmp_path: Path) -> None:
    """Only a1, a2, a4 sit in a cluster of size > 1; of those only a1/a2 share
    an entity, so exactly one positive pair is reachable."""
    graph, run = build_fixture(tmp_path)
    result = er.evaluate(graph, run, tmp_path, n_permutations=0)
    assert result.micro.ceiling_pairs == 1
    assert result.micro.recall_ceiling == pytest.approx(0.25)
    assert result.micro.recall_conditional == pytest.approx(1.0)
    assert result.coverage.evaluable_addresses == 3


def test_evaluate_coverage(tmp_path: Path) -> None:
    graph, run = build_fixture(tmp_path)
    result = er.evaluate(graph, run, tmp_path, n_permutations=0)
    assert result.coverage.n_labelled == 6
    assert result.coverage.n_entities_total == 3
    assert result.coverage.n_entities_multi == 2
    assert result.coverage.evaluable_fraction == pytest.approx(50.0)


def test_evaluate_verdicts(tmp_path: Path) -> None:
    graph, run = build_fixture(tmp_path)
    result = er.evaluate(graph, run, tmp_path, n_permutations=0)
    rows = result.per_entity.set_index("entity_norm")
    assert rows.loc["e1", "verdict"] == "SPLIT+MERGED"
    assert rows.loc["e2", "verdict"] == "SPLIT+MERGED"
    assert int(rows.loc["e1", "foreign_labelled"]) == 1


def test_verdict_perfect_and_unreachable(tmp_path: Path) -> None:
    """A cleanly recovered entity and one with no reachable address."""
    addresses = np.array([f"a{i}" for i in range(4)], dtype=object)
    graph = CoSpendGraph(
        n_addresses=4, edges=np.empty((0, 2), dtype=np.int32),
        edge_tx_ids=np.empty(0, dtype=np.int64),
        universe_codes=np.arange(4, dtype=np.int32), n_transactions=0,
        n_input_rows=0, n_input_pairs=0, n_universe_addresses=4,
        n_input_only_addresses=0, addresses=addresses,
    )
    run = ClusterRun(
        label="m", roots=np.array([0, 0, 2, 3], dtype=np.int32),
        sizes=np.array([2, 1, 1], dtype=np.int64), n_addresses=4,
        cospend_edges=0, cospend_merges=0,
    )
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "address": addresses,
            "entity_norm": ["good", "good", "gone", "gone"],
            "entity_display": ["Good", "Good", "Gone", "Gone"],
            "in_elliptic": [True] * 4,
        }
    ).to_csv(processed / er.LABELS_FILE, index=False)

    result = er.evaluate(graph, run, tmp_path, n_permutations=0)
    rows = result.per_entity.set_index("entity_norm")
    assert rows.loc["good", "verdict"] == "PERFECT"
    assert rows.loc["gone", "verdict"] == "UNREACHABLE"
    assert result.verdict_counts["PERFECT"] == 1
    assert result.verdict_counts["UNREACHABLE"] == 1


def test_missing_labels_file_names_the_remedy(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="entity-labels"):
        er.load_evaluation_set(tmp_path)


def test_evaluate_requires_address_labels(tmp_path: Path) -> None:
    graph, run = build_fixture(tmp_path)
    graph.addresses = None
    with pytest.raises(ValueError, match="keep_labels"):
        er.evaluate(graph, run, tmp_path, n_permutations=0)


def test_report_and_csv(tmp_path: Path) -> None:
    graph, run = build_fixture(tmp_path)
    result = er.evaluate(graph, run, tmp_path, n_permutations=10, seed=0)
    text = er.format_report(result)
    for expected in ("HEADLINE", "macro F1", "FALSE MERGES", "FALSE SPLITS",
                     "structural ceiling", "coverage", "random baseline"):
        assert expected in text, expected

    out = tmp_path / "processed" / "er.csv"
    assert er.write_per_entity_csv(result, out) == 3
    written = pd.read_csv(out)
    assert list(written.columns) == ["heuristics"] + er.PER_ENTITY_COLUMNS


def test_delta_table(tmp_path: Path) -> None:
    graph, run = build_fixture(tmp_path)
    a = er.evaluate(graph, run, tmp_path, n_permutations=0)
    b = er.evaluate(graph, run, tmp_path, n_permutations=0)
    text = er.format_delta(a, b)
    assert "MACRO F1 (headline)" in text and "FALSE MERGES" in text
