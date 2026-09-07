"""Tests for the Phase 3.1 evidence funnel.

Diagnosis instrumentation: these check that the funnel counts correctly and
records evidence *before* any test is applied. They add no mechanism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obsidianchain.eval import evidence_funnel as funnel
from obsidianchain.io.elliptic import CoSpendGraph
from obsidianchain.network import separation as sep


def stats_from(samples: np.ndarray) -> sep.GroupStats:
    present = ~np.isnan(samples)
    filled = np.nan_to_num(samples, nan=0.0)
    return sep.GroupStats(
        count=int(samples.shape[0]),
        dim_count=present.sum(axis=0).astype(np.int64),
        dim_sum=filled.sum(axis=0),
        dim_sumsq=(filled**2).sum(axis=0),
    )


def make_graph(edges, n_addresses=8) -> CoSpendGraph:
    array = np.array(edges, dtype=np.int32).reshape(-1, 2)
    return CoSpendGraph(
        n_addresses=n_addresses,
        edges=array,
        edge_tx_ids=np.arange(len(array), dtype=np.int64),
        universe_codes=np.arange(n_addresses, dtype=np.int32),
        n_transactions=len(array),
        n_input_rows=0,
        n_input_pairs=0,
        n_universe_addresses=n_addresses,
        n_input_only_addresses=0,
        addresses=np.array([f"a{i}" for i in range(n_addresses)], dtype=object),
    )


def make_oracle(per_node: dict[int, np.ndarray], n_dims=4) -> sep.SeparationOracle:
    return sep.SeparationOracle(
        n_dims=n_dims,
        observer_ids=[f"obs-{i}" for i in range(n_dims)],
        address_stats={k: stats_from(v) for k, v in per_node.items()},
    )


# ---- counting ----------------------------------------------------------


def test_redundant_edges_are_excluded_from_proposed_unions() -> None:
    """(0,1) twice: the second is already connected and proposes nothing."""
    graph = make_graph([(0, 1), (0, 1), (2, 3)])
    result = funnel.build_funnel(graph, make_oracle({}))
    assert result.n_edges == 3
    assert result.n_redundant == 1
    assert result.n_proposed == 2
    assert len(result.records) == 2


def test_proposed_plus_redundant_equals_edges() -> None:
    graph = make_graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 3)])
    result = funnel.build_funnel(graph, make_oracle({}))
    assert result.n_proposed + result.n_redundant == result.n_edges


def test_records_one_row_per_proposed_union() -> None:
    graph = make_graph([(0, 1), (2, 3), (4, 5)])
    result = funnel.build_funnel(graph, make_oracle({}))
    assert list(result.records.columns) == funnel.RECORD_COLUMNS
    assert len(result.records) == 3


# ---- evidence recorded BEFORE the test --------------------------------


def test_pooled_counts_are_recorded_as_they_were_before_the_merge() -> None:
    """Hand-checked: node 0 has 3 observations, node 1 has 5.

    The first union must record 3 and 5, not the merged 8.
    """
    rng = np.random.default_rng(0)
    oracle = make_oracle(
        {0: rng.normal(size=(3, 4)), 1: rng.normal(size=(5, 4))}
    )
    result = funnel.build_funnel(make_graph([(0, 1)]), oracle)
    row = result.records.iloc[0]
    assert {int(row["pooled_a"]), int(row["pooled_b"])} == {3, 5}
    assert int(row["min_pooled"]) == 3


def test_pooled_counts_accumulate_across_successive_unions() -> None:
    """Second union sees the first's merged total, which is the point."""
    rng = np.random.default_rng(1)
    oracle = make_oracle({node: rng.normal(size=(4, 4)) for node in (0, 1, 2)})
    result = funnel.build_funnel(make_graph([(0, 1), (1, 2)]), oracle)
    first, second = result.records.iloc[0], result.records.iloc[1]
    assert int(first["min_pooled"]) == 4
    assert max(int(second["pooled_a"]), int(second["pooled_b"])) == 8
    assert int(second["min_pooled"]) == 4


def test_min_pooled_is_the_smaller_side() -> None:
    rng = np.random.default_rng(2)
    oracle = make_oracle(
        {0: rng.normal(size=(30, 4)), 1: rng.normal(size=(2, 4))}
    )
    row = funnel.build_funnel(make_graph([(0, 1)]), oracle).records.iloc[0]
    assert int(row["min_pooled"]) == 2, "the smaller side binds"


def test_group_sizes_are_recorded_in_addresses_too() -> None:
    oracle = make_oracle({})
    result = funnel.build_funnel(make_graph([(0, 1), (2, 3), (1, 3)]), oracle)
    last = result.records.iloc[-1]
    assert int(last["size_a"]) == 2 and int(last["size_b"]) == 2
    assert int(last["min_size"]) == 2


# ---- the two gates ----------------------------------------------------


def test_a_single_observation_gives_zero_degrees_of_freedom() -> None:
    """Clearing a pooled minimum of 1 can never yield a decision."""
    rng = np.random.default_rng(3)
    oracle = make_oracle(
        {0: rng.normal(size=(1, 4)), 1: rng.normal(size=(40, 4))}
    )
    row = funnel.build_funnel(make_graph([(0, 1)]), oracle).records.iloc[0]
    assert int(row["min_pooled"]) == 1
    assert int(row["dof"]) == 0
    assert not np.isfinite(row["chi2"])


def test_decidable_requires_both_gates() -> None:
    rng = np.random.default_rng(4)
    oracle = make_oracle(
        {0: rng.normal(size=(1, 4)), 1: rng.normal(size=(40, 4)),
         2: rng.normal(size=(30, 4)), 3: rng.normal(size=(30, 4))}
    )
    result = funnel.build_funnel(make_graph([(0, 1), (2, 3)]), oracle)
    assert len(result.cleared(1)) == 2
    assert len(result.decidable(1)) == 1, "the 1-observation union is undecidable"


def test_statistic_is_computed_when_both_sides_reach_two() -> None:
    rng = np.random.default_rng(5)
    oracle = make_oracle(
        {0: rng.normal(size=(6, 4)), 1: rng.normal(size=(6, 4))}
    )
    row = funnel.build_funnel(make_graph([(0, 1)]), oracle).records.iloc[0]
    assert int(row["dof"]) >= 1
    assert np.isfinite(row["chi2"]) and np.isfinite(row["p_value"])


# ---- thresholds are monotone ------------------------------------------


def test_cleared_counts_are_non_increasing_in_the_threshold() -> None:
    rng = np.random.default_rng(6)
    oracle = make_oracle(
        {node: rng.normal(size=(node + 1, 4)) for node in range(8)}
    )
    graph = make_graph([(0, 1), (2, 3), (4, 5), (6, 7)])
    result = funnel.build_funnel(graph, oracle, thresholds=(1, 2, 5, 10, 25))
    counts = [len(result.cleared(t)) for t in result.thresholds]
    assert counts == sorted(counts, reverse=True)


def test_verdicts_at_threshold_never_exceed_decidable() -> None:
    rng = np.random.default_rng(7)
    oracle = make_oracle(
        {0: rng.normal(0, 0.1, (60, 4)), 1: rng.normal(1, 0.1, (60, 4))}
    )
    result = funnel.build_funnel(make_graph([(0, 1)]), oracle)
    for threshold in (1, 2, 5, 10, 25):
        verdicts = result.verdicts_at(threshold)
        total = verdicts["SEPARATED"] + verdicts["NOT_SEPARATED"]
        assert total == len(result.decidable(threshold))


# ---- production reproduction ------------------------------------------


def test_production_verdicts_sum_to_proposed_unions() -> None:
    rng = np.random.default_rng(8)
    oracle = make_oracle(
        {0: rng.normal(0, 0.1, (40, 4)), 1: rng.normal(1, 0.1, (40, 4)),
         2: rng.normal(size=(2, 4)), 3: rng.normal(size=(2, 4))}
    )
    result = funnel.build_funnel(make_graph([(0, 1), (2, 3)]), oracle)
    assert sum(result.production_verdicts.values()) == result.n_proposed


def test_clearly_separated_groups_are_reported_separated() -> None:
    rng = np.random.default_rng(9)
    oracle = make_oracle(
        {0: rng.normal(0.0, 0.05, (200, 4)), 1: rng.normal(1.0, 0.05, (200, 4))}
    )
    result = funnel.build_funnel(make_graph([(0, 1)]), oracle)
    assert result.production_verdicts["SEPARATED"] == 1
    assert result.verdicts_at(25)["SEPARATED"] == 1


def test_identical_groups_are_reported_not_separated() -> None:
    rng = np.random.default_rng(10)
    oracle = make_oracle(
        {0: rng.normal(0.0, 1.0, (200, 4)), 1: rng.normal(0.0, 1.0, (200, 4))}
    )
    result = funnel.build_funnel(make_graph([(0, 1)]), oracle)
    assert result.production_verdicts["NOT_SEPARATED"] == 1
    assert result.production_verdicts["SEPARATED"] == 0


# ---- trajectory is the baseline's -------------------------------------


def test_funnel_never_blocks_a_merge() -> None:
    """All thresholds must describe the same merge sequence.

    Separated groups are present, but the funnel still applies every union -
    otherwise the rows would describe different clusterings.
    """
    rng = np.random.default_rng(11)
    oracle = make_oracle(
        {0: rng.normal(0.0, 0.05, (200, 4)), 1: rng.normal(1.0, 0.05, (200, 4)),
         2: rng.normal(2.0, 0.05, (200, 4))}
    )
    graph = make_graph([(0, 1), (1, 2)])
    result = funnel.build_funnel(graph, oracle)
    assert result.n_proposed == 2, "both unions proposed despite separation"
    assert result.n_redundant == 0


# ---- output -----------------------------------------------------------


def test_report_renders_every_section() -> None:
    rng = np.random.default_rng(12)
    oracle = make_oracle(
        {0: rng.normal(0.0, 0.1, (40, 4)), 1: rng.normal(0.6, 0.1, (40, 4))}
    )
    text = funnel.format_funnel(funnel.build_funnel(make_graph([(0, 1)]), oracle))
    for expected in (
        "EVIDENCE FUNNEL", "proposed unions", "second gate",
        "decisions at each threshold", "group size at merge time", "reading",
    ):
        assert expected in text, expected


def test_records_can_be_written(tmp_path) -> None:
    oracle = make_oracle({})
    result = funnel.build_funnel(make_graph([(0, 1), (2, 3)]), oracle)
    out = tmp_path / "sub" / "funnel.csv"
    assert funnel.write_records(result, out) == 2
    assert list(pd.read_csv(out).columns) == funnel.RECORD_COLUMNS


def test_empty_graph_is_handled() -> None:
    graph = make_graph([], n_addresses=4)
    result = funnel.build_funnel(graph, make_oracle({}))
    assert result.n_proposed == 0
    assert "proposed unions" in funnel.format_funnel(result)
