"""Tests for the cumulative timestep analysis.

The incremental replay must give exactly what recomputing from scratch would,
and the super-linear detector must distinguish linear from accelerating growth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain.cluster.unionfind import UnionFind
from obsidianchain.eval import evolution as evo
from obsidianchain.eval import purity as pu


# ---- timestep bucketing -----------------------------------------------


def test_bucket_maps_transactions_to_timesteps() -> None:
    timesteps = pd.Series([1, 3, 3], index=[100, 101, 102], name="timestep")
    bucket, unknown = evo._bucket_by_timestep(
        np.array([100, 101, 102]), timesteps, n_timesteps=49
    )
    assert bucket.tolist() == [1, 3, 3]
    assert unknown == 0


def test_unknown_timestep_folds_into_the_final_step() -> None:
    """Otherwise the last snapshot would silently disagree with Phase 1."""
    timesteps = pd.Series([1], index=[100], name="timestep")
    bucket, unknown = evo._bucket_by_timestep(
        np.array([100, 999]), timesteps, n_timesteps=49
    )
    assert bucket.tolist() == [1, 49]
    assert unknown == 1


def test_bucket_clips_out_of_range_timesteps() -> None:
    timesteps = pd.Series([0, 80], index=[1, 2], name="timestep")
    bucket, _ = evo._bucket_by_timestep(np.array([1, 2]), timesteps, n_timesteps=49)
    assert bucket.tolist() == [1, 49]


# ---- snapshot statistics ----------------------------------------------


def test_snapshot_matches_the_purity_table() -> None:
    """The fast bincount path must agree with the canonical aggregation."""
    roots = np.array([0, 0, 0, 3, 3, 5, 6, 6, 6], dtype=np.int32)
    classes = np.array(
        [pu.ILLICIT, pu.LICIT, pu.UNKNOWN, pu.ILLICIT, pu.ILLICIT,
         pu.UNKNOWN, pu.ILLICIT, pu.ILLICIT, pu.LICIT],
        dtype=np.int8,
    )
    fast = evo._snapshot(roots, classes, len(roots))
    table = pu.build_cluster_table(roots, classes)

    assert fast["n_clusters"] == len(table)
    assert fast["largest"] == int(table["size"].max())
    assert fast["contaminated"] == int(table["contaminated"].sum())
    assert fast["second_largest"] == int(table["size"].tolist()[1])


def test_snapshot_second_largest_with_one_cluster() -> None:
    roots = np.zeros(4, dtype=np.int32)
    classes = np.full(4, pu.UNKNOWN, dtype=np.int8)
    stats = evo._snapshot(roots, classes, 4)
    assert stats["largest"] == 4
    assert stats["second_largest"] == 0


# ---- the append-only property -----------------------------------------


def test_incremental_replay_equals_recompute_from_scratch() -> None:
    """The core claim: one pass gives the same answer as 5 separate runs.

    This holds only because union-find is append-only. If edges could be
    removed, the incremental series would be invalid.
    """
    rng = np.random.default_rng(3)
    n = 200
    edges = rng.integers(0, n, size=(400, 2), dtype=np.int32)
    edges = edges[edges[:, 0] != edges[:, 1]]
    buckets = rng.integers(1, 6, size=len(edges))

    incremental = UnionFind(n)
    for step in range(1, 6):
        incremental.add_edges(edges[buckets == step])
        scratch = UnionFind(n)
        scratch.add_edges(edges[buckets <= step])
        assert (
            incremental.component_sizes().tolist()
            == scratch.component_sizes().tolist()
        ), f"diverged at timestep {step}"


def test_statistics_are_monotonic_in_the_right_directions() -> None:
    rng = np.random.default_rng(9)
    n = 300
    edges = rng.integers(0, n, size=(600, 2), dtype=np.int32)
    edges = edges[edges[:, 0] != edges[:, 1]]
    buckets = rng.integers(1, 8, size=len(edges))
    classes = np.full(n, pu.UNKNOWN, dtype=np.int8)

    forest = UnionFind(n)
    clusters, largest = [], []
    for step in range(1, 8):
        forest.add_edges(edges[buckets == step])
        stats = evo._snapshot(forest.roots(), classes, n)
        clusters.append(stats["n_clusters"])
        largest.append(stats["largest"])

    assert clusters == sorted(clusters, reverse=True), "clusters only ever merge"
    assert largest == sorted(largest), "the largest cluster never shrinks"


# ---- super-linear detection -------------------------------------------


def test_linear_growth_is_not_flagged() -> None:
    t = np.arange(1, 50)
    onset = evo.detect_superlinear(t, 100 * t, "linear")
    assert onset.detected is False
    assert onset.global_exponent == pytest.approx(1.0, abs=0.05)


def test_quadratic_growth_is_flagged() -> None:
    t = np.arange(1, 50)
    onset = evo.detect_superlinear(t, 10 * t**2, "quadratic")
    assert onset.detected is True
    assert onset.global_exponent == pytest.approx(2.0, abs=0.05)


def test_flat_growth_is_not_flagged() -> None:
    t = np.arange(1, 50)
    onset = evo.detect_superlinear(t, np.full(49, 500), "flat")
    assert onset.detected is False


def test_late_collapse_is_located_not_averaged_away() -> None:
    """A long linear run then a sharp blow-up must report the blow-up."""
    t = np.arange(1, 50)
    largest = np.where(t <= 35, 100 * t, 3500 + 40 * (t - 35) ** 3)
    onset = evo.detect_superlinear(t, largest, "late")
    assert onset.detected is True
    assert onset.timestep is not None and onset.timestep >= 25


def test_largest_jump_is_reported() -> None:
    t = np.arange(1, 11)
    largest = np.array([1, 2, 3, 4, 5, 900, 901, 902, 903, 904])
    onset = evo.detect_superlinear(t, largest, "jump")
    assert onset.largest_jump == 895
    assert onset.largest_jump_timestep == 6


def test_describe_mentions_the_mode_and_exponent() -> None:
    t = np.arange(1, 50)
    text = evo.detect_superlinear(t, 100 * t, "multi-input").describe()
    assert "multi-input" in text and "exponent" in text


# ---- outputs ----------------------------------------------------------


@pytest.fixture()
def series() -> evo.EvolutionSeries:
    rows = []
    for mode in ("multi-input", "multi-input+change"):
        for step in range(1, 6):
            rows.append(
                {
                    "heuristics": mode,
                    "timestep": step,
                    "n_clusters": 100 - step,
                    "largest": 10 * step,
                    "second_largest": 5 * step,
                    "coverage": 10.0 * step,
                    "contaminated": step,
                    "cospend_merges": 4 * step,
                    "change_merges": step if "change" in mode else 0,
                    "cospend_edges_applied": 5 * step,
                    "change_edges_applied": step if "change" in mode else 0,
                    "addresses_seen": 20 * step,
                    "coverage_of_seen": 5.0 * step,
                }
            )
    return evo.EvolutionSeries(frame=pd.DataFrame(rows, columns=evo.SERIES_COLUMNS))


def test_csv_holds_the_whole_series(series: evo.EvolutionSeries, tmp_path: Path) -> None:
    out = tmp_path / "processed" / "evolution.csv"
    rows = evo.write_series_csv(series, out)
    assert rows == 10
    written = pd.read_csv(out)
    assert list(written.columns) == evo.SERIES_COLUMNS
    assert set(written["heuristics"]) == {"multi-input", "multi-input+change"}


def test_chart_is_written_as_a_png(series: evo.EvolutionSeries, tmp_path: Path) -> None:
    out = tmp_path / "chart.png"
    evo.plot_largest_cluster(series, out, evo.detect_all(series), dpi=80)
    assert out.is_file()
    assert out.stat().st_size > 5_000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_chart_directory_is_created(series: evo.EvolutionSeries, tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "chart.png"
    evo.plot_largest_cluster(series, out, None, dpi=80)
    assert out.is_file()


def test_summary_renders(series: evo.EvolutionSeries) -> None:
    text = evo.format_evolution(series, evo.detect_all(series))
    assert "append-only" in text
    assert "multi-input" in text
    assert "super-linear" in text


def test_series_mode_accessors(series: evo.EvolutionSeries) -> None:
    assert series.modes == ["multi-input", "multi-input+change"]
    assert len(series.for_mode("multi-input")) == 5


def test_step_like_growth_is_distinguished_from_acceleration() -> None:
    """Flat-then-jump must not be reported as smooth super-linear growth."""
    t = np.arange(1, 50)
    stepped = np.where(t < 30, 6000.0, 12000.0)
    stepped = np.where(t >= 40, 14000.0, stepped)
    onset = evo.detect_superlinear(t, stepped, "stepped")
    if onset.detected:
        assert onset.is_step_like, "global exponent <= 1 means step, not trend"
    assert "step" in onset.describe() or not onset.detected


def test_smooth_acceleration_is_not_called_step_like() -> None:
    t = np.arange(1, 50)
    onset = evo.detect_superlinear(t, 10 * t**2, "quadratic")
    assert onset.detected is True
    assert onset.is_step_like is False
    assert "sustained acceleration" in onset.describe()
