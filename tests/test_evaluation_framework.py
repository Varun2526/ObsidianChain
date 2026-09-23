"""The metric definitions in ml/evaluation.py, checked against hand-computed cases."""

from __future__ import annotations

import numpy as np
import pytest

from obsidianchain.ml import evaluation as ev


def test_perfect_ranking() -> None:
    y = np.array([1, 1, 0, 0, 0])
    m = ev.ranking_metrics(y, [5, 4, 3, 2, 1], ks=(2, 3))
    assert m["nap"] == pytest.approx(1.0)
    assert m["r_precision"] == 1.0
    assert m["P@2"] == 1.0 and m["R@2"] == 1.0 and m["NDCG@2"] == pytest.approx(1.0)
    assert m["P@3"] == pytest.approx(2 / 3) and m["ceiling_P@3"] == pytest.approx(2 / 3)
    assert m["FP@3"] == 1 and m["FN@2"] == 0
    assert m["lift@2"] == pytest.approx(1 / 0.4)


def test_ceiling_normalised_recall() -> None:
    y = np.array([1] * 10 + [0] * 90)
    m = ev.ranking_metrics(y, -np.arange(100), ks=(5,))
    assert m["R@5"] == 0.5 and m["ceiling_R@5"] == 0.5 and m["nR@5"] == 1.0


def test_single_class_is_not_measurable_rather_than_zero() -> None:
    assert ev.ranking_metrics([0, 0, 0], [1, 2, 3])["measurable"] is False


def test_transaction_level_collapses_rows() -> None:
    m = ev.transaction_metrics([1, 0, 0, 0], [0.9, 0.8, 0.1, 0.2], ["t1", "t1", "t2", "t3"], ks=(1,))
    assert m["n"] == 3 and m["positives"] == 1 and m["P@1"] == 1.0


def test_calibration_of_a_calibrated_predictor() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    c = ev.calibration_metrics(y, p)
    assert c["ece"] < 0.02
    assert c["calibration_slope"] == pytest.approx(1.0, abs=0.1)
    assert c["calibration_intercept"] == pytest.approx(0.0, abs=0.1)


def test_aggregate_reports_interval_and_extremes() -> None:
    a = ev.aggregate([0.5, 0.6, 0.7])
    assert a["min"] == 0.5 and a["max"] == 0.7 and a["ci95"][0] < 0.6 < a["ci95"][1]
