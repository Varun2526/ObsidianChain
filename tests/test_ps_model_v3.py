"""Contract tests for the current PS-native model (ps_native_v3, schema /4).

Each test pins one of the four defects v2 exists to fix: the schema gap that
blocked inference, zero-imputed missing values, the importance-times-value
explanation proxy, and static severity bands.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain.ml import monitoring
from obsidianchain.ml.ps_model import (
    EXPLANATION_TREESHAP,
    SEVERITY_BUDGET,
    PsNativeRiskModel,
)
from obsidianchain.pipeline.features_ps import (
    CORE_PS_FEATURE_COLUMNS,
    PS_FEATURE_SCHEMA_VERSION,
)

V2_DIR = Path("data/models/ps_native/v3")
DATASETS = Path("data/models/ps_native/datasets")

pytestmark = pytest.mark.skipif(
    not (V2_DIR / "model.joblib").is_file(),
    reason="v2 artifact not built (research/reproduction/train_ps_model_v2.py)",
)


@pytest.fixture(scope="module")
def model() -> PsNativeRiskModel:
    return PsNativeRiskModel.load(V2_DIR)


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    frame = pd.read_parquet(DATASETS / "validation.parquet")
    return frame.sample(1500, random_state=0).reset_index(drop=True)


def test_v2_is_the_default_and_declares_the_live_schema(model) -> None:
    default = PsNativeRiskModel.load()
    assert default.version == "ps_native_v3"
    assert model.feature_schema_version == PS_FEATURE_SCHEMA_VERSION
    assert model.features == list(CORE_PS_FEATURE_COLUMNS)


def test_the_manifest_does_not_claim_a_holdout_result(model) -> None:
    """The holdout is still /1 and sealed. A v2 test metric would be invented."""
    manifest = json.loads((V2_DIR / "manifest.json").read_text())
    assert manifest["holdout_evaluated"] is False
    assert model.holdout_evaluated is False
    evaluation = json.loads((V2_DIR / "evaluation.json").read_text())
    assert len(evaluation["folds"]) == 12
    assert "test" not in json.dumps(manifest["training_data"]).lower()


def test_missing_values_reach_the_model_as_nan(model, sample) -> None:
    """NaN is passed through, never replaced with 0.0 by this layer."""
    rows = sample.head(50).copy()
    rows["fee"] = np.nan
    assert np.isnan(model._matrix(rows)[:, model.features.index("fee")]).all()


def test_unseen_missingness_is_reported_not_hidden(model, sample) -> None:
    """The development data has no missing fee, so LightGBM learned no
    missing-value branch for it and treats NaN as 0.0 at inference. A capture
    with missing fees is therefore scored as if the fee were zero - a regime
    the model associates with licit activity. The drift report is where that
    becomes visible: reference missing rate 0, observed missing rate > 0."""
    rows = sample.copy()
    rows["fee"] = np.nan
    report = model.drift_report(rows)
    fee = report["features"]["fee"]
    assert fee["missing_rate_reference"] == 0.0
    assert fee["missing_rate_observed"] == 1.0
    assert "fee" in report["unseen_missingness_features"]
    assert report["status"] != "STABLE"


def test_explanations_are_exact_treeshap_with_per_feature_sign(model, sample) -> None:
    rows = sample.head(50)
    preds = model.predict_address_features(rows)
    contribs = model._contributions(model._matrix(rows))
    booster = model.model.booster_
    full = booster.predict(model._matrix(rows), pred_contrib=True)
    margin = booster.predict(model._matrix(rows), raw_score=True)
    # Local accuracy: contributions plus bias reproduce the model's margin.
    np.testing.assert_allclose(full.sum(axis=1), margin, rtol=1e-6, atol=1e-6)
    for i, p in enumerate(preds):
        assert p.explanation_method == EXPLANATION_TREESHAP
        for e in p.explanations:
            j = model.features.index(e.feature_name)
            assert e.contribution == pytest.approx(contribs[i, j])
            expected = "INCREASES_RISK" if contribs[i, j] > 0 else "DECREASES_RISK"
            assert e.direction == expected


def test_a_row_can_carry_both_directions(model, sample) -> None:
    """v1 gave every explained feature the row's direction. Real rows mix."""
    preds = model.predict_address_features(sample)
    mixed = [p for p in preds if len({e.direction for e in p.explanations}) == 2]
    assert mixed, "no row has features pushing in both directions"


def test_calibration_is_monotone_so_it_never_reorders(model) -> None:
    raw = np.linspace(0.001, 0.999, 500)
    assert np.all(np.diff(model.calibrate(raw)) > 0)


def test_severity_is_a_rank_budget(model, sample) -> None:
    scores = model.scores(sample)
    sev = model.severity(scores)
    n = len(scores)
    critical_cap = int(np.ceil(SEVERITY_BUDGET[0][1] * n))
    assert (sev == "CRITICAL").sum() <= critical_cap
    assert (sev != "LOW").sum() <= int(np.ceil(SEVERITY_BUDGET[-1][1] * n))
    # Rank-consistent: no LOW row outranks a CRITICAL row.
    if (sev == "CRITICAL").any() and (sev == "LOW").any():
        assert scores[sev == "CRITICAL"].min() >= scores[sev == "LOW"].max() or \
            scores[sev == "LOW"].max() <= model._platt["reference_prevalence"]


def test_a_below_base_rate_row_is_never_promoted(model) -> None:
    """A three-address run must not produce a CRITICAL just because one row is first."""
    floor = model._platt["reference_prevalence"]
    sev = model.severity(np.array([floor / 10, floor / 20, floor / 30]))
    assert set(sev) == {"LOW"}


def test_the_artifact_carries_a_drift_reference(model, sample) -> None:
    assert model.reference_profile is not None
    report = model.drift_report(sample)
    assert report["status"] in {"STABLE", "SHIFTED", "MAJOR_SHIFT"}
    assert set(report["features"]) == set(model.features)


def test_psi_is_zero_for_identical_and_large_for_disjoint_distributions() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=5000)
    frame = pd.DataFrame({"x": values, "n_txs_asof_t": 0})
    ref = monitoring.build_reference_profile(frame, ["x"], values)
    same = monitoring.compare_to_reference(ref, frame, values)
    assert same["features"]["x"]["psi"] == pytest.approx(0.0, abs=1e-9)
    moved = monitoring.compare_to_reference(ref, frame.assign(x=values + 10), values)
    assert moved["features"]["x"]["reading"] == "MAJOR_SHIFT"
