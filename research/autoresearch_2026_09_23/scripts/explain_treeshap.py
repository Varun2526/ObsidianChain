"""Prototype explanation function for the v2 development candidate model.

Replaces ps_model.py's REJECTED explanation method (global
feature_importances_ * |value|, single row-level direction from whether
calibrated_p >= 0.5 - see 10_explanation_audit.md: 33.9% direction agreement
with true TreeSHAP, worse than chance) with exact per-prediction TreeSHAP,
verified locally exact to floating-point precision in exp05.

This is a PROTOTYPE, not wired into src/obsidianchain/ml/ps_model.py.
Promoting it is an integration decision for whoever promotes the v2
candidate artifact, not something this research program does unilaterally.

Usage:
    from explain_treeshap import explain_predictions
    explanations = explain_predictions(model, features_df, feature_names)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TreeShapExplanation:
    feature_name: str
    feature_value: float
    shap_contribution: float
    direction: str  # "INCREASES_RISK" or "DECREASES_RISK" - derived from
                     # THIS feature's own SHAP sign, never from the row's
                     # overall score crossing a threshold.

    def as_dict(self) -> dict:
        return {
            "feature": self.feature_name,
            "value": round(self.feature_value, 4),
            "contribution": round(self.shap_contribution, 4),
            "direction": self.direction,
        }


def explain_predictions(
    model,
    features_df: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 3,
) -> list[list[TreeShapExplanation]]:
    """Exact per-row TreeSHAP explanations for a fitted LightGBM model.

    Requires a LightGBM model specifically (Booster.predict(pred_contrib=True)
    is LightGBM's own built-in, exact TreeSHAP implementation - no external
    `shap` package needed, which matters because `shap` is not installed /
    not in this project's vendored offline wheel set (01_repo_audit.md).
    A different model family (RandomForest, HistGradientBoosting) would need
    the external `shap` package instead - not available in this environment,
    which is itself a concrete, evidenced reason (10_explanation_audit.md)
    to prefer LightGBM if this candidate is promoted.
    """
    booster = model.booster_ if hasattr(model, "booster_") else model
    X = np.nan_to_num(features_df[feature_names].to_numpy("float32"))
    shap_values = booster.predict(X, pred_contrib=True)  # (n, n_features + 1)
    shap_feature_part = shap_values[:, :-1]

    results = []
    for row_idx in range(len(features_df)):
        row_shap = shap_feature_part[row_idx]
        row_vals = X[row_idx]
        top_idx = np.argsort(-np.abs(row_shap))[:top_k]
        row_explanations = [
            TreeShapExplanation(
                feature_name=feature_names[i],
                feature_value=float(row_vals[i]),
                shap_contribution=float(row_shap[i]),
                direction="INCREASES_RISK" if row_shap[i] > 0 else "DECREASES_RISK",
            )
            for i in top_idx
        ]
        results.append(row_explanations)
    return results


def local_fidelity_check(model, features_df: pd.DataFrame, feature_names: list[str]) -> float:
    """Returns max|sum(shap) + bias - raw_margin| over the given rows.

    Call this before trusting any explanation output on a NEW model - this
    is what exp05 ran (result: 1.91e-14, floating-point exact) before
    treating TreeSHAP as ground truth. Cheap, and catches a booster/version
    mismatch immediately rather than silently.
    """
    booster = model.booster_ if hasattr(model, "booster_") else model
    X = np.nan_to_num(features_df[feature_names].to_numpy("float32"))
    shap_values = booster.predict(X, pred_contrib=True)
    raw_margin = booster.predict(X, raw_score=True)
    reconstructed = shap_values[:, :-1].sum(axis=1) + shap_values[:, -1]
    return float(np.max(np.abs(reconstructed - raw_margin)))
