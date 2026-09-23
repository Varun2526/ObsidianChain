"""Run-time trust checks for the PS-native model.

A risk score is only as good as the match between the data it is applied to
and the data it was trained on. On the development data a classifier can tell
t<=34 from t35-41 with AUC 0.999 and the median fee rises about 50x across the
period, so drift is the expected state, not an edge case. This module makes
that visible on every run instead of leaving it to be discovered.

The training script stores a reference profile inside the model artifact. At
inference the same profile is recomputed over the scored rows and compared:

* PSI per feature, on the reference's own quantile bins. The usual reading:
  < 0.10 stable, 0.10-0.25 shifted, >= 0.25 major shift.
* Missing-value rate per feature, reference against observed.
* Cold-start share - rows with no prior history. The model's weakest
  population (it is 88.6% of the development data).
* Score distribution: PSI of the raw score, and the mean score, which is the
  model's own estimate of prevalence.

Nothing here changes a score. It reports how far to trust one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: Quantile bins per feature. Ten is the conventional PSI resolution.
N_BINS = 10

#: PSI reading thresholds (conventional, Siddiqi 2006).
PSI_SHIFTED = 0.10
PSI_MAJOR = 0.25

#: Floor on a bin share before the log in PSI. Keeps an empty bin finite.
_EPS = 1e-4


def _edges(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    qs = np.unique(np.quantile(finite, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    return [float(q) for q in qs]


def _shares(values: np.ndarray, edges: list[float]) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0.0] * (len(edges) + 1)
    idx = np.searchsorted(np.asarray(edges), finite, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return [float(c) / finite.size for c in counts]


def psi(expected: list[float], observed: list[float]) -> float:
    """Population stability index between two share vectors on the same bins."""
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    o = np.clip(np.asarray(observed, dtype=float), _EPS, None)
    return float(np.sum((o - e) * np.log(o / e)))


def _cold_start_share(frame: pd.DataFrame) -> float:
    if "n_txs_asof_t" not in frame.columns or frame.empty:
        return float("nan")
    return float((frame["n_txs_asof_t"] == 0).mean())


def build_reference_profile(frame: pd.DataFrame, features: list[str],
                            raw_scores: np.ndarray) -> dict[str, Any]:
    """The training-time profile stored inside a model artifact."""
    per_feature = {}
    for name in features:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        edges = _edges(values)
        per_feature[name] = {
            "edges": edges,
            "shares": _shares(values, edges),
            "missing_rate": float(np.mean(~np.isfinite(values))),
        }
    scores = np.asarray(raw_scores, dtype=float)
    score_edges = _edges(scores)
    return {
        "n_rows": int(len(frame)),
        "features": per_feature,
        "score": {"edges": score_edges, "shares": _shares(scores, score_edges),
                  "mean": float(np.mean(scores)) if scores.size else float("nan")},
        "cold_start_share": _cold_start_share(frame),
    }


def _reading(value: float) -> str:
    if not np.isfinite(value):
        return "UNMEASURED"
    if value >= PSI_MAJOR:
        return "MAJOR_SHIFT"
    if value >= PSI_SHIFTED:
        return "SHIFTED"
    return "STABLE"


def compare_to_reference(reference: dict[str, Any] | None, frame: pd.DataFrame,
                         raw_scores: np.ndarray) -> dict[str, Any]:
    """Drift report for one run. ``reference`` None means the model has none."""
    if not reference:
        return {"status": "NO_REFERENCE_PROFILE",
                "reason": "the model artifact carries no training-time profile"}
    if frame.empty:
        return {"status": "NO_ROWS_SCORED"}

    features = {}
    for name, ref in reference["features"].items():
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        value = psi(ref["shares"], _shares(values, ref["edges"])) if ref["edges"] else float("nan")
        features[name] = {
            "psi": round(value, 4),
            "reading": _reading(value),
            "missing_rate_reference": round(ref["missing_rate"], 4),
            "missing_rate_observed": round(float(np.mean(~np.isfinite(values))), 4),
        }

    scores = np.asarray(raw_scores, dtype=float)
    score_ref = reference["score"]
    score_psi = psi(score_ref["shares"], _shares(scores, score_ref["edges"])) if score_ref["edges"] else float("nan")
    shifted = sorted((n for n, f in features.items() if f["reading"] != "STABLE"),
                     key=lambda n: -features[n]["psi"])
    major = [n for n in shifted if features[n]["reading"] == "MAJOR_SHIFT"]

    # A feature that was never missing in training has no learned
    # missing-value branch: a tree model routes NaN as if it were 0.0. That
    # is a silent substitution, so it is named separately and never STABLE.
    unseen_missing = sorted(
        n for n, f in features.items()
        if f["missing_rate_reference"] == 0.0 and f["missing_rate_observed"] > 0.0
    )

    status = "STABLE"
    if major or _reading(score_psi) == "MAJOR_SHIFT":
        status = "MAJOR_SHIFT"
    elif shifted or unseen_missing or _reading(score_psi) == "SHIFTED":
        status = "SHIFTED"

    return {
        "status": status,
        "n_rows_scored": int(len(frame)),
        "reference_rows": reference["n_rows"],
        "score_psi": round(score_psi, 4),
        "score_reading": _reading(score_psi),
        "mean_raw_score_reference": round(score_ref["mean"], 4),
        "mean_raw_score_observed": round(float(np.mean(scores)), 4),
        "cold_start_share_reference": round(reference["cold_start_share"], 4),
        "cold_start_share_observed": round(_cold_start_share(frame), 4),
        "shifted_features": shifted,
        "major_shift_features": major,
        "unseen_missingness_features": unseen_missing,
        "features": features,
        "reading_note": ("PSI < 0.10 stable, 0.10-0.25 shifted, >= 0.25 major. "
                         "A shifted run is still scored; the shift is reported so the "
                         "score is read with it."),
    }
