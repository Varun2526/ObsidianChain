"""PS-Native Supervised Risk Model Loader, Calibrator, and Explainer.

Loads a frozen model artifact (``data/models/ps_native/v3/`` by default;
v2 and v1 still loadable), validates cryptographic integrity and feature schemas,
executes calibrated inference, and emits explainable MODEL_SIGNAL outputs.

v2 differs from v1 in four ways, each fixing a measured defect:

* Missing values reach the model as NaN. LightGBM routes them natively; v1
  replaced them with 0.0, which is a value the data can genuinely take.
* The calibrator is Platt on the logit of the raw score, fit out-of-fold.
  Isotonic collapsed scores to a few dozen levels and did not improve Brier
  over the 12 development folds.
* Explanations are exact per-row TreeSHAP (``pred_contrib``), each feature
  with its own sign. v1 multiplied global importance by ``|value|`` and gave
  every feature the row's direction; 66% of those directions were wrong
  against TreeSHAP (``10_explanation_audit.md``).
* Severity is a rank budget within the run. Static probability bands,
  derived correctly, still failed out-of-time (``09_alert_policy_analysis.md``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from obsidianchain.ml import monitoring
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS

#: Default artifact directory: v3, trained on the live feature schema (/4).
#: v2 (/3) and v1 (/1) are kept, frozen, for reproducibility.
DEFAULT_MODEL_DIR = Path("data") / "models" / "ps_native" / "v3"

#: Share of a run's scored rows placed in each band, most severe first. A
#: row above the top budgets is LOW. Budgets rather than probability
#: thresholds, because an analyst's capacity is fixed and a probability
#: threshold drifts with prevalence.
SEVERITY_BUDGET = (("CRITICAL", 0.01), ("HIGH", 0.05), ("MEDIUM", 0.15))

EXPLANATION_TREESHAP = "TREESHAP_PRED_CONTRIB"
EXPLANATION_LEGACY = "LEGACY_IMPORTANCE_X_VALUE"


@dataclass(frozen=True)
class PsFeatureExplanation:
    feature_name: str
    feature_value: float
    contribution: float
    direction: str  # "INCREASES_RISK" or "DECREASES_RISK"


@dataclass
class PsModelPrediction:
    address: str
    txid: str | None
    raw_risk_score: float
    calibrated_risk_score: float
    severity: str
    explanations: list[PsFeatureExplanation]
    model_version: str
    model_sha256: str
    explanation_method: str = EXPLANATION_LEGACY

    def as_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "txid": self.txid,
            "raw_risk_score": round(self.raw_risk_score, 4),
            "calibrated_risk_score": round(self.calibrated_risk_score, 4),
            "severity": self.severity,
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
            "explanation_method": self.explanation_method,
            "explanations": [
                {
                    "feature": e.feature_name,
                    "value": round(e.feature_value, 4),
                    "contribution": round(e.contribution, 4),
                    "direction": e.direction,
                }
                for e in self.explanations
            ],
        }


class PsNativeRiskModel:
    """Offline runner for the frozen PS-native supervised risk model."""

    def __init__(
        self,
        model_artifact: dict[str, Any],
        manifest: dict[str, Any],
        model_sha256: str,
    ) -> None:
        self._artifact = model_artifact
        self._manifest = manifest
        self._sha256 = model_sha256

        self.model = model_artifact["model"]
        self.calibrator = model_artifact["calibrator"]
        self.features: list[str] = list(model_artifact["features"])
        self.severity_bands: list[dict[str, Any]] = model_artifact.get("severity_bands", [])
        self.version: str = manifest.get("model_version", "ps_native_v1")
        self.feature_schema_version: str | None = manifest.get("feature_schema_version")
        self.reference_profile: dict[str, Any] | None = model_artifact.get("reference_profile")
        self.holdout_evaluated: bool = bool(manifest.get("holdout_evaluated", True))
        # A dict calibrator is Platt (v2); anything else is a fitted
        # estimator with ``predict`` (v1 isotonic).
        self._platt = self.calibrator if isinstance(self.calibrator, dict) else None
        self._native_missing = self._platt is not None and hasattr(self.model, "booster_")

    @classmethod
    def load(cls, model_dir: str | Path | None = None) -> PsNativeRiskModel:
        """Load model from disk and verify cryptographic hash against manifest."""
        base_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        joblib_path = base_dir / "model.joblib"
        manifest_path = base_dir / "manifest.json"

        if not joblib_path.is_file():
            raise FileNotFoundError(f"PS-Native model artifact not found at {joblib_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"PS-Native model manifest not found at {manifest_path}")

        # Compute digest
        actual_sha256 = hashlib.sha256(joblib_path.read_bytes()).hexdigest()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha256 = manifest.get("model_sha256")

        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(
                f"PS-Native model integrity verification failed! "
                f"Expected SHA-256 {expected_sha256}, got {actual_sha256}"
            )

        artifact = joblib.load(joblib_path)
        return cls(model_artifact=artifact, manifest=manifest, model_sha256=actual_sha256)

    def assign_severity(self, calibrated_score: float) -> str:
        """v1 static bands. v2 has none and uses :meth:`severity` instead."""
        for b in self.severity_bands:
            if calibrated_score >= b["threshold"]:
                return b["band"]
        return "LOW"

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[self.features].to_numpy(dtype=np.float32)
        if self._native_missing:
            # Infinities are not a value; NaN is the model's missing marker.
            X[~np.isfinite(X)] = np.nan
            return X
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        # Named columns, because the v2 estimator was fitted on a frame.
        if self._native_missing:
            X = pd.DataFrame(X, columns=self.features)
        return self.model.predict_proba(X)[:, 1]

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Display probability. Monotone in the raw score, so order is kept."""
        raw_probs = np.asarray(raw_probs, dtype=float)
        if self._platt is None:
            return self.calibrator.predict(raw_probs)
        p = np.clip(raw_probs, 1e-6, 1 - 1e-6)
        z = self._platt["coef"] * np.log(p / (1 - p)) + self._platt["intercept"]
        return 1.0 / (1.0 + np.exp(-z))

    def raw_scores(self, frame: pd.DataFrame) -> np.ndarray:
        """The ranking score."""
        if frame.empty:
            return np.array([], dtype=np.float32)
        return self._predict(self._matrix(frame))

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated probabilities matching model scoring protocol."""
        if frame.empty:
            return np.array([], dtype=np.float32)
        return self.calibrate(self.raw_scores(frame))

    def severity(self, scores: Any) -> np.ndarray:
        """Severity per row.

        v1: static calibrated-probability bands. v2: rank budget within the
        scored batch (:data:`SEVERITY_BUDGET`), and a row is only placed
        above LOW when its probability is also above the training reference
        prevalence - a small run must not promote a below-base-rate row to
        CRITICAL just because something has to be first.
        """
        scores = np.asarray(scores, dtype=float)
        if self._platt is None:
            return np.array([self.assign_severity(float(s)) for s in scores], dtype=object)
        out = np.array(["LOW"] * len(scores), dtype=object)
        if not len(scores):
            return out
        order = np.argsort(-scores, kind="stable")
        floor = float(self._platt.get("reference_prevalence", 0.0))
        start = 0
        for band, share in SEVERITY_BUDGET:
            end = min(len(scores), int(np.ceil(share * len(scores))))
            for idx in order[start:end]:
                if scores[idx] > floor:
                    out[idx] = band
            start = max(start, end)
        return out

    def drift_report(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Training-reference comparison for the rows being scored."""
        if frame.empty or any(f not in frame.columns for f in self.features):
            return {"status": "NO_ROWS_SCORED"}
        return monitoring.compare_to_reference(
            self.reference_profile, frame, self.raw_scores(frame))

    def _contributions(self, X: np.ndarray) -> np.ndarray | None:
        """Per-row TreeSHAP in log-odds, bias column dropped. None if unsupported."""
        booster = getattr(self.model, "booster_", None)
        if booster is None:
            return None
        return np.asarray(booster.predict(X, pred_contrib=True))[:, :-1]

    def predict_address_features(
        self,
        features_df: pd.DataFrame,
    ) -> list[PsModelPrediction]:
        """Score a DataFrame of address-as-of-t feature rows."""
        if features_df.empty:
            return []

        # Validate required features are present
        missing = [f for f in self.features if f not in features_df.columns]
        if missing:
            raise ValueError(f"Feature schema mismatch: missing required features {missing[:5]}")

        # Extract strictly in expected feature order
        X_clean = self._matrix(features_df)

        # Raw probabilities
        raw_probs = self._predict(X_clean)
        # Calibrated probabilities
        cal_probs = self.calibrate(raw_probs)
        severities = self.severity(cal_probs)
        shap = self._contributions(X_clean) if self._native_missing else None

        # Feature importances / contributions
        # For tree models, use global feature_importances_ weighted by standardized values
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None and hasattr(self.model, "named_steps"):
            clf = self.model.named_steps.get("clf")
            importances = getattr(clf, "feature_importances_", None)

        if importances is None:
            importances = np.ones(len(self.features), dtype=np.float32) / len(self.features)

        results: list[PsModelPrediction] = []
        for i, row in features_df.iterrows():
            idx = features_df.index.get_loc(i)
            addr = str(row.get("address") or f"addr_{idx}")
            txid = str(row.get("txid") or "") or None
            raw_p = float(raw_probs[idx])
            cal_p = float(cal_probs[idx])
            sev = str(severities[idx])
            row_vals = X_clean[idx]

            if shap is not None:
                # Exact TreeSHAP: each feature's own signed contribution to
                # this row's log-odds.
                contrib_row = shap[idx]
                top = np.argsort(-np.abs(contrib_row), kind="stable")[:3]
                exps = [
                    PsFeatureExplanation(
                        feature_name=self.features[j],
                        feature_value=float(row_vals[j]),
                        contribution=float(contrib_row[j]),
                        direction="INCREASES_RISK" if contrib_row[j] > 0 else "DECREASES_RISK",
                    )
                    for j in top
                ]
                results.append(PsModelPrediction(
                    address=addr, txid=txid,
                    raw_risk_score=raw_p, calibrated_risk_score=cal_p,
                    severity=sev, explanations=exps,
                    model_version=self.version, model_sha256=self._sha256,
                    explanation_method=EXPLANATION_TREESHAP,
                ))
                continue

            # v1 legacy proxy, kept only so the frozen v1 artifact still runs.
            # Generate top 3 contributing factors
            # Contribution proxy: importance * normalized value
            contribs = importances * np.abs(row_vals)
            top_k_indices = np.argsort(-contribs)[:3]

            exps = []
            for feat_idx in top_k_indices:
                f_name = self.features[feat_idx]
                f_val = float(row_vals[feat_idx])
                f_contrib = float(contribs[feat_idx])
                # If feature is high and risk is high, it increases risk
                direction = "INCREASES_RISK" if cal_p >= 0.5 else "DECREASES_RISK"
                exps.append(PsFeatureExplanation(
                    feature_name=f_name,
                    feature_value=f_val,
                    contribution=f_contrib,
                    direction=direction,
                ))

            results.append(PsModelPrediction(
                address=addr,
                txid=txid,
                raw_risk_score=raw_p,
                calibrated_risk_score=cal_p,
                severity=sev,
                explanations=exps,
                model_version=self.version,
                model_sha256=self._sha256,
            ))

        return results
