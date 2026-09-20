"""PS-Native Supervised Risk Model Loader, Calibrator, and Explainer.

Loads the frozen model artifact from data/models/ps_native/v1/,
validates cryptographic integrity and feature schemas, executes calibrated
inference, and emits explainable MODEL_SIGNAL outputs.
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

from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "txid": self.txid,
            "raw_risk_score": round(self.raw_risk_score, 4),
            "calibrated_risk_score": round(self.calibrated_risk_score, 4),
            "severity": self.severity,
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
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

    @classmethod
    def load(cls, model_dir: str | Path | None = None) -> PsNativeRiskModel:
        """Load model from disk and verify cryptographic hash against manifest."""
        base_dir = Path(model_dir) if model_dir is not None else Path("data") / "models" / "ps_native" / "v1"
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
        for b in self.severity_bands:
            if calibrated_score >= b["threshold"]:
                return b["band"]
        return "LOW"

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated probabilities matching model scoring protocol."""
        if frame.empty:
            return np.array([], dtype=np.float32)
        X = frame[self.features].to_numpy(dtype=np.float32)
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        raw_probs = self.model.predict_proba(X_clean)[:, 1]
        return self.calibrator.predict(raw_probs)

    def severity(self, scores: Any) -> np.ndarray:
        """Severity bands matching model scoring protocol."""
        return np.array([self.assign_severity(float(s)) for s in scores], dtype=object)

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
        X = features_df[self.features].to_numpy(dtype=np.float32)
        # Handle any NaNs if model requires, or fill with 0 where appropriate
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Raw probabilities
        raw_probs = self.model.predict_proba(X_clean)[:, 1]
        # Calibrated probabilities
        cal_probs = self.calibrator.predict(raw_probs)

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
            sev = self.assign_severity(cal_p)

            # Generate top 3 contributing factors
            row_vals = X_clean[idx]
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
