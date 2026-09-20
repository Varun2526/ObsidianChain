"""Train, Validate, Calibrate, and Freeze the PS-Native Supervised Risk Model.

Evaluates 3 candidate models on TRAIN only:
    1. Logistic Regression (scaled baseline)
    2. Random Forest
    3. LightGBM

Model selection is strictly based on VALIDATION PR-AUC against the no-skill baseline.
Calibration is performed exclusively on VALIDATION data.
The final model is evaluated ONCE on the held-out TEST set.
Outputs immutable artifacts to data/models/ps_native/v1/.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from obsidianchain.pipeline.features_ps import (
    CORE_PS_FEATURE_COLUMNS,
    PS_FEATURE_GROUPS,
)


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if k <= 0 or len(y_true) == 0:
        return 0.0
    top_indices = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[top_indices]))


def _derive_severity_bands(y_val: np.ndarray, cal_scores: np.ndarray) -> list[dict[str, Any]]:
    """Derive severity thresholds using validation precision targets."""
    targets = [("CRITICAL", 0.90), ("HIGH", 0.75), ("MEDIUM", 0.50)]
    bands = []
    # Sort descending by calibrated score
    order = np.argsort(-cal_scores)
    sorted_y = y_val[order]
    sorted_s = cal_scores[order]

    cum_pos = np.cumsum(sorted_y)
    cum_total = np.arange(1, len(sorted_y) + 1)
    precisions = cum_pos / cum_total

    for name, target_p in targets:
        # Find the lowest score cutoff that meets precision target with min support 20
        valid_idx = np.where((precisions >= target_p) & (cum_total >= 20))[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[-1]
            bands.append({
                "band": name,
                "target_precision": target_p,
                "threshold": float(sorted_s[best_idx]),
                "support": int(cum_total[best_idx]),
                "validation_precision": float(precisions[best_idx]),
                "populated": True,
            })
        else:
            # Fallback based on score percentiles
            fallback_thresh = 0.8 if name == "CRITICAL" else (0.5 if name == "HIGH" else 0.2)
            mask = cal_scores >= fallback_thresh
            bands.append({
                "band": name,
                "target_precision": target_p,
                "threshold": fallback_thresh,
                "support": int(mask.sum()),
                "validation_precision": float(y_val[mask].mean()) if mask.sum() > 0 else 0.0,
                "populated": False,
            })
    return bands


def run_training_pipeline(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    root = Path(data_dir) if data_dir is not None else Path(os.environ.get("OBSIDIANCHAIN_DATA", "data"))
    datasets_dir = root / "models" / "ps_native" / "datasets"
    v1_dir = Path(output_dir) if output_dir is not None else root / "models" / "ps_native" / "v1"
    v1_dir.mkdir(parents=True, exist_ok=True)

    train_path = datasets_dir / "train.parquet"
    val_path = datasets_dir / "validation.parquet"
    test_path = datasets_dir / "test.parquet"
    manifest_path = datasets_dir / "manifest.json"

    for p in [train_path, val_path, test_path, manifest_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required dataset file: {p}")

    ds_manifest = json.loads(manifest_path.read_text())
    training_ds_sha256 = ds_manifest["artifact_hashes"]["train.parquet"]

    print("Loading train, validation, and test parquets...")
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    features = list(CORE_PS_FEATURE_COLUMNS)
    X_train, y_train = df_train[features].to_numpy(dtype=np.float32), df_train["y"].to_numpy(dtype=np.int8)
    X_val, y_val = df_val[features].to_numpy(dtype=np.float32), df_val["y"].to_numpy(dtype=np.int8)
    X_test, y_test = df_test[features].to_numpy(dtype=np.float32), df_test["y"].to_numpy(dtype=np.int8)

    no_skill_baseline = float(np.mean(y_val))
    print(f"Validation positive prevalence (No-Skill PR Baseline): {no_skill_baseline:.4f}")

    # -----------------------------------------------------------------------
    # Candidate Model Definitions
    # -----------------------------------------------------------------------
    candidates = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=20260919, class_weight="balanced")),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=20,
            random_state=20260919,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=20260919,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    val_results = {}
    fitted_models = {}

    print("\n--- Training & Evaluating Candidate Models on Validation ---")
    for name, model in candidates.items():
        t_m0 = time.time()
        print(f"Fitting {name} on TRAIN ({len(X_train)} samples)...")
        model.fit(X_train, y_train)
        fit_time = time.time() - t_m0

        raw_val_scores = model.predict_proba(X_val)[:, 1]
        val_pr_auc = float(average_precision_score(y_val, raw_val_scores))
        val_roc_auc = float(roc_auc_score(y_val, raw_val_scores))
        val_brier = float(brier_score_loss(y_val, raw_val_scores))
        p_at_100 = _precision_at_k(y_val, raw_val_scores, 100)
        p_at_500 = _precision_at_k(y_val, raw_val_scores, 500)
        f1 = float(f1_score(y_val, raw_val_scores >= 0.5))

        val_results[name] = {
            "val_pr_auc": val_pr_auc,
            "val_roc_auc": val_roc_auc,
            "val_brier": val_brier,
            "precision_at_100": p_at_100,
            "precision_at_500": p_at_500,
            "f1_at_0.5": f1,
            "fit_time_seconds": round(fit_time, 2),
            "beats_baseline": bool(val_pr_auc > no_skill_baseline),
        }
        fitted_models[name] = model
        print(f"  {name}: PR-AUC={val_pr_auc:.4f} (baseline={no_skill_baseline:.4f}), ROC-AUC={val_roc_auc:.4f}, P@100={p_at_100:.4f}")

    # -----------------------------------------------------------------------
    # Model Selection (Strictly on Validation PR-AUC)
    # -----------------------------------------------------------------------
    best_model_name = max(val_results, key=lambda k: val_results[k]["val_pr_auc"])
    best_model = fitted_models[best_model_name]
    print(f"\nWinning Model Selected: {best_model_name} (PR-AUC={val_results[best_model_name]['val_pr_auc']:.4f})")

    # -----------------------------------------------------------------------
    # Calibration on VALIDATION Data Only
    # -----------------------------------------------------------------------
    print("\nFitting Isotonic Calibrator on VALIDATION split...")
    raw_val_preds = best_model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val_preds, y_val)

    cal_val_preds = calibrator.predict(raw_val_preds)
    cal_val_brier = float(brier_score_loss(y_val, cal_val_preds))
    cal_val_pr_auc = float(average_precision_score(y_val, cal_val_preds))
    print(f"Calibrated Validation Brier: {cal_val_brier:.4f} (raw={val_results[best_model_name]['val_brier']:.4f})")

    # Severity bands from calibrated validation probabilities
    severity_bands = _derive_severity_bands(y_val, cal_val_preds)
    print("Derived Severity Bands:")
    for b in severity_bands:
        print(f"  {b['band']}: threshold={b['threshold']:.4f}, target={b['target_precision']:.2f}, support={b['support']}")

    # -----------------------------------------------------------------------
    # Final Test Set Evaluation (Held Out until Model Freeze)
    # -----------------------------------------------------------------------
    print("\n--- Final Single Evaluation on TEST Split ---")
    raw_test_preds = best_model.predict_proba(X_test)[:, 1]
    cal_test_preds = calibrator.predict(raw_test_preds)

    test_pr_auc = float(average_precision_score(y_test, cal_test_preds))
    test_roc_auc = float(roc_auc_score(y_test, cal_test_preds))
    test_brier = float(brier_score_loss(y_test, cal_test_preds))
    test_p_at_100 = _precision_at_k(y_test, cal_test_preds, 100)
    test_p_at_500 = _precision_at_k(y_test, cal_test_preds, 500)
    test_f1 = float(f1_score(y_test, cal_test_preds >= 0.5))
    test_prevalence = float(np.mean(y_test))

    test_metrics = {
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "test_brier": test_brier,
        "precision_at_100": test_p_at_100,
        "precision_at_500": test_p_at_500,
        "f1_at_0.5": test_f1,
        "test_positive_prevalence": test_prevalence,
    }
    print(f"TEST PR-AUC: {test_pr_auc:.4f} (baseline={test_prevalence:.4f})")
    print(f"TEST ROC-AUC: {test_roc_auc:.4f}")
    print(f"TEST P@100:   {test_p_at_100:.4f}")

    # -----------------------------------------------------------------------
    # Freeze Model Artifacts
    # -----------------------------------------------------------------------
    print("\nSaving frozen artifacts to data/models/ps_native/v1/...")
    model_artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "calibrator": calibrator,
        "features": features,
        "severity_bands": severity_bands,
    }
    model_joblib_path = v1_dir / "model.joblib"
    joblib.dump(model_artifact, model_joblib_path)
    model_sha256 = _compute_sha256(model_joblib_path)

    feature_schema = {
        "schema_version": "ps_native_features/1",
        "feature_count": len(features),
        "feature_list": features,
        "feature_groups": PS_FEATURE_GROUPS,
    }
    (v1_dir / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2))

    calibration_info = {
        "method": "IsotonicRegression",
        "dataset": "validation.parquet (timesteps 35-41)",
        "calibrated_brier": cal_val_brier,
        "raw_brier": val_results[best_model_name]["val_brier"],
        "severity_bands": severity_bands,
    }
    (v1_dir / "calibration.json").write_text(json.dumps(calibration_info, indent=2))

    metrics_report = {
        "no_skill_baseline_pr_auc": no_skill_baseline,
        "validation_candidate_comparison": val_results,
        "winning_model": best_model_name,
        "validation_winning_metrics": val_results[best_model_name],
        "test_metrics": test_metrics,
    }
    (v1_dir / "metrics.json").write_text(json.dumps(metrics_report, indent=2))

    model_manifest = {
        "schema": "obsidianchain.ps_model_manifest/1",
        "model_version": "ps_native_v1",
        "model_type": best_model_name,
        "feature_schema_version": "ps_native_features/1",
        "training_dataset_sha256": training_ds_sha256,
        "model_sha256": model_sha256,
        "features": features,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "artifacts": {
            "model.joblib": model_sha256,
            "feature_schema.json": _compute_sha256(v1_dir / "feature_schema.json"),
            "calibration.json": _compute_sha256(v1_dir / "calibration.json"),
            "metrics.json": _compute_sha256(v1_dir / "metrics.json"),
        },
        "metrics_summary": {
            "val_pr_auc": val_results[best_model_name]["val_pr_auc"],
            "test_pr_auc": test_pr_auc,
            "test_roc_auc": test_roc_auc,
            "test_precision_at_100": test_p_at_100,
        },
    }
    (v1_dir / "manifest.json").write_text(json.dumps(model_manifest, indent=2))
    print(f"Artifacts successfully written and hashed in {v1_dir}.")
    print(f"Model SHA-256: {model_sha256}")
    print(f"Total pipeline runtime: {time.time()-t0:.2f}s")
    return model_manifest


if __name__ == "__main__":
    run_training_pipeline()
