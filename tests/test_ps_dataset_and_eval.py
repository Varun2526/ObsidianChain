"""Dataset and Evaluation tests for PS-native supervised model pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

from obsidianchain.pipeline.features_ps import (
    CORE_PS_FEATURE_COLUMNS,
    PS_FEATURE_SCHEMA_VERSION,
)


DATASET_DIR = Path("data/models/ps_native/datasets")
MODEL_DIR = Path("data/models/ps_native/v1")


class TestPsDatasetIntegrity:
    """Validate dataset split boundaries, label correctness, and unknown exclusion."""

    def test_dataset_files_and_manifest_exist(self) -> None:
        assert (DATASET_DIR / "train.parquet").is_file()
        assert (DATASET_DIR / "validation.parquet").is_file()
        assert (DATASET_DIR / "test.parquet").is_file()
        assert (DATASET_DIR / "manifest.json").is_file()

        manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
        # The development files carry the live schema; the sealed holdout
        # was never regenerated and still declares /1. Both are recorded so
        # nothing can fit on one and score on the other unnoticed.
        assert manifest["feature_schema_version"] == PS_FEATURE_SCHEMA_VERSION
        assert manifest["development_schema_version"] == PS_FEATURE_SCHEMA_VERSION
        assert manifest["holdout_schema_version"] == "ps_native_features/1"
        assert manifest["provenance_type"] == "RESEARCH_DERIVED_CANONICAL_TRAINING_REPRESENTATION"
        assert manifest["timestamp_source"] == "ELLIPTIC_TIMESTEP_SURROGATE"

    def test_chronological_split_boundaries(self) -> None:
        """Verify train is timesteps 1-34, val is 35-41, test is 42-49."""
        manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
        splits = manifest["split_boundaries"]

        assert splits["train"] == "step 1-34"
        assert splits["validation"] == "step 35-41"
        assert splits["test"] == "step 42-49"

    def test_unknown_labels_strictly_excluded(self) -> None:
        """Verify only binary labels (0, 1) exist; unknown (class 3) must be excluded."""
        train_df = pd.read_parquet(DATASET_DIR / "train.parquet", columns=["y"])
        val_df = pd.read_parquet(DATASET_DIR / "validation.parquet", columns=["y"])
        test_df = pd.read_parquet(DATASET_DIR / "test.parquet", columns=["y"])

        for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            unique_labels = set(df["y"].unique())
            assert unique_labels == {0, 1}, f"{name} set contains unexpected labels: {unique_labels}"

    def test_dataset_sha256_reproducibility(self) -> None:
        """Assert dataset parquet files match the recorded SHA-256 in manifest."""
        manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
        for split_name in ["train", "validation", "test"]:
            filename = f"{split_name}.parquet"
            file_path = DATASET_DIR / filename
            actual_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
            expected_sha = manifest["artifact_hashes"][filename]
            assert actual_sha == expected_sha, f"Dataset hash mismatch for {split_name}!"


class TestPsModelArtifactsAndMetrics:
    """Validate frozen model artifacts, schema manifest, calibration, and metrics."""

    def test_model_artifacts_and_hashes(self) -> None:
        assert (MODEL_DIR / "model.joblib").is_file()
        assert (MODEL_DIR / "feature_schema.json").is_file()
        assert (MODEL_DIR / "calibration.json").is_file()
        assert (MODEL_DIR / "metrics.json").is_file()
        assert (MODEL_DIR / "manifest.json").is_file()

        manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
        actual_model_sha = hashlib.sha256((MODEL_DIR / "model.joblib").read_bytes()).hexdigest()
        assert manifest["model_sha256"] == actual_model_sha

    def test_evaluation_metrics_and_no_skill_baseline(self) -> None:
        metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
        no_skill = metrics["no_skill_baseline_pr_auc"]
        val_m = metrics["validation_winning_metrics"]
        test_m = metrics["test_metrics"]

        # Validation PR-AUC must beat no-skill baseline (prevalence)
        assert val_m["val_pr_auc"] > no_skill
        # Random Forest beat LightGBM and LogReg on Validation PR-AUC
        assert val_m["val_pr_auc"] >= 0.50
        assert val_m["val_roc_auc"] >= 0.80

        # Test PR-AUC must beat test no-skill baseline
        assert test_m["test_pr_auc"] > test_m["test_positive_prevalence"]
        assert test_m["test_roc_auc"] >= 0.70

    def test_calibration_parameters_validity(self) -> None:
        calib = json.loads((MODEL_DIR / "calibration.json").read_text())
        assert calib["method"] == "IsotonicRegression"
        assert calib["calibrated_brier"] < calib["raw_brier"]

        # Check severity thresholds
        bands = {b["band"]: b["threshold"] for b in calib["severity_bands"]}
        assert 0.0 < bands["MEDIUM"] < bands["HIGH"] < bands["CRITICAL"] <= 1.0
