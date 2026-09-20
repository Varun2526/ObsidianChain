"""Unit and Contract Tests for PS-Native Model Loading and Inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obsidianchain.ml.ps_model import PsNativeRiskModel
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS


@pytest.fixture(scope="module")
def ps_model():
    return PsNativeRiskModel.load("data/models/ps_native/v1")


class TestPsNativeModel:
    """Validate model loading, schema integrity, and inference contracts."""

    def test_model_loads_and_verifies_hash(self, ps_model) -> None:
        assert ps_model.version == "ps_native_v1"
        assert len(ps_model.features) == 30
        assert ps_model.features == CORE_PS_FEATURE_COLUMNS
        assert len(ps_model._sha256) == 64

    def test_model_inference_valid_range_and_severity(self, ps_model) -> None:
        # Create synthetic feature row
        row_data = {feat: 1.0 for feat in CORE_PS_FEATURE_COLUMNS}
        row_data["address"] = "1TestAddr"
        row_data["txid"] = "tx_test_01"

        df = pd.DataFrame([row_data])
        preds = ps_model.predict_address_features(df)

        assert len(preds) == 1
        p = preds[0]
        assert p.address == "1TestAddr"
        assert p.txid == "tx_test_01"
        assert 0.0 <= p.raw_risk_score <= 1.0
        assert 0.0 <= p.calibrated_risk_score <= 1.0
        assert p.severity in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        assert len(p.explanations) > 0
        assert p.model_version == "ps_native_v1"
        assert p.model_sha256 == ps_model._sha256

    def test_model_schema_mismatch_raises_error(self, ps_model) -> None:
        # Missing required features
        incomplete_df = pd.DataFrame([{"address": "1Incomplete", "input_count": 2}])
        with pytest.raises(ValueError, match="Feature schema mismatch"):
            ps_model.predict_address_features(incomplete_df)

    def test_deterministic_output(self, ps_model) -> None:
        row_data = {feat: float(i % 5) for i, feat in enumerate(CORE_PS_FEATURE_COLUMNS)}
        row_data["address"] = "1DetAddr"
        row_data["txid"] = "tx_det_01"
        df = pd.DataFrame([row_data])

        p1 = ps_model.predict_address_features(df)[0]
        p2 = ps_model.predict_address_features(df)[0]

        assert p1.raw_risk_score == pytest.approx(p2.raw_risk_score)
        assert p1.calibrated_risk_score == pytest.approx(p2.calibrated_risk_score)
        assert p1.severity == p2.severity
