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
        """The frozen artifact is intact and still declares its own schema.

        It declares ps_native_features/1 (30 columns). The live pipeline has
        moved to /2 (24 columns) after the generator fix, so the model is
        asserted against its OWN feature list rather than the live one - a
        model declares what it needs, and comparing it to whatever the
        pipeline currently emits would make this test fail for the wrong
        reason.
        """
        assert ps_model.version == "ps_native_v1"
        assert len(ps_model.features) == 30
        assert len(ps_model._sha256) == 64

    def test_the_frozen_model_predates_the_live_schema(self, ps_model) -> None:
        """The pending-retrain gap, pinned so it cannot be forgotten.

        This is the deliberate consequence of fixing the generator without
        retraining: the model cannot consume pipeline output until a v2 model
        is trained under ml/protocol.py. ps_model.py refuses the mismatch
        rather than scoring misaligned columns, and the orchestrator reports
        MODEL_UNAVAILABLE_FOR_SCHEMA. Delete this test when v2 ships.
        """
        assert set(ps_model.features) != set(CORE_PS_FEATURE_COLUMNS)
        removed = set(ps_model.features) - set(CORE_PS_FEATURE_COLUMNS)
        assert "output_entropy" in removed
        assert "is_peeling_candidate" not in removed, (
            "the peeling flag was redefined, not removed"
        )

    def test_model_inference_valid_range_and_severity(self, ps_model) -> None:
        # Create synthetic feature row
        # The model's own feature list, not the live schema: this test
        # is about inference behaviour, not about schema drift.
        row_data = {feat: 1.0 for feat in ps_model.features}
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
        row_data = {feat: float(i % 5)
                    for i, feat in enumerate(ps_model.features)}
        row_data["address"] = "1DetAddr"
        row_data["txid"] = "tx_det_01"
        df = pd.DataFrame([row_data])

        p1 = ps_model.predict_address_features(df)[0]
        p2 = ps_model.predict_address_features(df)[0]

        assert p1.raw_risk_score == pytest.approx(p2.raw_risk_score)
        assert p1.calibrated_risk_score == pytest.approx(p2.calibrated_risk_score)
        assert p1.severity == p2.severity
