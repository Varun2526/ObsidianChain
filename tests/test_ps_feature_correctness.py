"""Feature correctness, zero-division, and edge-case handling tests for PS features."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from obsidianchain.pipeline.features_ps import (
    CORE_PS_FEATURE_COLUMNS,
    GROUP_E_NETWORK,
    PsTemporalFeatureEngine,
    extract_ps_features,
)


def test_zero_division_and_empty_values_handling() -> None:
    """Zero fees, zero amounts, and empty fields must not produce ZeroDivisionError or Inf."""
    tx_zero = {
        "txid": "tx_zero",
        "timestamp": 1000.0,
        "input_addresses": ["addr_zero"],
        "input_amounts": [0.0],
        "output_addresses": ["out_zero"],
        "output_amounts": [0.0],
        "fee": 0.0,
        "script_type": "p2pkh",
    }
    df = pd.DataFrame([tx_zero])
    feats = extract_ps_features(df)
    assert len(feats) > 0
    row = feats.iloc[0]

    # Check that fee_ratio handled zero input amount cleanly
    assert row["fee_ratio"] == 0.0
    assert not math.isinf(row["fee_ratio"])
    assert not math.isnan(row["fee_ratio"])
    assert not math.isinf(row["tx_velocity_per_hour"])
    assert not math.isinf(row["output_entropy"])


def test_malformed_amounts_sanitization() -> None:
    """Invalid string amounts, negative amounts, or None must be safely sanitized."""
    tx_malformed = {
        "txid": "tx_malformed",
        "timestamp": 1050.0,
        "input_addresses": ["addr_bad"],
        "input_amounts": ["not_a_float", -5.0, None],
        "output_addresses": ["out_bad"],
        "output_amounts": [1.0],
        "fee": "invalid_fee",
        "script_type": "p2pkh",
    }
    df = pd.DataFrame([tx_malformed])
    feats = extract_ps_features(df)
    assert len(feats) > 0
    row = feats.iloc[0]
    # Total input amount should ignore non-numeric and negative values
    assert row["total_input_amount"] >= 0.0
    assert not math.isnan(row["total_input_amount"])
    assert row["fee"] == 0.0


def test_missing_network_evidence_is_nan_not_fabricated_zero() -> None:
    """Missing network evidence must be represented as NaN/missing, never fabricated 0."""
    tx_no_net = {
        "txid": "tx_pure_bc",
        "timestamp": 2000.0,
        "input_addresses": ["addr_net_test"],
        "input_amounts": [10.0],
        "output_addresses": ["out_net_test"],
        "output_amounts": [9.99],
        "fee": 0.01,
        "script_type": "p2pkh",
        # Network fields absent / None
        "src_ip": None,
        "dst_ip": None,
        "asn": None,
        "geo_country": None,
    }
    df = pd.DataFrame([tx_no_net])
    feats = extract_ps_features(df, include_network=True)
    row = feats.iloc[0]

    for col in GROUP_E_NETWORK:
        assert math.isnan(row[col]), f"Network feature '{col}' must be NaN when absent, not fabricated!"


def test_deterministic_feature_ordering() -> None:
    """Feature columns and records must follow deterministic ordering."""
    tx1 = {
        "txid": "tx1", "timestamp": 100.0,
        "input_addresses": ["addrA"], "input_amounts": [1.0],
        "output_addresses": ["out1"], "output_amounts": [0.99], "fee": 0.01,
    }
    tx2 = {
        "txid": "tx2", "timestamp": 200.0,
        "input_addresses": ["addrB"], "input_amounts": [2.0],
        "output_addresses": ["out2"], "output_amounts": [1.99], "fee": 0.01,
    }
    df = pd.DataFrame([tx1, tx2])
    feats1 = extract_ps_features(df)
    feats2 = extract_ps_features(df)

    assert list(feats1.columns) == list(feats2.columns)
    pd.testing.assert_frame_equal(feats1, feats2)
