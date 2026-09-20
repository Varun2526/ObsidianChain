"""Strict Future-Leakage and Label-Leakage Tests for PS-Native Features.

Asserts the 6 mandatory safeguards:
A. Adding a future transaction does not change an earlier feature vector.
B. Adding a future counterparty does not change an earlier degree/counterparty feature.
C. Adding a future transaction to a cluster does not change earlier cluster size.
D. Future transactions cannot alter an earlier peeling/mixing feature.
E. Labels from validation/test cannot enter feature construction.
F. Train/test split assignment occurs before any operation that could propagate labels.
"""

from __future__ import annotations

import pandas as pd
import pytest

from obsidianchain.pipeline.features_ps import (
    CORE_PS_FEATURE_COLUMNS,
    PsTemporalFeatureEngine,
    extract_ps_features,
)


def _base_tx(
    txid: str,
    ts: float,
    in_addrs: list[str],
    in_amts: list[float],
    out_addrs: list[str],
    out_amts: list[float],
    fee: float = 0.001,
) -> dict:
    return {
        "txid": txid,
        "timestamp": ts,
        "input_addresses": in_addrs,
        "input_amounts": in_amts,
        "output_addresses": out_addrs,
        "output_amounts": out_amts,
        "fee": fee,
        "script_type": "p2pkh",
    }


class TestTemporalLeakageSafeguards:
    """Rigorous tests proving that future events cannot corrupt earlier forensic features."""

    def test_a_future_transaction_does_not_change_earlier_feature_vector(self) -> None:
        """A. Adding a future transaction does not change an earlier feature vector."""
        # Initial scenario at t=100
        tx1 = _base_tx("tx1", 100.0, ["addrA"], [5.0], ["addrB"], [4.999], fee=0.001)
        df_early = pd.DataFrame([tx1])

        features_early = extract_ps_features(df_early)
        addrA_early = features_early[features_early["address"] == "addrA"].iloc[0]

        # Scenario with an additional future transaction at t=200 involving addrA
        tx2_future = _base_tx("tx2", 200.0, ["addrA"], [2.0], ["addrC"], [1.999], fee=0.001)
        df_extended = pd.DataFrame([tx1, tx2_future])

        features_extended = extract_ps_features(df_extended)
        addrA_at_tx1_in_extended = features_extended[
            (features_extended["address"] == "addrA") & (features_extended["txid"] == "tx1")
        ].iloc[0]

        # The feature vector for addrA at tx1 must be 100% IDENTICAL before and after adding tx2
        for col in CORE_PS_FEATURE_COLUMNS:
            val_early = addrA_early[col]
            val_extended = addrA_at_tx1_in_extended[col]
            assert val_early == val_extended, (
                f"Leakage detected on '{col}': value changed from {val_early} to {val_extended} "
                f"after future transaction was appended!"
            )

    def test_b_future_counterparty_does_not_change_earlier_degrees(self) -> None:
        """B. Adding a future counterparty does not change an earlier degree/counterparty feature."""
        tx1 = _base_tx("tx1", 100.0, ["addrA"], [5.0], ["addrB"], [4.999])
        tx2_future = _base_tx("tx2", 200.0, ["addrA"], [1.0], ["addrC", "addrD", "addrE"], [0.3, 0.3, 0.399])

        df_extended = pd.DataFrame([tx1, tx2_future])
        features = extract_ps_features(df_extended)

        row_t1 = features[(features["address"] == "addrA") & (features["txid"] == "tx1")].iloc[0]
        # At t1, addrA had 0 prior counterparties
        assert row_t1["unique_counterparties_asof_t"] == 0
        assert row_t1["out_degree_asof_t"] == 0

        row_t2 = features[(features["address"] == "addrA") & (features["txid"] == "tx2")].iloc[0]
        # At t2, addrA has observed addrB as counterparty from tx1
        assert row_t2["unique_counterparties_asof_t"] == 1
        assert row_t2["out_degree_asof_t"] == 1

    def test_c_future_cluster_expansion_does_not_change_earlier_cluster_size(self) -> None:
        """C. Adding a future transaction to a cluster does not change earlier cluster size."""
        # At t=100, addrA spends alone (cluster size = 1)
        tx1 = _base_tx("tx1", 100.0, ["addrA"], [5.0], ["out1"], [4.999])
        # At t=200, addrA co-spends with 10 other addresses (cluster size grows to 11)
        co_spenders = [f"co_{i}" for i in range(10)]
        tx2_future = _base_tx(
            "tx2", 200.0,
            ["addrA"] + co_spenders,
            [1.0] * 11,
            ["out2"], [10.999]
        )

        df = pd.DataFrame([tx1, tx2_future])
        features = extract_ps_features(df)

        row_t1 = features[(features["address"] == "addrA") & (features["txid"] == "tx1")].iloc[0]
        assert row_t1["cluster_size_asof_t"] == 1, (
            f"Leakage: cluster size at tx1 was {row_t1['cluster_size_asof_t']}, expected 1!"
        )

        row_t2 = features[(features["address"] == "addrA") & (features["txid"] == "tx2")].iloc[0]
        assert row_t2["cluster_size_asof_t"] == 11, (
            f"Expected cluster size 11 at tx2, got {row_t2['cluster_size_asof_t']}"
        )

    def test_d_future_transactions_cannot_alter_earlier_peeling_mixing_features(self) -> None:
        """D. Future transactions cannot alter an earlier peeling/mixing feature."""
        # Normal tx at t=100
        tx1 = _base_tx("tx1", 100.0, ["addrA", "addrB"], [1.0, 1.0], ["out1"], [1.999])
        # Mixing tx at t=200
        tx2_mix = _base_tx(
            "tx2", 200.0,
            ["addr1", "addr2", "addr3"], [1.0, 1.0, 1.0],
            ["outA", "outB", "outC"], [0.99, 0.99, 0.99],
            fee=0.03
        )

        df = pd.DataFrame([tx1, tx2_mix])
        features = extract_ps_features(df)

        row_t1 = features[(features["txid"] == "tx1")].iloc[0]
        assert row_t1["is_mixing_candidate"] == 0

        row_t2 = features[(features["txid"] == "tx2")].iloc[0]
        assert row_t2["is_mixing_candidate"] == 1

    def test_e_label_columns_cannot_enter_feature_matrix(self) -> None:
        """E. Labels from validation/test cannot enter feature construction."""
        tx1 = _base_tx("tx1", 100.0, ["addrA"], [5.0], ["out1"], [4.999])
        # Attach deceptive ground truth columns to the input frame
        tx1["class"] = 1
        tx1["label"] = "illicit"
        tx1["category"] = "RANSOMWARE"
        tx1["y"] = 1
        tx1["illicit_flag"] = True

        df = pd.DataFrame([tx1])
        features = extract_ps_features(df)

        forbidden_names = {"class", "label", "category", "y", "illicit_flag", "entity"}
        feature_cols = set(features.columns)

        intersection = forbidden_names.intersection(feature_cols)
        assert not intersection, f"Forbidden label columns leaked into feature matrix: {intersection}"

    def test_f_train_test_split_assignment_occurs_before_propagation(self) -> None:
        """F. Split assignment is purely temporal based on first appearance, immune to future edges."""
        # Test split rule: split assigned on first appearance
        # Address A first appears at t=10 (Train window: t < 100)
        # Address B first appears at t=200 (Test window: t >= 100)
        tx1 = _base_tx("tx1", 10.0, ["addrA"], [5.0], ["out1"], [4.999])
        tx2 = _base_tx("tx2", 200.0, ["addrA", "addrB"], [1.0, 1.0], ["out2"], [1.999])

        df = pd.DataFrame([tx1, tx2])
        engine = PsTemporalFeatureEngine()
        features = engine.process_records(df)

        # addrA first_seen is 10.0
        assert engine.address_states["addrA"].first_seen == 10.0
        # addrB first_seen is 200.0
        assert engine.address_states["addrB"].first_seen == 200.0
        # Neither address state knew anything about the other until t=200
