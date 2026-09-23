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

    # schema v3: a fee ratio over zero input value is not measurable, so it
    # is NaN - the same rule as output_spread below. 0.0 would claim the
    # ratio was measured and found to be zero.
    assert math.isnan(row["fee_ratio"])
    assert not math.isinf(row["gap_since_last_tx"])
    # schema v2: output_entropy was exactly log2(output_count) and was
    # removed. output_spread is the observable it replaced - and on a
    # zero-value transaction it must be NaN ("not measurable"), never
    # 0.0, which would read as "measured and perfectly uniform".
    assert not math.isinf(row["output_spread"])


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
    # schema v3: an unparseable fee is missing, not zero.
    assert math.isnan(row["fee"])
    assert math.isnan(row["fee_ratio"])


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


def test_repeated_observations_of_one_txid_are_one_chain_event() -> None:
    """A capture row is a network OBSERVATION; three peers announcing one
    transaction must not count as three transactions in anyone's history."""
    base = {
        "txid": "tx_seen_thrice", "timestamp": 1000.0,
        "input_addresses": ["payer"], "input_amounts": [1.0],
        "output_addresses": ["payee"], "output_amounts": [0.99],
        "fee": 0.01, "script_type": "p2pkh", "dst_ip": "10.0.0.1", "asn": 64500,
    }
    later = dict(base, txid="tx_next", timestamp=2000.0,
                 input_addresses=["payee"], input_amounts=[0.99],
                 output_addresses=["other"], output_amounts=[0.98])
    rows = [dict(base, src_ip=ip, timestamp=1000.0 + i)
            for i, ip in enumerate(("1.1.1.1", "2.2.2.2", "3.3.3.3"))]
    feats = extract_ps_features(pd.DataFrame(rows + [dict(later, src_ip="4.4.4.4")]),
                                include_network=True)
    assert not feats.duplicated(subset=["address", "txid"]).any()
    payee_next = feats[(feats.address == "payee") & (feats.txid == "tx_next")].iloc[0]
    assert payee_next["n_txs_asof_t"] == 1
    first = feats[feats.txid == "tx_seen_thrice"].iloc[0]
    assert first["network_observation_count"] == 3.0
    assert first["peer_count"] == 3.0
    assert first["asn_count"] == 1.0
    assert first["timestamp"] == 1000.0


def test_role_features_distinguish_addresses_in_one_transaction() -> None:
    """Group F exists so a payer and a payee of the same transaction do not
    share one feature vector."""
    history = {
        "txid": "tx_old", "timestamp": 500.0,
        "input_addresses": ["veteran"], "input_amounts": [2.0],
        "output_addresses": ["payer"], "output_amounts": [1.9],
        "fee": 0.1, "script_type": "p2pkh",
    }
    tx = {
        "txid": "tx_now", "timestamp": 1000.0,
        "input_addresses": ["payer"], "input_amounts": [1.9],
        "output_addresses": ["payee", "payer"], "output_amounts": [1.0, 0.8],
        "fee": 0.1, "script_type": "p2pkh",
    }
    feats = extract_ps_features(pd.DataFrame([history, tx]))
    now = feats[feats.txid == "tx_now"].set_index("address")
    assert now.loc["payer", "addr_is_sender"] == 1
    assert now.loc["payer", "addr_is_self_change"] == 1
    assert now.loc["payee", "addr_is_sender"] == 0
    # payee's counterparty is payer, who had one prior transaction.
    assert now.loc["payee", "counterparty_max_n_txs_asof_t"] == 1.0
    # payer's counterparty is payee, who had none; the current
    # transaction is never counted.
    assert now.loc["payer", "counterparty_max_n_txs_asof_t"] == 0.0


def test_upstream_group_reads_only_earlier_funding() -> None:
    """Group G describes the transactions that funded this one's inputs,
    as known BEFORE it. A later transaction must not change an earlier row."""
    def tx(txid, ts, ins, outs):
        return {"txid": txid, "timestamp": ts, "input_addresses": ins,
                "input_amounts": [1.0] * len(ins), "output_addresses": outs,
                "output_amounts": [0.9 / len(outs)] * len(outs), "fee": 0.01,
                "script_type": "p2pkh"}
    chain = [tx("t1", 100.0, ["src"], ["a"]), tx("t2", 200.0, ["a"], ["b"]),
             tx("t3", 300.0, ["b"], ["c"])]
    # Group G is a property of the transaction: one row per txid suffices.
    feats = extract_ps_features(pd.DataFrame(chain)).groupby("txid").first()
    first = feats.loc["t1"]
    assert first["upstream_funded_share"] == 0.0
    assert math.isnan(first["upstream_mean_output_count"])
    # t2 is funded by t1 (depth 1), t3 by t2 (depth 2): the chain lengthens.
    assert feats.loc["t2", "upstream_chain_depth"] == 1.0
    assert feats.loc["t3", "upstream_chain_depth"] == 2.0
    assert feats.loc["t3", "upstream_min_hold_seconds"] == 100.0
    # Adding a future transaction leaves every earlier row unchanged.
    more = extract_ps_features(pd.DataFrame(chain + [tx("t4", 400.0, ["c"], ["d"])])).groupby("txid").first()
    cols = [c for c in feats.columns if c.startswith("upstream_")]
    pd.testing.assert_frame_equal(feats.loc[["t1", "t2", "t3"], cols],
                                  more.loc[["t1", "t2", "t3"], cols])
