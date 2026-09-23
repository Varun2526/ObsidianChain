"""Automated leakage audit for the PS-native feature engine.

Each test makes the build fail if a feature can see information from after
the event it describes. The audit in words, with results:
research/autoresearch_2026_09_23/19_leakage_audit.md.

1. Prefix invariance - appending later events never changes an earlier row.
   Run on a real Elliptic++ prefix (skipped where the raw data is absent)
   and on the synthetic world v2 capture (real within-step timestamps).
2. Tie-order invariance - events sharing (timestamp, event_order) are
   simultaneous; shuffling their order must not change any feature.
3. Causal event order - every Elliptic++ spend edge goes forward in
   event_order (the builder's spend-DAG levels).
4. Label independence - label columns in the input change nothing, and the
   engine's source never names a label or truth file.
5. Catalog completeness - every emitted feature has an audited catalog
   entry, and development data satisfies the feature contract.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain.contracts.features import CATALOG, validate_feature_frame
from obsidianchain.pipeline import features_ps
from obsidianchain.pipeline.features_ps import (
    CORE_PS_FEATURE_COLUMNS, GROUP_E_NETWORK, extract_ps_features,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HAS_RAW = all((RAW / f).is_file() for f in (
    "AddrTx_edgelist.csv", "TxAddr_edgelist.csv", "txs_features.csv", "txs_edgelist.csv"))
FEATURES = list(CORE_PS_FEATURE_COLUMNS)


def _assert_rows_equal(a: pd.DataFrame, b: pd.DataFrame, cols) -> None:
    a = a.set_index(["address", "txid"]).sort_index()[cols]
    b = b.set_index(["address", "txid"]).sort_index().loc[a.index, cols]
    va, vb = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    same = (va == vb) | (np.isnan(va) & np.isnan(vb))
    if not same.all():
        r, c = np.argwhere(~same)[0]
        raise AssertionError(f"leak: {cols[c]} for {a.index[r]} changed {va[r, c]} -> {vb[r, c]}")


@pytest.fixture(scope="module")
def elliptic_prefix():
    if not HAS_RAW:
        pytest.skip("Elliptic++ raw data not present")
    sys.path.insert(0, str(ROOT / "research" / "reproduction"))
    from build_ps_dataset import build_canonical_frame
    return build_canonical_frame(RAW, max_step=4)


# ---- 1. prefix invariance -------------------------------------------------

def test_elliptic_later_timesteps_never_change_earlier_rows(elliptic_prefix) -> None:
    frame = elliptic_prefix
    early = frame[frame["_step"] <= 3]
    _assert_rows_equal(extract_ps_features(early), extract_ps_features(frame), FEATURES)


def test_elliptic_later_levels_in_a_step_never_change_earlier_levels(elliptic_prefix) -> None:
    frame = elliptic_prefix[elliptic_prefix["_step"] == 4]
    cut = int(frame.event_order.quantile(0.5)) + 1
    assert frame.event_order.max() >= cut
    _assert_rows_equal(extract_ps_features(frame[frame.event_order < cut]),
                       extract_ps_features(frame), FEATURES)


def _world_capture(tmp_path):
    from obsidianchain.io import ingest
    from obsidianchain.world import noisy
    capture, _labels, _entities = noisy.build_noisy_world(noisy.NoisyWorldConfig(entities_per_weight=4, seed=11))
    path = tmp_path / "capture.csv"
    capture.to_csv(path, index=False)
    return ingest.ingest(path)[0]


def test_synthetic_capture_prefix_invariance(tmp_path) -> None:
    frame = _world_capture(tmp_path)
    ts = pd.to_numeric(frame.timestamp)
    cut = ts.quantile(0.6)
    # Whole transactions only: a txid's observations may straddle the cut.
    early_tx = set(frame.loc[ts <= cut, "txid"]) - set(frame.loc[ts > cut, "txid"])
    cols = FEATURES + list(GROUP_E_NETWORK)
    _assert_rows_equal(extract_ps_features(frame[frame.txid.isin(early_tx)], include_network=True),
                       extract_ps_features(frame, include_network=True), cols)


# ---- 2. tie-order invariance ----------------------------------------------

def test_simultaneous_events_do_not_see_each_other(elliptic_prefix) -> None:
    frame = elliptic_prefix[elliptic_prefix["_step"] <= 2]
    shuffled = frame.sample(frac=1.0, random_state=7)
    _assert_rows_equal(extract_ps_features(frame), extract_ps_features(shuffled), FEATURES)


def test_tie_order_invariance_on_a_constructed_case() -> None:
    """Two transactions at one timestamp reusing an address: neither may
    count the other in its history, whichever is listed first."""
    a = {"txid": "a", "timestamp": 100.0, "input_addresses": ["x"], "input_amounts": [1.0],
         "output_addresses": ["shared"], "output_amounts": [0.9], "fee": 0.1}
    b = {"txid": "b", "timestamp": 100.0, "input_addresses": ["shared"], "input_amounts": [0.5],
         "output_addresses": ["y"], "output_amounts": [0.4], "fee": 0.1}
    ab = extract_ps_features(pd.DataFrame([a, b]))
    ba = extract_ps_features(pd.DataFrame([b, a]))
    _assert_rows_equal(ab, ba, FEATURES)
    row = ab[(ab.address == "shared") & (ab.txid == "b")].iloc[0]
    assert row["n_txs_asof_t"] == 0
    assert row["upstream_funded_share"] == 0.0


# ---- 3. causal event order --------------------------------------------------

def test_every_spend_edge_goes_forward_in_event_order(elliptic_prefix) -> None:
    edges = pd.read_csv(RAW / "txs_edgelist.csv")
    edges.columns = ["src", "dst"]
    level = dict(zip(elliptic_prefix.txid.astype(float), elliptic_prefix.event_order))
    step = dict(zip(elliptic_prefix.txid.astype(float), elliptic_prefix["_step"]))
    e = edges[edges.src.isin(level) & edges.dst.isin(level)]
    assert len(e) > 1000
    assert (e.src.map(step) == e.dst.map(step)).all(), "a spend edge crosses a timestep"
    assert (e.dst.map(level) > e.src.map(level)).all(), "a transaction precedes what it spends"


# ---- 4. label independence ------------------------------------------------

def test_label_columns_in_the_input_change_nothing() -> None:
    txs = [{"txid": f"t{i}", "timestamp": 100.0 * i, "input_addresses": [f"a{i}"], "input_amounts": [1.0],
            "output_addresses": [f"a{i + 1}"], "output_amounts": [0.9], "fee": 0.1} for i in range(6)]
    plain = extract_ps_features(pd.DataFrame(txs))
    labelled = extract_ps_features(pd.DataFrame([dict(t, y=i % 2, **{"class": 1}) for i, t in enumerate(txs)]))
    _assert_rows_equal(plain, labelled, FEATURES)
    assert not {"y", "class"} & set(labelled.columns)


def test_the_engine_source_never_names_a_label_or_truth_source() -> None:
    code = inspect.getsource(features_ps)
    for forbidden in ("wallets_classes", "world_truth", "network_truth", "FOR_EVALUATION_ONLY",
                      '["y"]', '["class"]', "labels.csv"):
        assert forbidden not in code, forbidden


# ---- 5. catalog completeness ------------------------------------------------

def test_every_emitted_feature_is_in_the_audited_catalog() -> None:
    txs = pd.DataFrame([{"txid": "t", "timestamp": 1.0, "input_addresses": ["a"], "input_amounts": [1.0],
                         "output_addresses": ["b"], "output_amounts": [0.9], "fee": 0.1, "src_ip": "10.0.0.1"}])
    emitted = set(extract_ps_features(txs, include_network=True).columns) - {"address", "txid", "timestamp"}
    assert emitted == set(CATALOG), (emitted ^ set(CATALOG))
    assert all(spec.labels == "none" for spec in CATALOG.values())


def test_development_data_satisfies_the_feature_contract() -> None:
    path = ROOT / "data" / "models" / "ps_native" / "datasets" / "train.parquet"
    if not path.is_file():
        pytest.skip("development data not built")
    report = validate_feature_frame(pd.read_parquet(path), FEATURES)
    assert report.ok, report.violations


def test_the_snapshot_row_is_one_whole_row() -> None:
    """groupby().last() would splice the latest non-null value of each column
    from different transactions; the snapshot must be one real row."""
    from obsidianchain.pipeline.features_ps import last_snapshot_per_address
    txs = [
        {"txid": "t1", "timestamp": 1.0, "input_addresses": ["src"], "input_amounts": [1.0],
         "output_addresses": ["a"], "output_amounts": [0.9], "fee": 0.1},
        {"txid": "t2", "timestamp": 2.0, "input_addresses": ["a"], "input_amounts": [0.9],
         "output_addresses": ["b"], "output_amounts": [0.8], "fee": None},
    ]
    feats = extract_ps_features(pd.DataFrame(txs))
    snap = last_snapshot_per_address(feats).set_index("address")
    assert snap.loc["a", "txid"] == "t2"
    assert np.isnan(snap.loc["a", "fee"]), "fee spliced in from an older transaction"
