"""The investigation read path over the real Phase 7 alerts and chain index.

Needs ``data/processed/alerts.parquet`` and the chain index
(``make run ARGS="build-chain-index"``); CI deselects it.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from obsidianchain.api import alerts, artifacts, boundary, investigation


@pytest.fixture(scope="module")
def alert_ids():
    frame, _ = artifacts.load_alerts(None)
    return list(frame.sort_values("rank")["alert_id"].head(5))


def test_alert_graph_has_every_layer_and_no_dangling_edge(alert_ids) -> None:
    g = alerts.get_alert_graph(alert_ids[0], hops=2)
    ids = {n["id"] for n in g["graph"]["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in g["graph"]["edges"])
    kinds = {n["kind"] for n in g["graph"]["nodes"]}
    assert {"cluster", "address", "transaction"} <= kinds
    assert any(e["kind"] == "MEMBER_OF" for e in g["graph"]["edges"])
    assert g["hops"] == 2
    boundary.assert_no_truth_fields(g)


def test_hops_change_the_graph(alert_ids) -> None:
    one = alerts.get_alert_graph(alert_ids[0], hops=1, max_nodes=1500)
    two = alerts.get_alert_graph(alert_ids[0], hops=2, max_nodes=1500)
    assert two["graph"]["node_count"] >= one["graph"]["node_count"]


def test_related_alerts_carry_real_signals(alert_ids) -> None:
    for aid in alert_ids:
        for rel in alerts.get_related_alerts(aid)["related_alerts"]:
            assert isinstance(rel["top_signals"], list)
            assert all(isinstance(s, str) for s in rel["top_signals"])


def test_transaction_drilldown_reads_the_mixing_scan() -> None:
    mixing = pd.read_parquet(artifacts.data_root(None) / "processed" / "tx_mixing.parquet",
                             columns=["txId"])
    txid = int(mixing["txId"].iloc[0])
    d = alerts.get_transaction_drilldown(txid)
    assert d["mixing"]["available"] is True and "mixing_score" in d["mixing"]
    assert d["timestep"] is not None


def test_read_path_latency_budget(alert_ids) -> None:
    """Warm-cache budgets: generous enough for CI noise, tight enough to catch a regression."""
    investigation.chain(None)
    alerts.get_alert_graph(alert_ids[0], hops=2)
    t = time.perf_counter()
    alerts.get_alert_graph(alert_ids[1], hops=2)
    assert time.perf_counter() - t < 1.0
    addr = alerts.get_alert_graph(alert_ids[1], hops=1)["graph"]["nodes"]
    a = next(n["data"]["address"] for n in addr if n["kind"] == "address")
    t = time.perf_counter()
    investigation.get_address(a)
    assert time.perf_counter() - t < 0.5
