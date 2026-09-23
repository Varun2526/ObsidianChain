"""The investigation read path: chain index, address profile, trace, search, models.

Everything runs on a hand-made Elliptic++-shaped raw folder, so it needs no
local data. The graph:

    t1  tx 1:  A, B      -> C
    t2  tx 2:  C         -> D, E
    t3  tx 3:  E         -> F
    t2  tx 9:  H1..H5    -> C     (a five-input transaction, the "hub" when
                                   max_tx_degree is small)
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from obsidianchain import chain_index
from obsidianchain.api import investigation, models
from obsidianchain.api.app import create_app
from obsidianchain.console import db, users

HUB_INPUTS = [f"1Hub{i}xxxxxxxx" for i in range(5)]
A, B, C, D, E, F = (f"1{x}ddrxxxxxxxxxx" for x in "ABCDEF")


def _write_raw(root):
    raw = root / "raw"
    raw.mkdir(parents=True)
    ins = [(A, 1), (B, 1), (C, 2), (E, 3)] + [(h, 9) for h in HUB_INPUTS]
    outs = [(1, C), (2, D), (2, E), (3, F), (9, C)]
    pd.DataFrame(ins, columns=["input_address", "txId"]).to_csv(raw / "AddrTx_edgelist.csv", index=False)
    pd.DataFrame(outs, columns=["txId", "output_address"]).to_csv(raw / "TxAddr_edgelist.csv", index=False)
    pd.DataFrame({"txId": [1, 2, 3, 9], "Time step": [1, 2, 3, 2], "fees": [0.001] * 4,
                  "in_BTC_total": [2.0, 1.5, 0.7, 5.0], "out_BTC_total": [1.999, 1.499, 0.699, 4.999],
                  "class": [1, 2, 1, 2]}).to_csv(raw / "txs_features.csv", index=False)
    watch = root / "watchlists"
    watch.mkdir()
    (watch / "analyst.csv").write_text(f"address,label\n{F},Test exchange\n")


@pytest.fixture()
def data_root(tmp_path):
    _write_raw(tmp_path)
    chain_index.build(tmp_path)
    return tmp_path


def _ids(graph):
    return {n["id"] for n in graph["nodes"]}


def _no_dangling(graph):
    ids = _ids(graph)
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])


def test_the_index_carries_no_class_labels(data_root) -> None:
    for name in (chain_index.CHAIN_EDGES, chain_index.CHAIN_TRANSACTIONS, chain_index.WATCHLIST_SEEDS):
        frame = pd.read_parquet(data_root / "processed" / name)
        assert not {"class", "label_class", "illicit"} & set(frame.columns)
        meta = json.loads((data_root / "processed" / f"{name}.meta.json").read_text())
        assert meta["artifact"]["contains_labels"] is False


def test_address_profile_is_observed_structure(data_root) -> None:
    p = investigation.get_address(C, data_root)
    assert p["observed"]["as_input"] == 1 and p["observed"]["as_output"] == 2
    assert {x["address"] for x in p["counterparties"]["paid_to"]} == {D, E}
    assert {x["address"] for x in p["counterparties"]["paid_by"]} == {A, B, *HUB_INPUTS}
    assert p["observed"]["first_timestep"] == 1 and p["observed"]["last_timestep"] == 2
    assert p["provenance"]["labels_served"] is False
    assert investigation.get_address(F, data_root)["watchlist"] == [
        {"source": "WATCHLIST:analyst.csv", "label": "Test exchange"}]
    with pytest.raises(investigation.AddressNotFoundError):
        investigation.get_address("1Nowhere", data_root)


def test_downstream_follows_value_forward_only(data_root) -> None:
    r = investigation.trace([A], data_root, direction="downstream", hops=3)
    ids = _ids(r["graph"])
    assert {f"addr:{x}" for x in (A, C, D, E, F)} <= ids
    assert f"addr:{B}" not in ids  # a co-input of tx 1 funded it; it is not downstream of A
    assert f"addr:{HUB_INPUTS[0]}" not in ids  # upstream of C, not downstream of A
    _no_dangling(r["graph"])
    kinds = {(e["source"], e["target"]): e["kind"] for e in r["graph"]["edges"]}
    assert kinds[(f"addr:{A}", "tx:1")] == "SPENDS" and kinds[("tx:1", f"addr:{C}")] == "PAYS"


def test_upstream_reaches_the_sources(data_root) -> None:
    r = investigation.trace([F], data_root, direction="upstream", hops=3)
    ids = _ids(r["graph"])
    assert {f"addr:{x}" for x in (A, B, E, C)} <= ids
    assert f"addr:{D}" not in ids
    _no_dangling(r["graph"])


def test_time_window_and_hub_skipping(data_root) -> None:
    r = investigation.trace([A], data_root, direction="downstream", hops=3, max_timestep=1)
    assert "tx:2" not in _ids(r["graph"])
    r = investigation.trace([C], data_root, direction="upstream", hops=1, max_tx_degree=4)
    assert "tx:9" not in _ids(r["graph"])
    assert r["hub_transactions_skipped"] == [{"txid": 9, "participants": 6}]


def test_truncation_is_reported(data_root) -> None:
    r = investigation.trace([C], data_root, direction="both", hops=3, max_nodes=10)
    assert r["truncated"] is True
    assert r["graph"]["node_count"] <= 11
    _no_dangling(r["graph"])


def test_bad_direction_is_refused(data_root) -> None:
    with pytest.raises(investigation.TraceRequestError):
        investigation.trace([A], data_root, direction="sideways")


def test_search(data_root) -> None:
    kinds = {(x["kind"], x["id"]) for x in investigation.search("1Cdd", data_root)["results"]}
    assert ("address", C) in kinds
    assert ("transaction", "2") in {(x["kind"], x["id"]) for x in investigation.search("2", data_root)["results"]}
    assert investigation.search("", data_root)["results"] == []


def _registry(root):
    base = root / "models" / "ps_native"
    (base / "v9").mkdir(parents=True)
    (base / "v9" / "manifest.json").write_text(json.dumps({"model_type": "LightGBM"}))
    (base / "registry.json").write_text(json.dumps({
        "schema": "x", "roles": {"champion": "v9", "candidate": None, "fallback": None},
        "models": {"v9": {"path": "v9", "feature_schema_version": "s/1", "notes": "n"}}, "history": []}))


def test_models_are_read_not_computed(tmp_path) -> None:
    _registry(tmp_path)
    listing = models.list_models(tmp_path)
    assert listing["models"][0]["role"] == "champion"
    detail = models.get_model("v9", tmp_path)
    assert detail["production"]["available"] is False
    assert "HOLDOUT" in detail["result_types"]
    with pytest.raises(models.ModelNotFoundError):
        models.get_model("v0", tmp_path)


@pytest.fixture()
def client(data_root):
    _registry(data_root)
    conn = db.connect(data_root)
    try:
        users.create(conn, username="analyst", password="analyst-password", role="INVESTIGATOR")
    finally:
        conn.close()
    c = TestClient(create_app(data_root))
    assert c.post("/api/auth/login", json={"username": "analyst", "password": "analyst-password"}).status_code == 200
    return c


def test_endpoints(client) -> None:
    assert client.get(f"/api/addresses/{C}").json()["address"] == C
    miss = client.get("/api/addresses/1Nowhere")
    assert miss.status_code == 404 and miss.json()["error"] == "address_not_found"

    r = client.get("/api/graph/trace", params={"address": A, "direction": "downstream", "hops": 2})
    assert r.status_code == 200 and r.json()["graph"]["node_count"] > 0
    assert client.get("/api/graph/trace").status_code == 400
    assert client.get("/api/graph/trace", params={"address": A, "direction": "x"}).status_code == 400
    assert client.get("/api/graph/trace", params={"txid": 2}).json()["seeds"]["txids"] == [2]

    tx = client.get("/api/transactions/2").json()
    assert tx["timestep"] == 2 and {x["address"] for x in tx["outputs"]} == {D, E}
    assert tx["announcing_peers"] == [] and tx["mixing"]["available"] is False
    assert client.get("/api/transactions/777").status_code == 404

    assert client.get("/api/search", params={"q": "1Fdd"}).json()["results"][0]["id"] == F
    assert client.get("/api/models").json()["roles"]["champion"] == "v9"
    assert client.get("/api/models/v0").status_code == 404


def test_endpoints_need_a_session(data_root) -> None:
    anon = TestClient(create_app(data_root))
    for path in (f"/api/addresses/{C}", "/api/graph/trace?address=x", "/api/search?q=1C", "/api/models"):
        assert anon.get(path).status_code == 401
