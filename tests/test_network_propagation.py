"""The network layer on real uploaded fields: ingestion, propagation, and the
boundary that keeps network evidence out of the risk score.

See docs/plans/2026-09-25-network-layer.md and
docs/audit/2026-09-25-system-truth-audit.md sections 1-3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from obsidianchain.correlation.engine import NetworkObservation
from obsidianchain.io import ingest
from obsidianchain.network import propagation as prop
from obsidianchain.pipeline.features import derive_canonical_address_features

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "canonical_acceptance_capture.json"


def obs(txid="t1", src_ip="8.8.8.8", ts="1600000000", observer=None, dst_ip="198.51.100.1", asn=15169, geo=None):
    return NetworkObservation(txid=txid, observer_id=observer, src_ip=src_ip, dst_ip=dst_ip, src_port=8333,
                              dst_port=8333, timestamp=ts, geo_country=geo, asn=asn, key=f"{txid}{src_ip}{ts}{observer}")


# ---- timestamps --------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("1600000000", 1_600_000_000_000),
    (1600000000.5, 1_600_000_000_500),
    ("1600000000123", 1_600_000_000_123),
    ("2020-09-13T12:26:40Z", 1_600_000_000_000),
    ("2020-09-13T12:26:40", 1_600_000_000_000),
    ("12", None),            # not a plausible Bitcoin-era time in either unit
    ("yesterday", None),
    (None, None),
    (float("nan"), None),
])
def test_timestamps_are_read_or_refused(value, expected) -> None:
    assert prop.timestamp_ms(value) == expected


# ---- ingestion keeps observer identity ---------------------------------------

def _capture(tmp_path, rows) -> Path:
    path = tmp_path / "cap.json"
    path.write_text(json.dumps({"records": rows}))
    return path


BASE = {"txid": "tx1", "input_addresses": ["1a"], "input_amounts": ["1.0"], "output_addresses": ["1b"],
        "output_amounts": ["0.9"], "fee": 0.1, "script_type": "p2pkh", "timestamp": "1600000000",
        "src_ip": "8.8.8.8", "dst_ip": "198.51.100.1", "src_port": 8333, "dst_port": 8333, "asn": 15169}


def test_observer_id_survives_ingestion(tmp_path) -> None:
    frame, _ = ingest.ingest(_capture(tmp_path, [{**BASE, "observer_id": "sensor-a"}]))
    assert frame["observer_id"].tolist() == ["sensor-a"]


def test_the_observer_alias_is_accepted(tmp_path) -> None:
    frame, _ = ingest.ingest(_capture(tmp_path, [{**BASE, "observer": "sensor-b"}]))
    assert frame["observer_id"].tolist() == ["sensor-b"]


def test_one_peer_seen_by_two_observers_is_two_observations(tmp_path) -> None:
    """Before the fix observer_id was dropped, so these collapsed into one."""
    frame, report = ingest.ingest(_capture(tmp_path, [{**BASE, "observer_id": "a"}, {**BASE, "observer_id": "b"}]))
    assert len(frame) == 2
    assert report.network_observations_preserved == 1


# ---- per-transaction propagation -----------------------------------------------

def test_first_seen_last_seen_and_spread() -> None:
    r = prop.analyse_transaction("t1", [
        obs(src_ip="8.8.8.8", ts="1600000003", observer="a"),
        obs(src_ip="1.1.1.1", ts="1600000001", observer="b", asn=13335),
        obs(src_ip="8.8.8.8", ts="1600000009", observer="b"),
    ])
    assert (r.first_seen_ms, r.last_seen_ms, r.spread_ms) == (1_600_000_001_000, 1_600_000_009_000, 8_000)
    assert r.first_seen_peers == ["1.1.1.1"] and r.first_seen_observers == ["b"]
    assert [p.peer_ip for p in r.peers] == ["1.1.1.1", "8.8.8.8"]          # ordered by first arrival
    eight = next(p for p in r.peers if p.peer_ip == "8.8.8.8")
    assert (eight.first_seen_ms, eight.observations, eight.observers) == (1_600_000_003_000, 2, ["a", "b"])
    assert r.dominant_peer_ip == "8.8.8.8" and r.dominant_peer_share == pytest.approx(2 / 3)
    assert r.asns == [13335, 15169] and r.observer_source == "observer_id"


def test_ties_for_first_seen_are_kept_not_broken() -> None:
    r = prop.analyse_transaction("t1", [obs(src_ip="8.8.8.8", ts="1600000001"), obs(src_ip="1.1.1.1", ts="1600000001")])
    assert r.first_seen_peers == ["1.1.1.1", "8.8.8.8"]


def test_a_single_observation_has_no_spread() -> None:
    r = prop.analyse_transaction("t1", [obs()])
    assert r.spread_ms is None and r.first_seen_ms == 1_600_000_000_000


def test_observer_falls_back_to_dst_ip_and_says_so() -> None:
    r = prop.analyse_transaction("t1", [obs(dst_ip="198.51.100.9")])
    assert r.observer_source.startswith("dst_ip") and r.observers == ["198.51.100.9"]


def test_unknown_observers_are_none_not_zero() -> None:
    r = prop.analyse_transaction("t1", [obs(dst_ip=None)])
    assert r.observer_source == "unknown" and r.observers is None
    assert r.as_dict()["observer_count"] is None


def test_non_routable_peers_are_classified() -> None:
    r = prop.analyse_transaction("t1", [obs(src_ip="192.0.2.7"), obs(src_ip="10.0.0.5"), obs(src_ip="8.8.8.8")])
    classes = {p.peer_ip: p.ip_class for p in r.peers}
    assert classes["8.8.8.8"] == "global" and classes["192.0.2.7"] != "global" and classes["10.0.0.5"] != "global"
    assert r.non_routable_peer_share == pytest.approx(2 / 3)


def test_countries_are_capture_supplied_and_labelled() -> None:
    d = prop.analyse_transaction("t1", [obs(geo="in")]).as_dict()
    assert d["countries"] == ["IN"] and "unverified" in d["country_source"]


def test_cluster_pooling() -> None:
    class Corr:
        all_observations = [obs("t1", "8.8.8.8", "1600000000"), obs("t1", "1.1.1.1", "1600000002", asn=13335),
                            obs("t2", "8.8.8.8", "1600000010"), obs("t9", "9.9.9.9", "1600000000", asn=19281)]
    res = prop.analyse(Corr())
    pooled = res.pooled(["t1", "t2", "t-absent"])
    assert pooled["transactions_observed"] == 2 and pooled["peer_count"] == 2
    assert pooled["dominant_peer_ip"] == "8.8.8.8" and pooled["dominant_peer_share"] == pytest.approx(2 / 3)
    assert pooled["earliest_first_seen_ms"] == 1_600_000_000_000 and pooled["median_spread_ms"] == 2_000
    assert res.pooled(["nothing"]) is None


# ---- the pipeline --------------------------------------------------------------

def test_repeated_observations_do_not_multiply_amounts() -> None:
    """One transaction seen twice moved its value once (audit defect 2)."""
    row = {"txid": "tx1", "timestamp": "1600000000", "input_addresses": ["1a"], "input_amounts": ["2.0"],
           "output_addresses": ["1b"], "output_amounts": ["1.9"], "fee": 0.1}
    once = derive_canonical_address_features(pd.DataFrame([row]))
    twice = derive_canonical_address_features(pd.DataFrame([row, {**row, "timestamp": "1600000005", "src_ip": "1.1.1.1"}]))
    pd.testing.assert_frame_equal(once.reset_index(drop=True), twice.reset_index(drop=True))


def _run(tmp_path, capture: Path, name: str):
    from obsidianchain.pipeline.orchestrator import run_pipeline
    return run_pipeline(capture, runs_dir=tmp_path / name / "runs")


def _stripped(tmp_path) -> Path:
    data = json.loads(SAMPLE.read_text())
    for r in data["records"]:
        for k in ("src_ip", "dst_ip", "src_port", "dst_port", "asn", "geo_country", "observer_id", "observer"):
            r.pop(k, None)
    path = tmp_path / "no_network.json"
    path.write_text(json.dumps(data))
    return path


def test_network_fields_do_not_change_any_score(tmp_path) -> None:
    """The boundary the audit found: network evidence is context, not risk.

    Before the observation double-count fix, stripping network fields moved
    fused scores for 10 clusters. Now model and fused scores must be equal.
    """
    with_net = _run(tmp_path, SAMPLE, "with")
    without = _run(tmp_path, _stripped(tmp_path), "without")
    a = {x.alert_id: x for x in with_net.alert_result.alerts}
    b = {x.alert_id: x for x in without.alert_result.alerts}
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].fused_risk_score == pytest.approx(b[k].fused_risk_score, abs=1e-12), k
        assert a[k].severity == b[k].severity, k


def test_the_run_publishes_propagation_and_names_features(tmp_path) -> None:
    out = _run(tmp_path, SAMPLE, "run")
    assert "network_propagation.json" in out.manifest["artifacts"]
    net = json.loads((out.run_dir / "network_propagation.json").read_text())
    assert net["schema"] == prop.SCHEMA and net["summary"]["transactions_with_observations"] > 0
    stage5 = next(s for s in out.manifest["stages"] if s["stage_number"] == 5)
    assert "propagation" in stage5["summary"]
    for alert in out.alert_result.alerts:
        for e in alert.evidence:
            if e.signal_name == "supervised_risk_model" and e.status == "PRESENT":
                assert ":  (" not in e.explanation and "unnamed feature" not in e.explanation
            if e.signal_name == "p2p_network_telemetry" and e.status == "PRESENT":
                assert e.details["fused"] is False and "not the sender" in e.explanation
