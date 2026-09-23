"""Seed-wallet risk propagation (ml/propagation.py) and its offline seed sources."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from obsidianchain import geoip
from obsidianchain.io import watchlist
from obsidianchain.ml.propagation import propagate, seed_paths
from obsidianchain.pipeline.orchestrator import run_pipeline


def _edges(pairs):
    return pd.DataFrame(pairs, columns=["address", "txid"])


#   seed --t1-- a --t2-- b --t3-- c        far: d --t4-- e (disconnected)
CHAIN = _edges([
    ("seed", "t1"), ("a", "t1"),
    ("a", "t2"), ("b", "t2"),
    ("b", "t3"), ("c", "t3"),
    ("d", "t4"), ("e", "t4"),
])


def test_risk_decays_with_distance_and_never_crosses_a_gap() -> None:
    r = propagate(CHAIN, ["seed"]).scores
    assert r["seed"] == 1.0
    assert r["a"] > r["b"] > r["c"] > 0
    assert r["d"] == 0.0 and r["e"] == 0.0


def test_a_large_transaction_dilutes_a_seed() -> None:
    """A seed paying one address is a stronger link than a seed inside a
    batch with forty strangers."""
    pair = _edges([("seed", "p"), ("x", "p")])
    crowd = _edges([("seed", "c")] + [(f"s{i}", "c") for i in range(40)] + [("x", "c")])
    assert propagate(pair, ["seed"]).scores["x"] > propagate(crowd, ["seed"]).scores["x"]


def test_hub_transactions_are_excluded_and_reported() -> None:
    hub = _edges([("seed", "hub")] + [(f"u{i}", "hub") for i in range(30)])
    r = propagate(hub, ["seed"], max_tx_degree=10)
    assert r.hub_txids_excluded == ["hub"]
    assert (r.scores.drop("seed") == 0).all()


def test_no_seed_in_graph_is_all_zero_and_says_so() -> None:
    r = propagate(CHAIN, ["not_here"])
    assert r.seeds_in_graph == 0
    assert (r.scores == 0).all()


def test_the_path_names_the_seed_and_every_hop() -> None:
    paths = seed_paths(CHAIN, ["seed"], ["c", "e"])
    assert "e" not in paths
    assert paths["c"]["seed"] == "seed"
    assert paths["c"]["hops"] == 3
    assert [h["via_txid"] for h in paths["c"]["path"]] == ["t1", "t2", "t3"]


def test_ofac_btc_addresses_are_read_from_the_zip(tmp_path: Path) -> None:
    xml = """<?xml version="1.0"?>
<sdnList xmlns="https://example.test/ns">
  <sdnEntry><lastName>EXAMPLE ENTITY</lastName><idList>
    <id><idType>Digital Currency Address - XBT</idType><idNumber>1SeedAddr</idNumber></id>
    <id><idType>Digital Currency Address - ETH</idType><idNumber>0xnotbitcoin</idNumber></id>
  </idList></sdnEntry>
</sdnList>"""
    path = tmp_path / "sdn_xml.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("SDN.XML", xml)
    seeds = watchlist.load_ofac(path)
    assert [(s.address, s.source, s.label) for s in seeds] == [("1SeedAddr", "OFAC_SDN", "EXAMPLE ENTITY")]


def test_csv_watchlists_and_missing_sources(tmp_path: Path) -> None:
    (tmp_path / "cases.csv").write_text("address,label\n1Watched,case 7\n\n")
    seeds = watchlist.load_default_seeds(tmp_path / "absent.zip", tmp_path)
    assert [(s.address, s.source, s.label) for s in seeds] == [("1Watched", "WATCHLIST:cases.csv", "case 7")]
    assert watchlist.load_default_seeds(tmp_path / "absent.zip", tmp_path / "nope") == []


def test_the_pipeline_attaches_propagation_evidence(tmp_path: Path) -> None:
    capture = Path("tests/data/synthetic_acceptance_capture.json")
    outcome = run_pipeline(capture, runs_dir=tmp_path, geoip_provider=geoip.TestFixtureProvider(),
                           seed_addresses=["1PeelSource"])
    s13 = next(s for s in outcome.stages if s.stage_number == 13)
    assert s13.summary["risk_propagation"]["seeds_in_graph"] == 1
    by_member = {a: al for al in outcome.alert_result.alerts for a in al.member_addresses}
    seed_ev = next(e for e in by_member["1PeelSource"].evidence if e.category == "PROPAGATION_CONTEXT")
    assert seed_ev.status == "PRESENT" and seed_ev.score == 1.0
    near = next(e for e in by_member["1PeelPayment1"].evidence if e.category == "PROPAGATION_CONTEXT")
    assert near.status == "PRESENT"
    assert near.details["nearest_seed"] == "1PeelSource"


def test_without_seeds_the_evidence_is_an_explicit_absence(tmp_path: Path) -> None:
    capture = Path("tests/data/synthetic_acceptance_capture.json")
    outcome = run_pipeline(capture, runs_dir=tmp_path, geoip_provider=geoip.TestFixtureProvider(),
                           seed_addresses=[])
    for alert in outcome.alert_result.alerts:
        ev = next(e for e in alert.evidence if e.category == "PROPAGATION_CONTEXT")
        assert ev.status == "NO_EVIDENCE"
        assert ev.details["reason"] == "NO_SEED_WALLET_IN_CAPTURE"


@pytest.mark.parametrize("alpha", [0.15, 0.3, 0.6])
def test_scores_stay_in_unit_interval(alpha: float) -> None:
    r = propagate(CHAIN, ["seed", "c"], alpha=alpha).scores
    assert ((r >= 0) & (r <= 1)).all()
