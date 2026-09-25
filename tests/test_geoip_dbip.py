"""DB-IP "IP to Country Lite" support: range lookup, attribution, wiring.

Uses a small fixture in the DB-IP layout, never the vendored file, so the
suite does not depend on data/reference/ being present.
"""

from __future__ import annotations

import gzip
import ipaddress
from types import SimpleNamespace

import pytest

from obsidianchain import geoip
from obsidianchain.network import propagation

ROWS = [
    "1.0.0.0,1.0.0.255,AU",
    "8.8.8.0,8.8.8.255,US",
    "49.44.0.0,49.47.255.255,IN",
    "100.0.0.0,100.0.0.255,ZZ",
    "2001:4860::,2001:4860:ffff:ffff:ffff:ffff:ffff:ffff,US",
    "not-an-ip,1.2.3.4,XX",
]


@pytest.fixture()
def data_root(tmp_path):
    ref = tmp_path / "reference"
    ref.mkdir()
    with gzip.open(ref / "dbip-country-lite-2026-09.csv.gz", "wt", encoding="utf-8") as fh:
        fh.write("\n".join(ROWS) + "\n")
    geoip._RANGE_CACHE.clear()
    return tmp_path


def test_range_database_lookup_ipv4_ipv6_and_gaps():
    db = geoip.RangeDatabase([r.split(",") for r in ROWS], source="t", version="v")
    assert db.ranges == 5  # the malformed line is skipped
    look = lambda ip: db.lookup(ipaddress.ip_address(ip))  # noqa: E731
    assert look("8.8.8.8") == "US"
    assert look("1.0.0.0") == "AU" and look("1.0.0.255") == "AU"
    assert look("1.0.1.0") is None  # between ranges
    assert look("0.0.0.1") is None  # before the first range
    assert look("49.46.1.1") == "IN"
    assert look("100.0.0.5") is None  # ZZ means unassigned, not a country
    assert look("2001:4860::8888") == "US"
    assert look("2001:db8::1") is None


def test_provider_picks_up_dbip_and_attributes_it(data_root):
    provider = geoip.OfflineCSVProvider(data_root)
    assert provider.version == "dbip-country-lite-2026-09"
    assert provider.attribution == geoip.DBIP_ATTRIBUTION
    assert len(provider.sha256) == 64
    facts = provider.resolve_ip("8.8.8.8")
    assert facts.country_iso == "US"
    assert "dbip-country-lite-2026-09" in facts.country_source
    assert "CC BY 4.0" in facts.country_source


def test_special_purpose_and_unassigned_never_get_a_country(data_root):
    provider = geoip.OfflineCSVProvider(data_root)
    for ip in ("10.0.0.1", "192.0.2.1", "127.0.0.1"):
        assert provider.resolve_ip(ip).country_iso is None
    unassigned = provider.resolve_ip("100.0.0.5")
    assert unassigned.country_iso is None
    assert "not assigned" in unassigned.country_source


def test_uninstalled_provider_reports_so(tmp_path):
    provider = geoip.OfflineCSVProvider(tmp_path)
    assert provider.version == "uninstalled"
    assert provider.attribution is None
    assert provider.resolve_ip("8.8.8.8").country_iso is None


def _obs(txid, src_ip, ts, geo=None):
    return SimpleNamespace(txid=txid, src_ip=src_ip, src_port=8333, dst_ip="198.51.100.1",
                           timestamp=ts, asn=15169, geo_country=geo, observer_id="obs-1")


def test_propagation_keeps_resolved_and_capture_countries_apart(data_root):
    provider = geoip.OfflineCSVProvider(data_root)
    corr = SimpleNamespace(all_observations=[
        _obs("t1", "8.8.8.8", 1_700_000_000, geo="DE"),
        _obs("t1", "49.44.1.1", 1_700_000_002),
        _obs("t1", "10.0.0.1", 1_700_000_003),
    ])
    result = propagation.analyse(corr, provider)
    row = result.transactions["t1"].as_dict()
    assert row["resolved_countries"] == ["IN", "US"]
    assert row["countries"] == ["DE"]  # the capture's claim, untouched
    assert "CC BY 4.0" in row["country_resolution"]
    by_ip = {p["peer_ip"]: p["country_iso"] for p in row["peers"]}
    assert by_ip == {"8.8.8.8": "US", "49.44.1.1": "IN", "10.0.0.1": None}
    pooled = result.pooled(["t1"])
    assert pooled["resolved_countries"] == ["IN", "US"]
    assert result.summary()["distinct_resolved_countries"] == 2


def test_propagation_without_provider_resolves_nothing():
    corr = SimpleNamespace(all_observations=[_obs("t1", "8.8.8.8", 1_700_000_000)])
    row = propagation.analyse(corr).transactions["t1"].as_dict()
    assert row["resolved_countries"] == [] and row["country_resolution"] is None
    assert row["peers"][0]["country_iso"] is None
