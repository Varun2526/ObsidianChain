"""Bulk ingestion (CSV/JSON/XML), GeoIP/ASN resolution, and correlation.

Three things are load-bearing here and each would be invisible if it broke:

1. **Every format produces the SAME normalised frame.** If XML silently
   produced a different column set from CSV, a user would get different
   answers for the same data depending on how they exported it.

2. **A GeoIP lookup never invents a country.** These addresses are RFC 5737
   documentation ranges; a real database returns nothing for them, and so
   must this one.

3. **An announcing peer is never presented as a sender.** 84.3% of
   transactions here were announced by several peers, which is what relay
   looks like.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from obsidianchain import geoip
from obsidianchain.io import ingest

RECORDS = [
    {
        "timestamp": "1500152257183", "src_ip": "198.51.100.191",
        "src_port": "8333", "txid": "1076",
        "input_addresses": ["1aaa", "1bbb"], "output_addresses": ["1ccc"],
        "input_amounts": ["0.5", "0.25"], "output_amounts": ["0.74"],
        "asn": "65162",
    },
    {
        "timestamp": "1500152257999", "src_ip": "203.0.113.7",
        "src_port": "8333", "txid": "1077",
        "input_addresses": ["1ddd"], "output_addresses": ["1eee", "1fff"],
        "input_amounts": ["1.0"], "output_amounts": ["0.6", "0.39"],
        "asn": "65001",
    },
]


def write_csv(path: Path) -> Path:
    header = ("timestamp,src_ip,src_port,txid,input_addresses,"
              "output_addresses,input_amounts,output_amounts,asn\n")
    lines = [header]
    for r in RECORDS:
        lines.append(
            f"{r['timestamp']},{r['src_ip']},{r['src_port']},{r['txid']},"
            f"{';'.join(r['input_addresses'])},{';'.join(r['output_addresses'])},"
            f"{';'.join(r['input_amounts'])},{';'.join(r['output_amounts'])},"
            f"{r['asn']}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def write_json(path: Path) -> Path:
    path.write_text(json.dumps({"records": RECORDS}), encoding="utf-8")
    return path


def write_xml(path: Path) -> Path:
    parts = ["<records>"]
    for r in RECORDS:
        parts.append("<record>")
        for key, value in r.items():
            if isinstance(value, list):
                inner = "".join(f"<item>{v}</item>" for v in value)
                parts.append(f"<{key}>{inner}</{key}>")
            else:
                parts.append(f"<{key}>{value}</{key}>")
        parts.append("</record>")
    parts.append("</records>")
    path.write_text("".join(parts), encoding="utf-8")
    return path


WRITERS = {"csv": write_csv, "json": write_json, "xml": write_xml}


@pytest.mark.parametrize("fmt", ["csv", "json", "xml"])
def test_every_format_is_ingested(tmp_path, fmt) -> None:
    path = WRITERS[fmt](tmp_path / f"sample.{fmt}")
    frame, report = ingest.ingest(path)
    assert report.ok, report.errors
    assert report.source_format == fmt
    assert report.rows_read == 2
    assert report.rows_valid == 2


@pytest.mark.parametrize("fmt", ["csv", "json", "xml"])
def test_the_normalised_shape_is_identical_across_formats(tmp_path, fmt) -> None:
    """The property that makes three parsers one ingestion path."""
    path = WRITERS[fmt](tmp_path / f"sample.{fmt}")
    frame, _ = ingest.ingest(path)
    assert list(frame.columns) == ingest.CANONICAL_COLUMNS
    assert frame["txid"].tolist() == ["1076", "1077"]
    assert frame["input_addresses"].iloc[0] == ["1aaa", "1bbb"]
    assert frame["output_addresses"].iloc[1] == ["1eee", "1fff"]
    assert frame["src_port"].iloc[0] == 8333


def test_a_missing_required_field_is_refused_not_guessed(tmp_path) -> None:
    path = tmp_path / "no_txid.csv"
    path.write_text("timestamp,src_ip\n1500,198.51.100.1\n", encoding="utf-8")
    frame, report = ingest.ingest(path)
    assert not report.ok
    assert "txid" in report.required_missing
    assert frame.empty


def test_rows_without_a_txid_are_rejected_and_counted(tmp_path) -> None:
    path = tmp_path / "partial.csv"
    path.write_text("txid,src_ip\n1076,198.51.100.1\n,203.0.113.2\n", encoding="utf-8")
    frame, report = ingest.ingest(path)
    assert report.rows_read == 2
    assert report.rows_valid == 1
    assert any("rejected" in e for e in report.errors)


def test_absent_optional_fields_are_reported_not_invented(tmp_path) -> None:
    path = tmp_path / "minimal.csv"
    path.write_text("txid\n1076\n", encoding="utf-8")
    frame, report = ingest.ingest(path)
    assert report.ok
    assert "src_ip" in report.columns_missing
    assert frame["src_ip"].isna().all()
    assert any("network-layer correlation is degraded" in w for w in report.warnings)


def test_a_single_value_is_not_split_into_characters(tmp_path) -> None:
    """A bare address must stay one address, not become a list of letters."""
    path = tmp_path / "one.csv"
    path.write_text("txid,input_addresses\n1076,1aaa\n", encoding="utf-8")
    frame, _ = ingest.ingest(path)
    assert frame["input_addresses"].iloc[0] == ["1aaa"]


def test_a_json_array_in_a_csv_cell_is_parsed(tmp_path) -> None:
    path = tmp_path / "jsoncell.csv"
    path.write_text(
        'txid,input_addresses\n1076,"[""1aaa"", ""1bbb""]"\n', encoding="utf-8"
    )
    frame, _ = ingest.ingest(path)
    assert frame["input_addresses"].iloc[0] == ["1aaa", "1bbb"]


def test_malformed_input_raises_rather_than_returning_empty(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ingest.IngestError):
        ingest.ingest(path)


def test_an_unknown_suffix_is_refused(tmp_path) -> None:
    path = tmp_path / "data.bin"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ingest.IngestError, match="(?i)supported"):
        ingest.ingest(path)


def test_correlation_summary_counts_only_joinable_records(tmp_path) -> None:
    path = write_csv(tmp_path / "s.csv")
    frame, _ = ingest.ingest(path)
    summary = ingest.correlation_summary(frame)
    assert summary["transactions"] == 2
    assert summary["source_ips"] == 2
    assert summary["addresses"] == 6
    # Both rows carry an IP AND wallets, so both are correlatable.
    assert summary["correlatable_records"] == 2


def test_correlation_summary_is_zero_without_network_fields(tmp_path) -> None:
    path = tmp_path / "walletsonly.csv"
    path.write_text("txid,input_addresses\n1076,1aaa\n", encoding="utf-8")
    frame, _ = ingest.ingest(path)
    assert ingest.correlation_summary(frame)["correlatable_records"] == 0


# ---- GeoIP / ASN --------------------------------------------------------


@pytest.mark.parametrize("ip,rfc", [
    ("192.0.2.164", "RFC 5737"),
    ("198.51.100.191", "RFC 5737"),
    ("203.0.113.7", "RFC 5737"),
    ("10.1.2.3", "RFC 1918"),
    ("127.0.0.1", "RFC 1122"),
])
def test_reserved_ranges_are_identified_and_get_no_country(ip, rfc) -> None:
    """The dataset's own addresses. A real database returns nothing for
    these too - reporting a country would be an invention."""
    facts = geoip.resolve_ip(ip)
    assert facts.valid
    assert facts.rfc == rfc
    assert facts.globally_routable is False
    assert facts.country_iso is None
    assert "not globally routable" in facts.country_source


def test_a_routable_address_has_no_country_without_a_database() -> None:
    facts = geoip.resolve_ip("8.8.8.8")
    assert facts.globally_routable is True
    assert facts.country_iso is None
    assert "no GeoIP database is installed" in facts.country_source


def test_a_vendored_database_resolves_a_country(tmp_path) -> None:
    """The pluggable path is real, not hypothetical."""
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "geoip_country.csv").write_text(
        "network,country_iso,country_name\n8.8.8.0/24,US,United States\n",
        encoding="utf-8",
    )
    database = geoip.load_geoip_database(tmp_path)
    assert database
    facts = geoip.resolve_ip("8.8.8.8", database)
    assert facts.country_iso == "US"
    assert facts.country_source == "geoip_database"


def test_a_reserved_address_is_still_refused_a_country_with_a_database(tmp_path) -> None:
    """Precedence matters: a bogus database entry must not override the
    registry and paint a country onto a documentation range."""
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "geoip_country.csv").write_text(
        "network,country_iso,country_name\n198.51.100.0/24,XX,Nowhere\n",
        encoding="utf-8",
    )
    database = geoip.load_geoip_database(tmp_path)
    facts = geoip.resolve_ip("198.51.100.191", database)
    assert facts.country_iso is None


def test_an_invalid_address_is_reported_as_invalid() -> None:
    facts = geoip.resolve_ip("not-an-ip")
    assert facts.valid is False
    assert facts.country_iso is None


@pytest.mark.parametrize("asn", [64512, 65000, 65534])
def test_private_use_asns_are_identified(asn) -> None:
    facts = geoip.resolve_asn(asn)
    assert facts.private_use is True
    assert facts.rfc == "RFC 6996"
    assert "identifies no real network operator" in facts.description


def test_a_public_asn_is_not_called_private() -> None:
    assert geoip.resolve_asn(15169).private_use is False


def test_summarise_reports_unavailable_rather_than_empty() -> None:
    facts = geoip.summarise(
        ["198.51.100.1", "203.0.113.2"], [65000, 65001],
    )
    assert facts["countries"] == []
    assert facts["country_resolution"].startswith("unavailable")
    assert facts["all_private_asns"] is True
    assert facts["geoip_database_installed"] is False
    assert facts["reserved_ranges"]


# ---- Duplicate Identity & Preservation ----------------------------------


def test_blockchain_key_preserves_address_amount_correspondence() -> None:
    """Inputs and amounts must remain paired as tuples. Swapping amounts across
    addresses must change the key, but reordering identical pairs must not."""
    rec1 = {
        "txid": "tx1",
        "input_addresses": ["addrA", "addrB"],
        "input_amounts": ["1.0", "5.0"],
        "output_addresses": ["addrC"],
        "output_amounts": ["5.99"],
        "fee": "0.01",
        "script_type": "p2pkh",
    }
    # Reordered input pairs: (addrB, 5.0), (addrA, 1.0) -> same canonical set
    rec2 = {
        "txid": "tx1",
        "input_addresses": ["addrB", "addrA"],
        "input_amounts": ["5.0", "1.0"],
        "output_addresses": ["addrC"],
        "output_amounts": ["5.99"],
        "fee": "0.01",
        "script_type": "p2pkh",
    }
    # Swapped amounts: addrA has 5.0, addrB has 1.0 -> DIFFERENT canonical tuples!
    rec3 = {
        "txid": "tx1",
        "input_addresses": ["addrA", "addrB"],
        "input_amounts": ["5.0", "1.0"],
        "output_addresses": ["addrC"],
        "output_amounts": ["5.99"],
        "fee": "0.01",
        "script_type": "p2pkh",
    }
    k1 = ingest.blockchain_record_key(rec1)
    k2 = ingest.blockchain_record_key(rec2)
    k3 = ingest.blockchain_record_key(rec3)
    assert k1 == k2, "Reordered canonical pairs must produce identical blockchain key"
    assert k1 != k3, "Swapped amounts between addresses must produce different blockchain key"


def test_network_observation_key_includes_observer_identity() -> None:
    """Distinct observers reporting the same txid, ip, port, timestamp are distinct."""
    obs1 = {
        "txid": "tx1", "observer_id": "probe-east-1",
        "src_ip": "1.2.3.4", "src_port": "8333", "timestamp": "1600000000",
    }
    obs2 = {
        "txid": "tx1", "observer_id": "probe-west-2",
        "src_ip": "1.2.3.4", "src_port": "8333", "timestamp": "1600000000",
    }
    obs_no_id = {
        "txid": "tx1",
        "src_ip": "1.2.3.4", "src_port": "8333", "timestamp": "1600000000",
    }
    k1 = ingest.observation_key(obs1)
    k2 = ingest.observation_key(obs2)
    k_none = ingest.observation_key(obs_no_id)

    assert k1 != k2, "Distinct observer IDs must produce distinct observation keys"
    assert k1 != k_none, "Absence of observer_id must not collide with present observer_id"


def test_exact_blockchain_duplicates_are_rejected_and_reported(tmp_path) -> None:
    path = tmp_path / "dup.csv"
    path.write_text(
        "txid,input_addresses,input_amounts,output_addresses,output_amounts\n"
        "tx1,addrA,1.0,addrB,0.99\n"
        "tx1,addrA,1.0,addrB,0.99\n"  # Exact duplicate
        "tx2,addrC,2.0,addrD,1.99\n",
        encoding="utf-8",
    )
    frame, report = ingest.ingest(path)
    assert len(frame) == 2
    assert report.rows_read == 3
    assert report.rows_valid == 2
    assert report.exact_duplicates_rejected == 1
    assert report.network_observations_preserved == 0
    assert any("duplicate" in w for w in report.warnings)


def test_multi_observer_network_observations_are_preserved(tmp_path) -> None:
    path = tmp_path / "multi_obs.csv"
    path.write_text(
        "txid,src_ip,src_port,timestamp\n"
        "tx1,198.51.100.1,8333,1600000001\n"
        "tx1,203.0.113.2,8333,1600000002\n"  # Different network vantage point
        "tx1,198.51.100.1,8333,1600000001\n",  # Exact duplicate of line 1
        encoding="utf-8",
    )
    frame, report = ingest.ingest(path)
    assert len(frame) == 2
    assert report.rows_read == 3
    assert report.rows_valid == 2
    assert report.exact_duplicates_rejected == 1
    assert report.network_observations_preserved == 1


def test_geoip_provider_interface_and_test_fixture() -> None:
    provider = geoip.TestFixtureProvider()
    assert provider.version == "test-fixture-1.0"
    assert len(provider.sha256) == 64

    # Routable mapped IP resolves to country
    us_facts = provider.resolve_ip("8.8.8.8")
    assert us_facts.country_iso == "US"
    assert us_facts.country_name == "United States"

    # RFC 5737 documentation IP is NEVER mapped to a country, even if in DB
    rfc_facts = provider.resolve_ip("198.51.100.1")
    assert rfc_facts.country_iso is None
    assert "reserved and not globally routable" in rfc_facts.country_source

    # Summary with provider
    summary = geoip.summarise(["8.8.8.8", "198.51.100.1"], [65000], provider=provider)
    assert summary["countries"] == ["US"]
    assert summary["provider_version"] == "test-fixture-1.0"
    assert summary["provider_sha256"] == provider.sha256


