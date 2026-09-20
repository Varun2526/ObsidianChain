"""Synthetic acceptance dataset generator for ObsidianChain full-system testing.

Generates CSV, JSON, and XML captures conforming strictly to the Problem Statement
(PS) canonical schema:
    timestamp, src_ip, dst_ip, src_port, dst_port, txid, input_addresses,
    output_addresses, input_amounts, output_amounts, fee, script_type,
    geo_country, asn

Contains:
1. Benign transactions (standard 1-in-2-out, normal fee ratio, moderate amounts)
2. Anomalous transactions (extreme velocity, outlier fee ratio, massive amounts)
3. Peeling chain (sequential peel with change addresses and peel amounts)
4. Mixing transactions (equal-output CoinJoin structure with multiple inputs)
5. Multi-observer network observations (same txid observed from distinct vantage points)
6. Edge cases:
   - Duplicate blockchain record (exact duplicate to verify deduplication)
   - NO_NETWORK_OBSERVATION (blockchain tx with no network telemetry)
   - UNMATCHED_NETWORK_OBSERVATION (network telemetry with no blockchain tx)
   - Reserved RFC 5737 / 1918 IPs vs routable IPs with GeoIP mappings
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_synthetic_records() -> list[dict[str, Any]]:
    """Build the canonical list of synthetic records."""
    records: list[dict[str, Any]] = [
        # ---- 1. Benign Transactions --------------------------------------
        {
            "txid": "tx_benign_01",
            "timestamp": "1600000010",
            "src_ip": "198.51.100.10",
            "dst_ip": "198.51.100.1",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": ["1BenignAddr1", "1BenignAddr2"],
            "input_amounts": ["1.50000000", "2.00000000"],
            "output_addresses": ["1MerchantA", "1BenignChange1"],
            "output_amounts": ["3.00000000", "0.49000000"],
            "fee": 0.01000000,
            "script_type": "p2pkh",
            "geo_country": None,
            "asn": 65001,
        },
        {
            "txid": "tx_benign_02",
            "timestamp": "1600000050",
            "src_ip": "8.8.8.8",
            "dst_ip": "198.51.100.1",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": ["1BenignChange1"],
            "input_amounts": ["0.49000000"],
            "output_addresses": ["1MerchantB", "1BenignChange2"],
            "output_amounts": ["0.35000000", "0.13800000"],
            "fee": 0.00200000,
            "script_type": "p2pkh",
            "geo_country": "US",
            "asn": 15169,
        },
        {
            "txid": "tx_benign_03",
            "timestamp": "1600000100",
            "src_ip": "1.1.1.1",
            "dst_ip": "198.51.100.2",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": ["1BenignAddr3"],
            "input_amounts": ["5.00000000"],
            "output_addresses": ["1BenignAddr4", "1BenignChange3"],
            "output_amounts": ["2.00000000", "2.99000000"],
            "fee": 0.01000000,
            "script_type": "p2pkh",
            "geo_country": "AU",
            "asn": 13335,
        },
        # ---- 2. Anomalous Entity Transactions -----------------------------
        # Address 1AnomWhale exhibits extreme velocity and an outlier fee ratio (50% fee!)
        {
            "txid": "tx_anom_01",
            "timestamp": "1600000200",
            "src_ip": "192.0.2.77",
            "dst_ip": "198.51.100.5",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": ["1AnomWhale"],
            "input_amounts": ["100.00000000"],
            "output_addresses": ["1DrainAddr1", "1DrainAddr2"],
            "output_amounts": ["50.00000000", "25.00000000"],
            "fee": 25.00000000,  # 25% fee ratio!
            "script_type": "p2wpkh",
            "geo_country": None,
            "asn": 65100,
        },
        {
            "txid": "tx_anom_02",
            "timestamp": "1600000201",  # 1 second later: extreme velocity
            "src_ip": "192.0.2.77",
            "dst_ip": "198.51.100.5",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": ["1AnomWhale"],
            "input_amounts": ["200.00000000"],
            "output_addresses": ["1DrainAddr3"],
            "output_amounts": ["150.00000000"],
            "fee": 50.00000000,  # 25% fee ratio!
            "script_type": "p2wpkh",
            "geo_country": None,
            "asn": 65100,
        },
        # ---- 3. Peeling Chain ---------------------------------------------
        # 1PeelSource peels off small payments: peel_1 -> peel_2 -> peel_3
        {
            "txid": "tx_peel_01",
            "timestamp": "1600000300",
            "src_ip": "203.0.113.20",
            "dst_ip": "198.51.100.1",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": ["1PeelSource"],
            "input_amounts": ["50.00000000"],
            "output_addresses": ["1PeelPayment1", "1PeelChange1"],
            "output_amounts": ["1.00000000", "48.99000000"],
            "fee": 0.01000000,
            "script_type": "p2pkh",
            "geo_country": None,
            "asn": 65002,
        },
        {
            "txid": "tx_peel_02",
            "timestamp": "1600000350",
            "src_ip": "203.0.113.21",
            "dst_ip": "198.51.100.1",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": ["1PeelChange1"],
            "input_amounts": ["48.99000000"],
            "output_addresses": ["1PeelPayment2", "1PeelChange2"],
            "output_amounts": ["1.00000000", "47.98000000"],
            "fee": 0.01000000,
            "script_type": "p2pkh",
            "geo_country": None,
            "asn": 65002,
        },
        {
            "txid": "tx_peel_03",
            "timestamp": "1600000400",
            "src_ip": "203.0.113.22",
            "dst_ip": "198.51.100.1",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": ["1PeelChange2"],
            "input_amounts": ["47.98000000"],
            "output_addresses": ["1PeelPayment3", "1PeelChange3"],
            "output_amounts": ["1.00000000", "46.97000000"],
            "fee": 0.01000000,
            "script_type": "p2pkh",
            "geo_country": None,
            "asn": 65002,
        },
        # ---- 4. Mixing / Equal-Output Pattern ----------------------------
        # 3 distinct input addresses, equal denomination outputs (10.0 BTC)
        {
            "txid": "tx_mix_01",
            "timestamp": "1600000500",
            "src_ip": "203.0.113.55",
            "dst_ip": "198.51.100.1",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": ["1MixerIn1", "1MixerIn2", "1MixerIn3"],
            "input_amounts": ["10.02000000", "10.02000000", "10.02000000"],
            "output_addresses": ["1MixerOut1", "1MixerOut2", "1MixerOut3", "1MixChange"],
            "output_amounts": ["10.00000000", "10.00000000", "10.00000000", "0.03000000"],
            "fee": 0.03000000,
            "script_type": "p2sh",
            "geo_country": None,
            "asn": 65003,
        },
        # ---- 5. Multi-Observer Network Observations ----------------------
        # The same txid observed from two distinct observer vantage points
        {
            "txid": "tx_multi_obs",
            "timestamp": "1600000600",
            "src_ip": "8.8.8.8",
            "dst_ip": "198.51.100.1",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": ["1MultiIn"],
            "input_amounts": ["2.00000000"],
            "output_addresses": ["1MultiOut"],
            "output_amounts": ["1.99000000"],
            "fee": 0.01000000,
            "script_type": "p2pkh",
            "geo_country": "US",
            "asn": 15169,
            "observer_id": "probe-us-east",
        },
        {
            "txid": "tx_multi_obs",
            "timestamp": "1600000604",
            "src_ip": "1.1.1.1",
            "dst_ip": "198.51.100.2",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": ["1MultiIn"],
            "input_amounts": ["2.00000000"],
            "output_addresses": ["1MultiOut"],
            "output_amounts": ["1.99000000"],
            "fee": 0.01000000,
            "script_type": "p2pkh",
            "geo_country": "AU",
            "asn": 13335,
            "observer_id": "probe-au-south",
        },
        # ---- 6. Edge Cases -----------------------------------------------
        # 6a. Blockchain record with NO network observation
        {
            "txid": "tx_no_net_01",
            "timestamp": "1600000700",
            "src_ip": None,
            "dst_ip": None,
            "src_port": None,
            "dst_port": None,
            "input_addresses": ["1NoNetIn"],
            "input_amounts": ["3.00000000"],
            "output_addresses": ["1NoNetOut"],
            "output_amounts": ["2.99000000"],
            "fee": 0.01000000,
            "script_type": "p2pkh",
            "geo_country": None,
            "asn": None,
        },
        # 6b. Unmatched network observation (no blockchain tx)
        {
            "txid": "tx_unmatched_net_01",
            "timestamp": "1600000800",
            "src_ip": "198.51.100.99",
            "dst_ip": "198.51.100.1",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": [],
            "input_amounts": [],
            "output_addresses": [],
            "output_amounts": [],
            "fee": None,
            "script_type": None,
            "geo_country": None,
            "asn": 65004,
        },
        # 6c. Exact duplicate of tx_benign_01 (should be rejected and counted)
        {
            "txid": "tx_benign_01",
            "timestamp": "1600000010",
            "src_ip": "198.51.100.10",
            "dst_ip": "198.51.100.1",
            "src_port": 8333,
            "dst_port": 8333,
            "input_addresses": ["1BenignAddr1", "1BenignAddr2"],
            "input_amounts": ["1.50000000", "2.00000000"],
            "output_addresses": ["1MerchantA", "1BenignChange1"],
            "output_amounts": ["3.00000000", "0.49000000"],
            "fee": 0.01000000,
            "script_type": "p2pkh",
            "geo_country": None,
            "asn": 65001,
        },
    ]
    return records


def write_synthetic_json(path: Path) -> Path:
    """Write synthetic dataset to JSON file."""
    records = build_synthetic_records()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"records": records}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_synthetic_csv(path: Path) -> Path:
    """Write synthetic dataset to CSV file."""
    records = build_synthetic_records()
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid",
        "input_addresses", "output_addresses", "input_amounts", "output_amounts",
        "fee", "script_type", "geo_country", "asn",
    ]
    lines = [",".join(headers) + "\n"]
    for r in records:
        row = []
        for h in headers:
            val = r.get(h)
            if val is None:
                row.append("")
            elif isinstance(val, list):
                row.append(";".join(str(v) for v in val))
            else:
                row.append(str(val))
        lines.append(",".join(row) + "\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def write_synthetic_xml(path: Path) -> Path:
    """Write synthetic dataset to XML file."""
    records = build_synthetic_records()
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = ["<records>\n"]
    for r in records:
        parts.append("  <record>\n")
        for key, value in r.items():
            if value is None:
                continue
            if isinstance(value, list):
                inner = "".join(f"<item>{v}</item>" for v in value)
                parts.append(f"    <{key}>{inner}</{key}>\n")
            else:
                parts.append(f"    <{key}>{value}</{key}>\n")
        parts.append("  </record>\n")
    parts.append("</records>\n")
    path.write_text("".join(parts), encoding="utf-8")
    return path


def generate_all(dir_path: Path) -> tuple[Path, Path, Path]:
    """Generate all three format files in the target directory."""
    json_path = write_synthetic_json(dir_path / "synthetic_acceptance_capture.json")
    csv_path = write_synthetic_csv(dir_path / "synthetic_acceptance_capture.csv")
    xml_path = write_synthetic_xml(dir_path / "synthetic_acceptance_capture.xml")
    return json_path, csv_path, xml_path


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "data"
    generate_all(out_dir)
    print(f"Generated synthetic captures in {out_dir}")
