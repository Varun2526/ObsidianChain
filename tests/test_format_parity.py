"""CSV, JSON and XML captures are one ingestion path, end to end.

The parser-level tests (test_phase9_ingest.py) show the three formats give
the same column shape. These go further, on the demonstration capture and
on a minimal two-record capture:

1. every canonical field, including ``script_type`` and the ``observer_id``
   extension, has the same value after ingestion whatever the format;
2. a JSON and an XML upload go through the product's own upload ->
   validation -> analysis path, exactly like the CSV, and produce the same
   validation report, the same ranked alerts and the same network
   propagation.

The JSON and XML files are derived from the CSV here, so the comparison is
of one capture in three encodings, not of three captures.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from obsidianchain.api.app import create_app
from obsidianchain.console.demo_reset import reset_clean_demo_database
from obsidianchain.io import ingest

ROOT = Path(__file__).resolve().parents[1]
DEMO_CSV = ROOT / "data" / "samples" / "demo_capture_synthetic.csv"
LISTS = ("input_addresses", "output_addresses", "input_amounts", "output_amounts")
NUMBERS = ("timestamp", "src_port", "dst_port", "fee", "asn")

MINIMAL_CSV = (
    "timestamp,src_ip,dst_ip,src_port,dst_port,txid,input_addresses,output_addresses,"
    "input_amounts,output_amounts,fee,script_type,geo_country,asn,observer_id\n"
    "1789977754.684,129.227.85.81,198.51.100.11,8333,8333,aa01,1AAA;1BBB,1CCC;1DDD,"
    "1.5;2.0,3.0;0.49,0.01,p2wpkh,,64519,obs-fra-1\n"
    "1789977755.100,62.180.106.68,198.51.100.12,8333,8333,aa02,1CCC,1EEE;1FFF,"
    "3.0,2.5;0.4999,0.0001,p2pkh,DE,64526,obs-sgp-1\n"
)


def _rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _typed(row: dict) -> dict:
    """A CSV row as a JSON producer would write it: lists as arrays, numbers
    as numbers, empty cells as null. A value that is not a number (the
    planted bad fee) stays a string, as it would in a real export."""
    out: dict = {}
    for key, value in row.items():
        if value == "":
            out[key] = None
        elif key in LISTS:
            out[key] = value.split(";")
        elif key in NUMBERS:
            try:
                out[key] = float(value) if "." in value else int(value)
            except ValueError:
                out[key] = value
        else:
            out[key] = value
    return out


def to_json(csv_path: Path, out: Path) -> Path:
    out.write_text(json.dumps({"records": [_typed(r) for r in _rows(csv_path)]}), encoding="utf-8")
    return out


def to_xml(csv_path: Path, out: Path) -> Path:
    parts = ["<records>"]
    for row in _rows(csv_path):
        parts.append("<record>")
        for key, value in row.items():
            if value == "":
                continue
            if key in LISTS:
                items = "".join(f"<item>{escape(v)}</item>" for v in value.split(";"))
                parts.append(f"<{key}>{items}</{key}>")
            else:
                parts.append(f"<{key}>{escape(value)}</{key}>")
        parts.append("</record>")
    parts.append("</records>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


@pytest.fixture(params=["demo", "minimal"])
def captures(request, tmp_path) -> dict[str, Path]:
    if request.param == "demo":
        source = DEMO_CSV
    else:
        source = tmp_path / "minimal.csv"
        source.write_text(MINIMAL_CSV, encoding="utf-8")
    return {"csv": source,
            "json": to_json(source, tmp_path / "capture.json"),
            "xml": to_xml(source, tmp_path / "capture.xml")}


def _normalised(path: Path) -> tuple[pd.DataFrame, dict]:
    frame, report = ingest.ingest(path)
    frame = frame.copy()
    for column in LISTS:
        frame[column] = frame[column].map(lambda v: [str(x) for x in v])
    return frame, report.as_dict()


def test_every_field_means_the_same_in_every_format(captures) -> None:
    base, base_report = _normalised(captures["csv"])
    assert base["script_type"].notna().all(), "script_type is preserved"
    assert base["observer_id"].notna().all(), "observer identity is preserved"
    for fmt in ("json", "xml"):
        frame, report = _normalised(captures[fmt])
        assert report["source_format"] == fmt
        for key in ("rows_read", "rows_valid", "exact_duplicates_rejected",
                    "network_observations_preserved", "errors", "warnings"):
            assert report[key] == base_report[key], (fmt, key)
        assert list(frame.columns) == list(base.columns)
        for column in base.columns:
            left = base[column].astype(object).where(base[column].notna(), None).tolist()
            right = frame[column].astype(object).where(frame[column].notna(), None).tolist()
            if column in NUMBERS:
                left = [None if v is None else float(v) for v in left]
                right = [None if v is None else float(v) for v in right]
            else:
                left = [v if v is None or isinstance(v, list) else str(v) for v in left]
                right = [v if v is None or isinstance(v, list) else str(v) for v in right]
            assert left == right, (fmt, column)


def test_json_and_xml_uploads_reach_the_same_analysis(tmp_path) -> None:
    creds = reset_clean_demo_database(tmp_path, confirm=True)
    client = TestClient(create_app(tmp_path))
    client.post("/api/auth/login", json={"username": creds["investigator"]["username"],
                                         "password": creds["investigator"]["password"]})
    files = {"csv": DEMO_CSV,
             "json": to_json(DEMO_CSV, tmp_path / "demo.json"),
             "xml": to_xml(DEMO_CSV, tmp_path / "demo.xml")}
    outcome = {}
    for fmt, path in files.items():
        case = client.post("/api/investigations", json={"name": f"{fmt} upload"}).json()["id"]
        uploaded = client.post(f"/api/investigations/{case}/datasets?filename={path.name}",
                               content=path.read_bytes(),
                               headers={"Content-Type": "application/octet-stream"})
        assert uploaded.status_code in (200, 201), uploaded.text
        dataset = uploaded.json()["dataset"]
        assert dataset["status"] == "VALIDATED" and dataset["format"] == fmt
        run = client.post(f"/api/investigations/{case}/datasets/{dataset['id']}/run?wait=true").json()
        results = client.get(f"/api/investigations/{case}/runs/{run['run_id']}/results?limit=500").json()
        network = client.get(f"/api/investigations/{case}/runs/{run['run_id']}/network").json()
        validation = dataset["validation"]["validation"]
        outcome[fmt] = {
            "validation": {k: validation[k] for k in ("rows_read", "rows_valid", "warnings", "errors")},
            "model": results["model"]["version"],
            "alerts": [(a["rank"], a["primary_address"], a["member_count"], a["severity"], a["fused_risk_score"])
                       for a in results["alerts"]],
            "network": network["summary"],
        }
    assert outcome["csv"]["model"] is not None
    assert len(outcome["csv"]["alerts"]) > 100
    assert outcome["csv"]["network"]["transactions_with_observations"] == 68
    assert outcome["json"] == outcome["csv"]
    assert outcome["xml"] == outcome["csv"]
