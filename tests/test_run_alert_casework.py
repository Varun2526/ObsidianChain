"""Alerts from a case's own uploaded run can be referenced and decided on.

Before this, only reference-artifact alerts could enter a case, so an
uploaded capture's alerts could be looked at but never reach a decision,
a report or a review.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from obsidianchain.api.app import create_app
from obsidianchain.console.demo_reset import reset_clean_demo_database

SAMPLE = Path(__file__).parent / "data" / "synthetic_acceptance_capture.json"


@pytest.fixture
def setup(tmp_path):
    creds = reset_clean_demo_database(tmp_path, confirm=True)
    app = create_app(tmp_path)

    def client(role):
        c = TestClient(app)
        c.post("/api/auth/login", json={"username": creds[role]["username"],
                                        "password": creds[role]["password"]})
        return c
    return client


def _analysed_case(inv):
    case_id = inv.post("/api/investigations", json={"name": "Uploaded run case"}).json()["id"]
    ds = inv.post(f"/api/investigations/{case_id}/datasets?filename=c.json", content=SAMPLE.read_bytes(),
                  headers={"Content-Type": "application/octet-stream"}).json()["dataset"]["id"]
    run = inv.post(f"/api/investigations/{case_id}/datasets/{ds}/run?wait=true").json()
    run_id = run["run_id"]
    results = inv.get(f"/api/investigations/{case_id}/runs/{run_id}/results").json()
    return case_id, run_id, results


def test_run_alerts_carry_a_case_reference_id(setup):
    inv = setup("investigator")
    _case, _run, results = _analysed_case(inv)
    fp = results["run_fingerprint"][:16]
    refs = [a["alert_ref"] for a in results["alerts"]]
    assert refs and all(r.startswith(fp + ":") for r in refs)
    assert refs[0] == f"{fp}:{results['alerts'][0]['rank']}"


def test_reference_decide_and_report_an_uploaded_run_alert(setup):
    inv = setup("investigator")
    case_id, run_id, results = _analysed_case(inv)
    alert_ref = results["alerts"][0]["alert_ref"]

    added = inv.post(f"/api/investigations/{case_id}/alerts", json={"alert_id": alert_ref})
    assert added.status_code == 201, added.text

    detail = inv.get(f"/api/investigations/{case_id}/alerts/{alert_ref}").json()
    analytical = detail["analytical"]
    assert analytical["available"] is True and analytical["source"] == "UPLOADED_RUN"
    view = analytical["run_alert"]
    assert view["run_id"] == run_id and view["alert"]["rank"] == 1
    assert view["members"], "the alert's member addresses are listed"
    kinds = {n["kind"] for n in view["graph"]["nodes"]}
    assert {"cluster", "address", "transaction"} <= kinds
    assert view["cross_layer"] is not None and "hop_pairs" in view["cross_layer"]["details"]
    ids = {n["id"] for n in view["graph"]["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in view["graph"]["edges"])

    listed = inv.get(f"/api/investigations/{case_id}/alerts").json()
    assert listed["alerts"][0]["stale"] is False
    assert listed["current_artifact_run"] == alert_ref.split(":")[0]

    decided = inv.post(f"/api/investigations/{case_id}/alerts/{alert_ref}/disposition",
                       json={"state": "CONFIRMED", "rationale": "Peel chain confirmed from the uploaded capture."})
    assert decided.status_code == 200, decided.text

    report = inv.post(f"/api/investigations/{case_id}/report",
                      json={"title": "R", "executive_summary": "S", "content": "C"}).json()
    assert report["analytical_run"]["status"] == "CURRENT"
    assert not report["stale_references"]
    assert inv.get(f"/api/investigations/{case_id}").json()["analytical_run"]["status"] == "CURRENT"


def test_unknown_rank_and_foreign_run_are_refused(setup):
    inv = setup("investigator")
    case_id, _run, results = _analysed_case(inv)
    fp = results["run_fingerprint"][:16]
    assert inv.post(f"/api/investigations/{case_id}/alerts", json={"alert_id": f"{fp}:9999"}).status_code == 404
    # another case cannot reference this case's run alerts
    other = inv.post("/api/investigations", json={"name": "Other"}).json()["id"]
    refused = inv.post(f"/api/investigations/{other}/alerts", json={"alert_id": results["alerts"][0]["alert_ref"]})
    assert refused.status_code != 201


def test_run_alert_preview_before_reference(setup):
    inv = setup("investigator")
    case_id, _run, results = _analysed_case(inv)
    ref = results["alerts"][1]["alert_ref"]
    preview = inv.get(f"/api/investigations/{case_id}/run-alerts/{ref}")
    assert preview.status_code == 200 and preview.json()["referenced"] is False
    assert preview.json()["run_alert"]["alert"]["rank"] == 2
    inv.post(f"/api/investigations/{case_id}/alerts", json={"alert_id": ref})
    assert inv.get(f"/api/investigations/{case_id}/run-alerts/{ref}").json()["referenced"] is True
    assert inv.get(f"/api/investigations/{case_id}/run-alerts/{ref.split(':')[0]}:999").status_code == 404
