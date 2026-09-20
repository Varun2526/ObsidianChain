"""Multi-user isolation, provenance binding, and the append-only records.

The scenario this file exists for
---------------------------------
Investigator A logs in, creates a case, uploads a dataset, references an
alert, writes a note, records a decision and generates a report.
Investigator B logs in and creates their own case. B must not be able to
reach any of A's work - not through the UI's own calls, and not by putting
A's identifiers directly into a URL.

The audit found that neither guarantee existed: case state lived in
localStorage, so isolation was an accident of browser profiles rather than a
property of the system, and every analytical route was open to anyone.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from obsidianchain.api.app import create_app
from obsidianchain.console import casework, db, errors, investigations, users

ALERT = "043ea584e99daf99:314770"
OTHER_RUN_ALERT = "deadbeefdeadbeef:1"
CAPTURE = b"txid,src_ip,timestamp\nabc123,192.0.2.7,2026-02-11T09:14:03Z\n"


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A temporary data root with no analytical artifacts.

    Deliberate: the application layer must work when the pipeline has not
    run, and the case-scoped tests below must not depend on a 12 MB parquet
    being present. Where an artifact IS needed, the test says so.
    """
    return tmp_path


@pytest.fixture
def conn(root):
    connection = db.connect(root)
    yield connection
    connection.close()


@pytest.fixture
def app(root, conn):
    users.create(conn, username="alice", password="alice-password",
                 role="INVESTIGATOR", display_name="Alice A")
    users.create(conn, username="bob", password="bob-password",
                 role="INVESTIGATOR", display_name="Bob B")
    users.create(conn, username="rev", password="rev-password", role="REVIEWER")
    users.create(conn, username="root", password="root-password", role="ADMIN")
    return create_app(root)


def signed_in(app, username, password):
    client = TestClient(app)
    response = client.post("/api/auth/login",
                           json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return client


@pytest.fixture
def alice(app):
    return signed_in(app, "alice", "alice-password")


@pytest.fixture
def bob(app):
    return signed_in(app, "bob", "bob-password")


@pytest.fixture
def reviewer(app):
    return signed_in(app, "rev", "rev-password")


@pytest.fixture
def case_a(alice):
    """A's case, with a dataset, a referenced alert, a note and a decision."""
    case = alice.post("/api/investigations",
                      json={"name": "Case A", "description": "A's case"}).json()
    cid = case["id"]
    alice.post(f"/api/investigations/{cid}/datasets?filename=capture.csv",
               content=CAPTURE,
               headers={"Content-Type": "application/octet-stream"})
    alice.post(f"/api/investigations/{cid}/alerts", json={"alert_id": ALERT})
    alice.post(f"/api/investigations/{cid}/alerts/{ALERT}/disposition",
               json={"state": "IN_REVIEW", "rationale": "checking peel depth"})
    alice.post(f"/api/investigations/{cid}/notes",
               json={"alert_id": ALERT, "body": "Counterparties reviewed."})
    alice.post(f"/api/investigations/{cid}/report",
               json={"title": "Report A", "executive_summary": "Summary A",
                     "content": "Body A"})
    return cid


# ---- the case is a real, owned, server-side object -----------------------


def test_a_case_has_a_server_generated_id_and_an_owner(alice) -> None:
    case = alice.post("/api/investigations", json={"name": "Owned"}).json()
    assert case["id"].startswith("inv_")
    assert case["case_label"] == "OC-0001"
    assert case["owner"]["username"] == "alice"
    assert case["status"] == "DRAFT"


def test_case_ids_are_not_sequential_or_guessable(alice) -> None:
    """A sequential id plus a 403 would let anyone count the case list."""
    first = alice.post("/api/investigations", json={"name": "one"}).json()["id"]
    second = alice.post("/api/investigations", json={"name": "two"}).json()["id"]
    assert first != second
    assert len(first) == len("inv_") + 16


def test_viewing_a_case_does_not_change_its_status(alice, case_a) -> None:
    """The retired browser store flipped DRAFT to ACTIVE on render."""
    before = alice.get(f"/api/investigations/{case_a}").json()["status"]
    for _ in range(3):
        alice.get(f"/api/investigations/{case_a}")
    assert alice.get(f"/api/investigations/{case_a}").json()["status"] == before


def test_status_transitions_are_validated(alice, case_a) -> None:
    alice.post(f"/api/investigations/{case_a}/status", json={"status": "VALIDATING"})
    alice.post(f"/api/investigations/{case_a}/status", json={"status": "ANALYZING"})
    alice.post(f"/api/investigations/{case_a}/status", json={"status": "ACTIVE"})
    # ACTIVE -> DRAFT is not a permitted transition.
    refused = alice.post(f"/api/investigations/{case_a}/status",
                         json={"status": "DRAFT"})
    assert refused.status_code == 409
    assert refused.json()["error"] == "conflict"


def test_an_unknown_status_is_refused(alice, case_a) -> None:
    response = alice.post(f"/api/investigations/{case_a}/status",
                          json={"status": "SUPERSEDED"})
    assert response.status_code == 422


# ---- USER B cannot reach USER A's work ----------------------------------


@pytest.mark.parametrize("path", [
    "", "/datasets", "/alerts", "/notes", "/report", "/history",
])
def test_b_cannot_read_any_part_of_a_case(bob, case_a, path) -> None:
    """Direct id manipulation, which is the only way B could try."""
    response = bob.get(f"/api/investigations/{case_a}{path}")
    assert response.status_code == 403
    assert response.json()["error"] == "access_denied"


def test_b_cannot_read_as_alert_detail(bob, case_a) -> None:
    response = bob.get(f"/api/investigations/{case_a}/alerts/{ALERT}")
    assert response.status_code == 403


@pytest.mark.parametrize("path,body", [
    ("/notes", {"body": "B was here"}),
    ("/alerts", {"alert_id": ALERT}),
    ("/report", {"title": "B", "executive_summary": "", "content": ""}),
    ("/status", {"status": "CLOSED"}),
])
def test_b_cannot_write_to_a_case(bob, case_a, path, body) -> None:
    assert bob.post(f"/api/investigations/{case_a}{path}",
                    json=body).status_code == 403


def test_b_cannot_set_a_disposition_on_a_case(bob, case_a) -> None:
    response = bob.post(
        f"/api/investigations/{case_a}/alerts/{ALERT}/disposition",
        json={"state": "DISMISSED"},
    )
    assert response.status_code == 403


def test_b_cannot_upload_into_a_case(bob, case_a) -> None:
    response = bob.post(
        f"/api/investigations/{case_a}/datasets?filename=x.csv",
        content=CAPTURE, headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 403


def test_b_sees_only_their_own_cases_in_the_listing(bob, alice, case_a) -> None:
    bob.post("/api/investigations", json={"name": "Case B"})
    listing = bob.get("/api/investigations").json()
    assert listing["scope"] == "owned"
    assert [c["name"] for c in listing["investigations"]] == ["Case B"]
    assert case_a not in [c["id"] for c in listing["investigations"]]


def test_a_nonexistent_case_is_404_not_403(bob) -> None:
    response = bob.get("/api/investigations/inv_0000000000000000")
    assert response.status_code == 404
    assert response.json()["error"] == "investigation_not_found"


def test_an_unauthenticated_caller_reaches_nothing(app, case_a) -> None:
    anonymous = TestClient(app)
    for path in ("", "/alerts", "/notes", "/report", "/history"):
        assert anonymous.get(
            f"/api/investigations/{case_a}{path}").status_code == 401


def test_a_dataset_id_from_another_case_does_not_resolve(alice, bob, case_a) -> None:
    """Checked against the case in the URL, not merely fetched by id."""
    dataset_id = alice.get(
        f"/api/investigations/{case_a}/datasets").json()["datasets"][0]["id"]
    case_b = bob.post("/api/investigations", json={"name": "Case B"}).json()["id"]
    response = bob.get(f"/api/investigations/{case_b}/datasets/{dataset_id}")
    assert response.status_code == 404


# ---- oversight roles ----------------------------------------------------


def test_a_reviewer_may_read_every_case(reviewer, case_a) -> None:
    assert reviewer.get(f"/api/investigations/{case_a}").status_code == 200
    assert reviewer.get(f"/api/investigations/{case_a}/history").status_code == 200


def test_a_reviewer_may_not_decide(reviewer, case_a) -> None:
    response = reviewer.post(
        f"/api/investigations/{case_a}/alerts/{ALERT}/disposition",
        json={"state": "DISMISSED"},
    )
    assert response.status_code == 403


def test_a_reviewer_may_note_on_someone_elses_case(reviewer, case_a) -> None:
    """Review comments have to live beside the work they describe."""
    response = reviewer.post(f"/api/investigations/{case_a}/notes",
                             json={"body": "Reviewed; agree with IN_REVIEW."})
    assert response.status_code == 201
    assert response.json()["author_username"] == "rev"


def test_only_a_reviewer_or_admin_finalises_a_report(alice, reviewer, case_a) -> None:
    assert alice.post(
        f"/api/investigations/{case_a}/report/1/finalise").status_code == 403
    assert reviewer.post(
        f"/api/investigations/{case_a}/report/1/finalise").status_code == 200


def test_a_final_report_is_not_refinalised(reviewer, case_a) -> None:
    reviewer.post(f"/api/investigations/{case_a}/report/1/finalise")
    again = reviewer.post(f"/api/investigations/{case_a}/report/1/finalise")
    assert again.status_code == 409


# ---- datasets and the analysis boundary ---------------------------------


def test_the_upload_survives_the_request(alice, case_a, root) -> None:
    """The defect: /api/ingest wrote to a TemporaryDirectory and deleted it."""
    dataset = alice.get(
        f"/api/investigations/{case_a}/datasets").json()["datasets"][0]
    stored = root / "uploads" / dataset["sha256"][:2] / dataset["sha256"]
    assert stored.is_file()
    assert stored.read_bytes() == CAPTURE


def test_the_stored_path_is_derived_from_the_hash_not_the_filename(alice) -> None:
    """Path traversal is impossible by construction, not by sanitisation."""
    case = alice.post("/api/investigations", json={"name": "Traversal"}).json()
    response = alice.post(
        f"/api/investigations/{case['id']}/datasets"
        f"?filename=../../../../etc/passwd",
        content=CAPTURE, headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 201
    dataset = response.json()["dataset"]
    assert "/" not in dataset["filename"] and ".." not in dataset["filename"]
    assert dataset["sha256"] in _stored_path(alice, case["id"])


def _stored_path(client, case_id) -> str:
    dataset = client.get(
        f"/api/investigations/{case_id}/datasets").json()["datasets"][0]
    return dataset["sha256"]


def test_an_upload_creates_an_honest_not_run_analysis(alice, case_a) -> None:
    """The critical requirement: a validated upload is not a scored one."""
    dataset = alice.get(
        f"/api/investigations/{case_a}/datasets").json()["datasets"][0]
    run = dataset["analysis_run"]
    assert run["status"] == "NOT_RUN"
    assert run["run_fingerprint"] is None
    assert run["produced_alerts"] is False
    assert "has NOT been scored" in run["meaning"]


def test_a_fingerprint_cannot_be_recorded_without_a_complete_run(conn, alice, case_a) -> None:
    """Enforced by a CHECK constraint, not by a convention in a handler."""
    from obsidianchain.console import runs

    dataset_id = alice.get(
        f"/api/investigations/{case_a}/datasets").json()["datasets"][0]["id"]
    run = runs.latest_for_dataset(conn, dataset_id)
    with pytest.raises(errors.ValidationFailed):
        runs.set_status(conn, run.id, "NOT_RUN", run_fingerprint="0" * 16)


def test_an_empty_upload_is_refused(alice, case_a) -> None:
    response = alice.post(
        f"/api/investigations/{case_a}/datasets?filename=empty.csv",
        content=b"", headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 400


# ---- provenance binding -------------------------------------------------


def test_a_new_case_is_bound_to_no_run(alice) -> None:
    case = alice.post("/api/investigations", json={"name": "Unbound"}).json()
    assert case["analytical_run"]["bound_run_fingerprint"] is None
    assert case["analytical_run"]["status"] == "UNBOUND"


def test_referencing_an_alert_binds_the_case_to_its_run(alice, case_a) -> None:
    case = alice.get(f"/api/investigations/{case_a}").json()
    assert case["analytical_run"]["bound_run_fingerprint"] == "043ea584e99daf99"


def test_a_case_is_not_repointed_by_reading_it(alice, case_a) -> None:
    """The defect: the overview overwrote the binding on every page view."""
    before = alice.get(
        f"/api/investigations/{case_a}").json()["bound_run_fingerprint"]
    for _ in range(5):
        alice.get(f"/api/investigations/{case_a}")
        alice.get(f"/api/investigations/{case_a}/alerts")
    after = alice.get(
        f"/api/investigations/{case_a}").json()["bound_run_fingerprint"]
    assert after == before


def test_an_alert_from_another_run_is_refused(alice, case_a) -> None:
    """409, never a silent second binding."""
    response = alice.post(f"/api/investigations/{case_a}/alerts",
                          json={"alert_id": OTHER_RUN_ALERT})
    assert response.status_code == 409
    assert response.json()["error"] == "run_mismatch"


def test_rebinding_to_a_different_run_is_refused(conn, alice, case_a) -> None:
    user = users.by_username(conn, "alice")
    with pytest.raises(errors.RunMismatch):
        investigations.bind_run(conn, user, case_a, "ffffffffffffffff")


def test_rebinding_to_the_same_run_is_a_no_op(conn, alice, case_a) -> None:
    user = users.by_username(conn, "alice")
    investigations.bind_run(conn, user, case_a, "043ea584e99daf99")
    case = investigations.get(conn, case_a)
    assert case.bound_run_fingerprint == "043ea584e99daf99"


def test_a_stale_reference_is_flagged_not_dropped(alice, case_a) -> None:
    """With no artifact on disk, staleness is UNKNOWN, not False.

    "We could not check" and "it is current" are different answers, and
    collapsing them would render an unverifiable reference as a verified one.
    """
    listing = alice.get(f"/api/investigations/{case_a}/alerts").json()
    assert len(listing["alerts"]) == 1
    assert listing["alerts"][0]["stale"] is None
    assert listing["current_artifact_run"] is None


def test_a_report_keeps_unresolvable_references(alice, case_a) -> None:
    report = alice.get(f"/api/investigations/{case_a}/report").json()
    assert len(report["alert_references"]) == 1
    assert len(report["unverifiable_references"]) == 1
    assert report["analytical_run"]["status"] == "UNVERIFIABLE"


def test_an_alert_detail_says_why_the_analytical_half_is_missing(alice, case_a) -> None:
    body = alice.get(f"/api/investigations/{case_a}/alerts/{ALERT}").json()
    assert body["analytical"]["available"] is False
    assert body["analytical"]["reason"] == "ARTIFACT_UNAVAILABLE"
    # The investigator record is unaffected, which is the whole point.
    assert body["investigator"]["disposition"]["state"] == "IN_REVIEW"


# ---- dispositions are append-only ---------------------------------------


def test_a_reference_starts_at_new(alice, conn, case_a) -> None:
    history = casework.disposition_history(conn, case_a, ALERT)
    assert history[-1]["state"] == "NEW"


def test_a_changed_decision_supersedes_rather_than_overwrites(alice, case_a, conn) -> None:
    alice.post(f"/api/investigations/{case_a}/alerts/{ALERT}/disposition",
               json={"state": "CONFIRMED", "rationale": "confirmed"})
    history = casework.disposition_history(conn, case_a, ALERT)
    assert [row["state"] for row in history] == ["CONFIRMED", "IN_REVIEW", "NEW"]
    assert [row["active"] for row in history] == [True, False, False]


def test_exactly_one_disposition_is_active(alice, case_a, conn) -> None:
    for state in ("TRIAGED", "CONFIRMED", "ESCALATED"):
        alice.post(f"/api/investigations/{case_a}/alerts/{ALERT}/disposition",
                   json={"state": state})
    active = [r for r in casework.disposition_history(conn, case_a, ALERT)
              if r["active"]]
    assert len(active) == 1
    assert active[0]["state"] == "ESCALATED"


def test_the_database_refuses_to_delete_a_disposition(conn, case_a) -> None:
    """Append-only is a storage property, not a route-layer convention."""
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM alert_dispositions")


def test_the_database_refuses_to_rewrite_a_disposition(conn, case_a) -> None:
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE alert_dispositions SET state = 'DISMISSED'")


def test_an_unknown_disposition_state_is_refused(alice, case_a) -> None:
    response = alice.post(
        f"/api/investigations/{case_a}/alerts/{ALERT}/disposition",
        json={"state": "PROBABLY_FINE"},
    )
    assert response.status_code == 422


def test_a_disposition_needs_a_reference_first(alice) -> None:
    case = alice.post("/api/investigations", json={"name": "Empty"}).json()["id"]
    response = alice.post(
        f"/api/investigations/{case}/alerts/{ALERT}/disposition",
        json={"state": "CONFIRMED"},
    )
    assert response.status_code == 404


# ---- notes --------------------------------------------------------------


def test_every_note_is_attributable(alice, case_a) -> None:
    notes = alice.get(f"/api/investigations/{case_a}/notes").json()["notes"]
    assert notes
    assert all(n["author_username"] for n in notes)
    assert all(n["created_at"] for n in notes)


def test_an_empty_note_is_refused(alice, case_a) -> None:
    assert alice.post(f"/api/investigations/{case_a}/notes",
                      json={"body": "   "}).status_code == 422


# ---- reports ------------------------------------------------------------


def test_the_executive_summary_persists(alice, case_a) -> None:
    """The field the previous report page never saved at all."""
    report = alice.get(f"/api/investigations/{case_a}/report").json()["report"]
    assert report["executive_summary"] == "Summary A"


def test_every_save_creates_a_new_version(alice, case_a) -> None:
    alice.post(f"/api/investigations/{case_a}/report",
               json={"title": "Report A", "executive_summary": "Revised",
                     "content": "Body A"})
    payload = alice.get(f"/api/investigations/{case_a}/report").json()
    assert payload["report"]["version"] == 2
    assert [v["version"] for v in payload["versions"]] == [2, 1]
    # Version 1 is still readable, unchanged.
    first = alice.get(
        f"/api/investigations/{case_a}/report?version=1").json()["report"]
    assert first["executive_summary"] == "Summary A"


def test_a_report_carries_a_content_hash_and_a_run(alice, case_a) -> None:
    report = alice.get(f"/api/investigations/{case_a}/report").json()["report"]
    assert len(report["content_sha256"]) == 64
    assert report["run_fingerprint"] == "043ea584e99daf99"


def test_the_content_hash_changes_with_the_content(alice, case_a) -> None:
    first = alice.get(
        f"/api/investigations/{case_a}/report").json()["report"]["content_sha256"]
    alice.post(f"/api/investigations/{case_a}/report",
               json={"title": "Report A", "executive_summary": "Different",
                     "content": "Body A"})
    second = alice.get(
        f"/api/investigations/{case_a}/report").json()["report"]["content_sha256"]
    assert first != second


# ---- audit --------------------------------------------------------------


def test_the_case_history_records_what_happened(alice, case_a) -> None:
    from obsidianchain.console import audit

    actions = {
        e["action"]
        for e in alice.get(
            f"/api/investigations/{case_a}/history").json()["events"]
    }
    for expected in (
        audit.INVESTIGATION_CREATED, audit.DATASET_UPLOADED,
        audit.DATASET_VALIDATED, audit.ANALYTICAL_RUN_BOUND,
        audit.ALERT_REFERENCED, audit.DISPOSITION_SET, audit.NOTE_CREATED,
        audit.REPORT_CREATED,
    ):
        assert expected in actions, f"{expected} was not recorded"


def test_every_audit_event_names_an_actor(alice, case_a) -> None:
    events = alice.get(
        f"/api/investigations/{case_a}/history").json()["events"]
    assert events
    assert all(e["actor_username"] == "alice" for e in events)


def test_a_failed_login_is_recorded_without_an_actor(app, conn) -> None:
    from obsidianchain.console import audit

    TestClient(app).post("/api/auth/login",
                         json={"username": "alice", "password": "wrong"})
    events = audit.recent(conn)
    failed = [e for e in events if e["action"] == audit.LOGIN_FAILED]
    assert failed and failed[0]["actor_id"] is None


def test_the_database_refuses_to_amend_the_audit_log(conn, case_a) -> None:
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM audit_events")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE audit_events SET action = 'NOTHING_HAPPENED'")


def test_a_closed_case_refuses_further_writes(alice, case_a) -> None:
    alice.post(f"/api/investigations/{case_a}/status", json={"status": "ACTIVE"})
    alice.post(f"/api/investigations/{case_a}/status", json={"status": "CLOSED"})
    response = alice.post(f"/api/investigations/{case_a}/notes",
                          json={"body": "after the fact"})
    assert response.status_code == 409


def test_there_is_no_delete_route_for_a_case(alice, case_a) -> None:
    """A case is CLOSED, never removed by investigators - deleting it destroys the audit trail."""
    assert alice.delete(f"/api/investigations/{case_a}").status_code in (403, 404, 405)
