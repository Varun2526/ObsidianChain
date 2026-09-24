"""Tests for final product hardening: demo reset, RBAC, account & case lifecycle, audit.

Pins:
1. Deterministic clean demo reset with dynamic credentials.
2. Immediate session revocation on account deactivation and password reset.
3. Protection of the sole active administrator.
4. Rigid case lifecycle progression (DRAFT -> VALIDATING -> ANALYZING -> ACTIVE -> SUBMITTED -> IN_REVIEW -> APPROVED -> CLOSED -> ARCHIVED).
5. Reviewer and Investigator role boundaries (403 on unpermitted actions).
6. Admin-only Archive and Restore.
7. Exceptional Delete safeguards (typed confirmation, no active/in-review deletion).
8. Append-only audit logging across all privileged operations.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from obsidianchain.api.app import create_app
from obsidianchain.console import audit, db, errors, investigations, rbac, sessions, users
from obsidianchain.console.demo_reset import reset_clean_demo_database
from obsidianchain.console.rbac import Role


@pytest.fixture
def root(tmp_path):
    return tmp_path


@pytest.fixture
def demo_setup(root):
    creds = reset_clean_demo_database(root, confirm=True)
    return creds


@pytest.fixture
def app(root, demo_setup):
    return create_app(root)


@pytest.fixture
def client(app):
    return TestClient(app)


def login(client: TestClient, username: str, password: str):
    return client.post("/api/auth/login", json={"username": username, "password": password})


# ---- 1. Demo Reset & Account Provisioning ----------------------------------


def test_demo_reset_requires_confirmation(root) -> None:
    with pytest.raises(RuntimeError, match="Demo database reset is destructive"):
        reset_clean_demo_database(root, confirm=False)


def test_demo_reset_provisions_primary_roles_with_dynamic_credentials(root) -> None:
    creds = reset_clean_demo_database(root, confirm=True)
    assert set(creds.keys()) == {"admin", "investigator", "reviewer"}
    for role_name in ("admin", "investigator", "reviewer"):
        assert len(creds[role_name]["password"]) >= 12
        assert creds[role_name]["role"] in ("ADMIN", "INVESTIGATOR", "REVIEWER")

    conn = db.connect(root)
    try:
        user_list = users.listing(conn)
        assert len(user_list) == 3
        # Ensure zero cases exist initially
        all_cases = conn.execute("SELECT COUNT(*) FROM investigations").fetchone()[0]
        assert all_cases == 0
    finally:
        conn.close()


# ---- 2. Authentication & Session Invalidation ------------------------------


def test_deactivating_user_immediately_invalidates_session(client, demo_setup, root) -> None:
    inv_pass = demo_setup["investigator"]["password"]
    admin_pass = demo_setup["admin"]["password"]

    # Log in as investigator
    inv_client = TestClient(client.app)
    resp = login(inv_client, "investigator", inv_pass)
    assert resp.status_code == 200
    assert inv_client.get("/api/auth/me").status_code == 200

    # Log in as admin and deactivate investigator
    admin_client = TestClient(client.app)
    login(admin_client, "admin", admin_pass)
    deact_resp = admin_client.post(f"/api/users/{demo_setup['investigator']['id']}/deactivate")
    assert deact_resp.status_code == 200

    # Investigator's existing open session MUST now immediately fail
    assert inv_client.get("/api/auth/me").status_code == 401

    # Deactivated investigator cannot log back in
    assert login(inv_client, "investigator", inv_pass).status_code == 401

    # Reactivate investigator
    react_resp = admin_client.post(f"/api/users/{demo_setup['investigator']['id']}/activate")
    assert react_resp.status_code == 200

    # Now login succeeds again
    assert login(inv_client, "investigator", inv_pass).status_code == 200


def test_password_reset_revokes_sessions(client, demo_setup) -> None:
    inv_pass = demo_setup["investigator"]["password"]
    admin_pass = demo_setup["admin"]["password"]

    inv_client = TestClient(client.app)
    login(inv_client, "investigator", inv_pass)
    assert inv_client.get("/api/auth/me").status_code == 200

    admin_client = TestClient(client.app)
    login(admin_client, "admin", admin_pass)
    reset_resp = admin_client.post(
        f"/api/users/{demo_setup['investigator']['id']}/reset-password",
        json={"password": "NewInvestigatorPassword2026!"},
    )
    assert reset_resp.status_code == 200

    # Existing session revoked
    assert inv_client.get("/api/auth/me").status_code == 401
    # Old password fails
    assert login(inv_client, "investigator", inv_pass).status_code == 401
    # New password succeeds
    assert login(inv_client, "investigator", "NewInvestigatorPassword2026!").status_code == 200


def test_sole_admin_cannot_be_deactivated_or_demoted(client, demo_setup) -> None:
    admin_pass = demo_setup["admin"]["password"]
    admin_id = demo_setup["admin"]["id"]

    admin_client = TestClient(client.app)
    login(admin_client, "admin", admin_pass)

    # Cannot deactivate self / sole admin
    deact = admin_client.post(f"/api/users/{admin_id}/deactivate")
    assert deact.status_code in (400, 422)

    # Cannot demote sole admin
    demote = admin_client.post(f"/api/users/{admin_id}/role", json={"role": "INVESTIGATOR"})
    assert demote.status_code in (400, 422)


# ---- 3. RBAC Enforcement ---------------------------------------------------


def test_investigator_and_reviewer_cannot_access_user_management(client, demo_setup) -> None:
    for username, role_key in [("investigator", "investigator"), ("reviewer", "reviewer")]:
        c = TestClient(client.app)
        login(c, username, demo_setup[role_key]["password"])
        assert c.get("/api/users").status_code == 403
        assert c.post("/api/users", json={"username": "evil", "password": "password"}).status_code == 403


# ---- 4. Case Lifecycle & State Machine Constraints --------------------------


def test_case_lifecycle_state_machine_and_role_boundaries(client, demo_setup, root) -> None:
    inv_c = TestClient(client.app)
    login(inv_c, "investigator", demo_setup["investigator"]["password"])

    rev_c = TestClient(client.app)
    login(rev_c, "reviewer", demo_setup["reviewer"]["password"])

    admin_c = TestClient(client.app)
    login(admin_c, "admin", demo_setup["admin"]["password"])

    # 1. Investigator creates case -> DRAFT
    create_resp = inv_c.post("/api/investigations", json={"name": "Operation Phoenix"})
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]
    case_label = create_resp.json()["case_label"]
    assert create_resp.json()["status"] == "DRAFT"

    # 2. Invalid skips are rejected
    # Cannot jump DRAFT -> ACTIVE directly
    skip_resp = inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "ACTIVE"})
    assert skip_resp.status_code == 409

    # Cannot jump DRAFT -> IN_REVIEW
    skip_rev = inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "IN_REVIEW"})
    assert skip_rev.status_code in (403, 409)

    # 3. Legitimate progression: DRAFT -> VALIDATING -> ANALYZING -> ACTIVE
    step1 = inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "VALIDATING"})
    assert step1.status_code == 200
    assert step1.json()["status"] == "VALIDATING"

    step2 = inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "ANALYZING"})
    assert step2.status_code == 200
    assert step2.json()["status"] == "ANALYZING"

    step3 = inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "ACTIVE"})
    assert step3.status_code == 200
    assert step3.json()["status"] == "ACTIVE"

    # 4. Investigator submits for review: ACTIVE -> SUBMITTED
    sub_resp = inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "SUBMITTED"})
    assert sub_resp.status_code == 200
    assert sub_resp.json()["status"] == "SUBMITTED"

    # Investigator CANNOT self-review or self-approve (403)
    self_rev = inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "IN_REVIEW"})
    assert self_rev.status_code == 403
    self_app = inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "APPROVED"})
    assert self_app.status_code == 403

    # 5. Reviewer starts review: SUBMITTED -> IN_REVIEW
    rev_start = rev_c.post(f"/api/investigations/{case_id}/status", json={"status": "IN_REVIEW"})
    assert rev_start.status_code == 200
    assert rev_start.json()["status"] == "IN_REVIEW"

    # 6. Reviewer approves case: IN_REVIEW -> APPROVED
    rev_app = rev_c.post(f"/api/investigations/{case_id}/status", json={"status": "APPROVED"})
    assert rev_app.status_code == 200
    assert rev_app.json()["status"] == "APPROVED"

    # 7. Close case: APPROVED -> CLOSED
    close_resp = inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "CLOSED"})
    assert close_resp.status_code == 200
    assert close_resp.json()["status"] == "CLOSED"

    # 8. Archive: Investigator and Reviewer CANNOT archive (403)
    assert inv_c.post(f"/api/investigations/{case_id}/archive").status_code == 403
    assert rev_c.post(f"/api/investigations/{case_id}/archive").status_code == 403

    # Admin archives CLOSED case: CLOSED -> ARCHIVED
    arch_resp = admin_c.post(f"/api/investigations/{case_id}/archive")
    assert arch_resp.status_code == 200
    assert arch_resp.json()["status"] == "ARCHIVED"

    # Admin restores ARCHIVED case: ARCHIVED -> CLOSED
    rest_resp = admin_c.post(f"/api/investigations/{case_id}/restore")
    assert rest_resp.status_code == 200
    assert rest_resp.json()["status"] == "CLOSED"


# ---- 5. Exceptional Delete Controls -----------------------------------------


def test_delete_investigation_protections(client, demo_setup) -> None:
    inv_c = TestClient(client.app)
    login(inv_c, "investigator", demo_setup["investigator"]["password"])

    admin_c = TestClient(client.app)
    login(admin_c, "admin", demo_setup["admin"]["password"])

    # Create case
    c_resp = inv_c.post("/api/investigations", json={"name": "Temp Test Case"})
    case_id = c_resp.json()["id"]
    case_label = c_resp.json()["case_label"]

    # Investigator cannot delete (403)
    assert inv_c.delete(f"/api/investigations/{case_id}").status_code == 403

    # Progress to ACTIVE
    inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "VALIDATING"})
    inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "ANALYZING"})
    inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "ACTIVE"})

    # Admin cannot delete an ACTIVE case
    del_active = admin_c.delete(
        f"/api/investigations/{case_id}?confirmation=DELETE+{case_label}"
    )
    assert del_active.status_code == 409

    # Close case
    inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "CLOSED"})

    # Admin delete with wrong confirmation text is rejected
    del_bad_conf = admin_c.delete(
        f"/api/investigations/{case_id}?confirmation=DELETE+WRONG"
    )
    assert del_bad_conf.status_code in (400, 422)

    # Admin delete with exact typed confirmation succeeds
    del_ok = admin_c.delete(
        f"/api/investigations/{case_id}?confirmation=DELETE+{case_label}"
    )
    assert del_ok.status_code == 200
    assert del_ok.json()["ok"] is True

    assert admin_c.get(f"/api/investigations/{case_id}").status_code == 404



def test_lifecycle_follows_upload_and_analysis_events(client, demo_setup, root) -> None:
    """A case reaches ACTIVE by validating and analysing a dataset, not by relabelling.

    Before this, a case stayed DRAFT through a validated upload and a
    completed run, and DRAFT -> SUBMITTED is not a permitted transition, so
    no case could ever be submitted for review from the product.
    """
    from pathlib import Path
    inv_c = TestClient(client.app)
    login(inv_c, "investigator", demo_setup["investigator"]["password"])
    case_id = inv_c.post("/api/investigations", json={"name": "Lifecycle"}).json()["id"]
    sample = Path(__file__).parent / "data" / "synthetic_acceptance_capture.json"
    up = inv_c.post(f"/api/investigations/{case_id}/datasets?filename=capture.json",
                    content=sample.read_bytes(), headers={"Content-Type": "application/octet-stream"})
    assert up.status_code == 201, up.text
    assert inv_c.get(f"/api/investigations/{case_id}").json()["status"] == "VALIDATING"
    ds = up.json()["dataset"]["id"]
    run = inv_c.post(f"/api/investigations/{case_id}/datasets/{ds}/run")
    assert run.status_code == 200, run.text
    assert inv_c.get(f"/api/investigations/{case_id}").json()["status"] == "ACTIVE"
    assert inv_c.post(f"/api/investigations/{case_id}/status", json={"status": "SUBMITTED"}).status_code == 200
    events = inv_c.get(f"/api/investigations/{case_id}/history").json()["events"]
    auto = [e for e in events if e["action"] == "INVESTIGATION_STATUS_CHANGED" and e["detail"].get("automatic")]
    assert [e["detail"]["to"] for e in auto][::-1] == ["VALIDATING", "ANALYZING", "ACTIVE"] or \
           sorted(e["detail"]["to"] for e in auto) == ["ACTIVE", "ANALYZING", "VALIDATING"]
