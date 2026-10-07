"""One-click demo sign-in (console/demo_access.py): off by default, ordinary sessions when on."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from obsidianchain.api.app import create_app
from obsidianchain.console import db
from obsidianchain.console.demo_reset import reset_clean_demo_database


@pytest.fixture
def root(tmp_path):
    reset_clean_demo_database(tmp_path, confirm=True)
    return tmp_path


def test_off_by_default(root, monkeypatch):
    monkeypatch.delenv("OBSIDIANCHAIN_DEMO_LOGIN", raising=False)
    c = TestClient(create_app(root))
    assert c.get("/api/auth/demo").json() == {"enabled": False, "roles": []}
    assert c.post("/api/auth/demo-login", json={"role": "ADMIN"}).status_code == 404
    assert c.get("/api/auth/me").status_code == 401


@pytest.mark.parametrize("role", ["INVESTIGATOR", "REVIEWER", "ADMIN"])
def test_each_role_signs_in_and_is_audited(root, monkeypatch, role):
    monkeypatch.setenv("OBSIDIANCHAIN_DEMO_LOGIN", "1")
    c = TestClient(create_app(root))
    status = c.get("/api/auth/demo").json()
    assert status["enabled"] and [r["role"] for r in status["roles"]] == ["INVESTIGATOR", "REVIEWER", "ADMIN"]
    r = c.post("/api/auth/demo-login", json={"role": role})
    assert r.status_code == 200, r.text
    assert c.get("/api/auth/me").json()["user"]["role"] == role
    conn = db.connect(root)
    row = conn.execute("SELECT detail_json FROM audit_events WHERE action = 'LOGIN' ORDER BY id DESC LIMIT 1").fetchone()
    assert "DEMO_LOGIN" in row["detail_json"]


def test_rbac_is_unchanged_for_demo_sessions(root, monkeypatch):
    monkeypatch.setenv("OBSIDIANCHAIN_DEMO_LOGIN", "1")
    c = TestClient(create_app(root))
    c.post("/api/auth/demo-login", json={"role": "REVIEWER"})
    assert c.post("/api/investigations", json={"name": "x"}).status_code == 403
    assert c.get("/api/users").status_code == 403


def test_missing_account_is_created_and_deactivated_account_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSIDIANCHAIN_DEMO_LOGIN", "1")
    db.connect(tmp_path, migrate_on_open=True).close()          # empty database, no users
    c = TestClient(create_app(tmp_path))
    assert c.post("/api/auth/demo-login", json={"role": "INVESTIGATOR"}).status_code == 200
    conn = db.connect(tmp_path)
    conn.execute("UPDATE users SET active = 0 WHERE username = 'investigator'")
    conn.commit()
    assert TestClient(create_app(tmp_path)).post(
        "/api/auth/demo-login", json={"role": "INVESTIGATOR"}).status_code == 401


def test_unknown_role_is_a_validation_error(root, monkeypatch):
    monkeypatch.setenv("OBSIDIANCHAIN_DEMO_LOGIN", "1")
    assert TestClient(create_app(root)).post("/api/auth/demo-login", json={"role": "ROOT"}).status_code == 422
