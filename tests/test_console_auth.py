"""Authentication, sessions and RBAC.

These are the properties the previous frontend-only "auth" did not have: a
password that is actually checked, a session the server owns, a logout that
revokes it, and a role the browser cannot assert for itself.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from obsidianchain.api.app import create_app
from obsidianchain.console import db, errors, passwords, rbac, sessions, users
from obsidianchain.console.rbac import Capability, Role


@pytest.fixture
def root(tmp_path):
    return tmp_path


@pytest.fixture
def conn(root):
    connection = db.connect(root)
    yield connection
    connection.close()


@pytest.fixture
def client(root, conn):
    users.create(conn, username="alice", password="alice-password",
                 role="INVESTIGATOR", display_name="Alice A")
    users.create(conn, username="root", password="root-password", role="ADMIN")
    users.create(conn, username="rev", password="rev-password", role="REVIEWER")
    return TestClient(create_app(root))


def login(client, username, password):
    return client.post("/api/auth/login",
                       json={"username": username, "password": password})


# ---- password hashing ---------------------------------------------------


def test_a_password_round_trips() -> None:
    encoded = passwords.hash_password("correct horse battery")
    assert passwords.verify("correct horse battery", encoded)
    assert not passwords.verify("wrong horse battery", encoded)


def test_the_hash_is_not_the_password() -> None:
    encoded = passwords.hash_password("a-real-password")
    assert "a-real-password" not in encoded
    assert encoded.startswith("scrypt$")


def test_two_hashes_of_one_password_differ() -> None:
    """Salted. Identical passwords must not produce identical rows."""
    assert (
        passwords.hash_password("same-password-twice")
        != passwords.hash_password("same-password-twice")
    )


def test_a_corrupt_hash_fails_closed() -> None:
    """A malformed stored value must fail the login, not raise."""
    for broken in ("", "not-a-hash", "scrypt$x$8$1$aa$bb", "scrypt$16384$8$1$zz$yy"):
        assert passwords.verify("anything", broken) is False


def test_a_short_password_is_refused() -> None:
    with pytest.raises(passwords.PasswordError):
        passwords.hash_password("short")


# ---- login --------------------------------------------------------------


def test_login_succeeds_with_the_right_password(client) -> None:
    response = login(client, "alice", "alice-password")
    assert response.status_code == 200
    assert response.json()["user"]["username"] == "alice"
    assert sessions.COOKIE_NAME in response.cookies


def test_login_fails_with_the_wrong_password(client) -> None:
    response = login(client, "alice", "not-the-password")
    assert response.status_code == 401
    assert response.json()["error"] == "invalid_credentials"


def test_an_unknown_user_is_indistinguishable_from_a_wrong_password(client) -> None:
    """Otherwise the login form is a username oracle."""
    wrong = login(client, "alice", "nope").json()
    missing = login(client, "nobody-at-all", "nope").json()
    assert wrong == missing


def test_a_deactivated_account_cannot_log_in(client, conn) -> None:
    user = users.by_username(conn, "alice")
    users.set_active(conn, user.id, False)
    assert login(client, "alice", "alice-password").status_code == 401


def test_the_session_cookie_is_httponly(client) -> None:
    """The token must be unreachable from any script on the page."""
    response = login(client, "alice", "alice-password")
    header = response.headers.get("set-cookie", "")
    assert "httponly" in header.lower()
    assert "samesite=lax" in header.lower()


def test_the_raw_token_is_never_stored(client, conn) -> None:
    response = login(client, "alice", "alice-password")
    token = response.cookies[sessions.COOKIE_NAME]
    rows = conn.execute("SELECT token_sha256 FROM sessions").fetchall()
    assert rows
    assert all(row["token_sha256"] != token for row in rows)


# ---- session lifecycle --------------------------------------------------


def test_me_requires_a_session(client) -> None:
    response = client.get("/api/auth/me")
    assert response.status_code == 401
    assert response.json()["error"] == "authentication_required"


def test_me_returns_the_identity_after_login(client) -> None:
    login(client, "alice", "alice-password")
    body = client.get("/api/auth/me").json()
    assert body["user"]["role"] == "INVESTIGATOR"
    assert "create_investigation" in body["capabilities"]


def test_logout_revokes_the_session(client) -> None:
    login(client, "alice", "alice-password")
    assert client.get("/api/auth/me").status_code == 200
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401


def test_a_revoked_session_stays_revoked_even_with_the_cookie(client, conn) -> None:
    """The cookie is not the session. Revocation is immediate, not on expiry."""
    response = login(client, "alice", "alice-password")
    token = response.cookies[sessions.COOKIE_NAME]
    user = users.by_username(conn, "alice")
    sessions.revoke_all_for_user(conn, user.id)

    fresh = TestClient(client.app)
    fresh.cookies.set(sessions.COOKIE_NAME, token)
    assert fresh.get("/api/auth/me").status_code == 401


def test_an_expired_session_is_refused(root, conn) -> None:
    users.create(conn, username="temp", password="temp-password",
                 role="INVESTIGATOR")
    user = users.by_username(conn, "temp")
    token = sessions.create(conn, user, ttl_seconds=-1)
    with pytest.raises(errors.SessionExpired):
        sessions.resolve(conn, token)


def test_an_invented_token_authenticates_nothing(client) -> None:
    client.cookies.set(sessions.COOKIE_NAME, "a" * 43)
    assert client.get("/api/auth/me").status_code == 401


def test_deactivating_an_account_kills_its_open_session(client, conn) -> None:
    login(client, "alice", "alice-password")
    user = users.by_username(conn, "alice")
    users.set_active(conn, user.id, False)
    assert client.get("/api/auth/me").status_code == 401


# ---- RBAC ---------------------------------------------------------------


def test_the_three_roles_are_the_policy() -> None:
    assert set(rbac.CAPABILITIES) == {
        Role.ADMIN, Role.INVESTIGATOR, Role.REVIEWER
    }


def test_an_admin_holds_every_capability() -> None:
    assert rbac.CAPABILITIES[Role.ADMIN] == frozenset(Capability)


def test_a_reviewer_cannot_decide_or_create(client) -> None:
    """A reviewer who could rewrite decisions would not be reviewing them."""
    for capability in (
        Capability.SET_DISPOSITION,
        Capability.CREATE_INVESTIGATION,
        Capability.UPLOAD_DATASET,
        Capability.ASSIGN_ALERT,
    ):
        assert not rbac.has(Role.REVIEWER, capability)


def test_an_investigator_cannot_finalise_their_own_report() -> None:
    """Dual control: the author does not sign off the report."""
    assert not rbac.has(Role.INVESTIGATOR, Capability.FINALISE_REPORT)
    assert rbac.has(Role.REVIEWER, Capability.FINALISE_REPORT)


def test_an_investigator_cannot_manage_users() -> None:
    assert not rbac.has(Role.INVESTIGATOR, Capability.MANAGE_USERS)


def test_only_an_admin_may_list_accounts(client) -> None:
    login(client, "alice", "alice-password")
    assert client.get("/api/users").status_code == 403
    client.post("/api/auth/logout")
    login(client, "root", "root-password")
    assert client.get("/api/users").status_code == 200


def test_an_unknown_role_is_refused_rather_than_defaulted() -> None:
    """"Role we do not recognise" must never become "role with no rights"."""
    with pytest.raises(ValueError):
        rbac.parse_role("SUPERUSER")


def test_a_reviewer_may_read_but_an_investigator_only_their_own() -> None:
    assert rbac.may_read_case(Role.REVIEWER, "u1", "u2")
    assert rbac.may_read_case(Role.ADMIN, "u1", "u2")
    assert not rbac.may_read_case(Role.INVESTIGATOR, "u1", "u2")
    assert rbac.may_read_case(Role.INVESTIGATOR, "u1", "u1")


def test_the_policy_listing_needs_a_session(client) -> None:
    """Reading the policy grants nothing, but strangers still do not get it."""
    assert client.get("/api/roles").status_code == 401


def test_the_published_policy_matches_the_enforced_one(client) -> None:
    """The /api/roles listing is generated from CAPABILITIES, not restated."""
    login(client, "alice", "alice-password")
    served = client.get("/api/roles").json()["roles"]
    for role, caps in rbac.CAPABILITIES.items():
        assert set(served[role.value]) == {c.value for c in caps}
