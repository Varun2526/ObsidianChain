"""Authenticated test clients for the analytical routes.

Why the existing analytical tests need this
-------------------------------------------
``/api/alerts``, ``/api/alerts/{id}``, ``/api/evidence/{id}``,
``/api/ingest`` and ``/api/demo/scenarios`` were public. They are not any
more: the console's login could be bypassed entirely by querying the
artifacts directly, so every route under ``/api`` now requires a session
except the three in ``app.PUBLIC_ROUTES``.

That changes how a test REACHES those routes and nothing about what they
return. Every assertion in the suites that use this helper is unchanged -
the responses, the error codes, the provenance refusals and the frozen
wordings are all exactly as before. The session is a precondition, not a
modification.

``tests/test_api_access.py`` is what proves the boundary itself. This module
deliberately does not assert anything; it only gets a caller through the
door so the contract behind it can still be tested.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

#: The account these helpers create. INVESTIGATOR rather than ADMIN on
#: purpose: the analytical surface must be reachable by an ordinary user, and
#: a test that signed in as an administrator would not show that.
USERNAME = "test-analyst"
PASSWORD = "test-analyst-password"
ROLE = "INVESTIGATOR"


def ensure_account(
    data_root, *, username: str = USERNAME, password: str = PASSWORD,
    role: str = ROLE,
) -> None:
    """Create the account in ``data_root``'s console database if absent.

    Idempotent, so a module-scoped root and a function-scoped one both work.
    """
    from obsidianchain.console import db, errors, users

    conn = db.connect(data_root)
    try:
        if users.by_username(conn, username) is None:
            try:
                users.create(conn, username=username, password=password,
                             role=role)
            except errors.Conflict:
                pass  # created concurrently by another fixture
    finally:
        conn.close()


def sign_in(app, data_root, *, username: str = USERNAME,
            password: str = PASSWORD, role: str = ROLE) -> TestClient:
    """A TestClient holding a real session cookie for ``app``.

    The session is established the same way the browser establishes one -
    ``POST /api/auth/login`` against the same scrypt hash and the same
    server-side session table. There is no test-only credential and no
    bypass: if authentication broke, this would fail too.
    """
    ensure_account(data_root, username=username, password=password, role=role)
    client = TestClient(app)
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, (
        f"the test helper could not sign in: {response.status_code} "
        f"{response.text}"
    )
    return client


def authed_client(app, data_root) -> TestClient:
    """Shorthand used by the analytical suites' ``client`` fixtures."""
    return sign_in(app, data_root)


def signed_client(data_root, **kwargs) -> TestClient:
    """Build the app for ``data_root`` and return a signed-in client.

    Drop-in for ``TestClient(create_app(root), **kwargs)`` at the call sites
    that existed before the analytical routes required a session. ``kwargs``
    (``raise_server_exceptions=False`` in particular) pass through unchanged,
    because several suites depend on seeing a 500 as a response rather than
    as a raised exception.
    """
    from obsidianchain.api.app import create_app

    ensure_account(data_root)
    app = create_app(data_root)
    client = TestClient(app, **kwargs)
    response = client.post(
        "/api/auth/login", json={"username": USERNAME, "password": PASSWORD}
    )
    assert response.status_code == 200, (
        f"the test helper could not sign in: {response.status_code} "
        f"{response.text}"
    )
    return client


#: A self-contained snippet for the suites that drive the API from a
#: SUBPROCESS. Those probes cannot import this module's fixtures, and they
#: need the session established inside their own interpreter.
SUBPROCESS_SIGN_IN = """
from tests.console_helpers import USERNAME, PASSWORD, ensure_account
ensure_account(_ROOT)
_client.post("/api/auth/login",
             json={"username": USERNAME, "password": PASSWORD})
"""
