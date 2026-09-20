"""No route answers an unauthenticated caller except the documented three.

Why this is behavioural rather than structural
----------------------------------------------
An earlier version of this check could have walked each route's declared
dependencies and looked for the session one. That would pass for a route
whose dependency was present but ineffective, and it would have to be taught
about every way FastAPI can attach one. Sending a real unauthenticated
request and reading the status cannot be fooled either way.

The rule this pins
------------------
Every route under ``/api`` refuses an unauthenticated caller, EXCEPT the
entries in ``app.PUBLIC_ROUTES``, each of which carries a written reason. A
new endpoint is protected by default: forgetting the dependency makes this
file fail rather than opening a door.

The gap it closes
-----------------
``/api/alerts``, ``/api/alerts/{id}``, ``/api/alerts/{id}/separation-evidence``
and ``/api/evidence/{id}`` were public. The console's login, its role checks
and its case ownership could all be bypassed by one GET: the ranked alerts,
the calibrated risk scores, the SHAP explanations and the separation evidence
were readable by anyone who could reach the port.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from obsidianchain.api import app as app_module
from obsidianchain.api.app import PUBLIC_ROUTES, create_app
from obsidianchain.console import db, sessions, users

#: The analytical routes this remediation closed, named individually so a
#: regression on any one of them fails with its own message.
ANALYTICAL_ROUTES = (
    "/api/alerts",
    "/api/alerts/043ea584e99daf99:1",
    "/api/alerts/043ea584e99daf99:1/separation-evidence",
    "/api/evidence/0123456789abcdef:1",
)

#: Valid-SHAPED stand-ins for path parameters. They need not exist: the
#: session dependency runs before the handler, so a protected route must 401
#: rather than 404. Using well-formed values keeps a 422 from masking that.
PLACEHOLDERS = {
    "{alert_id}": "043ea584e99daf99:1",
    "{evidence_id}": "0123456789abcdef:1",
    "{investigation_id}": "inv_0000000000000000",
    "{dataset_id}": "ds_0000000000000000",
    "{user_id}": "usr_0000000000000000",
    "{version}": "1",
    "{txid}": "1050841",
    "{filter_id}": "flt_0000000000000000",
    "{run_id}": "run_0000000000000000",
}


@pytest.fixture
def root(tmp_path):
    return tmp_path


@pytest.fixture
def app(root):
    connection = db.connect(root)
    try:
        users.create(connection, username="analyst",
                     password="analyst-password", role="INVESTIGATOR")
    finally:
        connection.close()
    return create_app(root)


@pytest.fixture
def anonymous(app):
    return TestClient(app)


@pytest.fixture
def signed_in(app):
    client = TestClient(app)
    response = client.post("/api/auth/login",
                           json={"username": "analyst",
                                 "password": "analyst-password"})
    assert response.status_code == 200, response.text
    return client


def all_api_routes(app) -> list[tuple[str, str]]:
    """Every ``(method, path)`` under /api, including mounted routers.

    Walked from the application's own table rather than from a hand-kept
    list, so a route added anywhere is covered the moment it exists.
    """
    seen: set[tuple[str, str]] = set()

    def walk(routes):
        for route in routes:
            path = getattr(route, "path", None)
            if path and path.startswith("/api"):
                for method in getattr(route, "methods", None) or ():
                    if method not in ("HEAD", "OPTIONS"):
                        seen.add((method, path))
            inner = getattr(route, "original_router", None)
            if inner is not None:
                walk(getattr(inner, "routes", ()))
            elif hasattr(route, "routes"):
                walk(route.routes)

    walk(app.routes)
    return sorted(seen)


def concrete(path: str) -> str:
    for placeholder, value in PLACEHOLDERS.items():
        path = path.replace(placeholder, value)
    return path


def send(client: TestClient, method: str, path: str):
    """One request, with a body where the handler declares one.

    An empty body on a route expecting JSON would return 422 before the
    session dependency was reached, which would look like protection without
    being it.
    """
    url = concrete(path)
    if method in ("POST", "PATCH", "PUT"):
        return client.request(method, url, json={})
    return client.request(method, url)


def test_the_route_table_is_not_empty(app) -> None:
    """Guard against a vacuous suite if routing ever changes shape."""
    routes = all_api_routes(app)
    assert len(routes) >= 25, routes
    # And the specific routes this file exists for are present.
    paths = {path for _method, path in routes}
    assert "/api/alerts" in paths
    assert "/api/evidence/{evidence_id}" in paths


def test_every_public_route_is_a_real_route(app) -> None:
    """A stale allowlist entry would excuse a route that no longer exists."""
    actual = set(all_api_routes(app))
    for entry in PUBLIC_ROUTES:
        assert entry in actual, f"{entry} is allowlisted but not routed"


def test_every_public_route_carries_a_written_reason() -> None:
    for entry, reason in PUBLIC_ROUTES.items():
        assert isinstance(reason, str) and len(reason) > 40, (
            f"{entry} is public with no explanation"
        )


def test_the_allowlist_is_small_and_contains_no_analytical_route() -> None:
    """The point of the remediation, asserted directly."""
    assert len(PUBLIC_ROUTES) == 3
    for _method, path in PUBLIC_ROUTES:
        assert "alert" not in path
        assert "evidence" not in path
        assert "investigation" not in path


def _protected_routes(app):
    return [r for r in all_api_routes(app) if r not in PUBLIC_ROUTES]


def test_there_are_protected_routes_to_check(app) -> None:
    assert len(_protected_routes(app)) >= 22


@pytest.mark.parametrize(
    "method,path",
    [
        pytest.param(m, p, id=f"{m} {p}")
        for m, p in all_api_routes(create_app(None))
        if (m, p) not in PUBLIC_ROUTES
    ],
)
def test_no_protected_route_answers_an_anonymous_caller(
    anonymous, method, path
) -> None:
    response = send(anonymous, method, path)
    assert response.status_code == 401, (
        f"{method} {path} answered {response.status_code} without a session. "
        f"Every route under /api requires one unless it is listed in "
        f"app.PUBLIC_ROUTES with a reason."
    )
    assert response.json()["error"] in (
        "authentication_required", "session_expired",
    )


# ---- the four analytical routes, individually --------------------------


@pytest.mark.parametrize("path", ANALYTICAL_ROUTES)
def test_an_analytical_route_refuses_without_a_session(anonymous, path) -> None:
    response = anonymous.get(path)
    assert response.status_code == 401
    assert response.json()["error"] == "authentication_required"


@pytest.mark.parametrize("path", ANALYTICAL_ROUTES)
def test_an_analytical_route_is_reached_once_authenticated(signed_in, path) -> None:
    """Past the gate the ORIGINAL contract applies, unchanged.

    No artifacts exist in this temporary root, so the honest answer is 503
    'artifact_not_generated' - the same answer the route gave before it was
    protected. What must not happen is a 401, which would mean the session
    was not accepted.
    """
    response = signed_in.get(path)
    assert response.status_code != 401
    assert response.status_code == 503
    assert response.json()["error"] == "artifact_not_generated"


def test_health_is_public_and_discloses_nothing(anonymous) -> None:
    response = anonymous.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "version", "artifacts_present"}
    # A liveness probe that leaked the dataset's size would be a disclosure
    # channel wearing a health check's name.
    assert "run_fingerprint" not in body
    assert "alert_count_total" not in body


# ---- the session lifecycle, on the analytical surface -------------------


def test_logging_out_closes_the_analytical_surface_again(signed_in) -> None:
    assert signed_in.get("/api/alerts").status_code != 401
    signed_in.post("/api/auth/logout")
    response = signed_in.get("/api/alerts")
    assert response.status_code == 401


def test_a_revoked_session_cannot_read_alerts(root, app, signed_in) -> None:
    connection = db.connect(root)
    try:
        user = users.by_username(connection, "analyst")
        sessions.revoke_all_for_user(connection, user.id)
    finally:
        connection.close()
    assert signed_in.get("/api/alerts").status_code == 401


def test_an_expired_session_cannot_read_alerts(root, app) -> None:
    connection = db.connect(root)
    try:
        user = users.by_username(connection, "analyst")
        token = sessions.create(connection, user, ttl_seconds=-1)
    finally:
        connection.close()

    client = TestClient(app)
    client.cookies.set(sessions.COOKIE_NAME, token)
    response = client.get("/api/alerts")
    assert response.status_code == 401
    assert response.json()["error"] == "session_expired"


def test_a_forged_cookie_cannot_read_alerts(app) -> None:
    client = TestClient(app)
    client.cookies.set(sessions.COOKIE_NAME, "a" * 43)
    assert client.get("/api/alerts").status_code == 401


def test_a_deactivated_account_cannot_read_alerts(root, app, signed_in) -> None:
    connection = db.connect(root)
    try:
        user = users.by_username(connection, "analyst")
        users.set_active(connection, user.id, False)
    finally:
        connection.close()
    assert signed_in.get("/api/alerts").status_code == 401


# ---- authentication is not authorisation -------------------------------


def test_authenticating_grants_no_case_access(root, app, signed_in) -> None:
    """The distinction the remediation must not blur.

    Being allowed to read the ANALYTICAL artifacts says nothing about whose
    case you may open. A second investigator's case stays closed.
    """
    connection = db.connect(root)
    try:
        users.create(connection, username="other", password="other-password",
                     role="INVESTIGATOR")
    finally:
        connection.close()

    other = TestClient(app)
    other.post("/api/auth/login",
               json={"username": "other", "password": "other-password"})
    case = other.post("/api/investigations", json={"name": "Not yours"}).json()

    # analyst is authenticated, and that is all.
    assert signed_in.get("/api/health").status_code == 200
    assert signed_in.get(f"/api/investigations/{case['id']}").status_code == 403


def test_the_public_surface_is_documented_in_the_module() -> None:
    """The written record §7 asks for, kept where it cannot drift."""
    source = app_module.__file__
    assert source
    assert "PUBLIC_ROUTES" in open(source, encoding="utf-8").read()
