"""GET /api/demo/scenarios - the Phase 5.1 endpoint.

The endpoint serves a file. These tests check that it serves *that* file
unchanged, refuses to serve anything a consumer could mistake for a
measurement, and computes nothing on the way.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from obsidianchain.api import artifacts, boundary, demo
from obsidianchain.api.app import create_app

# The analytical routes now require a session. Reaching them changed; what
# they return did not, and every assertion below is unchanged. See
# tests/console_helpers.py, and tests/test_api_access.py for the boundary
# itself.
from tests.console_helpers import signed_client


DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))
ROUTE = "/api/demo/scenarios"


def write_payload(root: Path, payload: dict) -> Path:
    path = root / artifacts.DEMO_SCENARIOS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def real_payload() -> dict:
    """The genuine artifact, so tests compare against what ``make demo`` wrote."""
    path = artifacts.demo_scenarios_path(DATA_ROOT)
    if not path.is_file():
        pytest.skip(f"{path} not generated; run 'make demo'")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def client(real_payload, tmp_path) -> TestClient:
    """A client over a COPY of the real payload in a temp root.

    Copied rather than pointed at ``data/`` so no test can write into the
    project's artifacts, and so the malformed-payload tests have somewhere
    safe to put a damaged file.
    """
    write_payload(tmp_path, real_payload)
    return signed_client(tmp_path)


# ---- 1. the endpoint responds ------------------------------------------


def test_returns_200(client) -> None:
    assert client.get(ROUTE).status_code == 200


def test_returns_the_persisted_payload_unchanged(client, real_payload) -> None:
    """Byte-for-byte the same object. No renaming, no reshaping, no additions."""
    assert client.get(ROUTE).json() == real_payload


def test_the_route_is_exactly_the_specified_path(client) -> None:
    # getattr rather than attribute access: since the console routers are
    # mounted, app.routes also holds FastAPI's own router-inclusion entries,
    # which carry no path. The assertion below is unchanged - the demo route
    # must be present under exactly this path, and a near-miss must 404.
    routes = {
        path for path in
        (getattr(r, "path", None) for r in create_app(None).routes)
        if path is not None
    }
    assert ROUTE in routes
    assert client.get("/api/demo/scenario").status_code == 404


# ---- 2. DEMO provenance markers ----------------------------------------


@pytest.mark.parametrize("field,expected", [
    ("demo", True),
    ("provenance", "SYNTHETIC_DEMONSTRATION"),
    ("provenance_type", "DEMO"),
    ("not_a_measurement", True),
])
def test_response_carries_each_demo_marker(client, field, expected) -> None:
    assert client.get(ROUTE).json()[field] == expected


def test_every_object_in_the_response_is_flagged(client) -> None:
    """Not just the envelope. A copied fragment must carry its own marking."""
    from obsidianchain.demo import api as demo_api

    demo_api.assert_demo_flagged(client.get(ROUTE).json())


def test_the_response_states_what_it_is_not(client) -> None:
    body = client.get(ROUTE).json()
    assert "NOT A MEASUREMENT" in body["banner"]
    assert body["statements"]["frozen_dataset"], (
        "the response must carry the statement that the frozen dataset does "
        "not trigger the mechanism"
    )


def test_provenance_type_distinguishes_demo_from_controlled_worlds(
    client
) -> None:
    """Phase 3.4 fixtures, not the Phase 3.3 worlds.

    The worlds are SYNTHETIC_CONTROL: a real experiment on generated data.
    Serving them under this route would relabel one as the other, which is
    what the three-type enum exists to prevent.
    """
    from obsidianchain import provenance as prov

    body = client.get(ROUTE).json()
    assert body["provenance_type"] == prov.ProvenanceType.DEMO.value
    assert body["provenance_type"] != prov.ProvenanceType.SYNTHETIC_CONTROL.value


# ---- 3 & 4. all five scenarios, with the persisted outcomes ------------


def test_all_five_scenarios_are_present(client) -> None:
    body = client.get(ROUTE).json()
    assert [s["key"] for s in body["scenarios"]] == ["A", "B", "C", "D", "E"]


def test_scenario_outcomes_match_the_persisted_artifact(
    client, real_payload
) -> None:
    served = {s["key"]: s["outcome"] for s in client.get(ROUTE).json()["scenarios"]}
    persisted = {s["key"]: s["outcome"] for s in real_payload["scenarios"]}
    assert served == persisted


def test_scenario_verdicts_match_the_persisted_artifact(
    client, real_payload
) -> None:
    served = {s["key"]: s["verdict"] for s in client.get(ROUTE).json()["scenarios"]}
    persisted = {s["key"]: s["verdict"] for s in real_payload["scenarios"]}
    assert served == persisted


def test_not_separated_is_not_flattened_into_something_else(client) -> None:
    """NOT_SEPARATED must survive transport as itself.

    It means "we looked and found nothing", and it is the most common
    decidable production outcome. Relabelling it as an affirmative state
    would convert an absence of evidence into a positive claim.
    """
    verdicts = {s["verdict"] for s in client.get(ROUTE).json()["scenarios"]}
    assert "NOT_SEPARATED" in verdicts
    assert "INFERRED" not in verdicts
    assert "CONFIRMED" not in verdicts


def test_decision_level_detail_survives(client) -> None:
    """The per-decision records are what answer "why was this flagged?"."""
    scenarios = {s["key"]: s for s in client.get(ROUTE).json()["scenarios"]}
    blocked = scenarios["C"]["decisions"][0]
    for field in ("verdict", "pooled_a", "pooled_b", "chi2", "p_value",
                  "effect", "reason", "merged"):
        assert field in blocked
    assert blocked["merged"] is False
    assert scenarios["D"]["contradiction"] is not None


# ---- 5. missing artifact fails explicitly ------------------------------


def test_missing_artifact_returns_503_naming_the_command(tmp_path) -> None:
    response = signed_client(tmp_path, raise_server_exceptions=False).get(ROUTE)
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "artifact_not_generated"
    assert "make demo" in body["detail"]


def test_the_loader_raises_rather_than_generating(tmp_path) -> None:
    with pytest.raises(artifacts.ArtifactMissingError, match="make demo"):
        artifacts.load_demo_scenarios(tmp_path)


# ---- 6. malformed or unflagged payloads are rejected -------------------


def test_unparseable_json_is_rejected(tmp_path) -> None:
    path = tmp_path / artifacts.DEMO_SCENARIOS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    response = signed_client(tmp_path, raise_server_exceptions=False).get(ROUTE)
    assert response.status_code == 500
    assert response.json()["error"] == "artifact_invalid"


def test_an_unflagged_payload_is_rejected(real_payload, tmp_path) -> None:
    """The flag is load-bearing, not decorative."""
    damaged = copy.deepcopy(real_payload)
    del damaged["demo"]
    write_payload(tmp_path, damaged)
    response = signed_client(tmp_path, raise_server_exceptions=False).get(ROUTE)
    assert response.status_code == 500
    assert response.json()["error"] in {"artifact_invalid",
                                        "demo_provenance_invalid"}


@pytest.mark.parametrize("field,value", [
    ("demo", False),
    ("provenance", "PRODUCTION"),
    ("provenance_type", "PRODUCTION"),
    ("not_a_measurement", False),
])
def test_a_weakened_marker_is_rejected(
    real_payload, tmp_path, field, value
) -> None:
    """Turning any marker to a production-looking value must fail the request.

    This is the attack the endpoint exists to make impossible: a payload that
    reads as a measurement because one field was flipped.
    """
    damaged = copy.deepcopy(real_payload)
    damaged[field] = value
    write_payload(tmp_path, damaged)
    response = signed_client(tmp_path, raise_server_exceptions=False).get(ROUTE)
    assert response.status_code == 500
    assert response.json()["error"] in {"artifact_invalid",
                                        "demo_provenance_invalid"}


def test_a_nested_object_losing_its_flag_is_rejected(
    real_payload, tmp_path
) -> None:
    """Depth is not an excuse - a decision record must be flagged too."""
    damaged = copy.deepcopy(real_payload)
    del damaged["scenarios"][2]["decisions"][0]["demo"]
    write_payload(tmp_path, damaged)
    response = signed_client(tmp_path, raise_server_exceptions=False).get(ROUTE)
    assert response.status_code == 500


def test_a_truncated_scenario_set_is_rejected(real_payload, tmp_path) -> None:
    """Three states served as five would misrepresent the engine."""
    damaged = copy.deepcopy(real_payload)
    damaged["scenarios"] = damaged["scenarios"][:3]
    write_payload(tmp_path, damaged)
    response = signed_client(tmp_path, raise_server_exceptions=False).get(ROUTE)
    assert response.status_code == 500
    assert response.json()["error"] == "demo_provenance_invalid"


def test_an_injected_truth_field_is_blocked(real_payload, tmp_path) -> None:
    """Truth can arrive through data as easily as through an import."""
    damaged = copy.deepcopy(real_payload)
    damaged["scenarios"][0]["decisions"][0]["truth_category"] = "PURE_SAME_ENTITY"
    write_payload(tmp_path, damaged)
    response = signed_client(tmp_path, raise_server_exceptions=False).get(ROUTE)
    assert response.status_code == 500
    assert response.json()["error"] == "truth_leak_blocked"


@pytest.mark.parametrize("field", boundary.FORBIDDEN_FIELDS)
def test_every_declared_truth_field_is_caught(field) -> None:
    with pytest.raises(boundary.TruthLeakInResponseError, match=field):
        boundary.assert_no_truth_fields({"scenarios": [{field: "x"}]})


def test_the_real_payload_carries_no_truth_field(real_payload) -> None:
    boundary.assert_no_truth_fields(real_payload)


# ---- 9. the endpoint computes nothing ----------------------------------


_SERVE_PROBE = """import json, sys
from fastapi.testclient import TestClient
from obsidianchain.api.app import create_app
from obsidianchain.console import db as _db, users as _users

# The analytical routes require a session, so the probe establishes one the
# same way a browser does. None of obsidianchain.console is on
# FORBIDDEN_RECOMPUTATION, so this does not affect what is being measured.
_root = sys.argv[1]
_conn = _db.connect(_root)
_users.create(_conn, username="probe", password="probe-password",
              role="INVESTIGATOR")
_conn.close()

_client = TestClient(create_app(_root))
_client.post("/api/auth/login",
             json={"username": "probe", "password": "probe-password"})
_client.get("/api/demo/scenarios")
print(json.dumps(sorted(n for n in sys.modules if n.startswith("obsidianchain"))))
"""


def test_serving_a_request_imports_no_computation_module(
    real_payload, tmp_path
) -> None:
    """Measured on a live import graph in a CLEAN interpreter.

    Run as a subprocess: measuring it in-process would mean clearing
    sys.modules, which corrupts every other test module in the session.
    """
    write_payload(tmp_path, real_payload)
    result = subprocess.run(
        [sys.executable, "-c", _SERVE_PROBE, str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    graph = set(json.loads(result.stdout.strip().splitlines()[-1]))
    leaked = sorted(
        name for name in graph
        for forbidden in boundary.FORBIDDEN_RECOMPUTATION
        if name == forbidden or name.startswith(forbidden + ".")
    )
    assert not leaked, f"serving the request imported {leaked}"


def artifact_files(root):
    """Every file under ``root`` EXCEPT the console's own database.

    The guarantee this pins is that a request does not rewrite an ARTIFACT.
    It was expressed as "the whole data root is byte-identical", which was
    the same thing until the application database moved in beside them.
    Resolving a session stamps ``last_seen_at``; that is session
    bookkeeping, not an artifact changing under a reader.
    """
    return [
        p for p in root.rglob("*")
        if p.is_file() and not p.name.startswith("obsidianchain.sqlite3")
    ]


def test_the_endpoint_does_not_write_anything(client, tmp_path) -> None:
    """A read-only layer must leave every artifact byte-identical."""
    before = {p: p.stat().st_mtime_ns for p in artifact_files(tmp_path)}
    client.get(ROUTE)
    after = {p: p.stat().st_mtime_ns for p in artifact_files(tmp_path)}
    assert before, "nothing was compared; the fixture wrote no artifacts"
    assert after == before


def test_repeated_requests_are_identical(client) -> None:
    """No hidden state, no cache warming that changes the second answer."""
    assert client.get(ROUTE).json() == client.get(ROUTE).json()


def test_the_app_can_be_built_without_touching_the_filesystem() -> None:
    """create_app must not read an artifact at construction time."""
    create_app(Path("/nonexistent-root"))
