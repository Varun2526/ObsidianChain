"""GET /api/alerts/{id}/separation-evidence - the alert -> funnel join.

The separation funnel has been served by ``/api/evidence/{id}`` since Phase
5.2 and nothing ever called it, because no route connected an alert to the
records made inside its cluster. The network layer's whole question - can a
network-derived constraint stop a harmful merge? - therefore had no answer
anywhere in the product.

These tests check that the join is CORRECT rather than merely plausible: the
two artifacts must agree on the inputs that determine an address code, the
alert's own members must resolve to its cluster, and a mismatch must be
refused rather than served as an empty result.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from obsidianchain import evidence_contract as contract
from obsidianchain import provenance as prov
from obsidianchain.api import artifacts, separation
from obsidianchain.api.app import create_app
from tests.console_helpers import signed_client

DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))

#: CLUSTERS is not used by the join itself; /api/evidence/{id} needs it, and
#: one test resolves an evidence id this endpoint hands out against that
#: route - which is the property that makes the ids worth emitting at all.
NEEDED = (
    artifacts.ALERTS,
    artifacts.EVIDENCE_FUNNEL,
    artifacts.ADDRESS_CLUSTERS,
    artifacts.CLUSTERS,
    *artifacts.ALERT_TABLES.values(),
)


@pytest.fixture(scope="module")
def real_root() -> Path:
    for relative in NEEDED:
        path = DATA_ROOT / relative
        if not path.is_file():
            pytest.skip(f"{path} not generated")
        if not Path(str(path) + prov.META_SUFFIX).is_file():
            pytest.skip(f"{path} has no sidecar")
    return DATA_ROOT


@pytest.fixture()
def root(real_root, tmp_path) -> Path:
    """A copy, so a damaged-artifact test cannot touch the project's data."""
    for relative in NEEDED:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(real_root / relative, target)
        shutil.copy2(
            str(real_root / relative) + prov.META_SUFFIX,
            str(target) + prov.META_SUFFIX,
        )
    return tmp_path


@pytest.fixture()
def client(root) -> TestClient:
    # The analytical routes require a session now. What they return is
    # unchanged, and every assertion below is unchanged with it; see
    # tests/test_api_access.py for the boundary itself.
    return signed_client(root)


@pytest.fixture(scope="module")
def alert_id(real_root) -> str:
    import pandas as pd

    frame = pd.read_parquet(real_root / artifacts.ALERTS, columns=["alert_id", "rank"])
    return str(frame.sort_values("rank").iloc[0]["alert_id"])


def route(alert_id: str) -> str:
    return f"/api/alerts/{alert_id}/separation-evidence"


# ---- the join -----------------------------------------------------------


def test_the_endpoint_responds(client, alert_id) -> None:
    assert client.get(route(alert_id)).status_code == 200


def test_it_returns_the_clusters_internal_merges(client, alert_id) -> None:
    """A cluster of n members is built by n-1 accepted merges."""
    import pandas as pd

    body = client.get(route(alert_id)).json()
    members = pd.read_parquet(
        Path(client.app.state.data_root) / artifacts.ALERT_TABLES["alert_members"],
        columns=["alert_id", "address"],
    )
    n_members = int((members["alert_id"] == alert_id).sum())
    assert body["proposed_merges_total"] == n_members - 1


def test_every_row_carries_a_resolvable_evidence_id(client, alert_id) -> None:
    """The ids must work against the endpoint that already exists."""
    body = client.get(route(alert_id)).json()
    assert body["rows"]
    evidence_id = body["rows"][0]["evidence_id"]
    assert evidence_id.startswith(body["evidence_run_fingerprint"] + ":")
    resolved = client.get(f"/api/evidence/{evidence_id}")
    assert resolved.status_code == 200


def test_the_two_runs_are_reported_separately(client, alert_id) -> None:
    """The alert and the funnel are different artifacts with different runs.

    Reporting one fingerprint for both would imply an identity that does not
    hold; the join is by address code, not by run.
    """
    body = client.get(route(alert_id)).json()
    assert body["alert_run_fingerprint"] != body["evidence_run_fingerprint"]
    assert len(body["alert_run_fingerprint"]) == 16
    assert len(body["evidence_run_fingerprint"]) == 16
    assert "address code" in body["join_basis"]


def test_the_verdict_counts_sum_to_the_row_total(client, alert_id) -> None:
    body = client.get(route(alert_id)).json()
    assert sum(body["verdicts"].values()) == body["proposed_merges_total"]


# ---- error taxonomy, matching the analytical layer's -------------------


def test_a_malformed_alert_id_is_400(client) -> None:
    assert client.get(route("not-an-alert")).status_code == 400


def test_an_alert_from_another_run_is_409(client) -> None:
    response = client.get(route("deadbeefdeadbeef:1"))
    assert response.status_code == 409
    assert response.json()["error"] == "alert_id_stale"


def test_an_unknown_cluster_is_404(client, alert_id) -> None:
    run = alert_id.split(":")[0]
    response = client.get(route(f"{run}:999999999"))
    assert response.status_code == 404


# ---- the join is verified, never assumed --------------------------------


def test_a_disagreeing_clustering_basis_is_refused(root, client, alert_id) -> None:
    """Different chain inputs mean the address codes address different things.

    This is the separation-evidence counterpart of ``require_same_run``: the
    join would succeed silently and describe other components.
    """
    sidecar = Path(str(root / artifacts.ADDRESS_CLUSTERS) + prov.META_SUFFIX)
    meta = json.loads(sidecar.read_text())
    meta["inputs"]["chain_universe_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(meta))

    response = client.get(route(alert_id))
    assert response.status_code == 500
    assert response.json()["error"] == "provenance_refused"


def test_a_missing_basis_key_is_refused_rather_than_assumed(root, client, alert_id) -> None:
    sidecar = Path(str(root / artifacts.ADDRESS_CLUSTERS) + prov.META_SUFFIX)
    meta = json.loads(sidecar.read_text())
    meta["inputs"].pop("heuristics", None)
    sidecar.write_text(json.dumps(meta))

    response = client.get(route(alert_id))
    assert response.status_code == 500


def test_the_basis_check_is_symmetric() -> None:
    left = {"inputs": {"chain_addr_tx_sha256": "a",
                       "chain_universe_sha256": "b", "heuristics": "multi-input"}}
    right = {"inputs": dict(left["inputs"])}
    separation.require_same_clustering_basis(left, right)

    right["inputs"]["heuristics"] = "multi-input+change"
    with pytest.raises(Exception):
        separation.require_same_clustering_basis(left, right)


# ---- what the response must never let a reader conclude ----------------


def test_the_frozen_wordings_travel_with_the_response(client, alert_id) -> None:
    """Read from the contract, not paraphrased, so they cannot drift."""
    body = client.get(route(alert_id)).json()
    assert body["not_separated_meaning"] == contract.NOT_SEPARATED_MEANING
    assert body["verdict_scope"] == contract.VERDICT_SCOPE
    assert body["frozen_run_limitation"] == contract.FROZEN_RUN_LIMITATION


def test_it_never_claims_two_components_are_the_same_entity(client, alert_id) -> None:
    body = json.dumps(client.get(route(alert_id)).json()).lower()
    for claim in ("same entity", "same person", "same owner", "must-link"):
        if claim in body:
            # The only permitted occurrences are inside a NEGATION - the
            # contract's own text saying what these verdicts do NOT mean.
            assert ("not evidence" in body or "never" in body), (
                f"the response contains {claim!r} outside a negation"
            )


def test_cannot_link_is_named_as_the_only_constraint(client, alert_id) -> None:
    body = client.get(route(alert_id)).json()
    assert "CANNOT-LINK" in body["cannot_link_meaning"]
    assert "never emits MUST-LINK" in body["cannot_link_meaning"]


def test_no_truth_field_reaches_the_response(client, alert_id) -> None:
    from obsidianchain.api import boundary

    boundary.assert_no_truth_fields(client.get(route(alert_id)).json())
