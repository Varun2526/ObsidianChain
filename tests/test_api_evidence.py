"""GET /api/evidence/{id} - the Phase 5.2 endpoint.

The endpoint joins two precomputed artifacts and reshapes one row. These
tests check that it serves only what is persisted, refuses everything else,
distinguishes a stale id from an unknown row, and computes nothing.

Fixture artifacts throughout for the mutation cases. The real frozen inputs
are never modified.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from obsidianchain import provenance as prov
from obsidianchain import run_fingerprint as rf
from obsidianchain.api import artifacts, boundary, evidence, provenance_gate
from obsidianchain.api.app import create_app

API_DIR = Path(__file__).resolve().parents[1] / "src" / "obsidianchain" / "api"


def code_only(source: str) -> str:
    """Executable lines only, via the project's existing convention.

    A module may legitimately *discuss* a forbidden artifact in prose:
    api/boundary.py explains at length which files it refuses and why. Only
    executable references matter, which is the same rule
    tests/test_truth_isolation.py has applied since Phase 4.1.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / "test_truth_isolation.py"
    spec = importlib.util.spec_from_file_location("_ti_for_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.code_only(source)

DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))


def route(evidence_id: str) -> str:
    return f"/api/evidence/{evidence_id}"


# ---- fixtures: a temp data root holding copies of the real artifacts ----


@pytest.fixture(scope="module")
def real_root() -> Path:
    for relative in (artifacts.EVIDENCE_FUNNEL, artifacts.ADDRESS_CLUSTERS,
                     artifacts.CLUSTERS):
        path = DATA_ROOT / relative
        if not path.is_file():
            pytest.skip(f"{path} not generated")
        if not Path(str(path) + prov.META_SUFFIX).is_file():
            pytest.skip(f"{path} has no sidecar")
    return DATA_ROOT


@pytest.fixture()
def root(real_root, tmp_path) -> Path:
    """A copy, so a damaged-artifact test cannot touch the project's data."""
    for relative in (artifacts.EVIDENCE_FUNNEL, artifacts.ADDRESS_CLUSTERS,
                     artifacts.CLUSTERS):
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
    return TestClient(create_app(root), raise_server_exceptions=False)


@pytest.fixture(scope="module")
def fingerprint(real_root) -> str:
    meta = json.loads(
        (Path(str(real_root / artifacts.EVIDENCE_FUNNEL) + prov.META_SUFFIX))
        .read_text(encoding="utf-8")
    )
    return evidence.current_run_fingerprint(meta)


@pytest.fixture(scope="module")
def a_decidable_row(real_root) -> int:
    """An edge_index whose statistics were actually computable.

    Only 1,515 of 253,429 rows have dof >= 1, so a fixed index would be a
    guess. This picks a real one.
    """
    frame = pd.read_parquet(
        real_root / artifacts.EVIDENCE_FUNNEL,
        columns=["edge_index", "dof", "min_pooled"],
    )
    rows = frame[(frame["dof"] >= 1) & (frame["min_pooled"] >= 25)]
    if rows.empty:
        pytest.skip("no decidable row in the funnel")
    return int(rows.iloc[0]["edge_index"])


@pytest.fixture(scope="module")
def a_no_evidence_row(real_root) -> int:
    frame = pd.read_parquet(
        real_root / artifacts.EVIDENCE_FUNNEL, columns=["edge_index", "dof"]
    )
    rows = frame[frame["dof"] < 1]
    if rows.empty:
        pytest.skip("no dof<1 row in the funnel")
    return int(rows.iloc[0]["edge_index"])


def rewrite_sidecar(root: Path, relative: Path, **changes) -> None:
    path = Path(str(root / relative) + prov.META_SUFFIX)
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(changes)
    path.write_text(json.dumps(meta), encoding="utf-8")


# ---- 7, 8. deterministic evidence id ------------------------------------


def test_the_response_echoes_a_canonical_id(client, fingerprint, a_decidable_row) -> None:
    body = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    assert body["evidence_id"] == f"{fingerprint}:{a_decidable_row}"
    assert body["run_fingerprint"] == fingerprint
    assert body["edge_index"] == a_decidable_row


def test_the_same_row_yields_the_same_id_every_time(
    client, fingerprint, a_decidable_row
) -> None:
    ident = f"{fingerprint}:{a_decidable_row}"
    first = client.get(route(ident)).json()
    second = client.get(route(ident)).json()
    assert first["evidence_id"] == second["evidence_id"]
    assert first == second


def test_the_id_round_trips(fingerprint) -> None:
    parsed = evidence.parse_evidence_id(f"{fingerprint}:6729")
    assert str(parsed) == f"{fingerprint}:6729"
    assert parsed.edge_index == 6729


def test_the_fingerprint_comes_from_the_sidecar_not_from_a_file(
    real_root, fingerprint
) -> None:
    """The API must never hash a raw input: raw data may not exist.

    Recomputing from the sidecar's own inputs block must agree with the
    persisted fingerprint, so an API-only deployment gets the same answer.
    """
    meta = json.loads(
        (Path(str(real_root / artifacts.EVIDENCE_FUNNEL) + prov.META_SUFFIX))
        .read_text(encoding="utf-8")
    )
    inputs = meta["inputs"]
    recomputed = rf.fingerprint_from_inputs(
        chain_addr_tx_sha256=inputs["chain_addr_tx_sha256"],
        chain_universe_sha256=inputs["chain_universe_sha256"],
        network_dataset_sha256=inputs["network_dataset_sha256"],
        heuristics=inputs["heuristics"],
        row_statistics_config=inputs["row_statistics_config"],
    )
    assert recomputed == fingerprint == rf.public_fingerprint(
        meta["run_fingerprint"]
    )


def test_the_api_source_never_hashes_a_file() -> None:
    """sha256_file is generation-side only."""
    for path in API_DIR.rglob("*.py"):
        assert "sha256_file" not in code_only(
            path.read_text(encoding="utf-8")
        ), f"{path.name} hashes a file"


# ---- 9, 10, 11. the four failure modes stay distinct --------------------


# NOTE: the empty string is deliberately absent. GET /api/evidence/ does not
# match this route at all, so it is a 404 from the router rather than a 400
# from the handler - a different failure, correctly reported as such.
@pytest.mark.parametrize("bad", [
    "abc", "notahexstring:1", "8756a67bfe8b8db8", "8756a67bfe8b8db8:",
    ":6729", "8756a67bfe8b8db8:-1", "8756a67bfe8b8db8:1.5",
    "8756a67bfe8b8db8:007", "8756A67BFE8B8DB8:1", "8756a67bfe8b8db:1",
    "8756a67bfe8b8db88:1", "8756a67bfe8b8db8:1:2",
])
def test_a_malformed_id_is_400(client, bad) -> None:
    response = client.get(route(bad))
    assert response.status_code == 400, f"{bad!r} gave {response.status_code}"
    assert response.json()["error"] == "evidence_id_invalid"


def test_leading_zeros_are_refused_not_normalised(fingerprint) -> None:
    """Two ids for one row would make equality comparison meaningless."""
    with pytest.raises(evidence.EvidenceIdInvalidError):
        evidence.parse_evidence_id(f"{fingerprint}:007")


def test_an_unknown_edge_index_is_404(client, fingerprint) -> None:
    response = client.get(route(f"{fingerprint}:999999999"))
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "evidence_not_found"
    assert "253,429" in body["detail"]


def test_a_stale_fingerprint_is_409_not_404(client, a_decidable_row) -> None:
    """THE distinction. A re-point must not look like a missing row."""
    response = client.get(route(f"{'0' * 16}:{a_decidable_row}"))
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "evidence_id_stale"
    assert "0000000000000000" in body["detail"]
    assert body["error"] != "evidence_not_found"


def test_a_missing_artifact_is_503_naming_the_command(tmp_path) -> None:
    response = TestClient(
        create_app(tmp_path), raise_server_exceptions=False
    ).get(route(f"{'0' * 16}:1"))
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "artifact_not_generated"
    assert "evidence-funnel" in body["detail"]


def test_the_loader_raises_rather_than_building(tmp_path) -> None:
    with pytest.raises(artifacts.ArtifactMissingError, match="evidence-funnel"):
        artifacts.load_evidence_funnel(tmp_path)


# ---- 12-14. provenance gate --------------------------------------------


def test_production_provenance_is_accepted(client, fingerprint, a_decidable_row) -> None:
    body = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    assert body["provenance"]["provenance_type"] == "PRODUCTION"
    assert body["provenance"]["is_measurement"] is False
    assert body["provenance"]["synthetic_network"] is True


@pytest.mark.parametrize("declared", ["SYNTHETIC_CONTROL", "DEMO", "SOMETHING", None])
def test_a_non_production_artifact_is_refused(
    root, fingerprint, a_decidable_row, declared
) -> None:
    """The realistic accident: reaching for phase33_decisions.csv."""
    rewrite_sidecar(root, artifacts.EVIDENCE_FUNNEL, provenance_type=declared)
    response = TestClient(
        create_app(root), raise_server_exceptions=False
    ).get(route(f"{fingerprint}:{a_decidable_row}"))
    assert response.status_code == 500
    assert response.json()["error"] == "provenance_refused"


def test_a_missing_sidecar_is_refused(root, fingerprint, a_decidable_row) -> None:
    """Absent provenance and production provenance must differ."""
    Path(str(root / artifacts.EVIDENCE_FUNNEL) + prov.META_SUFFIX).unlink()
    response = TestClient(
        create_app(root), raise_server_exceptions=False
    ).get(route(f"{fingerprint}:{a_decidable_row}"))
    assert response.status_code == 500
    assert response.json()["error"] == "provenance_refused"


def test_a_schema_1_sidecar_is_refused(root, fingerprint, a_decidable_row) -> None:
    rewrite_sidecar(
        root, artifacts.EVIDENCE_FUNNEL, schema="obsidianchain.provenance/1"
    )
    response = TestClient(
        create_app(root), raise_server_exceptions=False
    ).get(route(f"{fingerprint}:{a_decidable_row}"))
    assert response.status_code == 500
    assert "provenance/2" in response.json()["detail"]


def test_an_empty_inputs_block_is_refused(root, fingerprint, a_decidable_row) -> None:
    rewrite_sidecar(root, artifacts.EVIDENCE_FUNNEL, inputs={})
    response = TestClient(
        create_app(root), raise_server_exceptions=False
    ).get(route(f"{fingerprint}:{a_decidable_row}"))
    assert response.status_code == 500
    assert response.json()["error"] == "provenance_refused"


@pytest.mark.parametrize("name", [
    "phase33_decisions.csv", "phase33.csv", "world_diagnostics.csv",
])
def test_the_endpoint_never_names_a_synthetic_artifact(name) -> None:
    """Source-level: the forbidden artifacts must not be reachable by name."""
    for path in API_DIR.rglob("*.py"):
        executable = code_only(path.read_text(encoding="utf-8"))
        assert name not in executable, f"{path.name} names {name}"


# ---- 15-20. nothing unsupported is exposed -----------------------------


def test_no_truth_field_reaches_the_response(client, fingerprint, a_decidable_row) -> None:
    boundary.assert_no_truth_fields(
        client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    )


@pytest.mark.parametrize("field", [
    "verdict", "verdict_derived", "reason", "evidence_state",
    "truth_category", "entities_a", "entities_b", "blocked", "contested",
    "proposing_edges", "component_a", "component_b", "risk", "severity",
    "txid", "transaction", "observer_id", "peer_ip", "timestamp",
    "true_entity_id", "true_origin_id",
])
def test_a_forbidden_field_appears_nowhere_in_the_response(
    client, fingerprint, a_decidable_row, field
) -> None:
    body = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()

    def walk(node):
        if isinstance(node, dict):
            assert field not in node, f"{field} present at {list(node)}"
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(body)


@pytest.mark.parametrize("word", ["CONFIRMED", "INFERRED"])
def test_no_unsupported_certainty_word_appears(
    client, fingerprint, a_decidable_row, word
) -> None:
    assert word not in json.dumps(
        client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    )


@pytest.mark.parametrize("claim", [
    "belongs to", "owns these", "same IP", "same entity", "proves",
])
def test_no_ownership_claim_appears(
    client, fingerprint, a_decidable_row, claim
) -> None:
    """The response must never assert identity from a shared origin."""
    body = json.dumps(
        client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    ).lower()
    assert claim.lower() not in body


def test_the_statement_describes_rather_than_concludes(
    client, fingerprint, a_decidable_row
) -> None:
    body = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    assert body["statement"] == (
        "Here is the network evidence recorded for this proposed merge."
    )


def test_the_trajectory_is_declared_as_chain_only(
    client, fingerprint, a_decidable_row
) -> None:
    """It must be impossible to read the row as a fused decision."""
    body = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    assert body["trajectory"]["name"] == "chain-only"
    note = body["trajectory"]["note"]
    assert "NOT a record of a fused-engine decision" in note
    assert "no fused decision ledger is persisted" in note


def test_the_probe_configuration_mismatch_is_stated_not_hidden(
    client, fingerprint, a_decidable_row
) -> None:
    body = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    config = body["statistics_config"]
    assert config["name"] == "probe"
    assert config["min_pooled_observations"] == 1
    assert config["min_observer_observations"] == 2
    assert "NOT" in config["note"]
    rule = body["provenance"]["production_rule"]
    assert rule["min_pooled_observations"] == 25
    assert rule["min_observer_observations"] == 5


# ---- 21, 22. availability semantics ------------------------------------


def test_a_decidable_row_reports_evidence_available(
    client, fingerprint, a_decidable_row
) -> None:
    availability = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["availability"]
    assert availability["evidence_available"] is True
    assert availability["decidable_under_production_rule"] is True


def test_a_row_with_no_computable_statistic_reports_unavailable(
    client, fingerprint, a_no_evidence_row
) -> None:
    """NO_EVIDENCE semantics: nothing was computable, so nothing is claimed."""
    body = client.get(route(f"{fingerprint}:{a_no_evidence_row}")).json()
    assert body["availability"]["evidence_available"] is False
    assert body["statistics"]["chi2"] is None
    assert body["statistics"]["p_value"] is None
    assert body["statistics"]["effect"] is None
    assert body["statistics"]["dof"] == 0


def test_availability_is_two_booleans_and_nothing_else(
    client, fingerprint, a_decidable_row
) -> None:
    """Availability metadata, not a verdict wearing a different hat."""
    availability = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["availability"]
    assert set(availability) == {
        "evidence_available", "decidable_under_production_rule"
    }
    assert all(isinstance(v, bool) for v in availability.values())


def test_the_two_no_evidence_cases_are_distinguishable(real_root) -> None:
    """Evidence absent vs evidence present but below the production gate.

    The distinction the whole design turns on, checked against real rows.
    """
    frame = pd.read_parquet(
        real_root / artifacts.EVIDENCE_FUNNEL,
        columns=["edge_index", "dof", "min_pooled"],
    )
    absent = frame[frame["dof"] < 1]
    present_but_thin = frame[(frame["dof"] >= 1) & (frame["min_pooled"] < 25)]
    assert not absent.empty
    assert not present_but_thin.empty, (
        "the fixture must contain a row with a computable statistic that "
        "still fails the production pooled gate"
    )


# ---- 23. the address / cluster join ------------------------------------


def test_both_sides_resolve_to_addresses(client, fingerprint, a_decidable_row) -> None:
    merge = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["proposed_merge"]
    for side in ("side_a", "side_b"):
        assert merge[side]["address_resolved"] is True
        assert isinstance(merge[side]["address"], str)
        assert merge[side]["address"]
        assert merge[side]["cluster_id"] is not None
        assert merge[side]["cluster_size"] >= 2


def test_the_resolved_address_matches_the_stage_0_index(
    real_root, client, fingerprint, a_decidable_row
) -> None:
    """Verify the join rather than trusting it."""
    funnel = pd.read_parquet(
        real_root / artifacts.EVIDENCE_FUNNEL,
        columns=["edge_index", "node_a", "node_b"],
    )
    row = funnel[funnel["edge_index"] == a_decidable_row].iloc[0]
    index = pd.read_parquet(
        real_root / artifacts.ADDRESS_CLUSTERS,
        columns=["code", "address", "cluster_id"],
    ).set_index("code")
    merge = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["proposed_merge"]
    assert merge["side_a"]["address"] == index.loc[int(row["node_a"]), "address"]
    assert merge["side_b"]["address"] == index.loc[int(row["node_b"]), "address"]


def test_cluster_id_is_not_presented_as_a_decision_time_component(
    client, fingerprint, a_decidable_row
) -> None:
    """Accurate naming: cluster_id is final; component_size_at_record is not."""
    side = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["proposed_merge"]["side_a"]
    assert "cluster_id" in side
    assert "component_size_at_record" in side
    assert "component_a" not in side
    assert "component_id" not in side


def test_an_index_from_a_different_chain_is_refused(
    root, fingerprint, a_decidable_row
) -> None:
    """Resolving codes across two chains would return wrong addresses."""
    path = Path(str(root / artifacts.ADDRESS_CLUSTERS) + prov.META_SUFFIX)
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["inputs"]["chain_addr_tx_sha256"] = "f" * 64
    path.write_text(json.dumps(meta), encoding="utf-8")
    response = TestClient(
        create_app(root), raise_server_exceptions=False
    ).get(route(f"{fingerprint}:{a_decidable_row}"))
    assert response.status_code == 500
    assert response.json()["error"] == "evidence_join_failed"


def test_an_unresolvable_code_fails_rather_than_inventing_an_address(
    root, fingerprint, a_decidable_row
) -> None:
    """No plausible-looking address is ever substituted."""
    funnel = pd.read_parquet(root / artifacts.EVIDENCE_FUNNEL)
    row = funnel[funnel["edge_index"] == a_decidable_row].iloc[0]
    index = pd.read_parquet(root / artifacts.ADDRESS_CLUSTERS)
    index = index[index["code"] != int(row["node_a"])]
    index.to_parquet(root / artifacts.ADDRESS_CLUSTERS, index=False)
    response = TestClient(
        create_app(root), raise_server_exceptions=False
    ).get(route(f"{fingerprint}:{a_decidable_row}"))
    assert response.status_code == 500
    assert response.json()["error"] == "evidence_join_failed"
    assert str(int(row["node_a"])) in response.json()["detail"]


# ---- 24. response schema ------------------------------------------------


def test_the_response_has_exactly_the_declared_top_level_keys(
    client, fingerprint, a_decidable_row
) -> None:
    body = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    assert set(body) == {
        "evidence_id", "edge_index", "run_fingerprint", "provenance",
        "statistics_config", "trajectory", "proposed_merge", "statistics",
        "availability", "statement",
    }


def test_the_provenance_block_carries_all_three_input_hashes(
    client, fingerprint, a_decidable_row
) -> None:
    p = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()["provenance"]
    for key in ("network_dataset_sha256", "chain_addr_tx_sha256",
                "chain_universe_sha256"):
        assert isinstance(p[key], str) and len(p[key]) == 64, key
    assert p["heuristics"] == "multi-input"


def test_the_statistics_block_is_exactly_the_persisted_columns(
    real_root, client, fingerprint, a_decidable_row
) -> None:
    funnel = pd.read_parquet(real_root / artifacts.EVIDENCE_FUNNEL)
    row = funnel[funnel["edge_index"] == a_decidable_row].iloc[0]
    stats = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["statistics"]
    assert stats["pooled_observations_a"] == int(row["pooled_a"])
    assert stats["pooled_observations_b"] == int(row["pooled_b"])
    assert stats["min_pooled"] == int(row["min_pooled"])
    assert stats["dof"] == int(row["dof"])
    assert stats["chi2"] == pytest.approx(float(row["chi2"]))
    assert stats["p_value"] == pytest.approx(float(row["p_value"]))
    assert stats["effect"] == pytest.approx(float(row["effect"]))


# ---- 25-28. no computation, no mutation, determinism -------------------


_PROBE = """
import json, sys
from fastapi.testclient import TestClient
from obsidianchain.api.app import create_app
TestClient(create_app(sys.argv[1]), raise_server_exceptions=False).get(sys.argv[2])
print(json.dumps(sorted(n for n in sys.modules if n.startswith("obsidianchain"))))
"""


def test_serving_a_request_imports_no_computation_module(
    root, fingerprint, a_decidable_row
) -> None:
    """Measured in a clean interpreter, not inferred from the source."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(root),
         route(f"{fingerprint}:{a_decidable_row}")],
        capture_output=True, text=True, check=True,
    )
    graph = set(json.loads(result.stdout.strip().splitlines()[-1]))
    leaked = sorted(
        name for name in graph
        for forbidden in boundary.FORBIDDEN_RECOMPUTATION
        if name == forbidden or name.startswith(forbidden + ".")
    )
    assert not leaked, f"serving the request imported {leaked}"


def test_a_request_does_not_modify_any_artifact(
    root, client, fingerprint, a_decidable_row
) -> None:
    before = {
        p: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in root.rglob("*") if p.is_file()
    }
    client.get(route(f"{fingerprint}:{a_decidable_row}"))
    after = {
        p: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in root.rglob("*") if p.is_file()
    }
    assert after == before


def test_repeated_requests_are_byte_identical(
    client, fingerprint, a_decidable_row
) -> None:
    ident = route(f"{fingerprint}:{a_decidable_row}")
    first = client.get(ident)
    second = client.get(ident)
    assert first.content == second.content


def test_the_endpoint_reads_only_the_three_permitted_artifacts() -> None:
    """Source-level: no path into raw/, truth/, demo output or worlds."""
    # Precise names, not the substring "_truth": that would also match the
    # legitimate guard function assert_no_truth_fields, which exists to
    # REFUSE truth. The quarantined directories are what must be unreachable.
    forbidden = ("worlds/", "reach_stress", "/raw/", "network_truth",
                 "worlds_truth", "entity_assignment.csv", "ground_truth")
    for path in API_DIR.rglob("*.py"):
        executable = code_only(path.read_text(encoding="utf-8"))
        for bad in forbidden:
            assert bad not in executable, f"{path.name} names {bad!r}"


def test_provenance_gate_accepts_only_production() -> None:
    assert provenance_gate.ACCEPTED_TYPES == ("PRODUCTION",)
    assert provenance_gate.REQUIRED_SCHEMA == "obsidianchain.provenance/2"
