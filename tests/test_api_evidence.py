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

from obsidianchain import evidence_contract as contract
from obsidianchain import provenance as prov
from obsidianchain import run_fingerprint as rf
from obsidianchain.api import artifacts, boundary, evidence, provenance_gate
from obsidianchain.api.app import create_app

# The analytical routes now require a session. Reaching them changed; what
# they return did not, and every assertion below is unchanged. See
# tests/console_helpers.py, and tests/test_api_access.py for the boundary
# itself.
from tests.console_helpers import signed_client


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
    return signed_client(root, raise_server_exceptions=False)


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
    response = signed_client(tmp_path, raise_server_exceptions=False).get(route(f"{'0' * 16}:1"))
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
    response = signed_client(root, raise_server_exceptions=False).get(route(f"{fingerprint}:{a_decidable_row}"))
    assert response.status_code == 500
    assert response.json()["error"] == "provenance_refused"


def test_a_missing_sidecar_is_refused(root, fingerprint, a_decidable_row) -> None:
    """Absent provenance and production provenance must differ."""
    Path(str(root / artifacts.EVIDENCE_FUNNEL) + prov.META_SUFFIX).unlink()
    response = signed_client(root, raise_server_exceptions=False).get(route(f"{fingerprint}:{a_decidable_row}"))
    assert response.status_code == 500
    assert response.json()["error"] == "provenance_refused"


def test_a_schema_1_sidecar_is_refused(root, fingerprint, a_decidable_row) -> None:
    rewrite_sidecar(
        root, artifacts.EVIDENCE_FUNNEL, schema="obsidianchain.provenance/1"
    )
    response = signed_client(root, raise_server_exceptions=False).get(route(f"{fingerprint}:{a_decidable_row}"))
    assert response.status_code == 500
    assert "provenance/2" in response.json()["detail"]


def test_an_empty_inputs_block_is_refused(root, fingerprint, a_decidable_row) -> None:
    rewrite_sidecar(root, artifacts.EVIDENCE_FUNNEL, inputs={})
    response = signed_client(root, raise_server_exceptions=False).get(route(f"{fingerprint}:{a_decidable_row}"))
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


#: Fields that must not appear anywhere in the response, at any depth.
#:
#: ``verdict`` and ``reason`` were on this list through Phase 5.2, when
#: neither was persisted and any occurrence would have been a derivation.
#: Schema /2 records both, so they moved to
#: :data:`PERSISTED_ONLY_IN_PRODUCTION_BLOCK` below - permitted in exactly one
#: place and nowhere else. ``verdict_derived`` stays banned outright: that
#: name can only ever mean a reconstruction.
FORBIDDEN_ANYWHERE = [
    "verdict_derived", "evidence_state",
    "truth_category", "entities_a", "entities_b", "blocked", "contested",
    "proposing_edges", "component_a", "component_b", "risk", "severity",
    "txid", "transaction", "observer_id", "peer_ip", "timestamp",
    "true_entity_id", "true_origin_id",
]

#: Persisted /2 fields with exactly one legitimate home.
PERSISTED_ONLY_IN_PRODUCTION_BLOCK = ["verdict", "reason", "reason_code"]


def walk_dicts(node, path="$"):
    """Every dict in the payload, with the path that reached it."""
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            yield from walk_dicts(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from walk_dicts(value, f"{path}[{i}]")


@pytest.mark.parametrize("field", FORBIDDEN_ANYWHERE)
def test_a_forbidden_field_appears_nowhere_in_the_response(
    client, fingerprint, a_decidable_row, field
) -> None:
    body = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    for path, node in walk_dicts(body):
        assert field not in node, f"{field} present at {path}"


@pytest.mark.parametrize("field", PERSISTED_ONLY_IN_PRODUCTION_BLOCK)
def test_a_persisted_verdict_field_lives_only_in_the_production_block(
    client, fingerprint, a_decidable_row, field
) -> None:
    """One home, so a caller cannot pick up a verdict beside probe numbers.

    A ``verdict`` sitting next to the probe statistics would read as the
    verdict of those statistics. It is not - it is the verdict of a separate
    evaluation under a different configuration.
    """
    body = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    homes = [path for path, node in walk_dicts(body) if field in node]
    assert homes == ["$.production_evidence"], (
        f"{field} should appear only in production_evidence, found at {homes}"
    )


@pytest.mark.parametrize("word", ["CONFIRMED", "INFERRED"])
def test_no_unsupported_certainty_word_appears(
    client, fingerprint, a_decidable_row, word
) -> None:
    assert word not in json.dumps(
        client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    )


#: The frozen disclaimers, which are allowed to contain the banned phrases
#: because they exist to deny them. Excised before the substring scan below,
#: and pinned separately in
#: ``test_the_frozen_wordings_are_the_contract_constants_verbatim`` so they
#: cannot be reworded into the claim they currently refuse.
FROZEN_DISCLAIMERS = (
    contract.NOT_SEPARATED_MEANING,
    contract.FROZEN_RUN_LIMITATION,
    contract.VERDICT_SCOPE,
    contract.VERDICT_DEFINITION,
)


def body_outside_the_disclaimers(payload) -> str:
    """The serialised response with the frozen disclaimers removed.

    A naive substring ban cannot survive schema /2: NOT_SEPARATED must travel
    with a sentence that contains "the same entity" precisely in order to say
    the verdict is not evidence of it. Deleting the ban would be the wrong
    fix - it is the check that stops a claim appearing. So the disclaimers
    are cut out and verified verbatim against the contract instead, and
    everything else is scanned exactly as before.
    """
    text = json.dumps(payload)
    for wording in FROZEN_DISCLAIMERS:
        text = text.replace(json.dumps(wording)[1:-1], "")
    return text.lower()


@pytest.mark.parametrize("claim", [
    "belongs to", "owns these", "same IP", "same entity", "proves",
])
def test_no_ownership_claim_appears(
    client, fingerprint, a_decidable_row, claim
) -> None:
    """The response must never assert identity from a shared origin."""
    body = body_outside_the_disclaimers(
        client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    )
    assert claim.lower() not in body


def test_the_frozen_wordings_are_the_contract_constants_verbatim(
    client, fingerprint, a_decidable_row
) -> None:
    """No paraphrase. A paraphrase is where a denial becomes a claim."""
    block = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["production_evidence"]
    assert block["definition"] == contract.VERDICT_DEFINITION
    assert block["scope"] == contract.VERDICT_SCOPE
    assert block["frozen_run_limitation"] == contract.FROZEN_RUN_LIMITATION
    assert block["meaning"] == contract.NOT_SEPARATED_MEANING

    # and the constant itself still denies rather than asserts
    assert "is not evidence that" in contract.NOT_SEPARATED_MEANING
    assert "the same entity" in contract.NOT_SEPARATED_MEANING


def test_the_disclaimer_excision_is_not_a_blanket_exemption(
    client, fingerprint, a_decidable_row
) -> None:
    """The excision must remove the disclaimers and nothing else.

    Otherwise the substring ban above would be passing vacuously.
    """
    payload = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    full = json.dumps(payload)
    trimmed = body_outside_the_disclaimers(payload)
    removed = len(full) - len(trimmed)
    expected = sum(len(json.dumps(w)[1:-1]) for w in FROZEN_DISCLAIMERS)
    # VERDICT_DEFINITION also appears inside the persisted evaluation_config
    # note, so it is removed twice; nothing else may be.
    assert removed == expected + len(contract.VERDICT_DEFINITION), (
        f"excision removed {removed} characters, expected "
        f"{expected + len(contract.VERDICT_DEFINITION)}"
    )
    assert len(trimmed) > 1000, "the response was almost entirely excised"


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
    response = signed_client(root, raise_server_exceptions=False).get(route(f"{fingerprint}:{a_decidable_row}"))
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
    response = signed_client(root, raise_server_exceptions=False).get(route(f"{fingerprint}:{a_decidable_row}"))
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
        "production_evidence", "availability", "statement",
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


# ---- Phase 5.3 section 9. the persisted production evaluation ----------


def test_the_production_block_is_exactly_the_persisted_columns(
    real_root, client, fingerprint, a_decidable_row
) -> None:
    """Read, not derived. Every field traced back to its /2 column."""
    funnel = pd.read_parquet(real_root / artifacts.EVIDENCE_FUNNEL)
    row = funnel[funnel["edge_index"] == a_decidable_row].iloc[0]
    block = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["production_evidence"]

    assert block["verdict"] == row["verdict_production"]
    assert block["reason"] == row["reason_production"]
    assert block["reason_code"] == row["reason_code_production"]
    assert block["dof"] == int(row["dof_production"])
    assert block["chi2"] == pytest.approx(float(row["chi2_production"]))
    assert block["p_value"] == pytest.approx(float(row["p_value_production"]))
    assert block["effect"] == pytest.approx(float(row["effect_production"]))


def test_the_reason_code_matches_the_contract_mapping(
    real_root, client, fingerprint, a_decidable_row
) -> None:
    """The code the API serves is the code the contract assigns the reason."""
    block = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["production_evidence"]
    assert block["reason_code"] == contract.reason_code(block["reason"])


def test_both_statistic_blocks_declare_which_configuration_they_are(
    client, fingerprint, a_decidable_row
) -> None:
    """Section 9: a frontend holding two must not be able to plot the wrong one."""
    body = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    assert body["statistics"]["config"] == "probe"
    assert body["production_evidence"]["config"] == "production"


def test_the_production_block_carries_its_evaluation_configuration(
    client, fingerprint, a_decidable_row
) -> None:
    config = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["production_evidence"]["evaluation_config"]
    assert config["min_pooled_observations"] == 25
    assert config["min_observer_observations"] == 5
    assert config["alpha"] == 1e-4
    assert config["min_effect"] == 0.05
    assert config["evaluation_order"][0] == "POOLED_GATE"


def test_the_two_configurations_are_actually_different(
    client, fingerprint, a_decidable_row
) -> None:
    """If they were the same, labelling them would be pointless."""
    body = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    probe = body["statistics_config"]
    production = body["production_evidence"]["evaluation_config"]
    assert (probe["min_pooled_observations"],
            probe["min_observer_observations"]) != (
        production["min_pooled_observations"],
        production["min_observer_observations"],
    )


def test_dof_serialises_as_an_integer_not_a_float(
    client, fingerprint, a_decidable_row
) -> None:
    """Section 13. The nullable column reads back as a float in some pandas
    versions; the response must say 8, never 8.0."""
    raw = client.get(route(f"{fingerprint}:{a_decidable_row}")).text
    body = json.loads(raw)
    dof = body["production_evidence"]["dof"]
    assert isinstance(dof, int) and not isinstance(dof, bool)
    assert f'"dof": {dof}.0' not in raw
    assert f'"dof":{dof}.0' not in raw


def test_an_unevaluated_row_reports_null_not_a_dataclass_default(
    client, fingerprint, a_no_evidence_row
) -> None:
    """Section 13. chi2=0, p=1.0 would read as 'perfectly consistent'."""
    block = client.get(
        route(f"{fingerprint}:{a_no_evidence_row}")
    ).json()["production_evidence"]
    assert block["verdict"] == "NO_EVIDENCE"
    for field in ("dof", "chi2", "p_value", "effect"):
        assert block[field] is None, f"{field} was {block[field]!r}, expected null"
    # the reason code, not a number, is what discriminates the three gates
    assert block["reason_code"] in {
        "INSUFFICIENT_POOLED", "INSUFFICIENT_OBSERVER", "ZERO_VARIANCE"
    }


def test_not_separated_never_travels_without_its_meaning(
    real_root, client, fingerprint
) -> None:
    """Section 8, on every row that carries the verdict."""
    funnel = pd.read_parquet(
        real_root / artifacts.EVIDENCE_FUNNEL,
        columns=["edge_index", "verdict_production"],
    )
    rows = funnel[funnel["verdict_production"] == "NOT_SEPARATED"]
    # Not a skip. This artifact has 40 NOT_SEPARATED rows and the frozen
    # metrics pin that; zero of them would mean the production evaluation
    # stopped producing the verdict whose wording this test guards, which is
    # a failure to report and not a reason to stand down.
    assert not rows.empty, (
        "no NOT_SEPARATED row in the funnel. The frozen run has 40; if the "
        "production evaluation no longer yields the verdict, section 8's "
        "wording guarantee is untested rather than satisfied."
    )
    for edge in rows["edge_index"].tolist()[:5]:
        block = client.get(
            route(f"{fingerprint}:{int(edge)}")
        ).json()["production_evidence"]
        assert block["meaning"] == contract.NOT_SEPARATED_MEANING


def test_the_frozen_run_limitation_is_on_every_verdict_response(
    client, fingerprint, a_decidable_row, a_no_evidence_row
) -> None:
    """Section 10. A screenshot must not read as a claim about the method."""
    for edge in (a_decidable_row, a_no_evidence_row):
        block = client.get(
            route(f"{fingerprint}:{edge}")
        ).json()["production_evidence"]
        assert block["frozen_run_limitation"] == contract.FROZEN_RUN_LIMITATION
        assert "property of this frozen dataset, not" in block[
            "frozen_run_limitation"
        ]


def test_the_verdict_is_scoped_to_the_proposed_union_not_the_clusters(
    client, fingerprint, a_decidable_row
) -> None:
    """Section 18. cluster_id is final; component_size_at_record is not."""
    body = client.get(route(f"{fingerprint}:{a_decidable_row}")).json()
    scope = body["production_evidence"]["scope"]
    assert scope == contract.VERDICT_SCOPE
    assert "THIS proposed" in scope
    # and the two differently-timed fields it warns about are both present
    side = body["proposed_merge"]["side_a"]
    assert "cluster_id" in side and "component_size_at_record" in side


def test_the_run_verdict_counts_match_the_artifact(
    real_root, client, fingerprint, a_decidable_row
) -> None:
    """The run-level context is the artifact's own count, not a recount."""
    funnel = pd.read_parquet(
        real_root / artifacts.EVIDENCE_FUNNEL, columns=["verdict_production"]
    )
    actual = funnel["verdict_production"].value_counts().to_dict()
    served = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["production_evidence"]["run_verdict_counts"]
    for verdict, count in served.items():
        assert actual.get(verdict, 0) == count, verdict


def test_the_trajectory_equivalence_claim_is_carried_with_the_verdict(
    client, fingerprint, a_decidable_row
) -> None:
    """Section 7. The chain-only funnel may only be presented as the fused
    trajectory because a verification established the two coincide."""
    block = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["production_evidence"]
    assert block["veto_never_fired"] is True
    assert block["trajectory_equivalence"].startswith("VERIFIED:")


def test_a_schema_1_evidence_artifact_is_refused_naming_the_command(
    root, fingerprint, a_decidable_row
) -> None:
    """Section 1. Never served silently without the fields it lacks."""
    path = Path(str(root / artifacts.EVIDENCE_FUNNEL) + prov.META_SUFFIX)
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["artifact"]["artifact_schema"] = contract.ARTIFACT_SCHEMA_V1
    path.write_text(json.dumps(meta), encoding="utf-8")

    client = signed_client(root, raise_server_exceptions=False)
    response = client.get(route(f"{fingerprint}:{a_decidable_row}"))
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert contract.ARTIFACT_SCHEMA in detail
    assert "evidence-funnel" in detail


@pytest.mark.parametrize("declared", [None, "", "obsidianchain.other/2"])
def test_an_artifact_with_no_declared_schema_is_refused(
    root, fingerprint, a_decidable_row, declared
) -> None:
    path = Path(str(root / artifacts.EVIDENCE_FUNNEL) + prov.META_SUFFIX)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if declared is None:
        meta.pop("artifact", None)
    else:
        meta["artifact"]["artifact_schema"] = declared
    path.write_text(json.dumps(meta), encoding="utf-8")

    client = signed_client(root, raise_server_exceptions=False)
    response = client.get(route(f"{fingerprint}:{a_decidable_row}"))
    assert response.status_code == 500
    assert response.json()["error"] == "provenance_refused"


def test_a_phase_5_2_evidence_id_is_409_not_silently_repointed(
    client, a_decidable_row
) -> None:
    """Section 6/9. The Phase 5.2 fingerprint covered the probe row config
    only. Schema /2 rows are determined by the production configuration too,
    so the fingerprint moved and every 5.2 id must fail loudly."""
    stale = "8756a67bfe8b8db8"
    response = client.get(route(f"{stale}:{a_decidable_row}"))
    assert response.status_code == 409
    body = response.json()
    assert stale in body["detail"]
    assert body["error"] == "evidence_id_stale"


def test_the_phase_5_2_fingerprint_is_no_longer_current(fingerprint) -> None:
    """Guards the test above: it would pass vacuously on any wrong string."""
    assert fingerprint != "8756a67bfe8b8db8"


# ---- section 20. the frozen-input manifest ------------------------------


def test_the_served_run_is_the_frozen_input_manifest(
    real_root, client, fingerprint, a_decidable_row
) -> None:
    """The response's provenance must name the frozen network dataset.

    An artifact regenerated against a different capture would serve numbers
    that no frozen metric describes. The hash is checked against the file the
    project froze, not against whatever the sidecar happens to claim.
    """
    frozen = (
        "c405493d5bd904c0e17c840dca79386a7502cdde45518810fe8398d7adf9380a"
    )
    observations = real_root / "processed" / "network" / "observations.parquet"
    # Not a skip. The served provenance is only meaningful if it names the
    # dataset the project actually froze, so an absent dataset makes this
    # assertion unverifiable - which must be a failure. A skip here would
    # remove the one check that the response is not describing some other
    # capture, and remove it silently.
    assert observations.is_file(), (
        f"{observations} is absent, so the served run cannot be checked "
        f"against the frozen input manifest. Generate it with "
        f"'make run ARGS=\"network-generate\"'."
    )

    digest = rf.sha256_file(observations)
    assert digest == frozen, (
        "the frozen network dataset changed on disk; every downstream metric "
        "and every evidence id is invalidated"
    )
    served = client.get(
        route(f"{fingerprint}:{a_decidable_row}")
    ).json()["provenance"]["network_dataset_sha256"]
    assert served == frozen


# ---- 25-28. no computation, no mutation, determinism -------------------


_PROBE = """import json, sys
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
_client = TestClient(create_app(_root), raise_server_exceptions=False)
_client.post("/api/auth/login",
             json={"username": "probe", "password": "probe-password"})
_client.get(sys.argv[2])
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


def artifact_files(root):
    """Every file under ``root`` EXCEPT the console's own database.

    The read-only guarantee these tests pin is about ARTIFACTS: a request
    must not rewrite a parquet, a sidecar or a payload. It was expressed as
    "the whole data root is byte-identical", which was the same thing until
    the application database moved in beside them.

    A request now legitimately touches that database - resolving a session
    stamps ``last_seen_at`` - and that is session bookkeeping, not an
    artifact being modified. Excluding it keeps the assertion aimed at what
    it was always aiming at; every artifact is still compared.
    """
    return [
        p for p in root.rglob("*")
        if p.is_file() and not p.name.startswith("obsidianchain.sqlite3")
    ]


def test_a_request_does_not_modify_any_artifact(
    root, client, fingerprint, a_decidable_row
) -> None:
    before = {
        p: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in artifact_files(root)
    }
    client.get(route(f"{fingerprint}:{a_decidable_row}"))
    after = {
        p: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in artifact_files(root)
    }
    assert before, "nothing was compared; the fixture wrote no artifacts"
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
