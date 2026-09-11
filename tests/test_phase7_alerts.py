"""Phase 7: the alert artifacts and the investigation API.

Three properties dominate this file, and each exists because getting it wrong
would be invisible on the rendered page:

1. **No response computes anything.** The API's guarantee is that a response
   is a file the pipeline already wrote. A handler that scored on request
   would produce numbers no manifest describes, and would put a LightGBM
   booster one call away from an HTTP route.

2. **No artifact carries a label.** On real data there is no ground truth to
   show. A label column would render as just another number and nobody
   looking at the dashboard could tell it was the answer.

3. **Nothing asserts identity from network data.** Not "this IP owns this
   wallet", not "this IP sent it", not "these two addresses are the same
   person". One server broadcasts for tens of thousands of unrelated users,
   so the network layer can only ever say two groups look DIFFERENT.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from obsidianchain import provenance as prov
from obsidianchain.alerts import contract
from obsidianchain.api import alerts as alerts_api
from obsidianchain.api import artifacts, boundary, provenance_gate
from obsidianchain.api.app import API_PREFIX, create_app

DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))

#: Phrases a response must never contain. Checked against the whole rendered
#: body, because an ownership claim can arrive through a data field as
#: easily as through a hand-written string.
FORBIDDEN_CLAIMS = (
    "owns this wallet", "owns the wallet", "ip owns", "owned by ip",
    "sent by ip", "ip is the sender", "same person as", "identified as",
)

LABEL_COLUMNS = ("y", "class", "label", "truth", "is_illicit")


# ---- the contract itself -----------------------------------------------


def test_alert_id_round_trips() -> None:
    made = contract.make_alert_id("0123456789abcdef" + "0" * 48, 42)
    assert made == "0123456789abcdef:42"
    assert contract.parse_alert_id(made) == ("0123456789abcdef", 42)


@pytest.mark.parametrize("bad", [
    "", "abc", "0123456789abcdef", "0123456789abcdef:", "0123456789abcdef:-1",
    "0123456789abcdef:007", "ZZZZZZZZZZZZZZZZ:1", "0123456789abcdef:1:2",
])
def test_a_malformed_alert_id_is_refused(bad) -> None:
    """Anchored, no sign, no leading zeros: exactly one string per alert."""
    with pytest.raises(contract.AlertIdInvalidError):
        contract.parse_alert_id(bad)


def test_the_four_categories_are_distinct_and_complete() -> None:
    assert len(set(contract.CATEGORIES)) == 4
    assert contract.MODEL_SIGNAL in contract.CATEGORIES
    assert contract.INSUFFICIENT_EVIDENCE in contract.CATEGORIES


def test_only_network_features_are_network_context() -> None:
    """M0-M2 describe the ledger; only M3 describes announcements."""
    assert contract.GROUP_CATEGORY["M3"] == contract.NETWORK_CONTEXT
    for group in ("M0", "M1", "M2"):
        assert contract.GROUP_CATEGORY[group] == contract.BLOCKCHAIN_CONTEXT


def test_only_supported_relationships_exist() -> None:
    """Two, and both are statements about transactions, not about people."""
    assert set(contract.RELATIONSHIPS) == {contract.CO_SPEND, contract.FUNDED}
    for text in contract.RELATIONSHIPS.values():
        assert len(text) > 60


def test_the_network_wording_denies_ownership_explicitly() -> None:
    text = contract.NETWORK_CONTEXT_MEANING.lower()
    assert "do not establish that an ip address owns" in text
    assert "not thereby the same person" in text
    assert "different" in text


def test_the_alert_meaning_denies_that_a_cluster_is_a_person() -> None:
    text = contract.ALERT_MEANING.lower()
    assert "not a person" in text
    assert "not a finding" in text


def test_the_contract_imports_nothing_but_the_standard_library() -> None:
    """It exists so the API can name these without reaching a builder."""
    source = Path(contract.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("obsidianchain"), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("obsidianchain"), alias.name


def test_the_feature_group_table_matches_the_live_contract() -> None:
    """The API reads a literal copy; it must not drift from the real one."""
    from obsidianchain.alerts import feature_groups
    from obsidianchain.features import dataset as ds

    live = {c: g for g, cols in ds.FEATURE_GROUPS.items() for c in cols}
    assert feature_groups.GROUP_OF == live, (
        "alerts/feature_groups.py is stale; regenerate it from "
        "features.dataset.FEATURE_GROUPS"
    )


# ---- the API may not reach a computation path --------------------------


def test_the_api_cannot_import_the_model_or_the_feature_builders() -> None:
    """Phase 7's whole guarantee, enforced by the Phase 5.1 denylist."""
    assert "obsidianchain.ml" in boundary.FORBIDDEN_RECOMPUTATION
    assert "obsidianchain.features" in boundary.FORBIDDEN_RECOMPUTATION
    assert "obsidianchain.alerts.build" in boundary.FORBIDDEN_RECOMPUTATION


def test_the_alert_route_module_imports_no_builder() -> None:
    source = Path(alerts_api.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    offenders = [
        n for n in names
        for forbidden in boundary.FORBIDDEN_RECOMPUTATION
        if n == forbidden or n.startswith(forbidden + ".")
    ]
    assert not offenders, f"api/alerts.py imports {offenders}"


# ---- the generated artifacts -------------------------------------------


@pytest.fixture(scope="module")
def alerts_frame() -> pd.DataFrame:
    """The real artifact. A FAILURE, never a skip, when absent."""
    path = DATA_ROOT / artifacts.ALERTS
    assert path.is_file(), (
        f"{path} is absent, so the alert layer has not been verified against "
        f"real rows. Generate it with 'make run ARGS=\"phase7-alerts\"'."
    )
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(DATA_ROOT), raise_server_exceptions=False)


@pytest.mark.parametrize("name", [
    "alerts", "alert_members", "alert_explanations", "alert_timeline",
    "alert_relationships",
])
def test_no_alert_artifact_carries_a_label(name) -> None:
    """Property 2. An investigation tool must not be a label viewer."""
    path = DATA_ROOT / "processed" / f"{name}.parquet"
    assert path.is_file(), f"{path} is absent"
    columns = pd.read_parquet(path).columns
    offenders = [c for c in columns if c in LABEL_COLUMNS]
    assert not offenders, f"{name} carries label column(s) {offenders}"


@pytest.mark.parametrize("name,schema", [
    ("alerts", contract.ALERT_SCHEMA),
    ("alert_members", contract.MEMBER_SCHEMA),
    ("alert_explanations", contract.EXPLANATION_SCHEMA),
    ("alert_timeline", contract.TIMELINE_SCHEMA),
    ("alert_relationships", contract.RELATIONSHIP_SCHEMA),
])
def test_every_alert_artifact_is_production_and_identity_stamped(name, schema) -> None:
    path = DATA_ROOT / "processed" / f"{name}.parquet"
    meta = provenance_gate.require_production(path)
    assert meta["provenance_type"] == "PRODUCTION"
    embedded = prov.read_artifact_identity(path)
    assert embedded[prov.FINGERPRINT_METADATA_KEY] == meta["run_fingerprint"]
    # Each table declares its OWN layout, so a reader can tell from the
    # sidecar alone which schema a file has.
    assert meta["artifact"]["artifact_schema"] == schema


def test_the_five_artifacts_share_one_run(alerts_frame) -> None:
    """A stale detail table would join cleanly and describe another cluster."""
    index = provenance_gate.require_production(DATA_ROOT / artifacts.ALERTS)
    for name in artifacts.ALERT_TABLES:
        other = provenance_gate.require_production(
            DATA_ROOT / "processed" / f"{name}.parquet"
        )
        provenance_gate.require_same_run(index, other, name)


def test_alerts_are_ranked_densely_from_one(alerts_frame) -> None:
    ranks = sorted(alerts_frame["rank"].tolist())
    assert ranks == list(range(1, len(alerts_frame) + 1))
    ordered = alerts_frame.sort_values("rank")
    assert ordered["risk_score"].is_monotonic_decreasing


def test_every_alert_id_is_well_formed_and_matches_the_run(alerts_frame) -> None:
    meta = provenance_gate.require_production(DATA_ROOT / artifacts.ALERTS)
    current = meta["run_fingerprint"][:16]
    for alert_id, cluster_id in zip(
        alerts_frame["alert_id"], alerts_frame["cluster_id"]
    ):
        run, parsed = contract.parse_alert_id(alert_id)
        assert run == current
        assert parsed == int(cluster_id)


def test_severity_values_are_from_the_contract(alerts_frame) -> None:
    assert set(alerts_frame["severity"]) <= set(contract.SEVERITIES)


def test_relationships_are_only_the_supported_kinds() -> None:
    frame = pd.read_parquet(DATA_ROOT / "processed" / "alert_relationships.parquet")
    assert set(frame["relationship"]) <= set(contract.RELATIONSHIPS), (
        "an unsupported relationship kind reached the artifact"
    )


def test_explanation_categories_are_from_the_contract() -> None:
    frame = pd.read_parquet(
        DATA_ROOT / "processed" / "alert_explanations.parquet"
    )
    assert set(frame["signal_category"]) == {contract.MODEL_SIGNAL}
    assert set(frame["value_category"]) <= set(contract.CATEGORIES)


def test_a_null_feature_value_is_reported_as_insufficient_evidence() -> None:
    """Property: "not measured" never renders as a measured zero."""
    frame = pd.read_parquet(
        DATA_ROOT / "processed" / "alert_explanations.parquet"
    )
    missing = frame[frame["feature_value"].isna()]
    assert not missing.empty, "no null feature values; the check is vacuous"
    assert set(missing["value_category"]) == {contract.INSUFFICIENT_EVIDENCE}
    present = frame[frame["feature_value"].notna()]
    assert contract.INSUFFICIENT_EVIDENCE not in set(present["value_category"])


def test_network_features_are_categorised_as_network_context() -> None:
    frame = pd.read_parquet(
        DATA_ROOT / "processed" / "alert_explanations.parquet"
    )
    m3 = frame[(frame["feature_group"] == "M3") & frame["feature_value"].notna()]
    assert not m3.empty
    assert set(m3["value_category"]) == {contract.NETWORK_CONTEXT}


# ---- the endpoints ------------------------------------------------------


def test_the_list_endpoint_ranks_and_reports_its_counts(client) -> None:
    body = client.get(f"{API_PREFIX}/alerts?limit=5").json()
    assert body["alert_count_matched"] == body["alert_count_total"]
    assert len(body["alerts"]) == 5
    ranks = [a["rank"] for a in body["alerts"]]
    assert ranks == sorted(ranks)


def test_severity_filter_narrows_the_result(client) -> None:
    everything = client.get(f"{API_PREFIX}/alerts?limit=1").json()
    critical = client.get(f"{API_PREFIX}/alerts?severity=CRITICAL&limit=1").json()
    assert critical["alert_count_matched"] < everything["alert_count_total"]
    assert critical["alerts"][0]["severity"] == "CRITICAL"


def test_risk_and_timestep_filters_apply(client) -> None:
    body = client.get(
        f"{API_PREFIX}/alerts?min_risk=0.5&first_timestep=45&limit=100"
    ).json()
    for alert in body["alerts"]:
        assert alert["risk_score"] >= 0.5
        assert alert["last_timestep"] >= 45


def test_an_unknown_severity_is_a_400(client) -> None:
    response = client.get(f"{API_PREFIX}/alerts?severity=CATASTROPHIC")
    assert response.status_code == 400
    assert response.json()["error"] == "alert_filter_invalid"


def test_paging_does_not_repeat_an_alert(client) -> None:
    first = client.get(f"{API_PREFIX}/alerts?limit=10").json()["alerts"]
    second = client.get(f"{API_PREFIX}/alerts?limit=10&offset=10").json()["alerts"]
    assert not ({a["alert_id"] for a in first} & {a["alert_id"] for a in second})


@pytest.fixture(scope="module")
def one_alert(client) -> dict:
    listing = client.get(f"{API_PREFIX}/alerts?limit=1").json()
    alert_id = listing["alerts"][0]["alert_id"]
    response = client.get(f"{API_PREFIX}/alerts/{alert_id}")
    assert response.status_code == 200, response.text
    return response.json()


def test_the_detail_response_answers_every_investigator_question(one_alert) -> None:
    """The ten requirements, as one structural assertion."""
    for block in ("summary", "risk", "why_flagged", "evidence",
                  "relationships", "timeline", "network_context", "members",
                  "provenance"):
        assert block in one_alert, block
    assert one_alert["risk"]["severity"] in contract.SEVERITIES
    assert set(one_alert["risk"]["aggregations"]) == set(contract.AGGREGATIONS)


def test_the_explanation_is_the_models_own_output(one_alert) -> None:
    why = one_alert["why_flagged"]
    assert "log-odds" in why["units"]
    member = why["per_member"][0]
    assert member["contributions"]
    for row in member["contributions"]:
        assert row["signal_category"] == contract.MODEL_SIGNAL
        assert row["value_category"] in contract.CATEGORIES
        assert row["feature_group"] in ("M0", "M1", "M2", "M3")


def test_the_evidence_block_names_each_groups_category(one_alert) -> None:
    groups = one_alert["evidence"]["groups"]
    assert set(groups) == {"M0", "M1", "M2", "M3"}
    assert groups["M3"]["category"] == contract.NETWORK_CONTEXT
    assert groups["M0"]["category"] == contract.BLOCKCHAIN_CONTEXT
    assert "unavailable" in groups["M0"]["members"][0]


def test_the_provenance_block_carries_the_full_chain(one_alert) -> None:
    p = one_alert["provenance"]
    assert p["provenance_type"] == "PRODUCTION"
    assert len(p["run_fingerprint"]) == 64
    assert p["phase6_dataset_fingerprint"]
    assert p["model"]["family"] == "LightGBM"
    assert "VALIDATION" in p["model"]["calibration"]
    assert "as-of" in p["feature_semantics"]
    assert p["scored_split"] == "test"
    assert p["severity_bands"]


def test_the_network_block_is_context_and_denies_ownership(one_alert) -> None:
    net = one_alert["network_context"]
    assert net["category"] == contract.NETWORK_CONTEXT
    assert net["status"] in (contract.NETWORK_CONTEXT, contract.INSUFFICIENT_EVIDENCE)
    assert "SYNTHETIC" in net["synthetic_warning"]
    assert "owns" in net["meaning"]


def test_any_ownership_language_in_a_response_is_negated(one_alert) -> None:
    """Property 3, tested as the property rather than as a word ban.

    A plain substring scan is the wrong check here: it flags the sentence
    that says "these observations do NOT establish that an IP address owns a
    wallet", which is the disclaimer protecting against the exact claim. So
    every occurrence of ownership language must sit inside a NEGATED
    construction. Deleting the disclaimer would leave the phrase unnegated
    and fail this; so would adding a real claim.
    """
    rendered = json.dumps(one_alert).lower()
    negators = ("not ", "never", "cannot", "no ", "denies", "without")
    for phrase in FORBIDDEN_CLAIMS:
        start = 0
        while (found := rendered.find(phrase, start)) != -1:
            window = rendered[max(0, found - 90):found]
            assert any(n in window for n in negators), (
                f"the response uses {phrase!r} without a negation in the "
                f"preceding 90 characters: ...{window[-90:]}[{phrase}]"
            )
            start = found + len(phrase)


def test_the_ownership_scan_would_catch_a_real_claim() -> None:
    """Guards the test above: it must not pass on anything.

    Without this, a scan that never matched would look identical to a scan
    that matched only negated text.
    """
    negators = ("not ", "never", "cannot", "no ", "denies", "without")
    claim = "this ip owns the wallet"
    window = claim[:claim.find("ip owns")]
    assert not any(n in window for n in negators), (
        "a bare ownership claim must not be treated as negated"
    )


def test_no_response_carries_a_label(one_alert, client) -> None:
    rendered = json.dumps(one_alert)
    for key in ('"y":', '"is_illicit"', '"truth"'):
        assert key not in rendered, f"response leaks {key}"
    listing = json.dumps(client.get(f"{API_PREFIX}/alerts?limit=50").json())
    for key in ('"y":', '"is_illicit"', '"truth"'):
        assert key not in listing


def test_relationships_expose_only_supported_kinds(one_alert) -> None:
    block = one_alert["relationships"]
    assert set(block["supported"]) == {contract.CO_SPEND, contract.FUNDED}
    for edge in block["edges"]:
        assert edge["relationship"] in contract.RELATIONSHIPS
        assert edge["category"] == contract.BLOCKCHAIN_CONTEXT
    assert "heuristic" in block["note"].lower()


def test_the_timeline_is_ordered_and_labelled(one_alert) -> None:
    points = one_alert["timeline"]["points"]
    assert one_alert["timeline"]["category"] == contract.BLOCKCHAIN_CONTEXT
    steps = [p["timestep"] for p in points]
    assert steps == sorted(steps)


def test_a_stale_alert_id_is_409_not_404(client, one_alert) -> None:
    cluster = one_alert["summary"]["cluster_id"]
    response = client.get(f"{API_PREFIX}/alerts/{'f' * 16}:{cluster}")
    assert response.status_code == 409
    assert response.json()["error"] == "alert_id_stale"


def test_an_unknown_cluster_is_404(client, one_alert) -> None:
    run = one_alert["run_fingerprint"]
    response = client.get(f"{API_PREFIX}/alerts/{run}:999999999")
    assert response.status_code == 404
    assert response.json()["error"] == "alert_not_found"


def test_a_malformed_id_is_400(client) -> None:
    response = client.get(f"{API_PREFIX}/alerts/nonsense")
    assert response.status_code == 400
    assert response.json()["error"] == "alert_id_invalid"


def test_a_torn_alert_artifact_is_refused(tmp_path) -> None:
    """The Phase 5.3-B identity check, applied to the alert layer."""
    for relative in [artifacts.ALERTS, *artifacts.ALERT_TABLES.values()]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DATA_ROOT / relative, target)
        shutil.copy2(
            str(DATA_ROOT / relative) + prov.META_SUFFIX,
            str(target) + prov.META_SUFFIX,
        )
    sidecar = Path(str(tmp_path / artifacts.ALERTS) + prov.META_SUFFIX)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    meta["run_fingerprint"] = "b" * 64
    sidecar.write_text(json.dumps(meta), encoding="utf-8")

    client = TestClient(create_app(tmp_path), raise_server_exceptions=False)
    response = client.get(f"{API_PREFIX}/alerts?limit=1")
    assert response.status_code == 500
    assert response.json()["error"] == "provenance_refused"
    assert "TORN PUBLISH" in response.json()["detail"]


def test_a_missing_alert_artifact_is_503_naming_the_command(tmp_path) -> None:
    client = TestClient(create_app(tmp_path), raise_server_exceptions=False)
    response = client.get(f"{API_PREFIX}/alerts")
    assert response.status_code == 503
    assert "phase7-alerts" in response.json()["detail"]
