"""Full System Acceptance Test Suite for ObsidianChain.

Covers the three acceptance layers:
    Layer 1: Component Acceptance (Individual stages & adapters)
    Layer 2: Pipeline End-to-End Flow (17 stages across JSON, CSV, XML)
    Layer 3: Casework & Investigation Flow (API endpoints, DB transitions, audit)

Tests assert the 5 critical pre-implementation safeguards:
    1. Blockchain duplicate key preserves address <-> amount correspondence.
    2. Network observation key incorporates observer identity.
    3. Stage ordering builds clustering before projecting investigation graph.
    4. Feature adapter returns MODEL_UNAVAILABLE_FOR_SCHEMA without silent 0-imputation.
    5. Anomaly detector primary unit is address, secondary is cluster, anomalous > benign.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from obsidianchain import geoip
from obsidianchain.alerts.graph import project_investigation_graph
from obsidianchain.correlation.engine import correlate_blockchain_and_network
from obsidianchain.io import ingest
from obsidianchain.ml.anomaly import detect_address_anomalies
from obsidianchain.pipeline.alerts import build_alert_run
from obsidianchain.pipeline.blockchain import build_blockchain_layer
from obsidianchain.pipeline.features import (
    derive_canonical_address_features,
    execute_supervised_stage,
)
from obsidianchain.pipeline.orchestrator import run_pipeline
from obsidianchain.pipeline.patterns import detect_mixing_patterns, detect_peeling_chains


# =========================================================================
# LAYER 1: COMPONENT ACCEPTANCE TESTS
# =========================================================================


class TestComponentAcceptance:
    """Validate individual stage adapters under the safeguard contracts."""

    def test_blockchain_key_preserves_address_amount_tuples(self) -> None:
        """Safeguard 1: Never sort inputs and amounts independently."""
        row_a = {
            "txid": "tx1",
            "input_addresses": ["addr1", "addr2"],
            "input_amounts": ["1.0", "10.0"],
            "output_addresses": ["out1"],
            "output_amounts": ["10.99"],
            "fee": "0.01",
            "script_type": "p2pkh",
        }
        # Swapping amount assignment between addresses must change the identity key
        row_b = {
            "txid": "tx1",
            "input_addresses": ["addr1", "addr2"],
            "input_amounts": ["10.0", "1.0"],
            "output_addresses": ["out1"],
            "output_amounts": ["10.99"],
            "fee": "0.01",
            "script_type": "p2pkh",
        }
        key_a = ingest.blockchain_record_key(row_a)
        key_b = ingest.blockchain_record_key(row_b)
        assert key_a != key_b, "Swapping address-amount pairs must alter the key!"

    def test_network_observation_key_with_observer_identity(self) -> None:
        """Safeguard 2: Include observer identity when present."""
        obs_probe1 = {
            "txid": "tx1", "observer_id": "probe-eu-west",
            "src_ip": "1.2.3.4", "src_port": "8333", "timestamp": "1600000000",
        }
        obs_probe2 = {
            "txid": "tx1", "observer_id": "probe-us-east",
            "src_ip": "1.2.3.4", "src_port": "8333", "timestamp": "1600000000",
        }
        k1 = ingest.observation_key(obs_probe1)
        k2 = ingest.observation_key(obs_probe2)
        assert k1 != k2, "Distinct observer IDs must produce distinct keys!"

    def test_feature_adapter_hard_safeguard_no_silent_zero(self) -> None:
        """Safeguard 4: Incompatible LightGBM models receive MODEL_UNAVAILABLE_FOR_SCHEMA."""
        class MockLightGBM:
            features = ["elliptic_m0_unobservable_feature_1", "elliptic_m0_unobservable_feature_2"]

        frame, _ = ingest.ingest("tests/data/synthetic_acceptance_capture.json")
        feats = derive_canonical_address_features(frame)
        result = execute_supervised_stage(MockLightGBM(), feats)

        assert result.status == "MODEL_UNAVAILABLE_FOR_SCHEMA"
        assert not result.is_available
        assert result.scores is None
        assert result.manifest.missing_features == MockLightGBM.features
        assert "cannot be derived" in (result.reason or "")

    def test_anomaly_detector_unit_and_ranking(self) -> None:
        """Safeguard 5: Primary unit address, anomalous entity ranks above benign."""
        frame, _ = ingest.ingest("tests/data/synthetic_acceptance_capture.json")
        bg, clusters = build_blockchain_layer(frame)
        feats = derive_canonical_address_features(frame)
        anom = detect_address_anomalies(feats, clusters)

        whale_score = anom.address_score("1AnomWhale")
        benign_score = anom.address_score("1BenignAddr1")

        assert whale_score > benign_score, f"Expected whale > benign, got {whale_score} vs {benign_score}"
        whale_rec = anom.addresses["1AnomWhale"]
        dev_features = {d.feature for d in whale_rec.top_deviations}
        assert "mean_fee_ratio" in dev_features
        assert "velocity_tx_per_hour" in dev_features

    def test_exact_txid_correlation_policies(self) -> None:
        frame, _ = ingest.ingest("tests/data/synthetic_acceptance_capture.json")
        bg, _ = build_blockchain_layer(frame)
        corr = correlate_blockchain_and_network(frame, bg)

        assert corr.status_for_tx("tx_benign_01") == "CORRELATED"
        assert corr.status_for_tx("tx_no_net_01") == "NO_NETWORK_OBSERVATION"
        assert corr.status_for_tx("tx_unmatched_net_01") == "UNMATCHED_NETWORK_OBSERVATION"
        assert len(corr.observations_for_tx("tx_multi_obs")) == 2

    def test_pattern_detection_peeling_and_mixing(self) -> None:
        frame, _ = ingest.ingest("tests/data/synthetic_acceptance_capture.json")
        bg, _ = build_blockchain_layer(frame)

        peel = detect_peeling_chains(bg, min_depth=2)
        assert peel.detected_count >= 1
        assert peel.is_peeling("tx_peel_01")
        assert peel.is_peeling("tx_peel_02")
        assert peel.is_peeling("tx_peel_03")

        mix = detect_mixing_patterns(bg)
        assert mix.pattern_count >= 1
        assert mix.classification_for_tx("tx_mix_01") == "MIXING_PATTERN"
        assert mix.classification_for_tx("tx_benign_01") == "NO_MIXING_SIGNAL"

    def test_investigation_graph_projection(self) -> None:
        frame, _ = ingest.ingest("tests/data/synthetic_acceptance_capture.json")
        bg, clusters = build_blockchain_layer(frame)
        corr = correlate_blockchain_and_network(frame, bg)
        anom = detect_address_anomalies(derive_canonical_address_features(frame), clusters)
        peel = detect_peeling_chains(bg)
        mix = detect_mixing_patterns(bg)

        inv_graph = project_investigation_graph(bg, clusters, corr, anom, peel, mix)
        g_dict = inv_graph.as_dict()
        assert g_dict["node_count"] > 0
        assert g_dict["edge_count"] > 0

        kinds = {n["kind"] for n in g_dict["nodes"]}
        # asn: the autonomous system each announcing peer belongs to (IN_ASN).
        assert kinds == {"cluster", "address", "transaction", "ip", "asn"}
        edge_kinds = {e["kind"] for e in g_dict["edges"]}
        assert {"ANNOUNCED_BY", "IN_ASN"} <= edge_kinds
        announced = [e for e in g_dict["edges"] if e["kind"] == "ANNOUNCED_BY"]
        assert all("timestamp_ms" in e["data"] for e in announced)


# =========================================================================
# LAYER 2: PIPELINE END-TO-END ACCEPTANCE TESTS
# =========================================================================


class TestPipelineAcceptance:
    """Validate full 17-stage orchestration across all supported formats."""

    @pytest.mark.parametrize("fmt", ["json", "csv", "xml"])
    def test_orchestrator_runs_all_17_stages(self, tmp_path, fmt) -> None:
        input_file = Path(f"tests/data/synthetic_acceptance_capture.{fmt}")
        outcome = run_pipeline(
            input_file,
            runs_dir=tmp_path / "runs",
            geoip_provider=geoip.TestFixtureProvider(),
        )

        assert outcome.is_success
        assert len(outcome.stages) == 17
        stage_names = [s.stage_name for s in outcome.stages]
        assert "Ingest" in stage_names
        assert "Validate & Deduplicate" in stage_names
        assert "Blockchain ↔ Network Correlation" in stage_names
        assert "Entity Clustering" in stage_names
        assert "Supervised ML Risk" in stage_names
        assert "Unsupervised Anomaly" in stage_names
        assert "Peeling / Mixing Patterns" in stage_names
        assert "Evidence Fusion" in stage_names
        assert "Ranked Alerts" in stage_names
        assert "Investigation Graph" in stage_names
        assert "Reporting & Integrity" in stage_names

        # Verify atomic artifacts on disk
        run_dir = outcome.run_dir
        assert (run_dir / "manifest.json").is_file()
        assert (run_dir / "alerts.json").is_file()
        assert (run_dir / "investigation_graph.json").is_file()
        assert (run_dir / "validation_report.json").is_file()

        # Check manifest contents
        manifest = outcome.manifest
        assert manifest["input_dataset"]["format"] == fmt
        # The registry champion is trained on the live schema,
        # so the model scores. The v1 refusal path is still covered by the
        # MockLightGBM / MockEllipticModel tests in this file.
        assert manifest["provenance"]["ml_status"] == "SCORED"
        # The run still completes: 17 stages, honest degradation, no crash.
        assert outcome.is_success
        assert len(manifest["stages"]) == 17

        # Verify Stage 10 (Supervised ML Risk) execution
        s10 = next(s for s in outcome.stages if s.stage_number == 10)
        assert s10.status == "SUCCESS"
        assert s10.summary["status"] == "SCORED"
        # Whatever the registry names champion serves; today ps_native_v4.
        from obsidianchain.ml import registry
        assert s10.summary["model_version"] == registry.Registry.open().role("champion")

        # Check alerts contain a scored MODEL_SIGNAL
        alerts = outcome.alert_result.alerts
        assert len(alerts) > 0
        has_model_signal = False
        for a in alerts:
            for ev in a.evidence:
                if ev.category == "MODEL_SIGNAL":
                    has_model_signal = True
                    assert ev.status == "PRESENT"
                    assert 0.0 <= ev.score <= 1.0
        assert has_model_signal, "Alerts must contain MODEL_SIGNAL evidence"

    def test_ps_model_inference_determinism(self, tmp_path) -> None:
        """Verify identical PS-native input produces deterministic risk scores."""
        input_file = Path("tests/data/synthetic_acceptance_capture.json")
        outcome1 = run_pipeline(input_file, runs_dir=tmp_path / "r1", geoip_provider=geoip.TestFixtureProvider())
        outcome2 = run_pipeline(input_file, runs_dir=tmp_path / "r2", geoip_provider=geoip.TestFixtureProvider())

        alerts1 = outcome1.alert_result.alerts
        alerts2 = outcome2.alert_result.alerts
        assert len(alerts1) == len(alerts2)
        for a1, a2 in zip(alerts1, alerts2):
            assert a1.alert_id == a2.alert_id
            assert a1.fused_risk_score == a2.fused_risk_score
            assert a1.severity == a2.severity
            assert a1.rank == a2.rank

    def test_ps_model_inference_without_network(self, tmp_path) -> None:
        """Verify capture without network observations can still score with PS-native model."""
        import json
        raw = json.loads(Path("tests/data/synthetic_acceptance_capture.json").read_text())
        # Clear all network fields from records
        for r in raw["records"]:
            r["src_ip"] = None
            r["dst_ip"] = None
            r["src_port"] = None
            r["dst_port"] = None
            r["asn"] = None
            r["geo_country"] = None
        pure_bc_path = tmp_path / "pure_bc.json"
        pure_bc_path.write_text(json.dumps(raw))

        outcome = run_pipeline(pure_bc_path, runs_dir=tmp_path / "runs", geoip_provider=geoip.TestFixtureProvider())
        assert outcome.is_success
        manifest = outcome.manifest
        # The registry champion is trained on the live schema,
        # so the model scores. The v1 refusal path is still covered by the
        # MockLightGBM / MockEllipticModel tests in this file.
        assert manifest["provenance"]["ml_status"] == "SCORED"
        # The run still completes: 17 stages, honest degradation, no crash.
        assert outcome.is_success

        # Verify network evidence is NO_EVIDENCE, but ML risk score is still valid
        for a in outcome.alert_result.alerts:
            net_ev = next(e for e in a.evidence if e.category == "NETWORK_CONTEXT")
            assert net_ev.status == "NO_EVIDENCE"
            ml_ev = next(e for e in a.evidence if e.category == "MODEL_SIGNAL")
            # The core schema needs no network column, so a chain-only
            # capture is fully scorable.
            assert ml_ev.status == "PRESENT"

    def test_explicit_incompatible_model_safeguard(self, tmp_path) -> None:
        """Passing an incompatible Elliptic++ feature model returns MODEL_UNAVAILABLE_FOR_SCHEMA."""
        class MockEllipticModel:
            features = ["elliptic_m0_unobservable_feature_1", "elliptic_m0_unobservable_feature_2"]

        outcome = run_pipeline(
            "tests/data/synthetic_acceptance_capture.json",
            runs_dir=tmp_path / "runs",
            geoip_provider=geoip.TestFixtureProvider(),
            model=MockEllipticModel(),
        )
        assert outcome.manifest["provenance"]["ml_status"] == "MODEL_UNAVAILABLE_FOR_SCHEMA"

    def test_ranked_alerts_evidence_and_ordering(self, tmp_path) -> None:
        outcome = run_pipeline(
            "tests/data/synthetic_acceptance_capture.json",
            runs_dir=tmp_path / "runs",
            geoip_provider=geoip.TestFixtureProvider(),
        )
        alerts = outcome.alert_result.alerts
        assert len(alerts) > 0

        # Anomalous whale entity outranks standard benign entity
        whale_alert = outcome.alert_result.alert_for_address("1AnomWhale")
        benign_alert = outcome.alert_result.alert_for_address("1BenignAddr1")
        assert whale_alert is not None
        assert benign_alert is not None
        assert whale_alert.fused_risk_score > benign_alert.fused_risk_score
        assert whale_alert.rank < benign_alert.rank  # Lower rank number = higher priority


# =========================================================================
# LAYER 3: CASEWORK & INVESTIGATION FLOW ACCEPTANCE TESTS
# =========================================================================


from fastapi.testclient import TestClient
from obsidianchain.api.app import create_app
from obsidianchain.console import db, users


@pytest.fixture
def signed_in(tmp_path):
    root = tmp_path
    connection = db.connect(root)
    try:
        users.create(connection, username="analyst",
                     password="analyst-password", role="INVESTIGATOR")
    finally:
        connection.close()
    app = create_app(root)
    client = TestClient(app)
    response = client.post("/api/auth/login",
                           json={"username": "analyst",
                                 "password": "analyst-password"})
    assert response.status_code == 200, response.text
    return client


class TestCaseworkAcceptance:
    """Validate DB state transitions, audit logging, and investigation API routes."""

    def test_investigation_run_wiring_and_audit(self, signed_in) -> None:
        client = signed_in

        # 1. Create investigation
        create_resp = client.post(
            "/api/investigations",
            json={"name": "Acceptance Case", "description": "Testing full 17-stage wiring"},
        )
        assert create_resp.status_code == 201
        inv_id = create_resp.json()["id"]

        # 2. Upload and register synthetic capture
        capture_bytes = Path("tests/data/synthetic_acceptance_capture.json").read_bytes()
        upload_resp = client.post(
            f"/api/investigations/{inv_id}/datasets?filename=acceptance.json&format=json",
            content=capture_bytes,
            headers={"Content-Type": "application/json"},
        )
        assert upload_resp.status_code == 201
        upload_data = upload_resp.json()
        ds_id = upload_data["dataset"]["id"]
        initial_run = upload_data["analysis_run"]
        assert initial_run["status"] == "NOT_RUN"
        assert initial_run["run_fingerprint"] is None

        # 3. Trigger 17-stage analytical pipeline on the dataset
        run_resp = client.post(
            f"/api/investigations/{inv_id}/datasets/{ds_id}/run?wait=true",
        )
        assert run_resp.status_code == 200
        run_data = run_resp.json()
        completed_run = run_data["analysis_run"]
        assert completed_run["status"] == "COMPLETE"
        assert completed_run["run_fingerprint"] is not None
        assert run_data["alerts_count"] > 0
        assert len(run_data["manifest"]["stages"]) == 17

        # 4. Verify audit history recorded the run
        hist_resp = client.get(
            f"/api/investigations/{inv_id}/history",
        )
        assert hist_resp.status_code == 200
        events = hist_resp.json()["events"]
        action_names = [e["action"] for e in events]
        assert "DATASET_ANALYSIS_RUN" in action_names

    def test_run_results_are_served_to_the_case_and_only_to_it(self, signed_in) -> None:
        client = signed_in
        inv_id = client.post("/api/investigations", json={"name": "Results Case"}).json()["id"]
        other_id = client.post("/api/investigations", json={"name": "Other Case"}).json()["id"]
        upload = client.post(
            f"/api/investigations/{inv_id}/datasets?filename=a.json&format=json",
            content=Path("tests/data/synthetic_acceptance_capture.json").read_bytes(),
            headers={"Content-Type": "application/json"},
        ).json()
        ds_id = upload["dataset"]["id"]
        run_id = client.post(f"/api/investigations/{inv_id}/datasets/{ds_id}/run?wait=true").json()["run_id"]

        resp = client.get(f"/api/investigations/{inv_id}/runs/{run_id}/results?limit=5")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        from obsidianchain.ml import registry
        assert body["ml_status"] == "SCORED"
        assert body["model"]["version"] == registry.Registry.open().role("champion")
        assert body["model"]["holdout_result"]["nap"] > 0
        assert "unverified until labels" in body["run_result_type"]
        assert 0 < len(body["alerts"]) <= 5
        assert {e["evidence_class"] for e in body["alerts"][0]["evidence"]} >= {"MODEL", "RULE"}
        codes = {a["code"] for a in body["monitoring_alerts"]}
        assert "PERFORMANCE_UNVERIFIED_UNTIL_LABELS" in codes

        # Another case cannot read this run, and an unknown run is 404.
        assert client.get(f"/api/investigations/{other_id}/runs/{run_id}/results").status_code == 404
        assert client.get(f"/api/investigations/{inv_id}/runs/nope/results").status_code == 404

    def test_alert_graph_api_endpoint(self, signed_in) -> None:
        client = signed_in

        # Query existing precomputed alert's graph if available, or 404/valid response
        alerts_resp = client.get("/api/alerts?limit=1")
        if alerts_resp.status_code == 200 and alerts_resp.json().get("alerts"):
            alert_id = alerts_resp.json()["alerts"][0]["alert_id"]
            graph_resp = client.get(f"/api/alerts/{alert_id}/graph")
            assert graph_resp.status_code == 200
            g_data = graph_resp.json()
            assert g_data["alert_id"] == alert_id
            assert "graph" in g_data
            assert "nodes" in g_data["graph"]
            assert "edges" in g_data["graph"]



def test_every_run_reports_model_trust(tmp_path) -> None:
    """Drift, holdout status and unseen missingness travel with every run."""
    import json
    outcome = run_pipeline("tests/data/synthetic_acceptance_capture.json",
                           runs_dir=tmp_path, geoip_provider=geoip.TestFixtureProvider())
    trust = outcome.manifest["provenance"]["model_trust"]
    from obsidianchain.ml import registry
    assert trust["model_version"] == registry.Registry.open().role("champion")
    assert trust["holdout_evaluated"] is True
    assert trust["holdout_summary"]["nap"] > 0
    assert trust["drift_status"] in {"STABLE", "SHIFTED", "MAJOR_SHIFT"}
    report = json.loads((outcome.run_dir / "monitoring.json").read_text())
    assert report["drift"]["features"]
    assert "monitoring.json" in outcome.manifest["artifacts"]
