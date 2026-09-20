"""ObsidianChain Conductor Orchestrator: End-to-end analytical pipeline coordinator.

Conducts the 17-stage forensic pipeline across existing components and adapters:
    1.  Ingest
    2.  Validate & Deduplicate
    3.  GeoIP / ASN Provider
    4.  Blockchain Analysis
    5.  Network Analysis
    6.  Blockchain ↔ Network Correlation
    7.  Blockchain Transaction Graph
    8.  Entity Clustering (Union-Find)
    9.  Features & Compatibility Check
    10. Supervised ML Risk (or MODEL_UNAVAILABLE_FOR_SCHEMA)
    11. Unsupervised Anomaly (MAD per Address)
    12. Peeling & Mixing Patterns
    13. Evidence Fusion
    14. Ranked Alerts
    15. Forensic Explanations
    16. Investigation Graph Projection
    17. Reporting & Run Integrity Manifest

Conducts without rewriting analytical engines. Preserves offline, deterministic execution.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from obsidianchain.alerts.graph import InvestigationGraph, project_investigation_graph
from obsidianchain.correlation.engine import CorrelationResult, correlate_blockchain_and_network
from obsidianchain.geoip import GeoIPProvider, OfflineCSVProvider, TestFixtureProvider
from obsidianchain.io import ingest
from obsidianchain.ml.anomaly import AnomalyDetectionResult, detect_address_anomalies
from obsidianchain.pipeline.alerts import AlertRunResult, build_alert_run
from obsidianchain.pipeline.blockchain import BlockchainGraph, ClusterResult, build_blockchain_layer
from obsidianchain.pipeline.features import (
    FeatureManifest,
    MlStageResult,
    derive_canonical_address_features,
    execute_supervised_stage,
)
from obsidianchain.pipeline.patterns import (
    MixingResult,
    PeelingResult,
    detect_mixing_patterns,
    detect_peeling_chains,
)


@dataclass
class StageExecutionRecord:
    stage_number: int
    stage_name: str
    status: str
    """'SUCCESS', 'SKIPPED', 'DEGRADED', or 'FAILED'."""
    duration_seconds: float
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRunOutcome:
    """The complete result of an orchestrated pipeline run."""

    run_id: str
    run_dir: Path
    stages: list[StageExecutionRecord]
    validation_report: ingest.ValidationReport
    blockchain_graph: BlockchainGraph
    cluster_result: ClusterResult
    correlation_result: CorrelationResult
    feature_manifest: FeatureManifest
    ml_result: MlStageResult
    anomaly_result: AnomalyDetectionResult
    peeling_result: PeelingResult
    mixing_result: MixingResult
    alert_result: AlertRunResult
    investigation_graph: InvestigationGraph
    manifest: dict[str, Any]

    @property
    def is_success(self) -> bool:
        return all(s.status in ("SUCCESS", "DEGRADED") for s in self.stages)


def run_pipeline(
    input_path: str | Path,
    runs_dir: str | Path | None = None,
    run_id: str | None = None,
    model: Any | None = None,
    geoip_provider: GeoIPProvider | None = None,
    declared_format: str | None = None,
    stage_callback: Any | None = None,
) -> PipelineRunOutcome:
    """Execute the full 17-stage analytical pipeline on a canonical capture file."""
    input_file = Path(input_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input dataset file not found: {input_path}")

    # Deterministic or timestamped run ID
    t_start = datetime.datetime.now(datetime.timezone.utc)
    if run_id is None:
        input_hash = hashlib.sha256(input_file.read_bytes()).hexdigest()[:8]
        run_id = f"run_{t_start.strftime('%Y%m%d_%H%M%S')}_{input_hash}"

    base_runs_dir = Path(runs_dir) if runs_dir is not None else Path("data") / "runs"
    run_dir = base_runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stages: list[StageExecutionRecord] = []
    input_bytes = input_file.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()

    def _add_stage(rec: StageExecutionRecord) -> None:
        stages.append(rec)
        if stage_callback is not None:
            try:
                stage_callback(rec.stage_number, rec.stage_name, rec.status, rec.summary)
            except Exception:
                pass

    # 1. Ingest & 2. Validate + Deduplicate
    t0 = datetime.datetime.now(datetime.timezone.utc)
    frame, report = ingest.ingest(input_file, declared_format=declared_format)
    dt1 = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    _add_stage(StageExecutionRecord(
        stage_number=1, stage_name="Ingest",
        status="SUCCESS", duration_seconds=dt1,
        summary={"source_format": report.source_format, "rows_read": report.rows_read},
    ))
    _add_stage(StageExecutionRecord(
        stage_number=2, stage_name="Validate & Deduplicate",
        status="SUCCESS" if report.ok else "FAILED", duration_seconds=dt1,
        summary={
            "rows_valid": report.rows_valid,
            "exact_duplicates_rejected": report.exact_duplicates_rejected,
            "network_observations_preserved": report.network_observations_preserved,
            "errors": report.errors[:5],
            "warnings": report.warnings[:5],
        },
    ))

    # 3. GeoIP / ASN Provider
    t0 = datetime.datetime.now(datetime.timezone.utc)
    if geoip_provider is None:
        geoip_provider = OfflineCSVProvider()
    dt3 = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    _add_stage(StageExecutionRecord(
        stage_number=3, stage_name="GeoIP / ASN",
        status="SUCCESS", duration_seconds=dt3,
        summary={"provider_version": geoip_provider.version, "provider_sha256": geoip_provider.sha256},
    ))

    # 4. Blockchain Analysis & 7. Blockchain Transaction Graph & 8. Entity Clustering
    t0 = datetime.datetime.now(datetime.timezone.utc)
    bg, clusters = build_blockchain_layer(frame)
    dt_bc = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    _add_stage(StageExecutionRecord(
        stage_number=4, stage_name="Blockchain Analysis",
        status="SUCCESS", duration_seconds=dt_bc,
        summary={"n_transactions": bg.n_transactions, "n_addresses": bg.n_addresses},
    ))

    # 5. Network Analysis & 6. Blockchain <-> Network Correlation
    t0 = datetime.datetime.now(datetime.timezone.utc)
    corr = correlate_blockchain_and_network(frame, bg)
    dt_corr = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    _add_stage(StageExecutionRecord(
        stage_number=5, stage_name="Network Analysis",
        status="SUCCESS", duration_seconds=dt_corr,
        summary={"total_observations": len(corr.all_observations)},
    ))
    _add_stage(StageExecutionRecord(
        stage_number=6, stage_name="Blockchain ↔ Network Correlation",
        status="SUCCESS", duration_seconds=dt_corr,
        summary=corr.summary(),
    ))

    # Record 7 & 8 stages in presentation record
    _add_stage(StageExecutionRecord(
        stage_number=7, stage_name="Entity / Transaction Graph",
        status="SUCCESS", duration_seconds=dt_bc,
        summary={"n_nodes": bg.n_transactions + bg.n_addresses, "n_edges": bg.n_edges},
    ))
    _add_stage(StageExecutionRecord(
        stage_number=8, stage_name="Entity Clustering",
        status="SUCCESS", duration_seconds=dt_bc,
        summary=clusters.summary(),
    ))

    # 9. Features
    t0 = datetime.datetime.now(datetime.timezone.utc)
    address_features = derive_canonical_address_features(frame)
    from obsidianchain.pipeline.features_ps import extract_ps_features
    ps_features = extract_ps_features(frame)
    ps_addr_features = (
        ps_features.sort_values(by=["timestamp"], ascending=True)
        .groupby("address")
        .last()
        .reset_index()
    ) if not ps_features.empty else pd.DataFrame()

    dt_feats = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    _add_stage(StageExecutionRecord(
        stage_number=9, stage_name="Features",
        status="SUCCESS", duration_seconds=dt_feats,
        summary={
            "features_derived_count": len(address_features.columns) - 1,
            "ps_features_derived_count": len(ps_features.columns) - 3 if not ps_features.empty else 0,
            "addresses_scored": len(address_features),
        },
    ))

    # 10. Supervised ML Risk
    t0 = datetime.datetime.now(datetime.timezone.utc)
    active_model = model
    feats_to_use = address_features

    if active_model is None:
        ps_model_path = Path("data") / "models" / "ps_native" / "v1" / "model.joblib"
        if ps_model_path.is_file():
            try:
                from obsidianchain.ml.ps_model import PsNativeRiskModel
                active_model = PsNativeRiskModel.load(ps_model_path.parent)
                feats_to_use = ps_addr_features
            except Exception:
                active_model = None
    elif hasattr(active_model, "features") and not ps_addr_features.empty:
        if all(f in ps_addr_features.columns for f in active_model.features):
            feats_to_use = ps_addr_features

    ml_result = execute_supervised_stage(active_model, feats_to_use)
    dt_ml = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    ml_status = "SUCCESS" if ml_result.is_available else "DEGRADED"
    _add_stage(StageExecutionRecord(
        stage_number=10, stage_name="Supervised ML Risk",
        status=ml_status, duration_seconds=dt_ml,
        summary={
            "status": ml_result.status,
            "model_version": getattr(active_model, "version", "none") if active_model else "none",
            "is_compatible": ml_result.manifest.is_compatible,
            "reason": ml_result.reason,
            "missing_features_count": len(ml_result.manifest.missing_features),
        },
    ))

    # 11. Unsupervised Anomaly (MAD per Address)
    t0 = datetime.datetime.now(datetime.timezone.utc)
    anom_result = detect_address_anomalies(address_features, cluster_result=clusters)
    dt_anom = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    _add_stage(StageExecutionRecord(
        stage_number=11, stage_name="Unsupervised Anomaly",
        status="SUCCESS", duration_seconds=dt_anom,
        summary={
            "addresses_evaluated": len(anom_result.addresses),
            "clusters_aggregated": len(anom_result.clusters),
        },
    ))

    # 12. Peeling / Mixing Patterns
    t0 = datetime.datetime.now(datetime.timezone.utc)
    peel_result = detect_peeling_chains(bg)
    mix_result = detect_mixing_patterns(bg)
    dt_patt = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    _add_stage(StageExecutionRecord(
        stage_number=12, stage_name="Peeling / Mixing Patterns",
        status="SUCCESS", duration_seconds=dt_patt,
        summary={
            "peeling_chains_detected": peel_result.detected_count,
            "mixing_patterns_detected": mix_result.pattern_count,
            "mixing_likelihoods_detected": mix_result.likelihood_count,
        },
    ))

    # 13. Evidence Fusion & 14. Ranked Alerts & 15. Explanation
    t0 = datetime.datetime.now(datetime.timezone.utc)
    alert_result = build_alert_run(
        run_id=run_id,
        blockchain_graph=bg,
        cluster_result=clusters,
        correlation_result=corr,
        anomaly_result=anom_result,
        ml_result=ml_result,
        peeling_result=peel_result,
        mixing_result=mix_result,
        geoip_provider=geoip_provider,
    )
    dt_alerts = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    _add_stage(StageExecutionRecord(
        stage_number=13, stage_name="Evidence Fusion",
        status="SUCCESS", duration_seconds=dt_alerts,
        summary={"clusters_fused": len(alert_result.alerts)},
    ))
    _add_stage(StageExecutionRecord(
        stage_number=14, stage_name="Ranked Alerts",
        status="SUCCESS", duration_seconds=dt_alerts,
        summary={
            "total_ranked_alerts": alert_result.total_alerts,
            "critical_count": sum(1 for a in alert_result.alerts if a.severity == "CRITICAL"),
            "high_count": sum(1 for a in alert_result.alerts if a.severity == "HIGH"),
        },
    ))
    _add_stage(StageExecutionRecord(
        stage_number=15, stage_name="Explanation",
        status="SUCCESS", duration_seconds=dt_alerts,
        summary={"explanations_generated": sum(len(a.evidence) for a in alert_result.alerts)},
    ))

    # 16. Investigation Graph
    t0 = datetime.datetime.now(datetime.timezone.utc)
    inv_graph = project_investigation_graph(
        blockchain_graph=bg,
        cluster_result=clusters,
        correlation_result=corr,
        anomaly_result=anom_result,
        peeling_result=peel_result,
        mixing_result=mix_result,
    )
    dt_graph = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    _add_stage(StageExecutionRecord(
        stage_number=16, stage_name="Investigation Graph",
        status="SUCCESS", duration_seconds=dt_graph,
        summary=inv_graph.as_dict(),
    ))

    # 17. Reporting & Integrity (Atomic Artifact Publishing)
    t0 = datetime.datetime.now(datetime.timezone.utc)

    # Serialize alerts
    alerts_file = run_dir / "alerts.json"
    alerts_payload = {
        "run_id": run_id,
        "total_alerts": alert_result.total_alerts,
        "alerts": [a.as_dict() for a in alert_result.alerts],
    }
    alerts_bytes = json.dumps(alerts_payload, indent=2).encode("utf-8")
    alerts_file.write_bytes(alerts_bytes)
    alerts_sha = hashlib.sha256(alerts_bytes).hexdigest()

    # Serialize investigation graph
    graph_file = run_dir / "investigation_graph.json"
    graph_bytes = json.dumps(inv_graph.as_dict(), indent=2).encode("utf-8")
    graph_file.write_bytes(graph_bytes)
    graph_sha = hashlib.sha256(graph_bytes).hexdigest()

    # Serialize validation report
    val_file = run_dir / "validation_report.json"
    val_bytes = json.dumps(report.as_dict(), indent=2).encode("utf-8")
    val_file.write_bytes(val_bytes)
    val_sha = hashlib.sha256(val_bytes).hexdigest()

    dt_pub = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    manifest_file = run_dir / "manifest.json"

    _add_stage(StageExecutionRecord(
        stage_number=17, stage_name="Reporting & Integrity",
        status="SUCCESS", duration_seconds=dt_pub,
        summary={"manifest_path": str(manifest_file), "artifacts_published": 4},
    ))

    # Assemble complete run manifest
    t_end = datetime.datetime.now(datetime.timezone.utc)
    total_sec = (t_end - t_start).total_seconds()

    manifest = {
        "run_id": run_id,
        "created_at_utc": t_start.isoformat(),
        "completed_at_utc": t_end.isoformat(),
        "total_duration_seconds": round(total_sec, 4),
        "input_dataset": {
            "path": str(input_file.resolve()),
            "format": report.source_format,
            "sha256": input_sha256,
            "size_bytes": len(input_bytes),
        },
        "provenance": {
            "orchestrator_version": "obsidianchain-conductor-v2",
            "geoip_provider": geoip_provider.version,
            "geoip_provider_sha256": geoip_provider.sha256,
            "ml_status": ml_result.status,
            "ml_incompatibility_reason": ml_result.reason,
        },
        "artifacts": {
            "alerts.json": alerts_sha,
            "investigation_graph.json": graph_sha,
            "validation_report.json": val_sha,
        },
        "stages": [
            {
                "stage_number": s.stage_number,
                "stage_name": s.stage_name,
                "status": s.status,
                "duration_seconds": round(s.duration_seconds, 4),
                "summary": s.summary,
            }
            for s in stages
        ],
    }

    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    manifest_file.write_bytes(manifest_bytes)
    manifest["artifacts"]["manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()


    return PipelineRunOutcome(
        run_id=run_id,
        run_dir=run_dir,
        stages=stages,
        validation_report=report,
        blockchain_graph=bg,
        cluster_result=clusters,
        correlation_result=corr,
        feature_manifest=ml_result.manifest,
        ml_result=ml_result,
        anomaly_result=anom_result,
        peeling_result=peel_result,
        mixing_result=mix_result,
        alert_result=alert_result,
        investigation_graph=inv_graph,
        manifest=manifest,
    )
