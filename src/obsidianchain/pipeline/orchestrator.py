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


class CaptureContractError(ValueError):
    """The capture violates its contract badly enough to be refused whole."""


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
    seed_addresses: Any | None = None,
    registry_root: str | Path | None = None,
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
    from obsidianchain.contracts.capture import enforce_capture_contract
    frame, capture_contract = enforce_capture_contract(frame)
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
            "capture_contract": capture_contract.as_dict(),
        },
    ))
    if not capture_contract.ok:
        raise CaptureContractError(
            f"capture refused by {capture_contract.version}: "
            f"{len(capture_contract.quarantined)} of {capture_contract.transactions} "
            f"transactions violate the contract ({capture_contract.as_dict()['quarantine_kinds']})"
        )

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
    # Graph embeddings: cross-cluster link SUGGESTIONS beside co-spend
    # clustering. Nothing is merged on the strength of a similarity.
    from obsidianchain.ml.embeddings import embed, flows_from_capture, suggest_links
    t0 = datetime.datetime.now(datetime.timezone.utc)
    flows = flows_from_capture(frame)
    tx_counts: dict[str, int] = {}
    for fact in bg.transactions.values():
        for a in {x for x, _ in fact.inputs} | {x for x, _ in fact.outputs}:
            tx_counts[a] = tx_counts.get(a, 0) + 1
    link_suggestions = suggest_links(
        embed(flows), clusters.address_to_cluster, tx_counts=tx_counts,
    ) if not flows.empty else None
    dt_emb = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    _add_stage(StageExecutionRecord(
        stage_number=8, stage_name="Entity Clustering",
        status="SUCCESS", duration_seconds=dt_bc + dt_emb,
        summary={
            **clusters.summary(),
            "embedding_link_suggestions": (
                link_suggestions.summary() if link_suggestions else {"status": "NO_FLOWS"}
            ),
        },
    ))

    # 9. Features
    t0 = datetime.datetime.now(datetime.timezone.utc)
    address_features = derive_canonical_address_features(frame)
    from obsidianchain.pipeline.features_ps import extract_ps_features, last_snapshot_per_address
    ps_features = extract_ps_features(frame)
    ps_addr_features = last_snapshot_per_address(ps_features) if not ps_features.empty else pd.DataFrame()

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

    model_load_report: dict[str, Any] = {"source": "caller" if active_model is not None else "registry"}
    if active_model is None:
        # Serving goes through the registry only: champion, then fallback,
        # each verified against its registered hashes and required to declare
        # the engine's live feature schema. Never a directory scan, which
        # would serve any artifact whose column NAMES happen to match.
        active_model, model_load_report = _load_serving_model(registry_root)
        if active_model is not None:
            feats_to_use = ps_addr_features
    elif hasattr(active_model, "features") and not ps_addr_features.empty:
        if all(f in ps_addr_features.columns for f in active_model.features):
            feats_to_use = ps_addr_features

    t_score = datetime.datetime.now(datetime.timezone.utc)
    ml_result = execute_supervised_stage(active_model, feats_to_use)
    score_seconds = (datetime.datetime.now(datetime.timezone.utc) - t_score).total_seconds()
    if not ml_result.is_available and model_load_report.get("failures"):
        ml_result.reason = f"{ml_result.reason} | model load: {model_load_report['failures']}"
    # Feature contract: values outside the catalog never reach a score.
    feature_contract = None
    if ml_result.is_available and hasattr(active_model, "features"):
        from obsidianchain.contracts.features import validate_feature_frame
        feature_contract = validate_feature_frame(feats_to_use, list(active_model.features))
        if not feature_contract.ok:
            ml_result = MlStageResult(
                status="FEATURE_CONTRACT_VIOLATION", manifest=ml_result.manifest, scores=None,
                reason=f"{len(feature_contract.violations)} feature contract violation(s): "
                       f"{feature_contract.violations[:3]}",
            )
    # Model trust for THIS run: how far the scored rows sit from the data
    # the model was trained on. Reported, never used to change a score.
    if ml_result.is_available and hasattr(active_model, "drift_report"):
        monitoring_report = active_model.drift_report(feats_to_use)
    else:
        monitoring_report = {"status": "NOT_SCORED", "reason": ml_result.reason}
    model_trust = {
        "model_version": getattr(active_model, "version", None),
        "feature_schema_version": getattr(active_model, "feature_schema_version", None),
        "holdout_evaluated": getattr(active_model, "holdout_evaluated", None),
        "holdout_summary": getattr(active_model, "holdout_summary", None),
        "drift_relative_to_development": (monitoring_report.get("relative_to_development") or {}).get("status"),
        "drift_status": monitoring_report.get("status"),
        "score_psi": monitoring_report.get("score_psi"),
        "major_shift_features": monitoring_report.get("major_shift_features", []),
        "unseen_missingness_features": monitoring_report.get("unseen_missingness_features", []),
        "cold_start_share_observed": monitoring_report.get("cold_start_share_observed"),
    }
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
            "feature_contract": feature_contract.as_dict() if feature_contract else None,
            "model_load": model_load_report,
            "rows_per_second": round(len(feats_to_use) / score_seconds, 1) if score_seconds > 0 and ml_result.is_available else None,
            "model_trust": model_trust,
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
    # Seed-wallet risk propagation. Seeds come from offline sources (OFAC SDN,
    # data/watchlists/*.csv) unless the caller supplies them.
    from obsidianchain.io.watchlist import Seed, load_default_seeds
    from obsidianchain.ml.propagation import edges_from_capture, propagate
    if seed_addresses is None:
        seed_list = load_default_seeds()
    else:
        seed_list = [s if isinstance(s, Seed) else Seed(str(s), "CALLER") for s in seed_addresses]
    seed_sources = {s.address: s for s in seed_list}
    capture_edges = edges_from_capture(frame)
    prop_result = propagate(
        capture_edges, seed_sources, explain=set(capture_edges.address),
    ) if not capture_edges.empty else None
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
        propagation_result=prop_result,
        seed_sources=seed_sources,
        link_suggestions=link_suggestions,
        stacker=_load_stacker(active_model),
    )
    dt_alerts = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    _add_stage(StageExecutionRecord(
        stage_number=13, stage_name="Evidence Fusion",
        status="SUCCESS", duration_seconds=dt_alerts,
        summary={
            "clusters_fused": len(alert_result.alerts),
            "risk_propagation": prop_result.summary() if prop_result else {"status": "NOT_RUN"},
        },
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
        link_suggestions=link_suggestions,
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

    # Prediction audit log: one record per scored address, enough to trace
    # and recompute every score (model, schema, input hash, feature-row hash).
    extra_artifacts: dict[str, str] = {}
    if ml_result.is_available:
        audit = _prediction_audit(ml_result, feats_to_use, active_model, run_id, input_sha256)
        pred_file = run_dir / "predictions.parquet"
        audit.to_parquet(pred_file, index=False)
        extra_artifacts["predictions.parquet"] = hashlib.sha256(pred_file.read_bytes()).hexdigest()

    # Shadow: a registered candidate scores the same rows; nothing it
    # produces reaches an alert.
    shadow_report = _shadow(registry_root, active_model, ml_result, feats_to_use)
    if shadow_report is not None:
        shadow_file = run_dir / "shadow.json"
        shadow_bytes = json.dumps(shadow_report, indent=2, default=str).encode("utf-8")
        shadow_file.write_bytes(shadow_bytes)
        extra_artifacts["shadow.json"] = hashlib.sha256(shadow_bytes).hexdigest()

    # Serialize the model-trust report
    mon_file = run_dir / "monitoring.json"
    monitoring_alerts = _monitoring_alerts(model_trust, monitoring_report, capture_contract,
                                           feature_contract, model_load_report)
    import resource
    system = {
        "stage_seconds": {s.stage_name: round(s.duration_seconds, 4) for s in stages},
        "scoring_rows_per_second": round(len(feats_to_use) / score_seconds, 1)
        if score_seconds > 0 and ml_result.is_available else None,
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if __import__("sys").platform == "darwin" else 1 << 10), 1),
    }
    mon_bytes = json.dumps({"model_trust": model_trust, "drift": monitoring_report,
                            "alerts": monitoring_alerts, "system": system},
                           indent=2, default=str).encode("utf-8")
    mon_file.write_bytes(mon_bytes)
    mon_sha = hashlib.sha256(mon_bytes).hexdigest()

    dt_pub = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    manifest_file = run_dir / "manifest.json"

    _add_stage(StageExecutionRecord(
        stage_number=17, stage_name="Reporting & Integrity",
        status="SUCCESS", duration_seconds=dt_pub,
        summary={"manifest_path": str(manifest_file), "artifacts_published": 5},
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
            "model_trust": model_trust,
            "monitoring_alerts": monitoring_alerts,
        },
        "artifacts": {
            "alerts.json": alerts_sha,
            "investigation_graph.json": graph_sha,
            "validation_report.json": val_sha,
            "monitoring.json": mon_sha,
            **extra_artifacts,
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


def _load_stacker(model: Any) -> dict[str, Any] | None:
    """The stacker frozen in the serving model's own directory, if any.

    It was fitted on that model's out-of-fold scores, so it is never applied
    to a different model. Its hash is covered by the registry entry because
    it lives in the registered directory.
    """
    model_dir = getattr(model, "model_dir", None)
    if model_dir is None:
        return None
    from obsidianchain.ml.stacking import load_stacker
    return load_stacker(Path(model_dir) / "stacker.json")


def _load_serving_model(registry_root: str | Path | None) -> tuple[Any, dict[str, Any]]:
    """Champion, else fallback, from the registry. Every failure is kept."""
    from obsidianchain.ml.ps_model import PsNativeRiskModel
    report: dict[str, Any] = {"source": "registry", "failures": []}
    for role in ("champion", "fallback"):
        try:
            model = PsNativeRiskModel.from_registry(role, registry_root)
        except Exception as exc:  # recorded, then the next role is tried
            report["failures"].append({"role": role, "error": f"{type(exc).__name__}: {exc}"})
            continue
        report.update(role=role, version=model.version, model_sha256=model._sha256)
        return model, report
    return None, report


def _prediction_audit(ml_result, features: pd.DataFrame, model: Any, run_id: str,
                      input_sha256: str) -> pd.DataFrame:
    import numpy as np
    scores = ml_result.scores.reset_index(drop=True)
    feats = features.reset_index(drop=True)
    matrix = feats[list(model.features)].to_numpy(dtype="float64")
    row_hash = [hashlib.sha256(r.tobytes()).hexdigest()[:16] for r in matrix]
    raw = scores["ml_raw_score"] if "ml_raw_score" in scores else scores["ml_risk_score"]
    out = pd.DataFrame({
        "run_id": run_id,
        "address": scores["address"],
        "snapshot_txid": feats.get("txid"),
        "snapshot_timestamp": feats.get("timestamp"),
        "raw_score": raw.astype(float),
        "calibrated_probability": scores["ml_risk_score"].astype(float),
        "severity": scores.get("ml_severity"),
        "model_version": getattr(model, "version", None),
        "model_sha256": getattr(model, "_sha256", None),
        "feature_schema_version": getattr(model, "feature_schema_version", None),
        "input_sha256": input_sha256,
        "feature_row_sha256": row_hash,
    })
    out["rank"] = out["raw_score"].rank(ascending=False, method="first").astype(int)
    return out


def _shadow(registry_root, champion: Any, ml_result, features: pd.DataFrame) -> dict[str, Any] | None:
    """Candidate vs champion on the same rows, or None when no candidate is assigned."""
    if champion is None or not ml_result.is_available:
        return None
    from obsidianchain.ml import registry
    try:
        if registry.Registry.open(registry_root or registry.DEFAULT_ROOT).role("candidate") is None:
            return None
        from obsidianchain.ml.ps_model import PsNativeRiskModel
        candidate = PsNativeRiskModel.from_registry("candidate", registry_root)
    except Exception as exc:
        return {"status": "CANDIDATE_UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
    if candidate.version == getattr(champion, "version", None):
        return None
    import time

    import numpy as np
    from scipy.stats import spearmanr
    t0 = time.perf_counter()
    cand = np.asarray(candidate.raw_scores(features), dtype=float)
    seconds = time.perf_counter() - t0
    champ = np.asarray(champion.raw_scores(features), dtype=float)
    k = min(100, len(champ))
    top_c, top_p = set(np.argsort(-champ)[:k]), set(np.argsort(-cand)[:k])
    return {
        "status": "SHADOW_ONLY",
        "champion": champion.version, "candidate": candidate.version,
        "rows": int(len(cand)),
        "spearman": float(spearmanr(champ, cand).statistic) if len(cand) > 2 else None,
        "top100_overlap": len(top_c & top_p) / k if k else None,
        "mean_abs_score_difference": float(np.mean(np.abs(cand - champ))),
        "candidate_rows_per_second": round(len(cand) / seconds, 1) if seconds > 0 else None,
        "candidate_score_quantiles": np.quantile(cand, [0.5, 0.9, 0.99]).round(4).tolist() if len(cand) else [],
        "note": "Shadow scores never reach an alert. Promotion requires the gates in docs/runbook.md.",
    }


def _monitoring_alerts(model_trust, drift, capture_contract, feature_contract, model_load) -> list[dict[str, str]]:
    """Conditions an operator must see. Each names its severity and cause."""
    alerts: list[dict[str, str]] = []
    if model_load.get("role") == "fallback":
        alerts.append({"severity": "HIGH", "code": "SERVED_BY_FALLBACK",
                       "detail": str(model_load.get("failures"))})
    if model_load.get("source") == "registry" and model_load.get("role") is None:
        alerts.append({"severity": "CRITICAL", "code": "NO_MODEL_SERVED", "detail": str(model_load.get("failures"))})
    if feature_contract is not None and not feature_contract.ok:
        alerts.append({"severity": "CRITICAL", "code": "FEATURE_CONTRACT_VIOLATION",
                       "detail": str(feature_contract.violations[:3])})
    quarantined = len(capture_contract.quarantined)
    if capture_contract.transactions and quarantined / capture_contract.transactions > 0.05:
        alerts.append({"severity": "HIGH", "code": "CAPTURE_QUARANTINE_ABOVE_5PCT",
                       "detail": f"{quarantined}/{capture_contract.transactions}"})
    # Absolute PSI flags every two-week window of the development data
    # (exp23), so it is reported but never alerted on. The alert reads the run
    # against the model's development baseline. It cannot see concept drift
    # (exp23: t43/t45 read within baseline while precision collapsed); only
    # delayed labels can, which the INFO alert below says on every run.
    relative = (drift or {}).get("relative_to_development") or {}
    if relative.get("status") == "ABNORMAL":
        alerts.append({"severity": "HIGH", "code": "DRIFT_ABNORMAL_VS_DEVELOPMENT",
                       "detail": f"unusual features {relative.get('unusual_features', [])[:5]}, "
                                 f"score PSI above baseline: {relative.get('score_psi_above_baseline')}"})
    if drift and drift.get("status") not in (None, "NOT_SCORED", "NO_ROWS_SCORED"):
        alerts.append({"severity": "INFO", "code": "PERFORMANCE_UNVERIFIED_UNTIL_LABELS",
                       "detail": "input monitoring cannot detect concept drift; run `obsidianchain model health` "
                                 "when labels for this run arrive"})
    if (drift or {}).get("unseen_missingness_features"):
        alerts.append({"severity": "MEDIUM", "code": "UNSEEN_MISSINGNESS",
                       "detail": str(drift["unseen_missingness_features"])})
    if model_trust.get("holdout_evaluated") is False:
        alerts.append({"severity": "INFO", "code": "HOLDOUT_NOT_EVALUATED",
                       "detail": "scores come from a model without a sealed-holdout result"})
    return alerts
