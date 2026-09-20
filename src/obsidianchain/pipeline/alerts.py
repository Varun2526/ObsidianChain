"""Evidence fusion and alert ranking across all analytical stages.

Forensic Safeguards:
    - Explicit NO_EVIDENCE for absent data sources (e.g., NO_NETWORK_OBSERVATION).
    - Explicit MODEL_UNAVAILABLE_FOR_SCHEMA when supervised ML features cannot be derived.
    - Multiple orthogonal evidence categories:
        BLOCKCHAIN_CONTEXT, NETWORK_CONTEXT, ANOMALY_CONTEXT, PATTERN_CONTEXT, MODEL_SIGNAL.
    - Ordering-based ranking (anomalous entities and distinct patterns rank above benign baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from obsidianchain.correlation.engine import CorrelationResult
from obsidianchain.geoip import GeoIPProvider
from obsidianchain.ml.anomaly import AnomalyDetectionResult
from obsidianchain.pipeline.blockchain import BlockchainGraph, ClusterResult
from obsidianchain.pipeline.features import MlStageResult
from obsidianchain.pipeline.patterns import MixingResult, PeelingResult

PATTERN_CONTEXT = "PATTERN_CONTEXT"
ANOMALY_CONTEXT = "ANOMALY_CONTEXT"
BLOCKCHAIN_CONTEXT = "BLOCKCHAIN_CONTEXT"
NETWORK_CONTEXT = "NETWORK_CONTEXT"
MODEL_SIGNAL = "MODEL_SIGNAL"


@dataclass
class EvidenceItem:
    """A discrete forensic evidence item from an analytical stage."""

    category: str
    signal_name: str
    status: str
    """'PRESENT', 'NO_EVIDENCE', or 'UNAVAILABLE'."""
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "signal_name": self.signal_name,
            "status": self.status,
            "score": round(self.score, 4),
            "explanation": self.explanation,
            "details": self.details,
        }


@dataclass
class AlertItem:
    """A ranked forensic alert representing an entity or pattern requiring investigation."""

    alert_id: str
    cluster_id: str
    primary_address: str
    member_addresses: list[str]
    fused_risk_score: float
    severity: str
    rank: int = 0
    evidence: list[EvidenceItem] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "cluster_id": self.cluster_id,
            "primary_address": self.primary_address,
            "member_count": len(self.member_addresses),
            "fused_risk_score": round(self.fused_risk_score, 4),
            "severity": self.severity,
            "rank": self.rank,
            "summary": self.summary,
            "evidence": [e.as_dict() for e in self.evidence],
        }


@dataclass
class AlertRunResult:
    """The complete outcome of the alert ranking and evidence fusion run."""

    run_id: str
    alerts: list[AlertItem] = field(default_factory=list)

    @property
    def total_alerts(self) -> int:
        return len(self.alerts)

    def alert_for_address(self, address: str) -> AlertItem | None:
        for a in self.alerts:
            if address in a.member_addresses:
                return a
        return None

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for a in self.alerts:
            rows.append({
                "rank": a.rank,
                "alert_id": a.alert_id,
                "cluster_id": a.cluster_id,
                "primary_address": a.primary_address,
                "member_count": len(a.member_addresses),
                "fused_risk_score": a.fused_risk_score,
                "severity": a.severity,
                "evidence_count": len(a.evidence),
            })
        return pd.DataFrame(rows)


def _determine_severity(score: float) -> str:
    if score >= 0.80:
        return "CRITICAL"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    if score >= 0.20:
        return "LOW"
    return "INFORMATIONAL"


def build_alert_run(
    run_id: str,
    blockchain_graph: BlockchainGraph,
    cluster_result: ClusterResult,
    correlation_result: CorrelationResult,
    anomaly_result: AnomalyDetectionResult | None = None,
    ml_result: MlStageResult | None = None,
    peeling_result: PeelingResult | None = None,
    mixing_result: MixingResult | None = None,
    geoip_provider: GeoIPProvider | None = None,
) -> AlertRunResult:
    """Fuse multi-source evidence and rank alerts across all entity clusters."""
    alerts: list[AlertItem] = []

    # Map transactions to clusters
    tx_to_clusters: dict[str, set[str]] = {}
    for txid, fact in blockchain_graph.transactions.items():
        for in_addr, _ in fact.inputs:
            cid = cluster_result.address_to_cluster.get(in_addr)
            if cid:
                tx_to_clusters.setdefault(txid, set()).add(cid)
        for out_addr, _ in fact.outputs:
            cid = cluster_result.address_to_cluster.get(out_addr)
            if cid:
                tx_to_clusters.setdefault(txid, set()).add(cid)

    # Evaluate each cluster
    for cid, addrs in cluster_result.cluster_to_addresses.items():
        evidence_items: list[EvidenceItem] = []
        scores_to_fuse: list[tuple[float, float]] = []  # (score, weight)

        # 1. BLOCKCHAIN CONTEXT
        total_txs = set()
        for a in addrs:
            for txid, fact in blockchain_graph.transactions.items():
                if any(in_a == a for in_a, _ in fact.inputs) or any(out_a == a for out_a, _ in fact.outputs):
                    total_txs.add(txid)

        evidence_items.append(EvidenceItem(
            category=BLOCKCHAIN_CONTEXT,
            signal_name="cluster_topology",
            status="PRESENT",
            score=min(1.0, len(addrs) / 5.0),
            details={
                "member_addresses_count": len(addrs),
                "transactions_involved_count": len(total_txs),
            },
            explanation=f"Entity cluster contains {len(addrs)} co-spending address(es) across {len(total_txs)} transaction(s).",
        ))

        # 2. ANOMALY CONTEXT
        c_anom = 0.0
        lead_addr = addrs[0]
        if anomaly_result:
            c_anom = anomaly_result.cluster_score(cid)
            cl_rec = anomaly_result.clusters.get(cid)
            if cl_rec:
                lead_addr = cl_rec.lead_address
                dev_explanations = [d.description for d in cl_rec.top_deviations]
                evidence_items.append(EvidenceItem(
                    category=ANOMALY_CONTEXT,
                    signal_name="robust_mad_deviation",
                    status="PRESENT" if c_anom > 0.1 else "NO_EVIDENCE",
                    score=c_anom,
                    details={
                        "anomaly_score": c_anom,
                        "lead_address": lead_addr,
                        "deviations": [d.as_dict() for d in cl_rec.top_deviations],
                    },
                    explanation="; ".join(dev_explanations) if dev_explanations else "Within normal population parameters.",
                ))
            else:
                evidence_items.append(EvidenceItem(
                    category=ANOMALY_CONTEXT,
                    signal_name="robust_mad_deviation",
                    status="NO_EVIDENCE",
                    score=0.0,
                    explanation="No anomalous feature deviation detected across member addresses.",
                ))
        scores_to_fuse.append((c_anom, 0.40))

        # 3. PATTERN CONTEXT: Peeling & Mixing
        c_peel_score = 0.0
        c_peel_chains = []
        if peeling_result:
            for txid in total_txs:
                if peeling_result.is_peeling(txid):
                    depth = peeling_result.tx_to_depth.get(txid, 0)
                    chain_id = peeling_result.tx_to_chain.get(txid, "")
                    c_peel_chains.append((chain_id, depth))
            if c_peel_chains:
                max_depth = max(d for _, d in c_peel_chains)
                c_peel_score = 0.85 if max_depth >= 3 else 0.50
                evidence_items.append(EvidenceItem(
                    category=PATTERN_CONTEXT,
                    signal_name="peeling_chain",
                    status="PRESENT",
                    score=c_peel_score,
                    details={"max_depth": max_depth, "chains": c_peel_chains},
                    explanation=f"Entity participates in sequential peeling chain of depth {max_depth}.",
                ))
            else:
                evidence_items.append(EvidenceItem(
                    category=PATTERN_CONTEXT,
                    signal_name="peeling_chain",
                    status="NO_EVIDENCE",
                    score=0.0,
                    explanation="No peeling chain structure detected.",
                ))

        c_mix_score = 0.0
        c_mix_classes = []
        if mixing_result:
            for txid in total_txs:
                m_class = mixing_result.classification_for_tx(txid)
                if m_class in ("MIXING_PATTERN", "MIXING_LIKELIHOOD"):
                    c_mix_classes.append((txid, m_class))
            if any(cls == "MIXING_PATTERN" for _, cls in c_mix_classes):
                c_mix_score = 0.90
                evidence_items.append(EvidenceItem(
                    category=PATTERN_CONTEXT,
                    signal_name="coinjoin_mixing",
                    status="PRESENT",
                    score=c_mix_score,
                    details={"transactions": c_mix_classes},
                    explanation="Transaction matches CoinJoin equal-output collaborative spend pattern.",
                ))
            elif any(cls == "MIXING_LIKELIHOOD" for _, cls in c_mix_classes):
                c_mix_score = 0.60
                evidence_items.append(EvidenceItem(
                    category=PATTERN_CONTEXT,
                    signal_name="coinjoin_mixing",
                    status="PRESENT",
                    score=c_mix_score,
                    details={"transactions": c_mix_classes},
                    explanation="Transaction exhibits partial CoinJoin mixing characteristics.",
                ))
            else:
                evidence_items.append(EvidenceItem(
                    category=PATTERN_CONTEXT,
                    signal_name="coinjoin_mixing",
                    status="NO_EVIDENCE",
                    score=0.0,
                    explanation="No collaborative spend or CoinJoin structure detected.",
                ))

        pattern_score = max(c_peel_score, c_mix_score)
        scores_to_fuse.append((pattern_score, 0.40))

        # 4. NETWORK CONTEXT
        cluster_ips: set[str] = set()
        cluster_asns: set[int] = set()
        obs_count = 0
        for txid in total_txs:
            for obs in correlation_result.observations_for_tx(txid):
                obs_count += 1
                if obs.src_ip:
                    cluster_ips.add(obs.src_ip)
                if obs.asn:
                    cluster_asns.add(obs.asn)

        if cluster_ips:
            countries = []
            if geoip_provider:
                for ip in cluster_ips:
                    f = geoip_provider.resolve_ip(ip)
                    if f.country_iso:
                        countries.append(f.country_iso)

            evidence_items.append(EvidenceItem(
                category=NETWORK_CONTEXT,
                signal_name="p2p_network_telemetry",
                status="PRESENT",
                score=0.30 if len(cluster_ips) > 1 else 0.10,
                details={
                    "distinct_ips": list(cluster_ips),
                    "distinct_asns": list(cluster_asns),
                    "observation_count": obs_count,
                    "countries": sorted(list(set(countries))),
                },
                explanation=f"Observed via {obs_count} network announcement(s) across {len(cluster_ips)} peer IP(s).",
            ))
        else:
            evidence_items.append(EvidenceItem(
                category=NETWORK_CONTEXT,
                signal_name="p2p_network_telemetry",
                status="NO_EVIDENCE",
                score=0.0,
                details={"reason": "NO_NETWORK_OBSERVATION"},
                explanation="No network telemetry observations were recorded for transactions in this cluster.",
            ))

        # 5. SUPERVISED ML CONTEXT
        if ml_result and ml_result.is_available:
            # Score from model if available
            ml_sub = ml_result.scores[ml_result.scores["address"].isin(addrs)]
            ml_risk = float(ml_sub["ml_risk_score"].max()) if not ml_sub.empty else 0.0

            # Format model explanation
            exps_text = ""
            if not ml_sub.empty and "ml_explanations" in ml_sub.columns:
                top_exps = ml_sub.iloc[0].get("ml_explanations")
                if isinstance(top_exps, list) and top_exps:
                    exp_strs = [f"{e.get('feature', '')} ({str(e.get('direction', '')).lower().replace('_', ' ')})" for e in top_exps[:3]]
                    exps_text = f" Key contributing features: {', '.join(exp_strs)}."

            evidence_items.append(EvidenceItem(
                category=MODEL_SIGNAL,
                signal_name="supervised_risk_model",
                status="PRESENT",
                score=ml_risk,
                details={"model_scores": ml_sub.to_dict(orient="records")},
                explanation=f"Supervised ML risk model scored member address at {ml_risk:.4f} risk.{exps_text}",
            ))
            scores_to_fuse.append((ml_risk, 0.20))
        else:
            incomp_reason = ml_result.reason if ml_result else "No supervised model configured."
            evidence_items.append(EvidenceItem(
                category=MODEL_SIGNAL,
                signal_name="supervised_risk_model",
                status="UNAVAILABLE",
                score=0.0,
                details={"status": ml_result.status if ml_result else "MODEL_UNAVAILABLE"},
                explanation=f"{ml_result.status if ml_result else 'MODEL_UNAVAILABLE'}: {incomp_reason}",
            ))

        # Compute fused risk score (weighted average of active signals)
        total_w = sum(w for s, w in scores_to_fuse)
        if total_w > 0:
            fused_score = sum(s * w for s, w in scores_to_fuse) / total_w
        else:
            fused_score = 0.0

        # Reinforce if multiple distinct signals agree
        reinforcement = 0.0
        if c_anom > 0.5 and pattern_score > 0.5:
            reinforcement += 0.15
        if c_anom > 0.5 and len(addrs) > 1:
            reinforcement += 0.05
        fused_score = min(1.0, fused_score + reinforcement)

        alert_item = AlertItem(
            alert_id=f"alert_{cid}",
            cluster_id=cid,
            primary_address=lead_addr,
            member_addresses=addrs,
            fused_risk_score=fused_score,
            severity=_determine_severity(fused_score),
            evidence=evidence_items,
            summary={
                "anomaly_score": c_anom,
                "peeling_detected": c_peel_score > 0,
                "mixing_detected": c_mix_score > 0,
                "network_observations": obs_count,
            },
        )
        alerts.append(alert_item)

    # Sort descending by fused risk score and assign ranks
    alerts.sort(key=lambda a: a.fused_risk_score, reverse=True)
    for rank_idx, alert in enumerate(alerts):
        alert.rank = rank_idx + 1

    return AlertRunResult(run_id=run_id, alerts=alerts)
