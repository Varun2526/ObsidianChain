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
from obsidianchain.ml.propagation import PropagationResult
from obsidianchain.pipeline.blockchain import BlockchainGraph, ClusterResult
from obsidianchain.pipeline.features import MlStageResult
from obsidianchain.pipeline.patterns import MixingResult, PeelingResult

PATTERN_CONTEXT = "PATTERN_CONTEXT"
ANOMALY_CONTEXT = "ANOMALY_CONTEXT"
BLOCKCHAIN_CONTEXT = "BLOCKCHAIN_CONTEXT"
NETWORK_CONTEXT = "NETWORK_CONTEXT"
MODEL_SIGNAL = "MODEL_SIGNAL"
#: Structural closeness to seed illicit wallets (OFAC SDN, analyst
#: watchlists) by personalized PageRank. A claim about the GRAPH around an
#: entity, never about the entity itself.
PROPAGATION_CONTEXT = "PROPAGATION_CONTEXT"
#: Blockchain <-> network coherence: on-chain hops first announced by the
#: same relay beyond the capture's chance rate (correlation/cross_layer.py).
CROSS_LAYER_CONTEXT = "CROSS_LAYER_CONTEXT"


#: What KIND of claim each evidence category is, so model output, rules,
#: network observation and third-party lists are never read as one thing.
EVIDENCE_CLASS = {
    MODEL_SIGNAL: "MODEL",              # a learned association; not a cause
    ANOMALY_CONTEXT: "RULE",            # statistical deviation rule (MAD)
    PATTERN_CONTEXT: "RULE",            # structural pattern rule (peel, CoinJoin shape)
    NETWORK_CONTEXT: "NETWORK",         # what observers saw on the P2P layer
    PROPAGATION_CONTEXT: "WATCHLIST",   # graph distance to externally listed wallets
    BLOCKCHAIN_CONTEXT: "CONTEXT",      # descriptive; not risk evidence
    CROSS_LAYER_CONTEXT: "RULE",        # statistical coherence test across the two layers
}

EXPLANATION_STATEMENT = (
    "Model contributions (TreeSHAP) describe how this model's score was formed; "
    "they are associations learned from labelled history, not causes. Rule, "
    "network and watchlist evidence are separate claims with their own sources."
)


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
            "evidence_class": EVIDENCE_CLASS.get(self.category, "CONTEXT"),
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
            "explanation_statement": EXPLANATION_STATEMENT,
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


#: Weight of each evidence line in the noisy-OR fusion. POLICY, not fitted:
#: only the model line has labels behind it (it is a calibrated probability,
#: or the stacked model + propagation probability). The rule lines are
#: discounted because their scores are hand-set strengths, not
#: probabilities. A line's contribution is weight x score.
FUSION_WEIGHTS = {
    MODEL_SIGNAL: 1.0,
    PROPAGATION_CONTEXT: 0.8,
    PATTERN_CONTEXT: 0.6,
    ANOMALY_CONTEXT: 0.5,
}

#: The cross-layer line's policy weight, and whether it is fused. The
#: pre-registered exp-net2 (research/network_2026_09_25/RESULT_exp_net2.md)
#: found fusing it LOWERED alert-ranking nAP in both the SIGNAL and NULL
#: worlds: relay coherence marks operator-controlled flow, which benign
#: services produce too. So it is computed, shown and used to link
#: clusters across layers, and does not move the score.
CROSS_LAYER_WEIGHT = 0.5
FUSE_CROSS_LAYER = False
if FUSE_CROSS_LAYER:
    FUSION_WEIGHTS[CROSS_LAYER_CONTEXT] = CROSS_LAYER_WEIGHT

#: A line counts as corroborating when PRESENT with at least this score.
CORROBORATION_MIN = 0.3

#: Share of a run's alerts per band, most severe first; the rest are LOW or
#: INFORMATIONAL. Budgets rather than thresholds, because an analyst's
#: capacity is fixed and a static threshold failed out-of-time (exp09).
ALERT_SEVERITY_BUDGET = (("CRITICAL", 0.02), ("HIGH", 0.08), ("MEDIUM", 0.20))

#: Below this fused score an alert is INFORMATIONAL whatever its rank, so a
#: quiet capture does not produce CRITICAL alerts just because something
#: has to rank first.
SEVERITY_FLOOR = 0.20


def noisy_or(lines: list[tuple[float, float]]) -> float:
    """``1 - prod(1 - w*s)``. Any strong line raises priority; independent
    agreement compounds; a missing line (not appended) is neutral rather
    than dragging an average down, as the old weighted mean did."""
    remaining = 1.0
    for score, weight in lines:
        remaining *= 1.0 - max(0.0, min(1.0, weight * score))
    return 1.0 - remaining


def assign_budget_severity(alerts: list[AlertItem]) -> None:
    """Severity from rank position, for alerts already sorted best-first."""
    n = len(alerts)
    start = 0
    for band, share in ALERT_SEVERITY_BUDGET:
        end = min(n, max(start, int(-(-share * n // 1))))
        for alert in alerts[start:end]:
            alert.severity = band if alert.fused_risk_score >= SEVERITY_FLOOR else "INFORMATIONAL"
        start = end
    for alert in alerts[start:]:
        alert.severity = "LOW" if alert.fused_risk_score >= SEVERITY_FLOOR else "INFORMATIONAL"


def _model_line(ml_result: MlStageResult | None, propagation_result: PropagationResult | None,
                stacker: dict[str, Any] | None) -> pd.Series | None:
    """Per-address model probability, stacked with propagation when possible."""
    if ml_result is None or not ml_result.is_available:
        return None
    scores = ml_result.scores.set_index("address")
    line = scores["ml_risk_score"].astype(float)
    if (stacker is not None and propagation_result is not None
            and propagation_result.seeds_in_graph > 0 and "ml_raw_score" in scores.columns):
        from obsidianchain.ml.stacking import apply_stacker
        prop = propagation_result.scores.reindex(scores.index).fillna(0.0).to_numpy()
        line = pd.Series(apply_stacker(stacker, scores["ml_raw_score"].to_numpy(), prop), index=scores.index)
        line.attrs["stacked"] = True
    return line


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
    propagation_result: PropagationResult | None = None,
    seed_sources: dict[str, Any] | None = None,
    link_suggestions: Any | None = None,
    stacker: dict[str, Any] | None = None,
    network_propagation: Any | None = None,
    cross_layer: Any | None = None,
) -> AlertRunResult:
    """Fuse multi-source evidence and rank alerts across all entity clusters."""
    alerts: list[AlertItem] = []

    # Address -> transactions, built once. The previous per-cluster scan of
    # every transaction was O(clusters x transactions): 69 s of a 84 s run on
    # a 7,303-transaction capture.
    addr_txs: dict[str, set[str]] = {}
    for txid, fact in blockchain_graph.transactions.items():
        for a, _ in list(fact.inputs) + list(fact.outputs):
            addr_txs.setdefault(a, set()).add(txid)

    # Per-address model line. With seeds present and a frozen stacker, the
    # model score and the propagated seed risk are combined by the stacker
    # fitted out-of-fold on Elliptic++ (exp14); otherwise the calibrated
    # model probability is used alone.
    model_line = _model_line(ml_result, propagation_result, stacker)

    # Cross-layer flows: clusters the network layer ties together (ownership
    # leads, never merged into one entity).
    links_of: dict[str, list[dict]] = {}
    if cross_layer is not None:
        for flow in cross_layer.relay_flows(blockchain_graph, cluster_result.address_to_cluster):
            for own in flow["clusters"]:
                links_of.setdefault(own, []).append({
                    "relay": flow["relay"], "p_value": flow["p_value"],
                    "relay_coherent_pairs": flow["relay_coherent_pairs"], "expected": flow["expected"],
                    "relay_parent_hops": flow["relay_parent_hops"],
                    "linked_clusters": [c for c in flow["clusters"] if c != own][:25],
                    "flow_size": len(flow["clusters"]),
                })

    # Evaluate each cluster
    for cid, addrs in cluster_result.cluster_to_addresses.items():
        evidence_items: list[EvidenceItem] = []
        scores_to_fuse: list[tuple[float, float]] = []  # (score, weight)

        # 1. BLOCKCHAIN CONTEXT
        total_txs: set[str] = set()
        for a in addrs:
            total_txs |= addr_txs.get(a, set())

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
        scores_to_fuse.append((c_anom, FUSION_WEIGHTS[ANOMALY_CONTEXT]))

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
        scores_to_fuse.append((pattern_score, FUSION_WEIGHTS[PATTERN_CONTEXT]))

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

            pooled = network_propagation.pooled(total_txs) if network_propagation is not None else None
            explanation = f"Observed via {obs_count} network announcement(s) across {len(cluster_ips)} peer IP(s)"
            if pooled:
                explanation += f" and {pooled['asn_count']} ASN(s)"
                if pooled.get("median_spread_ms") is not None:
                    explanation += f"; median propagation spread {pooled['median_spread_ms'] / 1000:.1f} s"
                if pooled.get("dominant_peer_share") is not None and len(cluster_ips) > 1:
                    explanation += f"; top peer carried {pooled['dominant_peer_share'] * 100:.0f}% of announcements"
            explanation += ". A peer is a relay vantage point, not the sender. Context only: not part of the risk score."
            evidence_items.append(EvidenceItem(
                category=NETWORK_CONTEXT,
                signal_name="p2p_network_telemetry",
                status="PRESENT",
                score=0.30 if len(cluster_ips) > 1 else 0.10,
                details={
                    "distinct_ips": sorted(cluster_ips),
                    "distinct_asns": sorted(cluster_asns),
                    "observation_count": obs_count,
                    # Resolved by the offline GeoIP provider; empty without a database.
                    "countries": sorted(set(countries)),
                    "propagation": pooled,
                    "fused": False,
                },
                explanation=explanation,
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

        # 4b. CROSS-LAYER COHERENCE: do the network and on-chain layers agree?
        if cross_layer is not None:
            test = cross_layer.evaluate(total_txs)
            relay = test["relays"][0]["peer_ip"] if test["relays"] else None
            if test["status"] == "PRESENT":
                explanation = (
                    f"{test['coherent_pairs']} of {test['hop_pairs']} on-chain hop(s) through this cluster were "
                    f"first announced by the same relay"
                    + (f" ({relay})" if relay else "")
                    + f"; chance rate in this capture {test['chance_rate'] * 100:.1f}% "
                    f"(expected {test['expected_coherent']}), p = {test['p_value']:.2g}. "
                    "The network layer agrees with the money flow. A relay is a vantage point, not the sender."
                )
            elif test["hop_pairs"]:
                explanation = (
                    f"{test['coherent_pairs']} of {test['hop_pairs']} on-chain hop(s) shared a first-seen relay; "
                    f"chance alone predicts {test.get('expected_coherent', 0)}. No cross-layer coherence."
                )
            else:
                explanation = "No observed on-chain hops to compare against the network layer."
            evidence_items.append(EvidenceItem(
                category=CROSS_LAYER_CONTEXT,
                signal_name="cross_layer_relay_coherence",
                status=test["status"],
                score=test["score"],
                details={**test, "fused": FUSE_CROSS_LAYER, "weight": CROSS_LAYER_WEIGHT,
                         "linked_clusters": links_of.get(cid, [])[:10]},
                explanation=explanation + ("" if FUSE_CROSS_LAYER else " Not part of the risk score."),
            ))
            if FUSE_CROSS_LAYER and test["status"] == "PRESENT":
                scores_to_fuse.append((test["score"], CROSS_LAYER_WEIGHT))

        # 5. SUPERVISED ML CONTEXT
        if ml_result and ml_result.is_available:
            # Score from model if available
            ml_sub = ml_result.scores[ml_result.scores["address"].isin(addrs)]
            member_line = model_line.reindex(addrs).dropna() if model_line is not None else None
            if member_line is not None and not member_line.empty:
                ml_risk = float(member_line.max())
            else:
                ml_risk = float(ml_sub["ml_risk_score"].max()) if not ml_sub.empty else 0.0

            # Format model explanation
            # Explain the member the score came from (the highest-scoring
            # one), not whichever row happens to be first. Explanations are
            # stored from PsFeatureExplanation, whose name key is
            # ``feature_name``; ``feature`` is accepted for older payloads.
            exps_text = ""
            if not ml_sub.empty and "ml_explanations" in ml_sub.columns:
                top_row = ml_sub.loc[ml_sub["ml_risk_score"].idxmax()] if "ml_risk_score" in ml_sub.columns else ml_sub.iloc[0]
                top_exps = top_row.get("ml_explanations")
                if isinstance(top_exps, list) and top_exps:
                    exp_strs = [
                        f"{e.get('feature_name') or e.get('feature') or 'unnamed feature'} "
                        f"({str(e.get('direction', '')).lower().replace('_', ' ')})"
                        for e in top_exps[:3]
                    ]
                    exps_text = (f" Key contributing features for {top_row.get('address')}: "
                                 f"{', '.join(exp_strs)}.")

            evidence_items.append(EvidenceItem(
                category=MODEL_SIGNAL,
                signal_name="supervised_risk_model",
                status="PRESENT",
                score=ml_risk,
                details={
                    "model_scores": ml_sub.to_dict(orient="records"),
                    "combined_with_propagation": bool(model_line is not None and model_line.attrs.get("stacked")),
                },
                explanation=(
                    f"Supervised model{' + seed propagation (stacked)' if model_line is not None and model_line.attrs.get('stacked') else ''} "
                    f"scored member address at {ml_risk:.4f} probability.{exps_text}"
                ),
            ))
            scores_to_fuse.append((ml_risk, FUSION_WEIGHTS[MODEL_SIGNAL]))
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

        # 6. RISK PROPAGATION from seed wallets
        prop_ev, prop_score = _propagation_evidence(addrs, propagation_result, seed_sources or {})
        evidence_items.append(prop_ev)
        stacked = model_line is not None and model_line.attrs.get("stacked")
        if not stacked and propagation_result is not None and propagation_result.seeds_in_graph > 0:
            # Propagation counts on its own only when the stacker has not
            # already folded it into the model line - never twice.
            scores_to_fuse.append((prop_score, FUSION_WEIGHTS[PROPAGATION_CONTEXT]))

        fused_score = noisy_or(scores_to_fuse)
        corroborating = sum(
            1 for e in evidence_items
            if e.status == "PRESENT" and e.category in FUSION_WEIGHTS and e.score >= CORROBORATION_MIN
        )

        alert_item = AlertItem(
            alert_id=f"alert_{cid}",
            cluster_id=cid,
            primary_address=lead_addr,
            member_addresses=addrs,
            fused_risk_score=fused_score,
            severity="INFORMATIONAL",
            evidence=evidence_items,
            summary={
                "confidence": round(fused_score, 4),
                "corroborating_evidence_lines": corroborating,
                "anomaly_score": c_anom,
                "peeling_detected": c_peel_score > 0,
                "mixing_detected": c_mix_score > 0,
                "network_observations": obs_count,
                # Embedding-similar entities in OTHER clusters. Shown, not fused:
                # similarity is a hint about ownership, not about risk.
                # Other clusters joined to this one across layers (same relay
                # first announced the hops between them beyond chance).
                "cross_layer_links": links_of.get(cid, [])[:10],
                "suggested_links": (
                    link_suggestions.for_cluster(cid)[:5] if link_suggestions else []
                ),
            },
        )
        alerts.append(alert_item)

    # Rank by fused score, then by how many independent lines agree, then by
    # id for determinism. Severity is a budget over that ranking.
    alerts.sort(key=lambda a: (-a.fused_risk_score,
                               -a.summary["corroborating_evidence_lines"], a.alert_id))
    for rank_idx, alert in enumerate(alerts):
        alert.rank = rank_idx + 1
    assign_budget_severity(alerts)

    return AlertRunResult(run_id=run_id, alerts=alerts)


def _propagation_evidence(
    addrs: list[str],
    result: PropagationResult | None,
    seed_sources: dict[str, Any],
) -> tuple[EvidenceItem, float]:
    """One PROPAGATION_CONTEXT item for a cluster, and its score.

    Three distinct silences are kept distinct: propagation not run, no seed
    wallet present in this capture, and seeds present but no path to this
    cluster. Only the last is a measured absence of risk.
    """
    if result is None:
        return EvidenceItem(
            category=PROPAGATION_CONTEXT, signal_name="seed_risk_propagation",
            status="NO_EVIDENCE", score=0.0,
            details={"reason": "PROPAGATION_NOT_RUN"},
            explanation="Risk propagation was not run for this analysis.",
        ), 0.0
    if result.seeds_in_graph == 0:
        return EvidenceItem(
            category=PROPAGATION_CONTEXT, signal_name="seed_risk_propagation",
            status="NO_EVIDENCE", score=0.0,
            details={"reason": "NO_SEED_WALLET_IN_CAPTURE", "seeds_loaded": len(result.seeds)},
            explanation=(f"None of the {len(result.seeds)} seed wallet(s) appears in this "
                         f"capture, so no risk can be propagated."),
        ), 0.0

    member_scores = result.scores.reindex(addrs).fillna(0.0)
    best = str(member_scores.idxmax())
    score = float(member_scores.max())
    seeded = [a for a in addrs if a in seed_sources]
    if seeded:
        src = seed_sources[seeded[0]]
        label = f" ({src.label})" if getattr(src, "label", "") else ""
        return EvidenceItem(
            category=PROPAGATION_CONTEXT, signal_name="seed_risk_propagation",
            status="PRESENT", score=1.0,
            details={"seed_members": seeded, "source": getattr(src, "source", "CALLER")},
            explanation=(f"Member {seeded[0]} is a seed wallet listed by "
                         f"{getattr(src, 'source', 'the caller')}{label}."),
        ), 1.0
    path = result.paths.get(best)
    if score <= 0.0 or path is None:
        return EvidenceItem(
            category=PROPAGATION_CONTEXT, signal_name="seed_risk_propagation",
            status="NO_EVIDENCE", score=score,
            details={"reason": "NO_PATH_TO_SEED", "seeds_in_graph": result.seeds_in_graph},
            explanation="No short transaction path connects this entity to a seed wallet.",
        ), score
    return EvidenceItem(
        category=PROPAGATION_CONTEXT, signal_name="seed_risk_propagation",
        status="PRESENT", score=score,
        details={"member": best, "nearest_seed": path["seed"], "hops": path["hops"],
                 "path": path["path"], "alpha": result.alpha},
        explanation=(f"Member {best} is {path['hops']} transaction hop(s) from seed wallet "
                     f"{path['seed']}; propagated risk {score:.3f}."),
    ), score
