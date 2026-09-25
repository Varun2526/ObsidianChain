"""Investigation graph projection across cluster, address, transaction, and network layers.

Produces a multi-layer graph with stable identifiers:
    cluster:<cluster_id>
    addr:<address>
    tx:<txid>
    ip:<src_ip>

Connects:
    addr  --[MEMBER_OF]-->  cluster
    addr  --[SPENDS]----->  tx
    tx    --[RECEIVES]--->  addr
    tx    --[ANNOUNCED_BY]--> ip
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from obsidianchain.correlation.engine import CorrelationResult
from obsidianchain.pipeline.blockchain import BlockchainGraph, ClusterResult
from obsidianchain.pipeline.patterns import MixingResult, PeelingResult

if TYPE_CHECKING:
    from obsidianchain.ml.anomaly import AnomalyDetectionResult



@dataclass
class GraphNode:
    """A discrete node in the investigation graph."""

    id: str
    kind: str
    """'cluster', 'address', 'transaction', or 'ip'."""
    label: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "data": self.data,
        }


@dataclass
class GraphEdge:
    """A directed edge in the investigation graph."""

    id: str
    source: str
    target: str
    kind: str
    """'MEMBER_OF', 'SPENDS', 'RECEIVES', 'ANNOUNCED_BY', 'PEEL_HOP', 'SUGGESTED_LINK'."""
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "data": self.data,
        }


@dataclass
class InvestigationGraph:
    """Projected multi-layer forensic graph."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": [n.as_dict() for n in self.nodes],
            "edges": [e.as_dict() for e in self.edges],
        }

    def filter_subgraph(self, focal_ids: set[str], hops: int = 1) -> InvestigationGraph:
        """Extract a local k-hop neighborhood around one or more focal nodes."""
        current_node_ids = set(focal_ids)
        node_map = {n.id: n for n in self.nodes}

        for _ in range(hops):
            neighbor_ids: set[str] = set()
            for e in self.edges:
                if e.source in current_node_ids:
                    neighbor_ids.add(e.target)
                if e.target in current_node_ids:
                    neighbor_ids.add(e.source)
            current_node_ids.update(neighbor_ids)

        sub_nodes = [node_map[nid] for nid in current_node_ids if nid in node_map]
        sub_edges = [
            e for e in self.edges
            if e.source in current_node_ids and e.target in current_node_ids
        ]
        return InvestigationGraph(nodes=sub_nodes, edges=sub_edges)


def project_investigation_graph(
    blockchain_graph: BlockchainGraph,
    cluster_result: ClusterResult,
    correlation_result: CorrelationResult,
    anomaly_result: AnomalyDetectionResult | None = None,
    peeling_result: PeelingResult | None = None,
    mixing_result: MixingResult | None = None,
    link_suggestions: Any | None = None,
) -> InvestigationGraph:
    """Project the complete multi-layer investigation graph.

    ``link_suggestions`` (``ml.embeddings.LinkSuggestions``) adds
    cluster--[SUGGESTED_LINK]-->cluster edges. They are drawn, never merged.
    """
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    edge_ids: set[str] = set()

    def _add_edge(edge_id: str, src: str, dst: str, kind: str, data: dict[str, Any] | None = None):
        if edge_id not in edge_ids:
            edge_ids.add(edge_id)
            edges.append(GraphEdge(id=edge_id, source=src, target=dst, kind=kind, data=data or {}))

    # 1. Cluster nodes
    for cid, addrs in cluster_result.cluster_to_addresses.items():
        c_node_id = f"cluster:{cid}"
        c_anom = anomaly_result.cluster_score(cid) if anomaly_result else 0.0
        nodes[c_node_id] = GraphNode(
            id=c_node_id,
            kind="cluster",
            label=f"Cluster {cid}",
            data={
                "cluster_id": cid,
                "member_count": len(addrs),
                "anomaly_score": c_anom,
            },
        )

    # 2. Address nodes and MEMBER_OF edges
    for addr in blockchain_graph.addresses:
        a_node_id = f"addr:{addr}"
        cid = cluster_result.address_to_cluster.get(addr, f"cluster_{addr}")
        c_node_id = f"cluster:{cid}"

        anom_score = 0.0
        deviations = []
        if anomaly_result and addr in anomaly_result.addresses:
            rec = anomaly_result.addresses[addr]
            anom_score = rec.anomaly_score
            deviations = [d.as_dict() for d in rec.top_deviations]

        nodes[a_node_id] = GraphNode(
            id=a_node_id,
            kind="address",
            label=f"{addr[:8]}...",
            data={
                "address": addr,
                "cluster_id": cid,
                "anomaly_score": anom_score,
                "top_deviations": deviations,
            },
        )

        _add_edge(f"member:{addr}:{cid}", a_node_id, c_node_id, "MEMBER_OF")

    # 3. Transaction nodes and SPENDS / RECEIVES edges
    for txid, fact in blockchain_graph.transactions.items():
        t_node_id = f"tx:{txid}"
        is_peel = peeling_result.is_peeling(txid) if peeling_result else False
        peel_depth = peeling_result.tx_to_depth.get(txid, 0) if peeling_result else 0
        mixing_class = mixing_result.classification_for_tx(txid) if mixing_result else "NO_MIXING_SIGNAL"
        corr_status = correlation_result.status_for_tx(txid)

        nodes[t_node_id] = GraphNode(
            id=t_node_id,
            kind="transaction",
            label=f"{txid[:8]}...",
            data={
                "txid": txid,
                "fee": fact.fee,
                "script_type": fact.script_type,
                "timestamp": fact.timestamp,
                "correlation_status": corr_status,
                "is_peeling": is_peel,
                "peeling_depth": peel_depth,
                "mixing_classification": mixing_class,
            },
        )

        for in_addr, amt in fact.inputs:
            a_node_id = f"addr:{in_addr}"
            _add_edge(f"spends:{in_addr}:{txid}", a_node_id, t_node_id, "SPENDS", {"amount": amt})

        for out_addr, amt in fact.outputs:
            a_node_id = f"addr:{out_addr}"
            _add_edge(f"receives:{txid}:{out_addr}", t_node_id, a_node_id, "RECEIVES", {"amount": amt})

    # 4. Peer IP nodes, their ASN, and ANNOUNCED_BY edges. A peer is the IP
    #    that relayed the transaction to an observer: never the sender.
    from obsidianchain import geoip as _geoip
    from obsidianchain.network.propagation import timestamp_ms

    for obs in correlation_result.all_observations:
        if not obs.src_ip:
            continue
        ip_node_id = f"ip:{obs.src_ip}"
        if ip_node_id not in nodes:
            facts = _geoip.resolve_ip(obs.src_ip)
            nodes[ip_node_id] = GraphNode(
                id=ip_node_id,
                kind="ip",
                label=obs.src_ip,
                data={
                    "ip": obs.src_ip,
                    "asn": obs.asn,
                    "geo_country": obs.geo_country,
                    "geo_country_source": "capture-supplied (unverified)" if obs.geo_country else None,
                    "globally_routable": facts.globally_routable and facts.special_purpose is None,
                    "special_purpose": facts.special_purpose,
                },
            )
        if obs.asn is not None:
            asn_node_id = f"asn:{obs.asn}"
            if asn_node_id not in nodes:
                asn_facts = _geoip.resolve_asn(obs.asn)
                nodes[asn_node_id] = GraphNode(
                    id=asn_node_id, kind="asn", label=f"AS{obs.asn}",
                    data={"asn": obs.asn, "private_use": asn_facts.private_use,
                          "description": asn_facts.description},
                )
            _add_edge(f"in_asn:{obs.src_ip}:{obs.asn}", ip_node_id, asn_node_id, "IN_ASN", {})

        t_node_id = f"tx:{obs.txid}"
        if t_node_id in nodes:
            _add_edge(
                f"announced:{obs.txid}:{obs.src_ip}:{obs.key[:8]}",
                t_node_id,
                ip_node_id,
                "ANNOUNCED_BY",
                {
                    "src_port": obs.src_port,
                    "timestamp": obs.timestamp,
                    "timestamp_ms": timestamp_ms(obs.timestamp),
                    "observer_id": obs.observer_id,
                    "dst_ip": obs.dst_ip,
                },
            )

    if link_suggestions is not None:
        for pair in link_suggestions.pairs:
            a, b = f"cluster:{pair['cluster_a']}", f"cluster:{pair['cluster_b']}"
            if a in nodes and b in nodes:
                _add_edge(f"suggest:{a}:{b}", a, b, "SUGGESTED_LINK", {
                    "similarity": pair["similarity"],
                    "address_a": pair["address_a"], "address_b": pair["address_b"],
                    "status": "SUGGESTION_ONLY",
                })

    return InvestigationGraph(nodes=list(nodes.values()), edges=edges)


def get_alert_graph(alert_id: str, root=None, hops: int = 2) -> dict[str, Any]:
    """Retrieve or build the investigation graph for a given alert ID."""
    from obsidianchain.api import alerts as alerts_api

    detail = alerts_api.get_alert(alert_id, root=root)
    cid = detail["summary"]["cluster_id"]
    cluster_node_id = f"cluster:{cid}"

    nodes = [
        GraphNode(
            id=cluster_node_id,
            kind="cluster",
            label=f"Cluster {cid}",
            data={
                "cluster_id": cid,
                "severity": detail["summary"]["severity"],
                "risk_score": detail["summary"].get("risk_score"),
            },
        )
    ]
    edges = []

    # Member addresses
    member_addrs = set()
    for m in detail.get("members", {}).get("rows", []):
        addr = m["address"]
        member_addrs.add(addr)
        a_node_id = f"addr:{addr}"
        nodes.append(GraphNode(
            id=a_node_id,
            kind="address",
            label=f"{addr[:8]}...",
            data={
                "address": addr,
                "risk_score": m.get("risk_score"),
                "severity": m.get("severity"),
                "observed_at_timestep": m.get("observed_at_timestep"),
            },
        ))
        edges.append(GraphEdge(
            id=f"member:{addr}:{cid}",
            source=a_node_id,
            target=cluster_node_id,
            kind="MEMBER_OF",
        ))

    # Relationships between addresses
    for rel in detail.get("relationships", {}).get("edges", []):
        src_addr = rel.get("address_a")
        dst_addr = rel.get("address_b")
        if src_addr and dst_addr:
            edges.append(GraphEdge(
                id=f"rel:{src_addr}:{dst_addr}:{rel.get('relationship', 'COSPEND')}",
                source=f"addr:{src_addr}",
                target=f"addr:{dst_addr}",
                kind=rel.get("relationship", "COSPEND"),
                data={"timestep": rel.get("timestep")},
            ))

    # Network correlation: transactions & IPs
    tx_nodes = set()
    ip_nodes = set()
    for corr in detail.get("correlation", {}).get("records", []):
        txid = corr.get("txid")
        ip = corr.get("src_ip")
        if txid and txid not in tx_nodes:
            tx_nodes.add(txid)
            nodes.append(GraphNode(
                id=f"tx:{txid}",
                kind="transaction",
                label=f"{txid[:8]}...",
                data={
                    "txid": txid,
                    "timestamp": corr.get("timestamp"),
                },
            ))
        if ip and ip not in ip_nodes:
            ip_nodes.add(ip)
            nodes.append(GraphNode(
                id=f"ip:{ip}",
                kind="ip",
                label=ip,
                data={
                    "ip": ip,
                    "asn": corr.get("asn"),
                },
            ))
        if txid and ip:
            edges.append(GraphEdge(
                id=f"announced:{txid}:{ip}",
                source=f"tx:{txid}",
                target=f"ip:{ip}",
                kind="ANNOUNCED_BY",
                data={"port": corr.get("src_port")},
            ))

    inv_graph = InvestigationGraph(nodes=nodes, edges=edges)
    return {
        "alert_id": alert_id,
        "graph": inv_graph.as_dict(),
    }

