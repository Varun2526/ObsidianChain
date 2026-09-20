"""Blockchain transaction graph construction and Union-Find entity clustering.

SPEC Stage Order:
    Blockchain transaction graph
            ↓
    Entity clustering (Union-Find)
            ↓
    Investigation graph projection (deferred to Stage 16)

Transforms canonical PS-format records into a directed bipartite transaction-address
graph and groups co-spending addresses into candidate entities using Union-Find with
path compression and union-by-rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from obsidianchain.cluster.unionfind import UnionFind


@dataclass(frozen=True)
class TransactionFact:
    """Canonical facts for one blockchain transaction."""

    txid: str
    timestamp: str | float | None
    inputs: list[tuple[str, float]]  # (address, amount)
    outputs: list[tuple[str, float]]  # (address, amount)
    fee: float
    script_type: str | None


@dataclass
class BlockchainGraph:
    """Directed bipartite transaction-address graph."""

    transactions: dict[str, TransactionFact] = field(default_factory=dict)
    addresses: set[str] = field(default_factory=set)
    input_edges: list[dict[str, Any]] = field(default_factory=list)
    output_edges: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_transactions(self) -> int:
        return len(self.transactions)

    @property
    def n_addresses(self) -> int:
        return len(self.addresses)

    @property
    def n_edges(self) -> int:
        return len(self.input_edges) + len(self.output_edges)


@dataclass
class ClusterResult:
    """Outcome of multi-input clustering over canonical addresses."""

    address_to_cluster: dict[str, str]
    cluster_to_addresses: dict[str, list[str]]
    cluster_sizes: dict[str, int]
    n_clusters: int
    n_addresses: int
    singletons: int
    clustered: int
    largest_cluster_size: int

    def summary(self) -> dict[str, Any]:
        return {
            "n_addresses": self.n_addresses,
            "n_clusters": self.n_clusters,
            "singletons": self.singletons,
            "clustered": self.clustered,
            "largest_cluster_size": self.largest_cluster_size,
        }


def _parse_amount(v: Any) -> float:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0.0
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _clean_str(v: Any) -> str | None:
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    return s if s and s not in ("<NA>", "nan", "None") else None


def build_blockchain_graph(frame: pd.DataFrame) -> BlockchainGraph:
    """Build the directed transaction-address graph from canonical records."""
    bg = BlockchainGraph()

    for _, row in frame.iterrows():
        txid = _clean_str(row.get("txid"))
        if not txid:
            continue

        raw_in_addrs = row.get("input_addresses")
        raw_in_amts = row.get("input_amounts")
        raw_out_addrs = row.get("output_addresses")
        raw_out_amts = row.get("output_amounts")

        in_addrs = [str(a) for a in raw_in_addrs if a is not None and not pd.isna(a) and str(a).strip()] if isinstance(raw_in_addrs, (list, tuple)) else []
        out_addrs = [str(a) for a in raw_out_addrs if a is not None and not pd.isna(a) and str(a).strip()] if isinstance(raw_out_addrs, (list, tuple)) else []

        in_amts = [_parse_amount(a) for a in raw_in_amts] if isinstance(raw_in_amts, (list, tuple)) else []
        out_amts = [_parse_amount(a) for a in raw_out_amts] if isinstance(raw_out_amts, (list, tuple)) else []

        # Pad amounts to match addresses length if needed
        while len(in_amts) < len(in_addrs):
            in_amts.append(0.0)
        while len(out_amts) < len(out_addrs):
            out_amts.append(0.0)

        inputs = [(in_addrs[i], in_amts[i]) for i in range(len(in_addrs))]
        outputs = [(out_addrs[i], out_amts[i]) for i in range(len(out_addrs))]

        fee = _parse_amount(row.get("fee"))
        script_type = _clean_str(row.get("script_type"))
        ts = row.get("timestamp") if not pd.isna(row.get("timestamp")) else None

        # Only record as blockchain transaction if blockchain data exists
        has_bc = bool(inputs or outputs or fee > 0.0 or script_type is not None)
        if not has_bc:
            continue

        # Record addresses
        for a, _ in inputs:
            bg.addresses.add(a)
        for a, _ in outputs:
            bg.addresses.add(a)

        # Record edges and transaction facts
        if txid not in bg.transactions:
            bg.transactions[txid] = TransactionFact(
                txid=txid,
                timestamp=ts,
                inputs=inputs,
                outputs=outputs,
                fee=fee,
                script_type=script_type,
            )
            for addr, amt in inputs:
                bg.input_edges.append({
                    "source": addr,
                    "target": txid,
                    "amount": amt,
                    "script_type": script_type,
                })
            for addr, amt in outputs:
                bg.output_edges.append({
                    "source": txid,
                    "target": addr,
                    "amount": amt,
                    "script_type": script_type,
                })

    return bg


def cluster_entities(
    frame: pd.DataFrame,
    graph: BlockchainGraph | None = None,
) -> ClusterResult:
    """Cluster addresses using multi-input heuristic (Union-Find).

    Every address starts as a singleton. For transactions with multiple input
    addresses, an entity edge is asserted connecting them via union_star.
    """
    if graph is None:
        graph = build_blockchain_graph(frame)

    sorted_addrs = sorted(graph.addresses)
    n_addrs = len(sorted_addrs)
    addr_to_idx = {addr: i for i, addr in enumerate(sorted_addrs)}

    uf = UnionFind(n_addrs)

    for tx in graph.transactions.values():
        if len(tx.inputs) > 1:
            indices = [addr_to_idx[addr] for addr, _ in tx.inputs if addr in addr_to_idx]
            if len(indices) > 1:
                uf.union_star(indices)

    # Collect clusters
    roots = uf.roots()
    address_to_cluster: dict[str, str] = {}
    cluster_to_addresses: dict[str, list[str]] = {}

    for i, addr in enumerate(sorted_addrs):
        root_idx = int(roots[i])
        root_addr = sorted_addrs[root_idx]
        cluster_id = f"cluster_{root_addr}"
        address_to_cluster[addr] = cluster_id
        cluster_to_addresses.setdefault(cluster_id, []).append(addr)

    cluster_sizes = {cid: len(addrs) for cid, addrs in cluster_to_addresses.items()}
    singletons = sum(1 for sz in cluster_sizes.values() if sz == 1)
    clustered = sum(sz for sz in cluster_sizes.values() if sz > 1)
    largest = max(cluster_sizes.values()) if cluster_sizes else 0

    return ClusterResult(
        address_to_cluster=address_to_cluster,
        cluster_to_addresses=cluster_to_addresses,
        cluster_sizes=cluster_sizes,
        n_clusters=len(cluster_sizes),
        n_addresses=n_addrs,
        singletons=singletons,
        clustered=clustered,
        largest_cluster_size=largest,
    )


def build_blockchain_layer(frame: pd.DataFrame) -> tuple[BlockchainGraph, ClusterResult]:
    """Execute internal Stage 7 (Blockchain Graph) and Stage 8 (Clustering)."""
    bg = build_blockchain_graph(frame)
    clusters = cluster_entities(frame, graph=bg)
    return bg, clusters
