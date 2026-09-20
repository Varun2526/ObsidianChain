"""Exact-TXID blockchain-to-network correlation engine.

Forensic Invariant:
    Exact TXID match only:
        txid in blockchain AND txid in network  → CORRELATED
        txid in blockchain AND NOT in network  → NO_NETWORK_OBSERVATION
        txid in network AND NOT in blockchain  → UNMATCHED_NETWORK_OBSERVATION

NO FUZZY MATCHING:
    No timestamp-proximity matching. No IP-based inference.
    An alert does not require network evidence to be valid (NO_NETWORK_OBSERVATION
    is a fully legitimate forensic state, not a failure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from obsidianchain.io.ingest import observation_key
from obsidianchain.pipeline.blockchain import BlockchainGraph


@dataclass(frozen=True)
class NetworkObservation:
    """A discrete network telemetry observation from a specific vantage point."""

    txid: str
    observer_id: str | None
    src_ip: str | None
    dst_ip: str | None
    src_port: int | None
    dst_port: int | None
    timestamp: str | float | None
    geo_country: str | None
    asn: int | None
    key: str


@dataclass
class CorrelationResult:
    """The structured result of correlating blockchain transactions with network observations."""

    correlated_txids: dict[str, list[NetworkObservation]] = field(default_factory=dict)
    no_network_txids: list[str] = field(default_factory=list)
    unmatched_network_txids: dict[str, list[NetworkObservation]] = field(default_factory=dict)
    all_observations: list[NetworkObservation] = field(default_factory=list)

    def status_for_tx(self, txid: str) -> str:
        """Return the forensic correlation status for a given TXID."""
        if txid in self.correlated_txids:
            return "CORRELATED"
        if txid in self.no_network_txids:
            return "NO_NETWORK_OBSERVATION"
        if txid in self.unmatched_network_txids:
            return "UNMATCHED_NETWORK_OBSERVATION"
        return "UNKNOWN"

    def observations_for_tx(self, txid: str) -> list[NetworkObservation]:
        """Return network observations for a given TXID, or an empty list."""
        if txid in self.correlated_txids:
            return self.correlated_txids[txid]
        if txid in self.unmatched_network_txids:
            return self.unmatched_network_txids[txid]
        return []

    def summary(self) -> dict[str, Any]:
        distinct_observers = {
            obs.observer_id for obs in self.all_observations if obs.observer_id
        }
        return {
            "total_blockchain_transactions": len(self.correlated_txids) + len(self.no_network_txids),
            "correlated_count": len(self.correlated_txids),
            "no_network_count": len(self.no_network_txids),
            "unmatched_network_count": len(self.unmatched_network_txids),
            "total_network_observations": len(self.all_observations),
            "distinct_observers_count": len(distinct_observers),
        }


def _clean_str(v: Any) -> str | None:
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    return s if s and s not in ("<NA>", "nan", "None") else None


def _clean_int(v: Any) -> int | None:
    if v is None or pd.isna(v):
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def extract_network_observations(frame: pd.DataFrame) -> list[NetworkObservation]:
    """Extract discrete network observations from canonical records."""
    observations: list[NetworkObservation] = []

    for _, row in frame.iterrows():
        txid = _clean_str(row.get("txid"))
        src_ip = _clean_str(row.get("src_ip"))
        if not txid or not src_ip:
            continue

        obs_k = observation_key(row)
        obs = NetworkObservation(
            txid=txid,
            observer_id=_clean_str(row.get("observer_id") if "observer_id" in row else row.get("observer")),
            src_ip=src_ip,
            dst_ip=_clean_str(row.get("dst_ip")),
            src_port=_clean_int(row.get("src_port")),
            dst_port=_clean_int(row.get("dst_port")),
            timestamp=row.get("timestamp") if not pd.isna(row.get("timestamp")) else None,
            geo_country=_clean_str(row.get("geo_country")),
            asn=_clean_int(row.get("asn")),
            key=obs_k,
        )
        observations.append(obs)

    return observations


def correlate_blockchain_and_network(
    frame: pd.DataFrame,
    graph: BlockchainGraph | None = None,
) -> CorrelationResult:
    """Correlate blockchain transactions with network observations via exact TXID matching.

    Every transaction in the blockchain graph is classified as either CORRELATED
    or NO_NETWORK_OBSERVATION. Any network observations lacking a corresponding
    blockchain transaction are classified as UNMATCHED_NETWORK_OBSERVATION.
    """
    observations = extract_network_observations(frame)
    obs_by_txid: dict[str, list[NetworkObservation]] = {}
    for obs in observations:
        obs_by_txid.setdefault(obs.txid, []).append(obs)

    # Determine blockchain TXIDs
    if graph is not None:
        bc_txids = set(graph.transactions.keys())
    else:
        bc_txids = set()
        for _, row in frame.iterrows():
            txid = _clean_str(row.get("txid"))
            if not txid:
                continue
            has_bc = bool(
                (isinstance(row.get("input_addresses"), (list, tuple)) and len(row["input_addresses"]) > 0)
                or (isinstance(row.get("output_addresses"), (list, tuple)) and len(row["output_addresses"]) > 0)
                or (pd.notna(row.get("fee")) and str(row.get("fee")).strip() not in ("", "<NA>", "nan", "None"))
            )
            if has_bc:
                bc_txids.add(txid)

    res = CorrelationResult(all_observations=observations)

    # Classify blockchain txids
    for txid in bc_txids:
        if txid in obs_by_txid:
            res.correlated_txids[txid] = obs_by_txid[txid]
        else:
            res.no_network_txids.append(txid)

    # Classify unmatched network observations
    for txid, obs_list in obs_by_txid.items():
        if txid not in bc_txids:
            res.unmatched_network_txids[txid] = obs_list

    res.no_network_txids.sort()
    return res
