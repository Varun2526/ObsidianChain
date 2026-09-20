"""Unsupervised MAD-based anomaly detection for Bitcoin forensic analysis.

RIGOROUS UNIT DEFINITION:
    - Primary Anomaly Unit: Address
      Each address is evaluated against the observed population using Median
      Absolute Deviation (MAD) across multiple forensic dimensions.
    - Secondary Aggregation: Cluster
      Cluster anomaly is aggregated from member address scores (max / top-k).
    - Transaction-Level Deviations:
      Discrete deviations at the transaction layer (e.g., outlier fee ratio,
      extreme amount) serve as supporting evidence contributing to address score.

NO ARBITRARY THRESHOLDS:
    Acceptance is ordering-based (anomalous entity > benign entity) rather
    than requiring an arbitrary score cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from obsidianchain.pipeline.blockchain import ClusterResult


@dataclass
class AnomalyDeviation:
    """A specific feature deviation for an address relative to the population."""

    feature: str
    observed_value: float
    population_median: float
    population_mad: float
    modified_z_score: float
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "observed_value": self.observed_value,
            "population_median": self.population_median,
            "population_mad": self.population_mad,
            "modified_z_score": round(self.modified_z_score, 4),
            "description": self.description,
        }


@dataclass
class AddressAnomalyRecord:
    """Forensic anomaly evaluation for one address."""

    address: str
    anomaly_score: float
    composite_z_score: float
    top_deviations: list[AnomalyDeviation] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "anomaly_score": round(self.anomaly_score, 6),
            "composite_z_score": round(self.composite_z_score, 4),
            "top_deviations": [d.as_dict() for d in self.top_deviations],
        }


@dataclass
class ClusterAnomalyRecord:
    """Aggregated anomaly evaluation for one entity cluster."""

    cluster_id: str
    anomaly_score: float
    member_count: int
    lead_address: str
    lead_address_score: float
    top_deviations: list[AnomalyDeviation] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "anomaly_score": round(self.anomaly_score, 6),
            "member_count": self.member_count,
            "lead_address": self.lead_address,
            "lead_address_score": round(self.lead_address_score, 6),
            "top_deviations": [d.as_dict() for d in self.top_deviations],
        }


@dataclass
class AnomalyDetectionResult:
    """Complete result of unsupervised anomaly detection."""

    addresses: dict[str, AddressAnomalyRecord] = field(default_factory=dict)
    clusters: dict[str, ClusterAnomalyRecord] = field(default_factory=dict)
    feature_medians: dict[str, float] = field(default_factory=dict)
    feature_mads: dict[str, float] = field(default_factory=dict)

    def address_score(self, address: str) -> float:
        rec = self.addresses.get(address)
        return rec.anomaly_score if rec else 0.0

    def cluster_score(self, cluster_id: str) -> float:
        rec = self.clusters.get(cluster_id)
        return rec.anomaly_score if rec else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        rows = [rec.as_dict() for rec in self.addresses.values()]
        if not rows:
            return pd.DataFrame(columns=["address", "anomaly_score", "composite_z_score", "top_deviations"])
        df = pd.DataFrame(rows)
        return df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


ANOMALY_FEATURE_SPECS = [
    ("mean_fee_ratio", "Fee Ratio", "Fee paid relative to total transaction value"),
    ("velocity_tx_per_hour", "Velocity (tx/hr)", "Frequency of transactions over time window"),
    ("btc_sent", "Outflow Volume (BTC)", "Total BTC spent across transactions"),
]


def _compute_robust_mad(values: np.ndarray) -> tuple[float, float]:
    """Compute median and robust Median Absolute Deviation with zero-MAD safeguard."""
    if len(values) == 0:
        return 0.0, 1.0

    med = float(np.median(values))
    abs_devs = np.abs(values - med)
    mad = float(np.median(abs_devs))

    # Zero-MAD safeguard: when >50% of the sample has identical values (e.g. median=0),
    # MAD is 0. We fall back to mean absolute deviation or a non-zero scale to prevent div/0.
    if mad <= 1e-9:
        mean_dev = float(np.mean(abs_devs))
        if mean_dev > 1e-9:
            mad = 1.2533 * mean_dev  # Asymptotic normal relation between mean dev and MAD
        else:
            mad = 1.0  # Population is uniform; scale is normalized

    return med, mad


def detect_address_anomalies(
    address_features: pd.DataFrame,
    cluster_result: ClusterResult | None = None,
) -> AnomalyDetectionResult:
    """Compute MAD-based anomaly scores per address and aggregate to clusters.

    Primary unit: Address.
    Secondary aggregation: Cluster (max over member addresses).
    """
    result = AnomalyDetectionResult()
    if address_features.empty:
        return result

    # Compute population statistics for each feature
    z_scores_by_feature: dict[str, np.ndarray] = {}

    for feat_col, label, desc in ANOMALY_FEATURE_SPECS:
        if feat_col not in address_features.columns:
            continue
        vals = address_features[feat_col].to_numpy(dtype=float)
        med, mad = _compute_robust_mad(vals)
        result.feature_medians[feat_col] = med
        result.feature_mads[feat_col] = mad

        # One-sided deviation: we flag unusually HIGH values for risk
        diffs = vals - med
        z_col = 0.6745 * diffs / mad
        z_scores_by_feature[feat_col] = np.maximum(0.0, z_col)

    if not z_scores_by_feature:
        return result

    # Composite Z-score and sigmoidal mapping to [0.0, 1.0]
    feature_names = list(z_scores_by_feature.keys())
    z_matrix = np.column_stack([z_scores_by_feature[f] for f in feature_names])  # (N, K)
    composite_z = np.sqrt(np.sum(z_matrix ** 2, axis=1))  # Euclidean distance in z-space

    # Monotonic mapping to [0.0, 1.0]: 1.0 - exp(-0.5 * z)
    # At z=0 -> 0.0. At z=2 -> 0.63. At z=5 -> 0.91. Strictly bounded and monotonic.
    anomaly_scores = 1.0 - np.exp(-0.5 * composite_z)

    # Build address records
    for i, row in address_features.iterrows():
        addr = str(row["address"])
        score = float(anomaly_scores[i])
        cz = float(composite_z[i])

        # Identify top deviations
        deviations: list[AnomalyDeviation] = []
        for feat_col, label, desc in ANOMALY_FEATURE_SPECS:
            if feat_col not in z_scores_by_feature:
                continue
            z_val = float(z_scores_by_feature[feat_col][i])
            obs_val = float(row[feat_col])
            med = result.feature_medians[feat_col]
            mad = result.feature_mads[feat_col]

            if z_val > 1.5:  # Significant departure from median
                ratio_str = f"{obs_val / med:.1f}x" if med > 0 else f"+{obs_val:.2f}"
                description = (
                    f"{label} ({obs_val:.4f}) is {ratio_str} population median "
                    f"({med:.4f}), modified Z: {z_val:.2f}"
                )
                deviations.append(AnomalyDeviation(
                    feature=feat_col,
                    observed_value=obs_val,
                    population_median=med,
                    population_mad=mad,
                    modified_z_score=z_val,
                    description=description,
                ))

        # Sort deviations descending by modified Z
        deviations.sort(key=lambda d: d.modified_z_score, reverse=True)

        result.addresses[addr] = AddressAnomalyRecord(
            address=addr,
            anomaly_score=score,
            composite_z_score=cz,
            top_deviations=deviations,
        )

    # Secondary Aggregation: Cluster Layer
    if cluster_result is not None:
        for cid, addrs in cluster_result.cluster_to_addresses.items():
            member_records = [result.addresses[a] for a in addrs if a in result.addresses]
            if not member_records:
                continue
            lead_rec = max(member_records, key=lambda r: r.anomaly_score)
            result.clusters[cid] = ClusterAnomalyRecord(
                cluster_id=cid,
                anomaly_score=lead_rec.anomaly_score,
                member_count=len(addrs),
                lead_address=lead_rec.address,
                lead_address_score=lead_rec.anomaly_score,
                top_deviations=lead_rec.top_deviations,
            )

    return result
