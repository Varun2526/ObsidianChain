"""PS-Native Feature Engine: Temporal, As-Of-T Address Forensic Features.

P0 CRITICAL CONSTRAINTS:
1. Primary prediction unit is ADDRESS-AS-OF-TIMESTAMP.
2. Temporal state is strictly forward-only: information at t_future > t NEVER influences
   the feature vector at t.
3. Label-blind: labels, classes, categories, or ground-truth indicators NEVER enter
   the feature matrix.
4. Missing network telemetry is represented as NaN, NEVER fabricated as zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature Group Definitions
# ---------------------------------------------------------------------------

GROUP_A_TRANSACTION = [
    "input_count",
    "output_count",
    "total_input_amount",
    "total_output_amount",
    "fee",
    "fee_ratio",
    "input_amount_mean",
    "input_amount_max",
    "input_amount_std",
    "output_amount_mean",
    "output_amount_max",
    "output_amount_std",
]

GROUP_B_ADDRESS_HISTORY = [
    "n_txs_asof_t",
    "n_sent_asof_t",
    "n_recv_asof_t",
    "btc_sent_total_asof_t",
    "btc_recv_total_asof_t",
    "net_flow_asof_t",
    "mean_fee_ratio_asof_t",
    "active_duration_seconds",
    "tx_velocity_per_hour",
    "gap_since_last_tx",
]

GROUP_C_GRAPH = [
    "in_degree_asof_t",
    "out_degree_asof_t",
    "unique_counterparties_asof_t",
    "cluster_size_asof_t",
]

GROUP_D_PATTERNS = [
    "is_peeling_candidate",
    "is_mixing_candidate",
    "equal_output_count",
    "output_entropy",
]

GROUP_E_NETWORK = [
    "network_observation_count",
    "observer_diversity",
    "peer_count",
    "asn_count",
]

PS_FEATURE_GROUPS: dict[str, list[str]] = {
    "A_transaction": GROUP_A_TRANSACTION,
    "B_address_history": GROUP_B_ADDRESS_HISTORY,
    "C_graph": GROUP_C_GRAPH,
    "D_patterns": GROUP_D_PATTERNS,
    "E_network": GROUP_E_NETWORK,
}

# The core feature set without optional network features
CORE_PS_FEATURE_COLUMNS: list[str] = (
    GROUP_A_TRANSACTION + GROUP_B_ADDRESS_HISTORY + GROUP_C_GRAPH + GROUP_D_PATTERNS
)

# The full feature set including optional network features
ALL_PS_FEATURE_COLUMNS: list[str] = CORE_PS_FEATURE_COLUMNS + GROUP_E_NETWORK

# Metadata columns (not fed to ML booster)
METADATA_COLUMNS = ["address", "txid", "timestamp"]


def _safe_float(val: Any, default: float = 0.0, min_val: float | None = 0.0) -> float:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        f = float(val)
        if not math.isfinite(f):
            return default
        if min_val is not None and f < min_val:
            return default
        return f
    except (ValueError, TypeError):
        return default


def _calculate_entropy(amounts: list[float]) -> float:
    """Compute Shannon entropy of output amount distribution."""
    total = sum(amounts)
    if total <= 0:
        return 0.0
    probs = [a / total for a in amounts if a > 0]
    if len(probs) <= 1:
        return 0.0
    return -sum(p * math.log2(p) for p in probs)


# ---------------------------------------------------------------------------
# Incremental Union-Find for Cluster Size as-of-T
# ---------------------------------------------------------------------------

class IncrementalUnionFind:
    """Forward-replay Union-Find tracking component sizes as-of-t."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        curr = x
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            if self.size[root_a] < self.size[root_b]:
                root_a, root_b = root_b, root_a
            self.parent[root_b] = root_a
            self.size[root_a] += self.size[root_b]

    def get_cluster_size(self, x: str) -> int:
        root = self.find(x)
        return self.size[root]


# ---------------------------------------------------------------------------
# Per-Address Historical State Tracker
# ---------------------------------------------------------------------------

@dataclass
class AddressState:
    n_txs: int = 0
    n_sent: int = 0
    n_recv: int = 0
    btc_sent: float = 0.0
    btc_recv: float = 0.0
    fee_ratios: list[float] = field(default_factory=list)
    first_seen: float | None = None
    last_seen: float | None = None
    in_tx_ids: set[str] = field(default_factory=set)
    out_tx_ids: set[str] = field(default_factory=set)
    counterparties: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Temporal Feature Engine
# ---------------------------------------------------------------------------

class PsTemporalFeatureEngine:
    """Chronological state engine for PS-native feature extraction.

    Processes records in strict timestamp order to guarantee ZERO temporal leakage.
    """

    def __init__(self) -> None:
        self.address_states: dict[str, AddressState] = {}
        self.uf = IncrementalUnionFind()

    def process_records(
        self,
        frame: pd.DataFrame,
        include_network: bool = False,
    ) -> pd.DataFrame:
        """Extract address-as-of-t feature vectors for every address in each transaction.

        Guarantees:
        1. Records are processed strictly in increasing timestamp order.
        2. Features at event T use only state established up to T.
        3. Features are completely label-blind (labels are ignored).
        """
        if frame.empty:
            cols = METADATA_COLUMNS + (ALL_PS_FEATURE_COLUMNS if include_network else CORE_PS_FEATURE_COLUMNS)
            return pd.DataFrame(columns=cols)

        # Ensure frame has a timestamp column; sort chronologically
        df = frame.copy()
        if "timestamp" in df.columns:
            df["_sort_ts"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)
        else:
            df["_sort_ts"] = 0.0
        
        df = df.sort_values(by=["_sort_ts"], ascending=True, kind="stable")

        output_rows: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            t = float(row["_sort_ts"])
            txid = str(row.get("txid") or "").strip()
            if not txid:
                continue

            raw_in_addrs = row.get("input_addresses")
            raw_in_amts = row.get("input_amounts")
            raw_out_addrs = row.get("output_addresses")
            raw_out_amts = row.get("output_amounts")

            in_addrs = [str(a).strip() for a in raw_in_addrs if a and not pd.isna(a) and str(a).strip()] if isinstance(raw_in_addrs, (list, tuple)) else []
            out_addrs = [str(a).strip() for a in raw_out_addrs if a and not pd.isna(a) and str(a).strip()] if isinstance(raw_out_addrs, (list, tuple)) else []

            in_amts = [_safe_float(a) for a in raw_in_amts] if isinstance(raw_in_amts, (list, tuple)) else []
            out_amts = [_safe_float(a) for a in raw_out_amts] if isinstance(raw_out_amts, (list, tuple)) else []

            # Pad amounts if shorter than addresses
            while len(in_amts) < len(in_addrs):
                in_amts.append(0.0)
            while len(out_amts) < len(out_addrs):
                out_amts.append(0.0)

            fee = _safe_float(row.get("fee"))
            total_in = sum(in_amts)
            total_out = sum(out_amts)
            fee_ratio = (fee / total_in) if total_in > 0 else 0.0

            # --- Group A: Transaction Behaviour (Instantaneous at t) ---
            in_count = len(in_addrs)
            out_count = len(out_addrs)
            in_mean = float(np.mean(in_amts)) if in_amts else 0.0
            in_max = float(np.max(in_amts)) if in_amts else 0.0
            in_std = float(np.std(in_amts)) if len(in_amts) > 1 else 0.0
            out_mean = float(np.mean(out_amts)) if out_amts else 0.0
            out_max = float(np.max(out_amts)) if out_amts else 0.0
            out_std = float(np.std(out_amts)) if len(out_amts) > 1 else 0.0

            # --- Group D: Structural Patterns ---
            is_peel = 1 if (in_count == 1 and out_count == 2 and out_max >= 0.8 * total_out and total_out > 0) else 0
            counts_by_amt: dict[float, int] = {}
            for oa in out_amts:
                r_oa = round(oa, 6)
                counts_by_amt[r_oa] = counts_by_amt.get(r_oa, 0) + 1
            max_equal = max(counts_by_amt.values()) if counts_by_amt else 0
            is_mix = 1 if (in_count >= 3 and out_count >= 3 and max_equal >= 3) else 0
            entropy = _calculate_entropy(out_amts)

            # --- Group E: Optional Network Features ---
            net_obs_count = np.nan
            obs_div = np.nan
            peer_cnt = np.nan
            asn_cnt = np.nan
            if include_network:
                src_ip = row.get("src_ip")
                if src_ip is not None and not pd.isna(src_ip) and str(src_ip).strip():
                    net_obs_count = 1.0
                    obs_div = 1.0 if row.get("observer_id") is not None else 1.0
                    peer_cnt = 1.0
                    asn_cnt = 1.0 if row.get("asn") is not None and not pd.isna(row.get("asn")) else 0.0

            # --- Co-spend Union-Find Update as of t ---
            if len(in_addrs) > 1:
                first_in = in_addrs[0]
                for other_in in in_addrs[1:]:
                    self.uf.union(first_in, other_in)

            # --- Process Address State & Extract Features as-of-t ---
            participating = sorted(set(in_addrs) | set(out_addrs))

            for addr in participating:
                st = self.address_states.setdefault(addr, AddressState())

                prior_txs = st.n_txs
                prior_sent_count = st.n_sent
                prior_recv_count = st.n_recv
                prior_sent_btc = st.btc_sent
                prior_recv_btc = st.btc_recv
                prior_net_flow = prior_recv_btc - prior_sent_btc
                prior_mean_fee_ratio = float(np.mean(st.fee_ratios)) if st.fee_ratios else 0.0

                first_s = st.first_seen if st.first_seen is not None else t
                last_s = st.last_seen if st.last_seen is not None else t
                duration_sec = max(t - first_s, 0.0)
                gap_sec = max(t - last_s, 0.0)
                duration_hours = max(duration_sec / 3600.0, 1.0 / 3600.0)
                velocity = prior_txs / duration_hours if duration_sec > 0 else float(prior_txs)

                in_degree = len(st.in_tx_ids)
                out_degree = len(st.out_tx_ids)
                unique_cps = len(st.counterparties)
                cluster_size = self.uf.get_cluster_size(addr)

                feature_dict = {
                    "address": addr,
                    "txid": txid,
                    "timestamp": t,
                    # Group A
                    "input_count": in_count,
                    "output_count": out_count,
                    "total_input_amount": total_in,
                    "total_output_amount": total_out,
                    "fee": fee,
                    "fee_ratio": fee_ratio,
                    "input_amount_mean": in_mean,
                    "input_amount_max": in_max,
                    "input_amount_std": in_std,
                    "output_amount_mean": out_mean,
                    "output_amount_max": out_max,
                    "output_amount_std": out_std,
                    # Group B
                    "n_txs_asof_t": prior_txs,
                    "n_sent_asof_t": prior_sent_count,
                    "n_recv_asof_t": prior_recv_count,
                    "btc_sent_total_asof_t": prior_sent_btc,
                    "btc_recv_total_asof_t": prior_recv_btc,
                    "net_flow_asof_t": prior_net_flow,
                    "mean_fee_ratio_asof_t": prior_mean_fee_ratio,
                    "active_duration_seconds": duration_sec,
                    "tx_velocity_per_hour": velocity,
                    "gap_since_last_tx": gap_sec,
                    # Group C
                    "in_degree_asof_t": in_degree,
                    "out_degree_asof_t": out_degree,
                    "unique_counterparties_asof_t": unique_cps,
                    "cluster_size_asof_t": cluster_size,
                    # Group D
                    "is_peeling_candidate": is_peel,
                    "is_mixing_candidate": is_mix,
                    "equal_output_count": max_equal,
                    "output_entropy": entropy,
                }

                if include_network:
                    feature_dict.update({
                        "network_observation_count": net_obs_count,
                        "observer_diversity": obs_div,
                        "peer_count": peer_cnt,
                        "asn_count": asn_cnt,
                    })

                output_rows.append(feature_dict)

                # Now update state with this transaction
                st.n_txs += 1
                if st.first_seen is None:
                    st.first_seen = t
                st.last_seen = t

                if addr in in_addrs:
                    st.n_sent += 1
                    addr_idx = in_addrs.index(addr)
                    st.btc_sent += in_amts[addr_idx]
                    st.out_tx_ids.add(txid)
                    st.fee_ratios.append(fee_ratio)
                    st.counterparties.update(out_addrs)

                if addr in out_addrs:
                    st.n_recv += 1
                    addr_idx = out_addrs.index(addr)
                    st.btc_recv += out_amts[addr_idx]
                    st.in_tx_ids.add(txid)
                    st.counterparties.update(in_addrs)

        return pd.DataFrame(output_rows)


def extract_ps_features(
    frame: pd.DataFrame,
    include_network: bool = False,
) -> pd.DataFrame:
    """Convenience function to extract PS-native features using a fresh state engine."""
    engine = PsTemporalFeatureEngine()
    return engine.process_records(frame, include_network=include_network)
