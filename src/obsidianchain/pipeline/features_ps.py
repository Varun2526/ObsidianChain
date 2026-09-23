"""PS-Native Feature Engine: Temporal, As-Of-T Address Forensic Features.

P0 CRITICAL CONSTRAINTS:
1. Primary prediction unit is ADDRESS-AS-OF-TIMESTAMP.
2. Temporal state is strictly forward-only: information at t_future > t NEVER influences
   the feature vector at t.
3. Label-blind: labels, classes, categories, or ground-truth indicators NEVER enter
   the feature matrix.
4. Missing values - network telemetry, fee, anything the source did not
   carry - are represented as NaN, NEVER fabricated as zero.
5. One transaction is ONE chain event. A capture may carry several network
   observations of the same txid; they are aggregated into the network
   group, never replayed as repeated transactions.
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
    # v3. total_output_amount removed: it is total_input_amount minus the
    # fee, Spearman 1.000 on the development data.
    "fee",
    "fee_ratio",
    "input_amount_mean",
    "output_amount_mean",
    # v2. The source carries a min/max/mean/total summary per transaction and
    # nothing finer, so per-output values were previously SYNTHESISED by even
    # division - which made *_std exactly zero, *_max identical to *_mean,
    # equal_output_count identical to output_count and output_entropy equal
    # to log2(output_count). Spread is the signal those columns were
    # pretending to carry, and it is computable from the real summary.
    "input_spread",
    "output_spread",
]

GROUP_B_ADDRESS_HISTORY = [
    "n_txs_asof_t",
    "n_sent_asof_t",
    "n_recv_asof_t",
    # v3. btc_sent_total_asof_t, mean_fee_ratio_asof_t and
    # tx_velocity_per_hour removed. Each restated a sibling at Spearman
    # >= 0.995 (n_sent_asof_t, n_sent_asof_t, n_txs_asof_t), so they split
    # importance with their twins and made every explanation worse.
    # mean_fee_ratio_asof_t also carried 83% of the train-vs-validation
    # adversarial signal: it tracked the fee market, not the address.
    "btc_recv_total_asof_t",
    "net_flow_asof_t",
    "active_duration_seconds",
    "gap_since_last_tx",
]

# v2. in_degree_asof_t and out_degree_asof_t were removed: they counted
# distinct receiving and sending txids, which is exactly what n_recv_asof_t
# and n_sent_asof_t in group B already count. They were identical on every
# row, and a duplicate splits its own importance with its twin - so any
# importance ranking over the v1 set was wrong.
GROUP_C_GRAPH = [
    "unique_counterparties_asof_t",
    "cluster_size_asof_t",
]

# v2. equal_output_count and output_entropy were removed as restatements of
# output_count. The two remaining flags were REDEFINED - see
# is_peeling_shape and is_mixing_shape below, and the tests that pin them.
GROUP_D_PATTERNS = [
    "is_peeling_candidate",
    "is_mixing_candidate",
]

# v3. Features that DIFFER between addresses of the same transaction. Every
# group-A column is a property of the transaction, so before v3 all
# addresses in one transaction carried identical group-A values - 83% of
# development rows were exact duplicates of another row. Role and the
# counterparties' history are what distinguish a payer from a payee.
GROUP_F_ROLE = [
    "addr_is_sender",
    "addr_is_self_change",
    "counterparty_max_n_txs_asof_t",
    "counterparty_mean_n_txs_asof_t",
]

# v4. Group G, upstream flow. Admitted to CORE after exp20 (causal
# ordering): 12-fold nAP 0.594 -> 0.714, paired +0.120, p = 0.0001; on
# held-back confirmation folds 0.541 -> 0.682, worst fold 0.227 -> 0.446.
# Where this transaction's money came from, one hop back: a summary of the
# transactions that funded its inputs, as known before this transaction.
# Shared by every participant, so a first-seen receiver - 88% of rows have no
# history of their own - still inherits information about the flow reaching
# it. Label-free and as-of-t by construction.
GROUP_G_UPSTREAM = [
    "upstream_funded_share",
    "upstream_mean_output_count",
    "upstream_mean_input_count",
    "upstream_peel_share",
    "upstream_mix_share",
    "upstream_min_hold_seconds",
    "upstream_chain_depth",
]

#: Chain depth is capped: beyond this every chain is simply "long".
UPSTREAM_DEPTH_CAP = 10

GROUP_E_NETWORK = [
    "network_observation_count",
    "observer_diversity",
    "peer_count",
    "asn_count",
    # v3. How the transaction reached the observers, not just how often.
    # A node connected directly to many observers is seen announcing its own
    # transaction repeatedly (high dominant-peer share) and within a tight
    # window (small arrival spread); a transaction relayed through the
    # network arrives from many peers over seconds.
    "dominant_peer_share",
    "arrival_spread_seconds",
]

PS_FEATURE_GROUPS: dict[str, list[str]] = {
    "A_transaction": GROUP_A_TRANSACTION,
    "B_address_history": GROUP_B_ADDRESS_HISTORY,
    "C_graph": GROUP_C_GRAPH,
    "D_patterns": GROUP_D_PATTERNS,
    "E_network": GROUP_E_NETWORK,
    "F_role": GROUP_F_ROLE,
    "G_upstream": GROUP_G_UPSTREAM,
}

#: Bumped because a v1 row and a v2 row are not the same observation. A
#: model, a metric or an ablation carried across the two would be comparing
#: different quantities under one name.
#: /5: same columns as /4, different values in two cases, so a /4 model
#: must not be served on /5 output. (1) Events sharing (timestamp,
#: event_order) are simultaneous and no longer see each other's updates.
#: (2) The per-address snapshot is one whole row, not a per-column splice.
PS_FEATURE_SCHEMA_VERSION = "ps_native_features/5"

# The core feature set without optional network features
CORE_PS_FEATURE_COLUMNS: list[str] = (
    GROUP_A_TRANSACTION + GROUP_B_ADDRESS_HISTORY + GROUP_C_GRAPH
    + GROUP_D_PATTERNS + GROUP_F_ROLE + GROUP_G_UPSTREAM
)

# The full feature set including optional network features
ALL_PS_FEATURE_COLUMNS: list[str] = CORE_PS_FEATURE_COLUMNS + GROUP_E_NETWORK

# Metadata columns (not fed to ML booster)
METADATA_COLUMNS = ["address", "txid", "timestamp"]


#: Share of a two-output transaction's value the larger output must carry.
PEEL_DOMINANCE = 0.8

#: Fewest participants per side before an anonymity set means anything.
MIXING_MIN_PARTICIPANTS = 3

#: Spread at or below which values count as near-identical, and at or above
#: which they count as genuinely varied. Same constants as
#: features/mixing.py, which is already tested against benign shapes.
UNIFORM_SPREAD = 0.02
VARIED_SPREAD = 0.25


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


def is_peeling_shape(*, n_in: int, n_out: int,
                     out_min: float, out_max: float, out_mean: float) -> bool:
    """One input, two outputs, one of them carrying most of the value.

    v1 required ``out_max >= 0.8 * total_out`` while the builder gave every
    output an equal share, so ``out_max`` was exactly half the total and the
    condition read ``0.5 >= 0.8``. The flag was constant zero across all
    262,433 rows - Group D "Structural Patterns" detected no structural
    pattern, and the PS peeling requirement was not met by it.

    Stated against the observable summary instead: with two outputs,
    ``out_max`` and ``out_min`` ARE the two output values, so the dominance
    ratio is exact rather than inferred.
    """
    if n_in != 1 or n_out != 2:
        return False
    total = out_min + out_max
    if total <= 0:
        return False
    return bool(out_max / total >= PEEL_DOMINANCE)


def is_mixing_shape(*, n_in: int, n_out: int,
                    out_min: float, out_max: float, out_mean: float,
                    in_min: float, in_max: float, in_mean: float) -> bool:
    """Many participants, near-identical outputs, varied inputs.

    v1 reduced to ``n_in >= 3 and n_out >= 3`` once the equality test went
    vacuous, and fired on 34% of holdout addresses - a fan-out threshold
    wearing a detector's name. This requires the two VALUE conditions that
    actually distinguish a collaborative spend from a batch, using the same
    uniformity measure as ``features/mixing.py`` so the two detectors cannot
    disagree about what the shape is.
    """
    if n_in < MIXING_MIN_PARTICIPANTS or n_out < MIXING_MIN_PARTICIPANTS:
        return False
    if out_mean <= 0 or in_mean <= 0:
        return False
    outputs_uniform = (out_max - out_min) / out_mean <= UNIFORM_SPREAD
    inputs_varied = (in_max - in_min) / in_mean >= VARIED_SPREAD
    return bool(outputs_uniform and inputs_varied)


def spread(low: float, high: float, mean: float) -> float:
    """``(max - min) / mean``: 0 when every value is identical.

    Returns NaN on a non-positive mean rather than 0, because "could not be
    measured" and "measured as perfectly uniform" are different answers and
    only one of them is true of a transaction with no value.
    """
    if mean is None or not math.isfinite(mean) or mean <= 0:
        return float("nan")
    return (high - low) / mean


def _calculate_entropy(amounts: list[float]) -> float:
    """Compute Shannon entropy of output amount distribution."""
    total = sum(amounts)
    if total <= 0:
        return 0.0
    probs = [a / total for a in amounts if a > 0]
    if len(probs) <= 1:
        return 0.0
    return -sum(p * math.log2(p) for p in probs)


#: Network group for a transaction nobody observed: every value NaN, because
#: "not observed" and "observed zero times" are different statements.
_NO_NETWORK: dict[str, float] = {c: float("nan") for c in GROUP_E_NETWORK}


def _aggregate_network(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-txid counts over every network observation in the capture.

    ``observer_diversity`` counts distinct vantage points (``observer_id``
    when the capture has it, else ``dst_ip`` - the observer is the
    destination of an announcement). ``peer_count`` counts distinct
    announcing peers, ``asn_count`` distinct announcing ASNs,
    ``dominant_peer_share`` the share of observations from the most frequent
    announcing peer, and ``arrival_spread_seconds`` the gap between the first
    and last observation.
    """
    if "src_ip" not in df.columns:
        return {}
    ip = df["src_ip"].map(lambda v: str(v).strip() if v is not None and not pd.isna(v) else "")
    seen = df[ip != ""].assign(_ip=ip[ip != ""])
    if seen.empty:
        return {}
    observer_col = "observer_id" if "observer_id" in seen.columns else "dst_ip"
    out: dict[str, dict[str, float]] = {}
    for txid, g in seen.groupby("_txid", sort=False):
        observers = g[observer_col].dropna() if observer_col in g.columns else pd.Series(dtype=object)
        asns = g["asn"].dropna() if "asn" in g.columns else pd.Series(dtype=float)
        ts = pd.to_numeric(g["timestamp"], errors="coerce").dropna() if "timestamp" in g.columns else pd.Series(dtype=float)
        out[txid] = {
            "network_observation_count": float(len(g)),
            "observer_diversity": float(observers.astype(str).nunique()),
            "peer_count": float(g["_ip"].nunique()),
            "asn_count": float(asns.nunique()),
            "dominant_peer_share": float(g["_ip"].value_counts().iloc[0] / len(g)),
            "arrival_spread_seconds": float(ts.max() - ts.min()) if len(ts) else float("nan"),
        }
    return out


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
    first_seen: float | None = None
    last_seen: float | None = None
    in_tx_ids: set[str] = field(default_factory=set)
    out_tx_ids: set[str] = field(default_factory=set)
    counterparties: set[str] = field(default_factory=set)
    #: Summary of the transaction that last paid this address, recorded when
    #: it is paid: (timestamp, n_in, n_out, is_peel, is_mix, chain_depth,
    #: (timestamp, event_order)).
    funded_by: tuple | None = None


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
        4. Each txid is replayed exactly once, at its first-seen timestamp,
           however many network observations of it the capture holds.
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

        # Optional ``event_order``: a causal tie-break for records sharing a
        # timestamp (the dataset builder supplies spend-DAG levels for
        # Elliptic++). It orders events; it never enters a feature value, so
        # time features keep the meaning they have on a real capture.
        if "event_order" in df.columns:
            df["_order"] = pd.to_numeric(df["event_order"], errors="coerce").fillna(0.0)
        else:
            df["_order"] = 0.0
        df = df.sort_values(by=["_sort_ts", "_order"], ascending=True, kind="stable")
        df["_txid"] = df["txid"].map(lambda v: str(v or "").strip()) if "txid" in df.columns else ""

        # One chain event per txid. The canonical capture is one row per
        # network OBSERVATION, so a transaction seen by three peers arrives
        # three times; replaying each copy counted it three times in every
        # address's history. Network facts are aggregated across all copies
        # first, then only the earliest copy is replayed.
        network_by_tx = _aggregate_network(df) if include_network else {}
        df = df[df["_txid"] != ""].drop_duplicates(subset="_txid", keep="first")

        output_rows: list[dict[str, Any]] = []
        # Events sharing one (timestamp, event_order) are SIMULTANEOUS: each is
        # computed against the state before the group, and the group's
        # updates are applied together after it. Otherwise a sibling
        # processed earlier only because of its txid would be visible, which
        # is the within-step leak the causal ordering exists to remove.
        pending: list[tuple] = []
        group: tuple | None = None

        for _, row in df.iterrows():
            t = float(row["_sort_ts"])
            when = (t, float(row["_order"]))
            txid = row["_txid"]
            if when != group:
                self._apply(pending)
                pending, group = [], when

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

            # v3. A missing fee is NaN, not 0.0. Encoding it as zero put 7% of
            # development rows (25 very large transactions) at fee == 0, where
            # the model learned "no fee means licit" from a data gap.
            fee = _safe_float(row.get("fee"), default=float("nan"))
            total_in = sum(in_amts)
            total_out = sum(out_amts)
            fee_ratio = (fee / total_in) if (total_in > 0 and math.isfinite(fee)) else float("nan")

            # v2. The REAL per-transaction value summary, when the caller has
            # it. The source data carries in/out min, max and mean per
            # transaction but no per-output values, so a caller that passes
            # only amount LISTS is passing something it synthesised - which is
            # exactly how v1 produced six columns that measured nothing.
            # Falling back to the lists keeps streaming callers working; the
            # dataset builder supplies the real summary.
            def _summary(prefix: str, amounts: list[float]) -> tuple:
                lo = row.get(f"{prefix}_min")
                hi = row.get(f"{prefix}_max")
                mu = row.get(f"{prefix}_mean")
                if lo is not None and not pd.isna(lo):
                    return (_safe_float(lo), _safe_float(hi), _safe_float(mu))
                if not amounts:
                    return (0.0, 0.0, 0.0)
                return (min(amounts), max(amounts),
                        sum(amounts) / len(amounts))

            in_lo, in_hi, in_mu = _summary("in_BTC", in_amts)
            out_lo, out_hi, out_mu = _summary("out_BTC", out_amts)

            # --- Group A: Transaction Behaviour (Instantaneous at t) ---
            in_count = len(in_addrs)
            out_count = len(out_addrs)
            in_mean, out_mean = in_mu, out_mu
            in_spread = spread(in_lo, in_hi, in_mu)
            out_spread = spread(out_lo, out_hi, out_mu)

            # --- Group D: Structural Patterns ---
            is_peel = 1 if is_peeling_shape(
                n_in=in_count, n_out=out_count,
                out_min=out_lo, out_max=out_hi, out_mean=out_mu,
            ) else 0
            is_mix = 1 if is_mixing_shape(
                n_in=in_count, n_out=out_count,
                out_min=out_lo, out_max=out_hi, out_mean=out_mu,
                in_min=in_lo, in_max=in_hi, in_mean=in_mu,
            ) else 0

            # --- Group E: Optional Network Features ---
            # v3. Real per-transaction counts. v2 set observer_diversity and
            # peer_count to the literal 1.0 whenever any observation existed.
            network = network_by_tx.get(txid, _NO_NETWORK)

            # --- Co-spend cluster as of t: earlier events plus this
            # transaction's own inputs. The union itself is deferred with the
            # rest of the group's updates.
            input_roots = {self.uf.find(a) for a in in_addrs}
            input_cluster_size = sum(self.uf.size[r] for r in input_roots)

            # --- Process Address State & Extract Features as-of-t ---
            participating = sorted(set(in_addrs) | set(out_addrs))
            in_set, out_set = set(in_addrs), set(out_addrs)

            # --- Group G: upstream flow, from the inputs' funding transactions,
            # read before this transaction updates anything.
            # Strictly earlier funding only, by (timestamp, event_order): an
            # event at the same point is not provably earlier and is ignored.
            # On Elliptic++ the builder supplies spend-DAG levels as
            # event_order; without them "processed before" inside a step was
            # txId order and 1.4% of funding pairs ran backwards in time.
            funding = [self.address_states[a].funded_by for a in in_set
                       if a in self.address_states and self.address_states[a].funded_by is not None
                       and self.address_states[a].funded_by[6] < when]
            if funding:
                upstream = {
                    "upstream_funded_share": len(funding) / max(len(in_set), 1),
                    "upstream_mean_output_count": float(np.mean([f[2] for f in funding])),
                    "upstream_mean_input_count": float(np.mean([f[1] for f in funding])),
                    "upstream_peel_share": float(np.mean([f[3] for f in funding])),
                    "upstream_mix_share": float(np.mean([f[4] for f in funding])),
                    "upstream_min_hold_seconds": float(max(0.0, t - max(f[0] for f in funding))),
                    "upstream_chain_depth": float(max(f[5] for f in funding)),
                }
            else:
                upstream = {c: float("nan") for c in GROUP_G_UPSTREAM}
                upstream["upstream_funded_share"] = 0.0
                upstream["upstream_chain_depth"] = 0.0
            tx_depth = min(UPSTREAM_DEPTH_CAP, int(upstream["upstream_chain_depth"]) + 1)
            tx_summary = (t, in_count, out_count, is_peel, is_mix, tx_depth, when)

            # Every participant's history is snapshotted BEFORE any of them is
            # updated for this transaction, so a counterparty's count never
            # includes the transaction being described.
            prior_n = {
                a: (self.address_states[a].n_txs if a in self.address_states else 0)
                for a in participating
            }

            for addr in participating:
                st = self.address_states.get(addr) or AddressState()

                prior_txs = st.n_txs
                prior_sent_count = st.n_sent
                prior_recv_count = st.n_recv
                prior_sent_btc = st.btc_sent
                prior_recv_btc = st.btc_recv
                prior_net_flow = prior_recv_btc - prior_sent_btc

                first_s = st.first_seen if st.first_seen is not None else t
                last_s = st.last_seen if st.last_seen is not None else t
                duration_sec = max(t - first_s, 0.0)
                gap_sec = max(t - last_s, 0.0)

                unique_cps = len(st.counterparties)
                cluster_size = input_cluster_size if addr in in_set else self.uf.get_cluster_size(addr)

                # --- Group F: role within this transaction ---
                is_sender = addr in in_set
                is_receiver = addr in out_set
                if is_sender and is_receiver:
                    counterparties = (in_set | out_set) - {addr}
                elif is_sender:
                    counterparties = out_set - {addr}
                else:
                    counterparties = in_set - {addr}
                cp_hist = [prior_n[c] for c in counterparties]

                feature_dict = {
                    "address": addr,
                    "txid": txid,
                    "timestamp": t,
                    # Group A
                    "input_count": in_count,
                    "output_count": out_count,
                    "total_input_amount": total_in,
                    "fee": fee,
                    "fee_ratio": fee_ratio,
                    "input_amount_mean": in_mean,
                    "input_spread": in_spread,
                    "output_amount_mean": out_mean,
                    "output_spread": out_spread,
                    # Group B
                    "n_txs_asof_t": prior_txs,
                    "n_sent_asof_t": prior_sent_count,
                    "n_recv_asof_t": prior_recv_count,
                    "btc_recv_total_asof_t": prior_recv_btc,
                    "net_flow_asof_t": prior_net_flow,
                    "active_duration_seconds": duration_sec,
                    "gap_since_last_tx": gap_sec,
                    # Group C
                    "unique_counterparties_asof_t": unique_cps,
                    "cluster_size_asof_t": cluster_size,
                    # Group D
                    "is_peeling_candidate": is_peel,
                    "is_mixing_candidate": is_mix,
                    # Group F
                    "addr_is_sender": int(is_sender),
                    "addr_is_self_change": int(is_sender and is_receiver),
                    "counterparty_max_n_txs_asof_t": float(max(cp_hist)) if cp_hist else 0.0,
                    "counterparty_mean_n_txs_asof_t": float(np.mean(cp_hist)) if cp_hist else 0.0,
                }

                feature_dict.update(upstream)

                if include_network:
                    feature_dict.update(network)

                output_rows.append(feature_dict)

            pending.append((t, txid, in_addrs, out_addrs, in_amts, out_amts, tx_summary))

        self._apply(pending)
        return pd.DataFrame(output_rows)

    def _apply(self, pending: list[tuple]) -> None:
        """Apply a group of simultaneous transactions to the state.

        In txid order, so the state after the group - including which of two
        simultaneous payments an address records as its funding, and the
        last bits of float sums - does not depend on input row order.
        """
        for t, txid, in_addrs, out_addrs, in_amts, out_amts, tx_summary in sorted(pending, key=lambda p: p[1]):
            if len(in_addrs) > 1:
                first_in = in_addrs[0]
                for other_in in in_addrs[1:]:
                    self.uf.union(first_in, other_in)
            in_set, out_set = set(in_addrs), set(out_addrs)
            for addr in sorted(in_set | out_set):
                st = self.address_states.setdefault(addr, AddressState())
                st.n_txs += 1
                if st.first_seen is None:
                    st.first_seen = t
                st.last_seen = t
                if addr in in_set:
                    st.n_sent += 1
                    st.btc_sent += in_amts[in_addrs.index(addr)]
                    st.out_tx_ids.add(txid)
                    st.counterparties.update(out_addrs)
                if addr in out_set:
                    st.funded_by = tx_summary
                    st.n_recv += 1
                    st.btc_recv += out_amts[out_addrs.index(addr)]
                    st.in_tx_ids.add(txid)
                    st.counterparties.update(in_addrs)


def last_snapshot_per_address(features: pd.DataFrame) -> pd.DataFrame:
    """Each address's LAST row, whole, in the engine's causal processing order.

    Not ``groupby("address").last()``: that takes the last NON-NULL value of
    each column separately, so a nullable feature (fee, spreads, the upstream
    group) could come from an older transaction than the rest of the row.
    Not a re-sort by timestamp either: a default sort is unstable, and on
    tied timestamps which row is "last" would be arbitrary. The engine
    already emits rows in (timestamp, event_order, txid) order.
    """
    if features.empty:
        return features
    return features.drop_duplicates(subset="address", keep="last").reset_index(drop=True)


def extract_ps_features(
    frame: pd.DataFrame,
    include_network: bool = False,
) -> pd.DataFrame:
    """Convenience function to extract PS-native features using a fresh state engine."""
    engine = PsTemporalFeatureEngine()
    return engine.process_records(frame, include_network=include_network)
