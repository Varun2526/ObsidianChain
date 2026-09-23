"""Feature catalog and feature-frame contract (``obsidianchain.feature_contract/1``).

One entry per feature the PS-native engine emits. Each entry states, for the
leakage audit (research/autoresearch_2026_09_23/19_leakage_audit.md):

* ``source``       - which observed fields it is computed from
* ``as_of``        - the information set it may use. Every entry is one of:
    TX_ITSELF      the transaction being described, nothing else
    PRIOR_EVENTS   events strictly before this one in (timestamp, event_order)
    PRIOR_EVENTS_AND_TX  both of the above
* ``same_step``    - behaviour for events sharing a timestamp: which are
                     visible (only those earlier in event_order)
* ``nullable``     - whether NaN is a legitimate value, and what it means
* ``range``        - (low, high) inclusive; None = unbounded on that side
* ``labels``       - always "none": no feature reads a label, class or
                     truth file. Enforced structurally by
                     tests/test_leakage_audit.py.

``validate_feature_frame`` refuses a frame whose values break the catalog
before any model sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

FEATURE_CONTRACT_VERSION = "obsidianchain.feature_contract/1"

TX = "TX_ITSELF"
PRIOR = "PRIOR_EVENTS"
BOTH = "PRIOR_EVENTS_AND_TX"

SAME_STEP_NA = "not applicable: a property of this transaction only"
SAME_STEP_PRIOR = ("only events earlier in (timestamp, event_order) are counted; "
                   "an event at the same point is not visible")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    source: str
    as_of: str
    same_step: str
    nullable: str | None
    """None: NaN is a contract violation. Otherwise what NaN means."""
    low: float | None = 0.0
    high: float | None = None
    integer: bool = False
    labels: str = "none"
    notes: str = ""


def _spec(name, group, source, as_of, same_step, nullable=None, low=0.0, high=None,
          integer=False, notes="") -> FeatureSpec:
    return FeatureSpec(name, group, source, as_of, same_step, nullable, low, high, integer, "none", notes)


_TXS = "input/output address lists and amounts of this transaction"
_SUM = "in_BTC/out_BTC min/max/mean summary when supplied, else the amount lists"
_HIST = "this address's state, updated only after each earlier event"
_UF = "co-spend union-find over earlier events and this transaction's inputs"

CATALOG: dict[str, FeatureSpec] = {s.name: s for s in [
    # ---- A: transaction (instantaneous) ----
    _spec("input_count", "A_transaction", _TXS, TX, SAME_STEP_NA, integer=True),
    _spec("output_count", "A_transaction", _TXS, TX, SAME_STEP_NA, integer=True),
    _spec("total_input_amount", "A_transaction", _TXS, TX, SAME_STEP_NA),
    _spec("fee", "A_transaction", "fee field", TX, SAME_STEP_NA,
          nullable="fee not supplied by the capture"),
    _spec("fee_ratio", "A_transaction", "fee / total_input_amount", TX, SAME_STEP_NA,
          nullable="fee missing or zero input value (not measurable)", high=1.0),
    _spec("input_amount_mean", "A_transaction", _SUM, TX, SAME_STEP_NA),
    _spec("output_amount_mean", "A_transaction", _SUM, TX, SAME_STEP_NA),
    _spec("input_spread", "A_transaction", _SUM + ": (max-min)/mean", TX, SAME_STEP_NA,
          nullable="non-positive mean (not measurable)"),
    _spec("output_spread", "A_transaction", _SUM + ": (max-min)/mean", TX, SAME_STEP_NA,
          nullable="non-positive mean (not measurable)"),
    # ---- B: address history ----
    _spec("n_txs_asof_t", "B_address_history", _HIST, PRIOR, SAME_STEP_PRIOR, integer=True),
    _spec("n_sent_asof_t", "B_address_history", _HIST, PRIOR, SAME_STEP_PRIOR, integer=True),
    _spec("n_recv_asof_t", "B_address_history", _HIST, PRIOR, SAME_STEP_PRIOR, integer=True),
    _spec("btc_recv_total_asof_t", "B_address_history", _HIST, PRIOR, SAME_STEP_PRIOR,
          notes="per-address amounts are even splits on Elliptic++ (source has no per-output values)"),
    _spec("net_flow_asof_t", "B_address_history", _HIST, PRIOR, SAME_STEP_PRIOR, low=None,
          notes="per-address amounts are even splits on Elliptic++"),
    _spec("active_duration_seconds", "B_address_history", "timestamp - first seen", BOTH, SAME_STEP_PRIOR,
          notes="on Elliptic++ a multiple of one timestep (surrogate timestamps)"),
    _spec("gap_since_last_tx", "B_address_history", "timestamp - last seen", BOTH, SAME_STEP_PRIOR,
          notes="on Elliptic++ a multiple of one timestep (surrogate timestamps)"),
    # ---- C: graph ----
    _spec("unique_counterparties_asof_t", "C_graph", _HIST, PRIOR, SAME_STEP_PRIOR, integer=True),
    _spec("cluster_size_asof_t", "C_graph", _UF, BOTH, SAME_STEP_PRIOR, low=1.0, integer=True),
    # ---- D: patterns ----
    _spec("is_peeling_candidate", "D_patterns", _SUM, TX, SAME_STEP_NA, high=1.0, integer=True),
    _spec("is_mixing_candidate", "D_patterns", _SUM, TX, SAME_STEP_NA, high=1.0, integer=True),
    # ---- F: role ----
    _spec("addr_is_sender", "F_role", _TXS, TX, SAME_STEP_NA, high=1.0, integer=True),
    _spec("addr_is_self_change", "F_role", _TXS, TX, SAME_STEP_NA, high=1.0, integer=True),
    _spec("counterparty_max_n_txs_asof_t", "F_role",
          "prior n_txs of the opposite-side addresses, snapshotted before this tx", BOTH, SAME_STEP_PRIOR),
    _spec("counterparty_mean_n_txs_asof_t", "F_role",
          "prior n_txs of the opposite-side addresses, snapshotted before this tx", BOTH, SAME_STEP_PRIOR),
    # ---- G: upstream flow ----
    _spec("upstream_funded_share", "G_upstream",
          "share of inputs whose funding transaction is strictly earlier", BOTH, SAME_STEP_PRIOR, high=1.0),
    _spec("upstream_mean_output_count", "G_upstream", "funding transactions of the inputs", PRIOR,
          SAME_STEP_PRIOR, nullable="no input has a strictly earlier funding transaction"),
    _spec("upstream_mean_input_count", "G_upstream", "funding transactions of the inputs", PRIOR,
          SAME_STEP_PRIOR, nullable="no input has a strictly earlier funding transaction"),
    _spec("upstream_peel_share", "G_upstream", "funding transactions of the inputs", PRIOR,
          SAME_STEP_PRIOR, nullable="no input has a strictly earlier funding transaction", high=1.0),
    _spec("upstream_mix_share", "G_upstream", "funding transactions of the inputs", PRIOR,
          SAME_STEP_PRIOR, nullable="no input has a strictly earlier funding transaction", high=1.0),
    _spec("upstream_min_hold_seconds", "G_upstream", "timestamp - latest funding timestamp", BOTH,
          SAME_STEP_PRIOR, nullable="no input has a strictly earlier funding transaction",
          notes="0 for same-step funding on Elliptic++ (surrogate timestamps)"),
    _spec("upstream_chain_depth", "G_upstream", "funding-chain length, capped at 10", BOTH,
          SAME_STEP_PRIOR, high=10.0, integer=True),
    # ---- E: network (optional, not in the model's CORE) ----
    _spec("network_observation_count", "E_network", "capture rows for this txid", TX, SAME_STEP_NA,
          nullable="no observation of this txid", integer=True),
    _spec("observer_diversity", "E_network", "distinct observer_id / dst_ip", TX, SAME_STEP_NA,
          nullable="no observation of this txid", integer=True),
    _spec("peer_count", "E_network", "distinct announcing src_ip", TX, SAME_STEP_NA,
          nullable="no observation of this txid", integer=True),
    _spec("asn_count", "E_network", "distinct announcing ASN", TX, SAME_STEP_NA,
          nullable="no observation of this txid", integer=True),
    _spec("dominant_peer_share", "E_network", "share of the modal src_ip", TX, SAME_STEP_NA,
          nullable="no observation of this txid", high=1.0),
    _spec("arrival_spread_seconds", "E_network", "last - first observation time", TX, SAME_STEP_NA,
          nullable="no observation of this txid or no timestamps"),
]}

#: Tolerance for ranges on values that are ratios of summaries.
_EPS = 1e-9


@dataclass
class FeatureContractReport:
    ok: bool
    violations: list[dict[str, Any]] = field(default_factory=list)
    version: str = FEATURE_CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {"version": self.version, "ok": self.ok, "violations": self.violations[:50],
                "violation_count": len(self.violations)}


def validate_feature_frame(frame: pd.DataFrame, features: list[str]) -> FeatureContractReport:
    """Check every model feature against its catalog entry.

    A feature with no catalog entry, a non-numeric column, an infinity, a NaN
    where the catalog forbids it, an out-of-range value or a non-integer
    count is a violation. Nothing is repaired here.
    """
    violations: list[dict[str, Any]] = []
    for name in features:
        spec = CATALOG.get(name)
        if spec is None:
            violations.append({"feature": name, "kind": "NOT_IN_CATALOG"})
            continue
        if name not in frame.columns:
            violations.append({"feature": name, "kind": "MISSING_COLUMN"})
            continue
        col = frame[name]
        if not pd.api.types.is_numeric_dtype(col) and not col.isna().all():
            violations.append({"feature": name, "kind": "NON_NUMERIC", "dtype": str(col.dtype)})
            continue
        values = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
        nan = np.isnan(values)
        if np.isinf(values).any():
            violations.append({"feature": name, "kind": "INFINITE", "rows": int(np.isinf(values).sum())})
        if nan.any() and spec.nullable is None:
            violations.append({"feature": name, "kind": "UNEXPECTED_NULL", "rows": int(nan.sum())})
        finite = values[np.isfinite(values)]
        if spec.low is not None and (finite < spec.low - _EPS).any():
            violations.append({"feature": name, "kind": "BELOW_RANGE", "min": float(finite.min()), "low": spec.low})
        if spec.high is not None and (finite > spec.high + _EPS).any():
            violations.append({"feature": name, "kind": "ABOVE_RANGE", "max": float(finite.max()), "high": spec.high})
        if spec.integer and finite.size and not np.allclose(finite, np.round(finite)):
            violations.append({"feature": name, "kind": "NON_INTEGER"})
    return FeatureContractReport(ok=not violations, violations=violations)


def catalog_table() -> pd.DataFrame:
    """The catalog as a frame, for documentation and audit output."""
    return pd.DataFrame([{
        "feature": s.name, "group": s.group, "source": s.source, "as_of": s.as_of,
        "same_step": s.same_step, "nullable": s.nullable or "never", "range": f"[{s.low}, {s.high}]",
        "labels": s.labels, "notes": s.notes,
    } for s in CATALOG.values()])
