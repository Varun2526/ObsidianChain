"""Feature adapter and LightGBM schema compatibility enforcement.

P0 CRITICAL SAFEGUARD:
    The feature adapter must produce an explicit feature-availability manifest.
    The existing LightGBM model may execute ONLY when every required model feature
    has a defined derivation or an explicitly validated missing-value representation
    supported by the trained model. Otherwise the supervised ML stage must return
    MODEL_UNAVAILABLE_FOR_SCHEMA, not fabricate or silently substitute features.

    Prevents:
        PS input → missing feature → 0 inserted → LightGBM score → "valid risk score"
        (which is scientifically misleading).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FeatureManifest:
    """Explicit availability manifest for required model features."""

    required_features: list[str]
    derivable_features: list[str]
    missing_features: list[str]
    feature_status: dict[str, str] = field(default_factory=dict)
    is_compatible: bool = False
    incompatibility_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_count": len(self.required_features),
            "derivable_count": len(self.derivable_features),
            "missing_count": len(self.missing_features),
            "is_compatible": self.is_compatible,
            "incompatibility_reason": self.incompatibility_reason,
            "feature_status": self.feature_status,
        }


@dataclass
class MlStageResult:
    """Supervised ML inference result or schema incompatibility report."""

    status: str
    """'SCORED' or 'MODEL_UNAVAILABLE_FOR_SCHEMA'."""
    manifest: FeatureManifest
    scores: pd.DataFrame | None = None
    reason: str | None = None

    @property
    def is_available(self) -> bool:
        return self.status == "SCORED" and self.scores is not None


def _parse_amount(v: Any) -> float:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0.0
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def derive_canonical_address_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive mathematically well-defined forensic features per address from PS schema.

    Never invents unobservable quantities. Every derived feature has a defined
    calculation from txid, addresses, amounts, fee, and timestamp.
    """
    records: list[dict[str, Any]] = []

    # Map addresses to transactions
    addr_sent: dict[str, list[float]] = {}
    addr_recv: dict[str, list[float]] = {}
    addr_txs: dict[str, set[str]] = {}
    addr_timestamps: dict[str, list[float]] = {}
    addr_fee_ratios: dict[str, list[float]] = {}

    for _, row in frame.iterrows():
        txid = str(row.get("txid") or "").strip()
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

        while len(in_amts) < len(in_addrs):
            in_amts.append(0.0)
        while len(out_amts) < len(out_addrs):
            out_amts.append(0.0)

        fee = _parse_amount(row.get("fee"))
        total_in = sum(in_amts)
        fee_ratio = (fee / total_in) if total_in > 0 else 0.0

        ts_val = None
        if pd.notna(row.get("timestamp")):
            try:
                ts_val = float(row.get("timestamp"))
            except (ValueError, TypeError):
                ts_val = None

        for a, amt in zip(in_addrs, in_amts):
            addr_sent.setdefault(a, []).append(amt)
            addr_txs.setdefault(a, set()).add(txid)
            addr_fee_ratios.setdefault(a, []).append(fee_ratio)
            if ts_val is not None:
                addr_timestamps.setdefault(a, []).append(ts_val)

        for a, amt in zip(out_addrs, out_amts):
            addr_recv.setdefault(a, []).append(amt)
            addr_txs.setdefault(a, set()).add(txid)
            addr_fee_ratios.setdefault(a, []).append(fee_ratio)
            if ts_val is not None:
                addr_timestamps.setdefault(a, []).append(ts_val)

    all_addrs = sorted(set(addr_sent.keys()) | set(addr_recv.keys()))

    for a in all_addrs:
        s_list = addr_sent.get(a, [])
        r_list = addr_recv.get(a, [])
        tx_set = addr_txs.get(a, set())
        ts_list = addr_timestamps.get(a, [])
        fr_list = addr_fee_ratios.get(a, [])

        sent_total = sum(s_list)
        recv_total = sum(r_list)
        n_txs = len(tx_set)
        n_in = len(s_list)
        n_out = len(r_list)

        duration_sec = (max(ts_list) - min(ts_list)) if len(ts_list) > 1 else 0.0
        duration_hours = max(duration_sec / 3600.0, 1.0 / 3600.0)  # at least 1 second
        velocity_tx_per_hour = n_txs / duration_hours if duration_sec > 0 else float(n_txs)

        mean_fee_ratio = float(np.mean(fr_list)) if fr_list else 0.0
        max_fee_ratio = float(np.max(fr_list)) if fr_list else 0.0

        records.append({
            "address": a,
            "n_txs": n_txs,
            "n_in_txs": n_in,
            "n_out_txs": n_out,
            "btc_sent": sent_total,
            "btc_received": recv_total,
            "net_flow": recv_total - sent_total,
            "mean_fee_ratio": mean_fee_ratio,
            "max_fee_ratio": max_fee_ratio,
            "velocity_tx_per_hour": velocity_tx_per_hour,
            "duration_seconds": duration_sec,
        })

    if not records:
        return pd.DataFrame(columns=[
            "address", "n_txs", "n_in_txs", "n_out_txs", "btc_sent",
            "btc_received", "net_flow", "mean_fee_ratio", "max_fee_ratio",
            "velocity_tx_per_hour", "duration_seconds",
        ])

    return pd.DataFrame(records)


def check_feature_compatibility(
    model_required_features: list[str],
    derivable_feature_names: set[str],
    validated_missing_features: set[str] | None = None,
) -> FeatureManifest:
    """Build the explicit feature-availability manifest.

    LightGBM model may execute ONLY when every required feature has a defined
    derivation or an explicitly validated missing-value representation.
    """
    validated = validated_missing_features or set()
    derivable: list[str] = []
    missing: list[str] = []
    status: dict[str, str] = {}

    for f in model_required_features:
        if f in derivable_feature_names:
            derivable.append(f)
            status[f] = "DERIVED"
        elif f in validated:
            status[f] = "VALIDATED_MISSING"
        else:
            missing.append(f)
            status[f] = "FEATURE_UNAVAILABLE"

    is_compatible = len(missing) == 0
    incomp_reason = None
    if not is_compatible:
        incomp_reason = (
            f"Supervised model requires {len(missing)} feature(s) that cannot be "
            f"derived from Problem Statement canonical schema without unscientific "
            f"fabrication: {missing[:5]} (truncated)"
        )

    return FeatureManifest(
        required_features=list(model_required_features),
        derivable_features=derivable,
        missing_features=missing,
        feature_status=status,
        is_compatible=is_compatible,
        incompatibility_reason=incomp_reason,
    )


def execute_supervised_stage(
    model: Any,
    address_features: pd.DataFrame,
    validated_missing: set[str] | None = None,
) -> MlStageResult:
    """Execute supervised ML stage under the hard safeguard contract.

    If model is None or features are incompatible, returns MODEL_UNAVAILABLE_FOR_SCHEMA.
    NEVER silently substitutes 0.0 or fabricates missing features.
    """
    if model is None:
        manifest = FeatureManifest(
            required_features=[],
            derivable_features=list(address_features.columns),
            missing_features=[],
            is_compatible=False,
            incompatibility_reason="No trained supervised model was configured for this run.",
        )
        return MlStageResult(
            status="MODEL_UNAVAILABLE_FOR_SCHEMA",
            manifest=manifest,
            scores=None,
            reason="No supervised model configured.",
        )

    required_features = getattr(model, "features", [])
    derivable_names = set(address_features.columns)

    manifest = check_feature_compatibility(
        required_features,
        derivable_names,
        validated_missing_features=validated_missing,
    )

    if not manifest.is_compatible:
        return MlStageResult(
            status="MODEL_UNAVAILABLE_FOR_SCHEMA",
            manifest=manifest,
            scores=None,
            reason=manifest.incompatibility_reason,
        )

    # Compatible: model inference is permitted
    try:
        scored_df = address_features[["address"]].copy()
        if hasattr(model, "predict_address_features"):
            preds = model.predict_address_features(address_features)
            scored_df["ml_risk_score"] = [p.calibrated_risk_score for p in preds]
            # The RANKING score. Calibration is monotone, but the stacker and
            # the fusion consume the raw score the model was evaluated on.
            scored_df["ml_raw_score"] = [p.raw_risk_score for p in preds]
            scored_df["ml_severity"] = [p.severity for p in preds]
            scored_df["ml_explanations"] = [[e.__dict__ for e in p.explanations] for p in preds]
        else:
            raw_scores = model.scores(address_features)
            severity = model.severity(raw_scores)
            scored_df["ml_risk_score"] = raw_scores
            scored_df["ml_severity"] = severity

        return MlStageResult(
            status="SCORED",
            manifest=manifest,
            scores=scored_df,
            reason=None,
        )
    except Exception as exc:
        return MlStageResult(
            status="MODEL_UNAVAILABLE_FOR_SCHEMA",
            manifest=manifest,
            scores=None,
            reason=f"Model execution error: {exc}",
        )
