"""GET /api/alerts/{alert_id}/patterns - structural patterns behind an alert.

Two structural signals, in one place, for one alert:

``peeling``
    The PEEL-1 chain features the Phase 6 model already consumes (M2), read
    back off ``alert_members`` so the investigator can see WHY the model saw
    a chain signal rather than only that it did.

``mixing``
    The mixing / CoinJoin-like classification of the transactions this
    alert's members took part in, read from ``tx_mixing.parquet``.

Why mixing comes from a separate artifact
-----------------------------------------
``tx_mixing.parquet`` is ADDITIVE. It is written by ``mixing-scan`` and takes
no part in the Phase 7 run fingerprint, so an existing alert id, an existing
case binding and an existing stored alert reference all keep meaning exactly
what they meant. Folding the classification into the five alert artifacts
would have required regenerating them, which changes every alert id and
silently re-points every case that referenced one.

The cost of that choice is honest and stated in the response: the scan and
the alert run are separate artifacts, so the response carries both
identifiers and says the join is by txid.

Nothing here computes a pattern. The classification was made by
``mixing-scan``; this reads rows and groups them.
"""

from __future__ import annotations

from pathlib import Path

from obsidianchain.alerts import contract as alert_contract
from obsidianchain.api import artifacts, provenance_gate

# Nothing from obsidianchain.features is imported here. The mixing
# vocabulary lives in alerts/contract.py precisely so this layer can name
# a class without the detector that produces one being reachable from a
# request handler.

#: Written by ``mixing-scan``, relative to the data root.
TX_MIXING = Path("processed") / "tx_mixing.parquet"

#: M2 columns an investigator can actually read. The remaining M2 features
#: are model inputs rather than things to put in front of a person.
PEEL_FIELDS = (
    "in_chain", "chain_depth_max", "position_in_chain", "chain_count",
    "chain_fanout_mean", "hop_gap_median",
)

MIXING_UNAVAILABLE = (
    "The transaction-structure scan has not been generated for this "
    "deployment, so no mixing-like pattern has been measured. This is "
    "'not measured', not 'measured and clean'. Run 'make run "
    'ARGS="mixing-scan"\' to produce it.'
)

PEEL_UNAVAILABLE = (
    "This alert's artifact carries no PEEL-1 chain columns, so no "
    "peeling-chain-like structure has been measured for its members."
)


def _clean(value):
    import pandas as pd

    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def load_tx_mixing(root=None):
    """Read the scan, or return ``(None, None)`` when it was never made.

    Absence is a legitimate deployment state - the scan is additive and a
    deployment may not have run it - so it is reported rather than raised.
    An unreadable or non-production scan IS raised, because that is a
    different thing entirely.
    """
    import pandas as pd

    path = artifacts.data_root(root) / TX_MIXING
    if not path.is_file():
        return None, None
    meta = provenance_gate.require_production(path, require_inputs=False)
    return pd.read_parquet(path), meta


def get_patterns(raw_id: str, root=None, *, limit: int = 50) -> dict:
    """Peeling and mixing structure for one alert. Reads, never computes."""
    requested_run, cluster_id = alert_contract.parse_alert_id(raw_id)

    frame, sidecar = artifacts.load_alerts(root)
    current = str(sidecar.get("run_fingerprint") or "")[:16]
    if requested_run != current:
        raise alert_contract.AlertIdStaleError(
            f"alert id {raw_id!r} was minted against run {requested_run!r}; "
            f"the artifact on disk is run {current!r}."
        )
    match = frame[frame["cluster_id"] == cluster_id]
    if match.empty:
        raise alert_contract.AlertNotFoundError(
            f"no alert for cluster {cluster_id} in run {current}."
        )
    alert = match.iloc[0].to_dict()

    members = artifacts.load_alert_table(root, "alert_members")
    members = members[members["alert_id"] == raw_id]

    return {
        "alert_id": raw_id,
        "cluster_id": cluster_id,
        "run_fingerprint": current,
        "peeling": _peeling(alert, members, limit),
        "mixing": _mixing(raw_id, root, limit),
        "category": alert_contract.BLOCKCHAIN_CONTEXT,
    }


def _peeling(alert, members, limit: int) -> dict:
    """PEEL-1 structure, as the model already saw it.

    Deliberately reports ``members_in_chain`` out of ``members_scored``
    rather than a bare count: 3 of 4 members and 3 of 4,000 are different
    facts about a cluster.
    """
    present = [c for c in PEEL_FIELDS if c in members.columns]
    if not present or members.empty:
        return {
            "available": False,
            "status": alert_contract.INSUFFICIENT_EVIDENCE,
            "detail": PEEL_UNAVAILABLE,
            "meaning": alert_contract.PEEL_STRUCTURE_MEANING,
        }

    in_chain = members["in_chain"] if "in_chain" in members.columns else None
    n_in_chain = int(in_chain.fillna(0).astype(bool).sum()) if in_chain is not None else 0
    depths = (
        members["chain_depth_max"] if "chain_depth_max" in members.columns
        else None
    )

    rows = members.sort_values(
        "chain_depth_max" if depths is not None else present[0],
        ascending=False,
    ).head(limit)

    return {
        "available": True,
        "status": alert_contract.BLOCKCHAIN_CONTEXT,
        "members_scored": int(len(members)),
        "members_in_chain": n_in_chain,
        # The artifact already carries this on the alert row; reported
        # alongside so the two cannot silently disagree.
        "peel_chain_members_recorded": int(alert.get("peel_chain_members", 0)),
        "max_chain_depth": _clean(depths.max()) if depths is not None else None,
        "fields": present,
        "members": [
            {"address": row["address"],
             **{field: _clean(row[field]) for field in present}}
            for row in rows[["address", *present]].to_dict("records")
        ],
        "meaning": alert_contract.PEEL_STRUCTURE_MEANING,
    }


def _mixing(raw_id: str, root, limit: int) -> dict:
    """Mixing-like structure of this alert's correlated transactions.

    The transactions come from ``alert_network`` - the same table the
    correlation panel uses - because that is where an alert's txids are
    recorded. An alert with no correlated transactions therefore has no
    measurable mixing structure, which is reported as insufficient evidence
    rather than as an absence of pattern.
    """
    scan, meta = load_tx_mixing(root)
    if scan is None:
        return {
            "available": False,
            "status": alert_contract.INSUFFICIENT_EVIDENCE,
            "detail": MIXING_UNAVAILABLE,
            "meaning": alert_contract.MIXING_PATTERN_MEANING,
        }

    network = artifacts.load_alert_table(root, "alert_network")
    txids = set(network[network["alert_id"] == raw_id]["txid"].astype(str))
    if not txids:
        return {
            "available": False,
            "status": alert_contract.INSUFFICIENT_EVIDENCE,
            "detail": (
                "No transactions are correlated to this alert, so the "
                "structure of its transactions could not be measured."
            ),
            "meaning": alert_contract.MIXING_PATTERN_MEANING,
        }

    rows = scan[scan["txId"].astype(str).isin(txids)]
    counts = {
        str(k): int(v) for k, v in rows["mixing_class"].value_counts().items()
    }
    suppressed = {
        str(k): int(v) for k, v in rows["suppressor"].value_counts().items()
        if str(k)
    }

    shown = rows.sort_values("mixing_score", ascending=False).head(limit)
    return {
        "available": True,
        "status": alert_contract.BLOCKCHAIN_CONTEXT,
        "scan_id": str(meta.get("run_fingerprint") or "")[:16],
        "detector": (meta.get("artifact") or {}).get("detector"),
        "join_basis": (
            "The scan and the alert run are separate artifacts, joined by "
            "txid. The scan is additive and takes no part in the alert run "
            "fingerprint, so it changes no alert id."
        ),
        "transactions_measured": int(len(rows)),
        "transactions_correlated": int(len(txids)),
        "classes": counts,
        "pattern_count": counts.get(alert_contract.MIXING_PATTERN, 0),
        "suppressed": suppressed,
        "suppressor_meanings": dict(alert_contract.MIXING_SUPPRESSORS),
        "transactions": [
            {
                "txid": str(row["txId"]),
                "mixing_class": row["mixing_class"],
                "mixing_score": _clean(row["mixing_score"]),
                "output_uniformity": _clean(row["output_uniformity"]),
                "input_heterogeneity": _clean(row["input_heterogeneity"]),
                "participant_symmetry": _clean(row["participant_symmetry"]),
                "suppressor": row["suppressor"] or None,
                "n_inputs": _clean(row["n_inputs"]),
                "n_outputs": _clean(row["n_outputs"]),
            }
            for row in shown.to_dict("records")
        ],
        "meaning": alert_contract.MIXING_PATTERN_MEANING,
        "insufficient_data_meaning": alert_contract.MIXING_INSUFFICIENT_DATA_MEANING,
    }
