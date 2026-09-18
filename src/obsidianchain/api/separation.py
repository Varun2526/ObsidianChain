"""GET /api/alerts/{alert_id}/separation-evidence.

The gap this closes
-------------------
``GET /api/evidence/{evidence_id}`` has served the separation funnel since
Phase 5.2, and nothing ever called it, because no route connected an alert to
the evidence rows recorded inside it. The network-separation layer is this
project's actual contribution - whether network-derived constraints can
reduce harmful clustering errors - and it was invisible in the product.

The join, and why it is exact rather than assumed
-------------------------------------------------
A funnel row records one PROPOSED MERGE during co-spend clustering, keyed by
``edge_index`` and naming the two components as ``node_a``/``node_b``. Those
are ADDRESS CODES, the same codes ``address_clusters.parquet`` carries. So a
cluster's internal merges are the funnel rows whose ``node_a`` belongs to
that cluster.

Two things make this a real join rather than a plausible one:

1. ``evidence_funnel`` and ``address_clusters`` are required to declare the
   SAME ``chain_addr_tx_sha256``, ``chain_universe_sha256`` and
   ``heuristics``. Those three inputs are exactly what determines an address
   code and a cluster id - see ``run_fingerprint``'s module docstring - so
   equal inputs mean one code space, and unequal inputs are refused.

2. The alert artifact's sidecar does NOT carry the chain hashes, so its
   cluster ids cannot be proved to share that space from provenance alone.
   Rather than assume it, this module VERIFIES it per request: the alert's
   own member addresses are looked up in ``address_clusters`` and must
   resolve to the alert's own ``cluster_id``. If they do not, the response is
   refused instead of served.

Nothing is fabricated, nothing is recomputed, and no verdict is derived here.
The verdicts are read from the persisted ``*_production`` columns exactly as
``separation_evidence()`` returned them.

What the verdicts do and do not mean
------------------------------------
The frozen wordings in ``evidence_contract`` travel with the response. In
particular NOT_SEPARATED is not evidence of common ownership and NO_EVIDENCE
is not evidence of anything at all - the network layer emits cannot-link
only, and can never establish that two address groups are the same party.
"""

from __future__ import annotations

from obsidianchain import evidence_contract as contract
from obsidianchain.alerts import contract as alert_contract
from obsidianchain.api import artifacts, evidence, provenance_gate

#: Rows returned in full. A large cluster has thousands of internal merges
#: and the distribution matters more than the tail, so the counts below are
#: computed over ALL of them and only the listing is capped.
ROWS_IN_DETAIL = 100

#: The three sidecar inputs that fix the address-code and cluster-id space.
CLUSTERING_BASIS = (
    "chain_addr_tx_sha256", "chain_universe_sha256", "heuristics",
)

STATEMENT = (
    "These are the network-separation records for the proposed merges that "
    "built this cluster. Each row asks whether two candidate components look "
    "DIFFERENT on the network layer. None of them asserts that they are the "
    "same."
)

JOIN_BASIS = (
    "Funnel rows are selected by address code: node_a is looked up in "
    "address_clusters and matched against this alert's cluster_id. The two "
    "artifacts are required to declare identical chain inputs and clustering "
    "heuristic, and the alert's own member addresses are verified to resolve "
    "to its cluster before any row is returned."
)

CANNOT_LINK_MEANING = (
    "A SEPARATED verdict is the only one that carries a constraint, and the "
    "constraint it carries is CANNOT-LINK: the two components look different "
    "enough that merging them is refused. The network layer never emits "
    "MUST-LINK. Two groups sharing network characteristics is not evidence "
    "that they are one party - one server broadcasts for tens of thousands "
    "of unrelated users."
)


class SeparationBasisError(ValueError):
    """The alert and the funnel cannot be shown to share a cluster space."""


def require_same_clustering_basis(primary: dict, secondary: dict) -> None:
    """Refuse two artifacts whose address codes may not mean the same thing.

    A mismatch here would join cleanly and describe different components
    under the right-looking heading - the same silent re-pointing that
    ``require_same_run`` prevents between the five alert tables.
    """
    left = (primary.get("inputs") or {})
    right = (secondary.get("inputs") or {})
    for key in CLUSTERING_BASIS:
        if left.get(key) is None or right.get(key) is None:
            raise provenance_gate.ProvenanceRefusedError(
                f"one of these artifacts does not record {key!r}, so a shared "
                f"address-code space cannot be established. Refusing to join."
            )
        if left[key] != right[key]:
            raise provenance_gate.ProvenanceRefusedError(
                f"the evidence funnel and the cluster index disagree on "
                f"{key!r}; their address codes address different things. "
                f"Refusing to join."
            )


def _clean(value):
    import pandas as pd

    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def get_separation_evidence(raw_id: str, root=None, *, limit=ROWS_IN_DETAIL) -> dict:
    """Separation records for one alert's cluster. Reads three artifacts."""
    requested_run, cluster_id = alert_contract.parse_alert_id(raw_id)

    alerts_frame, alerts_sidecar = artifacts.load_alerts(root)
    current = str(alerts_sidecar.get("run_fingerprint") or "")[:16]
    if requested_run != current:
        raise alert_contract.AlertIdStaleError(
            f"alert id {raw_id!r} was minted against run {requested_run!r}; "
            f"the artifact on disk is run {current!r}."
        )
    if alerts_frame[alerts_frame["cluster_id"] == cluster_id].empty:
        raise alert_contract.AlertNotFoundError(
            f"no alert for cluster {cluster_id} in run {current}."
        )

    index, index_sidecar = artifacts.load_address_clusters(
        root, columns=["code", "address", "cluster_id"]
    )
    funnel, funnel_sidecar = artifacts.load_evidence_funnel(root)
    provenance_gate.require_artifact_schema(
        funnel_sidecar, artifacts.evidence_funnel_path(root)
    )
    require_same_clustering_basis(funnel_sidecar, index_sidecar)

    # Verify, rather than assume, that the alert's clusters live in this
    # index's cluster space. See the module docstring.
    members = artifacts.load_alert_table(root, "alert_members")
    addresses = set(members[members["alert_id"] == raw_id]["address"])
    resolved = index[index["address"].isin(addresses)]
    if addresses and (
        resolved.empty or set(resolved["cluster_id"]) != {cluster_id}
    ):
        raise SeparationBasisError(
            f"this alert's member addresses do not resolve to cluster "
            f"{cluster_id} in the cluster index on disk, so the two artifacts "
            f"do not share a cluster space and their rows cannot be joined. "
            f"Refusing to serve a join that would describe different "
            f"components."
        )

    codes = set(index[index["cluster_id"] == cluster_id]["code"])
    rows = funnel[funnel["node_a"].isin(codes) | funnel["node_b"].isin(codes)]

    funnel_run = str(funnel_sidecar.get("run_fingerprint") or "")[:16]
    verdicts = {
        str(k): int(v)
        for k, v in rows["verdict_production"].value_counts().items()
    }
    reason_codes = {
        str(k): int(v)
        for k, v in rows["reason_code_production"].value_counts().items()
    }

    shown = rows.sort_values("edge_index").head(max(1, int(limit)))
    return {
        "alert_id": raw_id,
        "cluster_id": cluster_id,
        "alert_run_fingerprint": current,
        "evidence_run_fingerprint": funnel_run,
        "statement": STATEMENT,
        "join_basis": JOIN_BASIS,
        "proposed_merges_total": int(len(rows)),
        "verdicts": verdicts,
        "reason_codes": reason_codes,
        "separated_count": int(verdicts.get("SEPARATED", 0)),
        "cannot_link_meaning": CANNOT_LINK_MEANING,
        "not_separated_meaning": contract.NOT_SEPARATED_MEANING,
        "verdict_scope": contract.VERDICT_SCOPE,
        "verdict_definition": contract.VERDICT_DEFINITION,
        "frozen_run_limitation": contract.FROZEN_RUN_LIMITATION,
        "reason_code_catalogue": dict(contract.REASON_CODES),
        "unreachable_reason_codes": dict(contract.UNREACHABLE_CODES),
        "rows_shown": int(len(shown)),
        "rows_withheld": int(max(0, len(rows) - len(shown))),
        "rows": [
            {
                # Resolvable through the existing GET /api/evidence/{id}.
                "evidence_id": f"{funnel_run}:{int(row['edge_index'])}",
                "edge_index": int(row["edge_index"]),
                "node_a": int(row["node_a"]),
                "node_b": int(row["node_b"]),
                "verdict": row["verdict_production"],
                "reason_code": row["reason_code_production"],
                "reason": row["reason_production"],
                "min_pooled": _clean(row["min_pooled"]),
                "size_a": _clean(row["size_a"]),
                "size_b": _clean(row["size_b"]),
                "chi2": _clean(row["chi2_production"]),
                "p_value": _clean(row["p_value_production"]),
                "effect": _clean(row["effect_production"]),
            }
            for row in shown.to_dict("records")
        ],
    }
