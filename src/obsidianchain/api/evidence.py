"""GET /api/evidence/{id} - the network evidence recorded for one proposed merge.

What this endpoint says, and what it refuses to say
---------------------------------------------------
It says: *here is the network evidence recorded for this proposed merge.*

It does not say the evidence proves the merge, and it does not say the
evidence caused the production engine to allow or block it. Both would be
false. The persisted funnel is the **chain-only trajectory** - every proposed
union was applied, none was vetoed - and no fused-engine decision ledger is
persisted anywhere in this project. ``run --mode fused`` and
``fusion-summary`` write zero files.

No verdict is served, derived or otherwise. The persisted statistics were
computed under ``PROBE_CONFIG`` (min_pooled=1, min_observer=2), not under the
production rule (25, 5), so degrees of freedom in particular may differ under
it. Deriving a production verdict from them would be an approximation dressed
as a fact. The response states the mismatch in its own body rather than
hiding it in a comment.

``reason`` is not served either. It exists on ``SeparationEvidence`` and is
dropped by the funnel writer, and two of the six reachable rationales are
indistinguishable from the persisted fields - both leave ``dof=0, chi2=0,
p=1.0, effect=0``. A reconstructed reason on a "why was this flagged?" screen
is exactly the inferred claim this project has spent four phases removing.

Availability, not judgement
---------------------------
Two booleans, each a direct function of a persisted number:

``evidence_available``
    A statistic was actually computable: ``dof >= 1`` and ``p_value`` is not
    NaN. False for 251,914 of the 253,429 rows.

``decidable_under_production_rule``
    The persisted pooled count clears the production pooled gate:
    ``min_pooled >= production_rule.min_pooled_observations``.

Neither is a verdict. Together they distinguish *no usable evidence* from
*evidence that exists and does not support separation* - the distinction the
whole design turns on.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from obsidianchain import run_fingerprint as rf
from obsidianchain.api import artifacts, boundary

#: ``<run_fingerprint>:<edge_index>``. Anchored, and the index may not carry a
#: sign or leading zeros, so exactly one string addresses one row.
EVIDENCE_ID = re.compile(r"^([0-9a-f]{16}):(0|[1-9][0-9]*)$")

STATEMENT = "Here is the network evidence recorded for this proposed merge."

TRAJECTORY_NOTE = (
    "Recorded along the unconstrained chain-only baseline: every proposed "
    "union was applied. This is the network evidence recorded for the "
    "proposed merge. It is NOT a record of a fused-engine decision, and no "
    "fused decision ledger is persisted."
)

STATISTICS_NOTE = (
    "These statistics were computed under the probe configuration below to "
    "extract a value wherever one was computable. They were NOT computed "
    "under provenance.production_rule, and degrees of freedom in particular "
    "may differ under it. No verdict is derived from them here."
)

UNRESOLVED_ADDRESS = None


class EvidenceIdInvalidError(ValueError):
    """The id is not of the form ``<16 hex>:<non-negative int>``."""


class EvidenceIdStaleError(ValueError):
    """The id is well-formed but was minted against a different run."""


class EvidenceNotFoundError(LookupError):
    """The run matches but no row carries that edge_index."""


class EvidenceJoinError(ValueError):
    """An address code could not be resolved, or the index is a different run."""


@dataclass(frozen=True)
class EvidenceId:
    run_fingerprint: str
    edge_index: int

    def __str__(self) -> str:
        return f"{self.run_fingerprint}:{self.edge_index}"


def parse_evidence_id(raw: str) -> EvidenceId:
    """Parse and validate, refusing anything ambiguous.

    Leading zeros are rejected rather than normalised: ``…:007`` and ``…:7``
    would otherwise be two ids for one row, and an id that is not canonical
    cannot be compared for equality.
    """
    if not isinstance(raw, str):
        raise EvidenceIdInvalidError(f"expected a string, got {type(raw).__name__}")
    match = EVIDENCE_ID.match(raw.strip())
    if match is None:
        raise EvidenceIdInvalidError(
            f"{raw!r} is not a valid evidence id. Expected "
            f"'<run_fingerprint>:<edge_index>' where run_fingerprint is "
            f"{rf.FINGERPRINT_LENGTH} lowercase hex characters and "
            f"edge_index is a non-negative integer without leading zeros."
        )
    return EvidenceId(match.group(1), int(match.group(2)))


def current_run_fingerprint(sidecar: dict) -> str:
    """The public fingerprint of the artifact on disk.

    Read from the sidecar's persisted ``run_fingerprint`` when present, and
    otherwise recomputed from the sidecar's own ``inputs`` block. Either way
    the API never hashes a file: raw data may not exist on an API-only
    deployment, and a response must not depend on a file the endpoint does
    not own.
    """
    persisted = sidecar.get("run_fingerprint")
    if isinstance(persisted, str) and persisted:
        return rf.public_fingerprint(persisted)

    inputs = sidecar.get("inputs") or {}
    return rf.fingerprint_from_inputs(
        chain_addr_tx_sha256=inputs.get("chain_addr_tx_sha256"),
        chain_universe_sha256=inputs.get("chain_universe_sha256"),
        network_dataset_sha256=inputs.get("network_dataset_sha256"),
        heuristics=inputs.get("heuristics", rf.HEURISTICS_MULTI_INPUT),
        row_statistics_config=inputs.get(
            "row_statistics_config", rf.PROBE_ROW_CONFIG
        ),
    )


def _clean(value):
    """NaN -> None, numpy scalar -> Python scalar.

    NaN is not representable in JSON and 251,914 rows carry it. Emitting
    ``null`` says "not computed"; emitting ``NaN`` would either break the
    encoder or arrive as a string.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _side(code: int, size_at_record: int, index_by_code: dict) -> dict:
    """One side of the proposed merge, resolved through the Stage 0 index.

    An unresolvable code is reported as unresolved. It is never replaced with
    a plausible-looking address, and the surrounding request is failed by the
    caller rather than served with a hole, because a wrong address on a
    forensics screen is worse than no answer.
    """
    row = index_by_code.get(int(code))
    if row is None:
        return {
            "code": int(code),
            "address": UNRESOLVED_ADDRESS,
            "address_resolved": False,
            "cluster_id": None,
            "cluster_size": None,
            "component_size_at_record": int(size_at_record),
        }
    return {
        "code": int(code),
        "address": str(row["address"]),
        "address_resolved": True,
        # The FINAL cluster this address ended up in. Deliberately not called
        # a component: it is not the union-find root that stood at the moment
        # the union was proposed, and conflating the two would misdate it.
        "cluster_id": int(row["cluster_id"]),
        "cluster_size": int(row["cluster_size"]),
        "component_size_at_record": int(size_at_record),
    }


def _provenance_block(sidecar: dict) -> dict:
    inputs = sidecar.get("inputs") or {}
    return {
        "provenance_type": sidecar.get("provenance_type"),
        "is_measurement": sidecar.get("is_measurement"),
        "synthetic_network": sidecar.get("synthetic_network"),
        "dataset_id": sidecar.get("dataset_id"),
        "network_dataset_sha256": inputs.get("network_dataset_sha256")
        or sidecar.get("dataset_sha256"),
        "chain_addr_tx_sha256": inputs.get("chain_addr_tx_sha256"),
        "chain_universe_sha256": inputs.get("chain_universe_sha256"),
        "generator_version": sidecar.get("generator_version"),
        "production_rule": sidecar.get("production_rule"),
        "heuristics": inputs.get("heuristics"),
        "notes": sidecar.get("notes", []),
    }


def get_evidence(raw_id: str, root=None) -> dict:
    """Load one evidence row and shape the response.

    Reads two precomputed artifacts and joins them. No clustering, no
    replay, no separation statistic, no evaluation.
    """
    parsed = parse_evidence_id(raw_id)

    funnel, sidecar = artifacts.load_evidence_funnel(root)
    current = current_run_fingerprint(sidecar)
    if parsed.run_fingerprint != current:
        raise EvidenceIdStaleError(
            f"evidence id {raw_id!r} was minted against run "
            f"{parsed.run_fingerprint!r}; the artifact on disk is run "
            f"{current!r}. One of the determining inputs changed - a chain "
            f"file, the network dataset, the heuristic, or the row "
            f"statistics configuration - so this id cannot be resolved "
            f"against the current run."
        )

    matches = funnel[funnel["edge_index"] == parsed.edge_index]
    if matches.empty:
        raise EvidenceNotFoundError(
            f"edge_index {parsed.edge_index} is not present in the evidence "
            f"funnel ({len(funnel):,} rows). Not every co-spend edge yields a "
            f"row: an edge whose endpoints were already connected proposes no "
            f"union and is not recorded."
        )
    row = matches.iloc[0]

    index, index_sidecar = artifacts.load_address_clusters(root)
    index_inputs = index_sidecar.get("inputs") or {}
    funnel_inputs = sidecar.get("inputs") or {}
    for key in ("chain_addr_tx_sha256", "chain_universe_sha256"):
        if index_inputs.get(key) != funnel_inputs.get(key):
            raise EvidenceJoinError(
                f"the address index was built from a different chain input "
                f"({key} differs from the evidence funnel's). Resolving codes "
                f"across two chains would return addresses that were never "
                f"party to this proposed merge. Rebuild both from one chain."
            )

    clusters, _ = artifacts.load_clusters(root)
    sizes = dict(
        zip(clusters["cluster_id"].tolist(), clusters["size"].tolist())
    )
    wanted = {int(row["node_a"]), int(row["node_b"])}
    index_by_code = {
        int(r.code): {
            "address": r.address,
            "cluster_id": int(r.cluster_id),
            "cluster_size": sizes.get(int(r.cluster_id)),
        }
        for r in index[index["code"].isin(wanted)].itertuples()
    }

    side_a = _side(row["node_a"], row["size_a"], index_by_code)
    side_b = _side(row["node_b"], row["size_b"], index_by_code)
    unresolved = [s["code"] for s in (side_a, side_b) if not s["address_resolved"]]
    if unresolved:
        raise EvidenceJoinError(
            f"address code(s) {unresolved} could not be resolved through the "
            f"Stage 0 index. No address is invented for an unresolved code; "
            f"the request fails instead."
        )

    dof = _clean(row["dof"])
    p_value = _clean(row["p_value"])
    evidence_available = bool(dof) and dof >= 1 and p_value is not None
    production_rule = sidecar.get("production_rule") or {}
    pooled_gate = production_rule.get("min_pooled_observations")
    min_pooled = int(row["min_pooled"])

    payload = {
        "evidence_id": str(EvidenceId(current, parsed.edge_index)),
        "edge_index": int(row["edge_index"]),
        "run_fingerprint": current,
        "provenance": _provenance_block(sidecar),
        "statistics_config": {
            "name": "probe",
            "min_pooled_observations": 1,
            "min_observer_observations": 2,
            "note": STATISTICS_NOTE,
        },
        "trajectory": {"name": "chain-only", "note": TRAJECTORY_NOTE},
        "proposed_merge": {"side_a": side_a, "side_b": side_b},
        "statistics": {
            "pooled_observations_a": int(row["pooled_a"]),
            "pooled_observations_b": int(row["pooled_b"]),
            "min_pooled": min_pooled,
            "dof": int(dof) if dof is not None else 0,
            "chi2": _clean(row["chi2"]),
            "p_value": p_value,
            "effect": _clean(row["effect"]),
        },
        "availability": {
            "evidence_available": evidence_available,
            "decidable_under_production_rule": (
                pooled_gate is not None and min_pooled >= pooled_gate
            ),
        },
        "statement": STATEMENT,
    }
    # Belt and braces. The funnel has no truth column, and this asserts the
    # response is clean anyway - a field can arrive through data as easily as
    # through an import.
    boundary.assert_no_truth_fields(payload)
    return payload
