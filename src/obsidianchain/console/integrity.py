"""ObsidianChain Tamper-Evident Merkle Case-Integrity Layer.

Tamper-evident integrity verification; does not constitute legal
admissibility or automated chain-of-custody warranty.

This module provides deterministic leaf extraction, canonical hashing,
Merkle tree construction, inclusion proof generation and verification,
and historical root comparison for investigations.

Cryptographic Domain Separation:
- 0x00: Leaf node prefix
- 0x01: Internal/branch node prefix
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from obsidianchain.alerts import contract
from obsidianchain.console import audit, casework, db, errors, reports
from obsidianchain.console.users import User

# Standard conceptual disclaimer required across all integrity responses
INTEGRITY_DISCLAIMER = (
    "Tamper-evident integrity verification; does not constitute legal "
    "admissibility or automated chain-of-custody warranty."
)

LEAF_PREFIX = b"\x00"
BRANCH_PREFIX = b"\x01"

# Empty SHA-256 digest for a tree with 0 leaves
EMPTY_ROOT = hashlib.sha256(b"").hexdigest()

# Opportunistic high-performance orjson, fallback to deterministic stdlib json
try:
    import orjson  # type: ignore

    def canonical_json_bytes(obj: Any) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)
except ImportError:
    def canonical_json_bytes(obj: Any) -> bytes:
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class MerkleLeaf:
    """A deterministic leaf record before hashing."""
    kind: str       # "alert" | "evidence" | "note" | "disposition" | "report_version"
    key: str        # Deterministic identifier: f"{kind}:{id}"
    data: dict[str, Any]

    def hash(self) -> str:
        """Domain-separated leaf hash: SHA-256(0x00 || canonical_json(data))."""
        raw = LEAF_PREFIX + canonical_json_bytes(self.data)
        return hashlib.sha256(raw).hexdigest()


@dataclass
class MerkleTree:
    leaves: list[MerkleLeaf]
    leaf_hashes: list[str]
    layers: list[list[str]]
    root: str
    leaf_count: int
    calculated_at: str

    def get_proof(self, target_leaf_hash: str) -> list[dict[str, str]]:
        """Generate Merkle inclusion proof for a leaf hash."""
        if target_leaf_hash not in self.leaf_hashes:
            raise errors.NotFound(f"leaf {target_leaf_hash} is not in this Merkle tree")

        index = self.leaf_hashes.index(target_leaf_hash)
        proof: list[dict[str, str]] = []

        for layer in self.layers[:-1]:
            # If layer has an odd number of items, last node was duplicated to form pairs
            is_odd = (len(layer) % 2 == 1)
            pair_layer = list(layer)
            if is_odd:
                pair_layer.append(pair_layer[-1])

            if index % 2 == 0:
                # Sibling is to the right
                sibling_index = index + 1
                direction = "right"
            else:
                # Sibling is to the left
                sibling_index = index - 1
                direction = "left"

            sibling_hash = pair_layer[sibling_index]
            proof.append({
                "direction": direction,
                "sibling_hash": sibling_hash,
            })
            index = index // 2

        return proof


def combine_hashes(left_hex: str, right_hex: str) -> str:
    """Domain-separated branch hash: SHA-256(0x01 || left_bytes || right_bytes)."""
    left_bytes = bytes.fromhex(left_hex)
    right_bytes = bytes.fromhex(right_hex)
    return hashlib.sha256(BRANCH_PREFIX + left_bytes + right_bytes).hexdigest()


def build_merkle_tree(leaves: list[MerkleLeaf], *, calculated_at: str | None = None) -> MerkleTree:
    """Construct a deterministic Merkle tree from leaves.
    
    Leaves are deterministically sorted by (kind, key) prior to tree construction.
    """
    sorted_leaves = sorted(leaves, key=lambda l: (l.kind, l.key))
    calculated_at = calculated_at or db.utcnow()

    if not sorted_leaves:
        return MerkleTree(
            leaves=[],
            leaf_hashes=[],
            layers=[[EMPTY_ROOT]],
            root=EMPTY_ROOT,
            leaf_count=0,
            calculated_at=calculated_at,
        )

    leaf_hashes = [leaf.hash() for leaf in sorted_leaves]
    if len(leaf_hashes) == 1:
        return MerkleTree(
            leaves=sorted_leaves,
            leaf_hashes=leaf_hashes,
            layers=[leaf_hashes],
            root=leaf_hashes[0],
            leaf_count=1,
            calculated_at=calculated_at,
        )

    layers: list[list[str]] = [leaf_hashes]
    current_layer = leaf_hashes

    while len(current_layer) > 1:
        next_layer: list[str] = []
        n = len(current_layer)
        # Duplicate last node if odd number of items in layer
        padded = list(current_layer)
        if n % 2 == 1:
            padded.append(padded[-1])

        for i in range(0, len(padded), 2):
            parent = combine_hashes(padded[i], padded[i + 1])
            next_layer.append(parent)

        layers.append(next_layer)
        current_layer = next_layer

    root = current_layer[0]
    return MerkleTree(
        leaves=sorted_leaves,
        leaf_hashes=leaf_hashes,
        layers=layers,
        root=root,
        leaf_count=len(sorted_leaves),
        calculated_at=calculated_at,
    )


def verify_leaf_proof(leaf_data: dict[str, Any], proof: list[Mapping[str, str]], root: str) -> bool:
    """Verify an inclusion proof for given leaf data against a Merkle root."""
    try:
        raw = LEAF_PREFIX + canonical_json_bytes(leaf_data)
        current = hashlib.sha256(raw).hexdigest()

        for step in proof:
            direction = step.get("direction")
            sibling = step.get("sibling_hash")
            if not sibling or direction not in ("left", "right"):
                return False

            if direction == "left":
                current = combine_hashes(sibling, current)
            else:
                current = combine_hashes(current, sibling)

        return current.lower() == root.lower()
    except Exception:
        return False


def extract_leaves(
    conn: sqlite3.Connection,
    investigation_id: str,
    *,
    current_run: str | None = None,
    alerts_data: list[dict] | None = None,
) -> list[MerkleLeaf]:
    """Extract deterministic Merkle leaves for all case entities.
    
    Includes:
    - alerts
    - evidence (including canonical features and provenance content hashes)
    - investigator notes
    - dispositions
    - report versions
    """
    leaves: list[MerkleLeaf] = []

    # 1. Alert leaves & Evidence leaves
    alert_refs = casework.references(conn, investigation_id, current_run=current_run)
    for ref in alert_refs:
        alert_id = ref["alert_id"]
        try:
            run_fp, cluster_id = contract.parse_alert_id(alert_id)
        except Exception:
            run_fp, cluster_id = ref.get("run_fingerprint") or "", 0

        leaves.append(MerkleLeaf(
            kind="alert",
            key=f"alert:{alert_id}",
            data={
                "kind": "alert",
                "alert_id": alert_id,
                "cluster_id": cluster_id,
                "run_fingerprint": run_fp,
                "added_at": ref.get("added_at"),
                "added_by": ref.get("added_by"),
                "assigned_to": ref.get("assigned_to"),
            },
        ))

        # Evidence leaf: captures category, provenance run, canonical content hash
        # If alerts_data was supplied (e.g. from rich alert resolution), we extract
        # full feature group content; otherwise we extract the canonical evidence
        # identity and references.
        rich_evidence = None
        if alerts_data:
            for a in alerts_data:
                if a.get("alert_id") == alert_id:
                    rich_evidence = a.get("evidence")
                    break

        evidence_content_hash = ""
        summary_metrics: dict[str, Any] = {}
        category = contract.BLOCKCHAIN_CONTEXT
        if rich_evidence and isinstance(rich_evidence, dict):
            category = rich_evidence.get("aggregate", {}).get("category") or contract.BLOCKCHAIN_CONTEXT
            summary_metrics = {
                "transactions": rich_evidence.get("aggregate", {}).get("transactions"),
                "btc_sent_total": rich_evidence.get("aggregate", {}).get("btc_sent_total"),
                "btc_received_total": rich_evidence.get("aggregate", {}).get("btc_received_total"),
                "unique_counterparties": rich_evidence.get("aggregate", {}).get("unique_counterparties"),
                "peel_chain_members": rich_evidence.get("aggregate", {}).get("peel_chain_members"),
            }
            evidence_content_hash = hashlib.sha256(canonical_json_bytes(rich_evidence)).hexdigest()
        else:
            # Baseline evidence reference bound to the analytical run
            evidence_content_hash = hashlib.sha256(
                canonical_json_bytes({
                    "alert_id": alert_id,
                    "run_fingerprint": run_fp,
                    "cluster_id": cluster_id,
                })
            ).hexdigest()

        leaves.append(MerkleLeaf(
            kind="evidence",
            key=f"evidence:{alert_id}:{category}",
            data={
                "kind": "evidence",
                "evidence_id": f"evidence:{alert_id}:{category}",
                "alert_id": alert_id,
                "category": category,
                "run_fingerprint": run_fp,
                "canonical_content_hash": evidence_content_hash,
                "summary_metrics": summary_metrics,
            },
        ))

    # 2. Investigator Notes
    notes = casework.notes(conn, investigation_id)
    for n in notes:
        body_text = str(n.get("body") or "")
        leaves.append(MerkleLeaf(
            kind="note",
            key=f"note:{n['id']}",
            data={
                "kind": "note",
                "note_id": n["id"],
                "alert_id": n.get("alert_id"),
                "author_id": n.get("author_id"),
                "content_hash": hashlib.sha256(body_text.encode("utf-8")).hexdigest(),
                "created_at": n.get("created_at"),
            },
        ))

    # 3. Dispositions (active and historical rows)
    disp_rows = conn.execute(
        "SELECT id, alert_id, state, rationale, decided_by, decided_at, superseded_by "
        "FROM alert_dispositions WHERE investigation_id = ? ORDER BY id ASC",
        (investigation_id,),
    ).fetchall()
    for d in disp_rows:
        rationale_text = str(d["rationale"] or "")
        leaves.append(MerkleLeaf(
            kind="disposition",
            key=f"disposition:{d['id']}",
            data={
                "kind": "disposition",
                "disposition_id": d["id"],
                "alert_id": d["alert_id"],
                "state": d["state"],
                "rationale_hash": hashlib.sha256(rationale_text.encode("utf-8")).hexdigest(),
                "decided_by": d["decided_by"],
                "decided_at": d["decided_at"],
                "superseded_by": d["superseded_by"],
            },
        ))

    # 4. Report Versions
    report_versions = reports.versions(conn, investigation_id)
    for rv in report_versions:
        leaves.append(MerkleLeaf(
            kind="report_version",
            key=f"report_version:{rv['version']}",
            data={
                "kind": "report_version",
                "report_id": rv.get("id"),
                "version": rv["version"],
                "title": rv.get("title"),
                "status": rv.get("status"),
                "content_sha256": rv.get("content_sha256"),
                "generated_at": rv.get("generated_at"),
            },
        ))

    return leaves


def compute_case_merkle_tree(
    conn: sqlite3.Connection,
    investigation_id: str,
    *,
    current_run: str | None = None,
    alerts_data: list[dict] | None = None,
) -> MerkleTree:
    """Extract leaves and build the deterministic Merkle tree for a case."""
    leaves = extract_leaves(conn, investigation_id, current_run=current_run, alerts_data=alerts_data)
    return build_merkle_tree(leaves)


def record_integrity(
    conn: sqlite3.Connection,
    actor: User,
    investigation_id: str,
    *,
    tree: MerkleTree | None = None,
    current_run: str | None = None,
    alerts_data: list[dict] | None = None,
) -> dict[str, Any]:
    """Calculate and persist a historical Merkle integrity snapshot."""
    if tree is None:
        tree = compute_case_merkle_tree(
            conn, investigation_id, current_run=current_run, alerts_data=alerts_data
        )
    record_id = uuid.uuid4().hex
    manifest = {
        "leaf_count": tree.leaf_count,
        "leaf_hashes": tree.leaf_hashes,
        "leaves": [
            {"kind": l.kind, "key": l.key, "hash": h}
            for l, h in zip(tree.leaves, tree.leaf_hashes)
        ],
    }
    manifest_json = json.dumps(manifest, sort_keys=True)

    with db.transaction(conn):
        conn.execute(
            "INSERT INTO case_integrity "
            "(id, investigation_id, merkle_root, leaf_count, manifest_json, calculated_by, calculated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record_id,
                investigation_id,
                tree.root,
                tree.leaf_count,
                manifest_json,
                actor.id,
                tree.calculated_at,
            ),
        )

    audit.record(
        conn,
        actor_id=actor.id,
        action=audit.INTEGRITY_RECORDED,
        object_type="case_integrity",
        object_id=record_id,
        investigation_id=investigation_id,
        detail={"merkle_root": tree.root, "leaf_count": tree.leaf_count},
    )

    return {
        "id": record_id,
        "investigation_id": investigation_id,
        "merkle_root": tree.root,
        "leaf_count": tree.leaf_count,
        "calculated_at": tree.calculated_at,
        "calculated_by": actor.id,
    }


def get_latest_recorded_integrity(conn: sqlite3.Connection, investigation_id: str) -> dict[str, Any] | None:
    """Retrieve the most recently recorded historical Merkle integrity snapshot."""
    row = conn.execute(
        "SELECT id, investigation_id, merkle_root, leaf_count, manifest_json, calculated_by, calculated_at "
        "FROM case_integrity WHERE investigation_id = ? "
        "ORDER BY calculated_at DESC LIMIT 1",
        (investigation_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def verify_case_integrity(
    conn: sqlite3.Connection,
    investigation_id: str,
    *,
    actor: User | None = None,
    current_run: str | None = None,
) -> dict[str, Any]:
    """Verify live case integrity against the recorded historical Merkle root.
    
    Returns:
    - current_merkle_root
    - recorded_merkle_root
    - status: "VERIFIED" | "MISMATCH" | "NO_RECORDED_ROOT"
    - leaf_count
    - calculated_at
    - verification_meaning
    - disclaimer
    """
    current_tree = compute_case_merkle_tree(conn, investigation_id, current_run=current_run)
    latest_recorded = get_latest_recorded_integrity(conn, investigation_id)

    recorded_root: str | None = None
    recorded_at: str | None = None
    if latest_recorded is not None:
        recorded_root = latest_recorded["merkle_root"]
        recorded_at = latest_recorded["calculated_at"]

    if recorded_root is None:
        status = "NO_RECORDED_ROOT"
        meaning = "No prior historical Merkle root has been recorded for this case."
    elif current_tree.root.lower() == recorded_root.lower():
        status = "VERIFIED"
        meaning = "Live case state strictly matches the recorded historical Merkle root."
    else:
        status = "MISMATCH"
        meaning = (
            "Live case state differs from the recorded historical Merkle root. "
            "Records have been modified or added since last snapshot."
        )

    if actor is not None:
        audit.record(
            conn,
            actor_id=actor.id,
            action=audit.INTEGRITY_VERIFIED,
            object_type="case_integrity",
            object_id=investigation_id,
            investigation_id=investigation_id,
            detail={
                "status": status,
                "current_merkle_root": current_tree.root,
                "recorded_merkle_root": recorded_root,
            },
        )

    return {
        "current_merkle_root": current_tree.root,
        "recorded_merkle_root": recorded_root,
        "status": status,
        "leaf_count": current_tree.leaf_count,
        "calculated_at": current_tree.calculated_at,
        "recorded_at": recorded_at,
        "verification_meaning": meaning,
        "disclaimer": INTEGRITY_DISCLAIMER,
    }
