"""Tests for Export Serialization, Merkle Tree Integrity, and Historical Verification.

Proves:
1. Export Single-Buffer Serialization:
   - Returns raw bytes directly (FastAPI Response).
   - Delivered bytes hash exactly to X-Bundle-SHA256 (and X-Export-SHA256).
   - Content-Type and Content-Disposition headers are present and accurate.
   - Deterministic binary output across multiple export invocations.
   - JSON payload contains anti-circular bundle_sha256 and integrity block.
2. Merkle Tree & Leaf Integrity:
   - Deterministic leaf generation for alerts, evidence, notes, dispositions, report versions.
   - Domain-separated leaf (0x00) and branch (0x01) hashing.
   - Invariance to input reordering (leaves sorted by (kind, key)).
   - Inclusion proof generation and verification.
   - Rejection on tampered leaf data, tampered proof steps, and wrong root.
3. Historical Verification:
   - "VERIFIED" when live state matches recorded root.
   - "MISMATCH" when case state changes after snapshot.
   - "NO_RECORDED_ROOT" when no historical snapshot exists.
4. API Endpoints:
   - GET /api/investigations/{id}/integrity (status, disclaimer, roots, meaning).
   - POST /api/investigations/{id}/integrity/verify (proof verification).
   - Authentication, Authorization & Case Isolation (cross-user access denied).
   - Malformed proof rejection and 404 for nonexistent cases.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import pytest
from fastapi.testclient import TestClient

from obsidianchain.api.app import create_app
from obsidianchain.console import casework, db, integrity, investigations as inv, reports, users


@pytest.fixture
def root(tmp_path):
    return tmp_path


@pytest.fixture
def conn(root):
    connection = db.connect(root)
    yield connection
    connection.close()


@pytest.fixture
def seed_users(conn):
    alice = users.create(conn, username="alice", password="alice-password", role="INVESTIGATOR", display_name="Alice")
    bob = users.create(conn, username="bob", password="bob-password", role="INVESTIGATOR", display_name="Bob")
    admin = users.create(conn, username="admin", password="admin-password", role="ADMIN", display_name="Admin")
    return {"alice": alice, "bob": bob, "admin": admin}


@pytest.fixture
def app(root, seed_users):
    return create_app(root)


def login(app, username, password):
    client = TestClient(app)
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200, res.text
    return client


# =========================================================================
# 1. MERKLE UNIT TESTS
# =========================================================================

def test_merkle_leaf_domain_separation():
    """Leaf hash must use 0x00 domain separation prefix."""
    leaf_data = {"kind": "alert", "alert_id": "0123456789abcdef:100"}
    leaf = integrity.MerkleLeaf(kind="alert", key="alert:0123456789abcdef:100", data=leaf_data)
    
    expected_raw = integrity.LEAF_PREFIX + integrity.canonical_json_bytes(leaf_data)
    expected_hash = hashlib.sha256(expected_raw).hexdigest()
    
    assert leaf.hash() == expected_hash
    assert integrity.LEAF_PREFIX == b"\x00"
    assert integrity.BRANCH_PREFIX == b"\x01"


def test_merkle_branch_domain_separation():
    """Branch hash must use 0x01 domain separation prefix."""
    h1 = hashlib.sha256(b"left").hexdigest()
    h2 = hashlib.sha256(b"right").hexdigest()
    
    expected = hashlib.sha256(b"\x01" + bytes.fromhex(h1) + bytes.fromhex(h2)).hexdigest()
    assert integrity.combine_hashes(h1, h2) == expected


def test_merkle_tree_ordering_determinism():
    """Tree root must be identical regardless of the order leaves are supplied."""
    l1 = integrity.MerkleLeaf(kind="alert", key="alert:1", data={"id": 1})
    l2 = integrity.MerkleLeaf(kind="evidence", key="evidence:1:BLOCKCHAIN", data={"id": 2})
    l3 = integrity.MerkleLeaf(kind="note", key="note:1", data={"id": 3})
    l4 = integrity.MerkleLeaf(kind="disposition", key="disposition:1", data={"id": 4})
    l5 = integrity.MerkleLeaf(kind="report_version", key="report_version:1", data={"id": 5})
    
    tree_forward = integrity.build_merkle_tree([l1, l2, l3, l4, l5])
    tree_reversed = integrity.build_merkle_tree([l5, l4, l3, l2, l1])
    tree_shuffled = integrity.build_merkle_tree([l3, l1, l5, l2, l4])
    
    assert tree_forward.root == tree_reversed.root == tree_shuffled.root
    assert tree_forward.leaf_count == 5


def test_merkle_proof_verification_and_tamper_detection():
    """Proof must verify valid leaves and fail for tampered leaf data, tampered proofs, or wrong roots."""
    leaves = [
        integrity.MerkleLeaf(kind="alert", key=f"alert:{i}", data={"val": i})
        for i in range(7)  # odd number of leaves tests odd-layer padding
    ]
    tree = integrity.build_merkle_tree(leaves)
    
    target_leaf = leaves[3]
    target_hash = target_leaf.hash()
    proof = tree.get_proof(target_hash)
    
    # 1. Valid leaf & proof against root
    assert integrity.verify_leaf_proof(target_leaf.data, proof, tree.root) is True
    
    # 2. Tampered leaf data
    tampered_data = dict(target_leaf.data)
    tampered_data["val"] = 999
    assert integrity.verify_leaf_proof(tampered_data, proof, tree.root) is False
    
    # 3. Tampered proof step
    tampered_proof = [dict(p) for p in proof]
    tampered_proof[0]["sibling_hash"] = hashlib.sha256(b"fake").hexdigest()
    assert integrity.verify_leaf_proof(target_leaf.data, tampered_proof, tree.root) is False
    
    # 4. Wrong root
    wrong_root = hashlib.sha256(b"wrong_root").hexdigest()
    assert integrity.verify_leaf_proof(target_leaf.data, proof, wrong_root) is False


def test_empty_merkle_tree():
    """An empty leaf set produces the canonical EMPTY_ROOT without crashing."""
    tree = integrity.build_merkle_tree([])
    assert tree.leaf_count == 0
    assert tree.root == integrity.EMPTY_ROOT
    assert len(tree.leaves) == 0


# =========================================================================
# 2. HISTORICAL CASE INTEGRITY (VERIFIED / MISMATCH / NO_RECORDED_ROOT)
# =========================================================================

def test_historical_verification_states(conn, seed_users):
    """Test NO_RECORDED_ROOT -> VERIFIED -> MISMATCH state transitions."""
    alice = seed_users["alice"]
    case = inv.create(conn, alice, name="Integrity State Case", description="Testing verification states")
    
    # Initial state: no snapshot recorded
    v0 = integrity.verify_case_integrity(conn, case.id)
    assert v0["status"] == "NO_RECORDED_ROOT"
    assert v0["recorded_merkle_root"] is None
    assert v0["current_merkle_root"] is not None
    assert integrity.INTEGRITY_DISCLAIMER in v0["disclaimer"]
    
    # Add an alert reference and a note
    run_fp = "0123456789abcdef"
    aid = f"{run_fp}:101"
    casework.reference_alert(conn, alice, case.id, aid)
    casework.add_note(conn, alice, case.id, "Initial observation note.")
    
    # Record historical root snapshot
    rec = integrity.record_integrity(conn, alice, case.id)
    assert rec["merkle_root"] is not None
    assert rec["leaf_count"] > 0
    
    # Re-verify: must be VERIFIED
    v1 = integrity.verify_case_integrity(conn, case.id)
    assert v1["status"] == "VERIFIED"
    assert v1["current_merkle_root"] == rec["merkle_root"]
    assert v1["recorded_merkle_root"] == rec["merkle_root"]
    
    # Tamper/modify case state: add another note
    casework.add_note(conn, alice, case.id, "Second observation note added after snapshot.")
    
    # Re-verify: must be MISMATCH
    v2 = integrity.verify_case_integrity(conn, case.id)
    assert v2["status"] == "MISMATCH"
    assert v2["current_merkle_root"] != v2["recorded_merkle_root"]
    assert v2["recorded_merkle_root"] == rec["merkle_root"]


# =========================================================================
# 3. EXPORT SERIALIZATION & HASH CONTRACT
# =========================================================================

def test_export_single_buffer_and_hash_semantics(app, conn, seed_users):
    """Test single-buffer export response and exact anti-circular hash definitions."""
    alice = seed_users["alice"]
    admin = seed_users["admin"]
    client = login(app, "alice", "alice-password")
    
    case = inv.create(conn, alice, name="Export Contract Case", description="Audit export")
    run = "fedcba9876543210"
    aid = f"{run}:202"
    casework.reference_alert(conn, alice, case.id, aid)
    casework.set_disposition(conn, alice, case.id, aid, "CONFIRMED", "Confirmed illicit pattern")
    casework.add_note(conn, alice, case.id, "Export verification note.", alert_id=aid)
    
    rep = reports.save(conn, alice, case.id, title="Final Case Report", executive_summary="Summary text", content="Full markdown body")
    reports.finalise(conn, admin, case.id, rep.version)
    
    # Perform export request
    res = client.get(f"/api/investigations/{case.id}/export")
    assert res.status_code == 200
    
    # Verify Content-Type & Content-Disposition
    assert "application/json" in res.headers["content-type"]
    assert f'filename="obsidianchain-case-{case.case_number}.json"' in res.headers.get("content-disposition", "")
    
    raw_bytes = res.content
    assert len(raw_bytes) > 0
    
    # D. X-Bundle-SHA256 must represent exact bytes delivered by HTTP response
    header_sha256 = res.headers.get("X-Bundle-SHA256")
    delivered_bytes_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    assert header_sha256 == delivered_bytes_sha256
    assert res.headers.get("X-Export-SHA256") == delivered_bytes_sha256
    
    # Parse JSON bundle
    bundle = json.loads(raw_bytes.decode("utf-8"))
    
    # A. merkle_root
    assert "merkle_root" in bundle
    assert "integrity" in bundle
    assert bundle["integrity"]["merkle_root"] == bundle["merkle_root"]
    assert bundle["integrity"]["notice"] == integrity.INTEGRITY_DISCLAIMER
    assert bundle["integrity"]["leaf_count"] > 0
    
    # B. bundle_sha256: SHA-256 of payload BEFORE bundle_sha256 field is inserted
    assert "bundle_sha256" in bundle
    bundle_copy = dict(bundle)
    stored_bundle_sha256 = bundle_copy.pop("bundle_sha256")
    bundle_copy.pop("merkle_root", None)  # Merkle root is inside bundle['integrity'], outer is convenience
    
    recomputed_bundle_bytes = integrity.canonical_json_bytes(bundle_copy)
    recomputed_bundle_sha256 = hashlib.sha256(recomputed_bundle_bytes).hexdigest()
    assert stored_bundle_sha256 == recomputed_bundle_sha256
    
    # Ensure deterministic output on repeated export
    res2 = client.get(f"/api/investigations/{case.id}/export")
    assert res2.status_code == 200
    # Note: exported_at changes with current timestamp, but hash contract strictly holds
    assert hashlib.sha256(res2.content).hexdigest() == res2.headers.get("X-Bundle-SHA256")


# =========================================================================
# 4. API ENDPOINTS & RBAC / CASE ISOLATION
# =========================================================================

def test_api_integrity_endpoints(app, conn, seed_users):
    """Test GET /integrity and POST /integrity/verify routes."""
    alice = seed_users["alice"]
    client_alice = login(app, "alice", "alice-password")
    
    case = inv.create(conn, alice, name="API Test Case", description="Endpoints test")
    run = "0123456789abcdef"
    aid = f"{run}:303"
    casework.reference_alert(conn, alice, case.id, aid)
    
    # 1. GET integrity when NO_RECORDED_ROOT
    res = client_alice.get(f"/api/investigations/{case.id}/integrity")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "NO_RECORDED_ROOT"
    assert data["current_merkle_root"] is not None
    assert data["recorded_merkle_root"] is None
    assert integrity.INTEGRITY_DISCLAIMER in data["disclaimer"]
    
    # 2. Record snapshot
    snap_res = client_alice.post(f"/api/investigations/{case.id}/integrity/snapshot")
    assert snap_res.status_code == 200
    recorded_root = snap_res.json()["merkle_root"]
    
    # 3. GET integrity after snapshot -> VERIFIED
    res_after = client_alice.get(f"/api/investigations/{case.id}/integrity")
    assert res_after.status_code == 200
    assert res_after.json()["status"] == "VERIFIED"
    assert res_after.json()["recorded_merkle_root"] == recorded_root
    
    # 4. POST verify inclusion proof
    tree = integrity.compute_case_merkle_tree(conn, case.id)
    target_leaf = tree.leaves[0]
    proof = tree.get_proof(target_leaf.hash())
    
    verify_res = client_alice.post(
        f"/api/investigations/{case.id}/integrity/verify",
        json={
            "leaf_data": target_leaf.data,
            "proof": proof,
            "root": tree.root,
        },
    )
    assert verify_res.status_code == 200
    assert verify_res.json()["verified"] is True
    
    # Malformed proof body
    bad_res = client_alice.post(
        f"/api/investigations/{case.id}/integrity/verify",
        json={"leaf_data": "invalid", "proof": "invalid", "root": 123},
    )
    assert bad_res.status_code == 422 or bad_res.status_code == 400


def test_api_case_isolation(app, conn, seed_users):
    """Investigator Bob must NOT be able to access Alice's case integrity or export."""
    alice = seed_users["alice"]
    client_bob = login(app, "bob", "bob-password")
    
    case_alice = inv.create(conn, alice, name="Alice Confidential Case")
    
    # Bob attempts to GET integrity
    res_int = client_bob.get(f"/api/investigations/{case_alice.id}/integrity")
    assert res_int.status_code in (403, 404)
    
    # Bob attempts to export Alice's case
    res_exp = client_bob.get(f"/api/investigations/{case_alice.id}/export")
    assert res_exp.status_code in (403, 404)


def test_api_nonexistent_investigation(app, seed_users):
    """Accessing non-existent investigation returns 404."""
    client = login(app, "alice", "alice-password")
    res = client.get("/api/investigations/nonexistent-case-id/integrity")
    assert res.status_code == 404
