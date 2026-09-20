"""Detailed profiler for the investigation export endpoint.

Measures each individual phase across standard and large cases:
1. case/bundle data gathering
2. leaf extraction
3. canonical leaf serialization
4. leaf hashing
5. Merkle tree construction
6. historical snapshot/database transaction
7. final bundle serialization
8. final SHA-256
9. response construction
"""

import time
import tempfile
import pathlib
import json
import hashlib
from fastapi import Response

from obsidianchain.api.app import create_app
from obsidianchain.console import db, users, investigations as inv, casework, reports as reports_mod, audit, integrity
from obsidianchain.console.routes_investigations import _case_payload


def setup_case(root, n_alerts, n_notes):
    conn = db.connect(root)
    alice = users.create(conn, username="alice", password="password123", role="INVESTIGATOR", display_name="Alice")
    case = inv.create(conn, alice, name="Profiling Case", description="Detailed timing")
    
    run = "0123456789abcdef"
    for i in range(n_alerts):
        aid = f"{run}:{1000 + i}"
        casework.reference_alert(conn, alice, case.id, aid)
        if i % 2 == 0:
            casework.set_disposition(conn, alice, case.id, aid, "CONFIRMED", f"Rationale {i}")
        else:
            casework.set_disposition(conn, alice, case.id, aid, "DISMISSED", f"Rationale {i}")
            
    for j in range(n_notes):
        aid = f"{run}:{1000 + (j % n_alerts)}"
        casework.add_note(conn, alice, case.id, f"Note text for testing note #{j}", alert_id=aid)
        
    admin = users.create(conn, username="admin", password="adminpassword", role="ADMIN", display_name="Admin")
    rep = reports_mod.save(conn, alice, case.id, title="Profiling Report", executive_summary="Summary text", content="Body text")
    reports_mod.finalise(conn, admin, case.id, rep.version)
    
    conn.close()
    return case.id


def profile_case(label, n_alerts, n_notes, n_iterations=15):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir)
        case_id = setup_case(root, n_alerts, n_notes)
        conn = db.connect(root)
        actor = users.by_username(conn, "alice")
        current_run = None
        
        # Accumulators
        times = {
            "1_data_gathering": 0.0,
            "2_leaf_extraction": 0.0,
            "3_leaf_serialization": 0.0,
            "4_leaf_hashing": 0.0,
            "5_tree_construction": 0.0,
            "6_snapshot_db": 0.0,
            "7_bundle_serialization": 0.0,
            "8_final_sha256": 0.0,
            "9_response_construction": 0.0,
            "total_endpoint": 0.0,
        }
        
        payload_size = 0
        
        for _ in range(n_iterations):
            t_total_start = time.perf_counter()
            
            # 1. case/bundle data gathering
            t0 = time.perf_counter()
            case = inv.require_readable(conn, actor, case_id)
            case_data = _case_payload(conn, case, current_run=current_run)
            alerts_data = casework.references(conn, case.id, current_run=current_run)
            notes_data = casework.notes(conn, case.id)
            report_data = reports_mod.latest(conn, case.id)
            versions_data = reports_mod.versions(conn, case.id)
            history_data = audit.for_investigation(conn, case.id, limit=1000)
            t1 = time.perf_counter()
            times["1_data_gathering"] += (t1 - t0)
            
            # 2. leaf extraction
            t0 = time.perf_counter()
            leaves = integrity.extract_leaves(conn, case.id, current_run=current_run)
            t1 = time.perf_counter()
            times["2_leaf_extraction"] += (t1 - t0)
            
            # 3 & 4. canonical leaf serialization & leaf hashing
            t0 = time.perf_counter()
            serialized_leaves = []
            for l in leaves:
                serialized_leaves.append(integrity.canonical_json_bytes(l.data))
            t1 = time.perf_counter()
            times["3_leaf_serialization"] += (t1 - t0)
            
            t0 = time.perf_counter()
            leaf_hashes = []
            for raw_leaf in serialized_leaves:
                leaf_hashes.append(hashlib.sha256(integrity.LEAF_PREFIX + raw_leaf).hexdigest())
            t1 = time.perf_counter()
            times["4_leaf_hashing"] += (t1 - t0)
            
            # 5. Merkle tree construction
            t0 = time.perf_counter()
            tree = integrity.build_merkle_tree(leaves)
            merkle_root = tree.root
            t1 = time.perf_counter()
            times["5_tree_construction"] += (t1 - t0)
            
            # 6. historical snapshot/database transaction
            t0 = time.perf_counter()
            integrity.record_integrity(conn, actor, case.id, current_run=current_run)
            t1 = time.perf_counter()
            times["6_snapshot_db"] += (t1 - t0)
            
            # 7. final bundle serialization
            t0 = time.perf_counter()
            bundle_preimage = {
                "export_format": "obsidianchain.investigation.bundle/v1",
                "exported_at": db.utcnow(),
                "exported_by": {
                    "id": actor.id,
                    "username": actor.username,
                    "display_name": actor.display_name,
                    "role": actor.role,
                },
                "investigation": case_data,
                "alerts": alerts_data,
                "notes": notes_data,
                "report": report_data.as_dict() if report_data else None,
                "report_versions": versions_data,
                "history": history_data,
                "integrity": {
                    "system": "obsidianchain.integrity.merkle/v1",
                    "notice": integrity.INTEGRITY_DISCLAIMER,
                    "merkle_root": merkle_root,
                    "leaf_count": tree.leaf_count,
                    "calculated_at": tree.calculated_at,
                },
            }
            bundle_preimage_bytes = integrity.canonical_json_bytes(bundle_preimage)
            bundle_sha256 = hashlib.sha256(bundle_preimage_bytes).hexdigest()
            bundle = dict(bundle_preimage)
            bundle["bundle_sha256"] = bundle_sha256
            bundle["merkle_root"] = merkle_root
            raw_bytes = json.dumps(bundle, sort_keys=True, indent=2).encode("utf-8")
            t1 = time.perf_counter()
            times["7_bundle_serialization"] += (t1 - t0)
            payload_size = len(raw_bytes)
            
            # 8. final SHA-256
            t0 = time.perf_counter()
            export_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            t1 = time.perf_counter()
            times["8_final_sha256"] += (t1 - t0)
            
            # 9. response construction
            t0 = time.perf_counter()
            filename = f"obsidianchain-case-{case.case_number or case.id}.json"
            headers = {
                "X-Bundle-SHA256": export_sha256,
                "X-Export-SHA256": export_sha256,
                "X-Content-SHA256": bundle_sha256,
                "X-Merkle-Root": merkle_root,
                "Content-Disposition": f'attachment; filename="{filename}"',
            }
            resp = Response(content=raw_bytes, media_type="application/json", headers=headers)
            t1 = time.perf_counter()
            times["9_response_construction"] += (t1 - t0)
            
            t_total_end = time.perf_counter()
            times["total_endpoint"] += (t_total_end - t_total_start)
            
        print(f"\n=======================================================")
        print(f"PROFILING: {label} (Iterations: {n_iterations})")
        print(f"Payload size: {payload_size} bytes ({payload_size / 1024:.2f} KB)")
        print(f"Leaves in tree: {tree.leaf_count}")
        print(f"-------------------------------------------------------")
        for key in [
            "1_data_gathering",
            "2_leaf_extraction",
            "3_leaf_serialization",
            "4_leaf_hashing",
            "5_tree_construction",
            "6_snapshot_db",
            "7_bundle_serialization",
            "8_final_sha256",
            "9_response_construction",
            "total_endpoint",
        ]:
            avg_ms = (times[key] / n_iterations) * 1000
            pct = (avg_ms / ((times["total_endpoint"] / n_iterations) * 1000)) * 100 if key != "total_endpoint" else 100.0
            print(f"  {key:25s}: {avg_ms:8.3f} ms  ({pct:5.1f}%)")
        print(f"=======================================================")
        conn.close()


if __name__ == "__main__":
    profile_case("STANDARD CASE (~50 alerts, ~20 notes)", 50, 20, 20)
    profile_case("LARGE CASE (~500 alerts, ~100 notes)", 500, 100, 15)
