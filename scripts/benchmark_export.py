"""Benchmark script to measure investigation export performance BEFORE and AFTER optimizations."""
import json
import time
import tracemalloc
import tempfile
import pathlib

from obsidianchain.api.app import create_app
from obsidianchain.console import db, users, investigations as inv, casework, reports
from fastapi.testclient import TestClient

def setup_benchmark_case(root, n_alerts=50, n_notes=20):
    conn = db.connect(root)
    alice = users.create(conn, username="alice", password="password123", role="INVESTIGATOR", display_name="Alice")
    case = inv.create(conn, alice, name="Operation DarkNet Benchmark", description="Benchmark Case")
    
    run = "0123456789abcdef"
    for i in range(n_alerts):
        aid = f"{run}:{1000 + i}"
        casework.reference_alert(conn, alice, case.id, aid)
        if i % 2 == 0:
            casework.set_disposition(conn, alice, case.id, aid, "ESCALATED", f"Escalation rationale {i}")
        else:
            casework.set_disposition(conn, alice, case.id, aid, "DISMISSED", f"Dismissed rationale {i}")
            
    for j in range(n_notes):
        aid = f"{run}:{1000 + (j % n_alerts)}"
        casework.add_note(conn, alice, case.id, f"Investigator observation note #{j} with some realistic analytical text and findings.", alert_id=aid)
        
    admin = users.create(conn, username="admin", password="adminpassword", role="ADMIN", display_name="Admin")
    rep = reports.save(conn, alice, case.id, title="Executive Investigation Summary", executive_summary="High-priority multi-cluster flow observed.", content="# Full Report Content\n\nDetailed markdown findings.")
    reports.finalise(conn, admin, case.id, rep.version)
    
    conn.close()
    return case.id

def run_suite():
    print("=== MEASURING CURRENT BASELINE (BEFORE) ===")
    for label, alerts, notes in [("Standard Case (50 alerts, 20 notes)", 50, 20), ("Large Case (500 alerts, 100 notes)", 500, 100)]:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            case_id = setup_benchmark_case(root, alerts, notes)
            app = create_app(root)
            
            client = TestClient(app)
            login_res = client.post("/api/auth/login", json={"username": "alice", "password": "password123"})
            assert login_res.status_code == 200
            
            # Warmup
            res = client.get(f"/api/investigations/{case_id}/export")
            assert res.status_code == 200
            payload_bytes = res.content
            payload_size = len(payload_bytes)
            
            tracemalloc.start()
            start_time = time.perf_counter()
            n_iter = 10 if alerts > 100 else 20
            for _ in range(n_iter):
                r = client.get(f"/api/investigations/{case_id}/export")
                assert r.status_code == 200
            total_time = time.perf_counter() - start_time
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            avg_endpoint_ms = (total_time / n_iter) * 1000
            
            data = res.json()
            ser_start = time.perf_counter()
            for _ in range(n_iter):
                _ = json.dumps(data, sort_keys=True, indent=2).encode("utf-8")
            ser_time = (time.perf_counter() - ser_start) / n_iter * 1000
            
            print(f"\n[{label}]")
            print(f"Payload size: {payload_size} bytes ({payload_size / 1024:.2f} KB)")
            print(f"Average total endpoint latency: {avg_endpoint_ms:.3f} ms")
            print(f"Average serialization time: {ser_time:.3f} ms")
            print(f"Peak memory: {peak_mem / 1024:.2f} KB")

if __name__ == "__main__":
    run_suite()
