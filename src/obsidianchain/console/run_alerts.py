"""Alerts from a case's own uploaded-dataset runs, as referenceable case alerts.

Until now only the reference alert artifact (the Elliptic++ Phase 7 run on
disk) could be referenced into a case, so an investigator who uploaded a
capture could look at its alerts but not decide on them. This module lets
the alerts a case's OWN completed run wrote be referenced exactly like
reference-run alerts: by id, never by copy, with the run recorded.

Identity
--------
An uploaded-run alert is addressed as ``<run fingerprint>:<rank>``, the
same ``<16 hex>:<integer>`` shape the alert contract validates. The integer
is the alert's rank within that run's ``alerts.json``. A run directory is
written once and never rewritten, so the rank names the same cluster for as
long as the run exists.

Which run is "current" for a case
---------------------------------
A case bound to one of its own completed runs is current against that run,
whatever reference artifact is on disk: the uploaded run is immutable and
belongs to the case. A case bound to the reference run (or unbound) keeps
the existing rule, current against the reference artifact on disk.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from obsidianchain.alerts import contract as alert_contract

SOURCE = "UPLOADED_RUN"

MEANING = (
    "An alert from this case's own uploaded-dataset run: a co-spend cluster "
    "the pipeline ranked for attention. The fused risk score combines the "
    "model score with rule evidence; network evidence is shown beside it and "
    "is not part of the score. It is a lead for review, not a finding."
)


def case_runs(conn: sqlite3.Connection, investigation_id: str) -> dict[str, str]:
    """``{run_fingerprint: run_id}`` for every COMPLETE run of this case's datasets."""
    rows = conn.execute(
        "SELECT r.id, r.run_fingerprint FROM analysis_runs r"
        " JOIN datasets d ON d.id = r.dataset_id"
        " WHERE d.investigation_id = ? AND r.status = 'COMPLETE'"
        " AND r.run_fingerprint IS NOT NULL",
        (investigation_id,),
    ).fetchall()
    return {row["run_fingerprint"][:16]: row["id"] for row in rows}


def belongs_to_another_case(conn: sqlite3.Connection, investigation_id: str,
                            run_fingerprint: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM analysis_runs r JOIN datasets d ON d.id = r.dataset_id"
        " WHERE substr(r.run_fingerprint, 1, 16) = ? AND d.investigation_id <> ?"
        " AND r.status = 'COMPLETE' LIMIT 1",
        (run_fingerprint, investigation_id),
    ).fetchone()
    return row is not None


def effective_run(conn: sqlite3.Connection, investigation_id: str | None,
                  reference_run: str | None) -> str | None:
    """The run this case's alerts are current against (see module docstring)."""
    if not investigation_id:
        return reference_run
    row = conn.execute(
        "SELECT bound_run_fingerprint FROM investigations WHERE id = ?",
        (investigation_id,),
    ).fetchone()
    bound = row["bound_run_fingerprint"] if row else None
    if bound and bound in case_runs(conn, investigation_id):
        return bound
    return reference_run


def alert_ref(run_fingerprint: str, rank: int) -> str:
    return alert_contract.make_alert_id(run_fingerprint, rank)


def _read(run_dir: Path, name: str) -> dict:
    path = run_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def find_alert(run_dir: Path, rank: int) -> dict | None:
    for alert in _read(run_dir, "alerts.json").get("alerts", []):
        if int(alert.get("rank", -1)) == int(rank):
            return alert
    return None


def _subgraph(graph: dict, cluster_node: str) -> dict:
    """The alert's neighbourhood: its members, their transactions, the
    counterpart addresses of those transactions, and the peers and ASNs that
    announced them. Everything else in the run graph is left out."""
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])
    members = {e["source"] for e in edges if e["kind"] == "MEMBER_OF" and e["target"] == cluster_node}
    txs = {e["target"] for e in edges if e["kind"] == "SPENDS" and e["source"] in members}
    txs |= {e["source"] for e in edges if e["kind"] == "RECEIVES" and e["target"] in members}
    keep = {cluster_node} | members | txs
    for e in edges:
        if e["kind"] in ("SPENDS", "RECEIVES") and (e["source"] in txs or e["target"] in txs):
            keep |= {e["source"], e["target"]}
        if e["kind"] == "ANNOUNCED_BY" and e["source"] in txs:
            keep.add(e["target"])
    for e in edges:
        if e["kind"] == "IN_ASN" and e["source"] in keep:
            keep.add(e["target"])
    kept_nodes = []
    for node_id in keep:
        node = nodes.get(node_id)
        if node is None:
            continue
        data = dict(node.get("data", {}))
        if node_id in members:
            data["member"] = True
        if node_id == cluster_node:
            data["seed"] = True
        kept_nodes.append({**node, "data": data})
    kept_edges = [e for e in edges if e["source"] in keep and e["target"] in keep]
    return {"nodes": kept_nodes, "edges": kept_edges,
            "members": sorted(m.split(":", 1)[1] for m in members),
            "transactions": sorted(t.split(":", 1)[1] for t in txs)}


def view(conn: sqlite3.Connection, data_root, investigation_id: str,
         alert_id: str) -> dict | None:
    """The analytical block for an uploaded-run alert, or None when the id
    does not belong to one of this case's runs."""
    fingerprint, rank = alert_contract.parse_alert_id(alert_id)
    run_id = case_runs(conn, investigation_id).get(fingerprint)
    if run_id is None:
        return None
    run_dir = Path(data_root) / "runs" / run_id
    alert = find_alert(run_dir, rank)
    if alert is None:
        return {
            "available": False, "source": SOURCE, "reason": "ALERT_NOT_IN_RUN",
            "referenced_run": fingerprint, "current_artifact_run": fingerprint,
            "detail": f"Run {run_id} has no alert of rank {rank}.",
        }
    # The cross-layer line, with every linked cluster resolved to its alert
    # in this run so the investigator can follow the lead.
    ranked = _read(run_dir, "alerts.json").get("alerts", [])
    alert_of = {a["cluster_id"]: a for a in ranked}
    line = next((e for e in alert.get("evidence", []) if e.get("signal_name") == "cross_layer_relay_coherence"), None)
    cross = None
    if line is not None:
        flows = []
        for flow in (line.get("details") or {}).get("linked_clusters", []):
            linked = []
            for cid in flow.get("linked_clusters", []):
                other = alert_of.get(cid)
                linked.append({"cluster_id": cid, "primary_address": other["primary_address"] if other else None,
                               "rank": other["rank"] if other else None,
                               "severity": other["severity"] if other else None,
                               "alert_ref": alert_ref(fingerprint, other["rank"]) if other else None})
            flows.append({**flow, "linked": sorted(linked, key=lambda x: (x["rank"] is None, x["rank"] or 0))})
        cross = {"status": line.get("status"), "score": line.get("score"), "explanation": line.get("explanation"),
                 "details": {k: v for k, v in (line.get("details") or {}).items() if k != "linked_clusters"},
                 "flows": flows}
    graph = _read(run_dir, "investigation_graph.json")
    sub = _subgraph(graph, f"cluster:{alert['cluster_id']}")
    propagation = _read(run_dir, "network_propagation.json")
    wanted = set(sub["transactions"])
    network_rows = [r for r in propagation.get("transactions", []) if r.get("txid") in wanted]
    manifest = _read(run_dir, "manifest.json")
    dataset = conn.execute(
        "SELECT d.filename, d.sha256 FROM analysis_runs r JOIN datasets d ON d.id = r.dataset_id"
        " WHERE r.id = ?", (run_id,),
    ).fetchone()
    trust = manifest.get("provenance", {}).get("model_trust", {})
    return {
        "available": True,
        "source": SOURCE,
        "meaning": MEANING,
        "run_alert": {
            "alert_ref": alert_id,
            "run_id": run_id,
            "run_fingerprint": fingerprint,
            "dataset": {"filename": dataset["filename"], "sha256": dataset["sha256"]} if dataset else None,
            "model_version": trust.get("model_version"),
            "alert": alert,
            "members": sub["members"],
            "transactions": sub["transactions"],
            "graph": {"nodes": sub["nodes"], "edges": sub["edges"]},
            "network": {
                "meaning": propagation.get("meaning"),
                "transactions": network_rows,
            },
            "cross_layer": cross,
        },
    }
