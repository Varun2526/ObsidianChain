"""/api/investigations/* - cases, datasets, analysis runs, history.

Every route here resolves the session first and the case second. There is no
path to case data that skips either. An investigation id in a URL grants
nothing: it names a row, and the row's owner decides whether the caller may
see it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Body, Depends, Query, Request, Response

from obsidianchain.console import (
    audit,
    casework,
    datasets as datasets_mod,
    db,
    deps,
    errors,
    integrity,
    investigations as inv,
    reports as reports_mod,
    runs as runs_mod,
    users as users_mod,
)
from obsidianchain.console.rbac import Capability, has
from obsidianchain.console.users import User

router = APIRouter(prefix="/api/investigations", tags=["investigations"])


def _owner_of(conn, investigation) -> object | None:
    return users_mod.get(conn, investigation.owner_id)


def _case_payload(conn, investigation, *, current_run) -> dict:
    """The case, its analytical binding, and its case-owned counts.

    Note what is NOT here: the number of alerts in the global artifact. That
    is a property of the pipeline's output, not of this investigation, and
    reporting it beside a case name is exactly how a baseline run came to
    look like the result of somebody's upload.
    """
    dataset_rows = datasets_mod.for_investigation(conn, investigation.id)
    with_runs = []
    for dataset in dataset_rows:
        run = runs_mod.latest_for_dataset(conn, dataset.id)
        with_runs.append({
            **dataset.as_dict(),
            "analysis_run": run.as_dict() if run else None,
        })

    return {
        **investigation.as_dict(owner=_owner_of(conn, investigation)),
        "analytical_run": {
            "bound_run_fingerprint": investigation.bound_run_fingerprint,
            "bound_run_at": investigation.bound_run_at,
            "current_artifact_run": current_run,
            "status": reports_mod._run_status(investigation, current_run),
            "meaning": (
                "The run this case's referenced alerts were taken from. It is "
                "set by an explicit action and is never re-pointed at "
                "whatever artifact happens to be on disk."
            ),
        },
        "datasets": with_runs,
        "summary": casework.summary(conn, investigation.id),
    }


@router.get("", summary="Investigations this user may see")
def list_investigations(
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    found = inv.listing(conn, actor)
    return {
        "investigations": [
            {
                **case.as_dict(owner=_owner_of(conn, case)),
                "summary": casework.summary(conn, case.id),
                "run_status": reports_mod._run_status(case, current_run),
            }
            for case in found
        ],
        "scope": (
            "all" if has(actor.role, Capability.VIEW_ALL_INVESTIGATIONS)
            else "owned"
        ),
        "current_artifact_run": current_run,
    }


@router.get("/activity/recent", summary="Recent casework activity")
def recent_activity(
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    """Recent audit events for cases this user may see."""
    if has(actor.role, Capability.VIEW_ALL_INVESTIGATIONS):
        events = audit.recent(conn, limit=limit)
    else:
        found = inv.listing(conn, actor)
        owned_ids = [c.id for c in found]
        if not owned_ids:
            return {"events": []}
        placeholders = ",".join("?" for _ in owned_ids)
        rows = conn.execute(
            f"SELECT e.*, u.username, u.display_name FROM audit_events e"
            f" LEFT JOIN users u ON u.id = e.actor_id"
            f" WHERE e.investigation_id IN ({placeholders})"
            f" ORDER BY e.id DESC LIMIT ?",
            (*owned_ids, limit),
        ).fetchall()
        events = [audit._shape(row) for row in rows]
    return {"events": events}


@router.post("", summary="Create an investigation", status_code=201)
def create_investigation(
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    case = inv.create(
        conn, actor,
        name=str(payload.get("name") or ""),
        description=str(payload.get("description") or ""),
    )
    return _case_payload(conn, case, current_run=current_run)


@router.get("/{investigation_id}", summary="One investigation")
def get_investigation(
    investigation_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    case = inv.require_readable(conn, actor, investigation_id)
    audit.record_standalone(
        conn, actor_id=actor.id, action=audit.INVESTIGATION_VIEWED,
        object_type="investigation", object_id=case.id,
        investigation_id=case.id, detail={},
    )
    return _case_payload(conn, case, current_run=current_run)


@router.patch("/{investigation_id}", summary="Rename or re-describe a case")
def patch_investigation(
    investigation_id: str,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    case = inv.update(
        conn, actor, investigation_id,
        name=payload.get("name"), description=payload.get("description"),
    )
    return _case_payload(conn, case, current_run=current_run)


@router.post("/{investigation_id}/status", summary="Move a case's status")
def set_status(
    investigation_id: str,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    """An explicit, validated, audited transition.

    Status never changes as a side effect of reading a page. The retired
    browser store flipped DRAFT to ACTIVE on render, which made the field
    describe traffic rather than the investigation.
    """
    case = inv.set_status(
        conn, actor, investigation_id, str(payload.get("status") or "")
    )
    return _case_payload(conn, case, current_run=current_run)


@router.post("/{investigation_id}/archive", summary="Archive an investigation (admin)")
def archive_investigation(
    investigation_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.require_capability(Capability.ARCHIVE_INVESTIGATION)),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    case = inv.archive(conn, actor, investigation_id)
    return _case_payload(conn, case, current_run=current_run)


@router.post("/{investigation_id}/restore", summary="Restore an archived investigation (admin)")
def restore_investigation(
    investigation_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.require_capability(Capability.ARCHIVE_INVESTIGATION)),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    case = inv.restore(conn, actor, investigation_id)
    return _case_payload(conn, case, current_run=current_run)


@router.delete("/{investigation_id}", summary="Delete an investigation (admin, exceptional)")
async def delete_investigation(
    investigation_id: str,
    request: Request,
    confirmation: str = Query(default=""),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.require_capability(Capability.DELETE_INVESTIGATION)),
) -> dict:
    confirm_text = confirmation
    if not confirm_text:
        try:
            body = await request.json()
            if isinstance(body, dict):
                confirm_text = str(body.get("confirmation") or body.get("confirm") or "")
        except Exception:
            pass
    inv.delete_case(conn, actor, investigation_id, confirmation=confirm_text)
    return {"ok": True, "deleted_id": investigation_id}


@router.post(
    "/{investigation_id}/analytical-run",
    summary="Bind this case to one analytical run",
)
def bind_run(
    investigation_id: str,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    """Explicit binding. Refuses to replace an existing, different binding.

    Referencing an alert binds the case automatically on first use, so this
    route exists for the case that wants to declare its run before it has
    referenced anything.
    """
    case = inv.require_writable(
        conn, actor, investigation_id, Capability.BIND_ANALYTICAL_RUN
    )
    fingerprint = str(payload.get("run_fingerprint") or "").strip()
    if not fingerprint:
        raise errors.ValidationFailed("run_fingerprint is required")
    inv.bind_run(conn, actor, case.id, fingerprint)
    return _case_payload(
        conn, inv.get(conn, case.id), current_run=current_run
    )


# ---- datasets -----------------------------------------------------------


@router.get("/{investigation_id}/datasets", summary="Datasets in this case")
def list_datasets(
    investigation_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    inv.require_readable(conn, actor, investigation_id)
    out = []
    for dataset in datasets_mod.for_investigation(conn, investigation_id):
        run = runs_mod.latest_for_dataset(conn, dataset.id)
        out.append({
            **dataset.as_dict(),
            "analysis_run": run.as_dict() if run else None,
        })
    return {"datasets": out}


@router.post(
    "/{investigation_id}/datasets",
    summary="Upload and persist a dataset for this case",
    status_code=201,
)
async def upload_dataset(
    investigation_id: str,
    request: Request,
    filename: str = Query(default="upload"),
    format: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    """Persist the bytes, validate them, and record an honest NOT_RUN.

    The body is the file's raw bytes rather than multipart, for the reason
    the existing ingest route already documents: ``python-multipart`` is not
    in the vendored wheel set and this image builds with no network.

    This validates and correlates. It does NOT score. The AnalysisRun created
    alongside carries status NOT_RUN and a NULL fingerprint, so the case
    cannot imply that alerts came from this upload.
    """
    inv.require_writable(
        conn, actor, investigation_id, Capability.UPLOAD_DATASET
    )
    payload = await request.body()
    dataset, run = datasets_mod.register(
        conn, actor,
        data_root=deps.data_root_of(request),
        investigation_id=investigation_id,
        payload=payload,
        filename=filename,
        declared_format=format,
    )
    if dataset.status == "VALIDATED":
        inv.advance_lifecycle(conn, actor, investigation_id, "VALIDATING",
                              reason=f"dataset {dataset.id} validated")
    return {"dataset": dataset.as_dict(), "analysis_run": run}


@router.get(
    "/{investigation_id}/datasets/{dataset_id}", summary="One dataset"
)
def get_dataset(
    investigation_id: str,
    dataset_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    inv.require_readable(conn, actor, investigation_id)
    dataset = datasets_mod.get(conn, dataset_id)
    # Checked against the case in the URL, not just fetched by id: a dataset
    # id from another case must not resolve through a case this caller can
    # see.
    if dataset is None or dataset.investigation_id != investigation_id:
        raise errors.NotFound(f"no dataset {dataset_id!r} in this investigation")
    runs = [r.as_dict() for r in runs_mod.for_dataset(conn, dataset_id)]
    return {"dataset": dataset.as_dict(), "analysis_runs": runs}


#: One worker: runs are CPU- and memory-heavy (a 40k-row capture peaks at
#: ~1.3 GB), so they are serialised rather than allowed to compete. Created
#: on first use so importing the router starts no thread.
_RUN_EXECUTOR = None


def _executor():
    global _RUN_EXECUTOR
    if _RUN_EXECUTOR is None:
        from concurrent.futures import ThreadPoolExecutor
        _RUN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis-run")
    return _RUN_EXECUTOR


def _execute_analysis(data_root, run_id: str, investigation_id: str, dataset_id: str,
                      storage_path: Path, declared_format, actor: User) -> dict:
    """Run the pipeline and record the outcome, on a connection of its own.

    Called on the worker (or inline with ?wait=true). SQLite connections are
    not shared across threads, so this opens and closes its own.
    """
    conn = db.connect(data_root)
    try:
        try:
            updated_run, outcome = runs_mod.execute_run(
                conn, run_id, storage_path,
                runs_dir=Path(data_root) / "runs", declared_format=declared_format,
            )
        except Exception as exc:  # recorded as FAILED by execute_run; audited here
            audit.record_standalone(
                conn, actor_id=actor.id, action=audit.DATASET_ANALYSIS_RUN,
                object_type="dataset", object_id=dataset_id, investigation_id=investigation_id,
                detail={"run_id": run_id, "status": "FAILED", "error": str(exc)[:500]},
            )
            raise
        audit.record_standalone(
            conn, actor_id=actor.id, action=audit.DATASET_ANALYSIS_RUN,
            object_type="dataset", object_id=dataset_id, investigation_id=investigation_id,
            detail={"run_id": run_id, "status": updated_run.status,
                    "run_fingerprint": updated_run.run_fingerprint,
                    "alerts_count": outcome.alert_result.total_alerts},
        )
        if updated_run.status == "COMPLETE":
            inv.advance_lifecycle(conn, actor, investigation_id, "ACTIVE",
                                  reason=f"analysis run {run_id} completed")
        return {"run_id": run_id, "analysis_run": updated_run.as_dict(),
                "alerts_count": outcome.alert_result.total_alerts, "manifest": outcome.manifest}
    finally:
        conn.close()


@router.post(
    "/{investigation_id}/datasets/{dataset_id}/run",
    summary="Start the 17-stage analytical pipeline on this dataset",
)
async def run_dataset_analysis(
    request: Request,
    investigation_id: str,
    dataset_id: str,
    wait: bool = Query(default=False, description="Run inline and return the finished run"),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    """Queue the run and return at once; poll /runs/{run_id}/progress.

    The pipeline used to execute inside this request: a 40k-row capture
    held the only server process for ~23 s, so every other user and the
    progress poll itself waited. It now runs on a background worker.
    ``?wait=true`` keeps the synchronous behaviour for scripts and tests.
    """
    inv.require_writable(
        conn, actor, investigation_id, Capability.UPLOAD_DATASET
    )
    dataset = datasets_mod.get(conn, dataset_id)
    if dataset is None or dataset.investigation_id != investigation_id:
        raise errors.NotFound(f"no dataset {dataset_id!r} in this investigation")

    latest = runs_mod.latest_for_dataset(conn, dataset_id)
    if latest is not None and latest.status in ("QUEUED", "RUNNING"):
        raise errors.Conflict(f"run {latest.id} for this dataset is already {latest.status}")
    if latest is None or latest.status in ("COMPLETE", "FAILED"):
        with db.transaction(conn):
            run_id = runs_mod.create_not_run(conn, dataset_id)["id"]
    else:
        run_id = latest.id

    data_root = deps.data_root_of(request)
    storage_path = Path(data_root) / dataset.storage_path
    if not storage_path.is_file():
        raise errors.NotFound(f"stored dataset file {storage_path} not found on disk")

    inv.advance_lifecycle(conn, actor, investigation_id, "ANALYZING",
                          reason=f"analysis run {run_id} started")
    job = (data_root, run_id, investigation_id, dataset_id, storage_path, dataset.format, actor)
    if wait:
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(_execute_analysis, *job)

    runs_mod.set_status(conn, run_id, "QUEUED")
    runs_mod.RUN_PROGRESS[run_id] = {
        "status": "QUEUED", "current_stage": 0, "total_stages": 17,
        "stage_name": "Queued", "stage_status": "PENDING", "progress_pct": 0,
    }
    future = _executor().submit(_execute_analysis, *job)
    future.add_done_callback(lambda f: f.exception())  # failures are recorded, never lost silently
    return {"run_id": run_id, "status": "QUEUED",
            "analysis_run": runs_mod.get(conn, run_id).as_dict(),
            "poll": f"/api/investigations/{investigation_id}/runs/{run_id}/progress"}


@router.get(
    "/{investigation_id}/runs/{run_id}/progress",
    summary="Live execution progress of a 17-stage analytical run",
)
def get_run_progress(
    investigation_id: str,
    run_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    inv.require_readable(conn, actor, investigation_id)
    return runs_mod.get_progress(conn, run_id)


def _scalar_summary(summary) -> dict:
    """A stage's recorded counts, without its embedded payloads (graphs, paths)."""
    if not isinstance(summary, dict):
        return {}
    out = {}
    for k, v in summary.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            if not (isinstance(v, str) and ("/" in v or len(v) > 80)):
                out[k] = v
        elif isinstance(v, list) and k in ("warnings", "errors"):
            out[k] = [str(x) for x in v[:5]]
    return out


def _complete_run_dir(request, conn, actor, investigation_id: str, run_id: str):
    """Resolve a run through the database (run -> dataset -> case), never the path alone."""
    inv.require_readable(conn, actor, investigation_id)
    run = runs_mod.get(conn, run_id)
    dataset = datasets_mod.get(conn, run.dataset_id) if run is not None else None
    if run is None or dataset is None or dataset.investigation_id != investigation_id:
        raise errors.NotFound(f"no run {run_id!r} in this investigation")
    if run.status != "COMPLETE":
        raise errors.Conflict(f"run {run_id} is {run.status}, results exist only for COMPLETE runs")
    return run, Path(deps.data_root_of(request)) / "runs" / run.id


@router.get(
    "/{investigation_id}/runs/{run_id}/results",
    summary="Ranked alerts, model provenance and model trust of a completed run",
)
def get_run_results(
    request: Request,
    investigation_id: str,
    run_id: str,
    limit: int = 50,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    """The result of one uploaded-dataset run, for the case it belongs to.

    The run id is resolved through the database (run -> dataset ->
    investigation), never used as a path fragment on its own, so a caller can
    neither read another case's run nor walk the filesystem. Every number
    carries the kind of result it is: the model's holdout result is a
    HOLDOUT result for the model; the run's own alerts have no measured
    precision until labels arrive.
    """
    run, run_dir = _complete_run_dir(request, conn, actor, investigation_id, run_id)
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        alerts = json.loads((run_dir / "alerts.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise errors.NotFound(f"run {run_id} artifacts are missing on disk") from exc
    limit = max(1, min(int(limit), 500))
    provenance = manifest.get("provenance", {})
    trust = provenance.get("model_trust", {})
    return {
        "run_id": run.id,
        "run_fingerprint": run.run_fingerprint,
        "created_at": manifest.get("created_at_utc"),
        "input_sha256": manifest.get("input_dataset", {}).get("sha256"),
        "ml_status": provenance.get("ml_status"),
        "model": {
            "version": trust.get("model_version"),
            "feature_schema_version": trust.get("feature_schema_version"),
            "holdout_result": trust.get("holdout_summary"),
            "holdout_result_type": "HOLDOUT (model-level, t42-49 of Elliptic++; not this run)",
        },
        "run_result_type": "PRODUCTION RUN - performance unverified until labels arrive",
        "monitoring_alerts": provenance.get("monitoring_alerts", []),
        "drift_relative_to_development": trust.get("drift_relative_to_development"),
        "total_alerts": alerts.get("total_alerts", 0),
        "alerts": alerts.get("alerts", [])[:limit],
        "stages": [{**{k: s.get(k) for k in ("stage_number", "stage_name", "status", "duration_seconds")},
                    "summary": _scalar_summary(s.get("summary"))}
                   for s in manifest.get("stages", [])],
    }


@router.get(
    "/{investigation_id}/runs/{run_id}/graph",
    summary="The investigation graph one uploaded-dataset run wrote",
)
def get_run_graph(
    request: Request,
    investigation_id: str,
    run_id: str,
    max_nodes: int = Query(default=1500, ge=10, le=5000),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    """``investigation_graph.json`` as the pipeline wrote it, case-scoped like results.

    Capped at ``max_nodes`` (in file order) and pruned to edges whose two
    ends are both present, so the client never receives a dangling edge.
    """
    run, run_dir = _complete_run_dir(request, conn, actor, investigation_id, run_id)
    try:
        graph = json.loads((run_dir / "investigation_graph.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise errors.NotFound(f"run {run_id} wrote no investigation graph") from exc
    nodes = graph.get("nodes", [])
    kept = nodes[:max_nodes]
    ids = {n["id"] for n in kept}
    edges = [e for e in graph.get("edges", []) if e.get("source") in ids and e.get("target") in ids]
    return {
        "run_id": run.id,
        "run_fingerprint": run.run_fingerprint,
        "graph": {"node_count": len(kept), "edge_count": len(edges), "nodes": kept, "edges": edges},
        "truncated": len(kept) < len(nodes),
        "nodes_total": len(nodes),
        "result_type": "PRODUCTION RUN - performance unverified until labels arrive",
        "meaning": ("Structure the pipeline projected from the uploaded dataset. Cluster membership is "
                    "an entity-resolution heuristic; an announcing peer is a relay, not a sender."),
    }


@router.get(
    "/{investigation_id}/runs/{run_id}/network",
    summary="Network propagation computed from the run's own observations",
)
def get_run_network(
    request: Request,
    investigation_id: str,
    run_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    """``network_propagation.json`` for one run, case-scoped like results.

    Runs made before propagation was computed have no such file; that is a
    404 with the reason, not an empty table that would read as "no network
    activity".
    """
    run, run_dir = _complete_run_dir(request, conn, actor, investigation_id, run_id)
    path = run_dir / "network_propagation.json"
    if not path.is_file():
        raise errors.NotFound(f"run {run_id} predates network propagation; re-run the analysis to compute it")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(payload.get("transactions", []),
                  key=lambda r: (r.get("first_seen_ms") is None, r.get("first_seen_ms") or 0))
    return {
        "run_id": run.id,
        "schema": payload.get("schema"),
        "meaning": payload.get("meaning"),
        "summary": payload.get("summary"),
        "transactions": rows[:limit],
        "transactions_total": len(rows),
    }


# ---- history ------------------------------------------------------------


@router.get("/{investigation_id}/history", summary="This case's audit trail")
def history(
    investigation_id: str,
    limit: int = Query(default=200, ge=1, le=2000),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    """Append-only. Readable by anyone who may read the case."""
    inv.require_readable(conn, actor, investigation_id)
    return {
        "events": audit.for_investigation(conn, investigation_id, limit=limit),
        "append_only": True,
    }


# ---- export (T2) --------------------------------------------------------


@router.get(
    "/{investigation_id}/export",
    summary="Export deterministic investigation bundle with Merkle integrity (T2)",
)
def export_investigation(
    investigation_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> Response:
    """Export an investigation bundle with single-pass deterministic serialization.

    Hash Definitions (Anti-Circular):
    --------------------------------
    A. merkle_root:
       Integrity root of the case records (alerts, evidence, notes, dispositions, report versions).
    B. bundle_sha256:
       SHA-256 of the canonical export payload BEFORE the 'bundle_sha256' field is inserted.
    C. export_sha256:
       SHA-256 of the FINAL JSON bytes actually delivered to the investigator.
    D. X-Bundle-SHA256:
       HTTP response header representing the exact bytes delivered by the HTTP response (== export_sha256).
    """
    import hashlib
    import json

    case = inv.require_readable(conn, actor, investigation_id)
    case_data = _case_payload(conn, case, current_run=current_run)
    alerts_data = casework.references(conn, case.id, current_run=current_run)
    notes_data = casework.notes(conn, case.id)
    report_data = reports_mod.latest(conn, case.id)
    versions_data = reports_mod.versions(conn, case.id)
    history_data = audit.for_investigation(conn, case.id, limit=1000)

    # 1. Compute Merkle tree and root over case records
    tree = integrity.compute_case_merkle_tree(conn, case.id, current_run=current_run)
    merkle_root = tree.root

    # Persist snapshot so future historical verifications can verify against it
    try:
        integrity.record_integrity(conn, actor, case.id, tree=tree, current_run=current_run)
    except Exception:
        pass

    # 2. Canonical export payload BEFORE bundle_sha256 is inserted
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

    # B. bundle_sha256: SHA-256 of the canonical export payload BEFORE bundle_sha256 is inserted
    bundle_preimage_bytes = integrity.canonical_json_bytes(bundle_preimage)
    bundle_sha256 = hashlib.sha256(bundle_preimage_bytes).hexdigest()

    # Insert bundle_sha256 and backward-compatible fields into final bundle
    bundle = dict(bundle_preimage)
    bundle["bundle_sha256"] = bundle_sha256
    bundle["merkle_root"] = merkle_root

    # Single deterministic serialization to raw bytes
    raw_bytes = json.dumps(bundle, sort_keys=True, indent=2).encode("utf-8")

    # C. export_sha256: SHA-256 of the FINAL JSON bytes actually delivered to the investigator
    export_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    audit.record_standalone(
        conn,
        actor_id=actor.id,
        action=audit.REPORT_EXPORTED,
        object_type="investigation_bundle",
        object_id=case.id,
        investigation_id=case.id,
        detail={
            "bundle_sha256": bundle_sha256,
            "export_sha256": export_sha256,
            "merkle_root": merkle_root,
            "leaf_count": tree.leaf_count,
        },
    )

    # D. X-Bundle-SHA256: Exact digest of delivered bytes
    filename = f"obsidianchain-case-{case.case_number or case.id}.json"
    headers = {
        "X-Bundle-SHA256": export_sha256,
        "X-Export-SHA256": export_sha256,
        "X-Content-SHA256": bundle_sha256,
        "X-Merkle-Root": merkle_root,
        "Content-Disposition": f'attachment; filename="{filename}"',
    }

    return Response(
        content=raw_bytes,
        media_type="application/json",
        headers=headers,
    )


# ---- integrity (T2 / Case-Integrity) ------------------------------------


@router.get(
    "/{investigation_id}/integrity",
    summary="Tamper-evident Merkle case-integrity verification",
)
def get_case_integrity(
    investigation_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    """Verify live case integrity against recorded historical Merkle root.

    Tamper-evident integrity verification; does not constitute legal
    admissibility or automated chain-of-custody warranty.
    """
    inv.require_readable(conn, actor, investigation_id)
    return integrity.verify_case_integrity(
        conn, investigation_id, actor=actor, current_run=current_run
    )


@router.post(
    "/{investigation_id}/integrity/verify",
    summary="Verify an individual Merkle leaf inclusion proof",
)
def verify_leaf_inclusion(
    investigation_id: str,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    """Verify an inclusion proof for an individual case record leaf against a root."""
    inv.require_readable(conn, actor, investigation_id)
    leaf_data = payload.get("leaf_data")
    proof = payload.get("proof")
    root = payload.get("root")

    if not isinstance(leaf_data, dict) or not isinstance(proof, list) or not isinstance(root, str):
        raise errors.ValidationFailed(
            "payload must contain 'leaf_data' (dict), 'proof' (list), and 'root' (str)"
        )

    verified = integrity.verify_leaf_proof(leaf_data, proof, root)
    return {
        "verified": verified,
        "root": root,
        "proof_steps": len(proof),
        "disclaimer": integrity.INTEGRITY_DISCLAIMER,
    }


@router.post(
    "/{investigation_id}/integrity/snapshot",
    summary="Record a historical Merkle integrity snapshot",
)
def create_integrity_snapshot(
    investigation_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    """Calculate and persist a historical Merkle root snapshot for this case."""
    inv.require_writable(conn, actor, investigation_id, Capability.EDIT_INVESTIGATION)
    return integrity.record_integrity(conn, actor, investigation_id, current_run=current_run)

