"""/api/investigations/* - cases, datasets, analysis runs, history.

Every route here resolves the session first and the case second. There is no
path to case data that skips either. An investigation id in a URL grants
nothing: it names a row, and the row's owner decides whether the caller may
see it.
"""

from __future__ import annotations

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


@router.post(
    "/{investigation_id}/datasets/{dataset_id}/run",
    summary="Execute the 17-stage analytical pipeline on this dataset",
)
async def run_dataset_analysis(
    request: Request,
    investigation_id: str,
    dataset_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    inv.require_writable(
        conn, actor, investigation_id, Capability.UPLOAD_DATASET
    )
    dataset = datasets_mod.get(conn, dataset_id)
    if dataset is None or dataset.investigation_id != investigation_id:
        raise errors.NotFound(f"no dataset {dataset_id!r} in this investigation")

    latest = runs_mod.latest_for_dataset(conn, dataset_id)
    if latest is None or latest.status in ("COMPLETE", "FAILED"):
        with db.transaction(conn):
            latest_dict = runs_mod.create_not_run(conn, dataset_id)
            run_id = latest_dict["id"]
    else:
        run_id = latest.id

    data_root = deps.data_root_of(request)
    storage_path = Path(data_root) / dataset.storage_path
    if not storage_path.is_file():
        raise errors.NotFound(f"stored dataset file {storage_path} not found on disk")

    runs_dir = Path(data_root) / "runs"
    updated_run, outcome = runs_mod.execute_run(
        conn,
        run_id,
        storage_path,
        runs_dir=runs_dir,
        declared_format=dataset.format,
    )

    audit.record_standalone(
        conn,
        actor_id=actor.id,
        action=audit.DATASET_ANALYSIS_RUN,
        object_type="dataset",
        object_id=dataset_id,
        investigation_id=investigation_id,
        detail={
            "run_id": run_id,
            "status": updated_run.status,
            "run_fingerprint": updated_run.run_fingerprint,
            "alerts_count": outcome.alert_result.total_alerts,
        },
    )
    return {
        "run_id": run_id,
        "analysis_run": updated_run.as_dict(),
        "alerts_count": outcome.alert_result.total_alerts,
        "manifest": outcome.manifest,
    }


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

