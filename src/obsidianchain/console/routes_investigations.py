"""/api/investigations/* - cases, datasets, analysis runs, history.

Every route here resolves the session first and the case second. There is no
path to case data that skips either. An investigation id in a URL grants
nothing: it names a row, and the row's owner decides whether the caller may
see it.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Body, Depends, Query, Request, Response

from obsidianchain.console import (
    audit,
    casework,
    datasets as datasets_mod,
    deps,
    errors,
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
