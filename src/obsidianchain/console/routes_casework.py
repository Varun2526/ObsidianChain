"""/api/investigations/{id}/alerts|notes|report - the casework surface.

The response shape is the point
------------------------------
Every case-scoped alert response has exactly two top-level blocks:

``analytical``
    Read from the immutable artifact. Risk, severity, SHAP, M0-M3 evidence,
    network context, provenance. Nothing in this block can be written.

``investigator``
    Read from SQLite. Disposition, its full history, assignment, notes. Every
    field attributable to a named person.

They are never merged, and neither one is ever presented as the other. A
model that scored a cluster at 0.94 and an investigator who marked it
DISMISSED are both saying something true, about different things.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Body, Depends, Query, Request

from obsidianchain.alerts import contract as alert_contract
from obsidianchain.console import (
    audit,
    casework,
    deps,
    errors,
    investigations as inv,
    reports as reports_mod,
    users as users_mod,
)
from obsidianchain.console.rbac import Capability, has
from obsidianchain.console.users import User

router = APIRouter(prefix="/api/investigations", tags=["casework"])


def _stale_block(reference, current_run) -> dict:
    return {
        "available": False,
        "reason": "STALE_REFERENCE",
        "referenced_run": reference.run_fingerprint,
        "current_artifact_run": current_run,
        "detail": casework.STALE_REFERENCE_MEANING.format(
            referenced=reference.run_fingerprint, current=current_run
        ),
    }


def _analytical(alert_id: str, reference, current_run, data_root) -> dict:
    """The immutable side, or an explicit statement of why it is unavailable.

    Never an empty object and never a silent omission. A reference whose run
    no longer matches the artifact returns a STALE_REFERENCE block carrying
    both fingerprints, so the investigator sees that the analytical half is
    missing and why - which is the failure the old report page hid behind a
    swallowed 409.
    """
    if current_run is None:
        return {
            "available": False,
            "reason": "ARTIFACT_UNAVAILABLE",
            "referenced_run": reference.run_fingerprint,
            "current_artifact_run": None,
            "detail": (
                "The analytical artifact could not be read, so this alert's "
                "risk, explanation and evidence are unavailable. The "
                "investigator record below is unaffected."
            ),
        }
    if reference.run_fingerprint != current_run:
        return _stale_block(reference, current_run)

    from obsidianchain.api import alerts as alerts_api

    try:
        return {"available": True, "alert": alerts_api.get_alert(alert_id, data_root)}
    except alert_contract.AlertIdStaleError:
        return _stale_block(reference, current_run)
    except alert_contract.AlertNotFoundError:
        return {
            "available": False,
            "reason": "ALERT_NOT_IN_RUN",
            "referenced_run": reference.run_fingerprint,
            "current_artifact_run": current_run,
            "detail": (
                f"Run {current_run} contains no alert for this cluster. The "
                f"reference is kept so the decision recorded against it is "
                f"not lost."
            ),
        }


def _investigator(conn, investigation_id: str, alert_id: str, reference) -> dict:
    assignee = (
        users_mod.get(conn, reference.assigned_to)
        if reference.assigned_to else None
    )
    return {
        "reference": {
            "alert_id": reference.alert_id,
            "run_fingerprint": reference.run_fingerprint,
            "added_by": reference.added_by,
            "added_at": reference.added_at,
        },
        "assignment": {
            "assigned_to": reference.assigned_to,
            "assigned_at": reference.assigned_at,
            "assignee": (
                {"id": assignee.id, "username": assignee.username,
                 "display_name": assignee.display_name}
                if assignee else None
            ),
        },
        "disposition": casework.current_disposition(
            conn, investigation_id, alert_id
        ),
        "disposition_history": casework.disposition_history(
            conn, investigation_id, alert_id
        ),
        "notes": casework.notes(conn, investigation_id, alert_id=alert_id),
        "states": list(casework.DISPOSITION_STATES),
        "meaning": casework.DISPOSITION_MEANING,
    }


# ---- alert references ---------------------------------------------------


@router.get("/{investigation_id}/alerts", summary="Alerts referenced by this case")
def list_case_alerts(
    investigation_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    """The case's OWN alerts. Never the global queue.

    A case with no referenced alerts returns an empty list, which is the
    honest answer. It does not fall back to the artifact's top-ranked alerts:
    those belong to the pipeline, not to this investigation.
    """
    inv.require_readable(conn, actor, investigation_id)
    refs = casework.references(conn, investigation_id, current_run=current_run)
    return {
        "alerts": refs,
        "current_artifact_run": current_run,
        "stale_count": sum(1 for r in refs if r["stale"] is True),
        "summary": casework.summary(conn, investigation_id),
        "meaning": casework.DISPOSITION_MEANING,
    }


@router.post(
    "/{investigation_id}/alerts", summary="Reference a global alert",
    status_code=201,
)
def reference_alert(
    investigation_id: str,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    """Attach an alert to the case by reference, never by copy.

    The alert must exist in the artifact currently on disk. Referencing one
    from a different run than the case is bound to is refused with 409
    rather than accepted alongside.
    """
    alert_id = str(payload.get("alert_id") or "").strip()
    run_fingerprint, _cluster = casework.parse_alert_id(alert_id)

    if current_run is not None and run_fingerprint != current_run:
        raise errors.RunMismatch(
            f"alert {alert_id} was minted against run {run_fingerprint}; the "
            f"artifact on disk is run {current_run}. Re-open the alert from "
            f"the current list before referencing it."
        )
    reference = casework.reference_alert(conn, actor, investigation_id, alert_id)
    return {
        "alert_id": reference.alert_id,
        "run_fingerprint": reference.run_fingerprint,
        "added_at": reference.added_at,
        "disposition": casework.current_disposition(
            conn, investigation_id, alert_id
        ),
    }


@router.get(
    "/{investigation_id}/alerts/{alert_id}",
    summary="One alert: analytical assessment and investigator assessment",
)
def get_case_alert(
    investigation_id: str,
    alert_id: str,
    request: Request,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    inv.require_readable(conn, actor, investigation_id)
    reference = casework.require_reference(conn, investigation_id, alert_id)

    audit.record_standalone(
        conn, actor_id=actor.id, action=audit.ALERT_VIEWED,
        object_type="alert", object_id=alert_id,
        investigation_id=investigation_id, detail={},
    )
    return {
        "alert_id": alert_id,
        "investigation_id": investigation_id,
        "analytical": _analytical(
            alert_id, reference, current_run, deps.data_root_of(request)
        ),
        "investigator": _investigator(
            conn, investigation_id, alert_id, reference
        ),
    }


@router.post(
    "/{investigation_id}/alerts/{alert_id}/disposition",
    summary="Record an investigator decision",
)
def set_disposition(
    investigation_id: str,
    alert_id: str,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    """Append a decision. The previous one is superseded, never erased."""
    current = casework.set_disposition(
        conn, actor, investigation_id, alert_id,
        str(payload.get("state") or ""), str(payload.get("rationale") or ""),
    )
    return {
        "disposition": current,
        "history": casework.disposition_history(
            conn, investigation_id, alert_id
        ),
    }


@router.post(
    "/{investigation_id}/alerts/{alert_id}/assignment",
    summary="Assign an alert to an investigator",
)
def assign_alert(
    investigation_id: str,
    alert_id: str,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    assignee = payload.get("assigned_to")
    reference = casework.assign(
        conn, actor, investigation_id, alert_id,
        str(assignee) if assignee else None,
    )
    return {
        "alert_id": reference.alert_id,
        "assigned_to": reference.assigned_to,
        "assigned_at": reference.assigned_at,
    }


# ---- notes --------------------------------------------------------------


@router.get("/{investigation_id}/notes", summary="Notes in this case")
def list_notes(
    investigation_id: str,
    alert_id: str | None = Query(default=None),
    case_level_only: bool = Query(default=False),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    inv.require_readable(conn, actor, investigation_id)
    return {
        "notes": casework.notes(
            conn, investigation_id, alert_id=alert_id,
            case_level_only=case_level_only,
        )
    }


@router.post("/{investigation_id}/notes", summary="Write a note", status_code=201)
def create_note(
    investigation_id: str,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    alert_id = payload.get("alert_id")
    note = casework.add_note(
        conn, actor, investigation_id, str(payload.get("body") or ""),
        alert_id=str(alert_id) if alert_id else None,
    )
    return note


# ---- reports ------------------------------------------------------------


@router.get("/{investigation_id}/report", summary="The case report")
def get_report(
    investigation_id: str,
    version: int | None = Query(default=None, ge=1),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    case = inv.require_readable(conn, actor, investigation_id)
    report = (
        reports_mod.version(conn, investigation_id, version)
        if version is not None
        else reports_mod.latest(conn, investigation_id)
    )
    if version is not None and report is None:
        raise errors.NotFound(f"no report version {version}")
    payload = reports_mod.render(
        conn, case, report, current_run=current_run,
        owner=users_mod.get(conn, case.owner_id),
    )
    payload["may_finalise"] = has(actor.role, Capability.FINALISE_REPORT)
    return payload


@router.post("/{investigation_id}/report", summary="Save a new report version")
def save_report(
    investigation_id: str,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
    current_run: str | None = Depends(deps.current_artifact_run),
) -> dict:
    """Every save is a new version. Nothing is overwritten, nothing is lost.

    The executive summary is persisted here - it was the field the previous
    report page never saved at all.
    """
    case = inv.require_readable(conn, actor, investigation_id)
    report = reports_mod.save(
        conn, actor, investigation_id,
        title=str(payload.get("title") or ""),
        executive_summary=str(payload.get("executive_summary") or ""),
        content=str(payload.get("content") or ""),
    )
    rendered = reports_mod.render(
        conn, inv.get(conn, case.id), report, current_run=current_run,
        owner=users_mod.get(conn, case.owner_id),
    )
    rendered["may_finalise"] = has(actor.role, Capability.FINALISE_REPORT)
    return rendered


@router.post(
    "/{investigation_id}/report/{version}/finalise",
    summary="Sign off a report version (reviewer or admin)",
)
def finalise_report(
    investigation_id: str,
    version: int,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    return reports_mod.finalise(
        conn, actor, investigation_id, version
    ).as_dict()


@router.post(
    "/{investigation_id}/report/{version}/export",
    summary="Record that a report version left the console",
)
def export_report(
    investigation_id: str,
    version: int,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    """Printing or exporting is itself an auditable act.

    The console cannot control what happens to a document after it leaves,
    which is exactly why the departure is recorded.
    """
    inv.require_readable(conn, actor, investigation_id)
    report = reports_mod.version(conn, investigation_id, version)
    if report is None:
        raise errors.NotFound(f"no report version {version}")
    reports_mod.record_export(conn, actor, investigation_id, report.id)
    return {"ok": True, "content_sha256": report.content_sha256}


# ---- saved filters (T2) -------------------------------------------------


@router.get(
    "/{investigation_id}/filters",
    summary="Saved filters for this user and investigation",
)
def list_filters(
    investigation_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    inv.require_readable(conn, actor, investigation_id)
    filters = casework.list_saved_filters(
        conn, user_id=actor.id, investigation_id=investigation_id
    )
    return {"filters": filters}


@router.post(
    "/{investigation_id}/filters",
    summary="Save an alert filter preset",
    status_code=201,
)
def create_filter(
    investigation_id: str,
    payload: dict = Body(...),
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    inv.require_readable(conn, actor, investigation_id)
    name = str(payload.get("name") or "")
    import json
    filter_json = json.dumps(payload.get("filter") or {})
    created = casework.create_saved_filter(
        conn,
        user_id=actor.id,
        name=name,
        filter_json=filter_json,
        investigation_id=investigation_id,
    )
    return {"filter": created}


@router.delete(
    "/{investigation_id}/filters/{filter_id}",
    summary="Delete a saved filter",
)
def delete_filter(
    investigation_id: str,
    filter_id: str,
    conn: sqlite3.Connection = Depends(deps.get_connection),
    actor: User = Depends(deps.current_user),
) -> dict:
    inv.require_readable(conn, actor, investigation_id)
    casework.delete_saved_filter(conn, user_id=actor.id, filter_id=filter_id)
    return {"ok": True}
