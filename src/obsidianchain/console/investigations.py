"""Investigations: the case object, server-side and owned.

Two identifiers, on purpose
---------------------------
``id`` is an opaque ``inv_<16 hex>`` used in URLs and the API. ``case_number``
is a sequential integer rendered as ``OC-0001`` for people. An investigator
needs a case number they can quote; a URL must not let anyone count how many
cases exist or probe for someone else's.

Status is never changed by rendering a page
-------------------------------------------
The retired browser store flipped a case from DRAFT to ACTIVE as a side
effect of viewing the overview. A case's status is a statement about the
investigation, so it changes only through :func:`set_status`, which is an
explicit authorised action that writes an audit event.

The run binding
---------------
``bound_run_fingerprint`` records which analytical run this case's referenced
results came from. It is set once, by an explicit action - binding it
directly, or referencing the first alert - and never by reading the current
artifact. That is the fix for the defect where opening a case silently
re-pointed it at whatever run happened to be on disk.
"""

from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass

from obsidianchain.console import audit, db, errors
from obsidianchain.console.rbac import Capability, Role, has, may_read_case
from obsidianchain.console.users import User

STATUSES = (
    "DRAFT", "VALIDATING", "ANALYZING", "ACTIVE",
    "SUBMITTED", "IN_REVIEW", "APPROVED", "RETURNED",
    "CLOSED", "ARCHIVED", "REVIEW",
)

#: Which status may follow which in the institutional casework lifecycle.
TRANSITIONS: dict[str, tuple[str, ...]] = {
    "DRAFT": ("VALIDATING", "CLOSED"),
    "VALIDATING": ("ANALYZING", "DRAFT", "CLOSED"),
    "ANALYZING": ("ACTIVE", "DRAFT"),
    "ACTIVE": ("SUBMITTED", "CLOSED"),
    "SUBMITTED": ("IN_REVIEW", "ACTIVE"),
    "IN_REVIEW": ("APPROVED", "RETURNED"),
    "REVIEW": ("APPROVED", "RETURNED", "ACTIVE"),  # backward compat alias
    "RETURNED": ("ACTIVE",),
    "APPROVED": ("CLOSED",),
    "CLOSED": ("ACTIVE", "ARCHIVED"),
    "ARCHIVED": ("CLOSED", "ACTIVE"),
}

MAX_NAME = 200
MAX_DESCRIPTION = 4000


@dataclass(frozen=True)
class Investigation:
    id: str
    case_number: int
    name: str
    description: str
    owner_id: str
    status: str
    bound_run_fingerprint: str | None
    bound_run_at: str | None
    bound_by: str | None
    created_at: str
    updated_at: str
    closed_at: str | None

    @property
    def case_label(self) -> str:
        return f"OC-{self.case_number:04d}"

    def as_dict(self, *, owner=None) -> dict:
        payload = {
            "id": self.id,
            "case_number": self.case_number,
            "case_label": self.case_label,
            "name": self.name,
            "description": self.description,
            "owner_id": self.owner_id,
            "status": self.status,
            "bound_run_fingerprint": self.bound_run_fingerprint,
            "bound_run_at": self.bound_run_at,
            "bound_by": self.bound_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "closed_at": self.closed_at,
        }
        if owner is not None:
            payload["owner"] = {
                "id": owner.id,
                "username": owner.username,
                "display_name": owner.display_name,
            }
        return payload


def _row(row: sqlite3.Row) -> Investigation:
    return Investigation(
        id=row["id"],
        case_number=int(row["case_number"]),
        name=row["name"],
        description=row["description"],
        owner_id=row["owner_id"],
        status=row["status"],
        bound_run_fingerprint=row["bound_run_fingerprint"],
        bound_run_at=row["bound_run_at"],
        bound_by=row["bound_by"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        closed_at=row["closed_at"],
    )


def new_id() -> str:
    return "inv_" + secrets.token_hex(8)


def create(
    conn: sqlite3.Connection, actor: User, *, name: str, description: str = ""
) -> Investigation:
    if not has(actor.role, Capability.CREATE_INVESTIGATION):
        raise errors.AccessDenied(
            f"role {actor.role.value} may not create an investigation"
        )
    name = (name or "").strip()
    if not name:
        raise errors.ValidationFailed("an investigation needs a name")
    if len(name) > MAX_NAME:
        raise errors.ValidationFailed(
            f"name is {len(name)} characters; the limit is {MAX_NAME}"
        )
    description = (description or "").strip()
    if len(description) > MAX_DESCRIPTION:
        raise errors.ValidationFailed(
            f"description is {len(description)} characters; the limit is "
            f"{MAX_DESCRIPTION}"
        )

    now = db.utcnow()
    identifier = new_id()
    with db.transaction(conn):
        # MAX+1 inside the write transaction. Safe because there is no
        # delete path, so the maximum never decreases and a number is never
        # reissued.
        case_number = int(
            conn.execute(
                "SELECT COALESCE(MAX(case_number), 0) + 1 FROM investigations"
            ).fetchone()[0]
        )
        conn.execute(
            "INSERT INTO investigations (id, case_number, name, description,"
            " owner_id, status, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, 'DRAFT', ?, ?)",
            (identifier, case_number, name, description, actor.id, now, now),
        )
        audit.record(
            conn, actor_id=actor.id, action=audit.INVESTIGATION_CREATED,
            object_type="investigation", object_id=identifier,
            investigation_id=identifier,
            detail={"name": name, "case_number": case_number},
        )
    return get(conn, identifier)


def get(conn: sqlite3.Connection, investigation_id: str) -> Investigation | None:
    row = conn.execute(
        "SELECT * FROM investigations WHERE id = ?", (investigation_id,)
    ).fetchone()
    return _row(row) if row else None


def require_readable(
    conn: sqlite3.Connection, actor: User, investigation_id: str
) -> Investigation:
    """Load a case the caller is entitled to see, or raise.

    404 for a case that does not exist, 403 for one that does and is not
    theirs. See :mod:`obsidianchain.console.errors` for why the two are kept
    distinct rather than both answering 404.
    """
    found = get(conn, investigation_id)
    if found is None:
        raise errors.InvestigationNotFound(
            f"no investigation {investigation_id!r}"
        )
    if not may_read_case(actor.role, actor.id, found.owner_id):
        raise errors.AccessDenied(
            "this investigation belongs to another investigator"
        )
    return found


def require_writable(
    conn: sqlite3.Connection, actor: User, investigation_id: str,
    capability: Capability,
) -> Investigation:
    """Readable, plus owned-or-admin, plus the specific capability."""
    found = require_readable(conn, actor, investigation_id)
    if not has(actor.role, capability):
        raise errors.AccessDenied(
            f"role {actor.role.value} may not {capability.value}"
        )
    if actor.role is not Role.ADMIN and actor.id != found.owner_id:
        raise errors.AccessDenied(
            "only the owning investigator may change this case"
        )
    if found.status == "CLOSED":
        raise errors.Conflict(
            "this investigation is CLOSED; reopen it before changing it"
        )
    return found


def listing(conn: sqlite3.Connection, actor: User) -> list[Investigation]:
    """Every case this user may see, most recently updated first."""
    if has(actor.role, Capability.VIEW_ALL_INVESTIGATIONS):
        rows = conn.execute(
            "SELECT * FROM investigations ORDER BY updated_at DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM investigations WHERE owner_id = ?"
            " ORDER BY updated_at DESC",
            (actor.id,),
        ).fetchall()
    return [_row(row) for row in rows]


def update(
    conn: sqlite3.Connection, actor: User, investigation_id: str,
    *, name=None, description=None,
) -> Investigation:
    found = require_writable(
        conn, actor, investigation_id, Capability.EDIT_INVESTIGATION
    )
    changes: dict[str, str] = {}
    if name is not None:
        cleaned = name.strip()
        if not cleaned:
            raise errors.ValidationFailed("an investigation needs a name")
        if len(cleaned) > MAX_NAME:
            raise errors.ValidationFailed(f"name exceeds {MAX_NAME} characters")
        changes["name"] = cleaned
    if description is not None:
        cleaned = description.strip()
        if len(cleaned) > MAX_DESCRIPTION:
            raise errors.ValidationFailed(
                f"description exceeds {MAX_DESCRIPTION} characters"
            )
        changes["description"] = cleaned
    if not changes:
        return found

    assignments = ", ".join(f"{key} = ?" for key in changes)
    with db.transaction(conn):
        conn.execute(
            f"UPDATE investigations SET {assignments}, updated_at = ?"
            " WHERE id = ?",
            (*changes.values(), db.utcnow(), investigation_id),
        )
        audit.record(
            conn, actor_id=actor.id, action=audit.INVESTIGATION_UPDATED,
            object_type="investigation", object_id=investigation_id,
            investigation_id=investigation_id,
            detail={"fields": sorted(changes)},
        )
    return get(conn, investigation_id)


def set_status(
    conn: sqlite3.Connection, actor: User, investigation_id: str, status: str
) -> Investigation:
    """Move a case through its lifecycle. Explicit, validated, audited."""
    target = (status or "").strip().upper()
    if target not in STATUSES:
        raise errors.ValidationFailed(
            f"{status!r} is not a status; expected one of {list(STATUSES)}"
        )

    found = require_readable(conn, actor, investigation_id)

    # Specific role constraints per target:
    if target == "ARCHIVED" or found.status == "ARCHIVED":
        if not has(actor.role, Capability.ARCHIVE_INVESTIGATION):
            raise errors.AccessDenied(
                "only administrators may archive or restore an investigation"
            )
    elif target in ("IN_REVIEW", "APPROVED", "RETURNED"):
        if not has(actor.role, Capability.REVIEW_INVESTIGATION):
            raise errors.AccessDenied(
                f"role {actor.role.value} may not review an investigation"
            )
    else:
        if not has(actor.role, Capability.CHANGE_INVESTIGATION_STATUS):
            raise errors.AccessDenied(
                f"role {actor.role.value} may not change investigation status"
            )
        if actor.role is not Role.ADMIN and actor.id != found.owner_id:
            raise errors.AccessDenied(
                "only the owning investigator may change this case's status"
            )

    if target == found.status:
        return found
    if target not in TRANSITIONS[found.status]:
        raise errors.Conflict(
            f"{found.status} -> {target} is not a permitted transition; "
            f"from {found.status} a case may move to "
            f"{list(TRANSITIONS[found.status]) or 'nothing'}"
        )

    now = db.utcnow()
    closed_at = now if target in ("CLOSED", "ARCHIVED") else None
    with db.transaction(conn):
        conn.execute(
            "UPDATE investigations SET status = ?, updated_at = ?,"
            " closed_at = ? WHERE id = ?",
            (target, now, closed_at, investigation_id),
        )
        audit.record(
            conn, actor_id=actor.id,
            action=audit.INVESTIGATION_STATUS_CHANGED,
            object_type="investigation", object_id=investigation_id,
            investigation_id=investigation_id,
            detail={"from": found.status, "to": target},
        )
    return get(conn, investigation_id)


#: The part of the lifecycle that follows system events rather than a
#: person's decision, in order.
PIPELINE_STAGES = ("DRAFT", "VALIDATING", "ANALYZING", "ACTIVE")


def advance_lifecycle(
    conn: sqlite3.Connection, actor: User, investigation_id: str, target: str, *, reason: str
) -> Investigation:
    """Move a case forward along DRAFT -> VALIDATING -> ANALYZING -> ACTIVE.

    Driven by real events (a validated upload, a completed analysis run),
    not by a button that only relabels the case. Only ever moves forward,
    only inside the pipeline stages, and a case that has already left them
    (submitted, in review, closed...) is not touched. Each step is audited
    with the event that caused it.
    """
    found = get(conn, investigation_id)
    if found is None or found.status not in PIPELINE_STAGES or target not in PIPELINE_STAGES:
        return found
    here, there = PIPELINE_STAGES.index(found.status), PIPELINE_STAGES.index(target)
    if there <= here:
        return found
    now = db.utcnow()
    with db.transaction(conn):
        conn.execute(
            "UPDATE investigations SET status = ?, updated_at = ? WHERE id = ?",
            (target, now, investigation_id),
        )
        audit.record(
            conn, actor_id=actor.id,
            action=audit.INVESTIGATION_STATUS_CHANGED,
            object_type="investigation", object_id=investigation_id,
            investigation_id=investigation_id,
            detail={"from": found.status, "to": target, "reason": reason, "automatic": True},
        )
    return get(conn, investigation_id)


def archive(
    conn: sqlite3.Connection, actor: User, investigation_id: str
) -> Investigation:
    """Archive an investigation. Standard administrative lifecycle action."""
    if not has(actor.role, Capability.ARCHIVE_INVESTIGATION):
        raise errors.AccessDenied(
            "only administrators may archive an investigation"
        )
    found = require_readable(conn, actor, investigation_id)
    if found.status == "ARCHIVED":
        return found
    if found.status not in ("CLOSED", "APPROVED"):
        raise errors.Conflict(
            f"investigation must be CLOSED before archiving; current status is {found.status}"
        )
    now = db.utcnow()
    with db.transaction(conn):
        conn.execute(
            "UPDATE investigations SET status = 'ARCHIVED', updated_at = ?,"
            " closed_at = COALESCE(closed_at, ?) WHERE id = ?",
            (now, now, investigation_id),
        )
        audit.record(
            conn, actor_id=actor.id, action=audit.INVESTIGATION_ARCHIVED,
            object_type="investigation", object_id=investigation_id,
            investigation_id=investigation_id,
            detail={"previous_status": found.status},
        )
    return get(conn, investigation_id)


def restore(
    conn: sqlite3.Connection, actor: User, investigation_id: str
) -> Investigation:
    """Restore an archived investigation back to CLOSED/ACTIVE."""
    if not has(actor.role, Capability.ARCHIVE_INVESTIGATION):
        raise errors.AccessDenied(
            "only administrators may restore an archived investigation"
        )
    found = require_readable(conn, actor, investigation_id)
    if found.status != "ARCHIVED":
        raise errors.Conflict(
            f"investigation is not ARCHIVED (status: {found.status})"
        )
    now = db.utcnow()
    with db.transaction(conn):
        conn.execute(
            "UPDATE investigations SET status = 'CLOSED', updated_at = ? WHERE id = ?",
            (now, investigation_id),
        )
        audit.record(
            conn, actor_id=actor.id, action=audit.INVESTIGATION_RESTORED,
            object_type="investigation", object_id=investigation_id,
            investigation_id=investigation_id,
            detail={"restored_status": "CLOSED"},
        )
    return get(conn, investigation_id)


def delete_case(
    conn: sqlite3.Connection, actor: User, investigation_id: str, confirmation: str = ""
) -> None:
    """Exceptional administrative maintenance action.
    Requires typed confirmation matching 'DELETE <case_label>'."""
    if not has(actor.role, Capability.DELETE_INVESTIGATION):
        raise errors.AccessDenied("only administrators may delete an investigation")
    found = require_readable(conn, actor, investigation_id)
    if found.status in ("ACTIVE", "SUBMITTED", "IN_REVIEW"):
        raise errors.Conflict(
            f"cannot delete an active or in-review investigation (status: {found.status}); "
            f"close or archive it first."
        )
    expected_confirm = f"DELETE {found.case_label}"
    if confirmation.strip() != expected_confirm:
        raise errors.ValidationFailed(
            f"typed confirmation mismatch: expected '{expected_confirm}', got '{confirmation}'"
        )

    # Check if case has recorded dispositions (which are schema-enforced append-only)
    disp_count = int(conn.execute(
        "SELECT COUNT(*) FROM alert_dispositions WHERE investigation_id = ?",
        (investigation_id,),
    ).fetchone()[0])
    if disp_count > 0:
        raise errors.Conflict(
            f"investigation has {disp_count} recorded disposition(s). Deleting it would "
            f"destroy immutable evidentiary history. Archive the investigation instead."
        )

    # Check if case has sealed cryptographic integrity records
    integrity_count = int(conn.execute(
        "SELECT COUNT(*) FROM case_integrity WHERE investigation_id = ?",
        (investigation_id,),
    ).fetchone()[0])
    if integrity_count > 0:
        raise errors.Conflict(
            f"investigation has {integrity_count} sealed integrity record(s). Deleting it would "
            f"destroy cryptographic provenance. Archive the investigation instead."
        )

    # Record audit event BEFORE deletion so that the deletion action itself is permanently logged
    audit.record_standalone(
        conn, actor_id=actor.id, action=audit.INVESTIGATION_DELETED,
        object_type="investigation", object_id=investigation_id,
        investigation_id=None,
        detail={
            "case_number": found.case_number,
            "case_label": found.case_label,
            "name": found.name,
        },
    )

    # Disable foreign keys temporarily for exceptional purge of this case
    # so append-only audit events retain their historical investigation_id
    # without foreign key violation on row deletion.
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        with db.transaction(conn):
            conn.execute("DELETE FROM saved_filters WHERE investigation_id = ?", (investigation_id,))
            conn.execute("DELETE FROM reports WHERE investigation_id = ?", (investigation_id,))
            conn.execute("DELETE FROM investigator_notes WHERE investigation_id = ?", (investigation_id,))
            conn.execute("DELETE FROM alert_references WHERE investigation_id = ?", (investigation_id,))
            dataset_ids = [r[0] for r in conn.execute(
                "SELECT id FROM datasets WHERE investigation_id = ?", (investigation_id,)
            ).fetchall()]
            for ds_id in dataset_ids:
                conn.execute("DELETE FROM analysis_runs WHERE dataset_id = ?", (ds_id,))
            conn.execute("DELETE FROM datasets WHERE investigation_id = ?", (investigation_id,))
            conn.execute("DELETE FROM investigations WHERE id = ?", (investigation_id,))
    finally:
        conn.execute("PRAGMA foreign_keys = ON")



def bind_run(
    conn: sqlite3.Connection, actor: User, investigation_id: str,
    run_fingerprint: str, *, inside_transaction: bool = False,
) -> None:
    """Bind a case to one analytical run. Once, explicitly, never silently.

    Re-binding to the SAME run is a no-op. Binding to a DIFFERENT one raises
    :class:`errors.RunMismatch` rather than overwriting: the previous
    binding is what every existing alert reference, disposition and report in
    this case was made against, and quietly replacing it would leave an
    investigator's decisions attached to clusters they never saw.
    """
    found = get(conn, investigation_id)
    if found is None:
        raise errors.InvestigationNotFound(f"no investigation {investigation_id!r}")
    if found.bound_run_fingerprint == run_fingerprint:
        return
    if found.bound_run_fingerprint is not None:
        raise errors.RunMismatch(
            f"this investigation is bound to analytical run "
            f"{found.bound_run_fingerprint}; the requested object belongs to "
            f"run {run_fingerprint}. A case's analytical run is not changed "
            f"silently - the decisions already recorded against it were made "
            f"on the first run's clusters."
        )

    def _write():
        conn.execute(
            "UPDATE investigations SET bound_run_fingerprint = ?,"
            " bound_run_at = ?, bound_by = ?, updated_at = ? WHERE id = ?",
            (run_fingerprint, db.utcnow(), actor.id, db.utcnow(),
             investigation_id),
        )
        audit.record(
            conn, actor_id=actor.id, action=audit.ANALYTICAL_RUN_BOUND,
            object_type="investigation", object_id=investigation_id,
            investigation_id=investigation_id,
            detail={"run_fingerprint": run_fingerprint},
        )

    if inside_transaction:
        _write()
    else:
        with db.transaction(conn):
            _write()


def touch(conn: sqlite3.Connection, investigation_id: str) -> None:
    """Bump updated_at. Caller supplies the transaction."""
    conn.execute(
        "UPDATE investigations SET updated_at = ? WHERE id = ?",
        (db.utcnow(), investigation_id),
    )
