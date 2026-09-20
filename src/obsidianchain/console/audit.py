"""The audit trail. Append-only, enforced in the schema.

Why this is not optional
------------------------
A forensic tool whose output may be tendered has to be able to answer who
looked at what, when, and what they concluded. Without that the analytical
rigour upstream - the fingerprints, the calibration, the refusal to serve a
torn artifact - defends a number whose handling nobody can account for.

Append-only is a storage property here, not a convention: ``audit_events``
carries BEFORE UPDATE and BEFORE DELETE triggers that ``RAISE(ABORT)``. A
handler that tried to amend history would fail at the database, whatever the
route layer believed it was allowed to do.

Recording failures
------------------
:func:`record` is called inside the same transaction as the action it
describes, so an action that rolls back takes its audit row with it. The one
exception is a failed login, which has no other write to join and is
recorded on its own.
"""

from __future__ import annotations

import json
import sqlite3

from obsidianchain.console import db

# ---- actions ------------------------------------------------------------
# String literals rather than an enum at the storage boundary: an audit row
# written under an action name this version does not know must still be
# readable by a later one, and an enum would refuse it on load.

LOGIN = "LOGIN"
LOGIN_FAILED = "LOGIN_FAILED"
LOGOUT = "LOGOUT"

INVESTIGATION_CREATED = "INVESTIGATION_CREATED"
INVESTIGATION_VIEWED = "INVESTIGATION_VIEWED"
INVESTIGATION_UPDATED = "INVESTIGATION_UPDATED"
INVESTIGATION_STATUS_CHANGED = "INVESTIGATION_STATUS_CHANGED"

DATASET_UPLOADED = "DATASET_UPLOADED"
DATASET_VALIDATED = "DATASET_VALIDATED"
DATASET_REJECTED = "DATASET_REJECTED"
DATASET_ANALYSIS_RUN = "DATASET_ANALYSIS_RUN"

ANALYTICAL_RUN_BOUND = "ANALYTICAL_RUN_BOUND"

ALERT_VIEWED = "ALERT_VIEWED"
ALERT_REFERENCED = "ALERT_REFERENCED"
ALERT_ASSIGNED = "ALERT_ASSIGNED"
DISPOSITION_SET = "DISPOSITION_SET"
NOTE_CREATED = "NOTE_CREATED"

REPORT_CREATED = "REPORT_CREATED"
REPORT_UPDATED = "REPORT_UPDATED"
REPORT_FINALISED = "REPORT_FINALISED"
REPORT_EXPORTED = "REPORT_EXPORTED"
INTEGRITY_RECORDED = "INTEGRITY_RECORDED"
INTEGRITY_VERIFIED = "INTEGRITY_VERIFIED"

USER_CREATED = "USER_CREATED"
USER_DEACTIVATED = "USER_DEACTIVATED"
USER_ACTIVATED = "USER_ACTIVATED"
USER_ROLE_CHANGED = "USER_ROLE_CHANGED"
USER_DELETED = "USER_DELETED"
PASSWORD_RESET = "PASSWORD_RESET"
PASSWORD_CHANGED = "PASSWORD_CHANGED"

INVESTIGATION_ARCHIVED = "INVESTIGATION_ARCHIVED"
INVESTIGATION_RESTORED = "INVESTIGATION_RESTORED"
INVESTIGATION_DELETED = "INVESTIGATION_DELETED"

#: Every action this version emits. Used by the tests to assert that the
#: audit surface covers the operations the console actually performs.
ACTIONS = (
    LOGIN, LOGIN_FAILED, LOGOUT,
    INVESTIGATION_CREATED, INVESTIGATION_VIEWED, INVESTIGATION_UPDATED,
    INVESTIGATION_STATUS_CHANGED, INVESTIGATION_ARCHIVED, INVESTIGATION_RESTORED,
    INVESTIGATION_DELETED,
    DATASET_UPLOADED, DATASET_VALIDATED, DATASET_REJECTED, DATASET_ANALYSIS_RUN,
    ANALYTICAL_RUN_BOUND,
    ALERT_VIEWED, ALERT_REFERENCED, ALERT_ASSIGNED,
    DISPOSITION_SET, NOTE_CREATED,
    REPORT_CREATED, REPORT_UPDATED, REPORT_FINALISED, REPORT_EXPORTED,
    INTEGRITY_RECORDED, INTEGRITY_VERIFIED,
    USER_CREATED, USER_DEACTIVATED, USER_ACTIVATED, USER_ROLE_CHANGED,
    USER_DELETED, PASSWORD_RESET, PASSWORD_CHANGED,
)


def record(
    conn: sqlite3.Connection,
    *,
    actor_id: str | None,
    action: str,
    object_type: str,
    object_id: str | None = None,
    investigation_id: str | None = None,
    detail: dict | None = None,
) -> int:
    """Append one event. Caller supplies the transaction.

    ``actor_id`` may be None for a failed login, where no identity was ever
    established. It is never None for anything else.
    """
    cursor = conn.execute(
        "INSERT INTO audit_events (actor_id, action, object_type, object_id,"
        " investigation_id, at, detail_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            actor_id, action, object_type, object_id, investigation_id,
            db.utcnow(), json.dumps(detail or {}, sort_keys=True,
                                    default=str),
        ),
    )
    return int(cursor.lastrowid)


def record_standalone(conn: sqlite3.Connection, **kwargs) -> int:
    """Append one event in its own transaction.

    For events with no accompanying write - a failed login, a read that is
    worth recording. Everything else should use :func:`record` inside the
    transaction of the change it describes, so the two cannot disagree.
    """
    with db.transaction(conn):
        return record(conn, **kwargs)


def for_investigation(
    conn: sqlite3.Connection, investigation_id: str, *, limit: int = 500
) -> list[dict]:
    """The case's history, newest first."""
    rows = conn.execute(
        "SELECT e.*, u.username, u.display_name FROM audit_events e"
        " LEFT JOIN users u ON u.id = e.actor_id"
        " WHERE e.investigation_id = ? ORDER BY e.id DESC LIMIT ?",
        (investigation_id, max(1, min(int(limit), 2000))),
    ).fetchall()
    return [_shape(row) for row in rows]


def recent(conn: sqlite3.Connection, *, limit: int = 200) -> list[dict]:
    """System-wide history. ADMIN and REVIEWER only; see rbac.VIEW_ALL_AUDIT."""
    rows = conn.execute(
        "SELECT e.*, u.username, u.display_name FROM audit_events e"
        " LEFT JOIN users u ON u.id = e.actor_id"
        " ORDER BY e.id DESC LIMIT ?",
        (max(1, min(int(limit), 2000)),),
    ).fetchall()
    return [_shape(row) for row in rows]


def _shape(row: sqlite3.Row) -> dict:
    try:
        detail = json.loads(row["detail_json"])
    except ValueError:
        # A row written by a future version, or a corrupt one. Surfaced as
        # unparsed text rather than dropped: an audit entry nobody can read
        # is still evidence that something happened.
        detail = {"unparsed": row["detail_json"]}
    return {
        "id": row["id"],
        "actor_id": row["actor_id"],
        "actor_username": row["username"],
        "actor_display_name": row["display_name"],
        "action": row["action"],
        "object_type": row["object_type"],
        "object_id": row["object_id"],
        "investigation_id": row["investigation_id"],
        "at": row["at"],
        "detail": detail,
    }
