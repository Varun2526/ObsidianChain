"""Casework: alert references, dispositions, assignment, notes.

The distinction this module exists to hold
------------------------------------------
An ALERT is an analytical prediction: a co-spend cluster the Phase 6 model
scored highly. It has a risk score, a severity band and a SHAP explanation,
all written by the pipeline and none of them changeable by a request.

A DISPOSITION is an investigator's decision about that alert within one
case. It has a state, a rationale, an author and a timestamp, and it is
attributable to a person.

They are different kinds of claim with different evidentiary weight, and
this module never lets one become the other. An alert is REFERENCED, never
copied; a disposition lives beside the reference, keyed by
``(investigation_id, alert_id)``, so two cases can reach different
conclusions about the same cluster without either one overwriting the other.

Append-only dispositions
------------------------
A changed decision inserts a new row and stamps ``superseded_by`` on the
previous one. It does not update the old row's state. A record that shows
only where an investigator ended up, and not that they changed their mind,
is not a forensic record. The database enforces this with triggers; this
module simply never asks for anything else.
"""

from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass

from obsidianchain.alerts import contract as alert_contract
from obsidianchain.console import audit, db, errors, investigations
from obsidianchain.console.rbac import Capability, Role, has, may_write_note
from obsidianchain.console.users import User

DISPOSITION_STATES = (
    "NEW", "TRIAGED", "IN_REVIEW", "CONFIRMED", "DISMISSED", "ESCALATED",
)

#: Carried on every case-scoped alert response. The frozen analytical
#: wording says what an alert is; this says what a disposition is, so the two
#: are never read as one assessment.
DISPOSITION_MEANING = (
    "A disposition is an INVESTIGATOR'S decision about this alert within "
    "this investigation. It is not a model output and it does not change "
    "the alert's risk score, severity or explanation, which are analytical "
    "artifacts and immutable. Model severity and investigator disposition "
    "are separate assessments and may legitimately disagree."
)

STALE_REFERENCE_MEANING = (
    "This alert was referenced against analytical run {referenced}, and the "
    "artifact currently on disk is run {current}. Alert ids do not survive a "
    "regeneration, because the cluster a number refers to can change. The "
    "reference is kept and marked STALE rather than removed or re-pointed: "
    "removing it would erase an investigator's decision, and re-pointing it "
    "would attach that decision to a cluster they never saw."
)

MAX_RATIONALE = 4000
MAX_NOTE = 20000


# ---- alert references ---------------------------------------------------


@dataclass(frozen=True)
class AlertReference:
    investigation_id: str
    alert_id: str
    run_fingerprint: str
    added_by: str
    added_at: str
    assigned_to: str | None
    assigned_at: str | None


def _reference(row: sqlite3.Row) -> AlertReference:
    return AlertReference(
        investigation_id=row["investigation_id"],
        alert_id=row["alert_id"],
        run_fingerprint=row["run_fingerprint"],
        added_by=row["added_by"],
        added_at=row["added_at"],
        assigned_to=row["assigned_to"],
        assigned_at=row["assigned_at"],
    )


def parse_alert_id(raw: str) -> tuple[str, int]:
    """Validate against the analytical layer's OWN contract.

    Imported rather than re-expressed: a second regex here could drift from
    the one the artifact writer uses, and then the console and the API would
    disagree about what addresses an alert.
    """
    try:
        return alert_contract.parse_alert_id(raw)
    except alert_contract.AlertIdInvalidError as exc:
        raise errors.ValidationFailed(str(exc)) from exc


def reference_alert(
    conn: sqlite3.Connection, actor: User, investigation_id: str, alert_id: str,
) -> AlertReference:
    """Attach a global alert to a case, recording the run it came from.

    The case is bound to that run on the first reference. A later reference
    from a different run raises :class:`errors.RunMismatch` rather than being
    accepted alongside, because a case whose alerts come from two runs cannot
    say which clusters its conclusions are about.
    """
    found = investigations.require_writable(
        conn, actor, investigation_id, Capability.REFERENCE_ALERT
    )
    run_fingerprint, _cluster = parse_alert_id(alert_id)

    if (
        found.bound_run_fingerprint is not None
        and found.bound_run_fingerprint != run_fingerprint
    ):
        raise errors.RunMismatch(
            f"alert {alert_id} belongs to analytical run {run_fingerprint}, "
            f"and this investigation is bound to run "
            f"{found.bound_run_fingerprint}. Referencing it would mix two "
            f"runs' clusters in one case."
        )

    existing = get_reference(conn, investigation_id, alert_id)
    if existing is not None:
        return existing

    now = db.utcnow()
    with db.transaction(conn):
        if found.bound_run_fingerprint is None:
            investigations.bind_run(
                conn, actor, investigation_id, run_fingerprint,
                inside_transaction=True,
            )
        conn.execute(
            "INSERT INTO alert_references (investigation_id, alert_id,"
            " run_fingerprint, added_by, added_at) VALUES (?, ?, ?, ?, ?)",
            (investigation_id, alert_id, run_fingerprint, actor.id, now),
        )
        # Every referenced alert starts at NEW. An alert nobody has triaged
        # and an alert someone triaged and left at NEW are the same state,
        # but the absence of a row would be indistinguishable from an alert
        # that was never referenced.
        _insert_disposition(
            conn, investigation_id, alert_id, "NEW",
            "Referenced into the investigation.", actor.id, now,
        )
        audit.record(
            conn, actor_id=actor.id, action=audit.ALERT_REFERENCED,
            object_type="alert", object_id=alert_id,
            investigation_id=investigation_id,
            detail={"run_fingerprint": run_fingerprint},
        )
        investigations.touch(conn, investigation_id)
    return get_reference(conn, investigation_id, alert_id)


def get_reference(
    conn: sqlite3.Connection, investigation_id: str, alert_id: str
) -> AlertReference | None:
    row = conn.execute(
        "SELECT * FROM alert_references WHERE investigation_id = ?"
        " AND alert_id = ?",
        (investigation_id, alert_id),
    ).fetchone()
    return _reference(row) if row else None


def require_reference(
    conn: sqlite3.Connection, investigation_id: str, alert_id: str
) -> AlertReference:
    found = get_reference(conn, investigation_id, alert_id)
    if found is None:
        raise errors.NotFound(
            f"alert {alert_id} is not referenced by this investigation"
        )
    return found


def references(
    conn: sqlite3.Connection, investigation_id: str, *, current_run=None
) -> list[dict]:
    """Every referenced alert with its current disposition and staleness.

    ``current_run`` is the fingerprint of the artifact on disk, or None when
    it could not be read. A reference is marked stale when the two differ;
    when the artifact is unavailable, staleness is reported as unknown rather
    than as False, because "we could not check" and "it is current" are
    different answers.
    """
    rows = conn.execute(
        "SELECT r.*, u.username AS added_by_username,"
        " a.username AS assigned_to_username,"
        " a.display_name AS assigned_to_display_name"
        " FROM alert_references r"
        " LEFT JOIN users u ON u.id = r.added_by"
        " LEFT JOIN users a ON a.id = r.assigned_to"
        " WHERE r.investigation_id = ? ORDER BY r.added_at ASC",
        (investigation_id,),
    ).fetchall()

    result = []
    for row in rows:
        current = current_disposition(
            conn, investigation_id, row["alert_id"]
        )
        result.append({
            "alert_id": row["alert_id"],
            "run_fingerprint": row["run_fingerprint"],
            "added_by": row["added_by"],
            "added_by_username": row["added_by_username"],
            "added_at": row["added_at"],
            "assigned_to": row["assigned_to"],
            "assigned_to_username": row["assigned_to_username"],
            "assigned_to_display_name": row["assigned_to_display_name"],
            "assigned_at": row["assigned_at"],
            "disposition": current,
            "stale": _staleness(row["run_fingerprint"], current_run),
        })
    return result


def _staleness(referenced: str, current_run) -> bool | None:
    if current_run is None:
        return None
    return referenced != current_run


def assign(
    conn: sqlite3.Connection, actor: User, investigation_id: str,
    alert_id: str, assignee_id: str | None,
) -> AlertReference:
    investigations.require_writable(
        conn, actor, investigation_id, Capability.ASSIGN_ALERT
    )
    require_reference(conn, investigation_id, alert_id)

    if assignee_id is not None:
        exists = conn.execute(
            "SELECT 1 FROM users WHERE id = ? AND active = 1", (assignee_id,)
        ).fetchone()
        if exists is None:
            raise errors.ValidationFailed(
                f"no active user {assignee_id!r} to assign this alert to"
            )

    now = db.utcnow()
    with db.transaction(conn):
        conn.execute(
            "UPDATE alert_references SET assigned_to = ?, assigned_at = ?"
            " WHERE investigation_id = ? AND alert_id = ?",
            (assignee_id, now if assignee_id else None,
             investigation_id, alert_id),
        )
        audit.record(
            conn, actor_id=actor.id, action=audit.ALERT_ASSIGNED,
            object_type="alert", object_id=alert_id,
            investigation_id=investigation_id,
            detail={"assigned_to": assignee_id},
        )
        investigations.touch(conn, investigation_id)
    return require_reference(conn, investigation_id, alert_id)


# ---- dispositions -------------------------------------------------------


def _insert_disposition(
    conn: sqlite3.Connection, investigation_id: str, alert_id: str,
    state: str, rationale: str, decided_by: str, at: str,
) -> str:
    """Append one disposition and supersede the previous active row."""
    new_id = "dsp_" + secrets.token_hex(8)
    conn.execute(
        "INSERT INTO alert_dispositions (id, investigation_id, alert_id,"
        " state, rationale, decided_by, decided_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (new_id, investigation_id, alert_id, state, rationale, decided_by, at),
    )
    conn.execute(
        "UPDATE alert_dispositions SET superseded_by = ?"
        " WHERE investigation_id = ? AND alert_id = ? AND id <> ?"
        " AND superseded_by IS NULL",
        (new_id, investigation_id, alert_id, new_id),
    )
    return new_id


def set_disposition(
    conn: sqlite3.Connection, actor: User, investigation_id: str,
    alert_id: str, state: str, rationale: str = "",
) -> dict:
    target = (state or "").strip().upper()
    if target not in DISPOSITION_STATES:
        raise errors.ValidationFailed(
            f"{state!r} is not a disposition; expected one of "
            f"{list(DISPOSITION_STATES)}"
        )
    rationale = (rationale or "").strip()
    if len(rationale) > MAX_RATIONALE:
        raise errors.ValidationFailed(
            f"rationale is {len(rationale)} characters; the limit is "
            f"{MAX_RATIONALE}"
        )

    investigations.require_writable(
        conn, actor, investigation_id, Capability.SET_DISPOSITION
    )
    require_reference(conn, investigation_id, alert_id)

    previous = current_disposition(conn, investigation_id, alert_id)
    now = db.utcnow()
    with db.transaction(conn):
        new_id = _insert_disposition(
            conn, investigation_id, alert_id, target, rationale, actor.id, now
        )
        audit.record(
            conn, actor_id=actor.id, action=audit.DISPOSITION_SET,
            object_type="alert", object_id=alert_id,
            investigation_id=investigation_id,
            detail={
                "from": (previous or {}).get("state"),
                "to": target,
                "disposition_id": new_id,
            },
        )
        investigations.touch(conn, investigation_id)
    return current_disposition(conn, investigation_id, alert_id)


def current_disposition(
    conn: sqlite3.Connection, investigation_id: str, alert_id: str
) -> dict | None:
    row = conn.execute(
        "SELECT d.*, u.username, u.display_name FROM alert_dispositions d"
        " LEFT JOIN users u ON u.id = d.decided_by"
        " WHERE d.investigation_id = ? AND d.alert_id = ?"
        " AND d.superseded_by IS NULL"
        " ORDER BY d.decided_at DESC, d.rowid DESC LIMIT 1",
        (investigation_id, alert_id),
    ).fetchone()
    return _disposition(row) if row else None


def disposition_history(
    conn: sqlite3.Connection, investigation_id: str, alert_id: str
) -> list[dict]:
    """Every decision ever recorded for this alert in this case, newest first."""
    rows = conn.execute(
        "SELECT d.*, u.username, u.display_name FROM alert_dispositions d"
        " LEFT JOIN users u ON u.id = d.decided_by"
        " WHERE d.investigation_id = ? AND d.alert_id = ?"
        " ORDER BY d.decided_at DESC, d.rowid DESC",
        (investigation_id, alert_id),
    ).fetchall()
    return [_disposition(row) for row in rows]


def _disposition(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "state": row["state"],
        "rationale": row["rationale"],
        "decided_by": row["decided_by"],
        "decided_by_username": row["username"],
        "decided_by_display_name": row["display_name"],
        "decided_at": row["decided_at"],
        "superseded_by": row["superseded_by"],
        "active": row["superseded_by"] is None,
    }


# ---- notes --------------------------------------------------------------


def add_note(
    conn: sqlite3.Connection, actor: User, investigation_id: str,
    body: str, *, alert_id: str | None = None,
) -> dict:
    """Attributed, timestamped, and attached to a case or to one alert."""
    body = (body or "").strip()
    if not body:
        raise errors.ValidationFailed("a note needs a body")
    if len(body) > MAX_NOTE:
        raise errors.ValidationFailed(
            f"note is {len(body)} characters; the limit is {MAX_NOTE}"
        )

    found = investigations.require_readable(conn, actor, investigation_id)
    if not may_write_note(actor.role, actor.id, found.owner_id):
        raise errors.AccessDenied(
            f"role {actor.role.value} may not write a note on this case"
        )
    if found.status == "CLOSED":
        raise errors.Conflict("this investigation is CLOSED")
    if alert_id is not None:
        parse_alert_id(alert_id)
        require_reference(conn, investigation_id, alert_id)

    note_id = "note_" + secrets.token_hex(8)
    now = db.utcnow()
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO investigator_notes (id, investigation_id, alert_id,"
            " body, author_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (note_id, investigation_id, alert_id, body, actor.id, now),
        )
        audit.record(
            conn, actor_id=actor.id, action=audit.NOTE_CREATED,
            object_type="note", object_id=note_id,
            investigation_id=investigation_id,
            detail={"alert_id": alert_id, "characters": len(body)},
        )
        investigations.touch(conn, investigation_id)
    return get_note(conn, note_id)


def get_note(conn: sqlite3.Connection, note_id: str) -> dict | None:
    row = conn.execute(
        "SELECT n.*, u.username, u.display_name FROM investigator_notes n"
        " LEFT JOIN users u ON u.id = n.author_id WHERE n.id = ?",
        (note_id,),
    ).fetchone()
    return _note(row) if row else None


def notes(
    conn: sqlite3.Connection, investigation_id: str,
    *, alert_id: str | None = None, case_level_only: bool = False,
) -> list[dict]:
    query = (
        "SELECT n.*, u.username, u.display_name FROM investigator_notes n"
        " LEFT JOIN users u ON u.id = n.author_id"
        " WHERE n.investigation_id = ?"
    )
    params: list = [investigation_id]
    if alert_id is not None:
        query += " AND n.alert_id = ?"
        params.append(alert_id)
    elif case_level_only:
        query += " AND n.alert_id IS NULL"
    query += " ORDER BY n.created_at DESC, n.rowid DESC"
    return [_note(row) for row in conn.execute(query, params).fetchall()]


def _note(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "investigation_id": row["investigation_id"],
        "alert_id": row["alert_id"],
        "body": row["body"],
        "author_id": row["author_id"],
        "author_username": row["username"],
        "author_display_name": row["display_name"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def summary(
    conn: sqlite3.Connection, investigation_id: str
) -> dict:
    """Counts the overview needs: how many referenced, reviewed, outstanding.

    Deliberately counts CASE-OWNED objects only. The number of alerts in the
    global artifact is not a property of this case and is never reported
    here.
    """
    referenced = int(conn.execute(
        "SELECT COUNT(*) FROM alert_references WHERE investigation_id = ?",
        (investigation_id,),
    ).fetchone()[0])
    by_state = {
        row["state"]: int(row["n"])
        for row in conn.execute(
            "SELECT state, COUNT(*) AS n FROM alert_dispositions"
            " WHERE investigation_id = ? AND superseded_by IS NULL"
            " GROUP BY state",
            (investigation_id,),
        ).fetchall()
    }
    note_count = int(conn.execute(
        "SELECT COUNT(*) FROM investigator_notes WHERE investigation_id = ?",
        (investigation_id,),
    ).fetchone()[0])
    outstanding = by_state.get("NEW", 0) + by_state.get("TRIAGED", 0)
    return {
        "alerts_referenced": referenced,
        "dispositions_by_state": {
            state: by_state.get(state, 0) for state in DISPOSITION_STATES
        },
        "outstanding": outstanding,
        "notes": note_count,
    }


def may_assign(role: Role) -> bool:
    return has(role, Capability.ASSIGN_ALERT)


# ---- saved filters (T2) -------------------------------------------------


def list_saved_filters(
    conn: sqlite3.Connection, user_id: str, investigation_id: str | None = None
) -> list[dict]:
    query = "SELECT * FROM saved_filters WHERE user_id = ?"
    params: list[str] = [user_id]
    if investigation_id:
        query += " AND (investigation_id IS NULL OR investigation_id = ?)"
        params.append(investigation_id)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    return [
        {
            "id": r["id"],
            "user_id": r["user_id"],
            "investigation_id": r["investigation_id"],
            "name": r["name"],
            "filter_json": r["filter_json"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def create_saved_filter(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    name: str,
    filter_json: str,
    investigation_id: str | None = None,
) -> dict:
    if not name.strip():
        raise errors.ValidationFailed("filter name cannot be empty")
    new_id = "flt_" + secrets.token_hex(8)
    now = db.utcnow()
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO saved_filters (id, user_id, investigation_id, name, filter_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (new_id, user_id, investigation_id, name.strip(), filter_json, now),
        )
    return {
        "id": new_id,
        "user_id": user_id,
        "investigation_id": investigation_id,
        "name": name.strip(),
        "filter_json": filter_json,
        "created_at": now,
    }


def delete_saved_filter(
    conn: sqlite3.Connection, *, user_id: str, filter_id: str
) -> bool:
    with db.transaction(conn):
        cur = conn.execute(
            "DELETE FROM saved_filters WHERE id = ? AND user_id = ?",
            (filter_id, user_id),
        )
        if cur.rowcount == 0:
            raise errors.NotFound(f"saved filter {filter_id!r} not found")
    return True
