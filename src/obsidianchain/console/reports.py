"""Reports: persisted, versioned, run-bound, and honest about staleness.

What replaced what
------------------
The previous report was ``window.print()`` over live API reads. Its
executive summary was never saved at all, and an alert whose id had gone
stale was dropped from the table by a ``.catch(() => null)`` - so a report
that had listed six alerts would quietly list four, and say nothing.

Both failures had the same cause: nothing persisted. A report here is a row.
Saving creates a new VERSION rather than overwriting, so the document that
was reviewed is still the document that was reviewed. ``content_sha256``
covers the fields a reader would rely on, which makes tampering with a
stored report detectable.

Stale references are SHOWN, never dropped
-----------------------------------------
:func:`render` resolves the case's alert references against the artifact on
disk and marks each one current, stale or unverifiable. A stale reference
stays in the report carrying the run it was made against, because deleting
it would erase an investigator's decision and re-pointing it would attach
that decision to a different cluster.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass

from obsidianchain.console import audit, casework, db, errors, investigations
from obsidianchain.console.rbac import Capability, Role, has
from obsidianchain.console.users import User

STATUSES = ("DRAFT", "FINAL")

MAX_TITLE = 300
MAX_TEXT = 100_000

SEPARATION_NOTE = (
    "Analytical results in this report are read from immutable pipeline "
    "artifacts and were not produced by this console. Investigator "
    "conclusions - dispositions, rationales and notes - are recorded by "
    "named people and are separate claims of a different kind."
)


@dataclass(frozen=True)
class Report:
    id: str
    investigation_id: str
    version: int
    title: str
    executive_summary: str
    content: str
    run_fingerprint: str | None
    generated_by: str
    generated_at: str
    content_sha256: str
    status: str
    finalised_by: str | None
    finalised_at: str | None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "investigation_id": self.investigation_id,
            "version": self.version,
            "title": self.title,
            "executive_summary": self.executive_summary,
            "content": self.content,
            "run_fingerprint": self.run_fingerprint,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "content_sha256": self.content_sha256,
            "status": self.status,
            "finalised_by": self.finalised_by,
            "finalised_at": self.finalised_at,
        }


def _row(row: sqlite3.Row) -> Report:
    return Report(
        id=row["id"],
        investigation_id=row["investigation_id"],
        version=int(row["version"]),
        title=row["title"],
        executive_summary=row["executive_summary"],
        content=row["content"],
        run_fingerprint=row["run_fingerprint"],
        generated_by=row["generated_by"],
        generated_at=row["generated_at"],
        content_sha256=row["content_sha256"],
        status=row["status"],
        finalised_by=row["finalised_by"],
        finalised_at=row["finalised_at"],
    )


def content_hash(
    *, title: str, executive_summary: str, content: str,
    run_fingerprint, version: int,
) -> str:
    """A digest over exactly what a reader relies on.

    Canonical JSON with sorted keys so the same content always hashes the
    same way regardless of field order.
    """
    payload = json.dumps(
        {
            "title": title,
            "executive_summary": executive_summary,
            "content": content,
            "run_fingerprint": run_fingerprint,
            "version": int(version),
        },
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save(
    conn: sqlite3.Connection, actor: User, investigation_id: str,
    *, title: str = "", executive_summary: str = "", content: str = "",
) -> Report:
    """Write a new DRAFT version. Never overwrites an existing one."""
    found = investigations.require_writable(
        conn, actor, investigation_id, Capability.CREATE_REPORT
    )
    title = (title or f"{found.case_label} - {found.name}").strip()
    executive_summary = (executive_summary or "").strip()
    content = (content or "").strip()
    for name, value, limit in (
        ("title", title, MAX_TITLE),
        ("executive_summary", executive_summary, MAX_TEXT),
        ("content", content, MAX_TEXT),
    ):
        if len(value) > limit:
            raise errors.ValidationFailed(
                f"{name} is {len(value)} characters; the limit is {limit}"
            )

    now = db.utcnow()
    report_id = "rpt_" + secrets.token_hex(8)
    with db.transaction(conn):
        version = int(
            conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM reports"
                " WHERE investigation_id = ?",
                (investigation_id,),
            ).fetchone()[0]
        )
        digest = content_hash(
            title=title, executive_summary=executive_summary,
            content=content, run_fingerprint=found.bound_run_fingerprint,
            version=version,
        )
        conn.execute(
            "INSERT INTO reports (id, investigation_id, version, title,"
            " executive_summary, content, run_fingerprint, generated_by,"
            " generated_at, content_sha256, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'DRAFT')",
            (report_id, investigation_id, version, title, executive_summary,
             content, found.bound_run_fingerprint, actor.id, now, digest),
        )
        audit.record(
            conn, actor_id=actor.id,
            action=audit.REPORT_CREATED if version == 1
            else audit.REPORT_UPDATED,
            object_type="report", object_id=report_id,
            investigation_id=investigation_id,
            detail={"version": version, "content_sha256": digest},
        )
        investigations.touch(conn, investigation_id)
    return get(conn, report_id)


def get(conn: sqlite3.Connection, report_id: str) -> Report | None:
    row = conn.execute(
        "SELECT * FROM reports WHERE id = ?", (report_id,)
    ).fetchone()
    return _row(row) if row else None


def latest(conn: sqlite3.Connection, investigation_id: str) -> Report | None:
    row = conn.execute(
        "SELECT * FROM reports WHERE investigation_id = ?"
        " ORDER BY version DESC LIMIT 1",
        (investigation_id,),
    ).fetchone()
    return _row(row) if row else None


def version(
    conn: sqlite3.Connection, investigation_id: str, number: int
) -> Report | None:
    row = conn.execute(
        "SELECT * FROM reports WHERE investigation_id = ? AND version = ?",
        (investigation_id, int(number)),
    ).fetchone()
    return _row(row) if row else None


def versions(conn: sqlite3.Connection, investigation_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT r.id, r.version, r.title, r.status, r.generated_at,"
        " r.content_sha256, u.username AS generated_by_username"
        " FROM reports r LEFT JOIN users u ON u.id = r.generated_by"
        " WHERE r.investigation_id = ? ORDER BY r.version DESC",
        (investigation_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def finalise(
    conn: sqlite3.Connection, actor: User, investigation_id: str,
    report_version: int,
) -> Report:
    """Mark a version FINAL. Reviewer or admin only; once, never twice.

    Dual control: the person who writes a report is not the person who
    signs it off. FINALISE_REPORT is held by REVIEWER and ADMIN, not by
    INVESTIGATOR.
    """
    investigations.require_readable(conn, actor, investigation_id)
    if not has(actor.role, Capability.FINALISE_REPORT):
        raise errors.AccessDenied(
            f"role {actor.role.value} may not finalise a report; a report is "
            f"signed off by a reviewer, not by its author"
        )
    found = version(conn, investigation_id, report_version)
    if found is None:
        raise errors.NotFound(
            f"no report version {report_version} for this investigation"
        )
    if found.status == "FINAL":
        raise errors.Conflict(
            f"version {report_version} is already FINAL; save a new version "
            f"rather than re-finalising this one"
        )

    now = db.utcnow()
    with db.transaction(conn):
        conn.execute(
            "UPDATE reports SET status = 'FINAL', finalised_by = ?,"
            " finalised_at = ? WHERE id = ?",
            (actor.id, now, found.id),
        )
        audit.record(
            conn, actor_id=actor.id, action=audit.REPORT_FINALISED,
            object_type="report", object_id=found.id,
            investigation_id=investigation_id,
            detail={"version": report_version,
                    "content_sha256": found.content_sha256},
        )
        investigations.touch(conn, investigation_id)
    return version(conn, investigation_id, report_version)


def record_export(
    conn: sqlite3.Connection, actor: User, investigation_id: str,
    report_id: str,
) -> None:
    audit.record_standalone(
        conn, actor_id=actor.id, action=audit.REPORT_EXPORTED,
        object_type="report", object_id=report_id,
        investigation_id=investigation_id, detail={},
    )


def render(
    conn: sqlite3.Connection, investigation: investigations.Investigation,
    report: Report | None, *, current_run=None, owner=None,
) -> dict:
    """The full report payload: case, run, references, decisions, notes.

    Analytical results are NOT embedded. The referenced alert ids and the run
    fingerprint they were taken under are, so a reader resolves the numbers
    from the artifact that produced them - which is also what makes the stale
    check meaningful.
    """
    refs = casework.references(
        conn, investigation.id, current_run=current_run
    )
    stale = [r for r in refs if r["stale"] is True]
    unverifiable = [r for r in refs if r["stale"] is None]

    payload = {
        "case": investigation.as_dict(owner=owner),
        "analytical_run": {
            "bound_run_fingerprint": investigation.bound_run_fingerprint,
            "current_artifact_run": current_run,
            "status": _run_status(investigation, current_run),
        },
        "report": report.as_dict() if report else None,
        "versions": versions(conn, investigation.id),
        "alert_references": refs,
        "stale_references": [
            {
                **r,
                "warning": casework.STALE_REFERENCE_MEANING.format(
                    referenced=r["run_fingerprint"], current=current_run
                ),
            }
            for r in stale
        ],
        "unverifiable_references": unverifiable,
        "notes": casework.notes(conn, investigation.id),
        "summary": casework.summary(conn, investigation.id),
        "disposition_meaning": casework.DISPOSITION_MEANING,
        "separation_note": SEPARATION_NOTE,
    }
    return payload


def _run_status(investigation, current_run) -> str:
    if investigation.bound_run_fingerprint is None:
        return "UNBOUND"
    if current_run is None:
        return "UNVERIFIABLE"
    if investigation.bound_run_fingerprint != current_run:
        return "STALE"
    return "CURRENT"


def may_finalise(role: Role) -> bool:
    return has(role, Capability.FINALISE_REPORT)
