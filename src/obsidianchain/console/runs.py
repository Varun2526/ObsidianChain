"""AnalysisRun: the object that makes "not scored" a fact, not a footnote.

Why this exists before any job runner does
------------------------------------------
There is no queue, no worker and no automatic ``upload -> phase6 -> phase7``
path, deliberately. What was missing was not execution but HONESTY: the UI
implied that an upload had produced the 2,128 baseline alerts, because there
was no object capable of saying otherwise.

Every dataset gets a run row at upload time with ``status = NOT_RUN`` and
``run_fingerprint = NULL``. The schema's CHECK constraints make that pairing
structural - a fingerprint may exist only on a COMPLETE run - so "this
dataset has not been scored" is something the database enforces rather than
something a template remembers to say.

When offline execution is wired in later, it fills these same rows. Nothing
here has to change.
"""

from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass

from obsidianchain.console import db, errors

STATUSES = ("NOT_RUN", "QUEUED", "RUNNING", "COMPLETE", "FAILED")

#: The command that actually produces scored artifacts. Named in the API
#: response for the same reason the artifact loaders name their build
#: commands: an operator should never have to guess.
OFFLINE_COMMAND = (
    'make run ARGS="phase6-dataset" && make run ARGS="phase7-alerts"'
)

NOT_RUN_MEANING = (
    "This dataset was received, parsed and validated. It has NOT been "
    "scored. Producing risk requires building the as-of-t feature matrix "
    "over the whole dataset and fitting on the temporal split, which is an "
    "offline pipeline run, not a request. No alert shown anywhere in this "
    "console was generated from this dataset."
)


@dataclass(frozen=True)
class AnalysisRun:
    id: str
    dataset_id: str
    status: str
    run_fingerprint: str | None
    started_at: str | None
    completed_at: str | None
    error: str | None
    created_at: str

    @property
    def produced_alerts(self) -> bool:
        return self.status == "COMPLETE" and bool(self.run_fingerprint)

    def as_dict(self) -> dict:
        payload = {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "run_fingerprint": self.run_fingerprint,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "created_at": self.created_at,
            "produced_alerts": self.produced_alerts,
        }
        if self.status == "NOT_RUN":
            payload["meaning"] = NOT_RUN_MEANING
            payload["command"] = OFFLINE_COMMAND
        return payload


def _row(row: sqlite3.Row) -> AnalysisRun:
    return AnalysisRun(
        id=row["id"],
        dataset_id=row["dataset_id"],
        status=row["status"],
        run_fingerprint=row["run_fingerprint"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        error=row["error"],
        created_at=row["created_at"],
    )


def create_not_run(conn: sqlite3.Connection, dataset_id: str) -> dict:
    """The run every upload gets. Caller supplies the transaction."""
    run_id = "run_" + secrets.token_hex(8)
    conn.execute(
        "INSERT INTO analysis_runs (id, dataset_id, status, created_at)"
        " VALUES (?, ?, 'NOT_RUN', ?)",
        (run_id, dataset_id, db.utcnow()),
    )
    return _row(
        conn.execute(
            "SELECT * FROM analysis_runs WHERE id = ?", (run_id,)
        ).fetchone()
    ).as_dict()


def get(conn: sqlite3.Connection, run_id: str) -> AnalysisRun | None:
    row = conn.execute(
        "SELECT * FROM analysis_runs WHERE id = ?", (run_id,)
    ).fetchone()
    return _row(row) if row else None


def for_dataset(conn: sqlite3.Connection, dataset_id: str) -> list[AnalysisRun]:
    rows = conn.execute(
        "SELECT * FROM analysis_runs WHERE dataset_id = ?"
        " ORDER BY created_at DESC",
        (dataset_id,),
    ).fetchall()
    return [_row(row) for row in rows]


def latest_for_dataset(
    conn: sqlite3.Connection, dataset_id: str
) -> AnalysisRun | None:
    found = for_dataset(conn, dataset_id)
    return found[0] if found else None


def for_investigation(
    conn: sqlite3.Connection, investigation_id: str
) -> list[AnalysisRun]:
    rows = conn.execute(
        "SELECT r.* FROM analysis_runs r"
        " JOIN datasets d ON d.id = r.dataset_id"
        " WHERE d.investigation_id = ? ORDER BY r.created_at DESC",
        (investigation_id,),
    ).fetchall()
    return [_row(row) for row in rows]


def set_status(
    conn: sqlite3.Connection, run_id: str, status: str,
    *, run_fingerprint: str | None = None, error: str | None = None,
) -> AnalysisRun:
    """Advance a run. The only writer today is the future offline integration.

    Kept small and total: the CHECK constraints in the schema refuse a
    COMPLETE run without a fingerprint and a fingerprint on anything else, so
    this function cannot record a result it did not get.
    """
    target = (status or "").strip().upper()
    if target not in STATUSES:
        raise errors.ValidationFailed(
            f"{status!r} is not an analysis-run status; expected one of "
            f"{list(STATUSES)}"
        )
    now = db.utcnow()
    started = now if target == "RUNNING" else None
    completed = now if target in ("COMPLETE", "FAILED") else None
    try:
        with db.transaction(conn):
            conn.execute(
                "UPDATE analysis_runs SET status = ?, run_fingerprint = ?,"
                " started_at = COALESCE(?, started_at),"
                " completed_at = ?, error = ? WHERE id = ?",
                (target, run_fingerprint, started, completed, error, run_id),
            )
    except sqlite3.IntegrityError as exc:
        raise errors.ValidationFailed(
            f"refused: a run fingerprint may exist only on a COMPLETE run "
            f"({exc})"
        ) from exc
    found = get(conn, run_id)
    if found is None:
        raise errors.NotFound(f"no analysis run {run_id!r}")
    return found


RUN_PROGRESS: dict[str, dict] = {}


def get_progress(conn: sqlite3.Connection, run_id: str) -> dict:
    """Return real-time execution progress of a 17-stage analytical run."""
    if run_id in RUN_PROGRESS:
        return RUN_PROGRESS[run_id]
    run = get(conn, run_id)
    if run is None:
        raise errors.NotFound(f"no analysis run {run_id!r}")
    if run.status == "COMPLETE":
        return {
            "status": "COMPLETE",
            "current_stage": 17,
            "total_stages": 17,
            "stage_name": "Reporting & Integrity",
            "stage_status": "SUCCESS",
            "progress_pct": 100,
            "run_fingerprint": run.run_fingerprint,
        }
    if run.status == "FAILED":
        return {
            "status": "FAILED",
            "current_stage": 0,
            "total_stages": 17,
            "stage_name": "Execution Failed",
            "stage_status": "FAILED",
            "progress_pct": 0,
            "error": run.error,
        }
    return {
        "status": run.status,
        "current_stage": 0,
        "total_stages": 17,
        "stage_name": "Pending Execution",
        "stage_status": "NOT_RUN",
        "progress_pct": 0,
    }


def _update_progress(run_id: str, stage_num: int, stage_name: str, status: str, summary: dict) -> None:
    RUN_PROGRESS[run_id] = {
        "status": "RUNNING",
        "current_stage": stage_num,
        "total_stages": 17,
        "stage_name": stage_name,
        "stage_status": status,
        "progress_pct": int(round((stage_num / 17) * 100)),
        "summary": summary,
    }


def execute_run(
    conn: sqlite3.Connection,
    run_id: str,
    dataset_file_path: str | Path,
    runs_dir: str | Path | None = None,
    geoip_provider=None,
    declared_format: str | None = None,
):
    """Execute the full 17-stage analytical pipeline for an existing AnalysisRun.

    Transitions:
        NOT_RUN / QUEUED → RUNNING → COMPLETE (or FAILED)
    """
    from obsidianchain.pipeline.orchestrator import run_pipeline

    run = get(conn, run_id)
    if run is None:
        raise errors.NotFound(f"no analysis run {run_id!r}")

    set_status(conn, run_id, "RUNNING")
    _update_progress(run_id, 0, "Initializing pipeline", "PENDING", {})

    def on_stage(num: int, name: str, st: str, summ: dict) -> None:
        _update_progress(run_id, num, name, st, summ)

    try:
        outcome = run_pipeline(
            input_path=dataset_file_path,
            runs_dir=runs_dir,
            run_id=run_id,
            geoip_provider=geoip_provider,
            declared_format=declared_format,
            stage_callback=on_stage,
        )
        fingerprint = outcome.manifest["artifacts"]["manifest.json"][:16]
        updated_run = set_status(conn, run_id, "COMPLETE", run_fingerprint=fingerprint)
        RUN_PROGRESS[run_id] = {
            "status": "COMPLETE",
            "current_stage": 17,
            "total_stages": 17,
            "stage_name": "Reporting & Integrity",
            "stage_status": "SUCCESS",
            "progress_pct": 100,
            "run_fingerprint": fingerprint,
            "alerts_count": outcome.alert_result.total_alerts,
        }
        return updated_run, outcome
    except Exception as exc:
        RUN_PROGRESS[run_id] = {
            "status": "FAILED",
            "current_stage": RUN_PROGRESS.get(run_id, {}).get("current_stage", 0),
            "total_stages": 17,
            "stage_name": "Pipeline Failed",
            "stage_status": "FAILED",
            "progress_pct": 0,
            "error": str(exc),
        }
        set_status(conn, run_id, "FAILED", error=str(exc))
        raise

