"""Uploaded datasets: persisted, hashed, and bound to a case.

The defect this fixes
---------------------
``POST /api/ingest`` writes the upload into a ``TemporaryDirectory`` and
deletes it when the response is sent. The case then stored a filename for a
file that no longer existed anywhere. An investigation whose evidence base
cannot be re-read is not an investigation.

Content-addressed storage
-------------------------
A file is stored at ``uploads/<sha256[:2]>/<sha256>``. The path is derived
from the content hash and NOTHING else - the user-supplied filename is
metadata, never a path component - so path traversal is impossible by
construction rather than by sanitisation. Two uploads of identical bytes
share one file and two dataset rows, which is correct: they are the same
evidence, registered twice.

The ingestion boundary is preserved
-----------------------------------
Persisting the file does not score it. Validation and correlation run
exactly as ``POST /api/ingest`` already ran them - the same
``io.ingest`` parser, the same report - and then stop. Producing risk means
building the as-of-t feature matrix over a whole dataset and fitting on the
temporal split, which is an offline pipeline run. The AnalysisRun created
alongside every dataset records that honestly as NOT_RUN with a NULL
fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from obsidianchain.console import audit, db, errors, runs
from obsidianchain.console.users import User

#: Same ceiling the existing ingest endpoint applies, for the same reason.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

STATUSES = ("RECEIVED", "VALIDATED", "REJECTED")


@dataclass(frozen=True)
class Dataset:
    id: str
    investigation_id: str
    filename: str
    sha256: str
    size_bytes: int
    format: str
    uploaded_by: str
    uploaded_at: str
    storage_path: str
    validation_report: dict
    status: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "investigation_id": self.investigation_id,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "format": self.format,
            "uploaded_by": self.uploaded_by,
            "uploaded_at": self.uploaded_at,
            "status": self.status,
            "validation": self.validation_report,
        }


def _row(row: sqlite3.Row) -> Dataset:
    try:
        report = json.loads(row["validation_report"])
    except ValueError:
        report = {}
    return Dataset(
        id=row["id"],
        investigation_id=row["investigation_id"],
        filename=row["filename"],
        sha256=row["sha256"],
        size_bytes=int(row["size_bytes"]),
        format=row["format"],
        uploaded_by=row["uploaded_by"],
        uploaded_at=row["uploaded_at"],
        storage_path=row["storage_path"],
        validation_report=report,
        status=row["status"],
    )


def upload_root(data_root) -> Path:
    return Path(data_root) / db.UPLOAD_DIRNAME


def storage_path_for(data_root, digest: str) -> Path:
    """Where bytes with this hash live. Derived from the hash alone.

    The digest is validated as 64 lowercase hex characters before it is used
    as a path component, so no caller-controlled string ever reaches the
    filesystem - not even one that has already been through a hash function.
    """
    if not _is_sha256(digest):
        raise errors.UploadRejected(
            f"{digest!r} is not a sha256 digest and cannot address storage"
        )
    return upload_root(data_root) / digest[:2] / digest


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def safe_display_name(filename) -> str:
    """The filename as METADATA. Never used to build a path.

    Reduced to its last component and stripped of control characters so it
    cannot smuggle a newline into an audit record or a report header.
    """
    raw = str(filename or "upload")
    base = raw.replace("\\", "/").split("/")[-1].strip() or "upload"
    cleaned = "".join(ch for ch in base if ch.isprintable())
    return (cleaned or "upload")[:255]


def _store_bytes(data_root, payload: bytes, digest: str) -> Path:
    """Write the bytes atomically. Idempotent for identical content."""
    destination = storage_path_for(data_root, digest)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".incoming-{secrets.token_hex(8)}"
    try:
        staging.write_bytes(payload)
        # Rename rather than write-in-place: a crash mid-write must not
        # leave a truncated file sitting at a path whose name claims to be
        # the hash of its contents.
        staging.replace(destination)
    finally:
        if staging.exists():
            staging.unlink()
    return destination


def register(
    conn: sqlite3.Connection,
    actor: User,
    *,
    data_root,
    investigation_id: str,
    payload: bytes,
    filename: str,
    declared_format: str | None = None,
) -> tuple[Dataset, dict]:
    """Persist, validate, and create the dataset's NOT_RUN analysis run.

    Returns ``(dataset, analysis_run)``. Authorisation is the caller's job -
    the route resolves the case through ``investigations.require_writable``
    with UPLOAD_DATASET before reaching here.
    """
    if not payload:
        raise errors.UploadRejected(
            "the request body was empty; send the file's bytes as the body"
        )
    if len(payload) > MAX_UPLOAD_BYTES:
        raise errors.UploadRejected(
            f"the upload is {len(payload):,} bytes; the limit is "
            f"{MAX_UPLOAD_BYTES:,}"
        )

    display = safe_display_name(filename)
    digest = hashlib.sha256(payload).hexdigest()

    # Validate with the SAME parser the existing /api/ingest route uses, so
    # the two endpoints cannot drift into disagreeing about what a valid
    # capture is.
    from obsidianchain.api import ingest as ingest_api
    from obsidianchain.io import ingest as ingest_io

    try:
        report = ingest_api.ingest_bytes(payload, display, declared_format)
        rejected = not report["validation"]["ok"]
    except ingest_io.IngestError as exc:
        report = {
            "filename": display,
            "bytes": len(payload),
            "parse_error": str(exc),
            "validation": {"ok": False, "errors": [str(exc)]},
        }
        rejected = True

    stored = _store_bytes(data_root, payload, digest)
    status = "REJECTED" if rejected else "VALIDATED"
    source_format = (
        report.get("validation", {}).get("source_format")
        or declared_format
        or Path(display).suffix.lstrip(".").lower()
        or "unknown"
    )

    dataset_id = "ds_" + secrets.token_hex(8)
    now = db.utcnow()
    # Stored relative to the data root so the database stays portable when
    # the deployment directory moves.
    relative = str(stored.relative_to(Path(data_root)))

    with db.transaction(conn):
        conn.execute(
            "INSERT INTO datasets (id, investigation_id, filename, sha256,"
            " size_bytes, format, uploaded_by, uploaded_at, storage_path,"
            " validation_report, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (dataset_id, investigation_id, display, digest, len(payload),
             source_format, actor.id, now, relative,
             json.dumps(report, default=str), status),
        )
        run = runs.create_not_run(conn, dataset_id)
        audit.record(
            conn, actor_id=actor.id, action=audit.DATASET_UPLOADED,
            object_type="dataset", object_id=dataset_id,
            investigation_id=investigation_id,
            detail={"filename": display, "sha256": digest,
                    "bytes": len(payload)},
        )
        audit.record(
            conn, actor_id=actor.id,
            action=audit.DATASET_REJECTED if rejected
            else audit.DATASET_VALIDATED,
            object_type="dataset", object_id=dataset_id,
            investigation_id=investigation_id,
            detail={"status": status},
        )
        conn.execute(
            "UPDATE investigations SET updated_at = ? WHERE id = ?",
            (now, investigation_id),
        )

    return get(conn, dataset_id), run


def get(conn: sqlite3.Connection, dataset_id: str) -> Dataset | None:
    row = conn.execute(
        "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
    ).fetchone()
    return _row(row) if row else None


def for_investigation(
    conn: sqlite3.Connection, investigation_id: str
) -> list[Dataset]:
    rows = conn.execute(
        "SELECT * FROM datasets WHERE investigation_id = ?"
        " ORDER BY uploaded_at DESC",
        (investigation_id,),
    ).fetchall()
    return [_row(row) for row in rows]


def open_stored(data_root, dataset: Dataset) -> Path:
    """Resolve a dataset's bytes on disk, refusing anything outside uploads.

    The stored path came from this module and should always be inside the
    upload root, so this check should never fire. It is here because a
    database is a mutable file: a row edited outside the application must not
    be able to point the reader at an arbitrary path.
    """
    root = upload_root(data_root).resolve()
    candidate = (Path(data_root) / dataset.storage_path).resolve()
    if not candidate.is_relative_to(root):
        raise errors.UploadRejected(
            f"dataset {dataset.id} points outside the upload root; refusing "
            f"to read it"
        )
    if not candidate.is_file():
        raise errors.NotFound(
            f"the stored bytes for dataset {dataset.id} are missing"
        )
    return candidate
