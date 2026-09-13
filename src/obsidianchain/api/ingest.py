"""POST /api/ingest - validate an uploaded bulk metadata file.

What this endpoint does, and what it deliberately does not
----------------------------------------------------------
It parses CSV, JSON or XML, normalises it to the PS field set, validates it,
and reports what could be correlated. That is the PS's "ingest & parse a bulk
metadata dataset" requirement, answerable in a request.

It does NOT score the upload. Scoring means building the as-of-t feature
matrix over a whole dataset, fitting on the temporal split and calibrating -
minutes of work whose output every other artifact is fingerprinted against.
Pretending a request could do that would produce numbers no manifest
describes, which is the one thing the whole artifact discipline exists to
prevent. The response says so explicitly and names the offline command.

Why the body is raw rather than multipart
-----------------------------------------
``python-multipart`` is not in the vendored wheel set, and this image builds
with ``--network none``. Taking the file as the raw request body needs no new
dependency, works from a browser ``fetch`` with a File object, and keeps the
offline guarantee intact.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

#: Refused above this. Large enough for a realistic bulk capture, small
#: enough that a stray request cannot exhaust memory.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


class UploadTooLargeError(ValueError):
    """The body exceeded MAX_UPLOAD_BYTES."""


class UploadEmptyError(ValueError):
    """No bytes were sent."""


def ingest_bytes(payload: bytes, filename: str, declared_format=None) -> dict:
    """Validate one uploaded file and describe what it contains."""
    from obsidianchain.io import ingest as ingest_io

    if not payload:
        raise UploadEmptyError(
            "the request body was empty; send the file's bytes as the body"
        )
    if len(payload) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError(
            f"the upload is {len(payload):,} bytes; the limit is "
            f"{MAX_UPLOAD_BYTES:,}"
        )

    suffix = Path(filename or "upload").suffix or ""
    with tempfile.TemporaryDirectory(prefix="oc-ingest-") as staging:
        staged = Path(staging) / f"upload{suffix}"
        staged.write_bytes(payload)
        frame, report = ingest_io.ingest(staged, declared_format)
        summary = ingest_io.correlation_summary(frame)
        preview = frame.head(10).to_dict("records") if len(frame) else []

    return {
        "filename": filename,
        "bytes": len(payload),
        "validation": report.as_dict(),
        "correlation": summary,
        "preview": _clean_preview(preview),
        "canonical_columns": ingest_io.CANONICAL_COLUMNS,
        "next_step": {
            "scored": False,
            "why": (
                "This endpoint validates and correlates an upload. It does "
                "not score it: producing risk requires building the as-of-t "
                "feature matrix over the whole dataset and fitting on the "
                "temporal split, which is an offline pipeline run, not a "
                "request. Scoring here would produce numbers no manifest "
                "describes."
            ),
            "command": 'make run ARGS="phase6-dataset" && '
                       'make run ARGS="phase7-alerts"',
        },
    }


def _clean_preview(rows) -> list:
    """NaN and numpy scalars out; JSON cannot carry either."""
    import pandas as pd

    cleaned = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, list):
                item[key] = [str(v) for v in value]
            elif value is None or (
                not isinstance(value, (list, dict)) and pd.isna(value)
            ):
                item[key] = None
            elif hasattr(value, "item"):
                item[key] = value.item()
            else:
                item[key] = value
        cleaned.append(item)
    return cleaned
