"""Only PRODUCTION artifacts may back a production endpoint.

The realistic accident this prevents
------------------------------------
``phase33_decisions.csv`` is the only artifact in the project with an
``evidence_state`` column. An implementer looking for a verdict will find it
there and reach for it. That file is ``SYNTHETIC_CONTROL`` and carries three
ground-truth columns, so a single import would both leak truth and relabel a
controlled-world experiment as production output.

``api.boundary.FORBIDDEN_FIELDS`` catches the truth columns if they reach a
response. This gate catches the file before it is read at all, which is the
earlier and better place: the artifact is refused on the strength of its own
sidecar, so no denylist of column names has to be exhaustive.

What is checked
---------------
The sidecar must declare ``provenance_type = PRODUCTION``. Nothing else is
accepted - not ``SYNTHETIC_CONTROL``, not ``DEMO``, not a missing sidecar, not
an unrecognised value. A missing sidecar is a refusal rather than a default,
because "no provenance" and "production provenance" must never be the same
answer.
"""

from __future__ import annotations

import json
from pathlib import Path

from obsidianchain import provenance as prov

#: Sidecar schema the evidence route requires.
#:
#: /2 is the first version carrying ``inputs`` and ``run_fingerprint``. On a
#: /1 sidecar the absence of those keys is ambiguous - written before inputs
#: were recorded, or recorded as empty? - and an evidence identity cannot be
#: built on that ambiguity, so /1 is refused with a message that says which
#: command to re-run.
REQUIRED_SCHEMA = "obsidianchain.provenance/2"

#: Every provenance type that may back a production endpoint. Exactly one.
ACCEPTED_TYPES = (prov.ProvenanceType.PRODUCTION.value,)


class ProvenanceRefusedError(ValueError):
    """Raised when an artifact's provenance does not permit serving it."""


def sidecar_path(artifact_path) -> Path:
    return Path(str(artifact_path) + prov.META_SUFFIX)


def load_sidecar(artifact_path) -> dict:
    """Read an artifact's sidecar, refusing a missing or unreadable one."""
    path = sidecar_path(artifact_path)
    if not path.is_file():
        raise ProvenanceRefusedError(
            f"{path} not found. An artifact with no provenance record cannot "
            f"be served: absent provenance and production provenance must not "
            f"resolve to the same answer."
        )
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ProvenanceRefusedError(
            f"{path} is not readable as JSON: {exc}"
        ) from exc
    if not isinstance(meta, dict):
        raise ProvenanceRefusedError(
            f"{path} holds {type(meta).__name__}, expected an object"
        )
    return meta


def require_production(artifact_path, *, require_inputs: bool = True) -> dict:
    """Return the sidecar, or refuse the artifact.

    Args:
        artifact_path: The artifact whose sidecar is checked.
        require_inputs: Demand schema /2 with a populated ``inputs`` block and
            a ``run_fingerprint``. True for anything that has to carry an
            evidence identity.
    """
    meta = load_sidecar(artifact_path)

    declared = meta.get("provenance_type")
    if declared not in ACCEPTED_TYPES:
        raise ProvenanceRefusedError(
            f"{Path(artifact_path).name} declares provenance_type="
            f"{declared!r}; this endpoint serves {ACCEPTED_TYPES[0]} artifacts "
            f"only. A SYNTHETIC_CONTROL or DEMO artifact served here would be "
            f"relabelled as production output."
        )

    if not require_inputs:
        return meta

    if meta.get("schema") != REQUIRED_SCHEMA:
        raise ProvenanceRefusedError(
            f"{Path(artifact_path).name} has sidecar schema "
            f"{meta.get('schema')!r}; {REQUIRED_SCHEMA} is required because "
            f"the evidence identity is built from its 'inputs' block. "
            f"Regenerate the artifact."
        )
    inputs = meta.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ProvenanceRefusedError(
            f"{Path(artifact_path).name} carries no 'inputs' block; the "
            f"evidence identity cannot be established without it."
        )
    return meta
