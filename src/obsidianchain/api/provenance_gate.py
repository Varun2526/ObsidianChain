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

from obsidianchain import evidence_contract as contract
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


def require_artifact_schema(meta: dict, artifact_path) -> None:
    """Refuse a /1 evidence artifact where /2 fields are about to be served.

    Section 1 of the Phase 5.3 contract: the artifact became an API contract
    the moment an endpoint served it, so its content layout is versioned
    separately from the provenance record. A /1 artifact has no verdict or
    reason columns; serving it would silently omit the two fields the caller
    asked for rather than saying the artifact predates them.
    """
    declared = (meta.get("artifact") or {}).get("artifact_schema")
    if declared != contract.ARTIFACT_SCHEMA:
        raise ProvenanceRefusedError(
            f"{Path(artifact_path).name} declares artifact_schema="
            f"{declared!r}; {contract.ARTIFACT_SCHEMA} is required to serve "
            f"the production verdict and reason. Regenerate with "
            f"'make run ARGS=\"evidence-funnel\"'."
        )


def require_artifact_identity(meta: dict, artifact_path) -> None:
    """Refuse an artifact that disagrees with its own sidecar.

    An artifact and its sidecar are two files, so no publish order makes
    swapping them one atomic step. The dangerous half of a torn publish is
    NEW PARQUET + OLD SIDECAR, because the API takes the current run
    fingerprint from the sidecar: that state does not read as broken, it
    reads as a valid run in which every previously minted evidence id
    resolves against rows it was never minted for. Silent re-pointing is what
    the fingerprint exists to prevent, so it must not be reachable by a
    crash.

    The writer therefore stamps the fingerprint into the parquet's own footer
    as well, and this compares the two. Either half being stale is a
    mismatch, so the check is order-independent and the failure is loud.

    Refused rather than repaired. Rewriting the sidecar from the parquet
    footer would make the artifact whole again, but it would also make a
    half-finished regeneration look like a completed one, and an operator
    who never learns that the last publish was interrupted does not re-run
    it. The message names the command instead.
    """
    declared = meta.get("run_fingerprint")
    if not declared:
        return  # no evidence identity to tear; require_production checks that

    path = Path(artifact_path)
    if path.suffix != ".parquet":
        return

    embedded = prov.read_artifact_identity(path).get(
        prov.FINGERPRINT_METADATA_KEY
    )
    if embedded is None:
        raise ProvenanceRefusedError(
            f"{path.name} carries no embedded run fingerprint while its "
            f"sidecar declares {str(declared)[:16]}. The artifact predates "
            f"the identity stamp, so a torn publish could not be "
            f"distinguished from a complete one. Regenerate with "
            f"'make run ARGS=\"evidence-funnel\"'."
        )
    if embedded != declared:
        raise ProvenanceRefusedError(
            f"{path.name} does not match its sidecar: the file carries run "
            f"fingerprint {embedded[:16]} and the sidecar declares "
            f"{str(declared)[:16]}. This is a TORN PUBLISH - one of the two "
            f"was replaced and the other was not - so every evidence id "
            f"would resolve against rows it was not minted for. Refusing to "
            f"serve. Re-run 'make run ARGS=\"evidence-funnel\"'."
        )


def require_same_run(primary: dict, secondary: dict, name: str) -> None:
    """Refuse a detail table from a different run than its index.

    Phase 7 publishes five artifacts per run and joins them by ``alert_id``.
    A stale detail table would join CLEANLY - the ids look the same - and
    describe a different cluster under the right-looking heading, which is
    the same silent re-pointing the run fingerprint exists to prevent. So
    the fingerprints are compared rather than the join being trusted.
    """
    expected = primary.get("run_fingerprint")
    found = secondary.get("run_fingerprint")
    if expected and found != expected:
        raise ProvenanceRefusedError(
            f"{name} was produced by run {str(found)[:16]} but the alert "
            f"index is run {str(expected)[:16]}. These five artifacts are "
            f"written together and joined by alert_id, so a mismatched pair "
            f"would describe a different cluster without erroring. Re-run "
            f"'make run ARGS=\"phase7-alerts\"'."
        )


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
    # Last, because it reads the artifact itself. The cheap sidecar checks
    # above should reject a wrong file before the footer is opened at all.
    require_artifact_identity(meta, artifact_path)
    return meta
