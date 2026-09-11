"""Provenance that travels with the file, not with the terminal.

The problem this fixes
----------------------
Every terminal renderer in this project carries a banner saying the network
data is synthetic. **No output file carried anything.** A row of
``phase33_decisions.csv`` reads::

    regime,decision_id,component_a,component_b,evidence_state,chi2,p_value,blocked
    D,417,88213,90114,SEPARATED,64.4,6.33e-11,True

Opened in a spreadsheet six weeks later, detached from the terminal that
produced it, that is indistinguishable from a measurement about two real
Bitcoin entities. It is a decision made by a synthetic engine on a synthetic
chain against a synthetic origin assignment. The banner was on the ephemeral
artifact and absent from the durable one.

So provenance is attached two ways, and both are deliberate:

* a ``provenance_type`` column **in every row**, because a row is the unit
  that gets copied into an email or a slide;
* a sibling ``<name>.meta.json`` holding the full record - dataset id and
  hash, world, generator version, the production rule that was in force, and
  the git revision when one is discoverable.

Three types, and why two are not enough
---------------------------------------
``DEMO`` alone would be wrong. The Phase 3.3 controlled worlds are synthetic
but are not demonstration scenarios: they are a real experiment on generated
data, and conflating them with the five hand-built demo fixtures would
misrepresent both. So:

``PRODUCTION``
    Produced by the production pipeline on the frozen dataset. Note that
    ``synthetic_network`` is still true for anything network-derived: the
    chain is real Elliptic++, the announcements are generated.

``SYNTHETIC_CONTROL``
    A controlled world (regimes A-E) or the reach-stress fixture. Generated
    to make a question answerable; not a measurement of anything.

``DEMO``
    One of the five demonstration scenarios. Built to reach a specific
    engine state. Not a result.

What this module does not do
----------------------------
It does not touch a single scientific value. It adds columns and a sidecar
around frames that are computed exactly as they were before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

import pandas as pd

#: Bump when the sidecar layout changes so an old file stays readable.
#:
#: /2 added ``inputs`` and ``run_fingerprint``. A /1 sidecar simply lacks
#: them; nothing that read /1 breaks, but a consumer that needs an evidence
#: identity can require /2 rather than guess whether the absence means "no
#: inputs" or "written before inputs were recorded".
PROVENANCE_SCHEMA = "obsidianchain.provenance/2"

#: The one column added to every exported row.
PROVENANCE_COLUMN = "provenance_type"

#: Added only to rows carrying network-derived inference. Not duplicated
#: metadata: it answers a different question from ``provenance_type``, namely
#: whether the announcements behind the statistic were generated.
SYNTHETIC_NETWORK_COLUMN = "synthetic_network"

#: Suffix for the sidecar. Sits next to the artifact so a copied directory
#: keeps its provenance.
META_SUFFIX = ".meta.json"

#: Parquet footer key carrying the artifact's own run fingerprint.
#:
#: An artifact and its sidecar are two files, so publishing them is two
#: filesystem operations and no ordering makes that one atomic step. A crash
#: between them leaves a new parquet beside an old sidecar - and because the
#: API reads the CURRENT fingerprint out of the sidecar, that state does not
#: look broken. It looks like a valid run, and every evidence id minted
#: against the old fingerprint resolves cheerfully against the new rows.
#: Silent re-pointing is the exact failure the fingerprint exists to prevent,
#: so a torn publish must not be able to produce it.
#:
#: Writing the fingerprint into the parquet's own footer makes the pair
#: self-describing: the identity is carried by BOTH files, so a mismatch is
#: detectable by reading them, whichever half is stale. The check lives in
#: ``api/provenance_gate.require_artifact_identity``.
#:
#: The footer, specifically, because it costs nothing: pyarrow reads
#: key-value metadata without touching a single row group, so the guard is a
#: few hundred microseconds and does not scale with the file.
FINGERPRINT_METADATA_KEY = "obsidianchain.run_fingerprint"

#: Same idea for the content schema, so a /1 artifact relabelled by a
#: hand-edited sidecar is caught by the file itself.
ARTIFACT_SCHEMA_METADATA_KEY = "obsidianchain.artifact_schema"


class ProvenanceType(str, Enum):
    """Which pipeline produced an artifact. See the module docstring."""

    PRODUCTION = "PRODUCTION"
    SYNTHETIC_CONTROL = "SYNTHETIC_CONTROL"
    DEMO = "DEMO"


def git_revision(start: Path | None = None) -> str | None:
    """Best-effort git revision, without running git.

    Reads ``.git/HEAD`` and resolves it by hand. A subprocess call would need
    git installed in the container, which it is not, and the whole point of
    this project is that it runs with no network and few assumptions. Returns
    None when there is no repository to read - inside the container there
    usually is not, and "unknown" is an honest answer.
    """
    here = Path(start) if start is not None else Path(__file__).resolve()
    for directory in [here, *here.parents]:
        git = directory / ".git"
        if not git.exists():
            continue
        try:
            head = (git / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref:"):
                ref = head.split(" ", 1)[1].strip()
                candidate = git / ref
                if candidate.is_file():
                    return candidate.read_text(encoding="utf-8").strip()[:40]
                packed = git / "packed-refs"
                if packed.is_file():
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line.endswith(f" {ref}"):
                            return line.split(" ", 1)[0][:40]
                return None
            return head[:40]
        except OSError:
            return None
    return None


@dataclass(frozen=True)
class Provenance:
    """Where an artifact came from, and what it is not.

    Only ``provenance_type`` and ``dataset_id`` are required. Everything else
    is genuinely optional, because a real dataset may not have a hash, a
    world, a generator version or a git revision, and inventing placeholders
    would make the record less trustworthy rather than more complete.
    """

    provenance_type: ProvenanceType
    dataset_id: str
    """What produced the inputs, e.g. "elliptic++" or "worlds/D"."""

    dataset_sha256: str | None = None
    world: str | None = None
    """Regime key for a controlled world; None for production."""

    synthetic_network: bool | None = None
    """True where the announcements behind the numbers were generated. None
    for artifacts derived from the chain alone, where the question does not
    arise."""

    generator_version: str | None = None
    production_rule: Mapping[str, float] | None = None
    """The separation rule in force. Recorded because it is settable from the
    CLI, and a blocked-merge count produced under a loosened rule is
    otherwise indistinguishable from one produced under the real one."""

    inputs: Mapping[str, str] | None = None
    """Every input that determines a row of this artifact, as name -> value.

    Added in schema /2 because ``dataset_sha256`` records the NETWORK dataset
    only, while a row of the evidence funnel is also determined by two raw
    chain files and by two settings that live in code. An identity built on
    ``dataset_sha256`` alone can silently re-point: edit a chain file, leave
    the network dataset alone, and the same identity resolves to a different
    row with no error anywhere.

    Values are hex digests for files and canonical labels for settings. See
    :mod:`obsidianchain.run_fingerprint` for what goes in and why."""

    artifact: Mapping[str, object] | None = None
    """Artifact-specific contract block.

    The sidecar's top level describes the OBSERVATION layer - which dataset,
    which generator, whether the network is synthetic. This block describes
    what was done to it: the artifact's own content schema, the trajectory
    walked, and any evaluation configuration applied. Kept separate so that
    ``provenance_type: PRODUCTION`` sitting beside a production rule cannot be
    read as "production data evaluated by the production rule" - one of those
    words means pipeline and the other means rule."""

    run_fingerprint: str | None = None
    """Digest over ``inputs``, so a consumer can compare one string instead of
    re-deriving the fold. Full 64 characters; the public identity truncates."""

    git_revision: str | None = field(default_factory=git_revision)
    notes: tuple[str, ...] = ()

    @property
    def demo(self) -> bool:
        return self.provenance_type is ProvenanceType.DEMO

    @property
    def is_measurement(self) -> bool:
        """False for anything a judge must not read as a measurement.

        Production output over synthetic announcements is not a measurement
        of Bitcoin either, which is why this is derived rather than set.
        """
        if self.provenance_type is not ProvenanceType.PRODUCTION:
            return False
        return not bool(self.synthetic_network)

    def row_fields(self) -> dict[str, object]:
        """The marker columns added to every exported row.

        Deliberately minimal - two columns at most. The full record lives in
        the sidecar; repeating it per row would bloat a 253,429-row export
        for no gain.
        """
        fields: dict[str, object] = {
            PROVENANCE_COLUMN: self.provenance_type.value
        }
        if self.synthetic_network is not None:
            fields[SYNTHETIC_NETWORK_COLUMN] = bool(self.synthetic_network)
        return fields

    def as_meta(self) -> dict:
        """The full record, for the sidecar."""
        return {
            "schema": PROVENANCE_SCHEMA,
            "provenance_type": self.provenance_type.value,
            "demo": self.demo,
            "is_measurement": self.is_measurement,
            "dataset_id": self.dataset_id,
            "dataset_sha256": self.dataset_sha256,
            "world": self.world,
            "synthetic_network": self.synthetic_network,
            "generator_version": self.generator_version,
            "production_rule": dict(self.production_rule)
            if self.production_rule is not None
            else None,
            "inputs": dict(self.inputs) if self.inputs is not None else None,
            "artifact": dict(self.artifact) if self.artifact is not None else None,
            "run_fingerprint": self.run_fingerprint,
            "git_revision": self.git_revision,
            "notes": list(self.notes),
        }


def rule_config(config) -> dict[str, float]:
    """A :class:`SeparationConfig` as plain data, for the record."""
    return {
        "min_pooled_observations": int(config.min_pooled_observations),
        "min_observer_observations": int(config.min_observer_observations),
        "alpha": float(config.alpha),
        "min_effect": float(config.min_effect),
    }


def dataset_hash(manifest: Mapping | None) -> str | None:
    """Pull the dataset hash out of a generator manifest, if present."""
    if not manifest:
        return None
    value = manifest.get("dataset_sha256")
    return str(value) if value else None


def write_meta(path, provenance: Provenance) -> Path:
    """Write the sidecar for the artifact at ``path``."""
    target = Path(str(path) + META_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(provenance.as_meta(), indent=2), encoding="utf-8"
    )
    return target


def read_meta(path) -> dict:
    """Read the sidecar for the artifact at ``path``."""
    return json.loads(
        Path(str(path) + META_SUFFIX).read_text(encoding="utf-8")
    )


def stamp(frame: pd.DataFrame, provenance: Provenance) -> pd.DataFrame:
    """Return ``frame`` with the provenance marker columns in front.

    Leading rather than trailing so the marker is the first thing visible
    when the file is opened, before any number.
    """
    fields = provenance.row_fields()
    stamped = frame.copy()
    for column, value in reversed(list(fields.items())):
        if column in stamped.columns:
            stamped = stamped.drop(columns=[column])
        stamped.insert(0, column, value)
    return stamped


def write_frame(
    frame: pd.DataFrame,
    path,
    provenance: Provenance,
    *,
    stamp_rows: bool = True,
    index: bool = False,
    **to_csv_kwargs,
) -> Path:
    """Write a durable artifact with its provenance attached both ways.

    Format follows the suffix: ``.parquet`` or anything else as CSV, which is
    what the existing writers did.

    ``stamp_rows=False`` skips the row columns and writes only the sidecar.
    Use it where a row cannot carry an extra column - none of the current
    artifacts is in that position, and the flag exists so that a future one
    does not quietly lose its sidecar too.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = stamp(frame, provenance) if stamp_rows else frame
    if path.suffix == ".parquet":
        write_parquet_with_identity(payload, path, provenance, index=index)
    else:
        payload.to_csv(path, index=index, **to_csv_kwargs)
    write_meta(path, provenance)
    return path


def artifact_identity(provenance: Provenance) -> dict[str, str]:
    """The identity keys this provenance contributes to a parquet footer.

    Empty when there is no fingerprint to carry. Several artifacts in this
    project legitimately have none - a purity CSV has no evidence identity -
    and inventing one so the dict is never empty would make "unidentified"
    and "identified" indistinguishable, which is the mistake
    ``run_fingerprint._require_digest`` already refuses to make.
    """
    identity: dict[str, str] = {}
    if provenance.run_fingerprint:
        identity[FINGERPRINT_METADATA_KEY] = str(provenance.run_fingerprint)
    schema = (dict(provenance.artifact or {})).get("artifact_schema")
    if schema:
        identity[ARTIFACT_SCHEMA_METADATA_KEY] = str(schema)
    return identity


def write_parquet_with_identity(
    payload: pd.DataFrame, path, provenance: Provenance, *, index: bool = False
) -> Path:
    """Write a parquet whose footer carries its own run fingerprint.

    Goes through pyarrow rather than ``DataFrame.to_parquet`` because the
    existing pandas metadata in the footer must be PRESERVED, not replaced:
    dropping it would change how the file reads back (index handling, dtypes
    including the nullable Int64 that section 1 pins) while looking like a
    pure metadata addition. So the identity keys are merged into whatever
    schema pyarrow derived, and nothing else about the write changes.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    identity = artifact_identity(provenance)
    table = pa.Table.from_pandas(payload, preserve_index=index or None)
    if identity:
        merged = dict(table.schema.metadata or {})
        merged.update(
            {key.encode(): value.encode() for key, value in identity.items()}
        )
        table = table.replace_schema_metadata(merged)
    pq.write_table(table, path)
    return path


def read_artifact_identity(path) -> dict[str, str]:
    """Read the identity keys out of a parquet footer.

    Returns an empty dict for a file that carries none, so a caller can tell
    "this artifact predates the identity keys" from "this artifact disagrees
    with its sidecar". Those need different answers: the first is an
    artifact to regenerate, the second is a torn publish.
    """
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(Path(path)).schema_arrow.metadata or {}
    decoded = {
        key.decode(): value.decode()
        for key, value in metadata.items()
        if key.decode() in (
            FINGERPRINT_METADATA_KEY, ARTIFACT_SCHEMA_METADATA_KEY
        )
    }
    return decoded
