"""The information boundary between the synthetic generator and Phase 3.

Phase 3 must work from evidence a real Bitcoin capture could supply. The
generator necessarily knows more than that - it invented the answer - so the
two must be kept apart by construction rather than by discipline. This module
is the only sanctioned way for an inference stage to read network data, and
it refuses to hand over anything the boundary excludes.

What crosses the boundary
-------------------------
::

    txid           which transaction was announced
    observer_id    which of our own vantage points saw it
    peer_ip        which peer relayed it to us
    peer_port      that peer's port
    peer_asn       that peer's autonomous system
    timestamp_ms   when our clock says we saw it

Plus two things a deployment genuinely has: the roster of vantage points it
operates, and a public list of known-broadcaster IPs (exchange and wallet
server infrastructure is documented).

What does not
-------------
``true_origin_id``, the origin roster with its regions and ASNs, the
observers' actual clock offsets, any entity label, and any generator
internal. A real capture supplies none of these. Clock offset in particular
is *estimated* in a deployment, never known, so handing over the true value
would let Phase 3 undo the very noise the generator added.

Why a module rather than a convention
-------------------------------------
The failure mode is not malice, it is a convenient join. Someone debugging
Phase 3 merges the origin roster to see what is going on, the numbers improve,
and the improvement is never traced back to the join. :func:`assert_no_leakage`
makes that mistake loud: it inspects a frame's columns and raises on anything
outside the permitted set, and it is wired into every load path here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from obsidianchain.network import synthetic

#: The complete set of fields an inference stage may consume.
PHASE3_ALLOWED_FIELDS = frozenset(
    {"txid", "observer_id", "peer_ip", "peer_port", "peer_asn", "timestamp_ms"}
)

#: Fields that must never reach inference. Matched case-insensitively, and
#: as substrings, so a renamed copy (`origin_id_x`, `true_origin`) is caught.
FORBIDDEN_SUBSTRINGS = (
    "true_origin",
    "origin_id",
    "origin_node",
    "node_id",
    "broadcaster_flag",
    "is_known_broadcaster",
    "clock_bias",
    "entity",
    "region",
    "ground_truth",
)


class GroundTruthLeakError(AssertionError):
    """Raised when data outside the information boundary reaches inference."""


def assert_no_leakage(frame: pd.DataFrame, context: str = "frame") -> None:
    """Raise if ``frame`` carries anything the boundary excludes.

    Checks column names only. It cannot detect a value smuggled inside an
    allowed column, which is why the loaders below also pin the exact schema
    rather than merely screening for forbidden names.
    """
    offenders = [
        column
        for column in frame.columns
        if any(bad in str(column).lower() for bad in FORBIDDEN_SUBSTRINGS)
    ]
    if offenders:
        raise GroundTruthLeakError(
            f"{context} exposes ground truth: {sorted(offenders)}. "
            f"Phase 3 may only consume {sorted(PHASE3_ALLOWED_FIELDS)}."
        )


def _reject_truth_path(path: Path) -> None:
    if synthetic.TRUTH_DIR in Path(path).parts:
        raise GroundTruthLeakError(
            f"{path} is inside {synthetic.TRUTH_DIR}/, which holds ground "
            f"truth. The inference stage must not read it."
        )


@dataclass
class Phase3Inputs:
    """Everything the inference stage is allowed to see, and nothing else."""

    observations: pd.DataFrame
    observers: pd.DataFrame
    broadcaster_ips: set[str]
    manifest: dict

    @property
    def n_transactions(self) -> int:
        return int(self.observations["txid"].nunique())

    @property
    def n_records(self) -> int:
        return int(len(self.observations))


def observations_dir(processed_root) -> Path:
    return Path(processed_root) / synthetic.OBSERVATIONS_DIR


def load_observations(processed_root, filename: str | None = None) -> pd.DataFrame:
    """Load announcement records, refusing anything off-boundary."""
    directory = observations_dir(processed_root)
    if filename is not None:
        path = directory / filename
    else:
        parquet = directory / "observations.parquet"
        path = parquet if parquet.is_file() else directory / "observations.csv"
    _reject_truth_path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run 'make run ARGS=\"network-generate\"' first."
        )
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    assert_no_leakage(frame, context=str(path.name))
    extra = set(frame.columns) - PHASE3_ALLOWED_FIELDS
    if extra:
        raise GroundTruthLeakError(
            f"{path.name} carries unexpected columns {sorted(extra)}; the "
            f"observation schema is fixed at {sorted(PHASE3_ALLOWED_FIELDS)}."
        )
    return frame


def load_observers(processed_root) -> pd.DataFrame:
    """The vantage points we operate. No clock offsets - those are estimated."""
    path = observations_dir(processed_root) / "observers.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["observer_id", "asn"])
    frame = pd.read_csv(path)
    # 'region' is generator metadata about our own nodes; harmless in itself,
    # but the screen is absolute so it is dropped rather than special-cased.
    frame = frame.loc[:, [c for c in frame.columns if c != "region"]]
    assert_no_leakage(frame, context="observers.csv")
    return frame


def load_broadcaster_ips(processed_root) -> set[str]:
    """Public list of known-broadcaster infrastructure IPs."""
    path = observations_dir(processed_root) / "broadcaster_ips.csv"
    if not path.is_file():
        return set()
    return set(pd.read_csv(path)["ip"])


def load_manifest(processed_root) -> dict:
    import json

    path = observations_dir(processed_root) / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_phase3_inputs(processed_root) -> Phase3Inputs:
    """The single sanctioned entry point for an inference stage."""
    return Phase3Inputs(
        observations=load_observations(processed_root),
        observers=load_observers(processed_root),
        broadcaster_ips=load_broadcaster_ips(processed_root),
        manifest=load_manifest(processed_root),
    )


# ---- evaluation side, deliberately named to be conspicuous -------------


def load_ground_truth_FOR_EVALUATION_ONLY(processed_root) -> pd.DataFrame:
    """Ground truth, for scoring Phase 3 output *after* it has been produced.

    The name is shouted so that a call site inside an inference path is
    obvious in review and in a grep. If this appears anywhere that computes
    a constraint, a similarity, or a merge decision, that result is invalid.
    """
    path = Path(processed_root) / synthetic.TRUTH_DIR / "ground_truth.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run 'make run ARGS=\"network-generate\"' first."
        )
    return pd.read_csv(path)
