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
    "true_entity",
    "entity_id",
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


#: Every directory holding ground truth. Both namespaces are refused by the
#: same check, so adding the controlled worlds did not create a second,
#: weaker path into truth.
TRUTH_DIRS = (synthetic.TRUTH_DIR, synthetic.WORLDS_TRUTH_DIR)


def _reject_truth_path(path: Path) -> None:
    parts = set(Path(path).parts)
    for truth_dir in TRUTH_DIRS:
        if truth_dir in parts:
            raise GroundTruthLeakError(
                f"{path} is inside {truth_dir}/, which holds ground truth. "
                f"The inference stage must not read it."
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


def observations_dir(processed_root, world: str | None = None) -> Path:
    """Directory of readable observations, for production or one world.

    ``world`` selects a controlled-world regime ("A".."E"). The frozen
    production dataset is the default and is unaffected by the worlds
    existing.
    """
    root = Path(processed_root)
    if world is None:
        return root / synthetic.OBSERVATIONS_DIR
    return root / synthetic.WORLDS_DIR / str(world)


def load_observations(
    processed_root, filename: str | None = None, world: str | None = None
) -> pd.DataFrame:
    """Load announcement records, refusing anything off-boundary."""
    directory = observations_dir(processed_root, world)
    if filename is not None:
        path = directory / filename
    else:
        parquet = directory / "observations.parquet"
        path = parquet if parquet.is_file() else directory / "observations.csv"
    _reject_truth_path(path)
    if not path.is_file():
        remedy = (
            'make run ARGS="network-generate"'
            if world is None
            else f'make run ARGS="world-generate --regime {world}"'
        )
        raise FileNotFoundError(f"{path} not found. Run '{remedy}' first.")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    assert_no_leakage(frame, context=str(path.name))
    extra = set(frame.columns) - PHASE3_ALLOWED_FIELDS
    if extra:
        raise GroundTruthLeakError(
            f"{path.name} carries unexpected columns {sorted(extra)}; the "
            f"observation schema is fixed at {sorted(PHASE3_ALLOWED_FIELDS)}."
        )
    return frame


def load_observers(processed_root, world: str | None = None) -> pd.DataFrame:
    """The vantage points we operate. No clock offsets - those are estimated."""
    path = observations_dir(processed_root, world) / "observers.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["observer_id", "asn"])
    frame = pd.read_csv(path)
    # 'region' is generator metadata about our own nodes; harmless in itself,
    # but the screen is absolute so it is dropped rather than special-cased.
    frame = frame.loc[:, [c for c in frame.columns if c != "region"]]
    assert_no_leakage(frame, context="observers.csv")
    return frame


def load_broadcaster_ips(processed_root, world: str | None = None) -> set[str]:
    """Public list of known-broadcaster infrastructure IPs."""
    path = observations_dir(processed_root, world) / "broadcaster_ips.csv"
    if not path.is_file():
        return set()
    return set(pd.read_csv(path)["ip"])


def load_manifest(processed_root, world: str | None = None) -> dict:
    import json

    path = observations_dir(processed_root, world) / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_phase3_inputs(processed_root, world: str | None = None) -> Phase3Inputs:
    """The single sanctioned entry point for an inference stage.

    ``world=None`` is the frozen production dataset. ``world="B"`` and so on
    select a controlled-world regime. Either way the caller receives the same
    six observation columns and nothing else.
    """
    return Phase3Inputs(
        observations=load_observations(processed_root, world=world),
        observers=load_observers(processed_root, world=world),
        broadcaster_ips=load_broadcaster_ips(processed_root, world=world),
        manifest=load_manifest(processed_root, world=world),
    )


# ---- evaluation side, deliberately named to be conspicuous -------------


def load_ground_truth_FOR_EVALUATION_ONLY(
    processed_root, world: str | None = None
) -> pd.DataFrame:
    """Ground truth, for scoring Phase 3 output *after* it has been produced.

    The name is shouted so that a call site inside an inference path is
    obvious in review and in a grep. If this appears anywhere that computes
    a constraint, a similarity, or a merge decision, that result is invalid.
    """
    root = Path(processed_root)
    if world is None:
        path = root / synthetic.TRUTH_DIR / "ground_truth.csv"
    else:
        path = root / synthetic.WORLDS_TRUTH_DIR / str(world) / "ground_truth.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; generate the dataset first.")
    return pd.read_csv(path)
