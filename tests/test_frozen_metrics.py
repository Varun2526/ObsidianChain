"""The frozen numbers, asserted against the real dataset when it is present.

Every value here is copied from HANDOFF.md section 4. They are the anchor for
every claim the project makes, and a change in any of them means something
broke rather than improved. Phase 4.1 was an additive/corrective change and
none of them may move.

Skips rather than fails when ``data/raw`` holds no Elliptic++ - the suite has
to run on a machine without the download. Inside the container ``make test``
mounts ``/data``, so these do run in the normal loop.

The two heavier verifications live outside the suite because they cost
minutes rather than seconds:

* **Phase 3.1** - the funnel over 253,429 proposed unions. Verified
  bit-identical on every one of its 13 columns after the pooled-evidence
  extraction; the in-suite equivalent is the golden master in
  ``test_replay.py``.
* **Phase 3.3** - five regimes on the real chain (~6 minutes) and the
  reach-stress fixture. Both verified byte-identical in the terminal report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))

#: Phase 2, HANDOFF section 4. The anchor for every Phase 2 and Phase 3
#: number in the project.
FROZEN_DATASET_SHA256 = (
    "c405493d5bd904c0e17c840dca79386a7502cdde45518810fe8398d7adf9380a"
)
FROZEN_RECORD_COUNT = 1_589_863
FROZEN_TRANSACTION_COUNT = 202_804
FROZEN_GENERATOR_VERSION = "2.0.0"

#: Phase 1, HANDOFF section 4.
PHASE1 = {
    "input_rows": 477_117,
    "transactions": 202_804,
    "star_edges": 274_313,
    "cospend_merges": 253_429,
    "clusters": 569_513,
    "largest_cluster": 14_885,
    "coverage_pct": 34.60,
}


def _skip_without(path: Path, what: str) -> None:
    if not path.exists():
        pytest.skip(f"{what} not present at {path}")


# ---- the frozen synthetic dataset --------------------------------------


def test_frozen_dataset_hash_is_unchanged() -> None:
    manifest = DATA_ROOT / "processed" / "network" / "manifest.json"
    _skip_without(manifest, "frozen network manifest")
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert recorded["dataset_sha256"] == FROZEN_DATASET_SHA256
    assert recorded["record_count"] == FROZEN_RECORD_COUNT
    assert recorded["transaction_count"] == FROZEN_TRANSACTION_COUNT
    assert recorded["generator_version"] == FROZEN_GENERATOR_VERSION


def test_the_frozen_observations_file_matches_its_recorded_hash() -> None:
    """The manifest is only trustworthy if it still describes the file."""
    import hashlib

    observations = DATA_ROOT / "processed" / "network" / "observations.parquet"
    _skip_without(observations, "frozen observations")
    digest = hashlib.sha256()
    with observations.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    assert digest.hexdigest() == FROZEN_DATASET_SHA256


# ---- Phase 1 -----------------------------------------------------------


@pytest.fixture(scope="module")
def phase1():
    from obsidianchain.io import elliptic

    _skip_without(DATA_ROOT / "raw", "Elliptic++ raw data")
    try:
        graph = elliptic.load_cospend_graph(DATA_ROOT)
    except FileNotFoundError as exc:
        pytest.skip(f"Elliptic++ not loadable: {exc}")

    from obsidianchain.cluster.pipeline import run_clustering

    return graph, run_clustering(graph, label="frozen-check")


def test_phase1_graph_shape_is_unchanged(phase1) -> None:
    graph, _ = phase1
    assert graph.n_input_rows == PHASE1["input_rows"]
    assert graph.n_transactions == PHASE1["transactions"]
    assert graph.n_edges == PHASE1["star_edges"]


def test_phase1_clustering_is_unchanged(phase1) -> None:
    _, run = phase1
    assert run.cospend_merges == PHASE1["cospend_merges"]
    assert run.n_clusters == PHASE1["clusters"]
    assert run.largest == PHASE1["largest_cluster"]
    assert round(run.coverage, 2) == PHASE1["coverage_pct"]


def test_phase1_is_unaffected_by_the_network_layer(phase1) -> None:
    """Sanity: Phase 1 is chain-only and no Phase 4.1 change can reach it.

    Recorded explicitly because "the clustering changed" and "the evidence
    changed" are different failures and the fix for each is different.
    """
    graph, run = phase1
    assert run.change_edges == 0
    assert run.change_merges == 0


# ---- the production rule -----------------------------------------------


def test_the_production_rule_is_unchanged() -> None:
    """25 pooled per side, alpha 1e-4, effect floor 0.05. HANDOFF invariant 3."""
    from obsidianchain.network.separation import SeparationConfig

    config = SeparationConfig()
    assert config.min_pooled_observations == 25
    assert config.min_observer_observations == 5
    assert config.alpha == 1e-4
    assert config.min_effect == 0.05


def test_sigma_is_unchanged() -> None:
    """HANDOFF invariant 2: the one parameter with a published source."""
    from obsidianchain.network import synthetic

    assert round(synthetic.DECKER_WATTENHOFER_SIGMA, 4) == 1.1506
    assert synthetic.FROZEN_SEPTEMBER_2026.propagation.sigma == pytest.approx(
        synthetic.DECKER_WATTENHOFER_SIGMA
    )


def test_the_verdict_space_is_still_cannot_link_only() -> None:
    """HANDOFF invariant 5. Phase 4.1 added no evidence state.

    Guards the specific risk of the enum-derivation fix: now that the export
    follows the enum, adding a member is cheap - and adding a must-link
    member would invert the whole design.
    """
    from obsidianchain.network.separation import Verdict

    assert {v.value for v in Verdict} == {
        "SEPARATED",
        "NOT_SEPARATED",
        "NO_EVIDENCE",
    }
    assert not any(
        "MUST" in v.value or "SAME_ORIGIN" in v.value for v in Verdict
    )
