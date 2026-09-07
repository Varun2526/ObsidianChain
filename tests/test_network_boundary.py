"""Regression tests for the Phase 2 -> Phase 3 information boundary.

======================= SCIENTIFIC VALIDATION NOTICE =======================
Phase 2 does NOT provide scientific validation of network-origin inference.
Every test in this file checks *plumbing*: that the generator's private
knowledge cannot reach an inference stage. Passing them says nothing about
whether origin inference works on Bitcoin.

Real validation requires real mainnet observations, plus controlled
wallets/nodes we operate, plus known ground truth. None of that exists yet.
============================================================================

The failure this guards against is not malice. It is somebody debugging
Phase 3, joining the origin roster to see what is happening, watching the
numbers improve, and never tracing the improvement back to the join.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain.network import arrivals, boundary, synthetic

TXIDS = np.arange(1, 601)


@pytest.fixture()
def written(tmp_path: Path):
    config = synthetic.NetworkConfig(
        n_observers=6, n_origins=24, broadcaster_fraction=0.2,
        missing_observation_rate=0.05, seed=4,
    )
    observations, nodes, observers, truth = synthetic.generate(TXIDS, config)
    synthetic.write_outputs(
        observations, nodes, observers, truth, tmp_path, config, fmt="csv"
    )
    return tmp_path, config


# ---- the boundary itself ------------------------------------------------


def test_observations_expose_only_the_permitted_fields(written) -> None:
    root, _ = written
    frame = boundary.load_observations(root)
    assert set(frame.columns) == boundary.PHASE3_ALLOWED_FIELDS


def test_true_origin_is_never_in_the_observations(written) -> None:
    root, _ = written
    frame = boundary.load_observations(root)
    assert not any("origin" in c.lower() for c in frame.columns)


def test_phase3_inputs_carry_no_ground_truth(written) -> None:
    root, _ = written
    inputs = boundary.load_phase3_inputs(root)
    boundary.assert_no_leakage(inputs.observations, "observations")
    boundary.assert_no_leakage(inputs.observers, "observers")
    assert not hasattr(inputs, "ground_truth")
    assert not hasattr(inputs, "true_origin_id")


def test_observer_table_hides_true_clock_bias(written) -> None:
    """A deployment estimates clock offset; it never knows it."""
    root, _ = written
    observers = boundary.load_observers(root)
    assert "clock_bias_ms" not in observers.columns


def test_ground_truth_lives_in_a_separate_directory(written) -> None:
    root, _ = written
    assert (root / synthetic.TRUTH_DIR / "ground_truth.csv").is_file()
    obs_dir = root / synthetic.OBSERVATIONS_DIR
    assert not any("truth" in p.name.lower() for p in obs_dir.iterdir())


def test_loading_from_the_truth_directory_is_refused(written) -> None:
    root, _ = written
    with pytest.raises(boundary.GroundTruthLeakError, match="ground truth"):
        boundary.load_observations(root / synthetic.TRUTH_DIR, filename="x.csv")


@pytest.mark.parametrize(
    "column",
    [
        "true_origin_id", "origin_id", "node_id", "broadcaster_flag",
        "is_known_broadcaster", "clock_bias_ms", "entity_norm", "region",
    ],
)
def test_forbidden_column_is_detected(column: str) -> None:
    frame = pd.DataFrame({"txid": [1], column: ["x"]})
    with pytest.raises(boundary.GroundTruthLeakError, match=column):
        boundary.assert_no_leakage(frame)


def test_renamed_ground_truth_is_still_caught() -> None:
    """A copy under a different name must not slip through."""
    for sneaky in ("my_true_origin", "origin_id_v2", "ORIGIN_ID", "tx_region"):
        with pytest.raises(boundary.GroundTruthLeakError):
            boundary.assert_no_leakage(pd.DataFrame({sneaky: [1]}))


def test_extra_column_on_observations_is_refused(tmp_path: Path) -> None:
    """Even a harmless-looking addition breaks the fixed schema."""
    directory = tmp_path / synthetic.OBSERVATIONS_DIR
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "txid": [1], "observer_id": ["a"], "peer_ip": ["192.0.2.1"],
            "peer_port": [8333], "peer_asn": [64512], "timestamp_ms": [1.0],
            "helpful_hint": ["origin-007"],
        }
    ).to_csv(directory / "observations.csv", index=False)
    with pytest.raises(boundary.GroundTruthLeakError, match="unexpected columns"):
        boundary.load_observations(tmp_path)


def test_ground_truth_accessor_is_conspicuously_named() -> None:
    """The name must be greppable, so misuse is visible in review."""
    assert hasattr(boundary, "load_ground_truth_FOR_EVALUATION_ONLY")
    assert "FOR_EVALUATION_ONLY" in "".join(dir(boundary))


def test_ground_truth_has_the_documented_shape(written) -> None:
    root, _ = written
    truth = boundary.load_ground_truth_FOR_EVALUATION_ONLY(root)
    assert set(truth.columns) == {"txid", "true_origin_id", "broadcaster_flag"}


def test_arrival_vectors_contain_no_origin_information(written) -> None:
    root, _ = written
    inputs = boundary.load_phase3_inputs(root)
    vectors = arrivals.build(inputs.observations)
    frame = vectors.with_evidence(inputs.broadcaster_ips)
    boundary.assert_no_leakage(frame, "arrival vectors")


def test_peer_ip_does_not_identify_the_origin_for_ordinary_tx(written) -> None:
    root, _ = written
    inputs = boundary.load_phase3_inputs(root)
    ordinary = inputs.observations[
        ~inputs.observations["peer_ip"].isin(inputs.broadcaster_ips)
    ]
    per_tx = ordinary.groupby("txid")["peer_ip"].nunique()
    assert per_tx.mean() > 1.5, "observers must disagree about the relaying peer"


# ---- manifest -----------------------------------------------------------


def test_manifest_records_what_is_needed_to_reproduce(written) -> None:
    root, _ = written
    manifest = boundary.load_manifest(root)
    for key in (
        "generator_version", "dataset_sha256", "record_count",
        "transaction_count", "observer_count", "configuration",
    ):
        assert key in manifest, key
    assert manifest["configuration"]["seed"] == 4


def test_manifest_has_no_machine_specific_detail(written) -> None:
    """Two people on different machines must produce identical manifests."""
    root, _ = written
    text = str(boundary.load_manifest(root)).lower()
    for leak in ("/users/", "/home/", "hostname", "tmp", "generated_at"):
        assert leak not in text, leak


def test_manifest_hash_changes_with_the_data(tmp_path: Path) -> None:
    def build(seed: int) -> str:
        cfg = synthetic.NetworkConfig(n_observers=3, n_origins=8, seed=seed)
        obs, nodes, observers, truth = synthetic.generate(TXIDS[:50], cfg)
        root = tmp_path / f"s{seed}"
        synthetic.write_outputs(obs, nodes, observers, truth, root, cfg, fmt="csv")
        return boundary.load_manifest(root)["dataset_sha256"]

    assert build(1) != build(2)
    assert build(1) == build(1)
