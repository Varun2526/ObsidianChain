"""World v2 (world/noisy.py): overlap, quarantine, and the NULL control."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obsidianchain.io import ingest
from obsidianchain.pipeline.features_ps import extract_ps_features
from obsidianchain.world import noisy

SMALL = dict(entities_per_weight=3, seed=7)


@pytest.fixture(scope="module")
def world():
    return noisy.build_noisy_world(noisy.NoisyWorldConfig(**SMALL))


def test_generation_is_deterministic() -> None:
    a = noisy.build_noisy_world(noisy.NoisyWorldConfig(**SMALL))[0]
    b = noisy.build_noisy_world(noisy.NoisyWorldConfig(**SMALL))[0]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_the_capture_carries_no_truth(world) -> None:
    capture, _labels, _entities = world
    forbidden = {"y", "label", "entity", "behaviour", "scenario", "connectivity", "class"}
    assert not forbidden & set(capture.columns)
    assert list(capture.columns) == [c for c in ingest.CANONICAL_COLUMNS]


def test_the_capture_parses_through_production_ingest(tmp_path, world) -> None:
    capture = world[0]
    path = tmp_path / "capture.csv"
    capture.to_csv(path, index=False)
    frame, report = ingest.ingest(path)
    assert report.ok
    feats = extract_ps_features(frame, include_network=True)
    assert not feats.duplicated(subset=["address", "txid"]).any()
    # Real within-step timestamps: durations are no longer step multiples.
    nonzero = feats.loc[feats.active_duration_seconds > 0, "active_duration_seconds"]
    assert (nonzero % noisy.TIMESTEP_SECONDS != 0).any()


def test_benign_and_positive_peel_ranges_overlap(world) -> None:
    """Some benign PAYMENT_CHAIN hops fire the peeling flag - that is the point."""
    capture, labels, entities = world
    capture = capture.drop_duplicates("txid")
    first_in = capture.input_addresses.str.split(";").str[0]
    entity = first_in.map(labels.set_index("address")["entity"])
    behaviour = entity.map(entities.set_index("entity")["behaviour"])
    chain = capture[behaviour == noisy.PAYMENT_CHAIN]
    outs = chain.output_amounts.str.split(";").apply(lambda v: [float(x) for x in v])
    dominant = outs.apply(lambda v: max(v) / sum(v) >= 0.8)
    assert dominant.any() and not dominant.all()


def test_label_noise_is_applied(world) -> None:
    entities = world[2]
    big = noisy.build_noisy_world(noisy.NoisyWorldConfig(entities_per_weight=20, seed=3))[2]
    assert big.label_flipped.any()
    assert big.label_flipped.mean() < 0.05


def test_null_world_removes_only_the_relay_network_signal() -> None:
    sig = noisy.build_noisy_world(noisy.NoisyWorldConfig(entities_per_weight=10, network_signal=True))[2]
    null = noisy.build_noisy_world(noisy.NoisyWorldConfig(entities_per_weight=10, network_signal=False))[2]
    relay_sig = sig[sig.behaviour == noisy.RELAY_LAUNDERING].connectivity.mean()
    relay_null = null[null.behaviour == noisy.RELAY_LAUNDERING].connectivity.mean()
    normal = null[null.behaviour == noisy.NORMAL].connectivity.mean()
    assert relay_sig > 0.5
    assert abs(relay_null - normal) < 0.1


def test_the_manifest_declares_synthetic_control(tmp_path) -> None:
    manifest = noisy.write_noisy_world(tmp_path, noisy.NoisyWorldConfig(**SMALL))
    assert manifest["provenance_type"] == "SYNTHETIC_CONTROL"
    assert (tmp_path / "world_truth" / "README.txt").read_text().startswith("FOR_EVALUATION_ONLY")
