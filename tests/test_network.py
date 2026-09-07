"""Tests for the synthetic announcement generator and arrival instrumentation.

TEST CLASSIFICATION
-------------------
INSTRUMENTATION tests check determinism and correct reshaping. They must pass
exactly, every time, and a failure is a bug.

MECHANISM-DEMONSTRATION tests check that the generator produces the structure
it was designed to produce - same origin looking similar, different origins
looking different. They demonstrate what the synthetic model represents. They
are NOT evidence that origin inference works on Bitcoin.

SCIENTIFIC VALIDATION is absent from this file and from Phase 2 entirely. It
requires real mainnet observations, controlled nodes we operate, and known
ground truth. Nothing here substitutes for it.

Each test below is prefixed [INSTRUMENTATION] or [MECHANISM] in its docstring.

The generator is synthetic by design, so these tests check that it is
*honest* about that and that its structural promises hold: determinism under
a seed, the documented record schema, peer_ip not leaking the origin, and
known broadcasters producing flat vectors that carry no ordering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain.network import arrivals, synthetic

TXIDS = np.arange(1, 401)


@pytest.fixture()
def config() -> synthetic.NetworkConfig:
    return synthetic.NetworkConfig(n_observers=6, n_origins=24, seed=11)


@pytest.fixture()
def generated(config: synthetic.NetworkConfig):
    return synthetic.generate(TXIDS, config)


# ---- the honesty requirement -------------------------------------------


def test_module_docstring_states_the_data_is_synthetic() -> None:
    """This is a stated deliverable; it must not be edited away silently."""
    doc = synthetic.__doc__ or ""
    lowered = doc.lower()
    assert "synthetic" in lowered
    assert "validates nothing" in lowered or "validating the method" in lowered
    assert "mainnet capture" in lowered
    assert "ground-truth" in lowered or "ground truth" in lowered


def test_docstring_cites_a_real_measurement_source() -> None:
    doc = synthetic.__doc__ or ""
    assert "Decker" in doc and "2013" in doc


def test_scale_assumption_is_flagged_not_presented_as_measured() -> None:
    doc = synthetic.__doc__ or ""
    assert "ASSUMPTION" in doc or "is NOT from that paper" in doc


def test_sigma_is_derived_from_the_cited_figures() -> None:
    """sigma = sqrt(2 ln(mean/median)) with the paper's 12.6 s and 6.5 s."""
    expected = float(np.sqrt(2 * np.log(12.6 / 6.5)))
    assert synthetic.DECKER_WATTENHOFER_SIGMA == pytest.approx(expected)
    assert synthetic.PropagationModel().sigma == pytest.approx(expected, abs=1e-3)


def test_summary_carries_the_warning(generated, config) -> None:
    observations, nodes, observers, _ = generated
    text = synthetic.format_summary(observations, nodes, observers, config)
    assert "SYNTHETIC DATA" in text
    assert "VALIDATES NOTHING" in text
    assert "ASSUMPTION" in text


# ---- determinism -------------------------------------------------------


def test_same_seed_gives_identical_output(config) -> None:
    a, _, _, _ = synthetic.generate(TXIDS, config)
    b, _, _, _ = synthetic.generate(TXIDS, config)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_output(config) -> None:
    a, _, _, _ = synthetic.generate(TXIDS, config)
    other = synthetic.NetworkConfig(
        n_observers=config.n_observers, n_origins=config.n_origins, seed=12
    )
    b, _, _, _ = synthetic.generate(TXIDS, other)
    assert not a["timestamp_ms"].equals(b["timestamp_ms"])


def test_node_and_observer_tables_are_seeded(config) -> None:
    pd.testing.assert_frame_equal(
        synthetic.build_nodes(config), synthetic.build_nodes(config)
    )


# ---- schema ------------------------------------------------------------


def test_record_schema(generated) -> None:
    observations, _, _, _ = generated
    assert list(observations.columns) == synthetic.RECORD_COLUMNS


def test_one_record_per_transaction_and_observer(generated, config) -> None:
    observations, _, _, _ = generated
    assert len(observations) == len(TXIDS) * config.n_observers
    assert not observations.duplicated(subset=["txid", "observer_id"]).any()


def test_observer_count_is_configurable() -> None:
    for n in (1, 3, 12):
        cfg = synthetic.NetworkConfig(n_observers=n, n_origins=20, seed=2)
        observations, _, observers, _ = synthetic.generate(TXIDS[:50], cfg)
        assert len(observers) == n
        assert observations["observer_id"].nunique() == n


def test_ips_are_documentation_range_only(generated) -> None:
    """RFC 5737 addresses belong to nobody and are not routable."""
    import ipaddress

    _, nodes, _, _ = generated
    allowed = [ipaddress.ip_network(b) for b in synthetic._TEST_NETS]
    for ip in nodes["ip"]:
        assert any(ipaddress.ip_address(ip) in net for net in allowed), ip


def test_nodes_have_asn_and_region(generated) -> None:
    _, nodes, _, _ = generated
    assert nodes["asn"].between(64512, 65534).all()
    assert set(nodes["region"]).issubset(set(synthetic.REGIONS))


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        synthetic.NetworkConfig(n_observers=0)
    with pytest.raises(ValueError):
        synthetic.NetworkConfig(broadcaster_fraction=1.5)


def test_empty_transaction_set(config) -> None:
    observations, _, _, _ = synthetic.generate(np.array([]), config)
    assert list(observations.columns) == synthetic.RECORD_COLUMNS
    assert len(observations) == 0


# ---- modelling promises -------------------------------------------------


def test_peer_ip_does_not_leak_the_origin(generated) -> None:
    """If peer_ip identified the origin, the next phase could cheat.

    For ordinary transactions the relaying peer is drawn from the observer's
    own peer table, so observers must disagree about who announced it.
    """
    observations, nodes, _, _ = generated
    broadcasters = synthetic.known_broadcaster_ips(nodes)
    ordinary = observations[~observations["peer_ip"].isin(broadcasters)]
    per_tx = ordinary.groupby("txid")["peer_ip"].nunique()
    assert per_tx.max() > 1, "observers must not all name the same relay"
    assert per_tx.mean() > 1.5


def test_broadcasters_are_heard_directly_from_one_ip(generated) -> None:
    observations, nodes, _, _ = generated
    broadcasters = synthetic.known_broadcaster_ips(nodes)
    bc = observations[observations["peer_ip"].isin(broadcasters)]
    if len(bc):
        assert bc.groupby("txid")["peer_ip"].nunique().max() == 1


def test_broadcaster_fraction_is_respected() -> None:
    cfg = synthetic.NetworkConfig(n_origins=100, broadcaster_fraction=0.25, seed=5)
    nodes = synthetic.build_nodes(cfg)
    assert int(nodes["is_known_broadcaster"].sum()) == 25


def test_zero_broadcasters_is_allowed() -> None:
    cfg = synthetic.NetworkConfig(n_origins=20, broadcaster_fraction=0.0, seed=5)
    assert not synthetic.build_nodes(cfg)["is_known_broadcaster"].any()


def test_delays_are_positive_and_heavy_tailed(generated, config) -> None:
    observations, _, _, _ = generated
    per_tx = observations.groupby("txid")["timestamp_ms"]
    spread = (per_tx.max() - per_tx.min()).to_numpy()
    assert (spread >= 0).all()
    assert np.median(spread) < spread.max() / 3, "lognormal tail expected"


def test_observers_have_distinct_clock_biases(config) -> None:
    nodes = synthetic.build_nodes(config)
    observers = synthetic.build_observers(config, nodes)
    assert observers["clock_bias_ms"].nunique() == len(observers)
    assert observers["clock_bias_ms"].abs().max() <= config.clock_bias_range_ms


# ---- arrival vectors ---------------------------------------------------


def test_arrival_vectors_hand_checked() -> None:
    """Three observers, one transaction, arrivals at 100, 250 and 175 ms."""
    records = pd.DataFrame(
        {
            "txid": [1, 1, 1],
            "observer_id": ["obs-00", "obs-01", "obs-02"],
            "timestamp_ms": [100.0, 250.0, 175.0],
        }
    )
    v = arrivals.build(records)
    assert v.n_transactions == 1 and v.n_observers == 3
    assert list(v.offsets_ms[0]) == [0.0, 150.0, 75.0]
    assert list(v.ranks[0]) == [0, 2, 1]
    assert v.first_observer[0] == "obs-00"
    assert v.spread_ms[0] == 150.0
    assert v.n_observed[0] == 3


def test_offsets_are_invariant_to_a_shared_clock_shift() -> None:
    base = pd.DataFrame(
        {
            "txid": [1, 1, 1],
            "observer_id": ["a", "b", "c"],
            "timestamp_ms": [10.0, 20.0, 40.0],
        }
    )
    shifted = base.assign(timestamp_ms=base["timestamp_ms"] + 1_000_000)
    np.testing.assert_allclose(
        arrivals.build(base).offsets_ms, arrivals.build(shifted).offsets_ms
    )


def test_ranks_survive_a_monotone_clock_error() -> None:
    """Ranks depend only on ordering, so a stretched clock cannot change them."""
    base = pd.DataFrame(
        {
            "txid": [1, 1, 1],
            "observer_id": ["a", "b", "c"],
            "timestamp_ms": [10.0, 20.0, 40.0],
        }
    )
    stretched = base.assign(timestamp_ms=base["timestamp_ms"] * 3.5 + 7)
    np.testing.assert_array_equal(
        arrivals.build(base).ranks, arrivals.build(stretched).ranks
    )


def test_unobserved_observer_is_nan_and_rank_minus_one() -> None:
    records = pd.DataFrame(
        {
            "txid": [1, 1],
            "observer_id": ["a", "b"],
            "timestamp_ms": [5.0, 9.0],
        }
    )
    v = arrivals.build(records, observer_ids=["a", "b", "c"])
    assert np.isnan(v.absolute_ms[0, 2])
    assert v.ranks[0, 2] == -1
    assert v.n_observed[0] == 2


def test_duplicate_sightings_collapse_to_the_earliest() -> None:
    records = pd.DataFrame(
        {
            "txid": [1, 1, 1],
            "observer_id": ["a", "a", "b"],
            "timestamp_ms": [300.0, 100.0, 500.0],
        }
    )
    v = arrivals.build(records)
    assert v.duplicates_collapsed == 1
    assert v.absolute_ms[0, 0] == 100.0


def test_flat_vectors_are_flagged() -> None:
    records = pd.DataFrame(
        {
            "txid": [1, 1, 2, 2],
            "observer_id": ["a", "b", "a", "b"],
            "timestamp_ms": [100.0, 110.0, 100.0, 5000.0],
        }
    )
    v = arrivals.build(records)
    flat = v.flag_flat(threshold_ms=50.0)  # explicit, not the default
    assert list(flat) == [True, False]


def test_missing_columns_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        arrivals.build(pd.DataFrame({"txid": [1]}))


def test_to_frame_shape(generated, config) -> None:
    observations, _, _, _ = generated
    v = arrivals.build(observations)
    frame = v.to_frame()
    assert len(frame) == len(TXIDS)
    assert sum(c.startswith("offset_ms__") for c in frame.columns) == config.n_observers
    assert sum(c.startswith("rank__") for c in frame.columns) == config.n_observers
    assert {"first_observer", "spread_ms", "n_observed", "is_flat"} <= set(frame.columns)


# ---- the two ends meeting ----------------------------------------------


def test_broadcaster_transactions_are_flat_ordinary_ones_are_not() -> None:
    """The central structural promise: broadcasters carry no ordering.

    A flat vector means the measurement is uninformative, so downstream code
    must be able to identify and drop these rather than infer from them.
    """
    cfg = synthetic.NetworkConfig(
        n_observers=8, n_origins=40, broadcaster_fraction=0.25, seed=3
    )
    observations, nodes, _, _ = synthetic.generate(np.arange(1, 1201), cfg)
    v = arrivals.build(observations)

    broadcaster_ips = synthetic.known_broadcaster_ips(nodes)
    bc_txids = set(
        observations.loc[observations["peer_ip"].isin(broadcaster_ips), "txid"]
    )
    is_bc = np.isin(v.txids, list(bc_txids))
    assert is_bc.any() and (~is_bc).any()

    median_bc = float(np.median(v.spread_ms[is_bc]))
    median_ord = float(np.median(v.spread_ms[~is_bc]))
    assert median_bc < median_ord / 10, (
        f"broadcaster spread {median_bc:.1f} ms should be far below "
        f"ordinary {median_ord:.1f} ms"
    )

    flat = v.flag_flat(threshold_ms=250.0)
    assert flat[is_bc].mean() > 0.7, "most broadcaster vectors must be flat"
    assert flat[~is_bc].mean() < 0.1, "ordinary vectors must retain ordering"


def test_write_and_read_round_trip(generated, tmp_path: Path) -> None:
    observations, nodes, observers, truth = generated
    written = synthetic.write_outputs(
        observations, nodes, observers, truth, tmp_path,
        synthetic.NetworkConfig(), fmt="csv",
    )
    reread = arrivals.read_observations(Path(written["observations"]))
    assert len(reread) == len(observations)

    v = arrivals.build(reread)
    out = tmp_path / "vectors.csv"
    assert arrivals.write(v, out, fmt="csv") == v.n_transactions


def test_unsupported_format_rejected(generated, tmp_path: Path) -> None:
    observations, nodes, observers, truth = generated
    with pytest.raises(ValueError, match="unsupported format"):
        synthetic.write_outputs(
            observations, nodes, observers, truth, tmp_path,
            synthetic.NetworkConfig(), fmt="xml",
        )


def test_missing_observations_file_names_the_remedy(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="network-generate"):
        arrivals.read_observations(tmp_path / "nope.parquet")


def test_summarise_reports_flatness_split(generated) -> None:
    observations, nodes, _, _ = generated
    v = arrivals.build(observations)
    bc_ips = synthetic.known_broadcaster_ips(nodes)
    bc = set(observations.loc[observations["peer_ip"].isin(bc_ips), "txid"])
    text = arrivals.summarise(v, broadcaster_txids=bc)
    assert "SYNTHETIC" in text
    assert "no ordering information" in text
    assert "known-broadcaster" in text
