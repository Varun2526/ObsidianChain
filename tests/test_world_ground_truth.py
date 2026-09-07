"""The experiment must be capable of the measurements it claims to make.

Two properties are load-bearing, and Phase 3.3 was run once without either
of them holding:

1. Some co-spend edges must connect DIFFERENT hidden entities. Without that
   the engine is never asked to separate two genuine entities, every block
   is a false split by construction, and constraint precision cannot be
   computed at all. Measured before the fix: zero of 274,313 edges.

2. Regime D must contain SAME-entity multi-origin candidates - one entity,
   one component, several origins. Without that the false-split trap cannot
   fire whatever the rule does, and D's pass means nothing.

These tests fail loudly if either property is lost again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obsidianchain.io.elliptic import CoSpendGraph
from obsidianchain.network import reach_stress, worlds


def make_graph(edges, n_addresses):
    array = np.array(edges, dtype=np.int32).reshape(-1, 2)
    return CoSpendGraph(
        n_addresses=n_addresses,
        edges=array,
        edge_tx_ids=np.arange(len(array), dtype=np.int64),
        universe_codes=np.arange(n_addresses, dtype=np.int32),
        n_transactions=len(array),
        n_input_rows=0,
        n_input_pairs=0,
        n_universe_addresses=n_addresses,
        n_input_only_addresses=0,
        addresses=np.array([f"a{i}" for i in range(n_addresses)], dtype=object),
    )


def chain_of_components(n_components: int, size: int):
    """n components, each a path over `size` addresses."""
    edges = []
    for c in range(n_components):
        base = c * size
        edges.extend((base + i, base + i + 1) for i in range(size - 1))
    return make_graph(edges, n_components * size)


# ---- property 1: genuine cross-entity co-spend candidates --------------


def test_some_components_hold_more_than_one_entity() -> None:
    graph = chain_of_components(200, 8)
    config = worlds.WorldConfig(n_entities=20, split_probability=0.5, base_seed=7)
    entity, entity_map = worlds.assign_entities(graph, config)

    assert int(entity_map["is_split"].sum()) > 0, "no component was split"
    from obsidianchain.cluster.pipeline import run_clustering

    roots = run_clustering(graph, "t").roots
    per = pd.DataFrame({"root": roots, "e": entity}).groupby("root")["e"].nunique()
    assert int((per > 1).sum()) > 0, "every component still holds one entity"


def test_genuine_cross_entity_edges_exist() -> None:
    """THE property whose absence made precision untestable in Phase 3.3."""
    graph = chain_of_components(200, 8)
    config = worlds.WorldConfig(n_entities=20, split_probability=0.5, base_seed=7)
    entity, _ = worlds.assign_entities(graph, config)

    cross = int((entity[graph.edges[:, 0]] != entity[graph.edges[:, 1]]).sum())
    assert cross > 0, (
        "no co-spend edge connects two different entities; constraint "
        "precision would be structurally 0 and untestable"
    )


def test_legitimate_same_entity_edges_also_exist() -> None:
    """Both kinds of candidate are needed, or precision is trivially 1 or 0."""
    graph = chain_of_components(200, 8)
    config = worlds.WorldConfig(n_entities=20, split_probability=0.5, base_seed=7)
    entity, _ = worlds.assign_entities(graph, config)

    same = int((entity[graph.edges[:, 0]] == entity[graph.edges[:, 1]]).sum())
    cross = int((entity[graph.edges[:, 0]] != entity[graph.edges[:, 1]]).sum())
    assert same > 0 and cross > 0
    share = cross / (same + cross)
    assert 0.01 < share < 0.99, f"cross-entity share {share:.3f} is degenerate"


def test_small_components_are_never_split() -> None:
    """Splitting a tiny component gives each entity too little to pool."""
    graph = chain_of_components(300, 2)  # every component has 2 addresses
    config = worlds.WorldConfig(
        n_entities=20, split_probability=1.0, min_split_size=4, base_seed=7
    )
    _, entity_map = worlds.assign_entities(graph, config)
    assert int(entity_map["is_split"].sum()) == 0


def test_split_entities_differ_from_the_primary() -> None:
    graph = chain_of_components(200, 8)
    config = worlds.WorldConfig(n_entities=20, split_probability=1.0, base_seed=7)
    _, entity_map = worlds.assign_entities(graph, config)
    split = entity_map[entity_map["is_split"]]
    assert len(split) > 0
    assert (split["primary_entity"] != split["secondary_entity"]).all()


def test_assignment_is_deterministic_and_shared_across_regimes() -> None:
    """Same seed, same assignment - that is what makes regimes comparable."""
    graph = chain_of_components(100, 6)
    config = worlds.WorldConfig(n_entities=15, base_seed=11)
    first, _ = worlds.assign_entities(graph, config)
    second, _ = worlds.assign_entities(graph, config)
    np.testing.assert_array_equal(first, second)


def test_split_probability_controls_how_many_split() -> None:
    graph = chain_of_components(400, 8)
    counts = []
    for probability in (0.0, 0.5, 1.0):
        config = worlds.WorldConfig(
            n_entities=20, split_probability=probability, base_seed=3
        )
        _, entity_map = worlds.assign_entities(graph, config)
        counts.append(int(entity_map["is_split"].sum()))
    assert counts[0] == 0
    assert counts[0] < counts[1] < counts[2]


# ---- property 2: regime D same-entity multi-origin candidates ----------


def test_regime_D_gives_each_entity_three_origins() -> None:
    config = worlds.WorldConfig(n_entities=12, n_origins=24)
    matrix = worlds.origin_distribution(
        worlds.Regime.D_MULTI_ORIGIN_ENTITY, config, 4
    )
    assert ((matrix > 0).sum(axis=1) == config.origins_per_entity).all()


def test_reach_stress_D_binds_addresses_to_distinct_origins() -> None:
    """The trap needs subgroups of ONE entity to sit on DIFFERENT origins.

    With a per-transaction draw every address of a multi-origin entity ends
    up with the same three-origin mixture, so no two subgroups differ and the
    trap cannot fire. Regime D in this fixture binds each address to one
    origin instead.
    """
    config = worlds.WorldConfig(n_entities=6, n_origins=24)
    rs_config = reach_stress.ReachStressConfig(n_entities=6, n_origins=24)
    distribution = worlds.origin_distribution(
        worlds.Regime.D_MULTI_ORIGIN_ENTITY, config, 4
    )
    entity_of_address = np.repeat(np.arange(6), 20).astype(np.int32)

    sticky = reach_stress.address_origins(
        worlds.Regime.D_MULTI_ORIGIN_ENTITY, entity_of_address,
        distribution, rs_config, seed=4,
    )
    assert sticky is not None

    frame = pd.DataFrame({"entity": entity_of_address, "origin": sticky})
    per_entity = frame.groupby("entity")["origin"].nunique()
    assert (per_entity > 1).all(), (
        "every entity must span several origins, or the trap cannot fire"
    )
    assert (per_entity <= 3).all(), "an entity must not exceed its three origins"

    # And each address's origin must be one its entity actually uses.
    for entity in range(6):
        support = set(np.flatnonzero(distribution[entity] > 0).tolist())
        used = set(frame.loc[frame["entity"] == entity, "origin"].tolist())
        assert used <= support


def test_other_regimes_keep_the_per_transaction_draw() -> None:
    """Only D diverges; A, B, C, E are untouched by the fixture."""
    config = worlds.WorldConfig(n_entities=6, n_origins=24)
    rs_config = reach_stress.ReachStressConfig(n_entities=6, n_origins=24)
    entity_of_address = np.repeat(np.arange(6), 20).astype(np.int32)
    for regime in worlds.Regime:
        if regime is worlds.Regime.D_MULTI_ORIGIN_ENTITY:
            continue
        distribution = worlds.origin_distribution(regime, config, 4)
        assert reach_stress.address_origins(
            regime, entity_of_address, distribution, rs_config, seed=4
        ) is None


# ---- the reach-stress fixture itself ----------------------------------


def test_reach_stress_threshold_is_not_lowered() -> None:
    """The fixture raises evidence; it must never move the production bar."""
    described = reach_stress.ReachStressConfig().describe()
    assert described["production_min_pooled"] == 25
    from obsidianchain.network.separation import SeparationConfig

    assert SeparationConfig().min_pooled_observations == 25


def test_reach_stress_holds_sigma_and_network_parameters() -> None:
    config = reach_stress.ReachStressConfig()
    described = config.describe()
    assert described["sigma"] == pytest.approx(
        __import__("obsidianchain.network.synthetic", fromlist=["x"]).
        DECKER_WATTENHOFER_SIGMA
    )
    frozen = __import__(
        "obsidianchain.network.synthetic", fromlist=["x"]
    ).FROZEN_SEPTEMBER_2026
    assert config.n_observers == frozen.n_observers
    assert config.n_origins == frozen.n_origins
    assert config.broadcaster_fraction == frozen.broadcaster_fraction
    assert config.missing_observation_rate == frozen.missing_observation_rate


def test_reach_stress_components_can_clear_the_threshold() -> None:
    """Each component must pool far more than 25 transactions."""
    config = reach_stress.ReachStressConfig()
    assert config.expected_pooled_per_component > 25 * 4, (
        "components must clear the threshold with room to spare, or the "
        "fixture tests reach again instead of the rule"
    )


def test_reach_stress_chain_is_well_formed() -> None:
    config = reach_stress.ReachStressConfig(
        n_components=5, addresses_per_component=6,
        transactions_per_component=10, inputs_per_transaction=3,
    )
    chain = reach_stress.build_chain(config)
    addr_tx = chain["addr_tx"]
    assert addr_tx["txId"].nunique() == config.n_transactions
    assert chain["wallets_classes"]["address"].nunique() == config.n_addresses
    per_tx = addr_tx.groupby("txId")["input_address"].nunique()
    assert (per_tx == config.inputs_per_transaction).all()


def test_reach_stress_transactions_stay_inside_one_component() -> None:
    """A transaction must not span components, or the chain is meaningless."""
    config = reach_stress.ReachStressConfig(
        n_components=6, addresses_per_component=5, transactions_per_component=8
    )
    addr_tx = reach_stress.build_chain(config)["addr_tx"]
    index = addr_tx["input_address"].str.replace("rsaddr", "").astype(int)
    component = index // config.addresses_per_component
    per_tx = pd.DataFrame(
        {"txId": addr_tx["txId"].to_numpy(), "component": component.to_numpy()}
    ).groupby("txId")["component"].nunique()
    assert (per_tx == 1).all()


# ---- address-level truth is persisted and used -------------------------


def _world_data(tmp_path, entity_of_address, n_addresses):
    """A minimal WorldData, enough to exercise write_world."""
    return worlds.WorldData(
        regime=worlds.Regime.B_AFFINITY,
        observations=pd.DataFrame(
            {
                "txid": [1], "observer_id": ["obs-00"], "peer_ip": ["192.0.2.1"],
                "peer_port": [8333], "peer_asn": [64512], "timestamp_ms": [1.0],
            }
        ),
        ground_truth=pd.DataFrame(
            {
                "txid": [1], "true_entity_id": [0],
                "true_origin_id": ["origin-000"], "broadcaster_flag": [False],
            }
        ),
        nodes=pd.DataFrame(
            {
                "node_id": ["origin-000"], "ip": ["192.0.2.1"], "port": [8333],
                "asn": [64512], "region": ["eu-west"],
                "is_known_broadcaster": [True],
            }
        ),
        observers=pd.DataFrame(
            {
                "observer_id": ["obs-00"], "asn": [64513], "region": ["eu-west"],
                "clock_bias_ms": [1.0],
            }
        ),
        origin_distribution=np.array([[1.0]]),
        entity_map=pd.DataFrame(
            {
                "component_root": [0], "component_size": [n_addresses],
                "primary_entity": [0], "secondary_entity": [1], "is_split": [True],
            }
        ),
        entity_of_address=entity_of_address,
    )


def test_write_world_persists_address_level_entities(tmp_path) -> None:
    """entity_map is component-keyed and cannot answer 'who owns this address'."""
    entity_of_address = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    world = _world_data(tmp_path, entity_of_address, entity_of_address.size)
    worlds.write_world(world, tmp_path, fmt="csv")

    path = tmp_path / "worlds_truth" / "B" / "entity_assignment.csv"
    assert path.is_file(), "address-level truth was not written"
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["address_code", "true_entity_id"]
    assert frame["address_code"].tolist() == [0, 1, 2, 3, 4]
    assert frame["true_entity_id"].tolist() == [0, 0, 1, 1, 2]


def test_address_truth_is_quarantined_not_readable_by_inference(tmp_path) -> None:
    entity_of_address = np.array([0, 1], dtype=np.int32)
    world = _world_data(tmp_path, entity_of_address, 2)
    worlds.write_world(world, tmp_path, fmt="csv")

    observable = {p.name for p in (tmp_path / "worlds" / "B").iterdir()}
    assert "entity_assignment.csv" not in observable
    assert not any("entity" in name for name in observable)

    from obsidianchain.network import boundary

    with pytest.raises(boundary.GroundTruthLeakError):
        boundary.load_observations(
            tmp_path / "worlds_truth" / "B", filename="entity_assignment.csv"
        )


def test_split_component_keeps_two_entities_at_address_level() -> None:
    """The property the component-keyed lookup destroyed."""
    graph = chain_of_components(200, 8)
    config = worlds.WorldConfig(n_entities=20, split_probability=1.0, base_seed=7)
    entity, entity_map = worlds.assign_entities(graph, config)

    from obsidianchain.cluster.pipeline import run_clustering

    roots = run_clustering(graph, "t").roots
    split_roots = set(entity_map.loc[entity_map["is_split"], "component_root"])
    assert split_roots

    per_component = pd.DataFrame({"root": roots, "e": entity}).groupby("root")["e"]
    for root in list(split_roots)[:20]:
        assert per_component.nunique()[root] == 2, (
            f"split component {root} collapsed to one entity at address level"
        )


def test_component_lookup_would_lose_the_cross_entity_candidates() -> None:
    """Guard against reintroducing the component-keyed lookup.

    Mapping primary_entity through component_root reports zero cross-entity
    edges, which is exactly the silent failure this fix removes.
    """
    graph = chain_of_components(200, 8)
    config = worlds.WorldConfig(n_entities=20, split_probability=1.0, base_seed=7)
    entity, entity_map = worlds.assign_entities(graph, config)

    from obsidianchain.cluster.pipeline import run_clustering

    roots = run_clustering(graph, "t").roots
    naive = entity_map.set_index("component_root")["primary_entity"].reindex(
        roots
    ).to_numpy()

    a, b = graph.edges[:, 0], graph.edges[:, 1]
    assert int((entity[a] != entity[b]).sum()) > 0, "address-level sees candidates"
    assert int((naive[a] != naive[b]).sum()) == 0, (
        "component-level sees none - this is why the lookup had to go"
    )


# ---- the scorer reads address-level truth ------------------------------


def _phase33_source() -> str:
    import pathlib

    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "src" / "obsidianchain" / "eval" / "phase33.py"
    ).read_text(encoding="utf-8")


def test_scorer_reads_address_level_truth() -> None:
    source = _phase33_source()
    assert "entity_assignment.csv" in source
    assert '["address_code"]' in source


def test_scorer_no_longer_uses_the_component_root_lookup() -> None:
    source = _phase33_source()
    assert 'set_index("component_root")' not in source
    assert "entity_map" not in source


def test_scorer_no_longer_reclusters_for_truth_alignment() -> None:
    """Re-deriving the baseline in the scorer was where the bug could return."""
    source = _phase33_source()
    assert "truth-alignment" not in source
    assert "run_clustering" not in source


# ---- no inference input changed ---------------------------------------


def test_writing_address_truth_does_not_touch_observations(tmp_path) -> None:
    """The fix adds a truth file; observations must be byte-identical."""
    entity_of_address = np.array([0, 1, 1, 2], dtype=np.int32)
    world = _world_data(tmp_path, entity_of_address, 4)

    first = tmp_path / "one"
    second = tmp_path / "two"
    worlds.write_world(world, first, fmt="csv")
    worlds.write_world(world, second, fmt="csv")

    a = (first / "worlds" / "B" / "observations.csv").read_bytes()
    b = (second / "worlds" / "B" / "observations.csv").read_bytes()
    assert a == b


def test_manifest_hashes_only_the_observations(tmp_path) -> None:
    """Adding a truth file must not move the dataset hash."""
    import hashlib

    entity_of_address = np.array([0, 1], dtype=np.int32)
    world = _world_data(tmp_path, entity_of_address, 2)
    written = worlds.write_world(world, tmp_path, fmt="csv")

    observations = (tmp_path / "worlds" / "B" / "observations.csv").read_bytes()
    assert written["manifest_data"]["dataset_sha256"] == (
        hashlib.sha256(observations).hexdigest()
    )


def test_production_thresholds_untouched() -> None:
    from obsidianchain.network.separation import SeparationConfig

    config = SeparationConfig()
    assert config.min_pooled_observations == 25
    assert config.alpha == 1e-4
    assert config.min_effect == 0.05
