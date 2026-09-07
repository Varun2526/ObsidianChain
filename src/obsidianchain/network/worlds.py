"""Controlled world model: five regimes with a known entity-origin relation.

Why this exists
---------------
The Phase 3.1 funnel showed that the frozen dataset carries no recoverable
signal at any pooling level - the separated share *falls* as evidence
accumulates (41.5% at two pooled observations, 0% at twenty-five), which is
the signature of small-sample noise rather than starved evidence. The cause
is structural: ``FROZEN_SEPTEMBER_2026`` draws each transaction's origin
uniformly at random, independent of which addresses it spends, so a
component's pooled centroid converges on the global mean and two large
components are identical by construction.

That makes the frozen dataset unable to answer the question Phase 3 exists
to ask. These worlds inject a *known* entity-origin relationship so the
question becomes answerable, and so a Phase 3 result can be attributed to
the method rather than to the absence of anything to find.

Still synthetic, still not validation
-------------------------------------
Every caveat from :mod:`obsidianchain.network.synthetic` applies unchanged.
These worlds are written by the same people as the analysis and therefore
contain the structure the analysis looks for. A good result here shows the
mechanism can work when the signal exists; it says nothing about Bitcoin.
Real validation still needs mainnet capture plus controlled ground truth.

One variable moves
------------------
sigma stays at 1.1506 - the Decker & Wattenhofer shape - in every regime.
So does the topology: the same origin nodes, the same observers, the same
clock offsets, and crucially **the same entity assignment**, all derived
from one base seed shared across regimes. The only thing that differs
between A and B is the probability with which an entity's transactions
choose each origin. Varying propagation noise and origin structure together
would make any difference between regimes uninterpretable.

The five regimes
----------------
``A INDEPENDENT``
    Origin drawn uniformly, ignoring entity. Replicates the frozen null
    under this new code path, which is the point: if A shows signal, the
    new generator has introduced it and every other regime is suspect.

``B AFFINITY``
    Each entity gets a sparse Dirichlet distribution over all origins -
    peaked, but with genuine overlap between entities. Deliberately *not*
    one entity to one origin: that would be trivially separable and would
    not survive a hostile question.

``C SHARED INFRASTRUCTURE``
    Entities are paired and each pair shares a distribution. The two members
    are indistinguishable by construction, so the correct behaviour is
    abstention, not separation.

``D MULTI-ORIGIN ENTITY``
    Each entity broadcasts from three origins. Its own transactions form
    three arrival clusters, so a naive method separates an entity from
    itself. This is the false-split trap and the most important regime.

``E NOISY AFFINITY``
    Ninety per cent from the entity's preferred distribution, ten per cent
    from an unrelated origin. Partial signal, imperfect attribution.

Ground truth
------------
``true_entity_id`` and ``true_origin_id`` generate the observations and are
written to ``processed/worlds_truth/<regime>/``, which
:mod:`obsidianchain.network.boundary` refuses to load. Inference sees the
same six columns it sees today and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.network import synthetic

WORLD_GENERATOR_VERSION = "1.0.0"


class Regime(str, Enum):
    A_INDEPENDENT = "A"
    B_AFFINITY = "B"
    C_SHARED_INFRASTRUCTURE = "C"
    D_MULTI_ORIGIN_ENTITY = "D"
    E_NOISY_AFFINITY = "E"


REGIME_NAMES = {
    Regime.A_INDEPENDENT: "INDEPENDENT",
    Regime.B_AFFINITY: "AFFINITY",
    Regime.C_SHARED_INFRASTRUCTURE: "SHARED INFRASTRUCTURE",
    Regime.D_MULTI_ORIGIN_ENTITY: "MULTI-ORIGIN ENTITY",
    Regime.E_NOISY_AFFINITY: "NOISY AFFINITY",
}

REGIME_EXPECTATIONS = {
    Regime.A_INDEPENDENT: "no separation signal (replicates the frozen null)",
    Regime.B_AFFINITY: "repeated transactions carry recoverable signal",
    Regime.C_SHARED_INFRASTRUCTURE: "ambiguous; system should abstain",
    Regime.D_MULTI_ORIGIN_ENTITY: "entity must NOT be split (false-split trap)",
    Regime.E_NOISY_AFFINITY: "partial signal, imperfect attribution",
}


@dataclass(frozen=True)
class WorldConfig:
    """Controlled-world parameters. sigma is fixed and not exposed here."""

    n_entities: int = 40
    n_origins: int = 64
    n_observers: int = 8
    broadcaster_fraction: float = 0.15
    missing_observation_rate: float = 0.02

    #: Dirichlet concentration for regimes B and E. Below 1 gives a sparse,
    #: peaked distribution; entities still share support, which is what
    #: produces real overlap rather than a one-to-one mapping.
    dirichlet_alpha: float = 0.30

    #: Regime C: probability mass a paired entity puts on its shared origin.
    shared_origin_mass: float = 0.90

    #: Regime D: origins per entity.
    origins_per_entity: int = 3

    #: Regime E: share drawn from the entity's preferred distribution.
    affinity_share: float = 0.90

    #: Probability that an eligible component is split between two hidden
    #: entities. This is what makes constraint precision measurable: a
    #: component holding two entities means some co-spend edges genuinely
    #: connect different owners, so a block can be correct rather than
    #: necessarily a false split.
    split_probability: float = 0.35

    #: Components smaller than this are never split. Splitting a two-address
    #: component leaves one address per entity, which cannot pool evidence
    #: and only adds noise to the ground truth.
    min_split_size: int = 4

    #: Share of a split component's addresses handed to the second entity,
    #: taken in address-code order so the partition is deterministic.
    split_fraction: float = 0.5

    #: Base seed for everything held constant across regimes: node topology,
    #: observers, clock offsets, and the entity assignment itself.
    base_seed: int = 1000

    def describe(self) -> dict:
        out = asdict(self)
        out["sigma"] = synthetic.DECKER_WATTENHOFER_SIGMA
        out["sigma_source"] = "Decker & Wattenhofer 2013; fixed across regimes"
        return out


#: Regime seeds. Independent per regime so one can be regenerated without
#: disturbing the others.
REGIME_SEEDS = {
    Regime.A_INDEPENDENT: 2001,
    Regime.B_AFFINITY: 2002,
    Regime.C_SHARED_INFRASTRUCTURE: 2003,
    Regime.D_MULTI_ORIGIN_ENTITY: 2004,
    Regime.E_NOISY_AFFINITY: 2005,
}


@dataclass
class WorldData:
    regime: Regime
    observations: pd.DataFrame
    ground_truth: pd.DataFrame
    nodes: pd.DataFrame
    observers: pd.DataFrame
    origin_distribution: np.ndarray
    entity_map: pd.DataFrame

    entity_of_address: np.ndarray
    """Hidden entity of every address code.

    Entities are sub-component, so a component root no longer identifies an
    entity and this array is the only correct address-to-entity mapping. It
    is generated here and written to the quarantined truth directory; the
    inference stage never sees it.
    """

    config: WorldConfig = field(default_factory=WorldConfig)
    manifest: dict = field(default_factory=dict)


# ---- entity assignment, identical in every regime ---------------------


def assign_entities(
    graph, config: WorldConfig
) -> tuple[np.ndarray, pd.DataFrame]:
    """Map every address code to a hidden entity, at SUB-COMPONENT granularity.

    Why not one entity per component
    --------------------------------
    The first version of this function gave each baseline co-spend component
    exactly one entity. That seemed natural - a wallet is a set of addresses
    that co-spend together - but it made the central measurement impossible:
    a co-spend edge never leaves its component, so *every* edge was
    within-entity, the engine was never asked to separate two genuine
    entities, and any block it emitted was a false split by construction.
    Measured on the frozen graph: zero of 274,313 edges crossed an entity.
    Constraint precision could not be evaluated at all.

    So a share of components now hold two entities. That is also the more
    honest model: co-spend is a heuristic, and a component containing two
    owners is exactly the false merge this project exists to prevent. A
    proposed merge can now genuinely be either:

        same entity      -> a legitimate merge the engine must not block
        different entity -> a false merge the engine should block

    Splitting is deterministic given ``base_seed``, and the seed is shared
    across regimes, so the same address belongs to the same entity in all
    five worlds. Only the entity-to-origin distribution differs between them.

    Small components are left whole: splitting a two-address component gives
    each entity a single address, which can never pool enough evidence to be
    judged and only adds noise to the ground truth.
    """
    from obsidianchain.cluster.pipeline import run_clustering

    run = run_clustering(graph, "entity-assignment")
    roots = run.roots
    unique_roots, sizes = np.unique(roots, return_counts=True)
    n_components = unique_roots.size

    rng = np.random.default_rng([config.base_seed, 11])
    primary = rng.integers(0, config.n_entities, size=n_components)

    # A distinct second entity for split components: offsetting by 1..n-1
    # modulo n guarantees it differs from the primary without a rejection
    # loop, so the draw count stays fixed and the assignment reproducible.
    offset = 1 + rng.integers(0, config.n_entities - 1, size=n_components)
    secondary = (primary + offset) % config.n_entities

    eligible = sizes >= config.min_split_size
    is_split = eligible & (rng.random(n_components) < config.split_probability)

    # Within a split component, the later half in address-code order goes to
    # the second entity. Deterministic, and independent of edge ordering.
    root_index = np.searchsorted(unique_roots, roots)
    codes = np.arange(roots.size)
    order = np.lexsort((codes, root_index))
    sorted_root = root_index[order]

    starts_mask = np.empty(sorted_root.size, dtype=bool)
    starts_mask[0] = True
    np.not_equal(sorted_root[1:], sorted_root[:-1], out=starts_mask[1:])
    group_start = np.flatnonzero(starts_mask)
    group_id = np.cumsum(starts_mask) - 1
    rank_in_component = np.arange(sorted_root.size) - group_start[group_id]
    component_size = sizes[sorted_root]

    take_secondary_sorted = is_split[sorted_root] & (
        rank_in_component >= component_size * config.split_fraction
    )
    take_secondary = np.empty(roots.size, dtype=bool)
    take_secondary[order] = take_secondary_sorted

    entity_of_address = np.where(
        take_secondary, secondary[root_index], primary[root_index]
    ).astype(np.int32)

    entity_map = pd.DataFrame(
        {
            "component_root": unique_roots,
            "component_size": sizes,
            "primary_entity": primary,
            "secondary_entity": np.where(is_split, secondary, -1),
            "is_split": is_split,
        }
    )
    return entity_of_address, entity_map


# ---- the five origin distributions ------------------------------------


def origin_distribution(regime: Regime, config: WorldConfig, seed: int) -> np.ndarray:
    """(n_entities, n_origins) row-stochastic matrix for one regime.

    This function is the entire difference between the regimes. Everything
    else - topology, delays, observers, entity assignment - is shared.
    """
    rng = np.random.default_rng([seed, 21])
    n_e, n_o = config.n_entities, config.n_origins

    if regime is Regime.A_INDEPENDENT:
        # Every entity has the identical uniform distribution, so knowing the
        # entity tells you nothing about the origin. This is the frozen null.
        return np.full((n_e, n_o), 1.0 / n_o)

    if regime is Regime.B_AFFINITY:
        # Sparse Dirichlet: peaked but overlapping. Not one entity per
        # origin - entities share support, so some pairs are genuinely hard.
        return rng.dirichlet(np.full(n_o, config.dirichlet_alpha), size=n_e)

    if regime is Regime.C_SHARED_INFRASTRUCTURE:
        # Pairs of entities share one dominant origin and therefore share a
        # distribution outright. Members of a pair are indistinguishable.
        matrix = np.full((n_e, n_o), (1.0 - config.shared_origin_mass) / n_o)
        n_pairs = (n_e + 1) // 2
        shared = rng.choice(n_o, size=n_pairs, replace=False)
        for entity in range(n_e):
            matrix[entity, shared[entity // 2]] += config.shared_origin_mass
        return matrix / matrix.sum(axis=1, keepdims=True)

    if regime is Regime.D_MULTI_ORIGIN_ENTITY:
        # Three origins per entity, equally weighted. An entity's own
        # transactions therefore form three arrival clusters.
        matrix = np.zeros((n_e, n_o))
        for entity in range(n_e):
            picks = rng.choice(n_o, size=config.origins_per_entity, replace=False)
            matrix[entity, picks] = 1.0 / config.origins_per_entity
        return matrix

    if regime is Regime.E_NOISY_AFFINITY:
        # Regime B's affinity, contaminated with unrelated origins.
        affinity = rng.dirichlet(np.full(n_o, config.dirichlet_alpha), size=n_e)
        uniform = np.full((n_e, n_o), 1.0 / n_o)
        return config.affinity_share * affinity + (1 - config.affinity_share) * uniform

    raise ValueError(f"unknown regime {regime!r}")


# ---- generation -------------------------------------------------------


def _shared_topology(config: WorldConfig) -> tuple[pd.DataFrame, pd.DataFrame, synthetic.NetworkConfig]:
    """Nodes and observers derived from ``base_seed``, identical per regime."""
    base = synthetic.NetworkConfig(
        n_observers=config.n_observers,
        n_origins=config.n_origins,
        broadcaster_fraction=config.broadcaster_fraction,
        missing_observation_rate=config.missing_observation_rate,
        seed=config.base_seed,
        propagation=synthetic.PropagationModel(),  # sigma fixed, never varied
    )
    nodes = synthetic.build_nodes(base)
    observers = synthetic.build_observers(base, nodes)
    return nodes, observers, base


def transaction_entities(
    graph, entity_of_address: np.ndarray, data_root: Path | None = None
) -> pd.DataFrame:
    """Attribute each transaction to one entity, via its representative input.

    The lowest-coded input represents the transaction, the same convention
    :mod:`obsidianchain.network.separation` uses for pooling. Since co-spend
    unions every input of a transaction, all its inputs share a component and
    therefore an entity, so the choice of representative does not change the
    answer - it only makes the attribution single-valued.
    """
    from obsidianchain.io import elliptic

    edges = elliptic.load_input_edges(data_root)
    codes = pd.Index(graph.addresses).get_indexer(edges["input_address"])
    edges = edges.assign(_code=codes)
    edges = edges[edges["_code"] >= 0]
    edges = edges.sort_values(["txId", "_code"], kind="stable").drop_duplicates(
        subset="txId", keep="first"
    )
    return pd.DataFrame(
        {
            "txid": edges["txId"].to_numpy(),
            "code": edges["_code"].to_numpy(),
            "true_entity_id": entity_of_address[edges["_code"].to_numpy()],
        }
    )


def generate_world(
    regime: Regime,
    graph,
    config: WorldConfig | None = None,
    data_root: Path | None = None,
    limit: int = 0,
    shared: tuple | None = None,
) -> WorldData:
    """Generate one regime's observations plus its hidden ground truth.

    ``shared`` lets a caller pass pre-computed (nodes, observers, base
    config, entity assignment, entity map) so all five regimes provably use
    the same topology and entity assignment rather than merely the same
    seed.
    """
    config = config or WorldConfig()
    seed = REGIME_SEEDS[regime]

    if shared is None:
        nodes, observers, base = _shared_topology(config)
        entity_of_address, entity_map = assign_entities(graph, config)
    else:
        nodes, observers, base, entity_of_address, entity_map = shared

    attribution = transaction_entities(graph, entity_of_address, data_root)
    if limit > 0:
        attribution = attribution.iloc[:limit]
    txids = attribution["txid"].to_numpy()
    entities = attribution["true_entity_id"].to_numpy()

    distribution = origin_distribution(regime, config, seed)

    # Draw one origin per transaction from its entity's distribution. Done
    # per entity rather than per transaction so the number of RNG calls does
    # not depend on transaction count within an entity.
    rng = np.random.default_rng([seed, 31])
    origin_idx = np.empty(txids.size, dtype=np.int64)
    for entity in np.unique(entities):
        rows = np.flatnonzero(entities == entity)
        origin_idx[rows] = rng.choice(
            config.n_origins, size=rows.size, p=distribution[entity]
        )

    observations, nodes, observers, base_truth = synthetic.generate(
        txids, base, nodes=nodes, observers=observers, origin_idx=origin_idx
    )

    ground_truth = pd.DataFrame(
        {
            "txid": txids,
            "true_entity_id": entities.astype(np.int32),
            "true_origin_id": nodes["node_id"].to_numpy()[origin_idx],
            "broadcaster_flag": nodes["is_known_broadcaster"].to_numpy()[origin_idx],
        }
    )

    return WorldData(
        regime=regime,
        observations=observations,
        ground_truth=ground_truth,
        nodes=nodes,
        observers=observers,
        origin_distribution=distribution,
        entity_map=entity_map,
        entity_of_address=entity_of_address,
        config=config,
    )


# ---- writing, in a namespace the pipeline keeps separate ---------------


def world_dirs(processed_root, regime: Regime) -> tuple[Path, Path]:
    processed_root = Path(processed_root)
    return (
        processed_root / synthetic.WORLDS_DIR / regime.value,
        processed_root / synthetic.WORLDS_TRUTH_DIR / regime.value,
    )


def write_world(
    world: WorldData, processed_root, fmt: str = "parquet"
) -> dict[str, object]:
    """Write one regime, observations and truth into separate directories."""
    obs_dir, truth_dir = world_dirs(processed_root, world.regime)
    obs_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        obs_path = obs_dir / "observations.parquet"
        world.observations.to_parquet(obs_path, index=False)
    elif fmt == "csv":
        obs_path = obs_dir / "observations.csv"
        world.observations.to_csv(obs_path, index=False)
    else:
        raise ValueError(f"unsupported format {fmt!r}; use 'parquet' or 'csv'")

    # --- observable side: the same six columns, plus what a deployment has
    world.observers.loc[:, ["observer_id", "asn", "region"]].to_csv(
        obs_dir / "observers.csv", index=False
    )
    world.nodes.loc[world.nodes["is_known_broadcaster"], ["ip"]].to_csv(
        obs_dir / "broadcaster_ips.csv", index=False
    )

    # --- ground truth: entity and origin, quarantined
    truth_path = truth_dir / "ground_truth.csv"
    world.ground_truth.to_csv(truth_path, index=False)
    world.nodes.to_csv(truth_dir / "origin_nodes.csv", index=False)
    world.entity_map.to_csv(truth_dir / "entity_map.csv", index=False)
    # Address-level truth. entity_map is keyed by component root and cannot
    # answer "which entity owns this address" now that a component may hold
    # two: collapsing a split component to its primary entity would report
    # zero cross-entity candidates and silently make precision unmeasurable.
    pd.DataFrame(
        {
            "address_code": np.arange(world.entity_of_address.size, dtype=np.int32),
            "true_entity_id": world.entity_of_address.astype(np.int32),
        }
    ).to_csv(truth_dir / "entity_assignment.csv", index=False)
    world.observers.loc[:, ["observer_id", "clock_bias_ms"]].to_csv(
        truth_dir / "observer_clocks.csv", index=False
    )
    np.savetxt(
        truth_dir / "origin_distribution.csv",
        world.origin_distribution,
        delimiter=",",
        fmt="%.10f",
    )
    (truth_dir / "README.txt").write_text(
        f"GROUND TRUTH - REGIME {world.regime.value} "
        f"({REGIME_NAMES[world.regime]}) - EVALUATION ONLY\n"
        "=" * 70 + "\n"
        "true_entity_id and true_origin_id generated these observations.\n"
        "Nothing in this directory may be read by an inference stage;\n"
        "network.boundary refuses to load it. A real capture supplies none\n"
        "of it. The evaluation harness may use it AFTER inference has run.\n",
        encoding="utf-8",
    )

    manifest = build_world_manifest(world, obs_path)
    manifest_path = obs_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    world.manifest = manifest

    return {
        "regime": world.regime.value,
        "observations": obs_path,
        "ground_truth": truth_path,
        "manifest": manifest_path,
        "rows": int(len(world.observations)),
        "bytes": int(obs_path.stat().st_size),
        "manifest_data": manifest,
    }


def build_world_manifest(world: WorldData, observations_path) -> dict:
    """Per-regime manifest: seed, parameters and dataset hash.

    Free of machine-specific detail, as in the production manifest, so two
    people generating regime B on different machines get identical hashes -
    and a Phase 3.3 comparison across regimes can prove it used these exact
    inputs.
    """
    observations_path = Path(observations_path)
    digest = hashlib.sha256()
    with observations_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    distribution = world.origin_distribution
    return {
        "world_generator_version": WORLD_GENERATOR_VERSION,
        "regime": world.regime.value,
        "regime_name": REGIME_NAMES[world.regime],
        "expectation": REGIME_EXPECTATIONS[world.regime],
        "regime_seed": REGIME_SEEDS[world.regime],
        "base_seed": world.config.base_seed,
        "dataset_sha256": digest.hexdigest(),
        "observations_filename": observations_path.name,
        "record_count": int(len(world.observations)),
        "transaction_count": int(world.observations["txid"].nunique())
        if len(world.observations)
        else 0,
        "observer_count": int(world.observations["observer_id"].nunique())
        if len(world.observations)
        else 0,
        "record_columns": list(synthetic.RECORD_COLUMNS),
        "entity_count": int(world.ground_truth["true_entity_id"].nunique()),
        "sigma": synthetic.DECKER_WATTENHOFER_SIGMA,
        "origin_distribution_shape": list(distribution.shape),
        "configuration": world.config.describe(),
    }


def generate_all(
    graph,
    config: WorldConfig | None = None,
    data_root: Path | None = None,
    limit: int = 0,
    regimes: tuple[Regime, ...] | None = None,
) -> dict[Regime, WorldData]:
    """Generate every regime from one shared topology and entity assignment."""
    config = config or WorldConfig()
    nodes, observers, base = _shared_topology(config)
    entity_of_address, entity_map = assign_entities(graph, config)
    shared = (nodes, observers, base, entity_of_address, entity_map)

    out: dict[Regime, WorldData] = {}
    for regime in (regimes or tuple(Regime)):
        out[regime] = generate_world(
            regime, graph, config, data_root, limit=limit, shared=shared
        )
    return out
