"""Reach-stress fixture: what the production rule does when evidence exists.

Two different questions, two different fixtures
----------------------------------------------
The controlled worlds in :mod:`obsidianchain.network.worlds` sit on the real
Elliptic++ co-spend graph and therefore inherit its evidence availability.
That is the right fixture for "how often can the rule act on realistic
data", and the answer measured in Phase 3.3 was: almost never - 47 of
253,429 unions reached the pooled minimum.

That leaves the important question unanswered. Regime D produced zero false
splits, but its within-entity sub-centroid separation was 6.48 standard
errors, so the evidence to split was present and the rule simply never got
near it. A pass earned by running out of reach is not evidence that a rule
is safe.

This fixture answers the other question: **given enough evidence, what does
the unchanged rule do?** It builds a small synthetic chain in which every
component accumulates far more than twenty-five pooled observations, so the
rule is actually forced to decide.

The threshold is NOT lowered. Twenty-five stays exactly as it is in
production; the fixture raises the evidence to meet it rather than dropping
the bar to meet the evidence. Lowering the threshold would test a different
rule, and the whole point is to test this one.

What is deliberately kept identical
-----------------------------------
sigma at 1.1506, the same observer count, the same clock jitter and bias,
the same missing-observation rate, the same broadcaster fraction, the same
five regimes A-E, and the same entity-splitting rule that lets a component
hold two hidden entities. Only the chain underneath changes, and it changes
in exactly one respect: components are dense enough to pool evidence.

One documented divergence, in regime D
--------------------------------------
In the normal worlds an origin is drawn per transaction, so every address of
a multi-origin entity ends up with the same three-origin mixture and any two
subgroups have the same expected centroid. The false-split trap therefore
cannot fire, whatever the rule does - D is untestable there, which is a
second instance of the flaw that made precision untestable.

Here, regime D binds each ADDRESS to one of its entity's three origins.
Subgroups of one entity then genuinely differ, which is what the trap
diagram describes: one entity, one component, three origins, and a rule that
must not cut it into three. This changes how D's origins are assigned in
this fixture only; regimes A, B, C and E draw per transaction exactly as
before, and the normal worlds are untouched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.network import synthetic, worlds

REACH_STRESS_VERSION = "1.0.0"


@dataclass(frozen=True)
class ReachStressConfig:
    """A chain dense enough that the production threshold is reachable."""

    n_components: int = 300
    addresses_per_component: int = 20
    transactions_per_component: int = 200
    inputs_per_transaction: int = 3

    n_entities: int = 40
    split_probability: float = 0.35
    split_fraction: float = 0.5

    #: Network parameters, held at the frozen production values.
    n_observers: int = 8
    n_origins: int = 64
    broadcaster_fraction: float = 0.15
    missing_observation_rate: float = 0.02

    base_seed: int = 3000

    @property
    def n_addresses(self) -> int:
        return self.n_components * self.addresses_per_component

    @property
    def n_transactions(self) -> int:
        return self.n_components * self.transactions_per_component

    @property
    def expected_pooled_per_component(self) -> int:
        """Transactions pooling into one component, before the usable filter."""
        return self.transactions_per_component

    def describe(self) -> dict:
        out = asdict(self)
        out["sigma"] = synthetic.DECKER_WATTENHOFER_SIGMA
        out["production_min_pooled"] = 25
        out["note"] = "threshold NOT lowered; evidence raised to meet it"
        return out


def build_chain(config: ReachStressConfig) -> dict[str, pd.DataFrame]:
    """A synthetic co-spend chain with dense, poolable components.

    Every transaction spends several addresses of one component, so the
    component assembles into a single co-spend cluster and accumulates one
    pooled observation per transaction.
    """
    rng = np.random.default_rng([config.base_seed, 1])

    addresses = np.array(
        [f"rsaddr{i:07d}" for i in range(config.n_addresses)], dtype=object
    )
    rows_addr: list[str] = []
    rows_tx: list[int] = []

    txid = 1
    for component in range(config.n_components):
        base = component * config.addresses_per_component
        members = np.arange(base, base + config.addresses_per_component)
        for _ in range(config.transactions_per_component):
            k = min(config.inputs_per_transaction, members.size)
            picked = rng.choice(members, size=k, replace=False)
            rows_addr.extend(addresses[picked])
            rows_tx.extend([txid] * k)
            txid += 1

    addr_tx = pd.DataFrame({"input_address": rows_addr, "txId": rows_tx})
    wallets = pd.DataFrame({"address": addresses, "class": 3})
    return {"addr_tx": addr_tx, "wallets_classes": wallets}


def write_chain(chain: dict[str, pd.DataFrame], data_root: Path) -> Path:
    """Write the fixture's raw files so the ordinary loaders can read them."""
    raw = Path(data_root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    chain["addr_tx"].to_csv(raw / "AddrTx_edgelist.csv", index=False)
    chain["wallets_classes"].to_csv(raw / "wallets_classes.csv", index=False)
    return raw


def assign_entities(graph, config: ReachStressConfig) -> tuple[np.ndarray, pd.DataFrame]:
    """Sub-component entity assignment, same rule as the normal worlds."""
    world_config = worlds.WorldConfig(
        n_entities=config.n_entities,
        split_probability=config.split_probability,
        split_fraction=config.split_fraction,
        min_split_size=4,
        base_seed=config.base_seed,
    )
    return worlds.assign_entities(graph, world_config)


def address_origins(
    regime: worlds.Regime,
    entity_of_address: np.ndarray,
    distribution: np.ndarray,
    config: ReachStressConfig,
    seed: int,
) -> np.ndarray | None:
    """Regime D only: bind each address to one of its entity's origins.

    Returns None for every other regime, which keeps their per-transaction
    draw exactly as in the normal worlds.
    """
    if regime is not worlds.Regime.D_MULTI_ORIGIN_ENTITY:
        return None
    rng = np.random.default_rng([seed, 41])
    sticky = np.empty(entity_of_address.size, dtype=np.int64)
    for entity in np.unique(entity_of_address):
        rows = np.flatnonzero(entity_of_address == entity)
        support = np.flatnonzero(distribution[entity] > 0)
        sticky[rows] = rng.choice(support, size=rows.size)
    return sticky


def generate_regime(
    regime: worlds.Regime,
    graph,
    entity_of_address: np.ndarray,
    entity_map: pd.DataFrame,
    config: ReachStressConfig,
    data_root: Path,
) -> dict:
    """Generate one regime's observations over the dense chain."""
    seed = worlds.REGIME_SEEDS[regime] + 500  # distinct from the normal worlds
    world_config = worlds.WorldConfig(
        n_entities=config.n_entities,
        n_origins=config.n_origins,
        n_observers=config.n_observers,
        broadcaster_fraction=config.broadcaster_fraction,
        missing_observation_rate=config.missing_observation_rate,
        base_seed=config.base_seed,
    )
    base = synthetic.NetworkConfig(
        n_observers=config.n_observers,
        n_origins=config.n_origins,
        broadcaster_fraction=config.broadcaster_fraction,
        missing_observation_rate=config.missing_observation_rate,
        seed=config.base_seed,
        propagation=synthetic.PropagationModel(),  # sigma fixed
    )
    nodes = synthetic.build_nodes(base)
    observers = synthetic.build_observers(base, nodes)

    distribution = worlds.origin_distribution(regime, world_config, seed)
    attribution = worlds.transaction_entities(graph, entity_of_address, data_root)
    txids = attribution["txid"].to_numpy()
    codes = attribution["code"].to_numpy()
    entities = attribution["true_entity_id"].to_numpy()

    sticky = address_origins(
        regime, entity_of_address, distribution, config, seed
    )
    rng = np.random.default_rng([seed, 31])
    if sticky is not None:
        # Regime D: the transaction's origin is its representative address's
        # origin, so subgroups of one entity genuinely differ.
        origin_idx = sticky[codes]
    else:
        origin_idx = np.empty(txids.size, dtype=np.int64)
        for entity in np.unique(entities):
            rows = np.flatnonzero(entities == entity)
            origin_idx[rows] = rng.choice(
                config.n_origins, size=rows.size, p=distribution[entity]
            )

    observations, nodes, observers, _ = synthetic.generate(
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
    return {
        "regime": regime,
        "observations": observations,
        "ground_truth": ground_truth,
        "nodes": nodes,
        "observers": observers,
        "distribution": distribution,
        "entity_map": entity_map,
        "entity_of_address": entity_of_address,
        "addresses": graph.addresses,
        "sticky_origins": sticky is not None,
        "seed": seed,
    }


def write_regime(payload: dict, data_root: Path, config: ReachStressConfig) -> dict:
    """Write one regime into the fixture's own processed namespace.

    Uses the same directory names the boundary already understands, under a
    separate data root, so nothing in the loader or the engine needs to know
    this fixture exists.
    """
    regime = payload["regime"]
    processed = Path(data_root) / "processed"
    obs_dir = processed / synthetic.WORLDS_DIR / regime.value
    truth_dir = processed / synthetic.WORLDS_TRUTH_DIR / regime.value
    obs_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    obs_path = obs_dir / "observations.parquet"
    payload["observations"].to_parquet(obs_path, index=False)
    payload["observers"].loc[:, ["observer_id", "asn", "region"]].to_csv(
        obs_dir / "observers.csv", index=False
    )
    nodes = payload["nodes"]
    nodes.loc[nodes["is_known_broadcaster"], ["ip"]].to_csv(
        obs_dir / "broadcaster_ips.csv", index=False
    )

    payload["ground_truth"].to_csv(truth_dir / "ground_truth.csv", index=False)
    pd.DataFrame(
        {
            "address": payload["addresses"],
            "address_code": np.arange(len(payload["addresses"])),
            "true_entity_id": payload["entity_of_address"],
        }
    ).to_csv(truth_dir / "entity_assignment.csv", index=False)
    payload["entity_map"].to_csv(truth_dir / "entity_map.csv", index=False)
    nodes.to_csv(truth_dir / "origin_nodes.csv", index=False)
    np.savetxt(
        truth_dir / "origin_distribution.csv",
        payload["distribution"], delimiter=",", fmt="%.10f",
    )
    (truth_dir / "README.txt").write_text(
        f"REACH-STRESS GROUND TRUTH - REGIME {regime.value} - EVALUATION ONLY\n"
        "Threshold NOT lowered; the chain is denser so 25 is reachable.\n",
        encoding="utf-8",
    )

    digest = hashlib.sha256()
    with obs_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    manifest = {
        "fixture": "reach-stress",
        "reach_stress_version": REACH_STRESS_VERSION,
        "regime": regime.value,
        "regime_name": worlds.REGIME_NAMES[regime],
        "regime_seed": payload["seed"],
        "base_seed": config.base_seed,
        "dataset_sha256": digest.hexdigest(),
        "record_count": int(len(payload["observations"])),
        "transaction_count": int(payload["observations"]["txid"].nunique()),
        "observer_count": int(payload["observations"]["observer_id"].nunique()),
        "record_columns": list(synthetic.RECORD_COLUMNS),
        "sigma": synthetic.DECKER_WATTENHOFER_SIGMA,
        "production_min_pooled": 25,
        "sticky_address_origins": bool(payload["sticky_origins"]),
        "configuration": config.describe(),
    }
    (obs_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {
        "regime": regime.value,
        "observations": obs_path,
        "rows": int(len(payload["observations"])),
        "bytes": int(obs_path.stat().st_size),
        "manifest": manifest,
    }


def build_fixture(
    data_root: Path, config: ReachStressConfig | None = None
) -> dict[str, dict]:
    """Build the whole reach-stress fixture: chain, then all five regimes."""
    from obsidianchain.io import elliptic

    config = config or ReachStressConfig()
    data_root = Path(data_root)

    write_chain(build_chain(config), data_root)
    graph = elliptic.load_cospend_graph(data_root, keep_labels=True)
    entity_of_address, entity_map = assign_entities(graph, config)

    written: dict[str, dict] = {}
    for regime in worlds.Regime:
        payload = generate_regime(
            regime, graph, entity_of_address, entity_map, config, data_root
        )
        written[regime.value] = write_regime(payload, data_root, config)
    written["_graph"] = {"graph": graph, "entity_of_address": entity_of_address}
    return written
