"""Generate the world: entities -> UTXOs -> transactions -> txids -> network.

Output layout
-------------
``<root>/raw/``            OBSERVABLE chain layer, in exactly the three
                           filenames the existing Phase 6 loader reads, so
                           the real pipeline runs over it unmodified:
                           ``AddrTx_edgelist.csv``, ``TxAddr_edgelist.csv``,
                           ``txs_features.csv``, plus ``wallets_classes.csv``
                           for the label, which the loader joins LAST and
                           drops - the same discipline as production.

``<root>/processed/network/``  OBSERVABLE network layer, written by the
                           existing frozen ``network.synthetic`` generator
                           over the txids produced above.

``<root>/world_truth/``    QUARANTINED. Entity identity, origin identity and
                           the behaviour that produced each transaction.
                           Nothing in the pipeline may read it;
                           ``tests/test_world.py`` asserts no observable file
                           carries any of its columns.

Why the chain layer reuses production filenames
-----------------------------------------------
"The existing pipeline runs end to end on this world" is a claim worth being
able to make literally rather than approximately. Emitting the three files
the loader already opens means nothing in ``features/`` or ``ml/`` needed a
synthetic code path, so what runs on the world is the same code that runs on
Elliptic++ - not a parallel implementation that could drift.

Determinism
-----------
One seed, threaded through a single ``numpy`` Generator. Entities, addresses,
values, timesteps and origins all derive from it, so two runs of the same
version with the same config produce byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.world import behaviours as bx

GENERATOR_VERSION = "1.0.0"


def _now() -> str:
    """Creation timestamp for the manifest.

    Recorded for the operator, and deliberately NOT part of the world
    fingerprint: a world regenerated from the same seed and config is the
    same world, and making its identity depend on the clock would make
    determinism unverifiable.
    """
    import datetime as dt

    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


WORLD_SCHEMA = "obsidianchain.world/1"

#: Where truth lives. Named so the source scanners quarantine it by the same
#: rule they already apply to ``network_truth`` and ``worlds_truth``.
TRUTH_DIR = "world_truth"
RAW_DIR = "raw"

#: Production's own split boundaries, so ``dataset.assign_split`` needs no
#: synthetic special case. Each entity is confined to ONE band, because an
#: entity straddling a boundary would have every address dropped as a
#: boundary spanner and the world would train on almost nothing.
SPLIT_BANDS = {
    "train": (1, 34),
    "validation": (35, 41),
    "test": (42, 49),
}

#: Elliptic's own encoding, reused so the label file means the same thing.
CLASS_POSITIVE = 1
CLASS_NEGATIVE = 2

POSITIVE_CLASS_MEANING = (
    "class 1 designates the behaviours this synthetic evaluation treats as "
    "the positive label. It is a labelling convention for a controlled "
    "experiment, not a claim that the behaviour is unlawful."
)


@dataclass(frozen=True)
class WorldConfig:
    """Everything that determines the world. Travels into provenance whole."""

    seed: int = 20260917

    #: Entities per behaviour, per split band. Small on purpose: the world
    #: must be inspectable by eye before it is trusted at any scale.
    entities_per_behaviour: int = 4

    #: Addresses an entity controls.
    addresses_per_entity: int = 6

    #: Network origins. An entity announces from one of these unless its
    #: behaviour gives it several.
    n_origins: int = 24

    #: Origins for a MULTI_HOP entity, which is the world's multi-origin
    #: case - the shape network world D exists to describe.
    origins_for_multi_origin: int = 3

    behaviour: bx.BehaviourConfig = field(
        default_factory=lambda: bx.DEFAULT_BEHAVIOUR_CONFIG
    )

    def digest(self) -> str:
        """A stable hash of the configuration, for the artifact identity."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class _Builder:
    """Mutable accumulation while the world is being written."""

    rng: np.random.Generator
    config: WorldConfig

    addresses: list = field(default_factory=list)      # address -> entity
    address_entity: dict = field(default_factory=dict)
    entity_rows: list = field(default_factory=list)
    tx_rows: list = field(default_factory=list)        # tx truth
    inputs: list = field(default_factory=list)         # (address, txid)
    outputs: list = field(default_factory=list)        # (txid, address)
    tx_values: list = field(default_factory=list)      # observable tx summary
    next_tx: int = 0

    def new_address(self, entity: str) -> str:
        """A deterministic, obviously-synthetic address string.

        Prefixed ``SYN`` so a value copied out of this world into a document
        is recognisable on sight as generated. Nothing here is a real
        Bitcoin address and the format does not pretend to be one.
        """
        index = len(self.addresses)
        address = f"SYN{index:08d}"
        self.addresses.append(address)
        self.address_entity[address] = entity
        return address

    def emit(self, entity: str, behaviour: str, timestep: int,
             ins: list, outs: list) -> str:
        """Record one transaction. ``ins``/``outs`` are (address, value)."""
        txid = f"SYNTX{self.next_tx:08d}"
        self.next_tx += 1

        in_values = np.array([v for _a, v in ins], dtype="float64")
        out_values = np.array([v for _a, v in outs], dtype="float64")
        # A fee makes the outputs sum to slightly less than the inputs, as a
        # real transaction does. Kept tiny so it does not perturb the output
        # uniformity a mixing round is defined by.
        fee = float(max(in_values.sum() - out_values.sum(), 0.0))

        for address, _value in ins:
            self.inputs.append({"input_address": address, "txId": txid})
        for address, _value in outs:
            self.outputs.append({"txId": txid, "output_address": address})

        self.tx_values.append({
            "txId": txid,
            "Time step": int(timestep),
            "total_BTC": float(out_values.sum()),
            "fees": fee,
            "size": int(180 + 70 * len(ins) + 35 * len(outs)),
            "num_input_addresses": len(ins),
            "num_output_addresses": len(outs),
            "in_BTC_total": float(in_values.sum()),
            "in_BTC_min": float(in_values.min()),
            "in_BTC_max": float(in_values.max()),
            "in_BTC_mean": float(in_values.mean()),
            "out_BTC_total": float(out_values.sum()),
            "out_BTC_min": float(out_values.min()),
            "out_BTC_max": float(out_values.max()),
            "out_BTC_mean": float(out_values.mean()),
        })
        # TRUTH. Never written beside the observable files.
        self.tx_rows.append({
            "txid": txid, "true_entity": entity, "scenario": behaviour,
            "timestep": int(timestep),
        })
        return txid

    def varied(self, n: int, low: float = 0.05, high: float = 4.0) -> np.ndarray:
        return np.round(self.rng.uniform(low, high, size=n), 6)


def _band(rng, band: str) -> tuple[int, int]:
    return SPLIT_BANDS[band]


def _generate_entity(builder: _Builder, entity: str, behaviour: str,
                     band: str) -> None:
    """Emit one entity's transactions in the shape its behaviour names."""
    cfg = builder.config.behaviour
    rng = builder.rng
    low, high = _band(rng, band)
    own = [builder.new_address(entity)
           for _ in range(builder.config.addresses_per_entity)]

    def t(offset: int = 0) -> int:
        """A timestep inside this entity's band. Clamped, never straddling."""
        return int(min(high, low + offset))

    if behaviour == bx.NORMAL:
        for k in range(3):
            step = t(rng.integers(0, max(1, high - low)))
            value = float(builder.varied(1, 0.4, 3.0)[0])
            payment = round(value * float(rng.uniform(0.2, 0.7)), 6)
            builder.emit(entity, behaviour, step,
                         [(own[0], value)],
                         [(own[1], payment), (own[2], round(value - payment - 0.0001, 6))])

    elif behaviour == bx.EXCHANGE_BATCH:
        for k in range(2):
            outs = [(builder.new_address(entity), float(v))
                    for v in builder.varied(cfg.batch_outputs, 0.01, 6.0)]
            total = sum(v for _a, v in outs)
            builder.emit(entity, behaviour, t(k),
                         [(own[0], round(total + 0.001, 6))], outs)

    elif behaviour == bx.MERCHANT_SWEEP:
        # Two funders paying many unrelated amounts: the fan-out of a batch
        # with a second input, which is the shape most likely to be mistaken
        # for a collaborative spend.
        outs = [(builder.new_address(entity), float(v))
                for v in builder.varied(cfg.batch_outputs, 0.02, 5.0)]
        total = sum(v for _a, v in outs)
        builder.emit(entity, behaviour, t(0),
                     [(own[0], round(total * 0.6, 6)),
                      (own[1], round(total * 0.4 + 0.001, 6))], outs)

    elif behaviour == bx.CONSOLIDATION:
        ins = [(builder.new_address(entity), float(v))
               for v in builder.varied(cfg.consolidation_inputs, 0.001, 0.9)]
        total = sum(v for _a, v in ins)
        builder.emit(entity, behaviour, t(0), ins,
                     [(own[0], round(total - 0.0005, 6))])

    elif behaviour == bx.PEELING:
        # Each hop peels a small amount off and forwards the remainder to a
        # NEW address at a strictly later timestep, which is what PEEL-1
        # walks. Two outputs per hop keeps every hop admissible.
        remaining = 20.0
        carrier = own[0]
        for hop in range(cfg.peel_depth):
            step = t(hop)
            peel = round(remaining * cfg.peel_fraction, 6)
            remainder = round(remaining - peel - 0.0001, 6)
            forward = builder.new_address(entity)
            builder.emit(entity, behaviour, step,
                         [(carrier, remaining)],
                         [(builder.new_address(entity), peel),
                          (forward, remainder)])
            carrier, remaining = forward, remainder

    elif behaviour == bx.MIXING_LIKE:
        n = cfg.mixing_participants
        for round_index in range(2):
            ins = [(builder.new_address(entity), float(v))
                   for v in builder.varied(n, 0.3, 3.0)]
            outs = [(builder.new_address(entity), cfg.mixing_denomination)
                    for _ in range(n)]
            builder.emit(entity, behaviour, t(round_index), ins, outs)

    elif behaviour == bx.UNIFORM_PAYOUT:
        # Equal outputs, ONE payer. Structurally similar to a mixing round
        # and benign; the detector must separate them on the input side.
        outs = [(builder.new_address(entity), cfg.payout_value)
                for _ in range(cfg.payout_outputs)]
        total = cfg.payout_value * cfg.payout_outputs
        builder.emit(entity, behaviour, t(0),
                     [(own[0], round(total + 0.001, 6))], outs)

    elif behaviour == bx.BENIGN_EQUAL_SPLIT:
        builder.emit(entity, behaviour, t(0),
                     [(own[0], 1.2), (own[1], 0.8)],
                     [(own[2], 0.999), (own[3], 0.999)])

    elif behaviour == bx.RAPID_MOVEMENT:
        carrier = own[0]
        value = 5.0
        for k in range(cfg.rapid_transfers):
            nxt = builder.new_address(entity)
            value = round(value * 0.93, 6)
            builder.emit(entity, behaviour, t(k % max(1, high - low)),
                         [(carrier, round(value / 0.93, 6))],
                         [(nxt, value), (own[1], 0.0005)])
            carrier = nxt

    elif behaviour == bx.MULTI_HOP:
        carrier = own[0]
        value = 3.0
        for hop in range(cfg.multi_hop_length):
            nxt = builder.new_address(entity)
            value = round(value - 0.001, 6)
            builder.emit(entity, behaviour, t(hop),
                         [(carrier, round(value + 0.001, 6))], [(nxt, value)])
            carrier = nxt

    else:  # pragma: no cover - BEHAVIOURS is the closed set
        raise ValueError(f"unknown behaviour {behaviour!r}")

    builder.entity_rows.append({
        "true_entity": entity, "scenario": behaviour, "band": band,
    })


def build_world(config: WorldConfig | None = None):
    """Generate the chain layer. Returns observable frames and truth frames.

    Truth is RETURNED SEPARATELY rather than as columns on the observable
    frames, so a caller cannot write it into the wrong directory by
    forgetting to drop a column - the same reason ``network.synthetic``
    returns its ground truth apart from its observations.
    """
    config = config or WorldConfig()
    builder = _Builder(rng=np.random.default_rng(config.seed), config=config)

    for band in SPLIT_BANDS:
        for behaviour in bx.BEHAVIOURS:
            for index in range(config.entities_per_behaviour):
                entity = f"E-{band[:2]}-{behaviour}-{index}"
                _generate_entity(builder, entity, behaviour, band)

    addr_tx = pd.DataFrame(builder.inputs)
    tx_addr = pd.DataFrame(builder.outputs)
    tx_features = pd.DataFrame(builder.tx_values)

    entities = pd.DataFrame(builder.entity_rows)
    tx_truth = pd.DataFrame(builder.tx_rows)
    address_truth = pd.DataFrame(
        [{"address": a, "true_entity": builder.address_entity[a]}
         for a in builder.addresses]
    )

    # Labels. Derived from the behaviour, which is truth - so the label file
    # is written beside raw/ exactly as Elliptic's is, and the pipeline joins
    # it LAST and drops it, never letting it reach a feature.
    positive = set(bx.POSITIVE_CLASS)
    scenario_of = dict(zip(entities["true_entity"], entities["scenario"]))
    labels = pd.DataFrame({
        "address": address_truth["address"],
        "class": [
            CLASS_POSITIVE if scenario_of[e] in positive else CLASS_NEGATIVE
            for e in address_truth["true_entity"]
        ],
    })

    return {
        "addr_tx": addr_tx,
        "tx_addr": tx_addr,
        "tx_features": tx_features,
        "labels": labels,
    }, {
        "entities": entities,
        "addresses": address_truth,
        "transactions": tx_truth,
    }


def origin_assignment(truth: dict, config: WorldConfig, rng) -> tuple:
    """Map each transaction to the network origin that announced it.

    An entity announces from one origin, except MULTI_HOP, which is given
    several - that is the multi-origin entity network world D describes, and
    it is the case a naive "same origin implies same entity" rule gets wrong.
    """
    entities = truth["entities"]
    n_origins = config.n_origins
    per_entity = {}
    for row in entities.to_dict("records"):
        if row["scenario"] == bx.MULTI_HOP:
            per_entity[row["true_entity"]] = rng.integers(
                0, n_origins, size=config.origins_for_multi_origin
            )
        else:
            per_entity[row["true_entity"]] = rng.integers(0, n_origins, size=1)

    transactions = truth["transactions"]
    origin_idx = np.array([
        int(rng.choice(per_entity[entity]))
        for entity in transactions["true_entity"]
    ], dtype=np.int64)
    return origin_idx, per_entity


def write_world(root: Path, config: WorldConfig | None = None) -> dict:
    """Generate and persist the whole world. Returns a summary for the CLI."""
    from obsidianchain import provenance as prov
    from obsidianchain.network import synthetic

    config = config or WorldConfig()
    root = Path(root)
    observable, truth = build_world(config)

    raw = root / RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    observable["addr_tx"].to_csv(raw / "AddrTx_edgelist.csv", index=False)
    observable["tx_addr"].to_csv(raw / "TxAddr_edgelist.csv", index=False)
    observable["tx_features"].to_csv(raw / "txs_features.csv", index=False)
    observable["labels"].to_csv(raw / "wallets_classes.csv", index=False)

    # ---- network layer, over the SAME txids ------------------------------
    # The frozen generator, called with this world's txids and origins.
    # Nothing about its propagation or noise model is changed; it is simply
    # told which transactions exist and who announced them.
    rng = np.random.default_rng(config.seed + 1)
    origin_idx, _per_entity = origin_assignment(truth, config, rng)
    net_config = synthetic.NetworkConfig(seed=config.seed)
    observations, nodes, observers, net_truth = synthetic.generate(
        truth["transactions"]["txid"].to_numpy(),
        config=net_config,
        origin_idx=origin_idx,
    )

    processed = root / "processed" / synthetic.OBSERVATIONS_DIR
    processed.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(processed / "observations.parquet", index=False)
    nodes.to_parquet(processed / "nodes.parquet", index=False)
    observers.to_parquet(processed / "observers.parquet", index=False)

    # ---- truth, quarantined ---------------------------------------------
    truth_dir = root / TRUTH_DIR
    truth_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in truth.items():
        frame.to_csv(truth_dir / f"{name}.csv", index=False)
    net_truth.to_csv(truth_dir / "network_origins.csv", index=False)
    (truth_dir / "README.txt").write_text(
        "FOR_EVALUATION_ONLY\n\n"
        "Entity identity, origin identity and the behaviour that produced "
        "each transaction. No inference stage may read this directory. It "
        "exists so a controlled experiment can be scored, and reading it "
        "from a feature builder would make every measurement meaningless.\n",
        encoding="utf-8",
    )

    # ---- provenance ------------------------------------------------------
    fingerprint = hashlib.sha256(
        (config.digest() + GENERATOR_VERSION).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": WORLD_SCHEMA,
        "provenance_type": prov.ProvenanceType.SYNTHETIC_CONTROL.value,
        "generator_version": GENERATOR_VERSION,
        "network_generator_version": synthetic.GENERATOR_VERSION,
        "seed": config.seed,
        "config": asdict(config),
        "config_sha256": config.digest(),
        "world_fingerprint": fingerprint,
        "created_at": _now(),
        "behaviours": list(bx.BEHAVIOURS),
        "adversarial_behaviours": list(bx.ADVERSARIAL),
        "positive_class": list(bx.POSITIVE_CLASS),
        "positive_class_meaning": POSITIVE_CLASS_MEANING,
        "behaviour_meaning": bx.MEANING,
        "counts": {
            "entities": int(len(truth["entities"])),
            "addresses": int(len(truth["addresses"])),
            "transactions": int(len(truth["transactions"])),
            "observations": int(len(observations)),
        },
        "notes": [
            "SYNTHETIC_CONTROL. Every number measured on this world is a "
            "property of a generator, not of Bitcoin.",
            "The chain layer and the network layer share txids because both "
            "were produced from one set of synthetic transactions.",
            "Truth lives in world_truth/ and is read by no inference stage.",
        ],
    }
    (root / "world_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return manifest
