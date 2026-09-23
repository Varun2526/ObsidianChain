"""World v2: a noisy, overlapping benchmark in the problem statement's own format.

Why a second world
------------------
World v1 (``generate.py``) was built to be inspected by eye: every behaviour
has exact shape parameters (peel fraction 0.08, mixing denomination 0.1) and
the detectors were written against those shapes. On it the full feature set
ranks perfectly (nAP 1.000 in every usable fold, exp12), so it cannot show
whether any feature group adds value. It also fits only 3 of the 12 protocol
folds. World v1 is kept unchanged; this module is additive.

What v2 changes
---------------
* **Overlap.** Shape parameters are drawn from ranges, and benign look-alikes
  share the range: a benign PAYMENT_CHAIN peels 5-60% per hop, PEELING peels
  2-30%, so some benign chains fire the peeling flag and some peeling chains
  do not. MIXING_LIKE outputs carry denomination noise, and a benign
  multi-payer UNIFORM_PAYOUT reaches the mixing shape.
* **Label noise.** A fraction of entities (default 2%) carry the wrong label,
  as real labels do.
* **Scale and spread.** Entities start anywhere in their split band, so every
  one of the 12 rolling folds has rows.
* **Real timestamps inside a step.** Transactions carry a within-step offset
  in seconds, so duration and gap features are not step counts.
* **One chain-ambiguous, network-resolvable behaviour.** RELAY_LAUNDERING
  emits exactly the NORMAL chain shape. What differs is how it broadcasts: it
  runs well-connected nodes, so observers tend to receive its announcements
  directly from the origin (one repeated peer, a tight arrival spread). Two
  benign service behaviours (EXCHANGE_BATCH, HOT_WALLET_SHUFFLE) are
  well-connected too, so connectivity alone does not decide either.

The NULL control
----------------
``network_signal=False`` gives RELAY_LAUNDERING the same connectivity
distribution as NORMAL. Its chain and network layers then carry no signal
about it at all. A network feature that "helps" in the NULL world is fitting
noise; the experiment compares both worlds.

What any number measured here means
-----------------------------------
It is a property of this generator. The generator was written by the same
team as the analysis, and the network signal was put there on purpose. The
world tests whether the pipeline can EXTRACT a network signal that exists; it
says nothing about whether real Bitcoin traffic carries one.

Output (``write_noisy_world``)
------------------------------
``<root>/capture.csv``              canonical PS capture, one row per network
                                    observation (chain-only rows for
                                    unobserved transactions)
``<root>/world_truth/labels.csv``   QUARANTINED address labels
``<root>/world_truth/entities.csv`` QUARANTINED entity -> behaviour
``<root>/world_manifest.json``      SYNTHETIC_CONTROL provenance
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GENERATOR_VERSION = "2.0.0"
WORLD_SCHEMA = "obsidianchain.world_noisy/1"
TRUTH_DIR = "world_truth"

BASE_TIMESTAMP = 1400000000
TIMESTEP_SECONDS = 1209600

SPLIT_BANDS = {"train": (1, 34), "validation": (35, 41), "test": (42, 49)}

NORMAL = "NORMAL"
EXCHANGE_BATCH = "EXCHANGE_BATCH"
CONSOLIDATION = "CONSOLIDATION"
PAYMENT_CHAIN = "PAYMENT_CHAIN"
UNIFORM_PAYOUT = "UNIFORM_PAYOUT"
HOT_WALLET_SHUFFLE = "HOT_WALLET_SHUFFLE"
MERCHANT_SWEEP = "MERCHANT_SWEEP"
PEELING = "PEELING"
MIXING_LIKE = "MIXING_LIKE"
RAPID_MOVEMENT = "RAPID_MOVEMENT"
RELAY_LAUNDERING = "RELAY_LAUNDERING"

#: Behaviour -> entities per split band, relative. NORMAL dominates, as
#: ordinary use dominates real traffic.
BEHAVIOUR_WEIGHTS = {
    NORMAL: 6, EXCHANGE_BATCH: 1, CONSOLIDATION: 1, PAYMENT_CHAIN: 2,
    UNIFORM_PAYOUT: 1, HOT_WALLET_SHUFFLE: 1, MERCHANT_SWEEP: 1,
    PEELING: 1, MIXING_LIKE: 1, RAPID_MOVEMENT: 1, RELAY_LAUNDERING: 1,
}

#: Labelling convention for this controlled experiment, not a claim that
#: the behaviour is unlawful.
POSITIVE_CLASS = (PEELING, MIXING_LIKE, RAPID_MOVEMENT, RELAY_LAUNDERING)

#: Behaviours whose operators run well-connected nodes.
WELL_CONNECTED = (EXCHANGE_BATCH, HOT_WALLET_SHUFFLE)


@dataclass(frozen=True)
class NoisyWorldConfig:
    seed: int = 20260923
    #: Entities per unit of BEHAVIOUR_WEIGHTS, per split band.
    entities_per_weight: int = 40
    label_noise: float = 0.02
    #: False builds the NULL control (no network signal for RELAY_LAUNDERING).
    network_signal: bool = True
    n_observers: int = 12
    n_relay_peers: int = 3000
    n_asns: int = 400
    #: Share of transactions no observer saw. Their network group is NaN.
    unobserved_share: float = 0.10

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass
class _W:
    rng: np.random.Generator
    cfg: NoisyWorldConfig
    txs: list = field(default_factory=list)
    address_label: dict = field(default_factory=dict)
    address_entity: dict = field(default_factory=dict)
    entities: list = field(default_factory=list)
    n_addr: int = 0
    n_tx: int = 0

    def addr(self, entity: str, label: int) -> str:
        a = f"SYN2{self.n_addr:09d}"
        self.n_addr += 1
        self.address_label[a] = label
        self.address_entity[a] = entity
        return a

    def ts(self, step: int) -> float:
        return BASE_TIMESTAMP + (step - 1) * TIMESTEP_SECONDS + float(self.rng.uniform(0, TIMESTEP_SECONDS - 1))

    def emit(self, entity: str, step: int, ins: list, outs: list, conn: float, origin: tuple) -> None:
        in_v = np.array([v for _a, v in ins], dtype=float)
        out_v = np.array([v for _a, v in outs], dtype=float)
        fee = float(max(in_v.sum() - out_v.sum(), 0.0))
        self.txs.append({
            "txid": f"SYN2TX{self.n_tx:09d}", "timestamp": self.ts(step), "entity": entity,
            "ins": ins, "outs": outs, "fee": round(fee, 8), "conn": conn, "origin": origin,
        })
        self.n_tx += 1


def _values(rng, n, low, high) -> list[float]:
    return [round(float(v), 8) for v in rng.uniform(low, high, size=n)]


def _entity(w: _W, entity: str, behaviour: str, band: str, flipped: bool) -> None:
    rng = w.rng
    lo, hi = SPLIT_BANDS[band]
    positive = behaviour in POSITIVE_CLASS
    label = int(positive) ^ int(flipped)
    start = int(rng.integers(lo, hi + 1))

    def step(k: int = 0) -> int:
        return int(min(hi, start + k))

    def own() -> str:
        return w.addr(entity, label)

    def external() -> str:
        # A third party paid by the entity: its label is its own (benign
        # background), which is why one transaction can carry mixed labels.
        return w.addr(f"EXT-{entity}", 0)

    if behaviour in WELL_CONNECTED:
        conn = float(rng.beta(4, 3))
    elif behaviour == RELAY_LAUNDERING and w.cfg.network_signal:
        conn = float(rng.beta(5, 2))
    else:
        conn = float(rng.beta(1, 6))
    origin = (f"198.18.{int(rng.integers(0, 256))}.{int(rng.integers(1, 255))}",
              int(64512 + rng.integers(0, w.cfg.n_asns)))

    def emit(k, ins, outs):
        w.emit(entity, step(k), ins, outs, conn, origin)

    if behaviour in (NORMAL, RELAY_LAUNDERING):
        # RELAY_LAUNDERING is chain-identical to NORMAL by construction.
        wallet = [own() for _ in range(3)]
        for k in range(int(rng.integers(2, 6))):
            n_in = int(rng.integers(1, 3))
            vals = _values(rng, n_in, 0.2, 3.0)
            total = sum(vals)
            pay = round(total * float(rng.uniform(0.15, 0.85)), 8)
            change = round(total - pay - float(rng.uniform(1e-5, 5e-4)), 8)
            emit(int(rng.integers(0, 3)), [(wallet[i % 3], v) for i, v in enumerate(vals)],
                 [(external(), pay), (wallet[(k + 1) % 3], max(change, 1e-6))])

    elif behaviour in (PAYMENT_CHAIN, PEELING):
        frac_lo, frac_hi = (0.05, 0.60) if behaviour == PAYMENT_CHAIN else (0.02, 0.30)
        depth = int(rng.integers(3, 11)) if behaviour == PEELING else int(rng.integers(2, 7))
        remaining = float(rng.uniform(2.0, 40.0))
        carrier = own()
        for k in range(depth):
            peel = round(remaining * float(rng.uniform(frac_lo, frac_hi)), 8)
            rest = round(remaining - peel - float(rng.uniform(1e-5, 5e-4)), 8)
            nxt = own()
            outs = [(external(), peel), (nxt, rest)]
            rng.shuffle(outs)
            emit(k, [(carrier, remaining)], outs)
            carrier, remaining = nxt, rest

    elif behaviour == MIXING_LIKE:
        for k in range(int(rng.integers(1, 4))):
            n = int(rng.integers(3, 11))
            denom = float(rng.choice([0.01, 0.05, 0.1, 0.5, 1.0]))
            sigma = float(rng.uniform(0.0, 0.04))
            ins = [(own(), v) for v in _values(rng, n, denom * 1.1, denom * 6)]
            outs = [(own(), round(denom * (1 + float(rng.normal(0, sigma))), 8)) for _ in range(n)]
            emit(k, ins, outs)

    elif behaviour == UNIFORM_PAYOUT:
        payers = int(rng.integers(1, 5))
        n = int(rng.integers(3, 12))
        value = round(float(rng.uniform(0.05, 1.0)), 8)
        per = value * n / payers
        ins = [(own(), round(per * float(rng.uniform(0.6, 1.6)), 8)) for _ in range(payers)]
        total_in = sum(v for _a, v in ins)
        outs = [(external(), value) for _ in range(n)]
        if total_in < value * n:
            ins.append((own(), round(value * n - total_in + 0.001, 8)))
        emit(0, ins, outs)

    elif behaviour == EXCHANGE_BATCH:
        hot = own()
        for k in range(int(rng.integers(1, 4))):
            outs = [(external(), v) for v in _values(rng, int(rng.integers(8, 31)), 0.005, 5.0)]
            emit(k, [(hot, round(sum(v for _a, v in outs) + 0.002, 8))], outs)

    elif behaviour == MERCHANT_SWEEP:
        outs = [(external(), v) for v in _values(rng, int(rng.integers(6, 16)), 0.02, 4.0)]
        total = sum(v for _a, v in outs)
        share = float(rng.uniform(0.3, 0.7))
        emit(0, [(own(), round(total * share, 8)), (own(), round(total * (1 - share) + 0.001, 8))], outs)

    elif behaviour == CONSOLIDATION:
        ins = [(own(), v) for v in _values(rng, int(rng.integers(4, 15)), 0.001, 0.9)]
        emit(0, ins, [(own(), round(sum(v for _a, v in ins) - 0.0005, 8))])

    elif behaviour in (RAPID_MOVEMENT, HOT_WALLET_SHUFFLE):
        # Both move value in many quick hops. RAPID_MOVEMENT peels a dust
        # output each hop; the benign shuffle keeps whole amounts but is run
        # from a well-connected service node.
        carrier = own()
        value = float(rng.uniform(1.0, 20.0))
        for k in range(int(rng.integers(4, 12))):
            nxt = own()
            if behaviour == RAPID_MOVEMENT:
                keep = round(value * float(rng.uniform(0.85, 0.97)), 8)
                emit(k // 3, [(carrier, value)], [(nxt, keep), (own(), round(value - keep - 1e-4, 8))])
            else:
                keep = round(value - float(rng.uniform(1e-5, 3e-4)), 8)
                emit(k // 3, [(carrier, value)], [(nxt, keep)])
            carrier, value = nxt, keep

    w.entities.append({"entity": entity, "behaviour": behaviour, "band": band,
                       "label": label, "label_flipped": flipped, "connectivity": round(conn, 4)})


def _observations(w: _W) -> pd.DataFrame:
    """One capture row per observation of each transaction."""
    rng, cfg = w.rng, w.cfg
    observers = [f"203.0.113.{i + 1}" for i in range(cfg.n_observers)]
    peers = [f"100.{64 + i // 250}.{(i % 250) // 10}.{i % 10 + 1}" for i in range(cfg.n_relay_peers)]
    peer_asn = rng.integers(64512, 64512 + cfg.n_asns, size=cfg.n_relay_peers)
    rows = []
    for tx in w.txs:
        chain = {
            "txid": tx["txid"],
            "input_addresses": ";".join(a for a, _v in tx["ins"]),
            "output_addresses": ";".join(a for a, _v in tx["outs"]),
            "input_amounts": ";".join(f"{v:.8f}" for _a, v in tx["ins"]),
            "output_amounts": ";".join(f"{v:.8f}" for _a, v in tx["outs"]),
            "fee": tx["fee"], "script_type": "p2wpkh",
        }
        if rng.random() < cfg.unobserved_share:
            rows.append({"timestamp": round(tx["timestamp"], 3), "src_ip": "", "dst_ip": "",
                         "src_port": "", "dst_port": "", **chain, "geo_country": "", "asn": ""})
            continue
        k = int(min(cfg.n_observers, 2 + rng.poisson(4)))
        for obs in rng.choice(cfg.n_observers, size=k, replace=False):
            direct = rng.random() < tx["conn"]
            if direct:
                ip, asn, delay = tx["origin"][0], tx["origin"][1], rng.exponential(0.3)
            else:
                j = int(rng.integers(0, cfg.n_relay_peers))
                ip, asn, delay = peers[j], int(peer_asn[j]), rng.exponential(2.5)
            rows.append({"timestamp": round(tx["timestamp"] + delay, 3), "src_ip": ip,
                         "dst_ip": observers[obs], "src_port": 8333, "dst_port": 8333,
                         **chain, "geo_country": "", "asn": asn})
    columns = ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid",
               "input_addresses", "output_addresses", "input_amounts", "output_amounts",
               "fee", "script_type", "geo_country", "asn"]
    return pd.DataFrame(rows, columns=columns).sort_values("timestamp", kind="stable")


def build_noisy_world(config: NoisyWorldConfig | None = None):
    cfg = config or NoisyWorldConfig()
    w = _W(rng=np.random.default_rng(cfg.seed), cfg=cfg)
    for band in SPLIT_BANDS:
        for behaviour, weight in BEHAVIOUR_WEIGHTS.items():
            for i in range(weight * cfg.entities_per_weight):
                flipped = bool(w.rng.random() < cfg.label_noise)
                _entity(w, f"E2-{band[:2]}-{behaviour}-{i}", behaviour, band, flipped)
    capture = _observations(w)
    labels = pd.DataFrame({"address": list(w.address_label),
                           "y": list(w.address_label.values()),
                           "entity": [w.address_entity[a] for a in w.address_label]})
    return capture, labels, pd.DataFrame(w.entities)


def write_noisy_world(root: Path, config: NoisyWorldConfig | None = None) -> dict:
    from obsidianchain import provenance as prov

    cfg = config or NoisyWorldConfig()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    capture, labels, entities = build_noisy_world(cfg)
    capture.to_csv(root / "capture.csv", index=False)
    truth = root / TRUTH_DIR
    truth.mkdir(exist_ok=True)
    labels.to_csv(truth / "labels.csv", index=False)
    entities.to_csv(truth / "entities.csv", index=False)
    (truth / "README.txt").write_text(
        "FOR_EVALUATION_ONLY\n\nAddress labels and entity behaviours for the "
        "noisy benchmark world. No inference stage may read this directory.\n",
        encoding="utf-8")
    manifest = {
        "schema": WORLD_SCHEMA,
        "provenance_type": prov.ProvenanceType.SYNTHETIC_CONTROL.value,
        "generator_version": GENERATOR_VERSION,
        "config": asdict(cfg),
        "config_sha256": cfg.digest(),
        "positive_class": list(POSITIVE_CLASS),
        "counts": {"transactions": int(capture.txid.nunique()), "capture_rows": int(len(capture)),
                   "addresses": int(len(labels)), "entities": int(len(entities)),
                   "positive_addresses": int(labels.y.sum())},
        "notes": [
            "SYNTHETIC_CONTROL. Every number measured on this world is a property "
            "of the generator, not of Bitcoin.",
            "RELAY_LAUNDERING is chain-identical to NORMAL. With network_signal=True "
            "it broadcasts from well-connected nodes; with False it does not (NULL control).",
            "The network signal was placed in the generator on purpose. The world "
            "tests extraction, not whether real traffic carries such a signal.",
        ],
    }
    (root / "world_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
