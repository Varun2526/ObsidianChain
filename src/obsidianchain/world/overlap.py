"""Overlap / degradation: what happens as the two layers stop sharing txids.

The question
------------
A real deployment will not have a network observation for every transaction.
Collectors miss announcements, capture windows start late, and a chain
dataset covers transactions nobody was watching. So: as the fraction of
chain transactions carrying network observations falls, what does the system
actually do?

The answer this measures is NOT "detection gets worse by X%". It is narrower
and more useful: how much network EVIDENCE remains available, and whether the
system converts its absence into a negative finding. Those are different
failures and only the second one is a defect.

    high overlap   ->  rich evidence
    low overlap    ->  reduced evidence
    no overlap     ->  evidence UNAVAILABLE, and said to be unavailable

The property under test
-----------------------
``net_has_evidence`` must fall with overlap while nothing turns into a
positive claim. An address with no observations must report "no evidence",
never "evidence of no link" - missing evidence is not negative evidence, and
the whole separation layer is built on that distinction.

Why this is honest about causation
----------------------------------
Nothing here asserts that lower overlap causes a particular detection
outcome. It measures coverage and abstention, which are the quantities the
degradation is actually visible in, and reports them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: The conditions the experiment sweeps. 1.0 is the generated world as-is;
#: 0.0 is the boundary case where the network layer contributes nothing and
#: must say so rather than contributing zeros.
DEFAULT_LEVELS = (1.0, 0.9, 0.7, 0.5, 0.3, 0.0)

EXPERIMENT_SCHEMA = "obsidianchain.overlap_experiment/1"

MEANING = (
    "Overlap is the fraction of this world's transactions that carry network "
    "observations. The experiment measures how much network evidence remains "
    "available as that fraction falls. It does not claim that lower overlap "
    "causes a particular detection outcome; it measures coverage and "
    "abstention, which is where the degradation is visible."
)

ABSTENTION_MEANING = (
    "Abstention is the share of addresses for which no network evidence "
    "could be established. It is the correct response to missing "
    "observations. Missing evidence is NOT evidence of separation and NOT "
    "evidence of a link, and a system that let coverage loss become a "
    "finding would be inventing one."
)


@dataclass(frozen=True)
class OverlapResult:
    """One condition, measured."""

    requested_overlap: float
    observed_overlap: float
    transactions_total: int
    transactions_with_observations: int
    observations: int
    addresses: int
    addresses_with_evidence: int
    addresses_reaching_minimum: int
    coverage: float
    abstention: float
    pooled_observations_mean: float
    #: False when the network layer could not be built at all. The real
    #: builder REFUSES on an empty observation set rather than returning
    #: zeros, which is the behaviour under test: with nothing observed there
    #: is no evidence, and zeros would read as measurements.
    evidence_available: bool = True
    unavailable_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "requested_overlap": self.requested_overlap,
            "observed_overlap": self.observed_overlap,
            "transactions_total": self.transactions_total,
            "transactions_with_observations": self.transactions_with_observations,
            "observations": self.observations,
            "addresses": self.addresses,
            "addresses_with_evidence": self.addresses_with_evidence,
            "addresses_reaching_minimum": self.addresses_reaching_minimum,
            "coverage": self.coverage,
            "abstention": self.abstention,
            "pooled_observations_mean": self.pooled_observations_mean,
            "evidence_available": self.evidence_available,
            "unavailable_reason": self.unavailable_reason,
        }


def _thin(observations: pd.DataFrame, txids: np.ndarray, level: float,
          seed: int) -> pd.DataFrame:
    """Keep observations for a deterministic subset of transactions.

    Thinning is by TRANSACTION, not by row: a collector that missed a
    transaction missed every announcement of it, and dropping random rows
    instead would model a different and less realistic failure.
    """
    if level >= 1.0:
        return observations
    if level <= 0.0:
        return observations.iloc[:0]
    rng = np.random.default_rng(seed)
    keep_n = int(round(len(txids) * level))
    keep = set(rng.choice(txids, size=keep_n, replace=False).tolist())
    return observations[observations["txid"].isin(keep)]


def run(world_root, levels=DEFAULT_LEVELS, *, seed: int = 4242) -> dict:
    """Sweep the overlap levels over an already-generated world.

    Each level writes a thinned observation set into a scratch root and runs
    the REAL M3 builder over it. Re-measuring with the production feature
    code is the point: a bespoke coverage calculation here could disagree
    with what the pipeline actually sees.
    """
    import shutil
    import tempfile

    from obsidianchain.features import netfeat
    from obsidianchain.features.incidence import load_incidence
    from obsidianchain.network import synthetic

    world_root = Path(world_root)
    incidence = load_incidence(world_root)
    all_txids = incidence.transactions.index.to_numpy()

    source = world_root / "processed" / synthetic.OBSERVATIONS_DIR
    observations = pd.read_parquet(source / "observations.parquet")

    results = []
    for level in levels:
        thinned = _thin(observations, all_txids, float(level), seed)
        scratch = Path(tempfile.mkdtemp(prefix="oc-overlap-"))
        try:
            target = scratch / "processed" / synthetic.OBSERVATIONS_DIR
            target.mkdir(parents=True, exist_ok=True)
            thinned.to_parquet(target / "observations.parquet", index=False)
            for name in ("nodes.parquet", "observers.parquet"):
                if (source / name).is_file():
                    shutil.copy2(source / name, target / name)
            # The world's own raw/ is needed for the address factorisation
            # the M3 builder maps through.
            (scratch / "raw").mkdir(parents=True, exist_ok=True)
            for name in ("AddrTx_edgelist.csv", "TxAddr_edgelist.csv",
                         "txs_features.csv", "wallets_classes.csv"):
                shutil.copy2(world_root / "raw" / name, scratch / "raw" / name)

            try:
                m3 = netfeat.build(scratch, incidence)
                unavailable = ""
            except ValueError as exc:
                # The real builder refuses an empty observation set rather
                # than manufacturing zeros. That refusal IS the graceful
                # degradation this experiment exists to demonstrate, so it
                # is recorded as a condition rather than treated as a crash.
                m3 = None
                unavailable = str(exc)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        covered_tx = int(thinned["txid"].nunique()) if len(thinned) else 0
        if m3 is None:
            results.append(OverlapResult(
                requested_overlap=float(level),
                observed_overlap=covered_tx / len(all_txids) if len(all_txids) else 0.0,
                transactions_total=int(len(all_txids)),
                transactions_with_observations=covered_tx,
                observations=int(len(thinned)),
                addresses=int(incidence.n_addresses),
                addresses_with_evidence=0,
                addresses_reaching_minimum=0,
                coverage=0.0,
                abstention=1.0,
                pooled_observations_mean=0.0,
                evidence_available=False,
                unavailable_reason=unavailable,
            ))
            continue

        with_evidence = int(m3["net_has_evidence"].astype(bool).sum())
        reaching = int(m3["net_reaches_production_minimum"].astype(bool).sum())
        n_addresses = int(len(m3))

        results.append(OverlapResult(
            requested_overlap=float(level),
            observed_overlap=(
                covered_tx / len(all_txids) if len(all_txids) else 0.0
            ),
            transactions_total=int(len(all_txids)),
            transactions_with_observations=covered_tx,
            observations=int(len(thinned)),
            addresses=n_addresses,
            addresses_with_evidence=with_evidence,
            addresses_reaching_minimum=reaching,
            coverage=with_evidence / n_addresses if n_addresses else 0.0,
            # The complement, named for what it MEANS: these addresses got no
            # network answer, which is an abstention and not a negative.
            abstention=(
                1.0 - (with_evidence / n_addresses) if n_addresses else 1.0
            ),
            pooled_observations_mean=float(
                pd.to_numeric(m3["net_pooled_observations"],
                              errors="coerce").fillna(0).mean()
            ),
        ))

    return {
        "schema": EXPERIMENT_SCHEMA,
        "provenance_type": "SYNTHETIC_CONTROL",
        "meaning": MEANING,
        "abstention_meaning": ABSTENTION_MEANING,
        "seed": seed,
        "levels": [r.as_dict() for r in results],
        "notes": [
            "SYNTHETIC evaluation. Every figure is a property of a generated "
            "world, not a measurement of Bitcoin.",
            "Thinning removes every observation of a transaction rather than "
            "random rows: a collector that misses a transaction misses all "
            "of its announcements.",
            "Coverage falling is expected. Missing evidence becoming a "
            "finding would be the defect, and does not occur: an address "
            "with no observations reports no evidence.",
        ],
    }


def write(world_root, destination: Path, levels=DEFAULT_LEVELS,
          *, seed: int = 4242) -> dict:
    """Run the sweep and persist it beside the world."""
    payload = run(world_root, levels, seed=seed)
    payload["layer_comparison"] = layer_comparison(world_root)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return payload


# ---- chain-only vs network-only vs fused --------------------------------


LAYER_MEANING = (
    "A comparison of what each EVIDENCE LAYER contributes on this synthetic "
    "world. It is not a comparison of three models - there is one model, and "
    "the network layer is not a model. CHAIN counts what the ledger's own "
    "structure yields, NETWORK counts what announcement observations yield, "
    "and FUSED is what an investigator sees with both."
)

NOT_MEASURED_HERE = (
    "False splits and constraint precision are properties of the constrained "
    "clustering run, which this experiment does not perform. They are "
    "reported by the Phase 3.3 world experiments and are deliberately absent "
    "here rather than estimated."
)


def _cospend_components(incidence) -> dict:
    """Co-spend components of the world, by the production heuristic.

    Addresses that were inputs to a common transaction are merged, which is
    exactly the common-input-ownership heuristic Phase 1 applies. Implemented
    here with a small union-find because the world is small and importing the
    production clustering package would drag an evaluation module into a
    generator path.
    """
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    spends = incidence.frame[incidence.frame["role"] == 0]
    for _txid, group in spends.groupby("txId"):
        codes = group["code"].tolist()
        for other in codes[1:]:
            union(codes[0], other)

    components: dict = {}
    for code in incidence.frame["code"].unique():
        components.setdefault(find(code), []).append(int(code))
    return components


def layer_comparison(world_root) -> dict:
    """What CHAIN, NETWORK and FUSED each contribute, measured on this world.

    Every figure is computed with the production detectors and the production
    M3 builder, over a world whose truth is known - which is the only setting
    in which a false merge can be counted at all. Metrics this experiment
    does NOT compute are named rather than estimated.
    """
    from obsidianchain.features import mixing, netfeat, peel
    from obsidianchain.features.incidence import load_incidence

    world_root = Path(world_root)
    incidence = load_incidence(world_root)
    cutoff = incidence.last_timestep()

    truth = pd.read_csv(world_root / "world_truth" / "addresses.csv")
    entity_of = dict(zip(truth["address"], truth["true_entity"]))

    # ---- CHAIN -----------------------------------------------------------
    components = _cospend_components(incidence)
    false_merges = 0
    for _root, codes in components.items():
        entities = {
            entity_of.get(incidence.addresses[code]) for code in codes
        }
        entities.discard(None)
        if len(entities) > 1:
            false_merges += 1

    chains = peel.build_chain_graph(incidence)
    m2 = peel.build(incidence, cutoff, chains=chains,
                    min_depth=peel.PRIMARY_MIN_DEPTH)
    scored = mixing.score_transactions(incidence.transactions)

    chain = {
        "addresses": int(incidence.n_addresses),
        "transactions": int(len(incidence.transactions)),
        "cospend_components": int(len(components)),
        "components_spanning_more_than_one_entity": int(false_merges),
        "addresses_in_a_peel_chain": int(m2["in_chain"].fillna(0).astype(bool).sum()),
        "transactions_with_mixing_pattern": int(
            (scored["mixing_class"] == mixing.MIXING_PATTERN).sum()
        ),
        "transactions_suppressed_as_benign": int(
            (scored["suppressor"] != mixing.SUPPRESSOR_NONE).sum()
        ),
    }

    # ---- NETWORK ---------------------------------------------------------
    try:
        m3 = netfeat.build(world_root, incidence)
        network = {
            "available": True,
            "addresses_with_evidence": int(
                m3["net_has_evidence"].astype(bool).sum()
            ),
            "addresses_reaching_production_minimum": int(
                m3["net_reaches_production_minimum"].astype(bool).sum()
            ),
            "addresses_abstaining": int(
                (~m3["net_has_evidence"].astype(bool)).sum()
            ),
        }
    except ValueError as exc:
        network = {"available": False, "reason": str(exc)}

    fused = {
        "addresses": chain["addresses"],
        "chain_signals": (
            chain["addresses_in_a_peel_chain"]
            + chain["transactions_with_mixing_pattern"]
        ),
        "network_constrained_addresses": (
            network.get("addresses_reaching_production_minimum", 0)
        ),
        "note": (
            "FUSED is what an investigator sees with both layers present. "
            "The network layer can only ever say two groups look DIFFERENT; "
            "it never merges and never attributes ownership."
        ),
    }

    return {
        "provenance_type": "SYNTHETIC_CONTROL",
        "meaning": LAYER_MEANING,
        "not_measured_here": NOT_MEASURED_HERE,
        "chain": chain,
        "network": network,
        "fused": fused,
        "notes": [
            "SYNTHETIC evaluation. Every figure is a property of a generated "
            "world, not a measurement of Bitcoin.",
            "A component spanning more than one entity is a FALSE MERGE, "
            "countable only because this world's truth is known.",
        ],
    }
