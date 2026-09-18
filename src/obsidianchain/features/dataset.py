"""Assemble the Phase 6 dataset: features, split, label, provenance.

This is the ONLY module in the package that opens a label file, and it does so
after every feature is built. A feature builder therefore cannot see an
outcome even by accident - the same inversion ``network/boundary.py`` uses for
ground truth.

Split construction (SPEC 6.1, 4.4)
----------------------------------
An address is assigned by FIRST appearance::

    TRAIN       t1  - t34
    VALIDATION  t35 - t41
    TEST        t42 - t49

and any address active on both sides of a boundary is REMOVED ENTIRELY. At
under 1% of labeled addresses (SPEC 0.5) that costs almost nothing and makes
the three splits address-disjoint by construction, which matters because
labels are static: an address appearing in train and test would be memorised
rather than predicted.

The observation point for an address is its LAST active timestep. Once
spanners are dropped, every transaction it touches lies inside its own split,
so aggregating to that point cannot reach across a boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.features import behaviour, graph, mixing, netfeat, peel
from obsidianchain.features.incidence import Incidence, load_incidence

#: SPEC 6.1. Inclusive upper bounds on first appearance.
TRAIN_END = 34
VALIDATION_END = 41
TEST_END = 49

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"

#: SPEC 1. class 3 (unknown) is EXCLUDED, never mapped to licit.
LABEL_ILLICIT = 1
LABEL_LICIT = 2

ARTIFACT_SCHEMA = "obsidianchain.phase6_dataset/1"

#: The FROZEN production contract. The Phase 6 experiment, the model and the
#: published dataset fingerprint are all defined over exactly these four
#: groups, so this literal does not change.
FEATURE_GROUPS = {
    "M0": behaviour.M0_COLUMNS,
    "M1": graph.M1_COLUMNS,
    "M2": peel.M2_COLUMNS,
    "M3": netfeat.M3_COLUMNS,
}

#: Groups a build may ADD on request. M4 is mixing / CoinJoin-like structure.
#:
#: Kept separate rather than appended to FEATURE_GROUPS on purpose. The
#: dataset fingerprint is the SHA-256 of the written matrix and it is what
#: every alert id, every case binding and every stored alert reference is
#: addressed by; silently widening the default build would re-point all of
#: them. A build that wants M4 asks for it, and the controlled synthetic
#: evaluation does exactly that. Promoting M4 into the production contract is
#: a deliberate model generation, not a side effect of adding a file.
OPTIONAL_FEATURE_GROUPS = {
    "M4": mixing.M4_COLUMNS,
}

#: Every group that exists. What the alert layer's group table is checked
#: against, so a served artifact carrying M4 can be grouped correctly.
ALL_FEATURE_GROUPS = {**FEATURE_GROUPS, **OPTIONAL_FEATURE_GROUPS}

KEY_COLUMNS = ["code", "address", "split", "observed_at_t", "first_t", "y"]


@dataclass(frozen=True)
class BuildReport:
    """What the build did, for the terminal and the sidecar."""

    n_addresses: int
    n_labeled: int
    n_spanners_dropped: int
    split_counts: dict
    prevalence: dict


def assign_split(first_t: pd.Series, last_t: pd.Series) -> pd.Series:
    """Split by first appearance; boundary-spanning addresses get NA."""
    split = pd.Series(pd.NA, index=first_t.index, dtype="object")
    split[first_t <= TRAIN_END] = SPLIT_TRAIN
    split[(first_t > TRAIN_END) & (first_t <= VALIDATION_END)] = SPLIT_VALIDATION
    split[first_t > VALIDATION_END] = SPLIT_TEST

    spans_first = (first_t <= TRAIN_END) & (last_t > TRAIN_END)
    spans_second = (first_t <= VALIDATION_END) & (last_t > VALIDATION_END)
    split[spans_first | spans_second] = pd.NA
    return split


def build_dataset(data_root: Path, min_depth: int = peel.PRIMARY_MIN_DEPTH,
                  *, groups: dict | None = None,
                  ) -> tuple[pd.DataFrame, BuildReport]:
    """Build the full feature matrix with keys, split and label.

    ``groups`` defaults to the FROZEN four. Passing
    :data:`ALL_FEATURE_GROUPS` adds M4; the controlled synthetic evaluation
    does that, and the production build does not, so the published dataset
    fingerprint is unaffected by M4 existing.
    """
    groups = FEATURE_GROUPS if groups is None else groups
    data_root = Path(data_root)
    incidence = load_incidence(data_root)

    first_t = incidence.first_timestep()
    last_t = incidence.last_timestep()
    split = assign_split(first_t, last_t)

    cutoff = last_t
    m0 = behaviour.build(incidence)
    sizes = graph.cospend_size_asof(data_root, cutoff, incidence)
    m1 = graph.build(incidence, cutoff, cospend_sizes=sizes)
    chains = peel.build_chain_graph(incidence)
    m2 = peel.build(incidence, cutoff, chains=chains, min_depth=min_depth)
    m3 = netfeat.build(data_root, incidence)

    blocks = [m0, m1, m2, m3]
    if "M4" in groups:
        blocks.append(mixing.build(incidence, cutoff))

    index = blocks[0].index
    for block in blocks[1:]:
        index = index.union(block.index)
    frame = pd.DataFrame(index=index)
    frame.index.name = "code"
    for block in blocks:
        frame = frame.join(block, how="left")

    frame.insert(0, "first_t", first_t.reindex(index).astype("Int16"))
    frame.insert(0, "observed_at_t", cutoff.reindex(index).astype("Int16"))
    frame.insert(0, "split", split.reindex(index))
    frame.insert(0, "address", incidence.addresses[index.to_numpy()])
    frame = frame.reset_index()

    # ---- labels, joined LAST ------------------------------------------
    labels = pd.read_csv(data_root / "raw" / "wallets_classes.csv")
    frame = frame.merge(labels, on="address", how="left")
    n_spanners = int(frame["split"].isna().sum())

    frame["y"] = np.where(
        frame["class"] == LABEL_ILLICIT, 1,
        np.where(frame["class"] == LABEL_LICIT, 0, np.nan),
    )
    frame = frame.drop(columns=["class"])

    usable = frame[frame["split"].notna() & frame["y"].notna()].copy()
    usable["y"] = usable["y"].astype(np.int8)

    counts = usable.groupby("split").size().to_dict()
    prevalence = usable.groupby("split")["y"].mean().to_dict()
    report = BuildReport(
        n_addresses=int(len(frame)),
        n_labeled=int(len(usable)),
        n_spanners_dropped=n_spanners,
        split_counts={k: int(v) for k, v in counts.items()},
        prevalence={k: float(v) for k, v in prevalence.items()},
    )
    ordered = KEY_COLUMNS + [c for group in groups.values() for c in group]
    return usable[ordered].reset_index(drop=True), report


def feature_columns(groups: tuple[str, ...]) -> list[str]:
    """Column list for an ablation stage, e.g. ``("M0", "M1")``."""
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def dataset_fingerprint(data_root: Path, min_depth: int) -> str:
    """Digest over every input that determines a row of this dataset.

    Same discipline as :mod:`obsidianchain.run_fingerprint`: the three raw
    files whose bytes fix the features, plus the configuration that lives in
    code and would otherwise change every value while leaving all file hashes
    identical.
    """
    from obsidianchain import run_fingerprint as rf

    raw = Path(data_root) / "raw"
    parts = [
        "obsidianchain.phase6_dataset/1",
        f"addr_tx={rf.sha256_file(raw / 'AddrTx_edgelist.csv')}",
        f"tx_addr={rf.sha256_file(raw / 'TxAddr_edgelist.csv')}",
        f"tx_features={rf.sha256_file(raw / 'txs_features.csv')}",
        f"labels={rf.sha256_file(raw / 'wallets_classes.csv')}",
        f"split=train<={TRAIN_END},val<={VALIDATION_END},test<={TEST_END}",
        f"peel_min_depth={int(min_depth)}",
        f"peel_outputs={list(peel.ADMISSIBLE_OUTPUTS)}",
        f"max_pairs_per_tx={graph.MAX_PAIRS_PER_TX}",
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
