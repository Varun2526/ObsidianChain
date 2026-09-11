"""M2 - PEEL-1 structural chain features. LABEL-BLIND.

SPEC 5. This module may not open ``wallets_classes.csv``,
``txs_classes.csv``, any illicit or licit label, or any model prediction.
``tests/test_phase6_leakage.py`` asserts that at source level.

PEEL-1 is **structural candidate generation, not laundering classification**.
It emits candidates and never asserts that a candidate is laundering. Whether
chain membership predicts illicitness is the question M2 exists to measure,
not an assumption it encodes.

What the detector is, and what it is not
----------------------------------------
Admissible transaction: ``num_output_addresses in {2, 3}``. There is
deliberately NO dominance filter - SPEC 5.4(E) measured that 61.3% of all
transactions have a dominant output, which makes dominance the ordinary
payment-plus-change shape rather than a signature. All discriminative weight
rests on the depth of repeated ordered transitions.

Hop: some address is an output of one admissible transaction and an input of
another, with STRICTLY increasing timestep. Strictness guarantees a DAG and
assumes no intra-timestep ordering, which Elliptic++ cannot supply - at the
cost of the 50.4% of candidate hops that are same-timestep (SPEC 5.4(D)).

As-of-t
-------
Depths are snapshotted per timestep in one forward pass rather than recomputed
per address, so an address observed at ``t`` sees only hops whose target
transaction landed at or before ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from obsidianchain.features.incidence import ROLE_IN, ROLE_OUT, Incidence

M2_COLUMNS = [
    "in_chain", "chain_depth_max", "position_in_chain", "chain_count",
    "chain_fanout_mean", "hop_gap_median",
    "value_retention_ratio", "value_retention_available",
]

#: SPEC 5.5. Small fan-out is a necessary filter, never a signature.
ADMISSIBLE_OUTPUTS = (2, 3)

#: SPEC 5.7. Primary threshold; 10 is the registered sensitivity analysis.
PRIMARY_MIN_DEPTH = 5
SENSITIVITY_MIN_DEPTH = 10


@dataclass(frozen=True)
class ChainGraph:
    """The PEEL-1 DAG and its per-timestep depth snapshots."""

    nodes: np.ndarray
    """txIds, in DAG node order."""

    node_step: np.ndarray
    """Timestep of each node."""

    hops: pd.DataFrame
    """Columns ``src``, ``dst`` (node positions), ``step`` (target timestep)."""

    depth_at: dict[int, np.ndarray]
    """timestep -> longest inbound path depth per node, as of that timestep."""

    single_input: np.ndarray
    """Whether each node's transaction has exactly one input address."""

    depth_si_at: dict[int, np.ndarray]
    """As ``depth_at``, but every node from position 2 on is single-input."""


def build_chain_graph(incidence: Incidence) -> ChainGraph:
    """Enumerate PEEL-1 hops and snapshot depth at every timestep."""
    tx = incidence.transactions
    admissible = tx.index[tx["num_output_addresses"].isin(ADMISSIBLE_OUTPUTS)]
    admissible_set = set(admissible)

    frame = incidence.frame
    subset = frame[frame["txId"].isin(admissible_set)]
    outs = subset[subset["role"] == ROLE_OUT][["code", "txId"]]
    ins = subset[subset["role"] == ROLE_IN][["code", "txId"]]

    linked = outs.merge(ins, on="code", suffixes=("_src", "_dst"))
    linked = linked[linked["txId_src"] != linked["txId_dst"]]
    step = tx["Time step"]
    src_step = linked["txId_src"].map(step).to_numpy()
    dst_step = linked["txId_dst"].map(step).to_numpy()
    forward = linked[dst_step > src_step][["txId_src", "txId_dst"]]
    forward = forward.drop_duplicates().reset_index(drop=True)

    nodes = pd.unique(
        np.concatenate([forward["txId_src"].to_numpy(),
                        forward["txId_dst"].to_numpy()])
    )
    # Keyed by the txId value itself, not int(txId): the id is opaque and
    # coercing it assumes a numbering scheme this module does not own.
    position = {t: i for i, t in enumerate(nodes)}
    node_step = step.reindex(nodes).to_numpy().astype(np.int32)
    single = (
        tx["num_input_addresses"].reindex(nodes).to_numpy() == 1
    )

    hops = pd.DataFrame({
        "src": [position[t] for t in forward["txId_src"]],
        "dst": [position[t] for t in forward["txId_dst"]],
    })
    hops["step"] = step.reindex(forward["txId_dst"]).to_numpy().astype(np.int32)
    hops = hops.sort_values("step", kind="stable").reset_index(drop=True)

    depth = np.ones(len(nodes), dtype=np.int32)
    depth_si = np.ones(len(nodes), dtype=np.int32)
    depth_at: dict[int, np.ndarray] = {}
    depth_si_at: dict[int, np.ndarray] = {}

    src = hops["src"].to_numpy()
    dst = hops["dst"].to_numpy()
    steps = hops["step"].to_numpy()
    cursor = 0
    for limit in range(1, 50):
        while cursor < len(hops) and steps[cursor] <= limit:
            s, d = src[cursor], dst[cursor]
            if depth[s] + 1 > depth[d]:
                depth[d] = depth[s] + 1
            if single[d] and depth_si[s] + 1 > depth_si[d]:
                depth_si[d] = depth_si[s] + 1
            cursor += 1
        depth_at[limit] = depth.copy()
        depth_si_at[limit] = depth_si.copy()

    return ChainGraph(
        nodes=np.asarray(nodes), node_step=node_step, hops=hops,
        depth_at=depth_at, single_input=single, depth_si_at=depth_si_at,
    )


def build(incidence: Incidence, cutoff: pd.Series,
          chains: ChainGraph | None = None,
          min_depth: int = PRIMARY_MIN_DEPTH) -> pd.DataFrame:
    """Per-address chain features, each as of that address's own timestep.

    An address is "in a chain" when it is an input or an output of a
    transaction whose as-of-``t`` depth reaches ``min_depth``.
    """
    chains = chains or build_chain_graph(incidence)
    index = pd.Index(sorted(cutoff.index), name="code")
    tx = incidence.transactions

    node_of = {t: i for i, t in enumerate(chains.nodes)}
    member = incidence.frame[incidence.frame["txId"].isin(node_of)].copy()
    member["node"] = member["txId"].map(node_of).astype(np.int32)
    member["cut"] = member["code"].map(cutoff)
    member = member.dropna(subset=["cut"])
    member["cut"] = member["cut"].astype(np.int32)

    depths = np.empty(len(member), dtype=np.int32)
    depths_si = np.empty(len(member), dtype=np.int32)
    nodes = member["node"].to_numpy()
    cuts = member["cut"].to_numpy()
    for limit in np.unique(cuts):
        mask = cuts == limit
        snapshot = chains.depth_at.get(int(limit))
        snapshot_si = chains.depth_si_at.get(int(limit))
        if snapshot is None:
            depths[mask] = 1
            depths_si[mask] = 1
            continue
        depths[mask] = snapshot[nodes[mask]]
        depths_si[mask] = snapshot_si[nodes[mask]]
    member["depth"] = depths
    member["depth_si"] = depths_si

    member["fanout"] = member["txId"].map(tx["num_output_addresses"])
    member["step"] = member["txId"].map(tx["Time step"])

    qualifying = member[member["depth"] >= min_depth]
    grouped = qualifying.groupby("code")

    out = pd.DataFrame(index=index)
    out["chain_depth_max"] = grouped["depth"].max().reindex(index)
    out["in_chain"] = out["chain_depth_max"].notna()
    out["position_in_chain"] = grouped["depth"].min().reindex(index)
    out["chain_count"] = grouped.size().reindex(index).fillna(0).astype(np.int32)
    out["chain_fanout_mean"] = grouped["fanout"].mean().reindex(index)
    out["hop_gap_median"] = _hop_gap(qualifying).reindex(index)

    # SPEC 5.6: emitted ONLY where value is attributable along the path.
    # NULL everywhere else - never imputed, never zero-filled.
    available = grouped["depth_si"].max().reindex(index) >= min_depth
    out["value_retention_available"] = available.fillna(False)
    out["value_retention_ratio"] = _retention(
        qualifying, incidence, min_depth
    ).reindex(index).where(out["value_retention_available"])
    out["chain_depth_max"] = out["chain_depth_max"].fillna(0)
    out["position_in_chain"] = out["position_in_chain"].fillna(0)
    return out[M2_COLUMNS]


def _hop_gap(qualifying: pd.DataFrame) -> pd.Series:
    """Median timestep gap between the chain transactions an address touches."""
    ordered = qualifying[["code", "step"]].drop_duplicates().sort_values(
        ["code", "step"]
    )
    ordered = ordered.assign(gap=ordered.groupby("code")["step"].diff())
    return ordered.groupby("code")["gap"].median()


def _retention(qualifying: pd.DataFrame, incidence: Incidence,
               min_depth: int) -> pd.Series:
    """Value retained across the address's deepest attributable chain position.

    Attributable only where the downstream transaction has a single input
    address, which is what makes ``in_BTC_total`` the value that arrived
    through this address rather than a total pooled from elsewhere
    (SPEC 5.6). Measured at 71.05% coverage at depth >= 5.
    """
    tx = incidence.transactions
    frame = qualifying[qualifying["depth_si"] >= min_depth].copy()
    if frame.empty:
        return pd.Series(dtype=float)
    frame["in_total"] = frame["txId"].map(tx["in_BTC_total"])
    frame["out_total"] = frame["txId"].map(tx["out_BTC_total"])
    ratio = frame["out_total"] / frame["in_total"].replace(0, np.nan)
    frame = frame.assign(ratio=ratio)
    return frame.groupby("code")["ratio"].median()
