"""M1 - graph features over the address-transaction bipartite graph.

SPEC 3.2 M1. NOT over ``AddrAddr_edgelist.csv``: that file carries no
timestep (SPEC 5.2), so "restricted to Time step <= t" is not expressible on
it. Counterparties are derived through the transaction layer instead, where
``Time step`` exists - address ``a`` is a counterparty of ``b`` as of ``t``
when some transaction with ``Time step <= t`` has one as an input and the
other as an output.

``cluster_size_final`` is NOT built. Final cluster membership must not be a
predictive feature; only ``cluster_size_asof_t`` enters M1, computed by
replaying the co-spend union-find forward and snapshotting it.

Where the as-of-t rule actually bites
-------------------------------------
An address's own edges all carry a timestep at or before its own last active
timestep, so degree and counterparty counts need no extra filter. The
clustering coefficient does: it asks whether two of ``a``'s neighbours are
themselves connected, and that connecting transaction can be in ``a``'s
future. Those edges are filtered explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from obsidianchain.features.incidence import ROLE_IN, ROLE_OUT, Incidence

M1_COLUMNS = [
    "in_degree_asof_t", "out_degree_asof_t",
    "weighted_in_degree_asof_t", "weighted_out_degree_asof_t",
    "unique_counterparties_asof_t", "counterparty_growth_rate_asof_t",
    "local_clustering_coefficient_asof_t",
    "cluster_size_asof_t",
]

#: Transactions wider than this contribute no counterparty edges.
#:
#: A transaction with 69 inputs and 1,001 outputs implies 69,069 pairs, and
#: the claim that all 69,069 are counterparties is not one this data supports:
#: a large payout batch says its recipients share a payer, not each other.
#: 28 transactions exceed this. Capping is a DESIGN choice recorded here
#: rather than a silent filter.
MAX_PAIRS_PER_TX = 10_000


def counterparty_edges(incidence: Incidence) -> pd.DataFrame:
    """Undirected address-address edges with the timestep that created them.

    One row per (input address, output address, timestep). Self-pairs are
    dropped: an address that funds and receives from the same transaction is
    not its own counterparty.
    """
    frame = incidence.frame
    ins = frame[frame["role"] == ROLE_IN][["code", "txId", "Time step"]]
    outs = frame[frame["role"] == ROLE_OUT][["code", "txId"]]

    width = (
        ins.groupby("txId").size() * outs.groupby("txId").size()
    ).dropna()
    keep = set(width[width <= MAX_PAIRS_PER_TX].index)
    ins = ins[ins["txId"].isin(keep)]
    outs = outs[outs["txId"].isin(keep)]

    edges = ins.merge(outs, on="txId", suffixes=("_a", "_b"))
    edges = edges[edges["code_a"] != edges["code_b"]]
    return edges[["code_a", "code_b", "Time step"]].reset_index(drop=True)


def build(incidence: Incidence, cutoff: pd.Series,
          cospend_sizes: pd.Series | None = None) -> pd.DataFrame:
    """One row per address code.

    Args:
        incidence: the substrate.
        cutoff: address code -> observation timestep ``t``.
        cospend_sizes: address code -> co-spend component size as of ``t``,
            from :func:`cospend_size_asof`. None leaves the column NULL.
    """
    frame = incidence.frame
    tx = incidence.transactions
    joined = frame.join(tx[["in_BTC_total", "out_BTC_total"]], on="txId")

    index = pd.Index(sorted(frame["code"].unique()), name="code")
    out = pd.DataFrame(index=index)

    ins = joined[joined["role"] == ROLE_IN].groupby("code")
    outs = joined[joined["role"] == ROLE_OUT].groupby("code")
    out["in_degree_asof_t"] = outs.size().reindex(index, fill_value=0)
    out["out_degree_asof_t"] = ins.size().reindex(index, fill_value=0)
    out["weighted_in_degree_asof_t"] = outs["out_BTC_total"].sum().reindex(index)
    out["weighted_out_degree_asof_t"] = ins["in_BTC_total"].sum().reindex(index)

    edges = counterparty_edges(incidence)
    undirected = pd.concat(
        [
            edges.rename(columns={"code_a": "u", "code_b": "v"}),
            edges.rename(columns={"code_b": "u", "code_a": "v"}),
        ],
        ignore_index=True,
    )[["u", "v", "Time step"]]

    counts = undirected.groupby("u")["v"].nunique()
    out["unique_counterparties_asof_t"] = counts.reindex(index, fill_value=0)

    span = (cutoff - incidence.first_timestep()).reindex(index)
    out["counterparty_growth_rate_asof_t"] = (
        out["unique_counterparties_asof_t"] / (span + 1).replace(0, np.nan)
    )

    out["local_clustering_coefficient_asof_t"] = _clustering(
        undirected, cutoff, index
    )

    out["cluster_size_asof_t"] = (
        cospend_sizes.reindex(index) if cospend_sizes is not None
        else pd.Series(np.nan, index=index)
    )
    return out[M1_COLUMNS]


def _clustering(undirected: pd.DataFrame, cutoff: pd.Series,
                index: pd.Index) -> pd.Series:
    """Local clustering coefficient, with neighbour-neighbour edges filtered.

    The filter is the whole point. A triangle is closed by an edge between two
    of ``a``'s neighbours, and that edge can carry a timestep after ``a``'s
    own - which would be future information about ``a``'s neighbourhood.
    Only edges with ``Time step <= t(a)`` may close a triangle for ``a``.

    Mean degree is 1.60 and p99 is 6, so neighbour sets are tiny and the
    direct computation is cheaper than any index that would avoid it.
    """
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for u, v, step in undirected.itertuples(index=False):
        adjacency.setdefault(int(u), []).append((int(v), int(step)))

    edge_step: dict[tuple[int, int], int] = {}
    for u, v, step in undirected.itertuples(index=False):
        key = (int(u), int(v))
        prior = edge_step.get(key)
        if prior is None or step < prior:
            edge_step[key] = int(step)

    values = np.full(len(index), np.nan)
    position = {int(code): i for i, code in enumerate(index)}
    for code, neighbours in adjacency.items():
        limit = cutoff.get(code)
        if limit is None:
            continue
        nbrs = sorted({v for v, step in neighbours if step <= limit})
        k = len(nbrs)
        if k < 2:
            # Undefined, not zero: a node with fewer than two neighbours has
            # no pair that could have been connected.
            continue
        links = 0
        for i in range(k):
            for j in range(i + 1, k):
                step = edge_step.get((nbrs[i], nbrs[j]))
                if step is not None and step <= limit:
                    links += 1
        values[position[code]] = 2.0 * links / (k * (k - 1))
    return pd.Series(values, index=index)


def cospend_size_asof(data_root, cutoff: pd.Series,
                      incidence: Incidence) -> pd.Series:
    """Co-spend component size for each address, as of its own timestep.

    Replays the Phase 1 star edges in timestep order and snapshots component
    sizes after each timestep, so an address observed at ``t`` gets the size
    its component had at ``t`` - never the final size, which SPEC 4.3 and the
    Phase 6 constraints exclude from prediction.

    Uses the frozen Phase 1 loader and union-find unchanged. Nothing here
    writes a Phase 1 artifact.
    """
    from obsidianchain.cluster.unionfind import UnionFind
    from obsidianchain.io import elliptic

    graph = elliptic.load_cospend_graph(data_root, keep_labels=True)
    tx_step = incidence.transactions["Time step"]
    edge_steps = pd.Series(graph.edge_tx_ids).map(
        dict(zip(tx_step.index, tx_step.to_numpy()))
    ).to_numpy()

    # Phase 1 address codes differ from this module's factorisation, so the
    # component sizes are mapped back through the address strings.
    phase1_of_string = pd.Series(
        np.arange(graph.n_addresses, dtype=np.int64), index=graph.addresses
    )
    ours_to_phase1 = phase1_of_string.reindex(incidence.addresses).to_numpy()

    forest = UnionFind(int(graph.n_addresses))
    order = np.argsort(np.nan_to_num(edge_steps, nan=np.inf), kind="stable")
    edges = graph.edges[order]
    steps = np.nan_to_num(edge_steps[order], nan=np.inf)

    result = pd.Series(np.nan, index=pd.Index(sorted(cutoff.index), name="code"))
    cutoffs = sorted({int(v) for v in cutoff.dropna().unique()})
    by_cutoff = {c: cutoff[cutoff == c].index.to_numpy() for c in cutoffs}

    position = 0
    for limit in cutoffs:
        while position < len(edges) and steps[position] <= limit:
            a, b = edges[position]
            forest.union(int(a), int(b))
            position += 1
        roots = forest.roots()
        sizes = np.bincount(roots, minlength=len(roots))
        codes = by_cutoff[limit]
        mapped = ours_to_phase1[codes]
        valid = ~pd.isna(mapped)
        idx = mapped[valid].astype(np.int64)
        result.loc[codes[valid]] = sizes[roots[idx]]
    return result
