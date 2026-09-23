"""Risk propagation from seed illicit wallets.

Problem-statement focus area "Risk Scoring: propagate risk scores from seed
illicit wallets via algorithms". Implemented as personalized PageRank (a
random walk with restart) on the address-transaction bipartite graph.

The walk
--------
A walker starts at a seed address. Each step it either restarts at a seed
(probability ``alpha``) or moves address -> one of that address's
transactions -> one of that transaction's addresses, uniformly at each hop.
The stationary visit rate of an address is its propagated risk: high when it
sits a short, narrow path away from seeds, low when the only paths are long
or pass through transactions with many participants.

Two structural choices, both about hubs:

* **Uniform choice among a transaction's addresses** divides a seed's
  influence by the transaction's size. A 2-address payment from a seed
  carries far more than a 7,000-address batch payout that happens to include
  one - which is correct, since in a batch the seed's co-participants are
  strangers.
* **Transactions above ``max_tx_degree`` are dropped from the walk.** Such
  hubs (exchange sweeps, large CoinJoins) connect unrelated users; walking
  through them spreads risk to everyone. They are reported, not hidden.

What the score is NOT
---------------------
It is not a probability, and it does not say an address is illicit. It says
the address is structurally close to addresses that are. The evidence item
carries the nearest seed and the path so an analyst can judge the link.

Seeds must be labels known BEFORE the addresses being scored - in evaluation,
labels up to the fold's training end only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

#: Restart probability. 0.3 keeps the walk local: the expected distance
#: from the last restart is about 1/0.3 - 1 ~ 2.3 address->address hops.
DEFAULT_ALPHA = 0.3

#: Transactions with more participants than this are excluded from the walk.
DEFAULT_MAX_TX_DEGREE = 500

#: Power-iteration steps. With alpha = 0.3 the residual after 40 steps is
#: 0.7**40 ~ 6e-7 of the seed mass.
DEFAULT_ITERATIONS = 40

#: Hops searched when attaching a path explanation (address-to-address).
PATH_MAX_HOPS = 3


@dataclass
class PropagationResult:
    """Propagated risk per address plus the path back to the nearest seed."""

    scores: pd.Series
    """Address -> propagated risk in [0, 1] (1.0 for seeds themselves)."""
    raw: pd.Series
    """Address -> stationary visit rate (sums to 1 over reachable addresses)."""
    seeds: list[str]
    seeds_in_graph: int
    hub_txids_excluded: list[str]
    alpha: float
    max_tx_degree: int
    paths: dict[str, dict[str, Any]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        reached = int(((self.scores > 0) & ~self.scores.index.isin(self.seeds)).sum())
        return {
            "seed_count": len(self.seeds),
            "seeds_in_graph": self.seeds_in_graph,
            "addresses_reached": reached,
            "hub_transactions_excluded": len(self.hub_txids_excluded),
            "alpha": self.alpha,
            "max_tx_degree": self.max_tx_degree,
        }


def edges_from_capture(frame: pd.DataFrame) -> pd.DataFrame:
    """``(address, txid)`` pairs from a canonical capture frame, one per role."""
    rows: list[tuple[str, str]] = []
    for _, row in frame.drop_duplicates(subset="txid").iterrows():
        txid = str(row.get("txid") or "").strip()
        if not txid:
            continue
        for col in ("input_addresses", "output_addresses"):
            addrs = row.get(col)
            if isinstance(addrs, (list, tuple)):
                rows.extend((str(a).strip(), txid) for a in addrs
                            if a is not None and not pd.isna(a) and str(a).strip())
    return pd.DataFrame(rows, columns=["address", "txid"]).drop_duplicates()


def _normalise_rows(m: sparse.csr_matrix) -> sparse.csr_matrix:
    deg = np.asarray(m.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=float), where=deg > 0)
    return sparse.diags(inv) @ m


def propagate(edges: pd.DataFrame, seeds: Iterable[str], *,
              alpha: float = DEFAULT_ALPHA,
              max_tx_degree: int = DEFAULT_MAX_TX_DEGREE,
              iterations: int = DEFAULT_ITERATIONS,
              explain: Iterable[str] | None = None) -> PropagationResult:
    """Personalized PageRank from ``seeds`` over ``edges`` (address, txid).

    ``explain`` names the addresses that get a seed path attached; None
    means none (paths cost a BFS each).
    """
    seeds = sorted({str(s) for s in seeds})
    edges = edges[["address", "txid"]].astype(str).drop_duplicates()
    tx_degree = edges.groupby("txid").size()
    hubs = tx_degree[tx_degree > max_tx_degree].index
    walk = edges[~edges.txid.isin(hubs)]

    addresses = pd.Index(sorted(set(edges.address)))
    empty = pd.Series(0.0, index=addresses)
    seed_idx = addresses.get_indexer(seeds)
    seed_idx = seed_idx[seed_idx >= 0]
    if walk.empty or seed_idx.size == 0:
        return PropagationResult(empty, empty.copy(), seeds, int(seed_idx.size),
                                 list(map(str, hubs)), alpha, max_tx_degree)

    txs = pd.Index(sorted(set(walk.txid)))
    a = addresses.get_indexer(walk.address)
    t = txs.get_indexer(walk.txid)
    incidence = sparse.csr_matrix((np.ones(len(walk)), (a, t)),
                                  shape=(len(addresses), len(txs)))
    addr_to_tx = _normalise_rows(incidence)
    tx_to_addr = _normalise_rows(incidence.T.tocsr())

    restart = np.zeros(len(addresses))
    restart[seed_idx] = 1.0 / seed_idx.size
    r = restart.copy()
    for _ in range(iterations):
        # Row vector r moves along address -> tx -> address. An address with
        # no walkable transaction keeps its mass nowhere; the restart term
        # re-injects the lost mass at the seeds.
        moved = (r @ addr_to_tx) @ tx_to_addr
        r = alpha * restart + (1 - alpha) * moved
    raw = pd.Series(r, index=addresses)
    # Scale so a seed's own retained mass (alpha / n_seeds) reads as 1.0:
    # the score is "how much of a seed's weight reaches this address".
    scaled = np.clip(r * seed_idx.size / alpha, 0.0, 1.0)
    scores = pd.Series(scaled, index=addresses)
    scores.iloc[seed_idx] = 1.0

    result = PropagationResult(scores, raw, seeds, int(seed_idx.size),
                               list(map(str, hubs)), alpha, max_tx_degree)
    if explain is not None:
        result.paths = seed_paths(walk, seeds, explain)
    return result


def seed_paths(edges: pd.DataFrame, seeds: Iterable[str],
               targets: Iterable[str], max_hops: int = PATH_MAX_HOPS) -> dict[str, dict[str, Any]]:
    """Shortest address->tx->address path from any seed to each target.

    One multi-source BFS, so the cost is one pass over the graph however
    many targets are asked for.
    """
    targets = {str(x) for x in targets}
    seeds = {str(s) for s in seeds}
    by_addr = edges.groupby("address")["txid"].apply(list).to_dict()
    by_tx = edges.groupby("txid")["address"].apply(list).to_dict()
    parent: dict[str, tuple[str | None, str | None]] = {s: (None, None) for s in seeds if s in by_addr}
    depth = {s: 0 for s in parent}
    queue = deque(parent)
    while queue:
        cur = queue.popleft()
        if depth[cur] >= max_hops:
            continue
        for tx in by_addr.get(cur, ()):
            for nxt in by_tx.get(tx, ()):
                if nxt not in parent:
                    parent[nxt] = (cur, tx)
                    depth[nxt] = depth[cur] + 1
                    queue.append(nxt)
    paths: dict[str, dict[str, Any]] = {}
    for target in targets:
        if target not in parent or target in seeds:
            continue
        hops, node = [], target
        while parent[node][0] is not None:
            prev, tx = parent[node]
            hops.append({"from": prev, "via_txid": tx, "to": node})
            node = prev
        paths[target] = {"seed": node, "hops": depth[target], "path": list(reversed(hops))}
    return paths
