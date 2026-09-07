"""Run a clustering from one or more evidence sources and record provenance.

Union-find has no notion of a weighted edge - a merge either happens or it
does not. "Lower weight" for change evidence is therefore expressed three
ways, all of which this module enforces:

1. a confidence threshold, so weak candidates never become edges at all;
2. ordering - co-spend edges are applied first, so a change edge is only
   ever credited with a merge that cryptographic evidence did not already
   make; and
3. separate accounting, so the contribution of each source stays visible in
   every figure downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from obsidianchain.cluster.unionfind import UnionFind
from obsidianchain.io.elliptic import CoSpendGraph


@dataclass
class ClusterRun:
    """The outcome of one clustering configuration."""

    label: str
    roots: np.ndarray
    sizes: np.ndarray
    n_addresses: int
    cospend_edges: int
    cospend_merges: int
    change_edges: int = 0
    change_merges: int = 0
    change_confidence_mean: float = float("nan")

    @property
    def n_clusters(self) -> int:
        return int(self.sizes.size)

    @property
    def largest(self) -> int:
        return int(self.sizes[0]) if self.sizes.size else 0

    @property
    def singletons(self) -> int:
        return int((self.sizes == 1).sum())

    @property
    def clustered(self) -> int:
        return int(self.sizes[self.sizes > 1].sum())

    @property
    def coverage(self) -> float:
        if not self.n_addresses:
            return 0.0
        return self.clustered / self.n_addresses * 100

    @property
    def total_merges(self) -> int:
        return self.cospend_merges + self.change_merges


def run_clustering(
    graph: CoSpendGraph,
    label: str,
    change_edges: np.ndarray | None = None,
    change_confidences: np.ndarray | None = None,
) -> ClusterRun:
    """Cluster addresses, optionally adding change evidence after co-spend.

    Co-spend edges go in first deliberately. Applying them first means
    ``change_merges`` counts only the merges that change detection actually
    contributed - not ones the multi-input heuristic would have made anyway -
    so the two sources never get credit for the same union.
    """
    forest = UnionFind(graph.n_addresses)
    cospend_merges = forest.add_edges(graph.edges)

    change_count = 0
    change_merges = 0
    if change_edges is not None and len(change_edges):
        change_count = int(len(change_edges))
        change_merges = forest.add_edges(change_edges)

    roots = forest.roots()
    sizes = forest.component_sizes()

    mean_confidence = float("nan")
    if change_confidences is not None and len(change_confidences):
        mean_confidence = float(np.mean(change_confidences))

    return ClusterRun(
        label=label,
        roots=roots,
        sizes=sizes,
        n_addresses=int(graph.n_addresses),
        cospend_edges=int(graph.n_edges),
        cospend_merges=int(cospend_merges),
        change_edges=change_count,
        change_merges=int(change_merges),
        change_confidence_mean=mean_confidence,
    )
