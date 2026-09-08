"""One replay of the co-spend edges, recording what the evidence said.

Why this module exists
----------------------
Four places walked the edge list, kept pooled network evidence per component,
merged it on union, and asked the oracle what it said: the constrained
clusterer itself, the Phase 3.1 funnel, the Phase 3.3 decision scorer, and the
demonstration runner. Two of them reached into a private method to observe a
decision without making it.

Four copies of one loop is four places for a bug to differ, and the funnel's
copy kept its own pooled-statistics dictionary over a plain union-find - a
second implementation of the aggregation :class:`ConstrainedUnionFind`
already performs. This module is that loop, once. After it, the pooled
evidence aggregation lives in exactly one place
(:meth:`ConstrainedUnionFind._merge_component_state`) and the loop that
drives it lives in exactly one place (here).

Behaviour-preserving by construction
------------------------------------
This is an extraction, not a rewrite. Nothing here computes a statistic:
pooling, transaction attribution, observer aggregation, the pooled minimum,
the chi-square, the effect size, the p-value, the evidence states and the
separation threshold all stay exactly where they were, in
:mod:`obsidianchain.network.separation`. The walker only decides the order in
which they are asked and what it writes down.

Two behaviours had to be reproduced exactly rather than tidied, and both are
worth naming because they look like bugs:

* **Double evaluation.** With ``veto=True`` the walker calls
  :meth:`ConstrainedUnionFind.evaluate_cannot_link` to record the answer and
  then :meth:`~ConstrainedUnionFind.union`, which evaluates again. So
  ``counters.evaluated`` and ``counters.abstained`` count each decision
  twice. The demonstration's reported "evaluated 4 / abstained 30" over 17
  decisions is that doubling, and correcting it here would silently change a
  published figure. It is preserved; the single-evaluation form is a
  behavioural change and belongs in its own phase.

* **Boundary de-duplication.** A component boundary re-proposed by a later
  edge is provenance, not a second decision. Scoring the edges instead
  turned one refusal into 166 in the reach-stress fixture. With
  ``veto=False`` de-duplication can never fire - an applied union makes the
  next edge across it redundant - so the funnel's counts are untouched by
  its presence.

Truth isolation
---------------
Nothing here reads ground truth, and it must stay that way. The Phase 3.3
scorer attributes entities to the member lists this walker hands back, *after*
the walk has finished. ``track_members`` exists for that and for nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from obsidianchain.cluster.constrained import ConstrainedUnionFind
from obsidianchain.network.separation import (
    SeparationConfig,
    SeparationEvidence,
    SeparationOracle,
    separation_evidence,
)


@dataclass
class UnionDecision:
    """One proposed union between two distinct components.

    Field naming follows the two callers that consume it, so neither had to
    change its own record shape: ``node_a``/``node_b`` are the edge endpoints
    the funnel records, ``component_a``/``component_b`` the sorted boundary
    the Phase 3.3 scorer keys on.
    """

    edge_index: int
    """Position in ``graph.edges``, so a decision can be traced to its edge."""

    node_a: int
    node_b: int

    root_a: int
    """``find(node_a)`` at decision time. ``evidence.n_a`` is this side."""
    root_b: int

    size_a: int
    """Members on ``root_a``'s side at decision time."""
    size_b: int

    evidence: SeparationEvidence
    merged: bool

    proposing_edges: int = 1
    """Later edges that re-proposed this same boundary. Provenance only."""

    members_a: tuple[int, ...] = ()
    """Populated only when ``track_members`` is set. Evaluation use only."""
    members_b: tuple[int, ...] = ()

    @property
    def component_a(self) -> int:
        return min(self.root_a, self.root_b)

    @property
    def component_b(self) -> int:
        return max(self.root_a, self.root_b)

    @property
    def blocked(self) -> bool:
        return not self.merged


@dataclass
class ReplayResult:
    """What one walk of the edge list produced."""

    forest: ConstrainedUnionFind
    decisions: list[UnionDecision] = field(default_factory=list)
    n_edges: int = 0
    n_redundant: int = 0
    """Edges whose endpoints were already in one component. No decision."""

    @property
    def n_proposed(self) -> int:
        return self.n_edges - self.n_redundant


def replay_unions(
    graph,
    oracle: SeparationOracle | None,
    *,
    veto: bool = True,
    config: SeparationConfig | None = None,
    track_members: bool = False,
    collect: bool = True,
    on_decision: Callable[[UnionDecision], None] | None = None,
) -> ReplayResult:
    """Walk ``graph.edges``, pooling evidence and recording each decision.

    Args:
        graph: A :class:`CoSpendGraph`. Edges are applied in array order,
            which is the baseline's order, so every caller describes the same
            sequence of proposed merges.
        oracle: Pooled network evidence, or None for a chain-only walk in
            which every answer is NO_EVIDENCE.
        veto: Whether a SEPARATED verdict refuses the union. True is
            production behaviour. False follows the baseline trajectory and
            is what the Phase 3.1 funnel needs - see the module docstring.
        config: Thresholds for the *recorded* comparison. None uses the
            oracle's own configuration through
            :meth:`ConstrainedUnionFind.evaluate_cannot_link`, which is the
            production path and also honours static cannot-links. Passing a
            config instead compares the pooled statistics directly, which is
            how the funnel extracts the raw statistic under its permissive
            probe settings without the production gates confounding it.
        track_members: Keep the member list of each side. O(addresses) memory
            and only the Phase 3.3 scorer needs it.
        collect: Retain decisions in the result. Pass False with
            ``on_decision`` to stream over a large graph without holding a
            quarter of a million records.
        on_decision: Called once per decision, in edge order, as it is made.

    Returns:
        A :class:`ReplayResult`. The forest inside it is the clustering.
    """
    forest = ConstrainedUnionFind(graph.n_addresses, oracle=oracle)
    sizes = np.ones(int(graph.n_addresses), dtype=np.int64)
    members: dict[int, list[int]] | None = (
        {i: [i] for i in range(int(graph.n_addresses))} if track_members else None
    )

    result = ReplayResult(forest=forest, n_edges=int(len(graph.edges)))
    seen: dict[tuple[int, int], UnionDecision] = {}

    for index, (a, b) in enumerate(graph.edges.tolist()):
        root_a, root_b = int(forest.find(a)), int(forest.find(b))
        if root_a == root_b:
            # Already connected: there is no merge to propose and nothing to
            # veto. Counted, never recorded as a decision.
            result.n_redundant += 1
            continue

        key = (min(root_a, root_b), max(root_a, root_b))
        previous = seen.get(key)
        if previous is not None:
            # The same boundary re-proposed by a later edge. Provenance, not
            # a second decision - counting these is what turned one refusal
            # into 166 in an earlier run.
            previous.proposing_edges += 1
            _apply(forest, a, b, veto)
            continue

        if config is None:
            evidence = forest.evaluate_cannot_link(root_a, root_b)
        else:
            evidence = separation_evidence(
                forest.pooled_stats(root_a), forest.pooled_stats(root_b), config
            )

        side_a = list(members[root_a]) if members is not None else None
        side_b = list(members[root_b]) if members is not None else None
        size_a, size_b = int(sizes[root_a]), int(sizes[root_b])

        merged = _apply(forest, a, b, veto)

        decision = UnionDecision(
            edge_index=index,
            node_a=int(a),
            node_b=int(b),
            root_a=root_a,
            root_b=root_b,
            size_a=size_a,
            size_b=size_b,
            evidence=evidence,
            merged=merged,
            members_a=tuple(side_a) if side_a is not None else (),
            members_b=tuple(side_b) if side_b is not None else (),
        )
        seen[key] = decision
        if collect:
            result.decisions.append(decision)
        if on_decision is not None:
            on_decision(decision)

        if merged:
            survivor = int(forest.find(a))
            absorbed = root_b if survivor == root_a else root_a
            sizes[survivor] = size_a + size_b
            if members is not None:
                members[survivor] = side_a + side_b
                if absorbed != survivor:
                    members.pop(absorbed, None)

    return result


def _apply(forest: ConstrainedUnionFind, a: int, b: int, veto: bool) -> bool:
    """Perform the union, with or without the veto."""
    return forest.union(a, b) if veto else forest.union_without_veto(a, b)
