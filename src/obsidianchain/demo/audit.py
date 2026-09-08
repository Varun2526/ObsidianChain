"""Post-hoc contradiction audit: the question the veto cannot ask in time.

The gap this closes
-------------------
:class:`~obsidianchain.cluster.constrained.ConstrainedUnionFind` consults the
oracle at the moment a union is proposed, comparing the pooled statistics of
the two components *as they stand then*. That is the right place for a veto -
it is the only place a merge can still be prevented - but it has a structural
blind spot.

A component's pooled count is the sum of its members'. So a component can be
assembled one thin address at a time, every single decision having a side
below the pooled minimum and therefore answering NO_EVIDENCE, and end up
holding two pools that would separate decisively if anyone compared them. No
decision boundary ever presented them as two sides, so no veto was possible.
The contradiction does not exist until clustering has finished.

What this audit does
--------------------
For each finished component, and for each member whose *own* pooled count
clears the production minimum, it asks the unchanged oracle a question the
run never asked: **does the network say this address does not belong with the
rest of its own component?** Both sides must clear the minimum, both gates
apply, and the configuration is the production one.

Everything it reads is observable. Member lists come from the union-find the
engine just produced, pooled statistics from the same oracle the engine used,
and nothing here touches ``processed/network_truth/``.

Why leave-one-out rather than a partition search
------------------------------------------------
A partition search would need a clustering step over member centroids, and
that step would be a new estimator with its own failure modes - a second
thing to validate before the first could be believed. Leave-one-out has none:
it is the same two-group comparison the engine already makes, with the groups
taken from the component itself. It cannot find every contradiction (two
thin halves of a component stay invisible to it, and that limitation is real)
but everything it does find is found by the production rule.

What happens to a contradiction
-------------------------------
It is recorded, never resolved. The pair is fed back as a static cannot-link
and :meth:`ConstrainedUnionFind.audit_existing_constraints` marks the
component CONTESTED. Union-find has no split, and inventing one would mean
choosing which cryptographically-backed co-spend edge to discard. A contested
component is one the analyst is being told not to trust, which is more useful
than either ignoring the contradiction or aborting the run.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from dataclasses import dataclass

from obsidianchain.network.separation import (
    GroupStats,
    SeparationEvidence,
    SeparationOracle,
    Verdict,
)


@dataclass
class Contradiction:
    """One member that network evidence places outside its own component."""

    member: int
    """Address code of the member that does not fit."""

    counterpart: int
    """A member of the rest, so the cannot-link has two concrete endpoints."""

    root: int
    """Component root at audit time."""

    component_size: int
    evidence: SeparationEvidence

    def describe(self) -> str:
        return (
            f"member {self.member} vs the other {self.component_size - 1} "
            f"members of its component: {self.evidence.describe()}"
        )


def _pool(oracle: SeparationOracle, codes) -> GroupStats:
    """Pooled statistics for an explicit set of addresses.

    Summed rather than subtracted from the component total. Subtraction would
    be one vector operation instead of many, but it accumulates floating-point
    error into a variance that a chi-square then amplifies, and these
    components are small enough that the difference does not matter.
    """
    stats = [oracle.stats_for(int(code)) for code in codes]
    if not stats:
        return oracle.empty_stats()
    return functools.reduce(lambda a, b: a.combine(b), stats)


def members_by_root(forest, n_addresses: int) -> dict[int, list[int]]:
    """Current component membership, keyed by root."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for code in range(int(n_addresses)):
        grouped[int(forest.find(code))].append(code)
    return dict(grouped)


def find_contradictions(
    forest, oracle: SeparationOracle, n_addresses: int
) -> list[Contradiction]:
    """Every member the production rule separates from its own component.

    Returns at most one contradiction per component: once a component is
    contested, further evidence that it is contested changes nothing about
    what the analyst is told, and reporting five rows for one component would
    repeat the edge-counting mistake that turned a single decision into 166
    false splits in Phase 3.3.
    """
    minimum = oracle.config.min_pooled_observations
    found: list[Contradiction] = []

    for root, members in sorted(members_by_root(forest, n_addresses).items()):
        if len(members) < 2:
            continue  # a singleton has no "rest" to be separated from
        for member in members:
            mine = oracle.stats_for(member)
            if mine.count < minimum:
                continue  # not enough of its own evidence to answer
            others = [code for code in members if code != member]
            rest = _pool(oracle, others)
            if rest.count < minimum:
                continue
            evidence = oracle.compare(mine, rest)
            if evidence.verdict is not Verdict.SEPARATED:
                continue
            counterpart = max(
                others, key=lambda code: oracle.stats_for(code).count
            )
            found.append(
                Contradiction(
                    member=int(member),
                    counterpart=int(counterpart),
                    root=int(root),
                    component_size=len(members),
                    evidence=evidence,
                )
            )
            break  # one per component; see the docstring
    return found


def record_contradictions(
    forest, contradictions: list[Contradiction]
) -> int:
    """Feed contradictions back as cannot-links and mark components CONTESTED.

    Uses the engine's own two-step path unchanged: ``add_cannot_link`` for a
    constraint that arrived too late to prevent anything, then
    ``audit_existing_constraints`` to mark every component that now violates
    one. Returns the number of components marked.
    """
    for contradiction in contradictions:
        forest.add_cannot_link(contradiction.member, contradiction.counterpart)
    return int(forest.audit_existing_constraints())
