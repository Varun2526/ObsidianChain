"""Union-find that can refuse a merge on network evidence.

This is the differentiator. Ordinary co-spend clustering only ever adds
edges, so a single wrong link is permanent and propagates: everything the
bad edge touches becomes one component, and no later evidence can undo it.
Here, network separation evidence gets a veto *before* the union happens, so
the super-cluster is prevented rather than cleaned up afterwards.

Cannot-link only
----------------
The veto is one-directional by design. Network evidence can say "these were
not broadcast from the same machine, do not merge"; it can never say "these
were, so merge them". One Electrum server broadcasts for tens of thousands
of unrelated users, so co-origination is not co-ownership. There is no
must-link path in this class and there should never be one.

The conflict case
-----------------
A cannot-link can arrive for a pair that co-spend has *already* merged
transitively - A-B and B-C were merged separately, and only now does
evidence say A and C are separate machines. The pair cannot be unmerged:
union-find has no split, and inventing one would mean choosing which of the
earlier, cryptographically-backed merges to discard.

So the conflict is recorded, not resolved. The component is marked
CONTESTED and clustering continues. A contested component is a component we
are telling the analyst not to trust, which is more useful than either
silently ignoring the contradiction or aborting the run. Contested count is
a headline output, not a diagnostic.

What is left for you
--------------------
:meth:`ConstrainedUnionFind.union` and :meth:`_evaluate_cannot_link`. The
bookkeeping around them - pooled statistics that merge in constant time, the
blocked-merge ledger, contested tracking, reporting - is written. Signatures
and contracts are below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from obsidianchain.cluster.unionfind import UnionFind
from obsidianchain.network.separation import (
    GroupStats,
    SeparationEvidence,
    SeparationOracle,
    Verdict,
)


class MergeOutcome(str, Enum):
    """What happened to one proposed union."""

    MERGED = "MERGED"
    """The union went ahead."""

    ALREADY_CONNECTED = "ALREADY_CONNECTED"
    """The pair was already in one component; nothing to do."""

    BLOCKED = "BLOCKED"
    """Network evidence refused the merge. This is the whole point."""

    CONTESTED = "CONTESTED"
    """Evidence says separate, but the pair is already merged transitively.
    Recorded, not resolved - see the module docstring."""


@dataclass
class BlockedMerge:
    """One refused union, kept so every block can be justified afterwards."""

    a: int
    b: int
    root_a: int
    root_b: int
    evidence: SeparationEvidence


@dataclass
class ConstrainedStats:
    """Counters for one constrained clustering run."""

    proposed: int = 0
    merged: int = 0
    already_connected: int = 0
    blocked: int = 0
    contested_events: int = 0
    evaluated: int = 0
    """Unions where both sides had enough pooled evidence to answer."""
    abstained: int = 0
    """Unions where the oracle returned NO_EVIDENCE."""

    @property
    def evaluable_fraction(self) -> float:
        total = self.evaluated + self.abstained
        return self.evaluated / total if total else 0.0


class ConstrainedUnionFind(UnionFind):
    """Union-find with a cannot-link veto and pooled network statistics.

    Two sources of veto, and they compose:

    * a static cannot-link set, added with :meth:`add_cannot_link`, for
      constraints computed ahead of time;
    * a live :class:`SeparationOracle`, consulted with the *current* pooled
      statistics of the two components being merged.

    The oracle path is what makes the repetition requirement structural. A
    component's pooled count is the sum of its members' counts, so a
    constraint cannot fire until enough observations have accumulated on both
    sides - there is no code path where one observation blocks a merge.

    >>> uf = ConstrainedUnionFind(4)
    >>> uf.add_cannot_link(0, 3)
    >>> sorted(uf.cannot_link_pairs())
    [(0, 3)]
    """

    __slots__ = (
        "_oracle",
        "_stats",
        "_cannot_link",
        "_forbidden",
        "_contested",
        "_blocked",
        "_counters",
        "_last_evidence",
    )

    def __init__(self, n: int, oracle: SeparationOracle | None = None) -> None:
        """Create ``n`` singletons, optionally with a separation oracle.

        Args:
            n: Number of nodes.
            oracle: Pooled network evidence. When None the class behaves as
                plain union-find plus any static cannot-links, which is how
                the chain-only baseline is produced from the same code path.
        """
        super().__init__(n)
        self._oracle = oracle
        self._cannot_link: set[tuple[int, int]] = set()
        # Component-level view of the same constraints: current root -> the
        # roots it may not merge with. A cannot-link is a statement about
        # two *components*, so once an endpoint is absorbed the prohibition
        # has to follow its new root or it silently stops applying.
        self._forbidden: dict[int, set[int]] = {}
        self._contested: set[int] = set()
        self._blocked: list[BlockedMerge] = []
        self._counters = ConstrainedStats()
        self._last_evidence: SeparationEvidence | None = None

        # Pooled statistics per component, keyed by current root. Seeded from
        # the oracle's per-address statistics; merged on every union so the
        # pooled centroid of a component costs one vector addition rather
        # than a re-scan of its members.
        self._stats: dict[int, GroupStats] = {}
        if oracle is not None:
            for code, stats in oracle.address_stats.items():
                if 0 <= code < n:
                    self._stats[int(code)] = stats

    # ---- the two methods you are implementing -------------------------
    def union(self, a: int, b: int) -> bool:
        root_a = int(self.find(a))
        root_b = int(self.find(b))

        # Already in the same component: there is no merge to veto.
        if root_a == root_b:
            self._record_outcome(MergeOutcome.ALREADY_CONNECTED)
            return False

        # Ask the static constraints / network oracle whether this merge
        # is permitted.
        evidence = self._evaluate_cannot_link(root_a, root_b)

        if evidence.verdict is Verdict.SEPARATED:
            self._record_blocked(
                a,
                b,
                root_a,
                root_b,
                evidence,
            )
            return False

        # No separation evidence: perform the ordinary union-by-rank.
        #
        # IMPORTANT: keep this equivalent to the base UnionFind behaviour.
        # The surviving root is the one whose rank is larger. On a tie,
        # root_a survives and its rank increases.
        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a

        self._parent[root_b] = root_a

        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1

        # Keep the pooled network statistics attached to the surviving root.
        self._merge_stats(root_a, root_b)

        # A contested component remains contested after absorption.
        self._inherit_contested(root_a, root_b)

        self._record_outcome(MergeOutcome.MERGED)
        return True

    def _evaluate_cannot_link(
        self, root_a: int, root_b: int
    ) -> SeparationEvidence:
        """Decide whether merging these two components is forbidden."""
        # Static cannot-link constraints take precedence over the oracle.
        if self.has_cannot_link(root_a, root_b):
            evidence = SeparationEvidence(
                verdict=Verdict.SEPARATED,
                reason="static cannot-link",
            )
            self._last_evidence = evidence
            return evidence

        # Chain-only mode: no network oracle means no network evidence.
        if self._oracle is None:
            evidence = SeparationEvidence(
                verdict=Verdict.NO_EVIDENCE,
                reason="no oracle",
            )
            self._last_evidence = evidence
            return evidence

        # Compare the pooled evidence accumulated by the two current
        # components. The oracle owns the thresholds and interpretation.
        stats_a = self.pooled_stats(root_a)
        stats_b = self.pooled_stats(root_b)

        evidence = self._oracle.compare(stats_a, stats_b)

        # Keep the reporting counters synchronized with the oracle result.
        self.note_evidence(evidence)

        return evidence

    # ---- written for you ----------------------------------------------

    def add_cannot_link(self, a: int, b: int) -> None:
        """Assert that ``a`` and ``b`` must never share a component."""
        a, b = int(a), int(b)
        self._cannot_link.add((min(a, b), max(a, b)))
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            # Already merged; the contradiction is reported by
            # audit_existing_constraints rather than prevented here.
            return
        self._forbidden.setdefault(root_a, set()).add(root_b)
        self._forbidden.setdefault(root_b, set()).add(root_a)

    def cannot_link_pairs(self) -> set[tuple[int, int]]:
        """The static cannot-link set, normalised so order does not matter."""
        return set(self._cannot_link)

    def has_cannot_link(self, a: int, b: int) -> bool:
        """Whether merging these two nodes' COMPONENTS is forbidden.

        Resolves both arguments to their current roots first. Comparing raw
        node ids would miss every constraint whose endpoints have since been
        absorbed into larger components - which is most of them once
        clustering is under way.
        """
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            key = (min(int(a), int(b)), max(int(a), int(b)))
            return key in self._cannot_link
        return root_b in self._forbidden.get(root_a, ())

    def pooled_stats(self, root: int) -> GroupStats:
        """Pooled network statistics for the component rooted at ``root``."""
        if self._oracle is None:
            return GroupStats.empty(1)
        return self._stats.get(int(root), self._oracle.empty_stats())

    def _merge_component_state(
        self, surviving_root: int, absorbed_root: int
    ) -> None:
        """Fold the absorbed component's state into the survivor.

        Two things move: the pooled network statistics, and the set of roots
        this component may not merge with. Both have to follow the surviving
        root or they stop applying the moment a component is absorbed.
        """
        surviving_root, absorbed_root = int(surviving_root), int(absorbed_root)
        if surviving_root == absorbed_root:
            return

        if self._oracle is not None:
            self._stats[surviving_root] = self.pooled_stats(
                surviving_root
            ).combine(self.pooled_stats(absorbed_root))
            self._stats.pop(absorbed_root, None)

        moved = self._forbidden.pop(absorbed_root, set())
        if moved:
            target = self._forbidden.setdefault(surviving_root, set())
            target |= moved
            # Re-point the other side of each prohibition at the new root.
            for other in moved:
                peers = self._forbidden.get(other)
                if peers is not None:
                    peers.discard(absorbed_root)
                    peers.add(surviving_root)
            target.discard(surviving_root)

    def _merge_stats(self, surviving_root: int, absorbed_root: int) -> None:
        """Compatibility name for :meth:`_merge_component_state`."""
        self._merge_component_state(surviving_root, absorbed_root)

    def _inherit_contested(self, surviving_root: int, absorbed_root: int) -> None:
        """Carry a contested mark onto the surviving root.

        A contested component that gets absorbed must not become clean by
        accident - the contradiction still applies to the addresses inside.
        """
        if int(absorbed_root) in self._contested:
            self._contested.discard(int(absorbed_root))
            self._contested.add(int(surviving_root))

    def _record_outcome(self, outcome: MergeOutcome) -> None:
        self._counters.proposed += 1
        if outcome is MergeOutcome.MERGED:
            self._counters.merged += 1
        elif outcome is MergeOutcome.ALREADY_CONNECTED:
            self._counters.already_connected += 1
        elif outcome is MergeOutcome.CONTESTED:
            self._counters.contested_events += 1

    def _record_blocked(
        self, a: int, b: int, root_a: int, root_b: int, evidence: SeparationEvidence
    ) -> None:
        """Log a refused merge and count it."""
        self._counters.proposed += 1
        self._counters.blocked += 1
        self._blocked.append(
            BlockedMerge(int(a), int(b), int(root_a), int(root_b), evidence)
        )

    def note_evidence(self, evidence: SeparationEvidence) -> None:
        """Count an oracle answer as evaluated or abstained.

        Call this from :meth:`_evaluate_cannot_link` if you want the
        evaluable-fraction reported; it is safe to skip, in which case that
        figure reads zero.
        """
        self._last_evidence = evidence
        if evidence.verdict is Verdict.NO_EVIDENCE:
            self._counters.abstained += 1
        else:
            self._counters.evaluated += 1

    def mark_contested(self, node: int) -> None:
        """Mark the component containing ``node`` as contested."""
        self._contested.add(int(self.find(node)))
        self._counters.contested_events += 1

    def contested_roots(self) -> set[int]:
        """Current roots of components carrying a contradiction."""
        return {int(self.find(root)) for root in self._contested}

    def n_contested_clusters(self) -> int:
        return len(self.contested_roots())

    def blocked_merges(self) -> list[BlockedMerge]:
        return list(self._blocked)

    @property
    def counters(self) -> ConstrainedStats:
        return self._counters

    def audit_existing_constraints(self) -> int:
        """Mark components that violate a static cannot-link, post hoc.

        Run after clustering. A cannot-link whose endpoints ended up in one
        component is a contradiction that could not be prevented - the pair
        was merged transitively before the constraint was reachable. It is
        recorded rather than resolved, because unmerging would mean choosing
        which cryptographically-backed co-spend edge to discard.

        Returns the number of distinct components marked contested.
        """
        marked: set[int] = set()
        for a, b in self._cannot_link:
            if a >= self.n_nodes or b >= self.n_nodes:
                continue
            root_a, root_b = self.find(a), self.find(b)
            if root_a == root_b and root_a not in marked:
                marked.add(root_a)
                self._contested.add(int(root_a))
                self._counters.contested_events += 1
        return len(marked)

    def component_sizes_excluding_contested(self) -> np.ndarray:
        """Component sizes with contested components removed.

        Useful for reporting what remains trustworthy, as distinct from what
        was produced.
        """
        contested = self.contested_roots()
        roots = self.roots()
        keep = ~np.isin(roots, list(contested)) if contested else np.ones(
            roots.size, dtype=bool
        )
        if not keep.any():
            return np.zeros(0, dtype=np.int64)
        counts = np.bincount(roots[keep], minlength=self.n_nodes)
        sizes = counts[counts > 0]
        return np.sort(sizes)[::-1]
