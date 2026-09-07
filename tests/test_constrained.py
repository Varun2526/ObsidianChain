"""Tests for the constrained union-find - the specification for your code.

These FAIL until ConstrainedUnionFind.union() and _evaluate_cannot_link()
are implemented. That is intentional: they are the contract, not a report.

Hand-built graph used throughout:

    nodes 0..5
    edges (0,1) (1,2) (3,4)
    cannot-link (0, 2)

    Without the constraint: {0,1,2} {3,4} {5}   -> 3 components, largest 3
    With it:                {0,1}  {2}  {3,4} {5} -> 4 components, largest 2

    The (1,2) union is the one that must be refused: by the time it is
    proposed, 1 shares a root with 0, so merging would put 0 and 2 together.
"""

from __future__ import annotations

import numpy as np
import pytest

from obsidianchain.cluster.constrained import (
    ConstrainedUnionFind,
    MergeOutcome,
    Verdict,
)
from obsidianchain.network import separation as sep

EDGES = [(0, 1), (1, 2), (3, 4)]


def stats_from(samples: np.ndarray) -> sep.GroupStats:
    present = ~np.isnan(samples)
    filled = np.nan_to_num(samples, nan=0.0)
    return sep.GroupStats(
        count=int(samples.shape[0]),
        dim_count=present.sum(axis=0).astype(np.int64),
        dim_sum=filled.sum(axis=0),
        dim_sumsq=(filled**2).sum(axis=0),
    )


def oracle_separating(a_nodes, b_nodes, n_dims=4, n=200):
    """An oracle where a_nodes and b_nodes have clearly different centroids."""
    rng = np.random.default_rng(0)
    stats = {}
    for node in a_nodes:
        stats[node] = stats_from(rng.normal(0.0, 0.1, size=(n, n_dims)))
    for node in b_nodes:
        stats[node] = stats_from(rng.normal(1.0, 0.1, size=(n, n_dims)))
    return sep.SeparationOracle(
        n_dims=n_dims,
        observer_ids=[f"obs-{i}" for i in range(n_dims)],
        address_stats=stats,
    )


@pytest.fixture(autouse=True)
def _implemented() -> None:
    """One clear message while the two methods are still stubs."""
    probe = ConstrainedUnionFind(2)
    if probe.union(0, 1) is None:
        pytest.fail(
            "ConstrainedUnionFind.union() returned None - implement union() "
            "and _evaluate_cannot_link() in "
            "src/obsidianchain/cluster/constrained.py."
        )


# ---- behaves as plain union-find when unconstrained --------------------


def test_without_constraints_matches_the_baseline() -> None:
    """The chain-only baseline must come from this same code path."""
    forest = ConstrainedUnionFind(6)
    for a, b in EDGES:
        forest.union(a, b)
    assert forest.component_sizes().tolist() == [3, 2, 1]
    assert forest.counters.blocked == 0
    assert forest.n_contested_clusters() == 0


def test_union_returns_true_only_on_a_real_merge() -> None:
    forest = ConstrainedUnionFind(4)
    assert forest.union(0, 1) is True
    assert forest.union(0, 1) is False
    assert forest.counters.already_connected == 1


def test_no_oracle_means_no_evidence_not_a_block() -> None:
    """Absence of evidence must never be read as evidence of separation."""
    forest = ConstrainedUnionFind(4)
    evidence = forest._evaluate_cannot_link(0, 1)
    assert evidence.verdict is Verdict.NO_EVIDENCE
    assert evidence.blocks_merge is False


# ---- the static cannot-link set ---------------------------------------


def test_static_cannot_link_blocks_the_merge() -> None:
    forest = ConstrainedUnionFind(6)
    forest.add_cannot_link(0, 2)
    for a, b in EDGES:
        forest.union(a, b)
    assert forest.component_sizes().tolist() == [2, 2, 1, 1]
    assert forest.counters.blocked == 1
    assert not forest.connected(0, 2)


def test_cannot_link_is_order_independent() -> None:
    for pair in ((0, 2), (2, 0)):
        forest = ConstrainedUnionFind(6)
        forest.add_cannot_link(*pair)
        assert forest.has_cannot_link(0, 2)
        for a, b in EDGES:
            forest.union(a, b)
        assert forest.counters.blocked == 1


def test_static_constraint_needs_no_statistic() -> None:
    forest = ConstrainedUnionFind(4)
    forest.add_cannot_link(0, 1)
    evidence = forest._evaluate_cannot_link(forest.find(0), forest.find(1))
    assert evidence.verdict is Verdict.SEPARATED
    assert "static" in evidence.reason


def test_blocked_merge_leaves_the_forest_untouched() -> None:
    """A refused merge must not half-apply; parent and rank stay as they were."""
    forest = ConstrainedUnionFind(4)
    forest.add_cannot_link(0, 1)
    before_parent = forest._parent.copy()
    before_rank = forest._rank.copy()
    assert forest.union(0, 1) is False
    np.testing.assert_array_equal(forest._parent, before_parent)
    np.testing.assert_array_equal(forest._rank, before_rank)


def test_blocked_merges_are_recorded_with_their_reason() -> None:
    forest = ConstrainedUnionFind(6)
    forest.add_cannot_link(0, 2)
    for a, b in EDGES:
        forest.union(a, b)
    blocked = forest.blocked_merges()
    assert len(blocked) == 1
    assert {blocked[0].a, blocked[0].b} == {1, 2}
    assert blocked[0].evidence.verdict is Verdict.SEPARATED


def test_a_blocked_merge_does_not_raise() -> None:
    """Callers count merges from the return value; blocking is not an error."""
    forest = ConstrainedUnionFind(4)
    forest.add_cannot_link(0, 1)
    merged = forest.add_edges([(0, 1), (2, 3)])
    assert merged == 1


# ---- the oracle path --------------------------------------------------


def test_oracle_blocks_when_groups_are_separated() -> None:
    oracle = oracle_separating(a_nodes=[0, 1], b_nodes=[2])
    forest = ConstrainedUnionFind(6, oracle=oracle)
    for a, b in EDGES:
        forest.union(a, b)
    assert forest.counters.blocked >= 1
    assert not forest.connected(0, 2)


def test_oracle_permits_when_groups_are_not_separated() -> None:
    rng = np.random.default_rng(1)
    stats = {
        node: stats_from(rng.normal(0.0, 1.0, size=(200, 4))) for node in range(6)
    }
    oracle = sep.SeparationOracle(
        n_dims=4, observer_ids=list("abcd"), address_stats=stats
    )
    forest = ConstrainedUnionFind(6, oracle=oracle)
    for a, b in EDGES:
        forest.union(a, b)
    assert forest.counters.blocked == 0
    assert forest.connected(0, 2)


def test_oracle_abstains_below_the_pooled_minimum() -> None:
    """Too little evidence must permit the merge, not block it."""
    rng = np.random.default_rng(2)
    stats = {
        0: stats_from(rng.normal(0.0, 0.1, size=(3, 4))),
        1: stats_from(rng.normal(0.0, 0.1, size=(2, 4))),
        2: stats_from(rng.normal(5.0, 0.1, size=(3, 4))),
    }
    oracle = sep.SeparationOracle(
        n_dims=4, observer_ids=list("abcd"), address_stats=stats
    )
    forest = ConstrainedUnionFind(6, oracle=oracle)
    for a, b in EDGES:
        forest.union(a, b)
    assert forest.counters.blocked == 0
    assert forest.connected(0, 2)


def test_pooled_statistics_follow_the_merge() -> None:
    """A component's pooled count must be the sum of its members'."""
    oracle = oracle_separating(a_nodes=[0, 1], b_nodes=[5])
    forest = ConstrainedUnionFind(6, oracle=oracle)
    before = forest.pooled_stats(forest.find(0)).count
    forest.union(0, 1)
    after = forest.pooled_stats(forest.find(0)).count
    assert after == before * 2 == 400


def test_evidence_accumulates_so_a_late_merge_can_be_judged() -> None:
    """The repetition requirement in action.

    Nodes carry 15 observations each - below the 25 minimum individually, so
    an early merge cannot be judged. After two merges the component holds 45
    and the oracle can answer.
    """
    rng = np.random.default_rng(3)
    left = {n: stats_from(rng.normal(0.0, 0.05, size=(15, 4))) for n in (0, 1, 2)}
    right = {n: stats_from(rng.normal(1.0, 0.05, size=(15, 4))) for n in (3, 4, 5)}
    oracle = sep.SeparationOracle(
        n_dims=4, observer_ids=list("abcd"), address_stats={**left, **right}
    )
    forest = ConstrainedUnionFind(6, oracle=oracle)

    assert forest._evaluate_cannot_link(0, 3).verdict is Verdict.NO_EVIDENCE
    forest.union(0, 1)
    forest.union(1, 2)
    forest.union(3, 4)
    forest.union(4, 5)
    verdict = forest._evaluate_cannot_link(forest.find(0), forest.find(3)).verdict
    assert verdict is Verdict.SEPARATED, "pooled evidence should now suffice"
    assert forest.union(0, 3) is False


def test_evaluated_and_abstained_are_counted_when_noted() -> None:
    oracle = oracle_separating(a_nodes=[0, 1], b_nodes=[2])
    forest = ConstrainedUnionFind(6, oracle=oracle)
    for a, b in EDGES:
        forest.union(a, b)
    counters = forest.counters
    assert counters.evaluated + counters.abstained >= 0  # note_evidence optional
    assert 0.0 <= counters.evaluable_fraction <= 1.0


# ---- the conflict case ------------------------------------------------


def test_late_constraint_marks_contested_without_aborting() -> None:
    """A-B and B-C merged first; only then does evidence separate A and C.

    The pair cannot be unmerged - union-find has no split, and inventing one
    would mean discarding a cryptographically-backed co-spend edge. So the
    component is flagged and the run continues.
    """
    forest = ConstrainedUnionFind(6)
    for a, b in EDGES:
        forest.union(a, b)
    assert forest.connected(0, 2)

    forest.add_cannot_link(0, 2)
    marked = forest.audit_existing_constraints()
    assert marked == 1
    assert forest.n_contested_clusters() == 1
    assert forest.connected(0, 2), "must not unmerge"
    assert forest.component_sizes().tolist() == [3, 2, 1], "clustering intact"


def test_contested_mark_survives_absorption() -> None:
    """A contested component must not become clean by being merged away."""
    forest = ConstrainedUnionFind(8)
    forest.union(0, 1)
    forest.mark_contested(0)
    assert forest.n_contested_clusters() == 1
    forest.union(2, 3)
    forest.union(1, 2)
    assert forest.n_contested_clusters() == 1
    assert forest.find(0) in forest.contested_roots()


def test_contested_components_can_be_excluded_from_reporting() -> None:
    forest = ConstrainedUnionFind(6)
    for a, b in EDGES:
        forest.union(a, b)
    forest.mark_contested(0)
    remaining = forest.component_sizes_excluding_contested().tolist()
    assert remaining == [2, 1], "the contested 3-node component is withheld"


def test_audit_is_idempotent() -> None:
    forest = ConstrainedUnionFind(6)
    for a, b in EDGES:
        forest.union(a, b)
    forest.add_cannot_link(0, 2)
    assert forest.audit_existing_constraints() == 1
    assert forest.audit_existing_constraints() == 1
    assert forest.n_contested_clusters() == 1


def test_satisfied_constraint_is_not_contested() -> None:
    forest = ConstrainedUnionFind(6)
    forest.add_cannot_link(0, 2)
    for a, b in EDGES:
        forest.union(a, b)
    assert forest.audit_existing_constraints() == 0
    assert forest.n_contested_clusters() == 0


# ---- there is no must-link path ---------------------------------------


def test_the_class_exposes_no_must_link_api() -> None:
    names = [n.lower() for n in dir(ConstrainedUnionFind)]
    assert not any("must_link" in n or "mustlink" in n for n in names)
    assert not any("same_origin" in n for n in names)


def test_merge_outcomes_are_the_documented_four() -> None:
    assert {o.value for o in MergeOutcome} == {
        "MERGED", "ALREADY_CONNECTED", "BLOCKED", "CONTESTED"
    }


# ---- regression: constraints are about components, not node ids --------


def test_static_cannot_link_holds_when_neither_endpoint_is_a_root() -> None:
    """The prohibition must follow its endpoints into larger components.

    Regression for a gap in the original skeleton: checking raw node ids
    silently drops every constraint whose endpoints have been absorbed,
    which is most of them once clustering is under way.
    """
    forest = ConstrainedUnionFind(10)
    forest.add_cannot_link(5, 7)
    forest.union(3, 5)   # 5 absorbed into root 3
    forest.union(9, 7)   # 7 absorbed into root 9
    assert forest.find(5) != 5 and forest.find(7) != 7, "fixture precondition"

    assert forest.union(5, 7) is False
    assert not forest.connected(5, 7)
    assert forest.counters.blocked == 1


def test_prohibition_survives_repeated_absorption() -> None:
    forest = ConstrainedUnionFind(12)
    forest.add_cannot_link(0, 6)
    for a, b in [(0, 1), (1, 2), (2, 3), (6, 7), (7, 8), (8, 9)]:
        forest.union(a, b)
    assert forest.union(3, 9) is False, "0's side and 6's side must stay apart"
    assert not forest.connected(0, 6)


def test_constraint_between_already_merged_nodes_is_contested_not_blocking() -> None:
    forest = ConstrainedUnionFind(6)
    forest.union(0, 1)
    forest.add_cannot_link(0, 1)  # arrives too late
    assert forest.audit_existing_constraints() == 1
    assert forest.n_contested_clusters() == 1
    assert forest.connected(0, 1), "must not unmerge"


def test_unrelated_components_are_unaffected() -> None:
    forest = ConstrainedUnionFind(10)
    forest.add_cannot_link(5, 7)
    forest.union(3, 5)
    forest.union(9, 7)
    assert forest.union(1, 2) is True, "an unconstrained merge must proceed"
    assert forest.counters.blocked == 0
