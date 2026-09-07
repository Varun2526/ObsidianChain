"""Decisions, not edges; and no binary verdict on an unanswerable boundary.

Two evaluator bugs are pinned here. Both produced confident, wrong numbers
rather than crashes, which is why they survived several reports.
"""

from __future__ import annotations

import numpy as np
import pytest

from obsidianchain.eval import phase33


# ---- truth categories --------------------------------------------------


def test_pure_same_entity() -> None:
    assert phase33.classify_truth({5}, {5}) is phase33.TruthCategory.PURE_SAME_ENTITY


def test_pure_cross_entity() -> None:
    assert phase33.classify_truth({5}, {8}) is phase33.TruthCategory.PURE_CROSS_ENTITY


@pytest.mark.parametrize(
    ("a", "b"),
    [({5, 8}, {5}), ({5}, {5, 8}), ({5, 8}, {5, 8}), ({1, 2}, {3, 4})],
)
def test_mixed_entity(a, b) -> None:
    """Either side holding several entities makes the boundary unanswerable."""
    assert phase33.classify_truth(a, b) is phase33.TruthCategory.MIXED_ENTITY


@pytest.mark.parametrize(("a", "b"), [(set(), {5}), ({5}, set()), (set(), set())])
def test_unresolved(a, b) -> None:
    assert phase33.classify_truth(a, b) is phase33.TruthCategory.UNRESOLVED


def test_only_pure_categories_are_binary_scorable() -> None:
    scorable = {c.value for c in phase33.BINARY_CATEGORIES}
    assert scorable == {"PURE_SAME_ENTITY", "PURE_CROSS_ENTITY"}
    for category in phase33.TruthCategory:
        decision = _decision(truth_category=category.value, blocked=True)
        assert decision.binary_scorable is (category.value in scorable)


def _decision(**kwargs):
    base = dict(
        decision_id=0, component_a=1, component_b=2,
        evidence_state="SEPARATED", chi2=30.0, p_value=1e-6,
        effect_size=0.11, pooled_n_a=100, pooled_n_b=50,
    )
    base.update(kwargs)
    return phase33.Decision(**base)


# ---- what counts as a false split -------------------------------------


def test_blocked_pure_same_entity_is_a_false_split() -> None:
    d = _decision(truth_category="PURE_SAME_ENTITY", blocked=True)
    assert d.is_false_split and not d.is_correct_separation


def test_blocked_pure_cross_entity_is_a_correct_separation() -> None:
    d = _decision(truth_category="PURE_CROSS_ENTITY", blocked=True)
    assert d.is_correct_separation and not d.is_false_split


def test_mixed_boundary_is_neither(  ) -> None:
    """The bug this pins: a mixed boundary was scored as both."""
    d = _decision(truth_category="MIXED_ENTITY", blocked=True)
    assert not d.is_false_split
    assert not d.is_correct_separation
    assert not d.binary_scorable


def test_an_allowed_merge_is_never_a_false_split() -> None:
    d = _decision(truth_category="PURE_SAME_ENTITY", blocked=False)
    assert not d.is_false_split


# ---- decisions, not edges ---------------------------------------------


class _Graph:
    def __init__(self, edges, n):
        self.edges = np.array(edges, dtype=np.int32).reshape(-1, 2)
        self.n_addresses = n


def _oracle_blocking(pairs, n_dims=4):
    """An oracle that separates the given address groups."""
    from obsidianchain.network import separation as sep

    rng = np.random.default_rng(0)
    stats = {}
    for group, centre in pairs.items():
        for node in group:
            samples = rng.normal(centre, 0.05, size=(200, n_dims))
            present = ~np.isnan(samples)
            stats[node] = sep.GroupStats(
                count=200,
                dim_count=present.sum(axis=0).astype(np.int64),
                dim_sum=samples.sum(axis=0),
                dim_sumsq=(samples**2).sum(axis=0),
            )
    return sep.SeparationOracle(
        n_dims=n_dims, observer_ids=[f"o{i}" for i in range(n_dims)],
        address_stats=stats,
    )


def test_one_boundary_proposed_many_times_is_one_decision() -> None:
    """THE bug: 166 edges across one boundary became 166 false splits."""
    # Two groups {0,1,2} and {3,4,5}, internally linked, then five edges
    # crossing between them - one boundary, five proposals.
    edges = [(0, 1), (1, 2), (3, 4), (4, 5)]
    edges += [(0, 3), (1, 4), (2, 5), (0, 4), (1, 5)]
    graph = _Graph(edges, 6)
    oracle = _oracle_blocking({(0, 1, 2): 0.0, (3, 4, 5): 1.0})
    entity = np.array([7, 7, 7, 7, 7, 7])  # all one entity

    decisions = phase33.replay_decisions(graph, oracle, entity)
    crossing = [d for d in decisions if d.blocked]
    assert len(crossing) == 1, f"expected one decision, got {len(crossing)}"
    assert crossing[0].proposing_edges == 5, "the other four are provenance"
    assert crossing[0].truth_category == "PURE_SAME_ENTITY"
    assert crossing[0].is_false_split


def test_provenance_does_not_inflate_the_metric() -> None:
    edges = [(0, 1), (2, 3)] + [(0, 2)] * 1 + [(1, 3), (0, 3), (1, 2)]
    graph = _Graph(edges, 4)
    oracle = _oracle_blocking({(0, 1): 0.0, (2, 3): 1.0})
    entity = np.array([1, 1, 1, 1])
    decisions = phase33.replay_decisions(graph, oracle, entity)
    blocked = [d for d in decisions if d.blocked]
    assert len(blocked) == 1
    assert blocked[0].proposing_edges >= 3
    assert sum(d.is_false_split for d in decisions) == 1


def test_decision_records_evidence_and_pooled_counts() -> None:
    graph = _Graph([(0, 1), (2, 3), (0, 2)], 4)
    oracle = _oracle_blocking({(0, 1): 0.0, (2, 3): 1.0})
    entity = np.array([1, 1, 2, 2])
    decisions = phase33.replay_decisions(graph, oracle, entity)
    blocked = [d for d in decisions if d.blocked][0]
    assert blocked.evidence_state == "SEPARATED"
    assert blocked.chi2 > 0 and blocked.p_value < 1e-4
    assert blocked.effect_size > 0
    assert blocked.pooled_n_a == 400 and blocked.pooled_n_b == 400
    assert blocked.truth_category == "PURE_CROSS_ENTITY"
    assert blocked.is_correct_separation


def test_mixed_boundary_is_recorded_as_mixed() -> None:
    graph = _Graph([(0, 1), (2, 3), (0, 2)], 4)
    oracle = _oracle_blocking({(0, 1): 0.0, (2, 3): 1.0})
    entity = np.array([1, 2, 3, 4])  # both sides hold two entities
    decisions = phase33.replay_decisions(graph, oracle, entity)
    blocked = [d for d in decisions if d.blocked][0]
    assert blocked.truth_category == "MIXED_ENTITY"
    assert not blocked.binary_scorable


def test_unresolved_when_truth_is_missing() -> None:
    graph = _Graph([(0, 1), (2, 3), (0, 2)], 4)
    oracle = _oracle_blocking({(0, 1): 0.0, (2, 3): 1.0})
    entity = np.array([-1, -1, 2, 2])  # side A has no truth
    decisions = phase33.replay_decisions(graph, oracle, entity)
    blocked = [d for d in decisions if d.blocked][0]
    assert blocked.truth_category == "UNRESOLVED"
    assert not blocked.binary_scorable


def test_every_proposed_boundary_appears_exactly_once() -> None:
    rng = np.random.default_rng(3)
    edges = rng.integers(0, 30, size=(200, 2))
    edges = edges[edges[:, 0] != edges[:, 1]]
    graph = _Graph(edges, 30)
    oracle = _oracle_blocking({tuple(range(30)): 0.0})
    entity = np.zeros(30, dtype=np.int64)
    decisions = phase33.replay_decisions(graph, oracle, entity)
    keys = [(d.component_a, d.component_b) for d in decisions]
    assert len(keys) == len(set(keys)), "a boundary was recorded twice"


def test_precision_is_nan_when_nothing_is_binary_scorable() -> None:
    outcome = phase33.RegimeOutcome(regime="X", name="x")
    outcome.blocked_by_truth = {"MIXED_ENTITY": 4}
    outcome.binary_scorable_blocked = 0
    assert np.isnan(outcome.decision_precision), (
        "a precision figure must not be invented from unscorable boundaries"
    )


def test_precision_uses_only_binary_scorable_blocks() -> None:
    outcome = phase33.RegimeOutcome(regime="X", name="x")
    outcome.correct_separation_decisions = 3
    outcome.false_split_decisions = 1
    outcome.binary_scorable_blocked = 4
    outcome.blocked_by_truth = {
        "PURE_CROSS_ENTITY": 3, "PURE_SAME_ENTITY": 1, "MIXED_ENTITY": 99,
    }
    assert outcome.decision_precision == pytest.approx(0.75)
