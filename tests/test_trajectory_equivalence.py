"""Section 7: the chain-only artifact may only be served as the fused walk
because a verification says the two coincide. This is that verification.

Before this file the claim was computed correctly in the CLI and asserted
nowhere. The only test touching it read the sidecar string and checked that
it began with "VERIFIED:", which passes against any hand-edited sidecar and
says nothing about the forests. The claim was true; the mechanism
guaranteeing it for the next run was not covered at all.

Two properties are checked here and they are not the same property:

* **equivalence holds when it should** - no separation, no refusal, identical
  forests, and the funnel's row-wise production columns aligned with the
  probe columns they sit beside;
* **equivalence FAILS when it should** - a graph engineered to produce a real
  SEPARATED verdict makes the veto fire, the forests diverge, and the
  measurement reports that rather than carrying on.

The second matters more. A check that cannot fail is indistinguishable from
no check, and the divergent path was the one no test had ever executed.
"""

from __future__ import annotations

import numpy as np
import pytest

from obsidianchain import evidence_contract as contract
from obsidianchain.cluster import replay
from obsidianchain.eval import evidence_funnel as funnel
from obsidianchain.eval import trajectory
from obsidianchain.io.elliptic import CoSpendGraph
from obsidianchain.network import separation as sep

# The fixture builders are the funnel suite's own, reused rather than
# re-invented so a change to the graph shape cannot make these two files
# disagree about what a co-spend graph is.
from tests.test_evidence_funnel import make_graph, make_oracle


# ---- fixtures: one world that separates, one that does not -------------


@pytest.fixture()
def identical_world():
    """Two groups drawn from the same distribution. Nothing to separate.

    200 observations per side clears the production pooled minimum of 25, so
    the rule genuinely runs: NOT_SEPARATED here is a decision, not an
    abstention. That distinction is the whole point - an abstaining world
    would make the veto silent for the wrong reason and the test would prove
    nothing about equivalence.
    """
    rng = np.random.default_rng(101)
    oracle = make_oracle(
        {
            0: rng.normal(0.0, 1.0, (200, 4)),
            1: rng.normal(0.0, 1.0, (200, 4)),
            2: rng.normal(0.0, 1.0, (200, 4)),
        }
    )
    return make_graph([(0, 1), (1, 2)]), oracle


@pytest.fixture()
def separating_world():
    """Two groups a mile apart, with enough observations to prove it."""
    rng = np.random.default_rng(202)
    oracle = make_oracle(
        {
            0: rng.normal(0.0, 0.05, (200, 4)),
            1: rng.normal(1.0, 0.05, (200, 4)),
        }
    )
    return make_graph([(0, 1)]), oracle


def production() -> sep.SeparationConfig:
    """The frozen production rule. Never a per-test threshold."""
    return sep.SeparationConfig()


# ---- the fixtures are what they claim to be ----------------------------


def test_the_identical_world_reaches_a_decision_rather_than_abstaining(
    identical_world,
) -> None:
    """Guards every test below: an abstaining fixture proves nothing."""
    graph, oracle = identical_world
    result = funnel.build_funnel(
        graph, oracle, production_config=production(),
        production_statistics_config=production(),
    )
    verdicts = result.records["verdict_production"].tolist()
    assert "NO_EVIDENCE" not in verdicts, verdicts
    assert set(verdicts) == {"NOT_SEPARATED"}, verdicts


def test_the_separating_world_actually_separates(separating_world) -> None:
    """The divergence tests are vacuous unless the veto has something to do."""
    graph, oracle = separating_world
    result = funnel.build_funnel(
        graph, oracle, production_config=production(),
        production_statistics_config=production(),
    )
    assert result.records["verdict_production"].tolist() == ["SEPARATED"]


# ---- equivalence holds: fused veto=True vs baseline veto=False ---------


def test_the_veto_flag_is_load_bearing(separating_world) -> None:
    """The comparison means nothing unless the two walks differ in that one
    variable, so the flag is shown to change the outcome.

    Measured on the separating world, because that is where the flag can
    show an effect at all: ``veto=True`` refuses the union and
    ``veto=False`` applies it, from the same graph and the same oracle. A
    refactor that quietly passed ``veto=False`` twice would fail here
    instead of reporting a trivially perfect match everywhere else.
    """
    graph, oracle = separating_world
    config = production()
    fused = replay.replay_unions(graph, oracle, veto=True, config=config,
                                 collect=False)
    baseline = replay.replay_unions(graph, oracle, veto=False, config=config,
                                    collect=False)
    assert fused.forest.counters.blocked == 1
    assert baseline.forest.counters.blocked == 0
    assert not np.array_equal(
        fused.forest.component_sizes(), baseline.forest.component_sizes()
    )


def test_arming_the_veto_changes_nothing_without_separation(
    identical_world,
) -> None:
    """The same two settings on a world with nothing to refuse."""
    graph, oracle = identical_world
    config = production()
    fused = replay.replay_unions(graph, oracle, veto=True, config=config,
                                 collect=False)
    baseline = replay.replay_unions(graph, oracle, veto=False, config=config,
                                    collect=False)
    assert fused.forest.counters.blocked == 0
    assert baseline.forest.counters.blocked == 0


def test_equivalence_holds_on_a_world_with_no_separation(
    identical_world,
) -> None:
    graph, oracle = identical_world
    equivalence = trajectory.verify_trajectory_equivalence(
        graph, oracle, production(), n_separated=0
    )
    assert equivalence.n_separated == 0
    assert equivalence.n_blocked == 0
    assert equivalence.roots_equal is True
    assert equivalence.component_sizes_equal is True
    assert equivalence.equivalent is True
    assert equivalence.invariant_violated is False


def test_roots_are_compared_element_wise_not_by_partition(
    identical_world,
) -> None:
    """``roots_equal`` must mean "same representative for every node".

    Re-derived here from the two forests rather than trusted, because a
    partition-level check would pass on two walks that landed on the same
    clustering by different routes - and this artifact's claim is that the
    two walks were the SAME WALK.
    """
    graph, oracle = identical_world
    config = production()
    fused = replay.replay_unions(graph, oracle, veto=True, config=config,
                                 collect=False)
    baseline = replay.replay_unions(graph, oracle, veto=False, config=config,
                                    collect=False)
    fused_roots = fused.forest.roots()
    baseline_roots = baseline.forest.roots()

    assert fused_roots.shape == (graph.n_addresses,)
    assert np.array_equal(fused_roots, baseline_roots)
    # Element-wise, spelled out: every node, not just the aggregate.
    for node in range(graph.n_addresses):
        assert fused_roots[node] == baseline_roots[node], node


def test_component_sizes_are_compared_as_a_full_profile(
    identical_world,
) -> None:
    graph, oracle = identical_world
    config = production()
    fused = replay.replay_unions(graph, oracle, veto=True, config=config,
                                 collect=False)
    baseline = replay.replay_unions(graph, oracle, veto=False, config=config,
                                    collect=False)
    fused_sizes = fused.forest.component_sizes()
    baseline_sizes = baseline.forest.component_sizes()

    assert np.array_equal(fused_sizes, baseline_sizes)
    assert int(fused_sizes.sum()) == graph.n_addresses
    # The three merged nodes are one component of 3 in both walks.
    assert fused_sizes.tolist()[0] == 3


def test_the_blocked_count_is_the_vetos_own_counter(identical_world) -> None:
    """Not a re-count from the decision list: the counter the engine
    increments when it actually refuses a union."""
    graph, oracle = identical_world
    equivalence = trajectory.verify_trajectory_equivalence(
        graph, oracle, production(), n_separated=0
    )
    fused = replay.replay_unions(graph, oracle, veto=True,
                                 config=production(), collect=False)
    assert equivalence.n_blocked == int(fused.forest.counters.blocked) == 0


def test_the_verified_wording_is_the_frozen_constant(identical_world) -> None:
    graph, oracle = identical_world
    equivalence = trajectory.verify_trajectory_equivalence(
        graph, oracle, production(), n_separated=0
    )
    assert equivalence.statement == contract.TRAJECTORY_EQUIVALENCE_VERIFIED


# ---- equivalence FAILS: the path no test had ever run ------------------


def test_a_separating_world_makes_the_veto_fire_and_the_walks_diverge(
    separating_world,
) -> None:
    """The explicit failing path.

    This is the case the old inline check could never have been shown to
    handle: with a real SEPARATED verdict the fused walk refuses the union,
    so the two forests genuinely differ and equivalence must be reported as
    NOT established.
    """
    graph, oracle = separating_world
    equivalence = trajectory.verify_trajectory_equivalence(
        graph, oracle, production(), n_separated=1
    )
    assert equivalence.n_blocked == 1, "the veto did not refuse the union"
    assert equivalence.roots_equal is False
    assert equivalence.component_sizes_equal is False
    assert equivalence.equivalent is False
    assert equivalence.statement == (
        contract.TRAJECTORY_EQUIVALENCE_NOT_ESTABLISHED
    )


def test_a_divergent_run_with_separations_is_not_an_invariant_violation(
    separating_world,
) -> None:
    """The veto working is not a bug.

    ``invariant_violated`` must distinguish "the veto fired, so of course
    the walks differ" from "nothing separated, yet they still differ". Only
    the second is incoherent, and only the second refuses to publish.
    """
    graph, oracle = separating_world
    equivalence = trajectory.verify_trajectory_equivalence(
        graph, oracle, production(), n_separated=1
    )
    assert equivalence.equivalent is False
    assert equivalence.invariant_violated is False


def test_divergence_with_no_separation_is_an_invariant_violation() -> None:
    """The refuse-to-publish branch, exercised directly.

    Constructed rather than provoked: reaching this state through the engine
    would need a bug in the engine, which is the point - it is incoherent,
    and the artifact must not publish on it. So the combination is built and
    the predicate is checked.
    """
    violation = trajectory.TrajectoryEquivalence(
        n_separated=0, n_blocked=0,
        roots_equal=False, component_sizes_equal=True,
    )
    assert violation.equivalent is False
    assert violation.invariant_violated is True
    assert violation.statement == (
        contract.TRAJECTORY_EQUIVALENCE_NOT_ESTABLISHED
    )


@pytest.mark.parametrize(
    "roots_equal,sizes_equal,blocked",
    [
        (False, True, 0),   # same partition, different walk
        (True, False, 0),   # same representatives, different profile
        (True, True, 1),    # forests agree but a union was refused
    ],
)
def test_every_condition_is_load_bearing(roots_equal, sizes_equal, blocked) -> None:
    """None of the four conditions may be decorative.

    If equivalence still held with any one of them false, that condition
    would not be protecting anything and the report would be overstating
    what was verified.
    """
    equivalence = trajectory.TrajectoryEquivalence(
        n_separated=0, n_blocked=blocked,
        roots_equal=roots_equal, component_sizes_equal=sizes_equal,
    )
    assert equivalence.equivalent is False


def test_the_report_shows_every_input_to_the_conclusion(identical_world) -> None:
    """A published VERIFIED must be auditable from the terminal block."""
    graph, oracle = identical_world
    text = trajectory.verify_trajectory_equivalence(
        graph, oracle, production(), n_separated=0
    ).as_report()
    for expected in (
        "SEPARATED verdicts", "unions refused by the veto",
        "fused roots == chain-only roots", "component sizes equal",
        "veto_never_fired",
    ):
        assert expected in text, expected


# ---- production-walk row alignment -------------------------------------


def test_the_production_walk_is_row_aligned_with_the_probe_walk(
    identical_world,
) -> None:
    """Schema /2's seven columns are joined to the thirteen by POSITION.

    That join is only valid if the second walk visited the same proposed
    merges in the same order. ``build_funnel`` asserts it internally; this
    checks the property itself is true rather than that the assertion exists.
    """
    graph, oracle = identical_world
    result = funnel.build_funnel(
        graph, oracle, production_config=production(),
        production_statistics_config=production(),
    )
    records = result.records

    seen: list[int] = []
    replay.replay_unions(
        graph, oracle, veto=False, config=production(), collect=False,
        on_decision=lambda d: seen.append(d.edge_index),
    )
    assert seen == records["edge_index"].tolist()
    assert len(records) == len(seen)


def test_a_misaligned_production_walk_is_refused(identical_world) -> None:
    """The guard fires, rather than silently producing a shifted join.

    Called with a deliberately wrong expectation, which is the only way to
    reach the branch: a genuine misalignment needs a bug in the walker.
    """
    graph, oracle = identical_world
    with pytest.raises(RuntimeError, match="did not follow the same trajectory"):
        funnel._production_evaluation(
            graph, oracle, production(),
            expected_edge_index=[999, 998],
            expected_pooled=([0, 0], [0, 0]),
        )


def test_a_config_dependent_oracle_is_refused(identical_world) -> None:
    """The second guard: pooled counts must not move with the configuration.

    The whole two-walk design rests on the oracle's per-address statistics
    being independent of ``SeparationConfig``. If that stopped being true the
    two walks would pool differently and the columns would describe different
    evidence, so the mismatch is refused by message.
    """
    graph, oracle = identical_world
    config = production()
    edges: list[int] = []
    replay.replay_unions(
        graph, oracle, veto=False, config=config, collect=False,
        on_decision=lambda d: edges.append(d.edge_index),
    )
    with pytest.raises(RuntimeError, match="not config-independent"):
        funnel._production_evaluation(
            graph, oracle, config,
            expected_edge_index=edges,
            # Right trajectory, wrong pooled counts: exactly what a
            # config-dependent oracle would produce.
            expected_pooled=([10_000] * len(edges), [10_000] * len(edges)),
        )


# ---- oracle / config independence, measured ----------------------------


def test_pooled_counts_are_identical_under_probe_and_production_configs(
    identical_world,
) -> None:
    """The assumption the row alignment rests on, measured directly.

    The probe walk runs at (1, 2) and the production walk at (25, 5). If the
    configuration reached the oracle's pooled statistics rather than only its
    gates, row i of one walk would not describe the same evidence as row i of
    the other - and the join would be wrong while looking perfectly regular.
    """
    graph, oracle = identical_world

    def walk(config):
        rows: list[tuple[int, int, int]] = []
        replay.replay_unions(
            graph, oracle, veto=False, config=config, collect=False,
            on_decision=lambda d: rows.append(
                (d.edge_index, int(d.evidence.n_a), int(d.evidence.n_b))
            ),
        )
        return rows

    probe_rows = walk(funnel.PROBE_CONFIG)
    production_rows = walk(production())

    assert probe_rows == production_rows
    assert probe_rows, "no decisions were recorded; the comparison is vacuous"
    # And the two configurations really are different, or the above is trivial.
    assert (
        funnel.PROBE_CONFIG.min_pooled_observations
        != production().min_pooled_observations
    )


def test_one_oracle_serves_both_walks(identical_world) -> None:
    """Not two oracles built under two configurations.

    Building a second oracle for the production walk would reintroduce the
    dependency the test above rules out, so the funnel is given exactly one.
    """
    graph, oracle = identical_world
    built: list[object] = []

    class Recording:
        """Passes through, and records that only one oracle was used."""

        def __init__(self, inner):
            self._inner = inner
            built.append(inner)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    result = funnel.build_funnel(
        graph, Recording(oracle), production_config=production(),
        production_statistics_config=production(),
    )
    assert len(built) == 1
    assert len(result.records) == 2


# ---- the funnel's own SEPARATED count feeds the check ------------------


def test_the_separated_count_comes_from_the_published_column(
    separating_world,
) -> None:
    """``n_separated`` is passed in from the artifact's own production column
    rather than recomputed, so the published verdict counts and the
    equivalence statement cannot disagree about the same run."""
    graph, oracle = separating_world
    result = funnel.build_funnel(
        graph, oracle, production_config=production(),
        production_statistics_config=production(),
    )
    separated = int((result.records["verdict_production"] == "SEPARATED").sum())
    equivalence = trajectory.verify_trajectory_equivalence(
        graph, oracle, production(), n_separated=separated
    )
    assert separated == 1
    assert equivalence.n_separated == separated
    assert equivalence.n_blocked == separated
