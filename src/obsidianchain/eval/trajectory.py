"""Section 7: is the chain-only artifact also the fused trajectory?

The claim and why it needs verifying
------------------------------------
``evidence_funnel.parquet`` records the **chain-only** walk: ``veto=False``,
every proposed union applied, because all thresholds must describe the same
merge sequence or "how many cleared 5" and "how many cleared 25" become
answers about different clusterings.

But the verdict served beside each row is the *production* verdict, and the
production engine runs with the veto armed. Presenting one as the other is
only honest if the two walks coincide. For this frozen run they do, and the
reason is specific rather than general: no union reached a SEPARATED verdict,
so the veto had nothing to act on.

That argument is sound but it is still an argument. This module measures the
thing instead:

1. no union was SEPARATED,
2. the fused walk refused nothing (``counters.blocked == 0``), and
3. the two forests agree **element-wise** - same root for every node, and the
   same component-size profile.

Step 3 is the one that matters. The first two could both hold while some
other divergence existed, so the forests are compared directly rather than
the conclusion being inferred from the premises.

Roots AND sizes, because they fail differently
----------------------------------------------
``roots()`` is the strict check: it compares the actual representative chosen
for every node, so it detects a divergence even between two forests that
happen to induce the same partition. ``component_sizes()`` is the partition
check: sorted descending, it is invariant to which member became the root, so
it answers "is this the same clustering?" rather than "is this the same walk?"

A run can fail the first and pass the second - union by rank picking a
different representative would do it - and that is worth reporting as a
divergence rather than waving through, because this artifact's whole claim is
that the two walks were the SAME WALK, not merely that they landed somewhere
equivalent. Both are therefore recorded separately.

Not an inference path
---------------------
This module lives under ``eval/`` and is called by the artifact generator, so
the API cannot reach it - ``api/boundary.py`` forbids the whole package and
``tests/test_api_boundary.py`` enforces it at source and import-graph level.
It reads no truth file.
"""

from __future__ import annotations

from dataclasses import dataclass

from obsidianchain import evidence_contract as contract


@dataclass(frozen=True)
class TrajectoryEquivalence:
    """The measured comparison between the fused and chain-only walks."""

    n_separated: int
    """SEPARATED verdicts under the production configuration."""

    n_blocked: int
    """Unions the fused walk refused. The veto's own counter, not a re-count."""

    roots_equal: bool
    """Element-wise: every node has the same representative in both forests."""

    component_sizes_equal: bool
    """The two forests induce the same partition profile."""

    @property
    def equivalent(self) -> bool:
        """Whether the chain-only record may be described as the fused walk.

        All four conditions, not a subset. This is the value persisted as
        ``artifact.veto_never_fired``.
        """
        return (
            self.n_separated == 0
            and self.n_blocked == 0
            and self.roots_equal
            and self.component_sizes_equal
        )

    @property
    def statement(self) -> str:
        """The frozen sidecar wording for this outcome."""
        return (
            contract.TRAJECTORY_EQUIVALENCE_VERIFIED if self.equivalent
            else contract.TRAJECTORY_EQUIVALENCE_NOT_ESTABLISHED
        )

    @property
    def invariant_violated(self) -> bool:
        """No SEPARATED verdict, yet the trajectories still differ.

        This is the incoherent case and the artifact must not be published on
        it. If nothing separated then the veto had nothing to act on, so a
        divergence means an assumption this artifact rests on is wrong - and
        the funnel rows would be describing merges the production engine
        never proposed.

        A run WITH separations that diverges is not a violation: that is the
        veto working, correctly reported by
        :data:`~obsidianchain.evidence_contract.TRAJECTORY_EQUIVALENCE_NOT_ESTABLISHED`.
        """
        return self.n_separated == 0 and not self.equivalent

    def as_report(self) -> str:
        """Terminal block. Every input to the conclusion is shown."""
        lines = [
            "-- trajectory equivalence (verified) " + "-" * 41,
            f"  SEPARATED verdicts               {self.n_separated:>12,}",
            f"  unions refused by the veto       {self.n_blocked:>12,}",
            f"  fused roots == chain-only roots  {str(self.roots_equal):>12}",
            f"  component sizes equal            "
            f"{str(self.component_sizes_equal):>12}",
            f"  veto_never_fired                 {str(self.equivalent):>12}",
        ]
        return "\n".join(lines)


def verify_trajectory_equivalence(
    graph, oracle, config, *, n_separated: int
) -> TrajectoryEquivalence:
    """Walk the graph twice and compare the two forests.

    Args:
        graph: The co-spend graph. Both walks use the same edge order.
        oracle: Pooled network evidence, shared by both walks.
        config: The production separation configuration.
        n_separated: SEPARATED count from the funnel's own production column,
            passed in rather than recomputed here so the published verdict
            counts and this check cannot disagree about the same run.

    Two full replays over 253,429 proposed unions, measured at about four
    seconds each on the frozen dataset. ``collect=False`` on both: only the
    forests are needed and retaining half a million decisions to compare two
    arrays would be the memory mistake the funnel already avoids.
    """
    from obsidianchain.cluster import replay

    import numpy as np

    fused = replay.replay_unions(
        graph, oracle, veto=True, config=config, collect=False
    )
    baseline = replay.replay_unions(
        graph, oracle, veto=False, config=config, collect=False
    )
    # np.array_equal, not `(a == b).all()`. A blocked union leaves the fused
    # forest with MORE components, so these two arrays differ in length on
    # exactly the divergent run this check exists to catch - and elementwise
    # `==` on mismatched lengths does not broadcast to an answer.
    return TrajectoryEquivalence(
        n_separated=int(n_separated),
        n_blocked=int(fused.forest.counters.blocked),
        roots_equal=bool(
            np.array_equal(fused.forest.roots(), baseline.forest.roots())
        ),
        component_sizes_equal=bool(
            np.array_equal(
                fused.forest.component_sizes(),
                baseline.forest.component_sizes(),
            )
        ),
    )
