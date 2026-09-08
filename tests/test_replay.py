"""One pooled-evidence loop, and proof it is the loop that was there before.

Phase 4 audit multiplier: the walk-the-edges/pool-the-evidence/ask-the-oracle
loop existed in four places, and two of them reached into a private method.
:mod:`obsidianchain.cluster.replay` is that loop, once.

These are **golden-master** tests. Each holds a verbatim copy of the loop that
was replaced and asserts the shared walker produces the identical result, field
by field, with no tolerance. Asserting only that the walker "looks right"
would not have caught a reordering, an off-by-one in the de-duplication, or a
changed union-by-rank tie-break - and those are exactly the mistakes an
extraction makes.

Comparisons are exact. Every value compared here is either an integer or a
float produced by the *same* arithmetic on the *same* inputs, so bit-identity
is the correct bar and a tolerance would only hide a real divergence. The
Phase 3.1 funnel was additionally checked bit-identical on all 253,429 rows
of the real dataset outside the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from obsidianchain.cluster import replay
from obsidianchain.cluster.constrained import ConstrainedUnionFind
from obsidianchain.cluster.unionfind import UnionFind
from obsidianchain.io.elliptic import CoSpendGraph
from obsidianchain.network import separation as sep


# ---- fixture -----------------------------------------------------------


def stats_from(samples: np.ndarray) -> sep.GroupStats:
    """Per-observer sufficient statistics for a block of arrival vectors."""
    return sep.GroupStats(
        count=int(samples.shape[0]),
        dim_count=np.full(samples.shape[1], samples.shape[0], dtype=np.int64),
        dim_sum=samples.sum(axis=0),
        dim_sumsq=(samples**2).sum(axis=0),
    )


@pytest.fixture()
def graph() -> CoSpendGraph:
    """Eighteen addresses, edges chosen to exercise every branch.

    Addresses 0-6 draw from one arrival population and 7-13 from another, so
    a within-population edge merges and a cross-population edge is refused.
    The edge order then produces all four cases the walker has to handle: a
    plain merge, a redundant edge (endpoints already connected), a refused
    merge, and the same boundary re-proposed by a later edge - which is only
    reachable *because* a merge was refused, and is the case that turned one
    refusal into 166 in reach-stress.
    """
    edges = np.array(
        [
            (0, 1),    # merge
            (1, 2),    # merge
            (5, 6),    # merge
            (7, 8),    # merge
            (2, 5),    # merge -> {0,1,2,5,6}
            (0, 2),    # redundant: already connected
            (2, 7),    # cross-population -> refused
            (5, 8),    # re-proposes the SAME boundary as (2,7)
            (9, 10),   # merge
            (11, 12),  # merge
            (10, 11),  # merge
            (12, 13),  # merge
            # Thin and absent evidence. On the real dataset 251,914 of
            # 253,429 proposed unions land here - the funnel records them
            # with dof 0 and NaN statistics rather than a fabricated number,
            # and that sentinel is the branch these edges exercise.
            (14, 15),  # no evidence at all on either side
            (16, 17),  # one pooled observation each: still no variance
            (14, 16),  # none against one
        ],
        dtype=np.int32,
    )
    return CoSpendGraph(
        n_addresses=18,
        edges=edges,
        edge_tx_ids=np.arange(len(edges)),
        universe_codes=np.arange(18, dtype=np.int32),
        n_transactions=len(edges),
        n_input_rows=len(edges) * 2,
        n_input_pairs=len(edges) * 2,
        n_universe_addresses=18,
        n_input_only_addresses=0,
        addresses=np.array([f"addr{i}" for i in range(18)], dtype=object),
    )


@pytest.fixture()
def oracle() -> sep.SeparationOracle:
    """Two arrival populations, far enough apart that a boundary separates."""
    rng = np.random.default_rng(4321)
    address_stats = {}
    for code in range(14):
        centre = 0.0 if code <= 6 else 1.0
        address_stats[code] = stats_from(
            rng.normal(centre, 0.05, size=(40, 4))
        )
    # 14 and 15 are absent entirely - the oracle returns empty statistics
    # for an address it has never seen. 16 and 17 hold a single observation,
    # which is not enough for a variance estimate however it is pooled.
    for code in (16, 17):
        address_stats[code] = stats_from(rng.normal(0.0, 0.05, size=(1, 4)))
    return sep.SeparationOracle(
        n_dims=4,
        observer_ids=list("abcd"),
        address_stats=address_stats,
        config=sep.SeparationConfig(),
    )


# ---- golden master: the Phase 3.3 / demo loop --------------------------


def reference_decision_loop(graph, oracle):
    """The loop as it stood in eval/phase33.py and demo/runner.py.

    Verbatim apart from the private method having been renamed. Kept here so
    the extraction is checked against what it replaced rather than against
    an idea of what it did.
    """
    forest = ConstrainedUnionFind(graph.n_addresses, oracle=oracle)
    members = {i: [i] for i in range(graph.n_addresses)}
    seen: dict[tuple[int, int], dict] = {}

    for a, b in graph.edges.tolist():
        root_a, root_b = forest.find(a), forest.find(b)
        if root_a == root_b:
            continue
        key = (min(root_a, root_b), max(root_a, root_b))
        if key in seen:
            seen[key]["proposing_edges"] += 1
            forest.union(a, b)
            continue

        evidence = forest.evaluate_cannot_link(root_a, root_b)
        side_a, side_b = list(members[root_a]), list(members[root_b])
        merged = forest.union(a, b)
        seen[key] = {
            "node_a": int(a),
            "node_b": int(b),
            "component_a": int(key[0]),
            "component_b": int(key[1]),
            "size_a": len(side_a),
            "size_b": len(side_b),
            "members_a": tuple(side_a),
            "members_b": tuple(side_b),
            "verdict": evidence.verdict.value,
            "chi2": float(evidence.chi2),
            "p_value": float(evidence.p_value),
            "effect": float(evidence.effect),
            "n_a": int(evidence.n_a),
            "n_b": int(evidence.n_b),
            "merged": merged,
            "proposing_edges": 1,
        }
        if merged:
            survivor = forest.find(a)
            absorbed = root_b if survivor == root_a else root_a
            members[survivor] = side_a + side_b
            if absorbed != survivor:
                members.pop(absorbed, None)
    return forest, list(seen.values())


def as_dicts(decisions):
    return [
        {
            "node_a": d.node_a,
            "node_b": d.node_b,
            "component_a": d.component_a,
            "component_b": d.component_b,
            "size_a": d.size_a,
            "size_b": d.size_b,
            "members_a": d.members_a,
            "members_b": d.members_b,
            "verdict": d.evidence.verdict.value,
            "chi2": float(d.evidence.chi2),
            "p_value": float(d.evidence.p_value),
            "effect": float(d.evidence.effect),
            "n_a": int(d.evidence.n_a),
            "n_b": int(d.evidence.n_b),
            "merged": d.merged,
            "proposing_edges": d.proposing_edges,
        }
        for d in decisions
    ]


def test_the_walker_reproduces_the_decision_loop_exactly(graph, oracle) -> None:
    expected_forest, expected = reference_decision_loop(graph, oracle)
    result = replay.replay_unions(graph, oracle, veto=True, track_members=True)
    assert as_dicts(result.decisions) == expected


def test_the_walker_reproduces_the_same_clustering(graph, oracle) -> None:
    expected_forest, _ = reference_decision_loop(graph, oracle)
    result = replay.replay_unions(graph, oracle, veto=True, track_members=True)
    assert result.forest.roots().tolist() == expected_forest.roots().tolist()
    assert (
        result.forest.component_sizes().tolist()
        == expected_forest.component_sizes().tolist()
    )


def test_the_walker_reproduces_the_counters_including_the_double_count(
    graph, oracle
) -> None:
    """The doubling is preserved deliberately, not overlooked.

    ``evaluate_cannot_link`` is called to record the answer and ``union``
    evaluates again, so each decision counts twice in evaluated/abstained.
    The demonstration's published "evaluated 4 / abstained 30" over 17
    decisions is that doubling. Correcting it here would silently change a
    reported figure, so it is reproduced exactly and left for its own phase.
    """
    expected_forest, expected = reference_decision_loop(graph, oracle)
    result = replay.replay_unions(graph, oracle, veto=True, track_members=True)
    for field in ("evaluated", "abstained", "blocked", "merged", "proposed"):
        assert getattr(result.forest.counters, field) == getattr(
            expected_forest.counters, field
        ), field
    total = result.forest.counters.evaluated + result.forest.counters.abstained
    reproposals = sum(d.proposing_edges - 1 for d in result.decisions)
    assert total == 2 * len(result.decisions) + reproposals, (
        "the double evaluation is load-bearing for reported figures: each "
        "decision is evaluated once to record it and once by union(), and a "
        "re-proposed boundary adds one more from union() alone"
    )


def test_members_are_only_tracked_when_asked(graph, oracle) -> None:
    """O(addresses) memory that only the Phase 3.3 scorer needs."""
    result = replay.replay_unions(graph, oracle, veto=True)
    assert all(d.members_a == () and d.members_b == () for d in result.decisions)


# ---- golden master: the Phase 3.1 funnel loop --------------------------


def reference_funnel_loop(graph, oracle):
    """The loop as it stood in eval/evidence_funnel.py, verbatim.

    Note the plain :class:`UnionFind` and the module's own pooled-statistics
    dictionary - a second implementation of the aggregation
    :class:`ConstrainedUnionFind` performs. That duplication is what the
    extraction removed.
    """
    from obsidianchain.eval.evidence_funnel import PROBE_CONFIG

    forest = UnionFind(graph.n_addresses)
    stats = {int(code): s for code, s in oracle.address_stats.items()}
    empty = oracle.empty_stats()
    sizes = np.ones(graph.n_addresses, dtype=np.int64)
    rows = []
    n_redundant = 0

    for index, (a, b) in enumerate(graph.edges.tolist()):
        root_a, root_b = forest.find(a), forest.find(b)
        if root_a == root_b:
            n_redundant += 1
            continue
        stats_a = stats.get(root_a, empty)
        stats_b = stats.get(root_b, empty)
        row = {
            "edge_index": index,
            "node_a": int(a),
            "node_b": int(b),
            "pooled_a": int(stats_a.count),
            "pooled_b": int(stats_b.count),
            "size_a": int(sizes[root_a]),
            "size_b": int(sizes[root_b]),
        }
        if min(stats_a.count, stats_b.count) >= 2:
            evidence = sep.separation_evidence(stats_a, stats_b, PROBE_CONFIG)
            row |= {
                "dof": int(evidence.dof),
                "chi2": float(evidence.chi2),
                "p_value": float(evidence.p_value),
                "effect": float(evidence.effect),
            }
        else:
            row |= {
                "dof": 0,
                "chi2": np.nan,
                "p_value": np.nan,
                "effect": np.nan,
            }
        rows.append(row)

        forest.union(a, b)
        survivor = forest.find(a)
        absorbed = root_b if survivor == root_a else root_a
        stats[survivor] = stats.get(survivor, empty).combine(
            stats.get(absorbed, empty)
        )
        stats.pop(absorbed, None)
        sizes[survivor] = sizes[root_a] + sizes[root_b]
    return rows, n_redundant


def test_the_funnel_records_are_reproduced_exactly(graph, oracle) -> None:
    """Every field, no tolerance. NaN must land where NaN landed before."""
    from obsidianchain.eval import evidence_funnel

    expected, expected_redundant = reference_funnel_loop(graph, oracle)
    result = evidence_funnel.build_funnel(graph, oracle)

    assert result.n_redundant == expected_redundant
    assert len(result.records) == len(expected)
    for got, want in zip(result.records.to_dict("records"), expected):
        for key, value in want.items():
            actual = got[key]
            if isinstance(value, float) and np.isnan(value):
                assert np.isnan(actual), f"{key}: expected NaN, got {actual}"
            else:
                assert actual == value, key


def test_the_funnel_follows_the_baseline_trajectory(graph, oracle) -> None:
    """No union may be refused, or the thresholds stop being comparable.

    If the funnel blocked merges, a lower pooled minimum would refuse unions
    a higher one allowed and the components would diverge - "how many
    cleared 5" and "how many cleared 25" would then be answers about
    different clusterings.
    """
    from obsidianchain.eval import evidence_funnel

    result = evidence_funnel.build_funnel(graph, oracle)
    baseline = UnionFind(graph.n_addresses)
    baseline.add_edges(graph.edges)
    walk = replay.replay_unions(graph, oracle, veto=False)

    assert result.n_proposed == int(len(result.records))
    assert (
        walk.forest.component_sizes().tolist()
        == baseline.component_sizes().tolist()
    ), "the funnel's trajectory must be the unconstrained baseline's"


def test_without_the_veto_no_merge_is_refused(graph, oracle) -> None:
    result = replay.replay_unions(graph, oracle, veto=False)
    assert all(d.merged for d in result.decisions)
    assert result.forest.counters.blocked == 0


def test_with_the_veto_at_least_one_merge_is_refused(graph, oracle) -> None:
    """Sanity: the fixture must actually exercise the blocking path."""
    result = replay.replay_unions(graph, oracle, veto=True)
    assert any(not d.merged for d in result.decisions), (
        "the fixture no longer reaches a SEPARATED verdict; the veto branch "
        "is untested"
    )


# ---- the loop's own invariants ------------------------------------------


def test_a_re_proposed_boundary_is_one_decision(graph, oracle) -> None:
    """Counting the edges turned one refusal into 166 in reach-stress."""
    result = replay.replay_unions(graph, oracle, veto=True)
    boundaries = [(d.component_a, d.component_b) for d in result.decisions]
    assert len(boundaries) == len(set(boundaries)), "a boundary was double-counted"
    assert any(d.proposing_edges > 1 for d in result.decisions), (
        "the fixture no longer re-proposes a boundary; the de-duplication "
        "branch is untested"
    )
    assert (
        sum(d.proposing_edges for d in result.decisions) + result.n_redundant
        == result.n_edges
    )


def test_redundant_edges_are_counted_not_recorded(graph, oracle) -> None:
    result = replay.replay_unions(graph, oracle, veto=True)
    assert result.n_redundant >= 1, "the fixture must contain a redundant edge"
    assert result.n_edges == int(len(graph.edges))


def test_no_oracle_means_every_answer_is_no_evidence(graph) -> None:
    result = replay.replay_unions(graph, None, veto=True)
    assert all(
        d.evidence.verdict is sep.Verdict.NO_EVIDENCE for d in result.decisions
    )
    assert all(d.merged for d in result.decisions)


def test_collect_false_streams_without_retaining(graph, oracle) -> None:
    seen = []
    result = replay.replay_unions(
        graph, oracle, veto=True, collect=False, on_decision=seen.append
    )
    assert result.decisions == []
    assert len(seen) >= 1


def test_streamed_and_collected_decisions_agree(graph, oracle) -> None:
    seen = []
    streamed = replay.replay_unions(
        graph, oracle, veto=True, collect=False, on_decision=seen.append
    )
    collected = replay.replay_unions(graph, oracle, veto=True)
    assert as_dicts(seen) == as_dicts(collected.decisions)
    assert streamed.n_redundant == collected.n_redundant


# ---- the single-aggregation guarantee -----------------------------------


def test_no_module_keeps_its_own_pooled_statistics_dictionary() -> None:
    """The funnel's private stats dict was the real duplicate aggregation.

    Source-level, because the property is "there is one implementation" and
    that is a property of the source, not of any single run.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "obsidianchain"
    offenders = []
    for path in sorted(src.rglob("*.py")):
        if path.name in ("constrained.py", "separation.py", "replay.py"):
            continue
        text = path.read_text(encoding="utf-8")
        executable = "\n".join(
            line.split("#", 1)[0] for line in text.splitlines()
        )
        if "oracle.address_stats" in executable:
            offenders.append(str(path.relative_to(src)))
    assert offenders == [], (
        f"{offenders} read the oracle's per-address statistics directly; "
        f"pooling belongs to ConstrainedUnionFind alone"
    )


def test_nothing_reaches_into_a_private_evidence_method() -> None:
    from pathlib import Path

    # Assembled at runtime so this file does not match its own scan. The
    # pattern is the attribute CALL, not a docstring mention - the method's
    # history is documented in constrained.py and that prose is fine.
    needle = "." + "_evaluate" + "_cannot_link("

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted((root / "src").rglob("*.py")) + sorted(
        (root / "tests").glob("*.py")
    ):
        text = path.read_text(encoding="utf-8")
        executable = "\n".join(
            line.split("#", 1)[0] for line in text.splitlines()
        )
        if needle in executable:
            offenders.append(str(path.relative_to(root)))
    assert offenders == [], f"{offenders} still call the private method"
