"""Union-find behaviour on a tiny graph whose answer is checkable by hand.

These tests FAIL until find() and union() are implemented in
src/obsidianchain/cluster/unionfind.py. That is intentional - they are the
specification for those two methods.

The fixture graph, 9 addresses (0-8) and 4 transactions:

    tx A inputs {0, 1, 2}   star edges (0,1), (0,2)
    tx B inputs {2, 3}      star edge  (2,3)
    tx C inputs {4, 5}      star edge  (4,5)
    tx D inputs {6}         single input -> no edge
    addresses 7 and 8 never appear as an input at all

tx A and tx B share address 2, so their inputs collapse into one entity:

    {0, 1, 2, 3}   size 4   <- largest
    {4, 5}         size 2
    {6}            size 1
    {7}            size 1
    {8}            size 1

    5 components, largest 4, 3 singletons,
    coverage = (4 + 2) / 9 = 66.67%
"""

from __future__ import annotations

import numpy as np
import pytest

from obsidianchain.cluster.unionfind import UnionFind

N_NODES = 9

# Star edges, exactly as io.elliptic.star_edges would emit them.
COSPEND_EDGES = [(0, 1), (0, 2), (2, 3), (4, 5)]

EXPECTED_SIZES = [4, 2, 1, 1, 1]
EXPECTED_N_COMPONENTS = 5
EXPECTED_LARGEST = 4
EXPECTED_SINGLETONS = 3


@pytest.fixture(autouse=True)
def _stubs_implemented() -> None:
    """Give one clear failure while find()/union() are unimplemented."""
    probe = UnionFind(1)
    if probe.find(0) is None:
        pytest.fail(
            "UnionFind.find() returned None - implement find() and union() in "
            "src/obsidianchain/cluster/unionfind.py."
        )


@pytest.fixture()
def uf() -> UnionFind:
    """The hand-built graph above, already unioned."""
    forest = UnionFind(N_NODES)
    for a, b in COSPEND_EDGES:
        forest.union(a, b)
    return forest


# ---- initial state ----------------------------------------------------


def test_starts_as_all_singletons() -> None:
    forest = UnionFind(N_NODES)
    assert forest.n_components() == N_NODES
    assert forest.largest_component_size() == 1
    assert [forest.find(i) for i in range(N_NODES)] == list(range(N_NODES))


def test_empty_forest() -> None:
    forest = UnionFind(0)
    assert forest.n_components() == 0
    assert forest.largest_component_size() == 0


def test_negative_size_rejected() -> None:
    with pytest.raises(ValueError):
        UnionFind(-1)


# ---- the hand-built graph ---------------------------------------------


def test_component_sizes(uf: UnionFind) -> None:
    assert uf.component_sizes().tolist() == EXPECTED_SIZES


def test_number_of_clusters(uf: UnionFind) -> None:
    assert uf.n_components() == EXPECTED_N_COMPONENTS


def test_largest_cluster_size(uf: UnionFind) -> None:
    """The headline number."""
    assert uf.largest_component_size() == EXPECTED_LARGEST


def test_sizes_sum_to_node_count(uf: UnionFind) -> None:
    assert int(uf.component_sizes().sum()) == N_NODES


def test_singleton_count(uf: UnionFind) -> None:
    sizes = uf.component_sizes()
    assert int((sizes == 1).sum()) == EXPECTED_SINGLETONS


def test_coverage(uf: UnionFind) -> None:
    sizes = uf.component_sizes()
    clustered = int(sizes[sizes > 1].sum())
    assert clustered == 6
    assert round(clustered / N_NODES * 100, 2) == 66.67


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        (0, 1, True),    # direct edge
        (0, 2, True),    # direct edge
        (1, 2, True),    # same transaction, no direct edge between them
        (0, 3, True),    # transitive: tx A and tx B share address 2
        (1, 3, True),    # transitive, two hops
        (4, 5, True),    # separate entity
        (0, 4, False),   # different entities
        (3, 5, False),
        (6, 0, False),   # single-input transaction stays alone
        (7, 8, False),   # never an input, never merged
    ],
)
def test_connectivity(uf: UnionFind, a: int, b: int, same: bool) -> None:
    assert uf.connected(a, b) is same


def test_addresses_never_an_input_stay_singletons(uf: UnionFind) -> None:
    for node in (6, 7, 8):
        assert uf.find(node) == node


# ---- contract of union() ----------------------------------------------


def test_union_returns_true_only_on_a_real_merge() -> None:
    forest = UnionFind(N_NODES)
    assert forest.union(0, 1) is True
    assert forest.union(0, 1) is False, "already connected"
    assert forest.union(1, 0) is False, "order must not matter"


def test_union_is_symmetric() -> None:
    left, right = UnionFind(4), UnionFind(4)
    left.union(0, 1)
    right.union(1, 0)
    assert left.component_sizes().tolist() == right.component_sizes().tolist()


def test_self_union_is_a_no_op() -> None:
    forest = UnionFind(4)
    assert forest.union(2, 2) is False
    assert forest.n_components() == 4


def test_add_edges_counts_merges() -> None:
    forest = UnionFind(N_NODES)
    merged = forest.add_edges(np.array(COSPEND_EDGES, dtype=np.int32))
    assert merged == 4, "all four edges join previously distinct components"
    assert forest.largest_component_size() == EXPECTED_LARGEST


def test_add_edges_ignores_redundant_edges() -> None:
    forest = UnionFind(N_NODES)
    forest.add_edges(COSPEND_EDGES)
    # (1, 2) is already implied by tx A; it must merge nothing new.
    assert forest.add_edges([(1, 2)]) == 0
    assert forest.component_sizes().tolist() == EXPECTED_SIZES


# ---- contract of find() -----------------------------------------------


def test_find_is_idempotent(uf: UnionFind) -> None:
    for node in range(N_NODES):
        root = uf.find(node)
        assert uf.find(root) == root


def test_find_agrees_within_a_component(uf: UnionFind) -> None:
    assert len({uf.find(n) for n in (0, 1, 2, 3)}) == 1
    assert len({uf.find(n) for n in (4, 5)}) == 1


def test_repeated_find_is_stable(uf: UnionFind) -> None:
    """Path compression must not change which nodes are grouped together."""
    before = [uf.find(n) for n in range(N_NODES)]
    for _ in range(3):
        assert [uf.find(n) for n in range(N_NODES)] == before
    assert uf.component_sizes().tolist() == EXPECTED_SIZES


def test_path_compression_flattens_a_chain() -> None:
    """After find(), every node on the path points straight at the root.

    Built as a worst-case chain, which is exactly what union by rank plus
    compression exists to defeat.
    """
    n = 1000
    forest = UnionFind(n)
    for i in range(n - 1):
        forest.union(i, i + 1)

    root = forest.find(0)
    for node in range(n):
        forest.find(node)

    parents = forest._parent  # noqa: SLF001 - asserting the optimisation
    assert all(int(parents[node]) == root for node in range(n)), (
        "every node should point directly at the root after find()"
    )
    assert forest.n_components() == 1
    assert forest.largest_component_size() == n


# ---- star union vs all-pairs ------------------------------------------


def test_union_star_does_k_minus_one_merges() -> None:
    forest = UnionFind(10)
    merged = forest.union_star([0, 1, 2, 3, 4])
    assert merged == 4, "5 inputs need 4 unions, not 10"
    assert forest.largest_component_size() == 5


def test_star_union_matches_all_pairs() -> None:
    """The optimisation must not change the answer.

    A 500-input transaction needs 499 star unions rather than 124,750
    all-pairs unions; both must give one component of 500.
    """
    from itertools import combinations

    members = list(range(500))

    star = UnionFind(600)
    star.union_star(members)

    all_pairs = UnionFind(600)
    for a, b in combinations(members, 2):
        all_pairs.union(a, b)

    assert star.component_sizes().tolist() == all_pairs.component_sizes().tolist()
    assert star.largest_component_size() == 500


def test_union_star_of_one_member_does_nothing() -> None:
    """A single-input transaction is not co-spend evidence."""
    forest = UnionFind(5)
    assert forest.union_star([3]) == 0
    assert forest.n_components() == 5
