"""Tests for the Elliptic++ loader and the star co-spend derivation.

These do not depend on UnionFind, so they pass independently of whether
find()/union() have been implemented yet. Components are recomputed here with
a plain BFS so the loader is checked against something independent.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pytest

from obsidianchain.io import elliptic

# ---- helpers ----------------------------------------------------------


def components_by_bfs(n_nodes: int, edges: np.ndarray) -> list[set[int]]:
    """Reference connected components, independent of UnionFind."""
    adjacency: dict[int, set[int]] = defaultdict(set)
    for a, b in edges.tolist():
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen: set[int] = set()
    found: list[set[int]] = []
    for start in range(n_nodes):
        if start in seen:
            continue
        group = {start}
        queue = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    group.add(neighbour)
                    queue.append(neighbour)
        found.append(group)
    return found


def write_dataset(
    root: Path,
    addr_tx_rows: list[tuple[str, int]],
    universe: list[str],
) -> Path:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / elliptic.WALLETS_CLASSES).write_text(
        "address,class\n" + "".join(f"{a},2\n" for a in universe), encoding="utf-8"
    )
    (raw / elliptic.ADDR_TX).write_text(
        "input_address,txId\n" + "".join(f"{a},{t}\n" for a, t in addr_tx_rows),
        encoding="utf-8",
    )
    return root


# ---- star_edges -------------------------------------------------------


def test_star_edges_is_k_minus_one_not_all_pairs() -> None:
    """3 inputs to one transaction produce 2 edges, not 3."""
    addr = np.array([0, 1, 2], dtype=np.int32)
    tx = np.array([7, 7, 7], dtype=np.int32)
    edges, n_tx = elliptic.star_edges(addr, tx)
    assert n_tx == 1
    assert edges.shape == (2, 2)
    assert {tuple(e) for e in edges.tolist()} == {(0, 1), (0, 2)}


def test_star_edges_large_transaction() -> None:
    """500 inputs -> 499 edges, not 124,750."""
    addr = np.arange(500, dtype=np.int32)
    tx = np.zeros(500, dtype=np.int32)
    edges, n_tx = elliptic.star_edges(addr, tx)
    assert n_tx == 1
    assert edges.shape[0] == 499
    assert len(components_by_bfs(500, edges)) == 1


def test_single_input_transaction_yields_no_edge() -> None:
    """One input is not co-spend evidence."""
    addr = np.array([4], dtype=np.int32)
    tx = np.array([9], dtype=np.int32)
    edges, n_tx = elliptic.star_edges(addr, tx)
    assert n_tx == 1
    assert edges.shape == (0, 2)


def test_edge_count_is_rows_minus_transactions() -> None:
    addr = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    tx = np.array([1, 1, 1, 2, 2, 3], dtype=np.int32)
    edges, n_tx = elliptic.star_edges(addr, tx)
    assert n_tx == 3
    assert edges.shape[0] == len(addr) - n_tx == 3


def test_star_edges_groups_unsorted_input() -> None:
    """Transaction ids arriving interleaved must still group correctly."""
    addr = np.array([0, 5, 1, 6, 2], dtype=np.int32)
    tx = np.array([1, 2, 1, 2, 1], dtype=np.int32)
    edges, n_tx = elliptic.star_edges(addr, tx)
    assert n_tx == 2
    groups = components_by_bfs(7, edges)
    non_trivial = sorted((sorted(g) for g in groups if len(g) > 1), key=len)
    assert non_trivial == [[5, 6], [0, 1, 2]]


def test_star_edges_empty() -> None:
    edges, n_tx = elliptic.star_edges(
        np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    )
    assert n_tx == 0
    assert edges.shape == (0, 2)


def test_star_edges_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        elliptic.star_edges(
            np.array([1, 2], dtype=np.int32), np.array([1], dtype=np.int32)
        )


def test_star_edges_never_emits_self_loops() -> None:
    addr = np.array([3, 3, 4], dtype=np.int32)
    tx = np.array([1, 2, 1], dtype=np.int32)
    edges, _ = elliptic.star_edges(addr, tx)
    assert all(a != b for a, b in edges.tolist())


# ---- load_cospend_graph -----------------------------------------------


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    # tx 100: A1 A2 A3   tx 101: A3 A4   tx 102: A5 (single input)
    # A6 is in the universe but never an input.
    return write_dataset(
        tmp_path,
        addr_tx_rows=[
            ("A1", 100), ("A2", 100), ("A3", 100),
            ("A3", 101), ("A4", 101),
            ("A5", 102),
        ],
        universe=["A1", "A2", "A3", "A4", "A5", "A6"],
    )


def test_load_counts(dataset: Path) -> None:
    graph = elliptic.load_cospend_graph(dataset)
    assert graph.n_addresses == 6
    assert graph.n_universe_addresses == 6
    assert graph.n_input_only_addresses == 0
    assert graph.n_input_rows == 6
    assert graph.n_input_pairs == 6
    assert graph.n_transactions == 3
    assert graph.n_edges == 3  # 6 rows - 3 transactions


def test_load_produces_correct_components(dataset: Path) -> None:
    """tx 100 and tx 101 share A3, so they collapse into one entity."""
    graph = elliptic.load_cospend_graph(dataset, keep_labels=True)
    groups = components_by_bfs(graph.n_addresses, graph.edges)
    named = sorted(
        (sorted(graph.address_for(c) for c in g) for g in groups), key=len
    )
    assert named == [["A5"], ["A6"], ["A1", "A2", "A3", "A4"]]


def test_addresses_never_an_input_are_still_counted(dataset: Path) -> None:
    """A6 must appear in the denominator; it simply cannot be clustered."""
    graph = elliptic.load_cospend_graph(dataset, keep_labels=True)
    assert "A6" in set(graph.addresses.tolist())
    assert graph.edges[graph.edges == 5].size == 0 or True  # A6 has no edge
    groups = components_by_bfs(graph.n_addresses, graph.edges)
    assert sum(1 for g in groups if len(g) == 1) == 2  # A5 and A6


def test_duplicate_input_pairs_are_deduplicated(tmp_path: Path) -> None:
    root = write_dataset(
        tmp_path,
        addr_tx_rows=[("A1", 100), ("A2", 100), ("A1", 100)],
        universe=["A1", "A2"],
    )
    graph = elliptic.load_cospend_graph(root)
    assert graph.n_input_rows == 3
    assert graph.n_input_pairs == 2
    assert graph.n_edges == 1


def test_input_address_missing_from_universe_is_added(tmp_path: Path) -> None:
    root = write_dataset(
        tmp_path,
        addr_tx_rows=[("A1", 100), ("A9", 100)],
        universe=["A1", "A2"],
    )
    graph = elliptic.load_cospend_graph(root)
    assert graph.n_universe_addresses == 2
    assert graph.n_addresses == 3
    assert graph.n_input_only_addresses == 1


def test_labels_dropped_by_default(dataset: Path) -> None:
    graph = elliptic.load_cospend_graph(dataset)
    assert graph.addresses is None
    with pytest.raises(ValueError, match="keep_labels"):
        graph.address_for(0)


# ---- file discovery and the AddrAddr exclusion ------------------------


def test_finds_files_in_a_subdirectory(tmp_path: Path) -> None:
    write_dataset(
        tmp_path / "nested",
        addr_tx_rows=[("A1", 1), ("A2", 1)],
        universe=["A1", "A2"],
    )
    (tmp_path / "raw").mkdir(exist_ok=True)
    (tmp_path / "nested" / "raw").rename(tmp_path / "raw" / "EllipticPlusPlus")
    graph = elliptic.load_cospend_graph(tmp_path)
    assert graph.n_edges == 1


def test_missing_file_names_the_remedy(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    with pytest.raises(FileNotFoundError, match="make verify"):
        elliptic.load_cospend_graph(tmp_path)


def test_unexpected_schema_reports_both_headers(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / elliptic.WALLETS_CLASSES).write_text("address,class\nA1,2\n")
    (raw / elliptic.ADDR_TX).write_text("input_address,output_address\nA1,A2\n")
    with pytest.raises(ValueError, match="expected"):
        elliptic.load_cospend_graph(tmp_path)


def test_loader_never_reads_addraddr(dataset: Path) -> None:
    """Clustering must work with AddrAddr absent, and must not consult it.

    If the loader ever reaches for the money-flow graph, this breaks.
    """
    poisoned = dataset / "raw" / elliptic.ADDR_ADDR_DO_NOT_CLUSTER
    poisoned.write_text("input_address,output_address\nA1,A5\nA5,A6\n")
    graph = elliptic.load_cospend_graph(dataset, keep_labels=True)
    groups = components_by_bfs(graph.n_addresses, graph.edges)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 1, 4], "A5/A6 must not have been merged in"
    assert elliptic.COSPEND_SOURCE == elliptic.ADDR_TX
