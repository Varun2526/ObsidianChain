"""Phase 5 Stage 0: the cluster index the read-only API reads.

Six properties are required of these artifacts and each has its own section:
deterministic generation, complete code/address coverage, complete
address/cluster coverage, provenance, truth isolation, and no modification of
frozen artifacts.

The heavy checks run against the real dataset when it is present and skip
otherwise, so the suite still runs on a machine without the Elliptic++
download. Inside the container ``make test`` mounts ``/data``, so they do run
in the normal loop.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain import provenance as prov
from obsidianchain.cluster import index
from obsidianchain.cluster.pipeline import run_clustering
from obsidianchain.io.elliptic import CoSpendGraph

DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))


# ---- a hand-built graph, so the invariants are checked without the dataset --


def make_graph(n: int, edges: list[tuple[int, int]]) -> CoSpendGraph:
    arr = np.array(edges, dtype=np.int32).reshape(-1, 2)
    return CoSpendGraph(
        n_addresses=n,
        edges=arr,
        edge_tx_ids=np.arange(len(arr)),
        universe_codes=np.arange(n, dtype=np.int32),
        n_transactions=len(arr),
        n_input_rows=len(arr) * 2,
        n_input_pairs=len(arr) * 2,
        n_universe_addresses=n,
        n_input_only_addresses=0,
        addresses=np.array([f"addr{i:04d}" for i in range(n)], dtype=object),
    )


@pytest.fixture()
def graph() -> CoSpendGraph:
    """Ten addresses: one 3-cluster, one 2-cluster, five singletons."""
    return make_graph(10, [(0, 1), (1, 2), (5, 6)])


@pytest.fixture()
def tables(graph) -> index.IndexTables:
    return index.build_tables(graph, run_clustering(graph, "t").roots)


# ---- 1. deterministic generation ---------------------------------------


def test_two_builds_produce_identical_tables(graph) -> None:
    first = index.build_tables(graph, run_clustering(graph, "a").roots)
    second = index.build_tables(graph, run_clustering(graph, "b").roots)
    pd.testing.assert_frame_equal(first.address_clusters, second.address_clusters)
    pd.testing.assert_frame_equal(first.clusters, second.clusters)


def test_two_writes_produce_byte_identical_files(graph, tmp_path) -> None:
    """Byte-identical, not merely equal: a stable artifact can be hashed."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first = index.write_tables(
        index.build_tables(graph, run_clustering(graph, "x").roots), a
    )
    second = index.write_tables(
        index.build_tables(graph, run_clustering(graph, "y").roots), b
    )
    assert first["address_clusters"]["sha256"] == second["address_clusters"]["sha256"]
    assert first["clusters"]["sha256"] == second["clusters"]["sha256"]


def test_rows_are_ordered_by_code(tables) -> None:
    """Order is part of determinism and makes the file a positional index."""
    codes = tables.address_clusters["code"].to_numpy()
    assert np.array_equal(codes, np.sort(codes))
    ids = tables.clusters["cluster_id"].to_numpy()
    assert np.array_equal(ids, np.sort(ids))


@pytest.mark.parametrize("frame,column,dtype", [
    ("address_clusters", "code", np.int32),
    ("address_clusters", "cluster_id", np.int32),
    ("clusters", "cluster_id", np.int32),
    ("clusters", "size", np.int32),
])
def test_integer_dtypes_are_pinned(tables, frame, column, dtype) -> None:
    """A dtype drift would change the bytes without changing the values."""
    assert getattr(tables, frame)[column].dtype == dtype


# ---- 2. complete code/address coverage ---------------------------------


def test_every_clustered_code_is_present(graph, tables) -> None:
    roots = run_clustering(graph, "t").roots
    sizes = np.bincount(roots, minlength=graph.n_addresses)
    expected = set(np.flatnonzero(sizes[roots] > 1).tolist())
    assert set(tables.address_clusters["code"].tolist()) == expected


def test_code_maps_to_the_right_address(graph, tables) -> None:
    for code, address in zip(
        tables.address_clusters["code"], tables.address_clusters["address"]
    ):
        assert address == graph.addresses[code]


def test_singletons_are_absent_and_that_is_recorded(graph, tables) -> None:
    """The deliberate omission, asserted so it cannot become accidental."""
    assert tables.n_singletons == 5
    assert tables.n_clustered_addresses == 5
    assert tables.n_universe_addresses == 10
    present = set(tables.address_clusters["code"].tolist())
    assert present.isdisjoint({3, 4, 7, 8, 9})


def test_the_index_covers_every_code_the_evidence_artifacts_reference() -> None:
    """THE coverage claim, on real data: evidence codes == clustered codes.

    This is the measurement the whole minimality decision rests on. If a
    future evidence artifact referenced a singleton, the index would no
    longer resolve it and this fails rather than the API returning a blank.
    """
    processed = DATA_ROOT / "processed"
    funnel = processed / "evidence_funnel.parquet"
    if not (processed / index.ADDRESS_CLUSTERS).is_file() or not funnel.is_file():
        pytest.skip("cluster index or evidence funnel not built")

    codes = set(index.load_address_clusters(processed)["code"].tolist())
    edges = pd.read_parquet(funnel, columns=["node_a", "node_b"])
    referenced = set(edges["node_a"].tolist()) | set(edges["node_b"].tolist())
    missing = referenced - codes
    assert not missing, (
        f"{len(missing):,} codes referenced by evidence_funnel are absent "
        f"from the index; the API could not resolve them"
    )


# ---- 3. complete address/cluster coverage ------------------------------


def test_every_cluster_id_resolves_to_a_cluster_row(tables) -> None:
    members = set(tables.address_clusters["cluster_id"].tolist())
    summary = set(tables.clusters["cluster_id"].tolist())
    assert members == summary, "a member points at a cluster with no row"


def test_cluster_size_equals_its_member_count(tables) -> None:
    counted = tables.address_clusters.groupby("cluster_id").size()
    declared = tables.clusters.set_index("cluster_id")["size"]
    # Compared as plain dicts: a groupby index is int64 while the column is
    # int32, and that difference is not what this test is about.
    assert {int(k): int(v) for k, v in counted.items()} == {
        int(k): int(v) for k, v in declared.items()
    }


def test_every_cluster_has_more_than_one_member(tables) -> None:
    assert (tables.clusters["size"] > 1).all()


def test_representative_address_is_the_cluster_id_address(graph, tables) -> None:
    """Matches purity.py's convention so contaminated_clusters.csv joins."""
    for cid, rep in zip(
        tables.clusters["cluster_id"], tables.clusters["representative_address"]
    ):
        assert rep == graph.addresses[cid]


def test_the_index_agrees_with_contaminated_clusters() -> None:
    """Cross-check against the artifact that already carries cluster_id."""
    processed = DATA_ROOT / "processed"
    contaminated = processed / "contaminated_clusters.csv"
    if not (processed / index.CLUSTERS).is_file() or not contaminated.is_file():
        pytest.skip("cluster index or contaminated clusters not built")

    clusters = index.load_clusters(processed).set_index("cluster_id")
    known = pd.read_csv(contaminated)
    for row in known.itertuples():
        assert row.cluster_id in clusters.index, (
            f"cluster {row.cluster_id} is in contaminated_clusters.csv but "
            f"not in the index; the two use different cluster identities"
        )
        found = clusters.loc[row.cluster_id]
        assert int(found["size"]) == int(row.size)
        assert found["representative_address"] == row.representative_address


# ---- 4. provenance ------------------------------------------------------


@pytest.mark.parametrize("name", [index.ADDRESS_CLUSTERS, index.CLUSTERS])
def test_written_artifacts_carry_row_and_sidecar_provenance(
    graph, tmp_path, name
) -> None:
    index.write_tables(
        index.build_tables(graph, run_clustering(graph, "t").roots), tmp_path
    )
    path = tmp_path / name
    frame = pd.read_parquet(path)
    assert (frame[prov.PROVENANCE_COLUMN] == "PRODUCTION").all()
    assert frame[prov.PROVENANCE_COLUMN].notna().all()

    meta = json.loads(Path(str(path) + prov.META_SUFFIX).read_text())
    assert meta["provenance_type"] == "PRODUCTION"
    assert meta["dataset_id"] == "elliptic++/cospend"
    assert meta["demo"] is False
    assert str(meta["schema"]).strip()


@pytest.mark.parametrize("name", [index.ADDRESS_CLUSTERS, index.CLUSTERS])
def test_the_sidecar_records_what_was_left_out(graph, tmp_path, name) -> None:
    """The singleton omission must be legible from the artifact alone."""
    index.write_tables(
        index.build_tables(graph, run_clustering(graph, "t").roots), tmp_path
    )
    notes = " ".join(prov.read_meta(tmp_path / name)["notes"])
    assert "n_universe_addresses=10" in notes
    assert "n_singletons=5" in notes
    assert "Singleton addresses are deliberately absent" in notes


@pytest.mark.parametrize("name", [index.ADDRESS_CLUSTERS, index.CLUSTERS])
def test_no_network_column_on_a_chain_only_artifact(graph, tmp_path, name) -> None:
    """Optional means optional: no announcement is involved, so no column."""
    index.write_tables(
        index.build_tables(graph, run_clustering(graph, "t").roots), tmp_path
    )
    frame = pd.read_parquet(tmp_path / name)
    assert prov.SYNTHETIC_NETWORK_COLUMN not in frame.columns
    assert prov.read_meta(tmp_path / name)["synthetic_network"] is None


def test_the_heuristic_is_named_not_implied(graph, tmp_path) -> None:
    """A cluster is a candidate entity; the artifact has to say so."""
    index.write_tables(
        index.build_tables(graph, run_clustering(graph, "t").roots), tmp_path
    )
    notes = " ".join(prov.read_meta(tmp_path / index.CLUSTERS)["notes"])
    assert "common-input-ownership" in notes
    assert "candidate entity, not an established one" in notes
    assert f"heuristics={index.HEURISTICS}" in notes


# ---- 5. truth isolation -------------------------------------------------


TRUTH_COLUMNS = (
    "true_entity_id", "true_origin_id", "entity_id", "entities_a",
    "entities_b", "truth_category", "ground_truth", "broadcaster_flag",
)


@pytest.mark.parametrize("name", [index.ADDRESS_CLUSTERS, index.CLUSTERS])
def test_no_truth_column_reaches_the_artifact(graph, tmp_path, name) -> None:
    index.write_tables(
        index.build_tables(graph, run_clustering(graph, "t").roots), tmp_path
    )
    columns = {c.lower() for c in pd.read_parquet(tmp_path / name).columns}
    assert not any(t in c for c in columns for t in TRUTH_COLUMNS)


@pytest.mark.parametrize("name", [index.ADDRESS_CLUSTERS, index.CLUSTERS])
def test_the_real_artifact_carries_only_the_declared_schema(name) -> None:
    processed = DATA_ROOT / "processed"
    if not (processed / name).is_file():
        pytest.skip("cluster index not built")
    expected = (
        index.ADDRESS_CLUSTER_COLUMNS if name == index.ADDRESS_CLUSTERS
        else index.CLUSTER_COLUMNS
    )
    columns = list(pd.read_parquet(processed / name).columns)
    assert columns == [prov.PROVENANCE_COLUMN] + expected


def test_the_builder_never_reads_truth() -> None:
    """Source-level, matching tests/test_truth_isolation.py's approach."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "obsidianchain" / "cluster" / "index.py"
    ).read_text(encoding="utf-8")
    executable = "\n".join(
        line.split("#", 1)[0] for line in source.splitlines()
        if not line.strip().startswith(('"""', "*"))
    )
    for marker in ("FOR_EVALUATION_ONLY", "network_truth", "worlds_truth"):
        assert marker not in executable, f"index.py references {marker}"


# ---- 6. no modification of frozen artifacts ----------------------------


FROZEN = (
    "processed/network/observations.parquet",
    "processed/network/manifest.json",
    "processed/worlds/A/observations.parquet",
    "processed/worlds/D/observations.parquet",
)


def _digests() -> dict[str, str]:
    out = {}
    for rel in FROZEN:
        path = DATA_ROOT / rel
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            out[rel] = digest.hexdigest()
    return out


def test_building_the_index_does_not_touch_frozen_artifacts(tmp_path) -> None:
    """Build into a scratch directory and prove the frozen set is untouched."""
    if not (DATA_ROOT / "raw").is_dir():
        pytest.skip("Elliptic++ raw data not present")
    before = _digests()
    if not before:
        pytest.skip("no frozen artifacts present")
    try:
        index.build(DATA_ROOT, processed_root=tmp_path)
    except FileNotFoundError as exc:
        pytest.skip(f"Elliptic++ not loadable: {exc}")
    assert _digests() == before, "the builder modified a frozen artifact"


def test_the_builder_writes_only_its_two_artifacts(graph, tmp_path) -> None:
    index.write_tables(
        index.build_tables(graph, run_clustering(graph, "t").roots), tmp_path
    )
    written = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {
        index.ADDRESS_CLUSTERS,
        index.ADDRESS_CLUSTERS + prov.META_SUFFIX,
        index.CLUSTERS,
        index.CLUSTERS + prov.META_SUFFIX,
    }


def test_the_frozen_observations_file_is_never_opened_for_writing() -> None:
    """Source-level: the builder must not name the frozen dataset at all."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "obsidianchain" / "cluster" / "index.py"
    ).read_text(encoding="utf-8")
    executable = "\n".join(
        line.split("#", 1)[0] for line in source.splitlines()
    )
    assert "observations.parquet" not in executable


# ---- the read side never builds ----------------------------------------


@pytest.mark.parametrize("loader", [
    index.load_address_clusters, index.load_clusters
])
def test_loaders_raise_rather_than_building(tmp_path, loader) -> None:
    """The API must never trigger a clustering pass on a cache miss."""
    with pytest.raises(FileNotFoundError, match="build-cluster-index"):
        loader(tmp_path)
