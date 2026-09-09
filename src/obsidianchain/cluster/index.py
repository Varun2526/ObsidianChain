"""Persist the clustering the API needs, so the API never recomputes it.

Why this exists
---------------
Three things the read-only API needs did not survive any pipeline run:

* **code -> address.** ``evidence_funnel.node_a/node_b``,
  ``phase33_decisions.component_a/component_b`` and
  ``contaminated_clusters.cluster_id`` are all ``pd.factorize`` codes assigned
  inside :func:`obsidianchain.io.elliptic.load_cospend_graph`. That mapping
  lived only in memory, so no evidence row could be turned back into an
  address.
* **address -> cluster.** Nothing wrote ``ClusterRun.roots``. 569,513 clusters
  existed in the terminal report and 68 had a durable row, in
  ``contaminated_clusters.csv``.
* **cluster -> size / representative.** Same gap.

Rebuilding any of it costs a full load and clustering pass - about two seconds
and half a gigabyte - which is fine once and unacceptable per request. This
module writes it once.

What is deliberately NOT written
--------------------------------
The 538,233 singleton addresses. Measured, not assumed: the set of codes
appearing in ``evidence_funnel`` and in ``phase33_decisions`` is **exactly**
the 284,709 addresses in clusters of size greater than one. That is not a
coincidence - a code reaches a proposed union only by being an input to a
multi-input transaction, which is the same thing that puts it in a
non-singleton cluster. Writing the other 538,233 rows would add 23 MB
referenced by nothing.

The cost of that choice is one real ambiguity, and it is recorded rather than
hidden: an address lookup that misses cannot distinguish "singleton" from
"absent from Elliptic++". Both are honestly reported as "not a member of any
multi-address cluster". ``n_universe_addresses`` in the sidecar says how many
addresses exist in total, so the size of the unlisted remainder is at least
visible. Closing the ambiguity means writing the full universe; the exact
cost is in the module docstring above and the decision belongs to whoever
builds ``/api/entity/{id}``.

Cluster identity
----------------
``cluster_id`` is the union-find root code, matching what
:func:`obsidianchain.eval.purity.build_cluster_table` already writes into
``contaminated_clusters.csv``. Choosing a different convention here - the
minimum member code would be the tidier one - would silently break the join
between these artifacts and that file.

Determinism
-----------
Codes come from one ``pd.factorize`` over ``wallets_classes.csv`` followed by
``AddrTx_edgelist.csv``, in file order. Roots come from union-by-rank over
star edges in transaction order. Both are deterministic functions of the input
files, so two runs on the same data produce byte-identical output.
:func:`obsidianchain.cluster.index` is tested for that rather than assumed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain import provenance as prov

#: Filenames under ``<data_root>/processed/``.
ADDRESS_CLUSTERS = "address_clusters.parquet"
CLUSTERS = "clusters.parquet"

#: Bump when the schema changes so a stale artifact is identifiable.
INDEX_VERSION = "1.0.0"

#: The clustering these artifacts describe. Co-spend only - the frozen Phase 1
#: baseline. ``multi-input+change`` produces a different clustering (562,894
#: clusters against 569,513) and would need its own artifact rather than
#: silently replacing this one.
HEURISTICS = "multi-input"

ADDRESS_CLUSTER_COLUMNS = ["code", "address", "cluster_id"]
CLUSTER_COLUMNS = ["cluster_id", "size", "representative_address"]


@dataclass
class IndexTables:
    """The two tables, before provenance stamping."""

    address_clusters: pd.DataFrame
    clusters: pd.DataFrame
    n_universe_addresses: int
    n_clusters_total: int
    n_singletons: int

    @property
    def n_clustered_addresses(self) -> int:
        return int(len(self.address_clusters))

    @property
    def n_non_singleton_clusters(self) -> int:
        return int(len(self.clusters))


def build_tables(graph, roots: np.ndarray) -> IndexTables:
    """Derive both tables from a graph and its computed roots.

    Split from the writer so tests can build tables from a hand-made graph
    without touching the filesystem.
    """
    if graph.addresses is None:
        raise ValueError(
            "cluster index needs address labels; "
            "call load_cospend_graph(keep_labels=True)"
        )
    roots = np.asarray(roots)
    n = int(graph.n_addresses)
    if roots.shape != (n,):
        raise ValueError(f"roots must have shape ({n},), got {roots.shape}")

    addresses = np.asarray(graph.addresses)
    sizes_by_root = np.bincount(roots, minlength=n)

    # Non-singleton membership. Sorted by code so the file is a positional
    # index as well as a keyed one, and so the byte layout is stable.
    clustered = np.flatnonzero(sizes_by_root[roots] > 1)
    address_clusters = pd.DataFrame(
        {
            "code": clustered.astype(np.int32),
            "address": addresses[clustered].astype(str),
            "cluster_id": roots[clustered].astype(np.int32),
        }
    )

    cluster_ids = np.flatnonzero(sizes_by_root > 1)
    clusters = pd.DataFrame(
        {
            "cluster_id": cluster_ids.astype(np.int32),
            "size": sizes_by_root[cluster_ids].astype(np.int32),
            # The root's own address, which is what purity.py writes as
            # representative_address in contaminated_clusters.csv.
            "representative_address": addresses[cluster_ids].astype(str),
        }
    )

    return IndexTables(
        address_clusters=address_clusters,
        clusters=clusters,
        n_universe_addresses=n,
        n_clusters_total=int((sizes_by_root > 0).sum()),
        n_singletons=int((sizes_by_root == 1).sum()),
    )


def chain_inputs(data_root) -> dict[str, str]:
    """Hash the two chain files that determine this index.

    Both, not one. ``AddrTx_edgelist.csv`` fixes edge and transaction
    ordering; ``wallets_classes.csv`` is concatenated FIRST in the single
    ``factorize`` inside ``load_cospend_graph``, so its length and content
    shift every address code - and therefore every ``code`` and
    ``cluster_id`` in these tables.

    Recording both lets the API check that the address index it is joining
    against was built from the same chain as the evidence artifact, rather
    than assuming it.
    """
    from obsidianchain import run_fingerprint as rf
    from obsidianchain.io import elliptic

    return {
        "chain_addr_tx_sha256": rf.sha256_file(
            elliptic.find_dataset_file(elliptic.ADDR_TX, data_root)
        ),
        "chain_universe_sha256": rf.sha256_file(
            elliptic.find_dataset_file(elliptic.WALLETS_CLASSES, data_root)
        ),
        "heuristics": HEURISTICS,
    }


def _provenance(
    tables: IndexTables, note: str, inputs: dict[str, str] | None = None
) -> prov.Provenance:
    """Phase 4.1 provenance for a chain-only derived artifact.

    ``synthetic_network`` is None, not False: these tables are derived from
    the Elliptic++ chain alone and no announcement is involved, so the
    question does not arise and the column is omitted rather than answered.
    """
    return prov.Provenance(
        provenance_type=prov.ProvenanceType.PRODUCTION,
        dataset_id="elliptic++/cospend",
        synthetic_network=None,
        inputs=inputs,
        notes=(
            note,
            "Co-spend clustering under the common-input-ownership heuristic. "
            "A cluster is a candidate entity, not an established one.",
            f"cluster_index_version={INDEX_VERSION}",
            f"heuristics={HEURISTICS}",
            f"n_universe_addresses={tables.n_universe_addresses}",
            f"n_clusters_total={tables.n_clusters_total}",
            f"n_singletons={tables.n_singletons}",
            "Singleton addresses are deliberately absent: no evidence "
            "artifact references them. A lookup that misses cannot "
            "distinguish a singleton from an address outside the dataset.",
        ),
    )


def write_tables(
    tables: IndexTables, processed_root, inputs: dict[str, str] | None = None
) -> dict:
    """Write both tables with Phase 4.1 provenance. Returns a summary.

    ``inputs`` are the chain hashes from :func:`chain_inputs`. Optional so a
    test can build tables from a hand-made graph with no files behind them.
    """
    processed_root = Path(processed_root)
    address_path = processed_root / ADDRESS_CLUSTERS
    cluster_path = processed_root / CLUSTERS

    prov.write_frame(
        tables.address_clusters,
        address_path,
        _provenance(
            tables,
            "Address-to-cluster index for the read-only API. "
            "Closes the code/address gap left by in-memory factorize codes.",
            inputs,
        ),
    )
    prov.write_frame(
        tables.clusters,
        cluster_path,
        _provenance(
            tables,
            "Cluster summary for the read-only API: size and representative "
            "address per non-singleton cluster.",
            inputs,
        ),
    )
    return {
        "address_clusters": {
            "path": str(address_path),
            "rows": tables.n_clustered_addresses,
            "sha256": _sha256(address_path),
        },
        "clusters": {
            "path": str(cluster_path),
            "rows": tables.n_non_singleton_clusters,
            "sha256": _sha256(cluster_path),
        },
        "n_universe_addresses": tables.n_universe_addresses,
        "n_clusters_total": tables.n_clusters_total,
        "n_singletons": tables.n_singletons,
    }


def build(data_root, processed_root=None) -> dict:
    """Load, cluster once, and persist both tables.

    This runs the same co-spend clustering that ``make run ARGS="cospend"``
    runs and then discards. That duplication is deliberate: persisting from
    inside ``cospend`` would change an existing command's side effects, and
    changing the pipeline silently is not something this should do on its own.
    """
    from obsidianchain.cluster.pipeline import run_clustering
    from obsidianchain.io import elliptic

    data_root = Path(data_root)
    processed_root = (
        Path(processed_root) if processed_root is not None
        else data_root / "processed"
    )
    graph = elliptic.load_cospend_graph(data_root, keep_labels=True)
    run = run_clustering(graph, label=HEURISTICS)
    tables = build_tables(graph, run.roots)
    summary = write_tables(tables, processed_root, chain_inputs(data_root))
    summary["largest_cluster"] = int(run.largest)
    summary["coverage"] = float(run.coverage)
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---- read side, for the API and for tests ------------------------------


def load_address_clusters(processed_root) -> pd.DataFrame:
    """Read the address index. Read-only; never builds."""
    path = Path(processed_root) / ADDRESS_CLUSTERS
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run 'make run ARGS=\"build-cluster-index\"' "
            f"first; the API must not build it on demand."
        )
    return pd.read_parquet(path)


def load_clusters(processed_root) -> pd.DataFrame:
    """Read the cluster summary. Read-only; never builds."""
    path = Path(processed_root) / CLUSTERS
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run 'make run ARGS=\"build-cluster-index\"' "
            f"first; the API must not build it on demand."
        )
    return pd.read_parquet(path)


def format_summary(summary: dict) -> str:
    """Terminal report for the CLI."""
    width = 74
    out = ["=" * width, "OBSIDIANCHAIN - CLUSTER INDEX FOR THE READ-ONLY API", "=" * width]
    add = out.append
    add("  Co-spend clustering under common-input-ownership. A cluster is a")
    add("  candidate entity, not an established one.")
    add("")
    add(f"  address rows (clusters of size > 1)  "
        f"{summary['address_clusters']['rows']:>12,}")
    add(f"  clusters (size > 1)                  "
        f"{summary['clusters']['rows']:>12,}")
    add(f"  largest cluster                      {summary['largest_cluster']:>12,}")
    add(f"  coverage                             {summary['coverage']:>11.2f}%")
    add("")
    add(f"  universe addresses                   "
        f"{summary['n_universe_addresses']:>12,}")
    add(f"  clusters incl. singletons            "
        f"{summary['n_clusters_total']:>12,}")
    add(f"  singletons (NOT written)             {summary['n_singletons']:>12,}")
    add("  Singletons are referenced by no evidence artifact. A lookup that")
    add("  misses cannot tell a singleton from an address outside the dataset.")
    add("")
    add(f"  {ADDRESS_CLUSTERS}")
    add(f"    sha256 {summary['address_clusters']['sha256'][:32]}...")
    add(f"  {CLUSTERS}")
    add(f"    sha256 {summary['clusters']['sha256'][:32]}...")
    add("=" * width)
    return "\n".join(out)
