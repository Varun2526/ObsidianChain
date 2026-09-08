"""Measure co-spend cluster correctness against the Elliptic++ labels.

The argument
------------
``wallets_classes.csv`` labels every address illicit (1), licit (2) or
unknown (3). A single real-world entity is not simultaneously a ransomware
operator and a legitimate exchange, so a cluster holding **both** an illicit
and a licit address has provably merged two distinct entities. We call such a
cluster CONTAMINATED. This is direct evidence of false merges taken from real
labelled data - no controlled wallets, no network layer required.

Handling unknowns
-----------------
557,588 of the 822,942 addresses are unlabelled, and an unlabelled address is
NOT a licit one. Every figure here is computed over labelled addresses only,
and label coverage is reported next to it so it is always visible how much of
a cluster we can actually see. A cluster of 10,000 addresses with 3 labels is
not evidence of anything much, and the output should never let you forget it.

Why the random baseline matters
-------------------------------
A raw contamination rate is uninterpretable on its own. With 265,354 labelled
addresses of which only 5.4% are illicit, a large cluster will contain both
classes by chance alone. The null model here keeps the clusters exactly as
they are, keeps each cluster's labelled count exactly as it is, and shuffles
which labels land where; contamination is then a hypergeometric tail, computed
in closed form rather than sampled. Observed-over-expected is the number that
carries meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

from obsidianchain.cluster.unionfind import UnionFind
from obsidianchain.io import elliptic

ILLICIT = 1
LICIT = 2
UNKNOWN = 3

CLASS_NAMES = {ILLICIT: "illicit", LICIT: "licit", UNKNOWN: "unknown"}

#: Columns of the contaminated-cluster CSV, in order.
CSV_COLUMNS = [
    "cluster_id",
    "size",
    "n_illicit",
    "n_licit",
    "n_unknown",
    "n_labelled",
    "label_coverage",
    "purity",
    "entropy",
    "majority_class",
]


@dataclass
class PurityReport:
    """Everything the summary and the CSV are rendered from."""

    clusters: pd.DataFrame
    """Per-cluster table, one row per cluster, sorted by size descending."""

    n_addresses: int
    n_illicit: int
    n_licit: int
    n_unknown: int

    expected_contaminated_clusters: float
    """Contaminated clusters under label shuffling (hypergeometric)."""

    expected_contaminated_addresses: float

    @property
    def n_labelled(self) -> int:
        return self.n_illicit + self.n_licit

    @property
    def n_clusters(self) -> int:
        return int(len(self.clusters))

    @property
    def contaminated(self) -> pd.DataFrame:
        return self.clusters[self.clusters["contaminated"]]

    @property
    def n_contaminated(self) -> int:
        return int(self.clusters["contaminated"].sum())

    @property
    def observed_over_expected(self) -> float:
        if self.expected_contaminated_clusters <= 0:
            return float("nan")
        return self.n_contaminated / self.expected_contaminated_clusters


# ---- inputs -----------------------------------------------------------


def load_classes_by_code(
    graph: elliptic.CoSpendGraph, data_root: Path | None = None
) -> np.ndarray:
    """Return the class label of every address code as an int8 array.

    Codes never seen in wallets_classes.csv (addresses that appear only in
    AddrTx) are filled with UNKNOWN, which is the honest reading: we have no
    label for them.
    """
    path = elliptic.find_dataset_file(elliptic.WALLETS_CLASSES, data_root)
    frame = pd.read_csv(path, usecols=["class"])
    values = frame["class"].to_numpy()

    if len(values) != len(graph.universe_codes):
        raise ValueError(
            f"{elliptic.WALLETS_CLASSES} has {len(values)} rows but the loader "
            f"mapped {len(graph.universe_codes)}; the file changed underneath us"
        )

    unexpected = set(np.unique(values).tolist()) - set(CLASS_NAMES)
    if unexpected:
        raise ValueError(
            f"unexpected class values in {elliptic.WALLETS_CLASSES}: "
            f"{sorted(unexpected)}; expected {sorted(CLASS_NAMES)}"
        )

    classes = np.full(graph.n_addresses, UNKNOWN, dtype=np.int8)
    classes[graph.universe_codes] = values.astype(np.int8)
    return classes


def cluster_roots(graph: elliptic.CoSpendGraph) -> np.ndarray:
    """Run co-spend union-find and return each address's cluster id."""
    forest = UnionFind(graph.n_addresses)
    forest.add_edges(graph.edges)
    return forest.roots()


# ---- per-cluster aggregation -----------------------------------------


def build_cluster_table(roots: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    """Aggregate addresses into one row per cluster.

    Counting is done with ``np.bincount`` over the root ids, which is a single
    pass per class rather than a group-by over 822k rows.
    """
    if roots.shape != classes.shape:
        raise ValueError("roots and classes must be the same length")

    n = int(roots.size)
    if n == 0:
        return pd.DataFrame(columns=CSV_COLUMNS + ["contaminated"])

    size = np.bincount(roots, minlength=n)
    n_illicit = np.bincount(roots, weights=(classes == ILLICIT), minlength=n)
    n_licit = np.bincount(roots, weights=(classes == LICIT), minlength=n)

    present = size > 0
    cluster_id = np.flatnonzero(present).astype(np.int64)
    size = size[present].astype(np.int64)
    n_illicit = n_illicit[present].astype(np.int64)
    n_licit = n_licit[present].astype(np.int64)
    n_labelled = n_illicit + n_licit
    n_unknown = size - n_labelled

    labelled = n_labelled > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        purity = np.where(
            labelled, np.maximum(n_illicit, n_licit) / n_labelled, np.nan
        )
        p_illicit = np.where(labelled, n_illicit / n_labelled, 0.0)
        p_licit = np.where(labelled, n_licit / n_labelled, 0.0)

    entropy = np.where(labelled, _binary_entropy(p_illicit, p_licit), np.nan)

    majority = np.full(len(cluster_id), "", dtype=object)
    majority[n_illicit > n_licit] = "illicit"
    majority[n_licit > n_illicit] = "licit"
    majority[labelled & (n_illicit == n_licit)] = "tie"

    table = pd.DataFrame(
        {
            "cluster_id": cluster_id,
            "size": size,
            "n_illicit": n_illicit,
            "n_licit": n_licit,
            "n_unknown": n_unknown,
            "n_labelled": n_labelled,
            "label_coverage": n_labelled / size,
            "purity": purity,
            "entropy": entropy,
            "majority_class": majority,
            # Provably wrong: one entity cannot be both.
            "contaminated": (n_illicit > 0) & (n_licit > 0),
        }
    )
    return table.sort_values(
        ["size", "cluster_id"], ascending=[False, True]
    ).reset_index(drop=True)


def _binary_entropy(p_illicit: np.ndarray, p_licit: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits over the illicit/licit split. 0 log 0 = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = []
        for p in (p_illicit, p_licit):
            terms.append(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0))
    return -(terms[0] + terms[1])


# ---- the random baseline ----------------------------------------------


def expected_random_contamination(
    size: np.ndarray, n_labelled: np.ndarray, total_illicit: int, total_licit: int
) -> tuple[float, float]:
    """Expected contamination if labels were shuffled across the same clusters.

    Null model: cluster sizes and each cluster's labelled count are held
    exactly as observed; only *which* labels land in which cluster is
    randomised. For a cluster holding k labelled addresses drawn without
    replacement from a pool of N = I + L,

        P(contaminated) = 1 - C(I,k)/C(N,k) - C(L,k)/C(N,k)

    i.e. one minus the chance of drawing all-illicit or all-licit. Computed
    with log-gamma so the binomial coefficients do not overflow at k ~ 15,000.

    Returns ``(expected_clusters, expected_addresses)``.
    """
    total = int(total_illicit + total_licit)
    k = n_labelled.astype(np.float64)

    if total <= 0:
        return 0.0, 0.0

    p_all_illicit = _hypergeometric_all_one_class(k, total_illicit, total)
    p_all_licit = _hypergeometric_all_one_class(k, total_licit, total)

    p_contaminated = 1.0 - p_all_illicit - p_all_licit
    # k = 0 gives 1 - 1 - 1 = -1; a cluster with no labels cannot be
    # contaminated. k = 1 correctly falls out as exactly 0.
    p_contaminated = np.where(n_labelled >= 2, p_contaminated, 0.0)
    p_contaminated = np.clip(p_contaminated, 0.0, 1.0)

    return float(p_contaminated.sum()), float((size * p_contaminated).sum())


def _hypergeometric_all_one_class(
    k: np.ndarray, class_total: int, pool_total: int
) -> np.ndarray:
    """P(all k draws come from a class of size ``class_total``)."""
    out = np.zeros_like(k, dtype=np.float64)
    feasible = k <= class_total
    kk = k[feasible]
    log_p = _log_choose(class_total, kk) - _log_choose(pool_total, kk)
    out[feasible] = np.exp(log_p)
    return out


def _log_choose(n: float, k: np.ndarray) -> np.ndarray:
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def monte_carlo_contamination(
    n_labelled: np.ndarray,
    total_illicit: int,
    total_licit: int,
    trials: int = 200,
    seed: int = 0,
) -> float:
    """Sampled contamination under the same null model.

    Slower and noisier than :func:`expected_random_contamination`; kept as an
    independent cross-check of the closed form.
    """
    rng = np.random.default_rng(seed)
    pool = np.concatenate(
        [np.ones(total_illicit, dtype=np.int8), np.zeros(total_licit, dtype=np.int8)]
    )
    counts = n_labelled[n_labelled >= 2]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    total_drawn = int(offsets[-1])
    if total_drawn == 0:
        return 0.0
    if total_drawn > pool.size:
        raise ValueError(
            f"cannot draw {total_drawn} labels from a pool of {pool.size}; "
            f"the labelled counts must sum to at most total_illicit + total_licit"
        )

    observed = np.empty(trials, dtype=np.float64)
    for trial in range(trials):
        shuffled = rng.permutation(pool)[:total_drawn]
        sums = np.add.reduceat(shuffled, offsets[:-1])
        observed[trial] = int(((sums > 0) & (sums < counts)).sum())
    return float(observed.mean())


# ---- top level --------------------------------------------------------


def report_for_roots(
    roots: np.ndarray, classes: np.ndarray, n_addresses: int
) -> PurityReport:
    """Score any clustering, given its per-address cluster ids.

    Kept separate from :func:`analyse` so alternative clusterings - change
    detection added, thresholds varied - can be scored without reloading.
    """
    table = build_cluster_table(roots, classes)
    n_illicit = int((classes == ILLICIT).sum())
    n_licit = int((classes == LICIT).sum())
    n_unknown = int((classes == UNKNOWN).sum())

    expected_clusters, expected_addresses = expected_random_contamination(
        table["size"].to_numpy(),
        table["n_labelled"].to_numpy(),
        n_illicit,
        n_licit,
    )
    return PurityReport(
        clusters=table,
        n_addresses=int(n_addresses),
        n_illicit=n_illicit,
        n_licit=n_licit,
        n_unknown=n_unknown,
        expected_contaminated_clusters=expected_clusters,
        expected_contaminated_addresses=expected_addresses,
    )


def analyse(
    data_root: Path | None = None, keep_addresses: bool = True
) -> tuple[PurityReport, elliptic.CoSpendGraph]:
    """Load, cluster on co-spend alone, join labels, and score."""
    graph = elliptic.load_cospend_graph(data_root, keep_labels=keep_addresses)
    classes = load_classes_by_code(graph, data_root)
    roots = cluster_roots(graph)
    return report_for_roots(roots, classes, graph.n_addresses), graph


def write_contaminated_csv(
    report: PurityReport,
    path: Path,
    graph: elliptic.CoSpendGraph | None = None,
    provenance=None,
) -> int:
    """Write the contaminated clusters to ``path``. Returns the row count.

    ``provenance`` attaches the record two ways - a marker column in every
    row and a sibling ``.meta.json``. Optional so the function keeps working
    for callers that only want the frame on disk; every CLI path passes it,
    and ``tests/test_provenance.py`` reads the artifacts back to check.
    """
    frame = report.contaminated.loc[:, CSV_COLUMNS].copy()
    if graph is not None and graph.addresses is not None:
        frame.insert(
            1,
            "representative_address",
            [graph.address_for(c) for c in frame["cluster_id"]],
        )
    frame["label_coverage"] = frame["label_coverage"].round(6)
    frame["purity"] = frame["purity"].round(6)
    frame["entropy"] = frame["entropy"].round(6)
    if provenance is not None:
        from obsidianchain import provenance as prov

        prov.write_frame(frame, path, provenance)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return int(len(frame))


# ---- rendering --------------------------------------------------------

_W = 78
_PURITY_BUCKETS = [
    (1.0, 1.0001, "1.00        (pure)"),
    (0.90, 1.0, "0.90 - 1.00"),
    (0.80, 0.90, "0.80 - 0.90"),
    (0.70, 0.80, "0.70 - 0.80"),
    (0.60, 0.70, "0.60 - 0.70"),
    (0.50, 0.60, "0.50 - 0.60 (worst)"),
]


def _pct(part: float, whole: float) -> str:
    return f"{part / whole * 100:6.2f}%" if whole else "     -"


def _rule(title: str) -> str:
    return f"-- {title} " + "-" * max(0, _W - len(title) - 4)


def format_summary(report: PurityReport, top: int = 10) -> str:
    """Render the whole analysis as a screenshot-friendly report."""
    table = report.clusters
    out: list[str] = []
    add = out.append

    add("=" * _W)
    add("obsidianchain :: cluster purity vs Elliptic++ labels")
    add("=" * _W)
    add("clusters   co-spend over AddrTx_edgelist.csv (grouped on txId)")
    add("labels     wallets_classes.csv  1=illicit  2=licit  3=unknown")
    add("")
    add("A cluster holding BOTH an illicit and a licit address has merged two")
    add("entities that cannot be the same. Unknown is never counted as licit.")
    add("")

    # -- label universe ------------------------------------------------
    add(_rule("label universe"))
    n = report.n_addresses
    add(f"  addresses                {n:>12,}")
    add(f"  illicit                  {report.n_illicit:>12,}   {_pct(report.n_illicit, n)}")
    add(f"  licit                    {report.n_licit:>12,}   {_pct(report.n_licit, n)}")
    add(f"  unknown                  {report.n_unknown:>12,}   {_pct(report.n_unknown, n)}")
    add(f"  labelled                 {report.n_labelled:>12,}   {_pct(report.n_labelled, n)}")
    if report.n_labelled:
        add(
            f"  illicit share of labelled              "
            f"{report.n_illicit / report.n_labelled * 100:6.2f}%"
        )
    add("")

    # -- clusters ------------------------------------------------------
    with_labels = table[table["n_labelled"] > 0]
    measurable = table[table["n_labelled"] >= 2]
    add(_rule("clusters"))
    add(f"  clusters                 {report.n_clusters:>12,}")
    add(
        f"  with >=1 labelled addr   {len(with_labels):>12,}   "
        f"{_pct(len(with_labels), report.n_clusters)}"
    )
    add(
        f"  with >=2 labelled addr   {len(measurable):>12,}   "
        f"{_pct(len(measurable), report.n_clusters)}"
    )
    add("     (only these can possibly show contamination)")
    add(f"  overall label coverage   {_pct(report.n_labelled, n):>12}")
    add("")

    # -- largest cluster -----------------------------------------------
    add(_rule("largest cluster"))
    if len(table):
        row = table.iloc[0]
        add(f"  cluster id               {int(row['cluster_id']):>12,}")
        add(f"  size                     {int(row['size']):>12,}   {_pct(row['size'], n)} of all addresses")
        size = int(row["size"])
        add(f"  illicit                  {int(row['n_illicit']):>12,}   {_pct(row['n_illicit'], size)} of cluster")
        add(f"  licit                    {int(row['n_licit']):>12,}   {_pct(row['n_licit'], size)} of cluster")
        add(f"  unknown                  {int(row['n_unknown']):>12,}   {_pct(row['n_unknown'], size)} of cluster")
        add(f"  labelled                 {int(row['n_labelled']):>12,}   {_pct(row['n_labelled'], size)} label coverage")
        if row["n_labelled"] > 0:
            add(f"  purity                   {row['purity']:>12.4f}   (of labelled only)")
            add(f"  entropy                  {row['entropy']:>12.4f}   bits")
            add(f"  majority class           {str(row['majority_class']):>12}")
        verdict = "YES - provably merged two entities" if row["contaminated"] else "no"
        add(f"  CONTAMINATED             {verdict:>12}")
    add("")

    # -- contamination --------------------------------------------------
    contaminated = report.contaminated
    addresses_inside = int(contaminated["size"].sum())
    add(_rule("contamination (both illicit and licit present)"))
    add(f"  contaminated clusters    {report.n_contaminated:>12,}")
    add(f"    of all clusters                       {_pct(report.n_contaminated, report.n_clusters)}")
    add(f"    of clusters with >=2 labelled         {_pct(report.n_contaminated, len(measurable))}")
    add(f"  addresses inside them    {addresses_inside:>12,}   {_pct(addresses_inside, n)} of all addresses")
    if len(contaminated):
        add(f"  largest contaminated     {int(contaminated['size'].max()):>12,}")
        add(f"  median size              {contaminated['size'].median():>12,.0f}")
    add("")

    # -- random baseline -------------------------------------------------
    add(_rule("random baseline (labels shuffled, clusters unchanged)"))
    add(f"  expected contaminated    {report.expected_contaminated_clusters:>12,.1f}   clusters")
    add(f"  observed contaminated    {report.n_contaminated:>12,}   clusters")
    ratio = report.observed_over_expected
    add(f"  observed / expected      {ratio:>12.3f}")
    add(f"  expected addr inside     {report.expected_contaminated_addresses:>12,.0f}")
    add(f"  observed addr inside     {addresses_inside:>12,}")
    add("")
    if np.isfinite(ratio):
        if ratio < 0.95:
            add("  Below 1.0: clusters are PURER than chance, i.e. co-spend is")
            add("  capturing real structure rather than merging arbitrarily.")
        elif ratio <= 1.05:
            add("  About 1.0: contamination is indistinguishable from chance.")
        else:
            add("  Above 1.0: clusters are contaminated MORE than chance would")
            add("  predict, which points at systematic false merging.")
    add("")

    # -- purity ----------------------------------------------------------
    add(_rule("purity distribution (clusters with >=1 labelled address)"))
    purity = with_labels["purity"].to_numpy()
    for low, high, label in _PURITY_BUCKETS:
        count = int(((purity >= low) & (purity < high)).sum())
        add(f"  {label:<22} {count:>12,}   {_pct(count, len(with_labels))}")
    add("")
    if len(with_labels):
        weights = with_labels["size"].to_numpy()
        labelled_weights = with_labels["n_labelled"].to_numpy()
        add(f"  mean purity, unweighted            {purity.mean():8.4f}")
        add(
            f"  mean purity, cluster-size weighted {np.average(purity, weights=weights):8.4f}"
        )
        add(
            f"  mean purity, labelled-count weighted "
            f"{np.average(purity, weights=labelled_weights):6.4f}"
        )
        add("     (size-weighted counts unlabelled addresses in the weight;")
        add("      labelled-weighted reflects only what we can actually see)")
    add("")

    # -- entropy ---------------------------------------------------------
    add(_rule("label entropy, bits (0 = pure, 1 = evenly split)"))
    if len(with_labels):
        entropy = with_labels["entropy"].to_numpy()
        weights = with_labels["size"].to_numpy()
        labelled_weights = with_labels["n_labelled"].to_numpy()
        add(f"  mean entropy, unweighted           {entropy.mean():8.4f}")
        add(
            f"  mean entropy, cluster-size weighted {np.average(entropy, weights=weights):7.4f}"
        )
        add(
            f"  mean entropy, labelled-count weighted "
            f"{np.average(entropy, weights=labelled_weights):5.4f}"
        )
        add(f"  clusters at zero entropy           {int((entropy == 0).sum()):>8,}")
    add("")

    # -- top contaminated -------------------------------------------------
    add(_rule(f"top {top} contaminated clusters by size"))
    if len(contaminated):
        add(
            f"  {'cluster':>10}  {'size':>8}  {'illic':>7}  {'licit':>7}  "
            f"{'unkn':>8}  {'cov':>6}  {'purity':>6}  {'H':>5}"
        )
        for _, row in contaminated.head(top).iterrows():
            add(
                f"  {int(row['cluster_id']):>10,}  {int(row['size']):>8,}  "
                f"{int(row['n_illicit']):>7,}  {int(row['n_licit']):>7,}  "
                f"{int(row['n_unknown']):>8,}  "
                f"{row['label_coverage'] * 100:>5.1f}%  "
                f"{row['purity']:>6.3f}  {row['entropy']:>5.3f}"
            )
    else:
        add("  none")
    add("=" * _W)
    return "\n".join(out)
