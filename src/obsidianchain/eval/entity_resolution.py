"""Score cluster assignments against external entity labels.

Ground truth here is entity ownership from ``data/processed/entity_labels.csv``
- two addresses sharing an entity are the same actor (must-link), two with
different entities are not (cannot-link). This is the first measurement in the
project that can show a cluster is *right*, not merely that it is not provably
wrong: licit/illicit purity could only detect contradictions.

Method (as approved)
--------------------
All unordered pairs of the labelled addresses that fall inside Elliptic++ are
classified by whether the two addresses share a cluster:

    same entity + same cluster        -> true positive
    different entity + same cluster   -> FALSE MERGE
    same entity + different cluster   -> FALSE SPLIT
    different entity + diff cluster   -> true negative

Counting is done from an entity x cluster contingency table rather than by
enumerating pairs: the pair counts are exactly the sums of ``C(n, 2)`` over its
cells, which is both faster and less error-prone than materialising 64,980
pairs, and makes 1,000 permutations cheap.

Two things this module refuses to do quietly
--------------------------------------------
**Recall has a structural ceiling.** An address that never appears as a
transaction input sits in a singleton cluster and can never be predicted
same-cluster with anything. Roughly 58% of the labelled addresses are in that
position, so raw recall is capped near a third. Raw, ceiling, and conditional
recall are always reported together; conditional recall alone would flatter
the result.

**Micro pair metrics are dominated by the largest entity.** One entity holds
44% of the positive pairs. The headline is therefore the macro figure, where
every entity counts once regardless of size; micro is reported alongside for
comparability, never on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.cluster.pipeline import ClusterRun
from obsidianchain.io import elliptic

LABELS_FILE = "entity_labels.csv"

DEFAULT_PERMUTATIONS = 1000
DEFAULT_SEED = 0

PER_ENTITY_COLUMNS = [
    "entity_display",
    "entity_norm",
    "n_addresses",
    "n_reachable",
    "n_clusters_spanned",
    "largest_cluster_share",
    "possible_pairs",
    "tp_pairs",
    "recall_e",
    "precision_e",
    "f1_e",
    "foreign_labelled",
    "verdict",
]

VERDICTS = ("PERFECT", "SPLIT", "MERGED", "SPLIT+MERGED", "UNREACHABLE")


def _choose2(n: np.ndarray | int) -> np.ndarray | int:
    """n choose 2, elementwise, in int64."""
    n = np.asarray(n, dtype=np.int64)
    return n * (n - 1) // 2


def _f1(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall):
        return 0.0
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---- results ----------------------------------------------------------


@dataclass
class MicroResult:
    """Pair-counting metrics pooled over every labelled pair."""

    n_addresses: int = 0
    n_pairs: int = 0
    n_positive: int = 0
    n_negative: int = 0

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    precision: float = 0.0
    recall_raw: float = 0.0
    f1: float = 0.0

    ceiling_pairs: int = 0
    recall_ceiling: float = 0.0
    recall_conditional: float = 0.0
    f1_conditional: float = 0.0

    @property
    def false_merges(self) -> int:
        return self.fp

    @property
    def false_splits(self) -> int:
        return self.fn

    @property
    def false_merge_rate(self) -> float:
        return 1.0 - self.precision if (self.tp + self.fp) else 0.0

    @property
    def false_split_rate(self) -> float:
        return 1.0 - self.recall_raw if self.n_positive else 0.0


@dataclass
class MacroResult:
    """Per-entity metrics averaged with every entity counting once."""

    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    n_entities: int = 0
    n_precision_defined: int = 0
    n_precision_undefined: int = 0
    recall_spread: tuple[float, float, float] = (0.0, 0.0, 0.0)  # min, median, max


@dataclass
class Coverage:
    label_coverage: float = 0.0
    n_labelled: int = 0
    n_universe: int = 0
    evaluable_addresses: int = 0
    evaluable_fraction: float = 0.0
    pair_reachability: float = 0.0
    n_entities_total: int = 0
    n_entities_multi: int = 0


@dataclass
class BaselineResult:
    n_permutations: int = 0
    seed: int = 0
    micro_f1_mean: float = 0.0
    micro_f1_sd: float = 0.0
    micro_precision_mean: float = 0.0
    macro_f1_mean: float = 0.0
    macro_f1_sd: float = 0.0
    micro_p_value: float = 1.0
    macro_p_value: float = 1.0

    def ratio(self, observed: float, expected: float) -> float:
        return observed / expected if expected > 0 else float("inf")


@dataclass
class EntityResolutionResult:
    mode: str
    micro: MicroResult
    macro: MacroResult
    coverage: Coverage
    baseline: BaselineResult | None
    per_entity: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdict_counts: dict[str, int] = field(default_factory=dict)


# ---- inputs -----------------------------------------------------------


def load_evaluation_set(data_root: Path | None = None) -> pd.DataFrame:
    """Labelled addresses inside the Elliptic++ universe.

    Requires ``entity-labels`` to have been run; that command owns validation
    and normalisation, and this one deliberately does not repeat it.
    """
    root = Path(data_root) if data_root is not None else elliptic.DEFAULT_DATA_ROOT
    path = root / "processed" / LABELS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run 'make run ARGS=\"entity-labels\"' first."
        )
    frame = pd.read_csv(path)
    missing = {"address", "entity_norm", "entity_display", "in_elliptic"} - set(
        frame.columns
    )
    if missing:
        raise ValueError(f"{LABELS_FILE} is missing columns: {sorted(missing)}")
    inside = frame[frame["in_elliptic"] == True].copy()  # noqa: E712
    if inside.empty:
        raise ValueError(
            f"{LABELS_FILE} has no rows flagged in_elliptic; re-run "
            f"'entity-labels' with --elliptic."
        )
    return inside.loc[:, ["address", "entity_norm", "entity_display"]]


# ---- core counting ----------------------------------------------------


def contingency(
    entity_codes: np.ndarray, cluster_codes: np.ndarray, n_entities: int, n_clusters: int
) -> np.ndarray:
    """Entity x cluster counts of labelled addresses."""
    flat = np.bincount(
        entity_codes.astype(np.int64) * n_clusters + cluster_codes.astype(np.int64),
        minlength=n_entities * n_clusters,
    )
    return flat.reshape(n_entities, n_clusters)


def micro_from_contingency(table: np.ndarray, n_addresses: int) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) pair counts.

    Standard pair-counting identity: agreements within a cell are true
    positives, everything else falls out of the row, column and grand totals.
    """
    tp = int(_choose2(table).sum())
    predicted_same = int(_choose2(table.sum(axis=0)).sum())
    actual_same = int(_choose2(table.sum(axis=1)).sum())
    total = int(_choose2(n_addresses))
    fp = predicted_same - tp
    fn = actual_same - tp
    tn = total - tp - fp - fn
    return tp, fp, fn, tn


def per_entity_metrics(table: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised per-entity recall, precision and F1 from the contingency.

    ``precision_e`` is entity-anchored: of the predicted-same pairs with at
    least one endpoint in the entity, the share whose other endpoint is also
    in it. Undefined when the entity contributes no predicted-same pair, in
    which case recall is necessarily zero and F1 is zero.
    """
    n_e = table.sum(axis=1).astype(np.int64)
    n_c = table.sum(axis=0).astype(np.int64)

    tp_e = _choose2(table).sum(axis=1).astype(np.int64)
    possible = _choose2(n_e)
    # pairs with exactly one endpoint in this entity, predicted same cluster
    cross_e = (table * (n_c[None, :] - table)).sum(axis=1).astype(np.int64)

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(possible > 0, tp_e / np.maximum(possible, 1), np.nan)
        denom = tp_e + cross_e
        precision = np.where(denom > 0, tp_e / np.maximum(denom, 1), np.nan)

    f1 = np.zeros(len(n_e), dtype=np.float64)
    both = np.isfinite(recall) & np.isfinite(precision) & ((recall + precision) > 0)
    f1[both] = (
        2 * precision[both] * recall[both] / (precision[both] + recall[both])
    )
    return {
        "n_e": n_e,
        "tp_e": tp_e,
        "possible": possible,
        "cross_e": cross_e,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def macro_from_contingency(table: np.ndarray) -> MacroResult:
    """Average per-entity metrics over entities with >=2 labelled addresses."""
    stats = per_entity_metrics(table)
    multi = stats["possible"] > 0  # exactly the entities with n_e >= 2
    if not multi.any():
        return MacroResult()

    recall = stats["recall"][multi]
    precision = stats["precision"][multi]
    f1 = stats["f1"][multi]
    defined = np.isfinite(precision)

    return MacroResult(
        macro_precision=float(precision[defined].mean()) if defined.any() else 0.0,
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        n_entities=int(multi.sum()),
        n_precision_defined=int(defined.sum()),
        n_precision_undefined=int((~defined).sum()),
        recall_spread=(
            float(recall.min()),
            float(np.median(recall)),
            float(recall.max()),
        ),
    )


# ---- baseline ---------------------------------------------------------


def permutation_baseline(
    entity_codes: np.ndarray,
    cluster_codes: np.ndarray,
    n_entities: int,
    n_clusters: int,
    observed_micro_f1: float,
    observed_macro_f1: float,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> BaselineResult:
    """Shuffle entity labels across addresses, holding clusters fixed.

    Permuting the label vector preserves both the cluster structure and the
    entity size distribution exactly, so the only thing destroyed is the
    association between them. That isolates what we are trying to measure.
    """
    rng = np.random.default_rng(seed)
    n_addresses = int(entity_codes.size)

    micro_f1 = np.empty(n_permutations, dtype=np.float64)
    micro_precision = np.empty(n_permutations, dtype=np.float64)
    macro_f1 = np.empty(n_permutations, dtype=np.float64)

    for i in range(n_permutations):
        shuffled = rng.permutation(entity_codes)
        table = contingency(shuffled, cluster_codes, n_entities, n_clusters)
        tp, fp, fn, _ = micro_from_contingency(table, n_addresses)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        micro_precision[i] = precision
        micro_f1[i] = _f1(precision, recall)
        macro_f1[i] = macro_from_contingency(table).macro_f1

    return BaselineResult(
        n_permutations=n_permutations,
        seed=seed,
        micro_f1_mean=float(micro_f1.mean()),
        micro_f1_sd=float(micro_f1.std(ddof=1)) if n_permutations > 1 else 0.0,
        micro_precision_mean=float(micro_precision.mean()),
        macro_f1_mean=float(macro_f1.mean()),
        macro_f1_sd=float(macro_f1.std(ddof=1)) if n_permutations > 1 else 0.0,
        micro_p_value=float(
            (1 + int((micro_f1 >= observed_micro_f1).sum())) / (n_permutations + 1)
        ),
        macro_p_value=float(
            (1 + int((macro_f1 >= observed_macro_f1).sum())) / (n_permutations + 1)
        ),
    )


# ---- top level --------------------------------------------------------


def evaluate(
    graph: elliptic.CoSpendGraph,
    run: ClusterRun,
    data_root: Path | None = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> EntityResolutionResult:
    """Score one clustering against the entity labels."""
    if graph.addresses is None:
        raise ValueError(
            "entity resolution needs address labels; "
            "call load_cospend_graph(keep_labels=True)"
        )
    labels = load_evaluation_set(data_root)

    codes = pd.Index(graph.addresses).get_indexer(labels["address"])
    if (codes < 0).any():
        raise ValueError(
            f"{int((codes < 0).sum())} labelled addresses are absent from the "
            f"Elliptic++ code space; regenerate entity_labels.csv"
        )

    entity_codes, entity_uniques = pd.factorize(labels["entity_norm"])
    roots = run.roots[codes]
    cluster_codes, _ = pd.factorize(roots)
    n_entities = int(len(entity_uniques))
    n_clusters = int(cluster_codes.max()) + 1 if len(cluster_codes) else 0
    n_addresses = int(len(labels))

    table = contingency(entity_codes, cluster_codes, n_entities, n_clusters)
    tp, fp, fn, tn = micro_from_contingency(table, n_addresses)

    # Structural ceiling: an address alone in its cluster can never be
    # predicted same-cluster with anything, so pairs touching one are
    # unreachable no matter how good the clustering is.
    full_sizes = np.bincount(run.roots, minlength=graph.n_addresses)
    reachable = full_sizes[roots] > 1
    ceiling_pairs = 0
    for code in range(n_entities):
        m = int((reachable & (entity_codes == code)).sum())
        ceiling_pairs += int(_choose2(m))

    n_positive = int(_choose2(table.sum(axis=1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall_raw = tp / n_positive if n_positive else 0.0
    recall_conditional = tp / ceiling_pairs if ceiling_pairs else 0.0

    micro = MicroResult(
        n_addresses=n_addresses,
        n_pairs=int(_choose2(n_addresses)),
        n_positive=n_positive,
        n_negative=int(_choose2(n_addresses)) - n_positive,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall_raw=recall_raw,
        f1=_f1(precision, recall_raw),
        ceiling_pairs=ceiling_pairs,
        recall_ceiling=ceiling_pairs / n_positive if n_positive else 0.0,
        recall_conditional=recall_conditional,
        f1_conditional=_f1(precision, recall_conditional),
    )
    macro = macro_from_contingency(table)

    sizes = table.sum(axis=1)
    coverage = Coverage(
        label_coverage=n_addresses / graph.n_addresses * 100,
        n_labelled=n_addresses,
        n_universe=int(graph.n_addresses),
        evaluable_addresses=int(reachable.sum()),
        evaluable_fraction=float(reachable.mean() * 100) if n_addresses else 0.0,
        pair_reachability=micro.recall_ceiling * 100,
        n_entities_total=n_entities,
        n_entities_multi=int((sizes >= 2).sum()),
    )

    per_entity, verdicts = _entity_table(
        table, labels, entity_codes, entity_uniques, cluster_codes, reachable
    )

    baseline = None
    if n_permutations > 0:
        baseline = permutation_baseline(
            entity_codes,
            cluster_codes,
            n_entities,
            n_clusters,
            micro.f1,
            macro.macro_f1,
            n_permutations=n_permutations,
            seed=seed,
        )

    return EntityResolutionResult(
        mode=run.label,
        micro=micro,
        macro=macro,
        coverage=coverage,
        baseline=baseline,
        per_entity=per_entity,
        verdict_counts=verdicts,
    )


def _entity_table(
    table: np.ndarray,
    labels: pd.DataFrame,
    entity_codes: np.ndarray,
    entity_uniques,
    cluster_codes: np.ndarray,
    reachable: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Per-entity rows with a verdict, ordered by size descending."""
    stats = per_entity_metrics(table)
    n_c = table.sum(axis=0)

    display = (
        labels.assign(_code=entity_codes)
        .groupby("_code")["entity_display"]
        .agg(lambda s: s.value_counts().idxmax())
    )

    rows = []
    for code in range(len(entity_uniques)):
        occupied = np.flatnonzero(table[code] > 0)
        n_addr = int(table[code].sum())
        n_reach = int((reachable & (entity_codes == code)).sum())
        foreign = int((n_c[occupied] - table[code][occupied]).sum())
        largest = int(table[code][occupied].max()) if occupied.size else 0

        if n_reach == 0:
            verdict = "UNREACHABLE"
        else:
            split = occupied.size > 1
            merged = foreign > 0
            verdict = (
                "SPLIT+MERGED" if split and merged
                else "SPLIT" if split
                else "MERGED" if merged
                else "PERFECT"
            )

        rows.append(
            {
                "entity_display": display.get(code, str(entity_uniques[code])),
                "entity_norm": str(entity_uniques[code]),
                "n_addresses": n_addr,
                "n_reachable": n_reach,
                "n_clusters_spanned": int(occupied.size),
                "largest_cluster_share": largest / n_addr if n_addr else 0.0,
                "possible_pairs": int(stats["possible"][code]),
                "tp_pairs": int(stats["tp_e"][code]),
                "recall_e": float(stats["recall"][code]),
                "precision_e": float(stats["precision"][code]),
                "f1_e": float(stats["f1"][code]),
                "foreign_labelled": foreign,
                "verdict": verdict,
            }
        )

    frame = pd.DataFrame(rows, columns=PER_ENTITY_COLUMNS).sort_values(
        ["n_addresses", "entity_norm"], ascending=[False, True]
    ).reset_index(drop=True)

    multi = frame[frame["n_addresses"] >= 2]
    counts = {v: int((multi["verdict"] == v).sum()) for v in VERDICTS}
    return frame, counts


def write_per_entity_csv(result: EntityResolutionResult, path: Path) -> int:
    frame = result.per_entity.copy()
    frame.insert(0, "heuristics", result.mode)
    for column in ("largest_cluster_share", "recall_e", "precision_e", "f1_e"):
        frame[column] = frame[column].round(6)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return int(len(frame))


# ---- rendering --------------------------------------------------------

_W = 78


def _rule(title: str) -> str:
    return f"-- {title} " + "-" * max(0, _W - len(title) - 4)


def format_report(result: EntityResolutionResult, top: int = 15) -> str:
    """Render one mode's evaluation."""
    m, mac, cov, base = result.micro, result.macro, result.coverage, result.baseline
    out: list[str] = []
    add = out.append

    add("=" * _W)
    add(f"obsidianchain :: entity resolution  [{result.mode}]")
    add("=" * _W)
    add("ground truth   entity ownership from address_labels/addresses.csv")
    add("               (not licit/illicit purity - that shows only contradictions)")
    add("")

    add(_rule("evaluation set"))
    add(f"  labelled addresses inside Elliptic++ {m.n_addresses:>10,}")
    add(f"  entities                             {cov.n_entities_total:>10,}"
        f"   ({cov.n_entities_multi} with >=2 addresses)")
    add(f"  all pairs                            {m.n_pairs:>10,}")
    add(f"  same-entity pairs (must-link)        {m.n_positive:>10,}"
        f"   {m.n_positive / m.n_pairs * 100:.2f}%")
    add(f"  different-entity pairs (cannot-link) {m.n_negative:>10,}")
    add("")

    add(_rule("HEADLINE - macro, every entity counts once"))
    add(f"  macro F1                             {mac.macro_f1:>10.4f}")
    add(f"  macro precision                      {mac.macro_precision:>10.4f}"
        f"   over {mac.n_precision_defined} entities")
    add(f"  macro recall                         {mac.macro_recall:>10.4f}")
    add(f"  entities averaged                    {mac.n_entities:>10,}")
    add(f"  precision undefined (no predicted pair) {mac.n_precision_undefined:>7,}"
        f"   excluded from macro precision")
    add(f"  per-entity recall  min / median / max "
        f"{mac.recall_spread[0]:.3f} / {mac.recall_spread[1]:.3f} / {mac.recall_spread[2]:.3f}")
    add("")

    add(_rule("micro - pooled over all pairs (dominated by the largest entity)"))
    add(f"  precision                            {m.precision:>10.4f}")
    add(f"  recall (raw)                         {m.recall_raw:>10.4f}")
    add(f"  F1                                   {m.f1:>10.4f}")
    add("")
    add(f"  true positives                       {m.tp:>10,}")
    add(f"  FALSE MERGES  (diff entity, same cl) {m.fp:>10,}"
        f"   rate {m.false_merge_rate:.4f}")
    add(f"  FALSE SPLITS  (same entity, diff cl) {m.fn:>10,}"
        f"   rate {m.false_split_rate:.4f}")
    add(f"  true negatives                       {m.tn:>10,}")
    add("")

    add(_rule("recall against its structural ceiling"))
    add(f"  raw recall           TP / {m.n_positive:<7,}     {m.recall_raw:>10.4f}")
    add(f"  ceiling              {m.ceiling_pairs:,} / {m.n_positive:,} reachable"
        f"      {m.recall_ceiling:>7.4f}")
    add(f"  conditional recall   TP / {m.ceiling_pairs:<7,}     {m.recall_conditional:>10.4f}")
    add(f"  conditional F1                       {m.f1_conditional:>10.4f}")
    add("")
    add("  An address that never appears as a transaction INPUT sits in a")
    add("  singleton cluster and can never be predicted same-cluster. Those")
    add("  pairs are unreachable by construction, not by failure. Conditional")
    add("  recall is never the headline.")
    add("")

    add(_rule("coverage"))
    add(f"  labelled share of address universe   {cov.label_coverage:>9.3f}%"
        f"   {cov.n_labelled:,} / {cov.n_universe:,}")
    add(f"  labelled addrs in a cluster of >1    {cov.evaluable_fraction:>9.1f}%"
        f"   {cov.evaluable_addresses:,} / {cov.n_labelled:,}")
    add(f"  positive pairs reachable             {cov.pair_reachability:>9.1f}%")
    add(f"  entities with >=2 addresses          {cov.n_entities_multi:>10,}"
        f" / {cov.n_entities_total:,}")
    add("")

    add(_rule("entity verdicts (entities with >=2 addresses)"))
    for verdict in VERDICTS:
        count = result.verdict_counts.get(verdict, 0)
        add(f"  {verdict:<16} {count:>4}")
    add("")

    add(_rule(f"largest {top} entities"))
    add(f"  {'entity':<22}{'addr':>5}{'reach':>6}{'clus':>5}"
        f"{'recall':>8}{'prec':>7}{'F1':>7}  verdict")
    shown = result.per_entity[result.per_entity["n_addresses"] >= 2].head(top)
    for _, row in shown.iterrows():
        prec = row["precision_e"]
        prec_text = f"{prec:>7.3f}" if np.isfinite(prec) else f"{'n/a':>7}"
        add(
            f"  {str(row['entity_display'])[:21]:<22}{int(row['n_addresses']):>5}"
            f"{int(row['n_reachable']):>6}{int(row['n_clusters_spanned']):>5}"
            f"{row['recall_e']:>8.3f}{prec_text}{row['f1_e']:>7.3f}  {row['verdict']}"
        )
    add("")

    if base is not None:
        add(_rule("random baseline - entity labels shuffled, clusters fixed"))
        add(f"  permutations                         {base.n_permutations:>10,}"
            f"   seed {base.seed}")
        add(f"  micro F1     observed {m.f1:.4f}   null {base.micro_f1_mean:.4f}"
            f" +/- {base.micro_f1_sd:.4f}")
        add(f"               observed / null       "
            f"{base.ratio(m.f1, base.micro_f1_mean):>10.1f}x"
            f"   p = {base.micro_p_value:.4f}")
        add(f"  macro F1     observed {mac.macro_f1:.4f}   null {base.macro_f1_mean:.4f}"
            f" +/- {base.macro_f1_sd:.4f}")
        add(f"               observed / null       "
            f"{base.ratio(mac.macro_f1, base.macro_f1_mean):>10.1f}x"
            f"   p = {base.macro_p_value:.4f}")
        add(f"  null precision (~= positive rate)    {base.micro_precision_mean:>10.4f}")
        add("")
        add("  Shuffling preserves cluster structure and entity sizes exactly,")
        add("  so only the association between them is destroyed.")
        add("")
    add("=" * _W)
    return "\n".join(out)


def format_delta(
    baseline_result: EntityResolutionResult, change_result: EntityResolutionResult
) -> str:
    """Side-by-side comparison of the two heuristic modes."""
    a, b = baseline_result, change_result
    out: list[str] = []
    add = out.append
    add("=" * _W)
    add("entity resolution :: co-spend vs co-spend + change")
    add("=" * _W)
    # Full mode names do not fit; the header below names them once.
    add(f"  left = {a.mode}    right = {b.mode}")
    add("")
    add(f"  {'':<32}{'co-spend':>13}{'+ change':>13}{'delta':>13}")
    add("  " + "-" * (_W - 4))

    def line(name: str, x: float, y: float, fmt: str = ".4f") -> None:
        add(f"  {name:<32}{x:>13{fmt}}{y:>13{fmt}}{y - x:>+13{fmt}}")

    def iline(name: str, x: int, y: int) -> None:
        add(f"  {name:<32}{x:>13,}{y:>13,}{y - x:>+13,}")

    line("MACRO F1 (headline)", a.macro.macro_f1, b.macro.macro_f1)
    line("macro precision", a.macro.macro_precision, b.macro.macro_precision)
    line("macro recall", a.macro.macro_recall, b.macro.macro_recall)
    add("")
    line("micro F1", a.micro.f1, b.micro.f1)
    line("micro precision", a.micro.precision, b.micro.precision)
    line("micro recall (raw)", a.micro.recall_raw, b.micro.recall_raw)
    line("micro recall (conditional)", a.micro.recall_conditional, b.micro.recall_conditional)
    add("")
    iline("true positives", a.micro.tp, b.micro.tp)
    iline("FALSE MERGES", a.micro.fp, b.micro.fp)
    iline("FALSE SPLITS", a.micro.fn, b.micro.fn)
    add("")
    iline("reachable positive pairs", a.micro.ceiling_pairs, b.micro.ceiling_pairs)
    iline("evaluable addresses", a.coverage.evaluable_addresses, b.coverage.evaluable_addresses)
    add("")
    for verdict in VERDICTS:
        iline(
            f"entities {verdict}",
            a.verdict_counts.get(verdict, 0),
            b.verdict_counts.get(verdict, 0),
        )
    add("=" * _W)
    return "\n".join(out)
