"""Compare clustering configurations and quantify the coverage/correctness trade.

Adding change detection buys merges the multi-input heuristic cannot make, and
pays for them in correctness. Neither number means much alone; the comparison
is the result. Contamination is expected to RISE - if it does not, either the
change signals are firing too rarely to matter or something is wrong with the
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.cluster import change as change_mod
from obsidianchain.cluster.pipeline import ClusterRun, run_clustering
from obsidianchain.eval import purity as purity_mod
from obsidianchain.io import elliptic

BASELINE = "multi-input"
WITH_CHANGE = "multi-input+change"
HEURISTIC_CHOICES = (BASELINE, WITH_CHANGE)


@dataclass
class Comparison:
    baseline: ClusterRun
    with_change: ClusterRun
    baseline_purity: purity_mod.PurityReport
    change_purity: purity_mod.PurityReport
    candidates: change_mod.ChangeCandidates
    threshold: float
    selected: pd.DataFrame


def run_comparison(
    data_root: Path | None = None,
    threshold: float = change_mod.DEFAULT_THRESHOLD,
    weights: change_mod.ChangeWeights | None = None,
    use_features: bool = True,
) -> Comparison:
    """Cluster both ways, score purity for each, and bundle the result."""
    graph = elliptic.load_cospend_graph(data_root, keep_labels=True)
    classes = purity_mod.load_classes_by_code(graph, data_root)

    candidates = change_mod.build_candidates(
        graph, data_root, weights=weights, use_features=use_features
    )
    selected = change_mod.select_change_rows(candidates, threshold)
    if selected is None:
        raise NotImplementedError(
            "change.select_change_rows() returned None - the merge decision is "
            "not yet implemented in src/obsidianchain/cluster/change.py."
        )

    edges = change_mod.edges_from_selection(selected)
    confidences = (
        selected["confidence"].to_numpy() if len(selected) else np.empty(0)
    )

    baseline = run_clustering(graph, BASELINE)
    with_change = run_clustering(
        graph, WITH_CHANGE, change_edges=edges, change_confidences=confidences
    )

    return Comparison(
        baseline=baseline,
        with_change=with_change,
        baseline_purity=purity_mod.report_for_roots(
            baseline.roots, classes, graph.n_addresses
        ),
        change_purity=purity_mod.report_for_roots(
            with_change.roots, classes, graph.n_addresses
        ),
        candidates=candidates,
        threshold=threshold,
        selected=selected,
    )


# ---- rendering --------------------------------------------------------

_W = 78


def _delta(before: float, after: float, fmt: str = ",.0f") -> str:
    diff = after - before
    if before:
        pct = diff / before * 100
        return f"{diff:+{fmt}} ({pct:+.1f}%)"
    return f"{diff:+{fmt}}"


def format_comparison(comparison: Comparison, top: int = 10) -> str:
    base, chg = comparison.baseline, comparison.with_change
    bp, cp = comparison.baseline_purity, comparison.change_purity
    cand = comparison.candidates
    out: list[str] = []
    add = out.append

    add("=" * _W)
    add("obsidianchain :: multi-input vs multi-input + change detection")
    add("=" * _W)
    add(f"change threshold   {comparison.threshold:.2f}")
    add(
        f"weights            self_ref={cand.weights.self_reference} "
        f"one_fresh={cand.weights.one_fresh} "
        f"fresher={cand.weights.fresher_first_block} "
        f"peer_reused={cand.weights.peer_reused}"
    )
    add(f"features used      {cand.features_used}")
    add("")

    # -- candidate funnel ------------------------------------------------
    add("-- change candidate funnel " + "-" * (_W - 27))
    add(f"  transactions with outputs      {cand.n_transactions_with_outputs:>12,}")
    add(
        f"  exactly two outputs            {cand.n_two_output:>12,}   "
        f"{cand.n_two_output / max(cand.n_transactions_with_outputs, 1) * 100:5.1f}%"
    )
    add(f"  skipped, output count != 2     {cand.n_skipped_output_count:>12,}   (batching)")
    add(f"  skipped, no recorded inputs    {cand.n_skipped_no_inputs:>12,}   (coinbase)")
    add(f"  scored candidate outputs       {cand.n_candidates:>12,}")
    add(
        f"  above threshold                {len(cand.above(comparison.threshold)):>12,}"
    )
    add(f"  selected as change             {len(comparison.selected):>12,}")
    add("")

    add("-- signal frequency among scored candidates " + "-" * (_W - 44))
    breakdown = change_mod.signal_breakdown(cand.table)
    add(f"  {'signal':<22} {'fired':>12}  {'share':>7}  {'mean conf':>9}")
    for _, row in breakdown.iterrows():
        add(
            f"  {row['signal']:<22} {int(row['fired']):>12,}  "
            f"{row['share'] * 100:6.2f}%  {row['mean_confidence']:9.3f}"
        )
    add("")

    # -- the comparison table --------------------------------------------
    add("-- comparison " + "-" * (_W - 15))
    add(f"  {'':<30}{'multi-input':>14}{'+ change':>14}{'delta':>18}")
    add("  " + "-" * (_W - 4))

    rows: list[tuple[str, str, str, str]] = [
        (
            "clusters",
            f"{base.n_clusters:,}",
            f"{chg.n_clusters:,}",
            _delta(base.n_clusters, chg.n_clusters),
        ),
        (
            "largest cluster",
            f"{base.largest:,}",
            f"{chg.largest:,}",
            _delta(base.largest, chg.largest),
        ),
        (
            "largest as % of addresses",
            f"{base.largest / base.n_addresses * 100:.2f}%",
            f"{chg.largest / chg.n_addresses * 100:.2f}%",
            f"{(chg.largest - base.largest) / base.n_addresses * 100:+.2f} pp",
        ),
        (
            "coverage (in cluster > 1)",
            f"{base.coverage:.2f}%",
            f"{chg.coverage:.2f}%",
            f"{chg.coverage - base.coverage:+.2f} pp",
        ),
        (
            "singletons",
            f"{base.singletons:,}",
            f"{chg.singletons:,}",
            _delta(base.singletons, chg.singletons),
        ),
        (
            "merges from co-spend",
            f"{base.cospend_merges:,}",
            f"{chg.cospend_merges:,}",
            _delta(base.cospend_merges, chg.cospend_merges),
        ),
        (
            "merges from change",
            "0",
            f"{chg.change_merges:,}",
            f"+{chg.change_merges:,}",
        ),
        (
            "contaminated clusters",
            f"{bp.n_contaminated:,}",
            f"{cp.n_contaminated:,}",
            _delta(bp.n_contaminated, cp.n_contaminated),
        ),
        (
            "addresses in contaminated",
            f"{int(bp.contaminated['size'].sum()):,}",
            f"{int(cp.contaminated['size'].sum()):,}",
            _delta(
                int(bp.contaminated["size"].sum()),
                int(cp.contaminated["size"].sum()),
            ),
        ),
        (
            "contamination obs/expected",
            f"{bp.observed_over_expected:.3f}",
            f"{cp.observed_over_expected:.3f}",
            f"{cp.observed_over_expected - bp.observed_over_expected:+.3f}",
        ),
    ]
    for name, before, after, delta in rows:
        add(f"  {name:<30}{before:>14}{after:>14}{delta:>18}")
    add("")

    # -- purity means -----------------------------------------------------
    add("-- purity, clusters with >=1 labelled address " + "-" * (_W - 46))
    for name, report in (("multi-input", bp), ("+ change", cp)):
        labelled = report.clusters[report.clusters["n_labelled"] > 0]
        if not len(labelled):
            continue
        purity = labelled["purity"].to_numpy()
        add(
            f"  {name:<20} mean {purity.mean():.5f}   "
            f"size-weighted {np.average(purity, weights=labelled['size'].to_numpy()):.5f}   "
            f"pure {int((purity == 1.0).sum()):,}"
        )
    add("")

    # -- newly contaminated ------------------------------------------------
    add(f"-- largest clusters that became contaminated " + "-" * (_W - 45))
    base_bad = set(
        bp.contaminated["cluster_id"].tolist()
    )
    new_bad = cp.contaminated[~cp.contaminated["cluster_id"].isin(base_bad)]
    if len(new_bad):
        add(
            f"  {'cluster':>10}  {'size':>8}  {'illic':>7}  {'licit':>7}  "
            f"{'cov':>6}  {'purity':>6}"
        )
        for _, row in new_bad.head(top).iterrows():
            add(
                f"  {int(row['cluster_id']):>10,}  {int(row['size']):>8,}  "
                f"{int(row['n_illicit']):>7,}  {int(row['n_licit']):>7,}  "
                f"{row['label_coverage'] * 100:>5.1f}%  {row['purity']:>6.3f}"
            )
        add("")
        add(f"  {len(new_bad):,} clusters became contaminated that were not before.")
    else:
        add("  none - no cluster crossed from clean to contaminated")
    add("")

    # -- reading ------------------------------------------------------------
    add("-- reading " + "-" * (_W - 12))
    if chg.largest > base.largest * 2:
        add("  The largest cluster more than doubled: change detection is")
        add("  collapsing distinct entities together, which is the documented")
        add("  super-cluster failure reproduced on real data.")
    elif chg.largest > base.largest:
        add("  The largest cluster grew but did not collapse. Elliptic++ is a")
        add("  curated subset, and its transaction coverage is too sparse for")
        add("  a single bad merge to chain across the graph the way it does")
        add("  on the full chain.")
    else:
        add("  The largest cluster did not grow at all.")
    add("")
    contamination_delta = cp.n_contaminated - bp.n_contaminated
    if contamination_delta > 0:
        pct = contamination_delta / bp.n_contaminated * 100 if bp.n_contaminated else 0.0
        add(
            f"  Contaminated clusters rose by {contamination_delta:,} "
            f"({pct:+.1f}%), and addresses inside them by "
            f"{int(cp.contaminated['size'].sum()) - int(bp.contaminated['size'].sum()):,}."
        )
        add("  That is the trade stated plainly: change detection buys coverage")
        add("  and pays in correctness.")
    elif contamination_delta == 0:
        add("  Contamination did not move. Either too few change merges landed")
        add("  in labelled regions to register, or the accepted merges were")
        add("  correct. Check the merge count before concluding either.")
    else:
        add("  Contamination FELL, which is not the expected direction and")
        add("  warrants inspection before it is reported.")
    add("=" * _W)
    return "\n".join(out)
