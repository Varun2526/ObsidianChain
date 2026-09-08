"""Cumulative clustering over Elliptic++'s 49 timesteps.

For each t, cluster using only transactions from timesteps 1..t and snapshot
the structure. Because union-find is **append-only** - a merge is never undone
and an edge is never removed - the whole series comes from one pass: add the
edges belonging to timestep t, take a snapshot, move on. Recomputing from
scratch 49 times would be 49x the work for identical output.

That property is the only reason this is cheap, and it is worth stating
explicitly: if the design ever needed edge removal, or a merge that could be
retracted on new evidence, this incremental replay would be invalid and the
series would have to be rebuilt per timestep.

Within a timestep, co-spend edges are applied before change edges, matching
the provenance rule used elsewhere: cryptographic evidence gets first claim on
any merge, so change detection is only credited with what it genuinely adds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.cluster import change as change_mod
from obsidianchain.cluster.unionfind import UnionFind
from obsidianchain.eval import compare as compare_mod
from obsidianchain.eval import purity as purity_mod
from obsidianchain.io import elliptic

TXS_FEATURES = "txs_features.csv"
TIMESTEP_COLUMN = "Time step"
N_TIMESTEPS = 49

_FEATURE_CHUNK = 100_000

SERIES_COLUMNS = [
    "heuristics",
    "timestep",
    "n_clusters",
    "largest",
    "second_largest",
    "coverage",
    "contaminated",
    "cospend_merges",
    "change_merges",
    "cospend_edges_applied",
    "change_edges_applied",
    "addresses_seen",
    "coverage_of_seen",
]


@dataclass
class EvolutionSeries:
    frame: pd.DataFrame
    n_unknown_timestep_edges: int = 0
    n_unknown_timestep_tx: int = 0

    def for_mode(self, mode: str) -> pd.DataFrame:
        return self.frame[self.frame["heuristics"] == mode]

    @property
    def modes(self) -> list[str]:
        return list(dict.fromkeys(self.frame["heuristics"].tolist()))


@dataclass
class SuperlinearOnset:
    """Where, if anywhere, the largest cluster stops growing linearly."""

    mode: str
    detected: bool
    timestep: int | None
    exponent_at_onset: float
    global_exponent: float
    largest_jump_timestep: int
    largest_jump: int

    @property
    def is_step_like(self) -> bool:
        """Whether the detection rests on discrete jumps rather than a trend.

        A global exponent at or below 1 with a super-linear local window means
        the curve is flat-then-jumping, not smoothly accelerating. The
        distinction matters: a step is two large clusters merging on a single
        transaction, which is a different phenomenon from runaway collapse.
        """
        return self.detected and self.global_exponent <= 1.0

    def describe(self) -> str:
        if not self.detected:
            return (
                f"{self.mode}: no sustained super-linear growth; "
                f"global exponent {self.global_exponent:.2f} "
                f"(largest single jump +{self.largest_jump:,} at t="
                f"{self.largest_jump_timestep})"
            )
        shape = (
            "step-like, not smooth acceleration"
            if self.is_step_like
            else "sustained acceleration"
        )
        return (
            f"{self.mode}: super-linear from timestep {self.timestep} "
            f"(local exponent {self.exponent_at_onset:.2f}, "
            f"global {self.global_exponent:.2f} - {shape}); "
            f"largest single jump +{self.largest_jump:,} at t="
            f"{self.largest_jump_timestep}"
        )


# ---- timesteps --------------------------------------------------------


def load_tx_timesteps(data_root: Path | None = None) -> pd.Series:
    """Map txId to its timestep, from txs_features.csv.

    That file is 662 MB across 184 columns, so it is streamed in chunks and
    only the two columns needed are retained.
    """
    path = elliptic.find_dataset_file(TXS_FEATURES, data_root)
    parts = []
    for chunk in pd.read_csv(
        path, usecols=["txId", TIMESTEP_COLUMN], chunksize=_FEATURE_CHUNK
    ):
        parts.append(chunk)
    frame = pd.concat(parts, ignore_index=True).drop_duplicates(subset="txId")
    return pd.Series(
        frame[TIMESTEP_COLUMN].to_numpy(dtype=np.int16),
        index=frame["txId"].to_numpy(),
        name="timestep",
    )


def _bucket_by_timestep(
    tx_ids: np.ndarray, timesteps: pd.Series, n_timesteps: int
) -> tuple[np.ndarray, int]:
    """Assign each item a timestep index, folding unknowns into the last one.

    A transaction with no row in txs_features.csv would otherwise be dropped,
    which would make the final snapshot disagree with the non-cumulative
    result. Folding unknowns into the final timestep keeps t=49 exactly equal
    to clustering over everything.
    """
    mapped = timesteps.reindex(tx_ids).to_numpy()
    unknown = np.isnan(mapped)
    mapped = np.where(unknown, n_timesteps, mapped).astype(np.int32)
    mapped = np.clip(mapped, 1, n_timesteps)
    return mapped, int(unknown.sum())


# ---- snapshots --------------------------------------------------------


def _snapshot(
    roots: np.ndarray, classes: np.ndarray, n_addresses: int
) -> dict[str, float]:
    """Cluster statistics from one set of roots.

    Deliberately avoids building the full per-cluster DataFrame: three
    bincounts over the address array is far cheaper, and this runs 98 times.
    """
    size = np.bincount(roots, minlength=n_addresses)
    illicit = np.bincount(roots, weights=(classes == purity_mod.ILLICIT), minlength=n_addresses)
    licit = np.bincount(roots, weights=(classes == purity_mod.LICIT), minlength=n_addresses)

    present = size > 0
    sizes = size[present]
    contaminated = int(((illicit[present] > 0) & (licit[present] > 0)).sum())

    ordered = np.sort(sizes)[::-1]
    clustered = int(sizes[sizes > 1].sum())
    return {
        "n_clusters": int(ordered.size),
        "largest": int(ordered[0]) if ordered.size else 0,
        "second_largest": int(ordered[1]) if ordered.size > 1 else 0,
        "coverage": clustered / n_addresses * 100 if n_addresses else 0.0,
        "contaminated": contaminated,
        "clustered": clustered,
    }


def _run_mode(
    mode: str,
    graph: elliptic.CoSpendGraph,
    classes: np.ndarray,
    cospend_bucket: np.ndarray,
    change_edges: np.ndarray | None,
    change_bucket: np.ndarray | None,
    n_timesteps: int,
) -> list[dict[str, float]]:
    forest = UnionFind(graph.n_addresses)
    rows: list[dict[str, float]] = []

    cospend_merges = change_merges = 0
    cospend_applied = change_applied = 0
    seen = np.zeros(graph.n_addresses, dtype=bool)

    for step in range(1, n_timesteps + 1):
        mask = cospend_bucket == step
        if mask.any():
            batch = graph.edges[mask]
            cospend_merges += forest.add_edges(batch)
            cospend_applied += int(len(batch))
            seen[batch.reshape(-1)] = True

        if change_edges is not None and change_bucket is not None:
            cmask = change_bucket == step
            if cmask.any():
                batch = change_edges[cmask]
                change_merges += forest.add_edges(batch)
                change_applied += int(len(batch))
                seen[batch.reshape(-1)] = True

        stats = _snapshot(forest.roots(), classes, graph.n_addresses)
        n_seen = int(seen.sum())
        rows.append(
            {
                "heuristics": mode,
                "timestep": step,
                "n_clusters": stats["n_clusters"],
                "largest": stats["largest"],
                "second_largest": stats["second_largest"],
                "coverage": stats["coverage"],
                "contaminated": stats["contaminated"],
                "cospend_merges": cospend_merges,
                "change_merges": change_merges,
                "cospend_edges_applied": cospend_applied,
                "change_edges_applied": change_applied,
                "addresses_seen": n_seen,
                "coverage_of_seen": (
                    stats["clustered"] / n_seen * 100 if n_seen else 0.0
                ),
            }
        )
    return rows


def run_evolution(
    data_root: Path | None = None,
    threshold: float = change_mod.DEFAULT_THRESHOLD,
    use_features: bool = True,
    n_timesteps: int = N_TIMESTEPS,
) -> EvolutionSeries:
    """Build the cumulative series for both heuristic modes."""
    graph = elliptic.load_cospend_graph(data_root, keep_labels=True)
    classes = purity_mod.load_classes_by_code(graph, data_root)
    timesteps = load_tx_timesteps(data_root)

    cospend_bucket, n_unknown_edges = _bucket_by_timestep(
        graph.edge_tx_ids, timesteps, n_timesteps
    )

    candidates = change_mod.build_candidates(
        graph, data_root, use_features=use_features
    )
    selected = change_mod.select_change_rows(candidates, threshold)
    if selected is None:
        raise NotImplementedError(
            "change.select_change_rows() returned None - the merge decision is "
            "not yet implemented in src/obsidianchain/cluster/change.py."
        )
    change_edges = change_mod.edges_from_selection(selected)
    change_bucket, _ = _bucket_by_timestep(
        selected["tx_id"].to_numpy() if len(selected) else np.empty(0),
        timesteps,
        n_timesteps,
    )

    unknown_tx = int(
        pd.Index(np.unique(graph.edge_tx_ids)).difference(timesteps.index).size
    )

    rows = _run_mode(
        compare_mod.BASELINE, graph, classes, cospend_bucket, None, None, n_timesteps
    )
    rows += _run_mode(
        compare_mod.WITH_CHANGE,
        graph,
        classes,
        cospend_bucket,
        change_edges,
        change_bucket,
        n_timesteps,
    )

    return EvolutionSeries(
        frame=pd.DataFrame(rows, columns=SERIES_COLUMNS),
        n_unknown_timestep_edges=n_unknown_edges,
        n_unknown_timestep_tx=unknown_tx,
    )


# ---- super-linear detection -------------------------------------------


def detect_superlinear(
    timestep: np.ndarray,
    largest: np.ndarray,
    mode: str,
    window: int = 8,
    exponent_threshold: float = 1.2,
    sustain: int = 3,
) -> SuperlinearOnset:
    """Find where the largest cluster starts growing faster than linearly.

    Fits ``log(largest) ~ alpha * log(t)`` over a trailing window. Growth
    proportional to t gives alpha = 1; alpha above ``exponent_threshold`` for
    ``sustain`` consecutive windows is reported as onset. Using a local
    exponent rather than a single global fit means a late collapse is not
    averaged away by a long linear start.

    With only 49 points this is a diagnostic, not a proof.
    """
    timestep = np.asarray(timestep, dtype=np.float64)
    largest = np.asarray(largest, dtype=np.float64)

    diffs = np.diff(largest, prepend=largest[0] if largest.size else 0.0)
    jump_index = int(np.argmax(diffs)) if diffs.size else 0

    usable = (timestep > 0) & (largest > 0)
    global_exponent = float("nan")
    if usable.sum() >= 2:
        global_exponent = float(
            np.polyfit(np.log(timestep[usable]), np.log(largest[usable]), 1)[0]
        )

    run = 0
    onset: int | None = None
    onset_exponent = float("nan")
    for end in range(window, len(largest) + 1):
        lo = end - window
        t_win, y_win = timestep[lo:end], largest[lo:end]
        good = (t_win > 0) & (y_win > 0)
        if good.sum() < 2 or len(np.unique(y_win[good])) < 2:
            run = 0
            continue
        alpha = float(np.polyfit(np.log(t_win[good]), np.log(y_win[good]), 1)[0])
        if alpha > exponent_threshold:
            run += 1
            if run >= sustain and onset is None:
                onset = int(timestep[lo])
                onset_exponent = alpha
        else:
            run = 0

    return SuperlinearOnset(
        mode=mode,
        detected=onset is not None,
        timestep=onset,
        exponent_at_onset=onset_exponent,
        global_exponent=global_exponent,
        largest_jump_timestep=int(timestep[jump_index]) if timestep.size else 0,
        largest_jump=int(diffs[jump_index]) if diffs.size else 0,
    )


def detect_all(series: EvolutionSeries) -> list[SuperlinearOnset]:
    return [
        detect_superlinear(
            series.for_mode(mode)["timestep"].to_numpy(),
            series.for_mode(mode)["largest"].to_numpy(),
            mode,
        )
        for mode in series.modes
    ]


# ---- outputs ----------------------------------------------------------

# The two trajectories very nearly coincide, so the baseline is drawn thick
# and solid underneath while the change line is drawn thin and dashed on top.
# That way the overlap is visible as overlap, instead of one line simply
# disappearing under the other.
_MODE_STYLE = {
    compare_mod.BASELINE: {
        "color": "#1b3a6b",
        "marker": "o",
        "label": "multi-input only",
        "linewidth": 4.5,
        "linestyle": "-",
        "markersize": 5.5,
        "zorder": 3,
    },
    compare_mod.WITH_CHANGE: {
        "color": "#f97316",
        "marker": "s",
        "label": "multi-input + change",
        "linewidth": 1.8,
        "linestyle": (0, (5, 2)),
        "markersize": 3.0,
        "zorder": 4,
    },
}
_STYLE_FALLBACK = {
    "color": "#444444",
    "marker": "o",
    "linewidth": 2.0,
    "linestyle": "-",
    "markersize": 4.0,
    "zorder": 3,
}


def write_series_csv(
    series: EvolutionSeries, path: Path, provenance=None
) -> int:
    """Write the full series. Returns the row count.

    ``provenance`` attaches the record two ways - a marker column in every
    row and a sibling ``.meta.json``. Optional so the function keeps working
    for callers that only want the frame on disk; every CLI path passes it,
    and ``tests/test_provenance.py`` reads the artifacts back to check.
    """
    frame = series.frame.copy()
    frame["coverage"] = frame["coverage"].round(4)
    frame["coverage_of_seen"] = frame["coverage_of_seen"].round(4)
    if provenance is not None:
        from obsidianchain import provenance as prov

        prov.write_frame(frame, path, provenance)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return int(len(frame))


def plot_largest_cluster(
    series: EvolutionSeries,
    path: Path,
    onsets: list[SuperlinearOnset] | None = None,
    dpi: int = 200,
) -> Path:
    """Plot largest cluster size against timestep, one line per mode.

    matplotlib is imported here rather than at module scope so the rest of
    this module stays importable without it, and the Agg backend is forced
    before pyplot loads - there is no display inside the container.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")

    for mode in series.modes:
        data = series.for_mode(mode)
        style = {**_STYLE_FALLBACK, "label": mode, **_MODE_STYLE.get(mode, {})}
        ax.plot(
            data["timestep"],
            data["largest"],
            color=style["color"],
            marker=style["marker"],
            markersize=style["markersize"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            label=style["label"],
            zorder=style["zorder"],
        )

    # Annotate only the earliest onset; the modes coincide, and stacking two
    # near-identical labels on the same x would be unreadable.
    flagged = [o for o in (onsets or []) if o.detected and o.timestep is not None]
    if flagged:
        onset = min(flagged, key=lambda o: o.timestep or 0)
        ax.axvline(
            onset.timestep,
            color="#6b7280",
            linestyle=(0, (4, 3)),
            linewidth=1.4,
            alpha=0.8,
            zorder=1,
        )
        shape = "step-like" if onset.is_step_like else "sustained"
        top = ax.get_ylim()[1]
        ax.annotate(
            f"onset t={onset.timestep}\n"
            f"local exponent {onset.exponent_at_onset:.2f}\n"
            f"global {onset.global_exponent:.2f} ({shape})",
            xy=(onset.timestep, top * 0.30),
            xytext=(-14, 0),
            textcoords="offset points",
            fontsize=10,
            color="#374151",
            ha="right",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "white",
                "edgecolor": "#d1d5db",
                "alpha": 0.95,
            },
        )

    ax.set_title(
        "Largest co-spend cluster over Elliptic++ timesteps",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("timestep (cumulative: transactions from 1..t)", fontsize=12.5)
    ax.set_ylabel("largest cluster size (addresses)", fontsize=12.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0.5, series.frame["timestep"].max() + 0.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11.5, frameon=True, framealpha=0.95, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    subtitle = (
        "The two modes very nearly coincide: change detection shifts the "
        "largest cluster by under 0.3%.\n"
        "Union-find is append-only, so the series is one incremental pass, "
        "not 49 recomputations."
    )
    fig.text(
        0.5, 0.012, subtitle, ha="center", va="bottom", fontsize=9.5,
        color="#555", linespacing=1.5,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def format_evolution(
    series: EvolutionSeries, onsets: list[SuperlinearOnset]
) -> str:
    """Render the series as a readable summary."""
    width = 78
    out: list[str] = []
    add = out.append

    add("=" * width)
    add("obsidianchain :: cumulative clustering across timesteps")
    add("=" * width)
    add("Union-find is append-only: edges are added in timestep order and the")
    add("statistics are snapshotted after each step. No edge is ever removed,")
    add("which is the only reason one pass can produce the whole series.")
    add("")
    if series.n_unknown_timestep_tx:
        add(
            f"note: {series.n_unknown_timestep_tx:,} transactions "
            f"({series.n_unknown_timestep_edges:,} edges) have no row in "
            f"{TXS_FEATURES};"
        )
        add("      they are folded into the final timestep so t=49 matches the")
        add("      non-cumulative result exactly.")
        add("")

    for mode in series.modes:
        data = series.for_mode(mode)
        add(f"-- {mode} " + "-" * max(0, width - len(mode) - 4))
        add(
            f"  {'t':>3}  {'clusters':>10}  {'largest':>8}  {'2nd':>7}  "
            f"{'cover%':>7}  {'contam':>7}  {'co-spend':>9}  {'change':>8}"
        )
        for _, row in data.iterrows():
            step = int(row["timestep"])
            if step % 4 and step != int(data["timestep"].max()):
                continue
            add(
                f"  {step:>3}  {int(row['n_clusters']):>10,}  "
                f"{int(row['largest']):>8,}  {int(row['second_largest']):>7,}  "
                f"{row['coverage']:>7.2f}  {int(row['contaminated']):>7,}  "
                f"{int(row['cospend_merges']):>9,}  {int(row['change_merges']):>8,}"
            )
        add("     (every 4th timestep shown; the CSV holds all 49)")
        add("")

    add("-- super-linear growth of the largest cluster " + "-" * (width - 46))
    for onset in onsets:
        add(f"  {onset.describe()}")
    add("")
    add("  Exponent is the slope of log(largest) against log(t) over a")
    add("  trailing 8-step window. 1.0 is linear growth; onset is reported")
    add("  when it exceeds 1.2 for 3 consecutive windows.")
    if any(o.is_step_like for o in onsets):
        add("")
        add("  Read the step-like cases carefully. A global exponent below 1")
        add("  with a super-linear local window means the largest cluster sat")
        add("  flat and then absorbed another large cluster outright. That is")
        add("  a discrete merge event, not runaway collapse, and it should not")
        add("  be presented as the latter.")
    add("=" * width)
    return "\n".join(out)
