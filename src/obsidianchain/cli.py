"""obsidianchain command line entry point.

Runs fully air-gapped. No command here may perform network I/O.
"""

from __future__ import annotations

import os
import platform
import resource
import socket
import sys
import time
from enum import Enum
from pathlib import Path

import typer

from obsidianchain import __version__


def synthetic_dir_name() -> str:
    """Directory under processed/ that holds Phase-3-visible network data."""
    from obsidianchain.network import synthetic

    return synthetic.OBSERVATIONS_DIR


class Mode(str, Enum):
    """Clustering mode. Same data, same seed, one flag between them."""

    CHAIN_ONLY = "chain-only"
    FUSED = "fused"


class Heuristics(str, Enum):
    """Which merge evidence to cluster on."""

    MULTI_INPUT = "multi-input"
    MULTI_INPUT_CHANGE = "multi-input+change"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Offline Bitcoin forensics prototype (NTRO PS 26146).",
)

def _default_data_root() -> Path:
    env = os.environ.get("OBSIDIANCHAIN_DATA")
    if env:
        return Path(env)
    repo_data = Path(__file__).resolve().parents[2] / "data"
    if repo_data.is_dir():
        return repo_data
    return Path("/data")


DATA_ROOT = _default_data_root()


def _evidence_run_inputs(
    data_root: Path, network_manifest: dict, production_config
) -> tuple[dict, str]:
    """Hash every input that determines an evidence row, and fold them.

    Computed HERE, on the artifact-generation side, and persisted into the
    sidecar. The API must never do this: an API-only deployment has no
    ``data/raw/``, and hashing an input per request would make a response
    depend on a file the endpoint does not own.

    Two chain files, not one. ``AddrTx_edgelist.csv`` fixes edge ordering;
    ``wallets_classes.csv`` is concatenated first in the single factorize, so
    it shifts every address code. Both determine a row, so both are hashed.
    """
    from obsidianchain import evidence_contract as contract
    from obsidianchain import run_fingerprint as rf
    from obsidianchain.eval.evidence_funnel import PROBE_CONFIG
    from obsidianchain.io import elliptic

    addr_tx = elliptic.find_dataset_file(elliptic.ADDR_TX, data_root)
    universe = elliptic.find_dataset_file(elliptic.WALLETS_CLASSES, data_root)
    # BOTH configurations, because schema /2 rows are determined by both:
    # the thirteen probe columns by the first, the seven production columns
    # by the second. A fingerprint covering only the probe half would let the
    # production rule change while every evidence id kept resolving.
    row_config = contract.combined_row_config_label(
        probe_min_pooled=PROBE_CONFIG.min_pooled_observations,
        probe_min_observer=PROBE_CONFIG.min_observer_observations,
        production_min_pooled=production_config.min_pooled_observations,
        production_min_observer=production_config.min_observer_observations,
        production_alpha=production_config.alpha,
        production_min_effect=production_config.min_effect,
    )
    inputs = {
        "chain_addr_tx_sha256": rf.sha256_file(addr_tx),
        "chain_universe_sha256": rf.sha256_file(universe),
        "network_dataset_sha256": str(network_manifest.get("dataset_sha256", "")),
        "heuristics": rf.HEURISTICS_MULTI_INPUT,
        "row_statistics_config": row_config,
    }
    full = rf.build_evidence_run_fingerprint(
        chain_addr_tx_sha256=inputs["chain_addr_tx_sha256"],
        chain_universe_sha256=inputs["chain_universe_sha256"],
        network_dataset_sha256=inputs["network_dataset_sha256"],
        heuristics=inputs["heuristics"],
        row_statistics_config=inputs["row_statistics_config"],
    )
    return inputs, full


def _publish_pair(staged: Path, destination: Path, run_fingerprint: str) -> None:
    """Move a staged artifact and its sidecar into place, tear-safely.

    Two files means two ``os.replace`` calls and there is no ordering that
    makes them one atomic step. What CAN be guaranteed is that no torn
    intermediate state is ever mistaken for a complete one, and that is done
    in three parts:

    1. **The identity is in both files.** ``provenance.write_frame`` stamps
       the run fingerprint into the parquet footer as well as the sidecar, so
       a mismatched pair is detectable by reading it. The API's gate refuses
       one rather than serving rows under the wrong fingerprint - which is
       the specific accident that matters, because new-parquet-with-old-
       sidecar does not look broken, it looks like a valid earlier run.

    2. **Both halves are durable before either is visible.** The staged
       files are fsynced, so a power loss cannot publish a name that points
       at unwritten blocks. Without this the window is not the microseconds
       between two renames; it is however long the page cache holds the data.

    3. **The sidecar goes last, and its absence is checked for.** Ordering
       does not affect detectability - the gate catches either half being
       stale - but it decides which failure an operator meets. Parquet first
       means the incomplete state is "new rows, no matching sidecar", and
       ``load_sidecar`` already refuses an artifact whose sidecar is missing
       or stale rather than defaulting to serving it.

    The verification in :func:`_verify_funnel_artifact` has already run at
    this point, so nothing here decides whether to publish - only how.
    """
    import os as _os

    staged_meta = Path(str(staged) + ".meta.json")
    if not staged_meta.is_file():
        typer.secho(
            f"HARD STOP: {staged_meta.name} was not written beside the "
            f"staged artifact. Publishing the parquet alone would leave it "
            f"with the PREVIOUS run's sidecar. Refusing to publish.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=8)

    # Durable before visible. Directory fsync too: on POSIX the rename
    # itself is only durable once the parent directory's entry is flushed.
    for path in (staged, staged_meta):
        handle = _os.open(path, _os.O_RDONLY)
        try:
            _os.fsync(handle)
        finally:
            _os.close(handle)

    _os.replace(staged, destination)
    _os.replace(staged_meta, str(destination) + ".meta.json")

    directory = _os.open(destination.parent, _os.O_RDONLY)
    try:
        _os.fsync(directory)
    finally:
        _os.close(directory)

    # Read the published pair back through the same gate the API uses. A
    # publish that cannot be served is a failed publish, and finding that out
    # now - with the staging directory still on disk and the operator still
    # watching - is worth one footer read.
    from obsidianchain.api import provenance_gate as _gate
    from obsidianchain import provenance as _prov

    meta = _gate.load_sidecar(destination)
    if meta.get("run_fingerprint") != run_fingerprint:
        typer.secho(
            f"HARD STOP: the published sidecar carries run fingerprint "
            f"{str(meta.get('run_fingerprint'))[:16]}, not the "
            f"{run_fingerprint[:16]} just computed. The pair is torn.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=8)
    embedded = _prov.read_artifact_identity(destination).get(
        _prov.FINGERPRINT_METADATA_KEY
    )
    if embedded != run_fingerprint:
        typer.secho(
            f"HARD STOP: the published parquet carries embedded fingerprint "
            f"{str(embedded)[:16]}, not {run_fingerprint[:16]}. The API "
            f"would refuse this artifact as a torn publish.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=8)
    _gate.require_artifact_identity(meta, destination)


def _verify_funnel_artifact(staged: Path, published: Path, counts: dict) -> None:
    """Section 14: what must be identical, and what is expected to change.

    The thirteen probe columns must be bit-identical in VALUE against the
    artifact being replaced. The whole-file hash is expected to change - it
    must, once seven columns are added - so it is never compared.
    """
    import pandas as pd

    from obsidianchain import evidence_contract as contract

    import pyarrow.parquet as pq

    fresh = pd.read_parquet(staged)
    for column in contract.PRODUCTION_COLUMNS:
        if column not in fresh.columns:
            typer.secho(f"missing production column {column!r}", fg=typer.colors.RED)
            raise typer.Exit(code=5)

    # Section 1 pins the on-disk types, so the gate checks them. Without this
    # a nullable integer silently lands as a double and the API serves 8.0
    # where the contract promises 8.
    expected = {
        "verdict_production": "string",
        "reason_code_production": "string",
        "reason_production": "string",
        "dof_production": "int64",
        "chi2_production": "double",
        "p_value_production": "double",
        "effect_production": "double",
    }
    actual = dict(
        zip(
            pq.ParquetFile(staged).schema_arrow.names,
            [str(t) for t in pq.ParquetFile(staged).schema_arrow.types],
        )
    )
    for column, want in expected.items():
        got = actual.get(column, "")
        if want not in got:
            typer.secho(
                f"HARD STOP: {column} has on-disk type {got!r}, contract "
                f"requires {want!r}. Refusing to publish.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=7)
    if int(len(fresh)) != sum(counts.values()):
        raise typer.Exit(code=5)

    if not published.is_file():
        return  # first generation: nothing to compare against
    old = pd.read_parquet(published)
    import numpy as np

    for column in old.columns:
        if column in contract.PRODUCTION_COLUMNS:
            continue
        a, b = old[column].to_numpy(), fresh[column].to_numpy()
        same = (
            np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f"
            else np.array_equal(a, b)
        )
        if not same:
            typer.secho(
                f"HARD STOP: existing column {column!r} changed during "
                f"regeneration. Refusing to publish; the artifact on disk is "
                f"untouched.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=6)


def _provenance(
    kind: str,
    dataset_id: str,
    *,
    world=None,
    synthetic_network=None,
    manifest=None,
    config=None,
    inputs=None,
    run_fingerprint=None,
    artifact=None,
    notes=(),
):
    """Provenance for one durable artifact.

    Every artifact this CLI writes gets one. The terminal banner is not
    enough: a CSV row detached from the run that produced it is
    indistinguishable from a real-world measurement, which is exactly how a
    synthetic decision ends up quoted as a result.
    """
    from obsidianchain import provenance as prov

    return prov.Provenance(
        provenance_type=prov.ProvenanceType(kind),
        dataset_id=dataset_id,
        dataset_sha256=prov.dataset_hash(manifest),
        world=world,
        synthetic_network=synthetic_network,
        generator_version=(manifest or {}).get("generator_version"),
        production_rule=prov.rule_config(config) if config is not None else None,
        inputs=inputs,
        run_fingerprint=run_fingerprint,
        artifact=artifact,
        notes=tuple(notes),
    )


@app.command()
def version() -> None:
    """Print the obsidianchain version."""
    typer.echo(f"obsidianchain {__version__}")


@app.command()
def info() -> None:
    """Show runtime environment and data directory status."""
    typer.echo(f"obsidianchain      {__version__}")
    typer.echo(f"python             {platform.python_version()}")
    typer.echo(f"platform           {platform.system()} {platform.machine()}")
    typer.echo(f"hashseed           {os.environ.get('PYTHONHASHSEED', '<unset>')}")
    typer.echo(f"data root          {DATA_ROOT}")

    for sub in ("raw", "processed"):
        path = DATA_ROOT / sub
        if path.is_dir():
            n = sum(1 for p in path.iterdir() if p.name != ".gitkeep")
            typer.echo(f"  {sub:<16} {path}  ({n} entries)")
        else:
            typer.echo(f"  {sub:<16} {path}  (missing)")

    typer.echo(f"network            {_network_state()}")


@app.command()
def isolation() -> None:
    """Assert the container has no usable network. Exits non-zero if it does."""
    state = _network_state()
    typer.echo(f"network: {state}")
    if not state.startswith("isolated"):
        typer.secho(
            "FAIL: a network route is reachable; run with --network none",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    typer.secho("PASS: air-gapped", fg=typer.colors.GREEN)


def _cluster(
    data_root: Path,
    heuristics: Heuristics,
    threshold: float,
    use_features: bool = True,
    need_labels: bool = False,
):
    """Load and cluster under the chosen heuristics.

    Returns (graph, run, candidates|None, selected|None).
    """
    from obsidianchain.cluster import change as change_mod
    from obsidianchain.cluster.pipeline import run_clustering
    from obsidianchain.io import elliptic

    wants_change = heuristics is Heuristics.MULTI_INPUT_CHANGE
    # keep_labels only controls whether address strings are retained; it has
    # no effect on edges or clustering.
    graph = elliptic.load_cospend_graph(
        data_root, keep_labels=wants_change or need_labels
    )

    if not wants_change:
        return graph, run_clustering(graph, heuristics.value), None, None

    candidates = change_mod.build_candidates(
        graph, data_root, use_features=use_features
    )
    selected = change_mod.select_change_rows(candidates, threshold)
    if selected is None:
        typer.secho(
            "change.select_change_rows() returned None - the merge decision is "
            "not yet implemented in src/obsidianchain/cluster/change.py.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    edges = change_mod.edges_from_selection(selected)
    run = run_clustering(
        graph,
        heuristics.value,
        change_edges=edges,
        change_confidences=selected["confidence"].to_numpy(),
    )
    return graph, run, candidates, selected


@app.command()
def compare(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    threshold: float = typer.Option(
        0.7, "--threshold", min=0.0, max=1.0, help="Minimum change confidence."
    ),
    top: int = typer.Option(10, "--top"),
    features: bool = typer.Option(
        True,
        "--features/--no-features",
        help="Use wallets_features.csv for the finer freshness signal.",
    ),
) -> None:
    """Compare multi-input clustering against multi-input + change detection."""
    from obsidianchain.eval import compare as compare_mod

    started = time.perf_counter()
    result = compare_mod.run_comparison(
        data_root, threshold=threshold, use_features=features
    )
    typer.echo(compare_mod.format_comparison(result, top=top))
    typer.echo("")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


@app.command("network-generate")
def network_generate(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    observers: int = typer.Option(8, "--observers", min=1, help="Vantage points."),
    origins: int = typer.Option(64, "--origins", min=1, help="Simulated origin nodes."),
    broadcaster_fraction: float = typer.Option(
        0.15, "--broadcaster-fraction", min=0.0, max=1.0,
        help="Share of origins that are known broadcasters.",
    ),
    median_ms: float = typer.Option(
        1200.0, "--median-ms", help="Median propagation delay (ASSUMPTION)."
    ),
    seed: int = typer.Option(0, "--seed"),
    limit: int = typer.Option(
        0, "--limit", help="Only the first N transactions (0 = all)."
    ),
    fmt: str = typer.Option("parquet", "--format", help="parquet or csv."),
    frozen: bool = typer.Option(
        True,
        "--frozen/--custom",
        help="Use the frozen September configuration, ignoring the tuning "
        "options above. Phase 3 must consume the frozen dataset.",
    ),
) -> None:
    """Generate SYNTHETIC network announcements for the Elliptic++ transactions.

    This data demonstrates the mechanism. It validates nothing: no packet was
    captured and the generator contains the structure the analysis looks for.
    Real validation needs mainnet capture plus controlled ground truth.
    """
    from obsidianchain.network import synthetic

    from dataclasses import replace

    started = time.perf_counter()
    config = synthetic.FROZEN_SEPTEMBER_2026
    if not frozen:
        config = replace(
            config,
            n_observers=observers,
            n_origins=origins,
            broadcaster_fraction=broadcaster_fraction,
            seed=seed,
            propagation=synthetic.PropagationModel(median_ms=median_ms),
        )
    txids = synthetic.load_transaction_ids(data_root)
    if limit > 0:
        txids = txids[:limit]

    observations, nodes, observer_table, ground_truth = synthetic.generate(
        txids, config
    )
    typer.echo(
        synthetic.format_summary(observations, nodes, observer_table, config)
    )

    written = synthetic.write_outputs(
        observations, nodes, observer_table, ground_truth,
        data_root / "processed", config, fmt=fmt,
    )
    typer.echo("")
    typer.echo(
        f"wrote {written['rows']:,} records "
        f"({written['bytes'] / 1e6:,.1f} MB) -> {written['observations']}"
    )
    typer.echo(f"      observers             -> {written['observers']}")
    typer.echo(f"      broadcaster IP list   -> {written['broadcaster_ips']}")
    typer.echo(f"      manifest              -> {written['manifest']}")
    typer.echo(
        f"      GROUND TRUTH (quarantined, not for Phase 3) "
        f"-> {written['ground_truth']}"
    )
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


def _mode_rows(label, run, fused=None) -> dict:
    """Common row set for one mode, for the before/after table."""
    return {
        "label": label,
        "clusters": run.n_clusters,
        "largest": run.largest,
        "coverage": run.coverage,
        "singletons": run.singletons,
        "merges": run.cospend_merges,
        "blocked": fused.blocked if fused else 0,
        "contested": fused.contested if fused else 0,
        "evaluated": fused.evaluated if fused else 0,
        "abstained": fused.abstained if fused else 0,
    }


def _run_mode(data_root: Path, mode: Mode, config=None):
    """Cluster in one mode. Returns (graph, run, fused_or_None, oracle_or_None)."""
    from obsidianchain.cluster.pipeline import run_clustering, run_fused
    from obsidianchain.io import elliptic
    from obsidianchain.network import separation

    graph = elliptic.load_cospend_graph(data_root, keep_labels=True)
    if mode is Mode.CHAIN_ONLY:
        return graph, run_clustering(graph, mode.value), None, None

    oracle = separation.build_oracle(
        graph, processed_root=data_root / "processed", data_root=data_root,
        config=config,
    )
    fused = run_fused(graph, oracle, label=mode.value)
    return graph, fused.run, fused, oracle


@app.command("run")
def run_mode(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    mode: Mode = typer.Option(
        Mode.CHAIN_ONLY, "--mode", help="chain-only or fused."
    ),
    min_pooled: int = typer.Option(
        25, "--min-pooled", help="Pooled observations required on each side."
    ),
    alpha: float = typer.Option(1e-4, "--alpha", help="Significance floor."),
    min_effect: float = typer.Option(0.05, "--min-effect", help="Effect floor."),
    top: int = typer.Option(5, "--top", help="Blocked merges to show."),
) -> None:
    """Cluster addresses, optionally with network cannot-link constraints.

    chain-only  co-spend union-find, nothing vetoes a merge
    fused       the same edges, with network separation able to refuse one

    Same data and same seed in both; the flag is the only difference.
    """
    from obsidianchain.network.separation import SeparationConfig

    started = time.perf_counter()
    config = SeparationConfig(
        min_pooled_observations=min_pooled, alpha=alpha, min_effect=min_effect
    )
    typer.echo("=" * 70)
    typer.echo(f"obsidianchain :: mode = {mode.value}")
    typer.echo("=" * 70)
    try:
        graph, run, fused, oracle = _run_mode(data_root, mode, config)
    except NotImplementedError as exc:
        # Show that the evidence layer loaded before reporting the gap, so it
        # is clear which half is missing.
        from obsidianchain.io import elliptic
        from obsidianchain.network import separation as sep

        graph = elliptic.load_cospend_graph(data_root, keep_labels=True)
        oracle = sep.build_oracle(
            graph, processed_root=data_root / "processed",
            data_root=data_root, config=config,
        )
        typer.echo(sep.format_oracle_summary(oracle))
        typer.echo("")
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from None

    if oracle is not None:
        from obsidianchain.network import separation as sep

        typer.echo(sep.format_oracle_summary(oracle))
        typer.echo("")

    typer.echo("-- results " + "-" * 58)
    typer.echo(f"  cluster count            {run.n_clusters:>14,}")
    typer.echo(f"  largest cluster          {run.largest:>14,}")
    typer.echo(f"  coverage                 {run.coverage:>13.2f}%")
    typer.echo(f"  merges applied           {run.cospend_merges:>14,}")
    if fused is not None:
        typer.echo(f"  merges BLOCKED           {fused.blocked:>14,}")
        typer.echo(f"  contested clusters       {fused.contested:>14,}")
        typer.echo(f"  unions with evidence     {fused.evaluated:>14,}")
        typer.echo(f"  unions abstained         {fused.abstained:>14,}"
                   f"   below pooled minimum")
        typer.echo(f"  evaluable fraction       "
                   f"{fused.evaluable_fraction * 100:>13.2f}%")
        if fused.blocked_merges:
            typer.echo("")
            typer.echo(
                f"-- first {min(top, len(fused.blocked_merges))} of "
                f"{len(fused.blocked_merges):,} blocked merges " + "-" * 26
            )
            for blocked in fused.blocked_merges[:top]:
                typer.echo(f"  {blocked.a:>8} x {blocked.b:<8} "
                           f"{blocked.evidence.describe()}")
    else:
        typer.echo("  merges BLOCKED           " + f"{0:>14,}   (no constraints)")
        typer.echo("  contested clusters       " + f"{0:>14,}")
    typer.echo("=" * 70)
    typer.echo("")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


@app.command("fusion-summary")
def fusion_summary(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    min_pooled: int = typer.Option(25, "--min-pooled"),
    alpha: float = typer.Option(1e-4, "--alpha"),
    min_effect: float = typer.Option(0.05, "--min-effect"),
) -> None:
    """Run both modes and print the before/after table.

    This is the demo. Chain-only and fused see identical data; the only
    difference is whether network evidence may veto a merge.
    """
    from obsidianchain.network.separation import SeparationConfig

    started = time.perf_counter()
    config = SeparationConfig(
        min_pooled_observations=min_pooled, alpha=alpha, min_effect=min_effect
    )
    _, chain_run, _, _ = _run_mode(data_root, Mode.CHAIN_ONLY, config)
    try:
        _, fused_run, fused, oracle = _run_mode(data_root, Mode.FUSED, config)
    except NotImplementedError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        typer.echo("")
        typer.echo("The chain-only baseline ran; the fused column needs the")
        typer.echo("constrained union-find implemented before this table can")
        typer.echo("be produced.")
        raise typer.Exit(code=2) from None

    width = 70
    e = typer.echo
    e("=" * width)
    e("OBSIDIANCHAIN - CHAIN-ONLY vs NETWORK-FUSED CLUSTERING")
    e("=" * width)
    e("  Same transactions, same seed, same co-spend edges in the same")
    e("  order. The only difference is whether network separation evidence")
    e("  may refuse a merge.")
    e("")
    e("  Network data is SYNTHETIC: this demonstrates the mechanism and")
    e("  validates nothing. Real validation needs mainnet capture.")
    e("")
    e(f"  {'':<28}{'chain-only':>13}{'fused':>13}{'delta':>14}")
    e("  " + "-" * (width - 4))

    def row(name, a, b, fmt=",", suffix=""):
        delta = b - a
        e(f"  {name:<28}{a:>13{fmt}}{b:>13{fmt}}{delta:>+14{fmt}}{suffix}")

    row("cluster count", chain_run.n_clusters, fused_run.n_clusters)
    row("largest cluster", chain_run.largest, fused_run.largest)
    e(f"  {'largest as % of addresses':<28}"
      f"{chain_run.largest / chain_run.n_addresses * 100:>12.2f}%"
      f"{fused_run.largest / fused_run.n_addresses * 100:>12.2f}%"
      f"{(fused_run.largest - chain_run.largest) / chain_run.n_addresses * 100:>+13.2f} pp")
    e(f"  {'coverage':<28}{chain_run.coverage:>12.2f}%"
      f"{fused_run.coverage:>12.2f}%"
      f"{fused_run.coverage - chain_run.coverage:>+13.2f} pp")
    row("singletons", chain_run.singletons, fused_run.singletons)
    row("merges applied", chain_run.cospend_merges, fused_run.cospend_merges)
    row("merges BLOCKED", 0, fused.blocked)
    row("contested clusters", 0, fused.contested)
    e("")
    e("-- constraint reach " + "-" * (width - 21))
    e(f"  unions with enough evidence   {fused.evaluated:>12,}")
    e(f"  unions abstained (NO_EVIDENCE){fused.abstained:>12,}")
    e(f"  evaluable fraction            {fused.evaluable_fraction * 100:>11.2f}%")
    e("")
    if fused.blocked == 0:
        e("  READING: no merge was blocked. On this dataset that is the")
        e("  expected outcome and not a bug - the Phase 2 audit measured")
        e("  per-transaction separability at 1.003, and only a small share of")
        e("  components accumulate enough pooled observations for the")
        e("  centroid difference to clear its own noise. The mechanism is")
        e("  wired end to end and abstains honestly; demonstrating a blocked")
        e("  merge needs either a longer capture per origin or a generator")
        e("  whose sigma is not borrowed from block propagation.")
    else:
        e(f"  READING: {fused.blocked:,} merge(s) refused on network evidence.")
        e("  Each is a super-cluster link prevented rather than cleaned up")
        e("  afterwards. Contested clusters are contradictions that arrived")
        e("  too late to prevent and are flagged, not silently resolved.")
    e("=" * width)
    typer.echo("")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


@app.command("world-experiment")
def world_experiment(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    regimes: str = typer.Option("all", "--regimes", help="A,B,C,D,E or all."),
    chain_root: Path = typer.Option(
        None, "--chain-root",
        help="Blockchain the graph is built from, holding raw/ "
        "[default: --data-root].",
    ),
    network_root: Path = typer.Option(
        None, "--network-root",
        help="Network observations and truth, holding processed/worlds/ "
        "[default: --data-root].",
    ),
    out: Path = typer.Option(
        None, "--out", help="[default: <network-root>/processed/phase33.csv]"
    ),
) -> None:
    """Phase 3.3: run the UNCHANGED engine across all five controlled worlds.

    The production rule is fixed - 25 pooled per side, alpha 1e-4, effect
    floor 0.05 - and identical in every regime. Tuning per world would make
    the experiment meaningless.
    """
    from obsidianchain.eval import phase33
    from obsidianchain.io import elliptic
    from obsidianchain.network import boundary, separation, worlds

    started = time.perf_counter()
    config = separation.SeparationConfig()  # production defaults, untouched
    selected = (
        tuple(worlds.Regime)
        if regimes.lower() == "all"
        else tuple(
            worlds.Regime(p.strip().upper())
            for p in regimes.split(",") if p.strip()
        )
    )

    # The chain and the network dataset are named separately because a
    # mismatched pair fails silently: unresolvable transactions are dropped,
    # the oracle ends up empty, and the report reads as a clean 100%
    # abstention. run_regime guards the pairing.
    chain = chain_root or data_root
    network = network_root or data_root

    graph = elliptic.load_cospend_graph(chain, keep_labels=True)
    outcomes: dict[str, phase33.RegimeOutcome] = {}
    try:
        for regime in selected:
            typer.echo(f"running regime {regime.value} ...", err=True)
            outcomes[regime.value] = phase33.run_regime(
                regime, graph, chain, config, network_root=network
            )
    except phase33.DatasetMismatchError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from None

    manifest = boundary.load_manifest(
        network / "processed", world=selected[0].value
    )
    fixture = manifest.get("fixture", "controlled-world")
    chain_description = (
        f"{graph.n_addresses:,} addresses, {graph.n_edges:,} co-spend edges "
        f"({'synthetic reach-stress chain' if fixture == 'reach-stress' else 'Elliptic++'})"
    )
    typer.echo(
        phase33.format_experiment(
            outcomes, config, fixture=fixture,
            chain_description=chain_description,
        )
    )
    from obsidianchain import provenance as prov

    # SYNTHETIC_CONTROL, not DEMO: these are controlled worlds, a real
    # experiment on generated data. Calling them DEMO would misrepresent
    # them, and calling them PRODUCTION would be far worse.
    artifact_provenance = _provenance(
        "SYNTHETIC_CONTROL",
        dataset_id=f"{fixture}/{','.join(o for o in outcomes)}",
        synthetic_network=True,
        manifest=manifest,
        config=config,
        notes=(
            "Controlled-world experiment. Generated to make the question "
            "answerable; not a measurement of Bitcoin.",
            chain_description,
        ),
    )
    destination = out or (network / "processed" / "phase33.csv")
    prov.write_frame(phase33.to_frame(outcomes), destination, artifact_provenance)
    decisions_path = destination.with_name(
        destination.stem + "_decisions.csv"
    )
    blocked_only = phase33.decisions_to_frame(outcomes)
    # Per-decision rows are the ones most likely to be copied out on their
    # own, so they carry the marker in the row as well as the sidecar.
    prov.write_frame(blocked_only, decisions_path, artifact_provenance)
    typer.echo("")
    typer.echo(f"wrote per-regime outcomes -> {destination}")
    typer.echo(f"wrote {len(blocked_only):,} decisions -> {decisions_path}")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


@app.command("world-generate")
def world_generate(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    regime: str = typer.Option(
        "all", "--regime", help="A, B, C, D, E, or all."
    ),
    entities: int = typer.Option(40, "--entities"),
    limit: int = typer.Option(0, "--limit", help="First N transactions (0=all)."),
    fmt: str = typer.Option("parquet", "--format"),
) -> None:
    """Generate the controlled-world regimes in a separate namespace.

    Writes to processed/worlds/<regime>/ and processed/worlds_truth/<regime>/.
    Does not touch FROZEN_SEPTEMBER_2026, its hash, or network_truth/.

    Still SYNTHETIC: these worlds inject a known entity-origin relationship
    so Phase 3 has something to find. They demonstrate mechanism and
    validate nothing.
    """
    from obsidianchain.io import elliptic
    from obsidianchain.network import worlds

    started = time.perf_counter()
    config = worlds.WorldConfig(n_entities=entities)

    if regime.lower() == "all":
        selected = tuple(worlds.Regime)
    else:
        selected = tuple(
            worlds.Regime(part.strip().upper())
            for part in regime.split(",")
            if part.strip()
        )

    graph = elliptic.load_cospend_graph(data_root, keep_labels=True)
    generated = worlds.generate_all(
        graph, config, data_root, limit=limit, regimes=selected
    )

    typer.echo("=" * 74)
    typer.echo("obsidianchain :: controlled world generation")
    typer.echo("=" * 74)
    typer.echo("  SYNTHETIC. Demonstrates mechanism, validates nothing.")
    typer.echo(f"  sigma fixed at {worlds.synthetic.DECKER_WATTENHOFER_SIGMA:.4f} "
               f"in every regime; only entity->origin differs.")
    typer.echo("")
    for reg, world in generated.items():
        written = worlds.write_world(world, data_root / "processed", fmt=fmt)
        manifest = written["manifest_data"]
        typer.echo(f"  {reg.value}  {worlds.REGIME_NAMES[reg]}")
        typer.echo(f"      seed {manifest['regime_seed']}   "
                   f"records {written['rows']:,}   "
                   f"entities {manifest['entity_count']}   "
                   f"{written['bytes'] / 1e6:.1f} MB")
        typer.echo(f"      sha256 {manifest['dataset_sha256'][:32]}")
        typer.echo(f"      observations -> {written['observations']}")
        typer.echo(f"      TRUTH (quarantined) -> {written['ground_truth']}")
    typer.echo("=" * 74)
    typer.echo("")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


@app.command("reach-stress-build")
def reach_stress_build(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    fixture_root: Path = typer.Option(
        None, "--fixture-root",
        help="[default: <data-root>/reach_stress]",
    ),
    components: int = typer.Option(300, "--components"),
    addresses: int = typer.Option(20, "--addresses-per-component"),
    transactions: int = typer.Option(200, "--transactions-per-component"),
) -> None:
    """Build the reach-stress fixture: dense chain, all five regimes.

    The production threshold of 25 is NOT lowered. This fixture raises the
    available evidence so the rule is actually forced to decide, which the
    normal worlds could not do - only 47 of 253,429 unions reached 25 there.
    """
    from obsidianchain.network import reach_stress

    started = time.perf_counter()
    root = fixture_root or (data_root / "reach_stress")
    config = reach_stress.ReachStressConfig(
        n_components=components,
        addresses_per_component=addresses,
        transactions_per_component=transactions,
    )
    written = reach_stress.build_fixture(root, config)

    typer.echo("=" * 74)
    typer.echo("obsidianchain :: reach-stress fixture")
    typer.echo("=" * 74)
    typer.echo("  SYNTHETIC. Threshold stays at 25; evidence is raised to meet it.")
    typer.echo(f"  components {config.n_components:,}   "
               f"addresses {config.n_addresses:,}   "
               f"transactions {config.n_transactions:,}")
    typer.echo(f"  expected pooled per component ~{config.expected_pooled_per_component}"
               f"  (production minimum 25)")
    typer.echo("")
    for key in ("A", "B", "C", "D", "E"):
        entry = written[key]
        manifest = entry["manifest"]
        typer.echo(f"  {key}  {manifest['regime_name']}")
        typer.echo(f"      records {entry['rows']:,}   "
                   f"sha256 {manifest['dataset_sha256'][:24]}   "
                   f"sticky-origins {manifest['sticky_address_origins']}")
    typer.echo("=" * 74)
    typer.echo("")
    typer.echo(f"fixture root -> {root}")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


@app.command("world-diagnostics")
def world_diagnostics_cmd(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    entities: int = typer.Option(40, "--entities"),
    sample: int = typer.Option(4000, "--sample"),
    out: Path = typer.Option(
        None,
        "--out",
        help="[default: <data-root>/processed/world_diagnostics.csv]",
    ),
) -> None:
    """Verify the five regimes produced the distributions intended.

    Runs before any Phase 3 inference. Regime A is the control: if it shows
    aggregate signal, this generator introduced structure the frozen dataset
    lacks and no other regime can be trusted.
    """
    from obsidianchain.eval import world_diagnostics as wd
    from obsidianchain.network import worlds

    started = time.perf_counter()
    config = worlds.WorldConfig(n_entities=entities)
    processed = data_root / "processed"

    reports: dict[str, wd.RegimeDiagnostics] = {}
    for regime in worlds.Regime:
        try:
            reports[regime.value] = wd.diagnose_regime(
                regime, processed, sample=sample
            )
        except FileNotFoundError as exc:
            typer.secho(f"regime {regime.value}: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from None

    checks = wd.check_expectations(reports, config)
    typer.echo(wd.format_diagnostics(reports, config, checks))

    from obsidianchain import provenance as prov
    from obsidianchain.network import boundary as _boundary

    destination = out or (processed / "world_diagnostics.csv")
    prov.write_frame(
        wd.to_frame(reports),
        destination,
        _provenance(
            "SYNTHETIC_CONTROL",
            dataset_id="worlds/" + ",".join(sorted(reports)),
            synthetic_network=True,
            manifest=_boundary.load_manifest(
                processed, world=sorted(reports)[0] if reports else None
            ),
            notes=("Generator verification, not a result.",),
        ),
    )
    typer.echo("")
    typer.echo(f"wrote diagnostics -> {destination}")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )
    if any(not ok for _, _, ok, _ in checks):
        raise typer.Exit(code=1)


@app.command("evidence-funnel")
def evidence_funnel(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    min_pooled: int = typer.Option(
        25, "--min-pooled", help="Production pooled minimum, for the headline."
    ),
    alpha: float = typer.Option(1e-4, "--alpha"),
    min_effect: float = typer.Option(0.05, "--min-effect"),
    thresholds: str = typer.Option(
        "1,2,5,10,25", "--thresholds", help="Comma-separated pooled minimums."
    ),
    out: Path = typer.Option(
        None,
        "--out",
        help="Per-union records "
        "[default: <data-root>/processed/evidence_funnel.parquet]",
    ),
) -> None:
    """Diagnose where network evidence disappears across proposed unions.

    Records, for every proposed union and before any evidence test, the
    pooled observations available on each side, then reports the funnel at
    several pooled minimums and the distribution of the statistic where it
    could be computed.

    Adds no mechanism and changes nothing: the frozen generator, the
    production observations, network_truth, the Phase 1 baseline and the
    merge-time constraint logic are all untouched.
    """
    from obsidianchain.eval import evidence_funnel as funnel
    from obsidianchain.io import elliptic
    from obsidianchain.network import separation

    started = time.perf_counter()
    config = separation.SeparationConfig(
        min_pooled_observations=min_pooled, alpha=alpha, min_effect=min_effect
    )
    levels = tuple(int(t) for t in thresholds.split(",") if t.strip())

    graph = elliptic.load_cospend_graph(data_root, keep_labels=True)
    oracle = separation.build_oracle(
        graph, processed_root=data_root / "processed", data_root=data_root,
        config=config,
    )
    result = funnel.build_funnel(
        graph, oracle, thresholds=levels, production_config=config,
        production_statistics_config=config,
    )
    typer.echo(funnel.format_funnel(result))

    from obsidianchain import evidence_contract as contract
    from obsidianchain.network import boundary as _boundary

    destination = out or (data_root / "processed" / "evidence_funnel.parquet")
    # PRODUCTION pipeline on the frozen dataset - but synthetic_network is
    # true, because the chain is real Elliptic++ and the announcements
    # behind every chi-square here are generated.
    network_manifest = _boundary.load_manifest(data_root / "processed")
    run_inputs, run_fp = _evidence_run_inputs(data_root, network_manifest, config)

    # ---- section 7: trajectory equivalence, VERIFIED not inferred -----
    #
    # The measurement lives in eval/trajectory.py so it can be tested; this
    # command only reports it and decides whether to publish. It used to be
    # inline here, where the one thing standing between a re-pointed
    # evidence id and a published artifact was untestable by construction.
    from obsidianchain.eval import trajectory as _trajectory

    separated = int((result.records["verdict_production"] == "SEPARATED").sum())
    equivalence = _trajectory.verify_trajectory_equivalence(
        graph, oracle, config, n_separated=separated
    )
    veto_never_fired = equivalence.equivalent
    typer.echo("")
    typer.echo(equivalence.as_report())
    if equivalence.invariant_violated:
        typer.secho(
            "INVARIANT VIOLATED: no SEPARATED verdict, yet the fused and "
            "chain-only trajectories differ. Refusing to publish.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=4)

    counts = {
        state: int((result.records["verdict_production"] == state).sum())
        for state in ("SEPARATED", "NOT_SEPARATED", "NO_EVIDENCE")
    }
    artifact_block = {
        "artifact_schema": contract.ARTIFACT_SCHEMA,
        "trajectory": "chain-only",
        "probe_statistics_config": {
            "min_pooled_observations": funnel.PROBE_CONFIG.min_pooled_observations,
            "min_observer_observations":
                funnel.PROBE_CONFIG.min_observer_observations,
        },
        "production_evaluation_config": {
            "min_pooled_observations": config.min_pooled_observations,
            "min_observer_observations": config.min_observer_observations,
            "alpha": config.alpha,
            "min_effect": config.min_effect,
            "evaluation_order": ["POOLED_GATE", "OBSERVER_GATE",
                                 "VARIANCE_GATE", "COMPUTE", "DECIDE"],
            "note": contract.VERDICT_DEFINITION,
        },
        "production_verdict_counts": counts,
        "veto_never_fired": veto_never_fired,
        # The frozen wording, from the contract rather than retyped here, so
        # a test can compare the SERVED string against the constant instead
        # of settling for a "VERIFIED:" prefix match.
        "trajectory_equivalence": equivalence.statement,
        "trajectory_equivalence_evidence": {
            "separated_verdicts": equivalence.n_separated,
            "unions_refused_by_veto": equivalence.n_blocked,
            "roots_equal": equivalence.roots_equal,
            "component_sizes_equal": equivalence.component_sizes_equal,
        },
        "not_separated_meaning": contract.NOT_SEPARATED_MEANING,
        "frozen_run_limitation": contract.FROZEN_RUN_LIMITATION,
        "verdict_scope": contract.VERDICT_SCOPE,
    }

    evidence_provenance = _provenance(
        "PRODUCTION",
        dataset_id="elliptic++/frozen-september-2026",
        synthetic_network=True,
        manifest=network_manifest,
        config=config,
        inputs=run_inputs,
        run_fingerprint=run_fp,
        artifact=artifact_block,
        notes=(
            "Network announcements are SYNTHETIC. Demonstrates the "
            "mechanism; validates nothing about Bitcoin.",
            "CHAIN-ONLY trajectory: every proposed union was applied.",
            "The thirteen probe columns were computed under PROBE_CONFIG "
            "(min_pooled=1, min_observer=2). The seven *_production columns "
            "were computed under the production rule. Both configurations "
            "are in artifact.",
        ),
    )

    # ---- sections 15/16: validate, then publish atomically ------------
    #
    # Written to a temporary directory and moved into place only after the
    # invariants pass. A half-finished regeneration must never leave a new
    # parquet beside an old sidecar, and the fingerprint must not become
    # visible before the artifact it names.
    import shutil as _shutil
    import tempfile as _tempfile

    # Staged inside the DESTINATION's directory: os.replace is atomic only
    # within one filesystem, and /tmp is a different device from the mounted
    # data root.
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(_tempfile.mkdtemp(prefix=".oc-funnel-", dir=destination.parent))
    try:
        staged = staging / destination.name
        rows = funnel.write_records(result, staged, provenance=evidence_provenance)
        _verify_funnel_artifact(staged, destination, counts)
        _publish_pair(staged, destination, run_fp)
    finally:
        _shutil.rmtree(staging, ignore_errors=True)

    typer.echo("")
    typer.echo(f"run fingerprint  {run_fp[:16]}   (full digest in the sidecar)")
    typer.echo(f"artifact schema  {contract.ARTIFACT_SCHEMA}")
    typer.echo("")
    typer.echo(f"wrote {rows:,} per-union records -> {destination}")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


@app.command("network-audit")
def network_audit(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    scenarios: bool = typer.Option(
        True, "--scenarios/--no-scenarios", help="Run the stress scenarios."
    ),
    scenario_transactions: int = typer.Option(
        4000, "--scenario-transactions", help="Transactions per stress scenario."
    ),
    sample: int = typer.Option(
        4000, "--sample", help="Transactions sampled for the distance comparison."
    ),
) -> None:
    """Audit the Phase 2 network layer before Phase 3 consumes it.

    Checks the information boundary, measures how separable the synthetic
    origins are, and runs the stress scenarios. Reads ground truth only for
    the identifiability description, which is an evaluation activity, never
    an inference one.
    """
    import numpy as np

    from obsidianchain.network import arrivals, audit, boundary, synthetic

    started = time.perf_counter()
    processed = data_root / "processed"

    leakage = audit.audit_leakage(processed)
    manifest = boundary.load_manifest(processed)

    identifiability = audit.IdentifiabilityReport()
    checks: dict[str, bool] = {}
    if leakage.boundary_error is None:
        inputs = boundary.load_phase3_inputs(processed)
        vectors = arrivals.build(
            inputs.observations,
            observer_ids=sorted(inputs.observers["observer_id"].tolist())
            or sorted(inputs.observations["observer_id"].unique().tolist()),
        )
        truth = boundary.load_ground_truth_FOR_EVALUATION_ONLY(processed)
        config = synthetic.FROZEN_SEPTEMBER_2026
        identifiability = audit.audit_identifiability(
            vectors, truth, config, sample=sample
        )

        # --- instrumentation checks -----------------------------------
        bc = set(
            inputs.observations.loc[
                inputs.observations["peer_ip"].isin(inputs.broadcaster_ips), "txid"
            ]
        )
        labels = arrivals.classify_evidence(vectors, bc)
        blocked = set(
            labels.loc[
                labels["evidence"] == arrivals.Evidence.NO_EVIDENCE.value, "txid"
            ]
        )
        checks["broadcaster_exclusion"] = bc.issubset(blocked)
        checks["vector_reconstruction"] = (
            vectors.n_transactions == inputs.observations["txid"].nunique()
        )
        checks["missing_observer_handling"] = bool(
            np.isnan(vectors.absolute_ms).any()
        ) or config.missing_observation_rate == 0.0
        checks["config_frozen"] = manifest.get("configuration", {}).get(
            "seed"
        ) == synthetic.FROZEN_SEPTEMBER_2026.seed
        checks["manifest_present"] = bool(manifest)
        checks["ground_truth_separated"] = bool(leakage.truth_files_present)

    scenario_results = []
    if scenarios and leakage.boundary_error is None:
        txids = synthetic.load_transaction_ids(data_root)[:scenario_transactions]
        scenario_results = audit.run_stress_scenarios(txids)
        checks["stress_scenarios_ran"] = len(scenario_results) > 0
        checks["broadcaster_excluded_under_stress"] = all(
            s.reasons.get("known_broadcaster", 0) > 0
            for s in scenario_results
            if s.name in {"baseline", "broadcaster-heavy"}
        )

    typer.echo(
        audit.format_audit(
            leakage, identifiability, scenario_results, manifest, checks
        )
    )
    typer.echo("")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )
    if not (leakage.clean and all(checks.values())):
        raise typer.Exit(code=1)


@app.command("network-arrivals")
def network_arrivals(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    observations: Path = typer.Option(
        None,
        "--observations",
        help="[default: <data-root>/processed/network_observations.parquet]",
    ),
    out: Path = typer.Option(
        None, "--out", help="[default: <data-root>/processed/arrival_vectors.parquet]"
    ),
    flat_threshold_ms: float = typer.Option(
        250.0, "--flat-threshold-ms", help="Spread at or below this carries no ordering."
    ),
    fmt: str = typer.Option("parquet", "--format", help="parquet or csv."),
) -> None:
    """Build per-transaction arrival-time vectors from announcement records.

    Instrumentation only: reshapes observations and describes them. It infers
    nothing and produces no constraints.
    """
    from obsidianchain.network import arrivals, boundary

    started = time.perf_counter()
    processed = data_root / "processed"

    # Read through the boundary: this path structurally cannot reach truth.
    if observations is not None:
        records = arrivals.read_observations(observations)
        boundary.assert_no_leakage(records, context=str(observations))
        observer_ids = sorted(records["observer_id"].unique().tolist())
        broadcaster_ips = boundary.load_broadcaster_ips(processed)
    else:
        inputs = boundary.load_phase3_inputs(processed)
        records = inputs.observations
        observer_ids = sorted(inputs.observers["observer_id"].tolist()) or sorted(
            records["observer_id"].unique().tolist()
        )
        broadcaster_ips = inputs.broadcaster_ips

    broadcaster_txids = set(
        records.loc[records["peer_ip"].isin(broadcaster_ips), "txid"].unique()
    )

    vectors = arrivals.build(records, observer_ids=observer_ids)
    typer.echo(
        arrivals.summarise(
            vectors, broadcaster_txids=broadcaster_txids,
            threshold_ms=flat_threshold_ms,
        )
    )

    labels = arrivals.classify_evidence(
        vectors, broadcaster_txids, threshold_ms=flat_threshold_ms
    )
    typer.echo("")
    typer.echo("-- evidence classification " + "-" * 51)
    for name, count in sorted(labels["evidence"].value_counts().items()):
        typer.echo(f"  {name:<24}{count:>12,}")
    for reason, count in labels.loc[
        labels["no_evidence_reason"] != "", "no_evidence_reason"
    ].value_counts().items():
        typer.echo(f"    reason: {reason:<16}{count:>12,}")

    suffix = ".parquet" if fmt == "parquet" else ".csv"
    destination = out or (
        processed / synthetic_dir_name() / f"arrival_vectors{suffix}"
    )
    frame = vectors.with_evidence(broadcaster_txids, threshold_ms=flat_threshold_ms)
    from obsidianchain import provenance as prov

    prov.write_frame(
        frame,
        destination,
        _provenance(
            "PRODUCTION",
            dataset_id="elliptic++/frozen-september-2026",
            synthetic_network=True,
            manifest=boundary.load_manifest(processed),
            notes=(
                "Instrumentation over SYNTHETIC announcements. Infers "
                "nothing and produces no constraints.",
            ),
        ),
    )
    rows = len(frame)
    typer.echo("")
    typer.echo(f"wrote {rows:,} arrival vectors -> {destination}")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


@app.command("entity-resolution")
def entity_resolution(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    heuristics: Heuristics = typer.Option(
        Heuristics.MULTI_INPUT, "--heuristics", help="Merge evidence to use."
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", min=0.0, max=1.0, help="Minimum change confidence."
    ),
    top: int = typer.Option(15, "--top", help="Entities to list."),
    permutations: int = typer.Option(
        1000, "--permutations", help="Label shuffles for the random baseline."
    ),
    seed: int = typer.Option(0, "--seed", help="Baseline seed."),
    out: Path = typer.Option(
        None,
        "--out",
        help="Per-entity CSV "
        "[default: <data-root>/processed/entity_resolution_<mode>.csv].",
    ),
) -> None:
    """Score cluster assignments against external entity labels.

    Requires 'entity-labels' to have been run first. Reports macro metrics as
    the headline, micro alongside, and a shuffled-label baseline.
    """
    from obsidianchain.eval import entity_resolution as er

    started = time.perf_counter()
    graph, run, _, _ = _cluster(
        data_root, heuristics, threshold, need_labels=True
    )
    result = er.evaluate(
        graph, run, data_root, n_permutations=permutations, seed=seed
    )
    typer.echo(er.format_report(result, top=top))

    destination = out or (
        data_root / "processed" / f"entity_resolution_{heuristics.value}.csv"
    )
    rows = er.write_per_entity_csv(
        result,
        destination,
        provenance=_provenance(
            "PRODUCTION",
            dataset_id="elliptic++/address_labels",
            notes=(
                "Chain and entity labels only; no network layer is involved.",
            ),
        ),
    )
    typer.echo("")
    typer.echo(f"wrote {rows:,} entity rows -> {destination}")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


@app.command("entity-labels")
def entity_labels(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    out: Path = typer.Option(
        None,
        "--out",
        help="[default: <data-root>/processed/entity_labels.csv]",
    ),
    elliptic_flag: bool = typer.Option(
        True,
        "--elliptic/--no-elliptic",
        help="Flag which labelled addresses fall inside the Elliptic++ universe.",
    ),
) -> None:
    """Validate and normalise the entity-label dataset for entity resolution.

    Reads raw/address_labels/addresses.csv, drops malformed addresses and
    addresses claimed by more than one entity, folds entity-name case, and
    writes a small processed table. Computes no metrics.
    """
    from obsidianchain.io import entity_labels as el

    started = time.perf_counter()
    frame, report = el.build(data_root, check_elliptic=elliptic_flag)
    destination = out or (data_root / "processed" / "entity_labels.csv")
    typer.echo(el.format_report(report, destination))

    rows = el.write(
        frame,
        destination,
        provenance=_provenance(
            "PRODUCTION",
            dataset_id="elliptic++/address_labels",
            notes=("Normalised real address labels; no inference.",),
        ),
    )
    size_kb = destination.stat().st_size / 1024
    typer.echo("")
    typer.echo(f"wrote {rows:,} rows ({size_kb:,.0f} KB) -> {destination}")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )
    if not report.accounted():
        raise typer.Exit(code=1)


@app.command()
def evolution(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    threshold: float = typer.Option(
        0.7, "--threshold", min=0.0, max=1.0, help="Minimum change confidence."
    ),
    timesteps: int = typer.Option(49, "--timesteps", help="Number of timesteps."),
    out_csv: Path = typer.Option(
        None, "--out-csv", help="[default: <data-root>/processed/evolution.csv]"
    ),
    out_png: Path = typer.Option(
        None, "--out-png", help="[default: <data-root>/processed/largest_cluster.png]"
    ),
    dpi: int = typer.Option(200, "--dpi", help="Chart resolution."),
    features: bool = typer.Option(True, "--features/--no-features"),
) -> None:
    """Cluster cumulatively over timesteps and chart how structure evolves.

    Runs both heuristic modes, writes the full series as CSV and the largest
    cluster trajectory as a PNG, and reports whether growth turns
    super-linear.
    """
    from obsidianchain.eval import evolution as evo

    started = time.perf_counter()
    series = evo.run_evolution(
        data_root,
        threshold=threshold,
        use_features=features,
        n_timesteps=timesteps,
    )
    onsets = evo.detect_all(series)
    typer.echo(evo.format_evolution(series, onsets))

    csv_path = out_csv or (data_root / "processed" / "evolution.csv")
    png_path = out_png or (data_root / "processed" / "largest_cluster.png")
    rows = evo.write_series_csv(
        series,
        csv_path,
        provenance=_provenance(
            "PRODUCTION",
            dataset_id="elliptic++/cospend",
            notes=("Chain only; no network layer is involved.",),
        ),
    )
    evo.plot_largest_cluster(series, png_path, onsets, dpi=dpi)

    typer.echo("")
    typer.echo(f"wrote {rows:,} rows -> {csv_path}")
    typer.echo(f"wrote chart      -> {png_path}")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


@app.command()
def cospend(
    data_root: Path = typer.Option(
        DATA_ROOT, "--data-root", help="Directory containing raw/."
    ),
    top: int = typer.Option(10, "--top", help="How many largest clusters to list."),
    heuristics: Heuristics = typer.Option(
        Heuristics.MULTI_INPUT, "--heuristics", help="Merge evidence to use."
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", min=0.0, max=1.0, help="Minimum change confidence."
    ),
) -> None:
    """Cluster addresses by co-spend and report the cluster statistics.

    Co-spend is derived from AddrTx_edgelist.csv by grouping on txId: the
    input addresses sharing a transaction are controlled by one entity.
    AddrAddr_edgelist.csv is a money-flow graph and is never used here.
    """
    from obsidianchain.io import elliptic

    started = time.perf_counter()

    typer.echo("=" * 68)
    typer.echo("obsidianchain :: co-spend clustering")
    typer.echo("=" * 68)
    typer.echo(f"source           {elliptic.COSPEND_SOURCE} (grouped on txId)")
    typer.echo(f"excluded         {elliptic.ADDR_ADDR_DO_NOT_CLUSTER} (money flow)")
    typer.echo(f"heuristics       {heuristics.value}")
    if heuristics is Heuristics.MULTI_INPUT_CHANGE:
        typer.echo(f"change threshold {threshold:.2f}")
    typer.echo(f"data root        {data_root}")
    typer.echo("")

    typer.echo("loading and clustering ...")
    load_started = time.perf_counter()
    graph, run, candidates, selected = _cluster(data_root, heuristics, threshold)
    load_seconds = time.perf_counter() - load_started

    typer.echo(f"  input rows       {graph.n_input_rows:>14,}")
    typer.echo(f"  distinct pairs   {graph.n_input_pairs:>14,}")
    typer.echo(f"  transactions     {graph.n_transactions:>14,}")
    typer.echo(f"  star edges       {graph.n_edges:>14,}")
    typer.echo(f"  load time        {load_seconds:>14.2f} s")
    if graph.n_input_only_addresses:
        typer.echo(
            f"  note: {graph.n_input_only_addresses:,} input addresses are absent "
            f"from wallets_classes.csv"
        )
    if candidates is not None:
        typer.echo("")
        typer.echo(f"  change candidates{candidates.n_candidates:>14,}")
        typer.echo(f"  above threshold  {len(candidates.above(threshold)):>14,}")
        typer.echo(f"  selected as change{len(selected):>13,}")
    typer.echo("")

    sizes = run.sizes
    merges = run.cospend_merges
    n_clusters = run.n_clusters
    largest = run.largest
    singletons = run.singletons
    clustered = run.clustered
    coverage = run.coverage
    union_seconds = 0.0
    elapsed = time.perf_counter() - started

    typer.echo("")
    typer.echo("-- results " + "-" * 57)
    typer.echo(f"  total addresses          {graph.n_addresses:>14,}")
    typer.echo(f"  total edges processed    {graph.n_edges:>14,}")
    typer.echo(f"  merges from co-spend     {merges:>14,}")
    typer.echo(f"  redundant edges          {graph.n_edges - merges:>14,}")
    if run.change_edges:
        typer.echo(f"  change edges applied     {run.change_edges:>14,}")
        typer.echo(f"  merges from change       {run.change_merges:>14,}")
        typer.echo(f"  mean change confidence   {run.change_confidence_mean:>14.3f}")
    typer.echo(f"  number of clusters       {n_clusters:>14,}")
    typer.echo("")
    typer.secho(
        f"  LARGEST CLUSTER SIZE     {largest:>14,}",
        fg=typer.colors.GREEN,
        bold=True,
    )
    typer.echo("")

    typer.echo(f"-- {top} largest clusters " + "-" * 44)
    for rank, size in enumerate(sizes[:top].tolist(), start=1):
        share = size / graph.n_addresses * 100 if graph.n_addresses else 0.0
        typer.echo(f"  #{rank:<3} {size:>12,}  ({share:.4f}% of all addresses)")
    typer.echo("")

    typer.echo("-- coverage " + "-" * 56)
    typer.echo(f"  singleton clusters       {singletons:>14,}")
    typer.echo(f"  addresses in size > 1    {clustered:>14,}")
    typer.echo(f"  coverage                 {coverage:>13.2f}%")
    typer.echo("")
    typer.echo("  Coverage well under 50% is expected, not a defect: an address")
    typer.echo("  that never appears as a transaction INPUT can never be")
    typer.echo("  co-spend clustered, and most Elliptic++ addresses appear only")
    typer.echo("  on the output side.")
    typer.echo("")

    typer.echo("-- cost " + "-" * 60)
    typer.echo(f"  peak memory              {_peak_rss_mb():>13.1f} MB")
    typer.echo(f"  load + cluster time      {load_seconds:>13.2f} s")
    typer.echo(f"  wall time                {elapsed:>13.2f} s")
    typer.echo("=" * 68)


@app.command()
def purity(
    data_root: Path = typer.Option(
        DATA_ROOT, "--data-root", help="Directory containing raw/ and processed/."
    ),
    top: int = typer.Option(10, "--top", help="Contaminated clusters to list."),
    out: Path = typer.Option(
        None,
        "--out",
        help="CSV of contaminated clusters "
        "[default: <data-root>/processed/contaminated_clusters.csv].",
    ),
    addresses: bool = typer.Option(
        True,
        "--addresses/--no-addresses",
        help="Include a representative address per cluster in the CSV. "
        "Costs ~75 MB of peak memory.",
    ),
    heuristics: Heuristics = typer.Option(
        Heuristics.MULTI_INPUT, "--heuristics", help="Merge evidence to use."
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", min=0.0, max=1.0, help="Minimum change confidence."
    ),
) -> None:
    """Measure co-spend cluster correctness against the Elliptic++ labels.

    A cluster containing both an illicit and a licit address has provably
    merged two entities. Unknown is never treated as licit.
    """
    from obsidianchain.eval import purity as purity_eval

    started = time.perf_counter()
    if heuristics is Heuristics.MULTI_INPUT:
        report, graph = purity_eval.analyse(data_root, keep_addresses=addresses)
    else:
        graph, run, _, _ = _cluster(data_root, heuristics, threshold)
        classes = purity_eval.load_classes_by_code(graph, data_root)
        report = purity_eval.report_for_roots(
            run.roots, classes, graph.n_addresses
        )
    typer.echo(f"heuristics: {heuristics.value}")
    typer.echo(purity_eval.format_summary(report, top=top))

    destination = out or (data_root / "processed" / "contaminated_clusters.csv")
    rows = purity_eval.write_contaminated_csv(
        report,
        destination,
        graph if addresses else None,
        provenance=_provenance(
            "PRODUCTION",
            dataset_id="elliptic++/wallets_classes",
            notes=("Chain and licit/illicit labels only; no network layer.",),
        ),
    )
    typer.echo("")
    typer.echo(f"wrote {rows:,} contaminated clusters -> {destination}")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


@app.command("serve")
def serve(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Serve the read-only API over precomputed artifacts.

    Binds to localhost by default: this is a forensics prototype that runs on
    one laptop, and an API that reads a quarantined data directory has no
    business listening on 0.0.0.0 without somebody deciding that on purpose.
    """
    import uvicorn

    from obsidianchain.api.app import create_app

    typer.echo(f"obsidianchain API  data root {data_root}")
    typer.echo("  read-only: no clustering, no evidence, no evaluation")
    typer.echo("  network data in this project is SYNTHETIC")
    uvicorn.run(create_app(data_root), host=host, port=port, log_level="info")


@app.command("build-cluster-index")
def build_cluster_index(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    out_dir: Path = typer.Option(
        None, "--out-dir", help="[default: <data-root>/processed]"
    ),
) -> None:
    """Persist the clustering the read-only API needs. Run once.

    Writes address_clusters.parquet (code, address, cluster_id) and
    clusters.parquet (cluster_id, size, representative_address), both
    provenance-stamped.

    Only clusters of size > 1 are written. Measured, not assumed: the codes
    referenced by evidence_funnel and phase33_decisions are exactly the
    284,709 addresses in non-singleton clusters, so the 538,233 singletons
    would add 23 MB referenced by nothing.

    This re-derives the same co-spend clustering that 'cospend' computes and
    discards. Persisting from inside 'cospend' would change that command's
    side effects, so it is a separate command rather than a silent change to
    the pipeline.
    """
    from obsidianchain.cluster import index

    started = time.perf_counter()
    summary = index.build(data_root, processed_root=out_dir)
    typer.echo(index.format_summary(summary))
    typer.echo("")
    typer.echo(f"wrote {summary['address_clusters']['path']}")
    typer.echo(f"wrote {summary['clusters']['path']}")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


@app.command("demo")
def demo(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    demo_root: Path = typer.Option(
        None, "--demo-root",
        help="Where the fixture and its outputs live "
        "[default: <data-root>/demo]. Must end in 'demo'.",
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Regenerate the fixture from the seed."
    ),
    json_out: Path = typer.Option(
        None, "--json-out", help="[default: <demo-root>/output/scenarios.json]"
    ),
    html_out: Path = typer.Option(
        None, "--html-out", help="[default: <demo-root>/output/index.html]"
    ),
) -> None:
    """Five DEMONSTRATION scenarios - SYNTHETIC, not a measurement.

    Runs the unchanged constraint engine over a seeded fixture built so that
    each of its five states is actually reached:

      A  blockchain evidence proposes a merge          -> CANDIDATE
      B  network evidence present but insufficient     -> merge proceeds
      C  network evidence strong and separating        -> merge BLOCKED
      D  contradiction after transitive clustering     -> CONTESTED
      E  no usable network evidence                    -> abstain

    Writes a DEMO-flagged JSON payload and a self-contained HTML page under
    <demo-root>. Everything is confined to that namespace; the frozen
    production dataset in processed/ is never read or written.
    """
    from obsidianchain.demo import api, report, runner, scenarios

    started = time.perf_counter()
    root = demo_root or scenarios.default_demo_root(data_root)
    try:
        result = runner.run(root, rebuild=rebuild)
    except scenarios.DemoNamespaceError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from None
    except (runner.DemoExpectationError, runner.DemoRuleError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=3) from None

    envelope = runner.to_envelope(result)
    typer.echo(report.format_terminal(envelope))

    output = Path(root) / "output"
    json_path = api.write_json(envelope, json_out or output / "scenarios.json")
    html_path = report.write_html(envelope, html_out or output / "index.html")
    typer.echo("")
    typer.echo(f"wrote DEMO-flagged payload -> {json_path}")
    typer.echo(f"wrote DEMO-marked page     -> {html_path}")
    typer.echo(
        f"peak memory {_peak_rss_mb():.1f} MB   "
        f"wall time {time.perf_counter() - started:.2f} s"
    )


def _peak_rss_mb() -> float:
    """Peak resident set size of this process, in megabytes.

    ru_maxrss is bytes on macOS but kilobytes on Linux, which is where this
    actually runs.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / scale


def _network_state() -> str:
    """Best-effort probe for outbound connectivity. Never blocks for long."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(0.25)
        # UDP connect() only sets the peer address; it sends no packet.
        # It fails immediately when no route exists, which is what
        # --network none produces.
        sock.connect(("10.255.255.255", 9))
        return f"reachable via {sock.getsockname()[0]}"
    except OSError as exc:
        return f"isolated ({exc.strerror or exc})"
    finally:
        sock.close()


def main() -> None:
    app()




# ---- Phase 6: entity risk ranking ---------------------------------------


@app.command("phase6-dataset")
def phase6_dataset(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    out: Path = typer.Option(None, "--out"),
    min_depth: int = typer.Option(
        5, "--peel-min-depth",
        help="PEEL-1 primary depth (SPEC 5.7). 10 is the sensitivity run.",
    ),
) -> None:
    """Build the Phase 6 address-level feature matrix.

    Address-level prediction, as-of-t features recomputed from the
    transaction layer, address-disjoint temporal split. Reads no
    wallets_features.csv column - SPEC 0.3 found 52 of 55 of them leak the
    future - and joins labels only after every feature is built.
    """
    from obsidianchain.features import dataset as ds

    started = time.perf_counter()
    frame, report = ds.build_dataset(data_root, min_depth=min_depth)
    destination = out or (
        data_root / "processed" / f"phase6_dataset_d{min_depth}.parquet"
    )

    typer.echo("=" * 78)
    typer.echo("OBSIDIANCHAIN - PHASE 6.0 DATASET")
    typer.echo("=" * 78)
    typer.echo(f"  addresses seen                      {report.n_addresses:>12,}")
    typer.echo(f"  dropped: spans a split boundary     {report.n_spanners_dropped:>12,}")
    typer.echo(f"  labeled rows retained               {report.n_labeled:>12,}")
    typer.echo("")
    for split in ("train", "validation", "test"):
        n = report.split_counts.get(split, 0)
        p = report.prevalence.get(split, float("nan"))
        typer.echo(f"  {split:<12} {n:>10,} rows   prevalence {p*100:6.3f}%")

    fingerprint = ds.dataset_fingerprint(data_root, min_depth)
    provenance = _provenance(
        "PRODUCTION",
        dataset_id="elliptic++/frozen-september-2026",
        synthetic_network=True,
        inputs={"phase6_dataset_sha256": fingerprint},
        run_fingerprint=fingerprint,
        artifact={
            "artifact_schema": ds.ARTIFACT_SCHEMA,
            "prediction_unit": "address",
            "label_policy": "class1=1, class2=0, class3 EXCLUDED (never licit)",
            "split": {"train": "t1-t34", "validation": "t35-t41", "test": "t42-t49"},
            "peel_min_depth": int(min_depth),
            "feature_groups": {k: len(v) for k, v in ds.FEATURE_GROUPS.items()},
            "excluded": [
                "Aggregate_feature_1..72 (UNRESOLVED provenance, SPEC 0.7)",
                "Local_feature_1..93 (anonymised, SPEC 3.3)",
                "52 of 55 wallets_features.csv columns (whole-life, SPEC 0.3)",
                "cluster_size_final (final membership is not predictive input)",
                "k-hop illicit exposure (SPEC 4.2)",
            ],
            "split_counts": report.split_counts,
            "prevalence": report.prevalence,
        },
        notes=(
            "Network features are derived from SYNTHETIC announcements. They "
            "are behavioural summaries, never ownership claims.",
            "Unknown-class addresses are EXCLUDED, never treated as licit.",
            "Features are as-of the address's last active timestep; "
            "boundary-spanning addresses are removed entirely.",
        ),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _publish_frame(frame, destination, provenance, fingerprint)

    typer.echo("")
    typer.echo(f"dataset fingerprint  {fingerprint[:16]}")
    typer.echo(f"wrote {len(frame):,} rows x {len(frame.columns)} columns -> {destination}")
    typer.echo(f"peak memory {_peak_rss_mb():.1f} MB   "
               f"wall time {time.perf_counter() - started:.1f} s")


def _publish_frame(frame, destination: Path, provenance, fingerprint: str) -> None:
    """Stage, then move both halves into place (the Phase 5.3-B discipline)."""
    import shutil as _shutil
    import tempfile as _tempfile

    from obsidianchain import provenance as prov

    staging = Path(_tempfile.mkdtemp(prefix=".oc-phase6-", dir=destination.parent))
    try:
        staged = staging / destination.name
        prov.write_frame(frame, staged, provenance)
        _publish_pair(staged, destination, fingerprint)
    finally:
        _shutil.rmtree(staging, ignore_errors=True)


@app.command("phase6-experiment")
def phase6_experiment(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    dataset_path: Path = typer.Option(None, "--dataset"),
    min_depth: int = typer.Option(5, "--peel-min-depth"),
) -> None:
    """Run the M0-M3 ablation, cluster aggregation and P1-P4.

    Every threshold, label definition and prediction was frozen in
    docs/PHASE6_SPEC.md before this ran. Nothing here is selected on the
    strength of a result.
    """
    import pandas as pd

    from obsidianchain.features import dataset as ds
    from obsidianchain.ml import experiment, metrics, model

    started = time.perf_counter()
    path = dataset_path or (
        data_root / "processed" / f"phase6_dataset_d{min_depth}.parquet"
    )
    if not path.is_file():
        typer.secho(
            f"{path} not found. Run 'phase6-dataset' first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    frame = pd.read_parquet(path)

    typer.echo("=" * 78)
    typer.echo(f"OBSIDIANCHAIN - PHASE 6.0 EXPERIMENT   (PEEL-1 depth >= {min_depth})")
    typer.echo("=" * 78)

    stages = experiment.run_ablation(frame)
    typer.echo("")
    typer.echo("-- ablation, TEST split " + "-" * 54)
    typer.echo(f"  {'stage':<14}{'PR-AUC':>9}{'base':>8}{'nAP':>8}{'ROC':>8}"
               f"{'P@10':>7}{'P@50':>7}{'P@100':>7}{'Brier':>8}")
    previous = None
    for s in stages:
        r = s.test
        typer.echo(
            f"  {s.name:<14}{r.pr_auc:>9.4f}{r.pr_auc_baseline:>8.4f}"
            f"{r.normalised_ap:>8.4f}{r.roc_auc:>8.4f}"
            f"{r.precision_at[10]:>7.2f}{r.precision_at[50]:>7.2f}"
            f"{r.precision_at[100]:>7.2f}{r.brier:>8.5f}"
        )
        previous = r
    typer.echo("")
    for a, b in zip(stages, stages[1:]):
        typer.echo(f"  delta({b.name} - {a.name}) PR-AUC "
                   f"{b.test.pr_auc - a.test.pr_auc:+.5f}")

    trained, scored = experiment.fit_full(frame)
    typer.echo("")
    typer.echo("-- severity bands (validation only, SPEC 6.3.1) " + "-" * 30)
    for band in trained.bands:
        if band.populated:
            typer.echo(f"  {band.name:<9} target {band.target:.2f}  "
                       f"threshold {band.threshold:.6f}  "
                       f"support {band.support:>6,}  "
                       f"validation precision {band.precision:.4f}")
        else:
            typer.echo(f"  {band.name:<9} target {band.target:.2f}  UNPOPULATED "
                       f"- no threshold meets the target at support >= "
                       f"{model.MIN_SUPPORT}")
    counts = pd.Series(scored["severity"]).value_counts().to_dict()
    typer.echo(f"  test severity distribution: {counts}")

    clusters_path = data_root / "processed" / "address_clusters.parquet"
    cluster_table = None
    if clusters_path.is_file():
        index = pd.read_parquet(clusters_path, columns=["address", "cluster_id"])
        cluster_table = experiment.aggregate_clusters(scored, index)
        cluster_metrics = experiment.evaluate_clusters(cluster_table)
        typer.echo("")
        typer.echo("-- cluster aggregation, TEST members " + "-" * 41)
        typer.echo(f"  clusters {len(cluster_table):,}   "
                   f"positive (share>=0.10) {int(cluster_table['label_primary'].sum()):,}   "
                   f"positive (>=1 illicit) {int(cluster_table['label_sensitivity'].sum()):,}")
        typer.echo(f"  {'label':<24}{'agg':<11}{'PR-AUC':>9}{'P@10':>7}{'P@50':>7}"
                   f"{'P@100':>7}{'R@10':>7}{'R@50':>7}{'R@100':>7}")
        for row in cluster_metrics.itertuples():
            typer.echo(
                f"  {row.cluster_label:<24}{row.aggregation:<11}{row.pr_auc:>9.4f}"
                f"{row.precision_at_10:>7.2f}{row.precision_at_50:>7.2f}"
                f"{row.precision_at_100:>7.2f}{row.recall_at_10:>7.3f}"
                f"{row.recall_at_50:>7.3f}{row.recall_at_100:>7.3f}"
            )
    else:
        cluster_metrics = pd.DataFrame()

    predictions = experiment.evaluate_predictions(
        stages, cluster_table if cluster_table is not None else pd.DataFrame(),
        scored,
    )
    typer.echo("")
    typer.echo("-- pre-registered predictions " + "-" * 48)
    for row in predictions.itertuples():
        verdict = "HELD" if row.held else "REFUTED"
        typer.echo(f"  {row.prediction}  {verdict:<8} observed "
                   f"{row.observed:+.5f}  threshold {row.threshold}")
        typer.echo(f"      {row.statement}")
        detail = getattr(row, "detail", None)
        if isinstance(detail, str):
            typer.echo(f"      {detail}")

    contributions = model.shap_contributions(trained, scored)
    mean_abs = contributions.drop(columns=["base_value"]).abs().mean()
    typer.echo("")
    typer.echo("-- SHAP, mean |contribution| in log-odds, top 12 " + "-" * 29)
    for name, value in mean_abs.sort_values(ascending=False).head(12).items():
        typer.echo(f"  {value:>8.4f}  {name}")

    fingerprint = ds.dataset_fingerprint(data_root, min_depth)
    processed = data_root / "processed"
    _write_phase6_outputs(
        processed, min_depth, fingerprint, stages, scored, cluster_metrics,
        predictions, mean_abs, trained,
    )
    typer.echo("")
    typer.echo(f"wall time {time.perf_counter() - started:.1f} s   "
               f"peak memory {_peak_rss_mb():.1f} MB")


def _write_phase6_outputs(processed: Path, min_depth: int, fingerprint: str,
                          stages, scored, cluster_metrics, predictions,
                          mean_abs, trained) -> None:
    """Persist every Phase 6 result with provenance."""
    import pandas as pd

    from obsidianchain.features import dataset as ds
    from obsidianchain.ml import metrics as ml_metrics

    suffix = f"_d{min_depth}"
    ablation = pd.DataFrame([s.test.as_dict() for s in stages])
    ablation["stage"] = [s.name for s in stages]
    ablation["n_features"] = [len(s.features) for s in stages]

    bands = pd.DataFrame([
        {"band": b.name, "target": b.target, "threshold": b.threshold,
         "support": b.support, "validation_precision": b.precision,
         "populated": b.populated}
        for b in trained.bands
    ])
    calibration = ml_metrics.calibration_curve(scored["y"], scored["risk"])
    shap_frame = mean_abs.rename("mean_abs_contribution").reset_index()
    shap_frame.columns = ["feature", "mean_abs_contribution"]

    artifact = {
        "artifact_schema": "obsidianchain.phase6_results/1",
        "peel_min_depth": int(min_depth),
        "cluster_label_primary": "illicit_share_among_labeled >= 0.10",
        "cluster_label_sensitivity": ">=1 illicit labeled member",
        "predictions": predictions.to_dict("records"),
        "note": (
            "PEEL-1 is structural candidate generation, not laundering "
            "classification. Network features are behavioural summaries of "
            "SYNTHETIC announcements, never ownership claims."
        ),
    }
    for name, frame in (
        (f"phase6_ablation{suffix}.csv", ablation),
        (f"phase6_severity_bands{suffix}.csv", bands),
        (f"phase6_calibration{suffix}.csv", calibration),
        (f"phase6_cluster_metrics{suffix}.csv", cluster_metrics),
        (f"phase6_predictions{suffix}.csv", predictions),
        (f"phase6_shap{suffix}.csv", shap_frame),
    ):
        if frame is None or len(frame) == 0:
            continue
        provenance = _provenance(
            "PRODUCTION",
            dataset_id="elliptic++/frozen-september-2026",
            synthetic_network=True,
            inputs={"phase6_dataset_sha256": fingerprint},
            run_fingerprint=fingerprint,
            artifact=artifact,
            notes=(
                "Network features are SYNTHETIC in origin and are not "
                "ownership claims.",
                "Unknown-class addresses were excluded, never treated licit.",
            ),
        )
        from obsidianchain import provenance as prov

        prov.write_frame(frame, processed / name, provenance)


# ---- Phase 7: investigation and alert layer ------------------------------


@app.command("phase7-alerts")
def phase7_alerts(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    dataset_path: Path = typer.Option(None, "--dataset"),
    min_depth: int = typer.Option(5, "--peel-min-depth"),
) -> None:
    """Generate the five Phase 7 alert artifacts.

    Scoring runs HERE, once, not in a request handler. The API's guarantee is
    that a response is a file the pipeline already wrote, and
    api/boundary.py makes obsidianchain.ml and obsidianchain.features
    unreachable by import so that stays true.

    Alerts are generated from the TEST split only: train and validation
    scores are optimistic by construction, and ranking them beside test
    scores would put the most confidently wrong rows at the top of an
    investigator's queue.
    """
    import pandas as pd

    from obsidianchain.alerts import build as alert_build
    from obsidianchain.alerts import contract as alert_contract
    from obsidianchain.features import dataset as ds

    started = time.perf_counter()
    path = dataset_path or (
        data_root / "processed" / f"phase6_dataset_d{min_depth}.parquet"
    )
    if not path.is_file():
        typer.secho(
            f"{path} not found. Run 'phase6-dataset' first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    frame = pd.read_parquet(path)
    fingerprint = ds.dataset_fingerprint(data_root, min_depth)

    typer.echo("=" * 78)
    typer.echo("OBSIDIANCHAIN - PHASE 7.0 ALERT ARTIFACTS")
    typer.echo("=" * 78)
    typer.echo("  An alert is a co-spend CLUSTER, not a person or an account.")
    typer.echo("  Network observations are SYNTHETIC and are investigative")
    typer.echo("  context only - never an ownership or identity claim.")
    typer.echo("")

    artifacts, summary = alert_build.build(data_root, frame, fingerprint)
    named = artifacts.named()
    alert_build.assert_no_labels(named)

    typer.echo(f"  alerts                    {summary['n_alerts']:>12,}")
    typer.echo(f"  scored member addresses   {summary['n_members']:>12,}")
    typer.echo(f"  explanation rows          {len(artifacts.explanations):>12,}")
    typer.echo(f"  timeline points           {len(artifacts.timeline):>12,}")
    typer.echo(f"  relationship edges        {len(artifacts.relationships):>12,}")
    typer.echo(f"  network correlation rows  {len(artifacts.network):>12,}")
    typer.echo(f"  correlated transactions   {summary['n_correlated_transactions']:>12,}")
    typer.echo(f"  ranking aggregation       {summary['ranking_aggregation']:>12}")
    typer.echo("")
    for band, count in sorted(summary["severity_counts"].items()):
        typer.echo(f"  severity {band:<10} {count:>12,}")

    # Each table declares its OWN content schema. Stamping the index's
    # schema on all five would leave a reader unable to tell, from the
    # sidecar alone, which layout a file actually has.
    table_schema = {
        "alerts": alert_contract.ALERT_SCHEMA,
        "alert_members": alert_contract.MEMBER_SCHEMA,
        "alert_explanations": alert_contract.EXPLANATION_SCHEMA,
        "alert_timeline": alert_contract.TIMELINE_SCHEMA,
        "alert_relationships": alert_contract.RELATIONSHIP_SCHEMA,
        "alert_network": alert_contract.NETWORK_SCHEMA,
    }
    artifact_block = {
        "index_schema": alert_contract.ALERT_SCHEMA,
        "model": {
            "family": "LightGBM",
            "calibration": "isotonic, fitted on the VALIDATION split only",
            "n_features": summary["n_features"],
            "explainability": "TreeSHAP via LightGBM pred_contrib (log-odds)",
        },
        "scored_split": "test",
        "split": {"train": "t1-t34", "validation": "t35-t41", "test": "t42-t49"},
        "feature_semantics": (
            "Every feature is as-of the address's LAST active timestep, "
            "computed only from transactions with Time step <= t. "
            "Boundary-spanning addresses were removed entirely, so the "
            "splits are address-disjoint."
        ),
        "ranking_aggregation": summary["ranking_aggregation"],
        "aggregations_served": list(alert_contract.AGGREGATIONS),
        "severity_bands": summary["bands"],
        "explanation_categories": list(alert_contract.CATEGORIES),
        "relationships_supported": list(alert_contract.RELATIONSHIPS),
        "announcing_peer_meaning": alert_contract.ANNOUNCING_PEER_MEANING,
        "alert_meaning": alert_contract.ALERT_MEANING,
        "network_context_meaning": alert_contract.NETWORK_CONTEXT_MEANING,
        "score_scope": alert_contract.SCORE_SCOPE,
        "peel_min_depth": int(min_depth),
    }
    notes = (
        "Network observations are SYNTHETIC and are investigative context "
        "only. They never establish that an IP owns, controls or sent from "
        "a wallet.",
        "A cluster is an inference from the common-input-ownership "
        "heuristic, not a person or a legal entity.",
        "No artifact here carries a ground-truth label: on real data there "
        "is none, and displaying one would make this a label viewer.",
        "Scores are out-of-sample; only the TEST split is ranked.",
    )

    processed = data_root / "processed"
    for name, table in named.items():
        destination = processed / f"{name}.parquet"
        provenance = _provenance(
            "PRODUCTION",
            dataset_id="elliptic++/frozen-september-2026",
            synthetic_network=True,
            inputs={"phase6_dataset_sha256": fingerprint},
            run_fingerprint=fingerprint,
            artifact={
                **artifact_block, "table": name,
                "artifact_schema": table_schema[name],
            },
            notes=notes,
        )
        _publish_frame(table, destination, provenance, fingerprint)

    typer.echo("")
    typer.echo(f"run fingerprint  {fingerprint[:16]}")
    typer.echo(f"wrote {len(named)} artifacts -> {processed}")
    typer.echo(f"peak memory {_peak_rss_mb():.1f} MB   "
               f"wall time {time.perf_counter() - started:.1f} s")


# ---- the operational console: accounts and database -----------------------
#
# Account creation lives here and not behind an unauthenticated HTTP route.
# An offline forensic workstation has no reason to let an anonymous caller
# mint an identity, and the absence of that path is the simplest way to
# guarantee it. The first ADMIN is therefore created by whoever has shell
# access to the machine, which is the correct trust root for this deployment.


@app.command("console-init")
def console_init() -> None:
    """Create or migrate the application database. Idempotent."""
    from obsidianchain.console import db as console_db

    conn = console_db.connect(DATA_ROOT)
    try:
        version = console_db.schema_version(conn)
        tables = [
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
                " AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
    finally:
        conn.close()

    typer.echo(f"database      {console_db.database_path(DATA_ROOT)}")
    typer.echo(f"schema        v{version}")
    typer.echo(f"tables        {', '.join(tables)}")
    typer.echo("")
    typer.echo("This database holds MUTABLE APPLICATION STATE only.")
    typer.echo("No analytical artifact is ever written to it.")


@app.command("console-user-add")
def console_user_add(
    username: str = typer.Argument(..., help="Login name."),
    role: str = typer.Option(
        "INVESTIGATOR", "--role",
        help="ADMIN, INVESTIGATOR or REVIEWER.",
    ),
    display_name: str = typer.Option("", "--display-name"),
    password: str = typer.Option(
        None, "--password",
        help="Omit to be prompted. Prompting keeps it out of shell history.",
    ),
) -> None:
    """Create a console account."""
    from obsidianchain.console import db as console_db, errors, users

    if password is None:
        password = typer.prompt("Password", hide_input=True,
                                confirmation_prompt=True)

    conn = console_db.connect(DATA_ROOT)
    try:
        created = users.create(
            conn, username=username, password=password, role=role,
            display_name=display_name,
        )
    except errors.ConsoleError as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()

    typer.echo(f"created  {created.username}  {created.role.value}  "
               f"({created.id})")


@app.command("console-user-list")
def console_user_list() -> None:
    """List console accounts. Never prints a password hash."""
    from obsidianchain.console import db as console_db, users

    conn = console_db.connect(DATA_ROOT)
    try:
        found = users.listing(conn)
    finally:
        conn.close()

    if not found:
        typer.echo("no accounts. Create one with 'console-user-add'.")
        return
    for user in found:
        state = "active" if user.active else "DISABLED"
        typer.echo(f"{user.username:<24} {user.role.value:<14} {state:<9} "
                   f"{user.id}")


@app.command("console-user-password")
def console_user_password(
    username: str = typer.Argument(...),
    password: str = typer.Option(None, "--password"),
) -> None:
    """Reset an account's password and revoke its open sessions.

    Revocation is not optional: a password changed because it may be known
    to someone else has achieved nothing while that person's session is
    still open.
    """
    from obsidianchain.console import db as console_db, errors, sessions, users

    if password is None:
        password = typer.prompt("New password", hide_input=True,
                                confirmation_prompt=True)

    conn = console_db.connect(DATA_ROOT)
    try:
        user = users.by_username(conn, username)
        if user is None:
            typer.echo(f"no account {username!r}", err=True)
            raise typer.Exit(code=1)
        users.set_password(conn, user.id, password)
        revoked = sessions.revoke_all_for_user(conn, user.id)
    except errors.ConsoleError as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(code=1)
    finally:
        conn.close()

    typer.echo(f"password updated for {username}; {revoked} session(s) revoked")


@app.command("demo-reset")
def demo_reset(
    confirm: bool = typer.Option(
        False, "--confirm",
        help="Acknowledge that mutable casework database and uploads will be wiped.",
    ),
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
) -> None:
    """Deterministically initialize a clean demonstration database.

    Wipes mutable cases and uploads, runs all migrations, and generates
    fresh temporary credentials for Admin, Investigator, and Reviewer.
    Leaves analytical and research artifacts (data/raw/, data/processed/) untouched.
    """
    from obsidianchain.console.demo_reset import reset_clean_demo_database

    if not confirm:
        typer.echo("Error: --confirm flag is required to reset the demo database.", err=True)
        typer.echo("This will wipe all mutable casework, sessions, and uploads.", err=True)
        raise typer.Exit(code=1)

    creds = reset_clean_demo_database(data_root, confirm=True)
    typer.echo("=" * 64)
    typer.echo("OBSIDIANCHAIN — CLEAN DEMO DATABASE INITIALIZED")
    typer.echo("=" * 64)
    typer.echo("Database:   Clean (schema v4)")
    typer.echo("Casework:   0 cases, 0 uploaded datasets")
    typer.echo("Analytics:  Preserved untouched (data/processed)")
    typer.echo("-" * 64)
    typer.echo(f"{'ROLE':<14} {'USERNAME':<16} {'TEMPORARY PASSWORD'}")
    typer.echo("-" * 64)
    for role_name in ("admin", "investigator", "reviewer"):
        info = creds[role_name]
        typer.echo(f"{info['role']:<14} {info['username']:<16} {info['password']}")
    typer.echo("=" * 64)
    typer.echo("Note: Save these temporary credentials for this demonstration session.")


# ---- structural pattern scan (additive, never mutates an alert artifact) --


@app.command("mixing-scan")
def mixing_scan(
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
) -> None:
    """Classify every transaction's mixing / CoinJoin-like structure.

    Writes ``processed/tx_mixing.parquet``. ADDITIVE: it touches none of the
    five Phase 7 alert artifacts, changes no run fingerprint and therefore
    invalidates no alert id, no case binding and no stored alert reference.
    An investigator can see the structural pattern behind an existing alert
    without the whole analytical run being regenerated under them.

    The classification is about transaction STRUCTURE. It does not assert
    that a mixing service was used, and using one is not itself unlawful.
    """
    import hashlib

    import pandas as pd

    from obsidianchain import run_fingerprint as rf
    from obsidianchain.features import mixing
    from obsidianchain.features.incidence import TX_COLUMNS

    started = time.perf_counter()
    raw = data_root / "raw" / "txs_features.csv"
    if not raw.is_file():
        typer.echo(f"{raw} not found", err=True)
        raise typer.Exit(code=1)

    transactions = pd.read_csv(raw, usecols=TX_COLUMNS)
    scored = mixing.score_transactions(transactions)
    scored = scored[mixing.TX_COLUMNS]

    counts = scored["mixing_class"].value_counts().to_dict()
    suppressed = scored["suppressor"].value_counts().to_dict()
    suppressed.pop(mixing.SUPPRESSOR_NONE, None)

    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(scored, index=False).values.tobytes()
    ).hexdigest()

    provenance = _provenance(
        "PRODUCTION",
        dataset_id="elliptic++/frozen-september-2026",
        inputs={"txs_features_sha256": rf.sha256_file(raw),
                "detector": mixing.DEFAULT_CONFIG.label},
        run_fingerprint=fingerprint,
        artifact={
            "artifact_schema": "obsidianchain.tx_mixing/1",
            "detector": mixing.DEFAULT_CONFIG.label,
            "signal_weights": dict(mixing.SIGNAL_WEIGHTS),
            "classes": list(mixing.CLASSES),
            "suppressors": dict(mixing.SUPPRESSORS),
            "meaning": mixing.MEANING,
            "insufficient_data_meaning": mixing.INSUFFICIENT_DATA_MEANING,
        },
        notes=(
            "A mixing-like pattern is an observation about transaction "
            "structure. It is not proof that a mixing service was used, and "
            "mixing is not itself unlawful.",
            "Individual output values are not present in the source data; "
            "uniformity is measured from the min/max/mean summary and no "
            "claim is made beyond it.",
            "This artifact is ADDITIVE. It does not participate in the "
            "Phase 7 alert run fingerprint and changes no alert id.",
        ),
    )

    destination = data_root / "processed" / "tx_mixing.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _publish_frame(scored, destination, provenance, fingerprint)

    typer.echo(f"scanned {len(scored):,} transactions")
    for name in mixing.CLASSES:
        typer.echo(f"  {name:<20} {counts.get(name, 0):>8,}")
    typer.echo("")
    typer.echo("suppressed as a benign shape:")
    for code, n in sorted(suppressed.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {code:<24} {n:>8,}")
    typer.echo("")
    typer.echo(f"detector        {mixing.DEFAULT_CONFIG.label}")
    typer.echo(f"scan id         {fingerprint[:16]}")
    typer.echo(f"wrote           {destination}")
    typer.echo(f"wall time       {time.perf_counter() - started:.1f} s")
    typer.echo("")
    typer.echo("A mixing-like pattern is a transaction STRUCTURE, not a")
    typer.echo("finding about any person and not evidence of an offence.")


# ---- the coherent TXID-correlated synthetic world -------------------------
#
# Named 'synthetic-world-*' to stay clearly apart from 'world-generate',
# which builds the Phase 3.3 controlled network worlds A-E. Those are a
# network-only experiment; this is a chain AND network world whose two layers
# share transactions because both were produced from the same ones.


@app.command("synthetic-world-build")
def synthetic_world_build(
    out: Path = typer.Option(None, "--out",
                             help="Defaults to <data-root>/synthetic_world."),
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
    seed: int = typer.Option(None, "--seed"),
    entities_per_behaviour: int = typer.Option(None, "--entities"),
) -> None:
    """Generate the coherent synthetic world. SYNTHETIC_CONTROL throughout.

    Writes an observable chain layer in the three filenames the Phase 6
    loader already reads, an observable network layer over the SAME txids,
    and a quarantined world_truth/ directory no inference stage may open.

    Nothing here touches the production artifacts. Every number measured on
    this world is a property of a generator, not of Bitcoin.
    """
    from obsidianchain.world import behaviours as bx
    from obsidianchain.world.generate import WorldConfig, write_world

    destination = out or (data_root / "synthetic_world")
    overrides = {}
    if seed is not None:
        overrides["seed"] = seed
    if entities_per_behaviour is not None:
        overrides["entities_per_behaviour"] = entities_per_behaviour

    started = time.perf_counter()
    manifest = write_world(destination, WorldConfig(**overrides))

    typer.echo(f"world            {destination}")
    typer.echo(f"fingerprint      {manifest['world_fingerprint'][:16]}")
    typer.echo(f"seed             {manifest['seed']}")
    for key, value in manifest["counts"].items():
        typer.echo(f"  {key:<14} {value:>8,}")
    typer.echo("")
    typer.echo(f"behaviours       {len(bx.BEHAVIOURS)} "
               f"({len(bx.ADVERSARIAL)} adversarial)")
    typer.echo(f"positive class   {', '.join(bx.POSITIVE_CLASS)}")
    typer.echo(f"wall time        {time.perf_counter() - started:.1f} s")
    typer.echo("")
    typer.echo("SYNTHETIC_CONTROL. The positive-class designation is a")
    typer.echo("labelling convention for a controlled experiment, not a claim")
    typer.echo("that any behaviour is unlawful.")


@app.command("synthetic-world-overlap")
def synthetic_world_overlap(
    world: Path = typer.Option(None, "--world",
                               help="Defaults to <data-root>/synthetic_world."),
    data_root: Path = typer.Option(DATA_ROOT, "--data-root"),
) -> None:
    """Measure network-evidence coverage as chain/network overlap falls.

    Runs the REAL M3 builder at each level rather than a bespoke coverage
    calculation, so what is reported is what the pipeline actually sees.

    Falling coverage is expected. The property under test is that missing
    evidence stays missing - it must never become a negative finding.
    """
    from obsidianchain.world import overlap as overlap_mod

    root = world or (data_root / "synthetic_world")
    if not (root / "raw" / "txs_features.csv").is_file():
        typer.echo(f"no world at {root}; run 'synthetic-world-build' first",
                   err=True)
        raise typer.Exit(code=1)

    started = time.perf_counter()
    destination = root / "overlap_experiment.json"
    payload = overlap_mod.write(root, destination)

    typer.echo(f"{'overlap':>8} {'tx obs':>8} {'coverage':>9} "
               f"{'abstention':>11}  evidence")
    for level in payload["levels"]:
        state = "available" if level["evidence_available"] else "UNAVAILABLE"
        typer.echo(
            f"{level['observed_overlap']:>8.2f} "
            f"{level['transactions_with_observations']:>8,} "
            f"{level['coverage']:>9.3f} {level['abstention']:>11.3f}  {state}"
        )
    typer.echo("")
    typer.echo(f"wrote            {destination}")
    typer.echo(f"wall time        {time.perf_counter() - started:.1f} s")
    typer.echo("")
    typer.echo("SYNTHETIC evaluation. Lower overlap is not claimed to cause a")
    typer.echo("detection outcome; coverage and abstention are what is")
    typer.echo("measured. Missing evidence is not negative evidence.")


# ---- model registry: list / verify / register / promote / rollback / health ----

model_app = typer.Typer(no_args_is_help=True, help="Model registry and model health.")
app.add_typer(model_app, name="model")


def _registry(root: Path | None):
    from obsidianchain.ml import registry
    return registry.Registry.open(root)


@model_app.command("list")
def model_list(root: Path = typer.Option(None, help="registry directory")) -> None:
    """Registered versions and the role each holds."""
    reg = _registry(root)
    roles = {v: r for r, v in reg.data["roles"].items() if v}
    for version, entry in sorted(reg.data["models"].items()):
        typer.echo(f"{version:<18} {roles.get(version, '-'):<10} {entry['feature_schema_version']:<22} "
                   f"{entry['lineage'].get('source_commit', '?')}")


@model_app.command("verify")
def model_verify(version: str, root: Path = typer.Option(None)) -> None:
    """Check every artifact file against its registered hash."""
    _registry(root).verify(version)
    typer.echo(f"{version}: all artifacts match the registry")


@model_app.command("register")
def model_register(version: str, path: str, root: Path = typer.Option(None)) -> None:
    """Freeze a trained model directory as an immutable registered version."""
    import json as _json
    reg = _registry(root)
    manifest = _json.loads((reg.root / path / "manifest.json").read_text())
    if manifest.get("model_version") != version:
        raise typer.BadParameter(f"manifest says {manifest.get('model_version')}, not {version}")
    reg.register(version, path, feature_schema_version=manifest["feature_schema_version"],
                 lineage=manifest.get("lineage", {}), notes=manifest.get("status", ""))
    reg.save()
    typer.echo(f"registered {version}")


@model_app.command("promote")
def model_promote(version: str, role: str = typer.Option("champion"),
                  reason: str = typer.Option(..., help="why; recorded permanently"),
                  gate_report: Path = typer.Option(None, help="production gate report required for champion"),
                  root: Path = typer.Option(None)) -> None:
    """Assign a role. Champion requires a passing production gate report for this version."""
    import json as _json
    reg = _registry(root)
    if role == "champion":
        if gate_report is None or not gate_report.is_file():
            raise typer.BadParameter("promotion to champion needs --gate-report")
        gate = _json.loads(gate_report.read_text())
        if gate.get("model_version") != version or gate.get("decision") != "PASS":
            raise typer.BadParameter(f"gate report does not PASS {version}: {gate.get('decision')}")
        reason = f"{reason} [gate {gate_report.name}: PASS]"
    reg.assign(role, version, reason)
    reg.save()
    typer.echo(f"{version} -> {role}")


@model_app.command("rollback")
def model_rollback(reason: str = typer.Option(...), root: Path = typer.Option(None)) -> None:
    """Swap champion and fallback, after verifying the fallback."""
    from obsidianchain.pipeline.features_ps import PS_FEATURE_SCHEMA_VERSION
    reg = _registry(root)
    served = reg.rollback(reason, PS_FEATURE_SCHEMA_VERSION)
    reg.save()
    typer.echo(f"champion is now {served}")


@model_app.command("attest")
def model_attest(version: str, commit: str = typer.Option("HEAD"), root: Path = typer.Option(None)) -> None:
    """Verify that the training code recorded in the manifest is exactly the code at COMMIT."""
    import subprocess
    sha = subprocess.run(["git", "rev-parse", commit], capture_output=True, text=True, check=True).stdout.strip()

    def read_blob(c: str, path: str) -> bytes:
        return subprocess.run(["git", "show", f"{c}:{path}"], capture_output=True, check=True).stdout

    reg = _registry(root)
    reg.attest(version, sha, read_blob)
    reg.save()
    typer.echo(f"{version}: training code verified at {sha}")


@model_app.command("health")
def model_health_cmd(run_dir: Path, labels: Path) -> None:
    """Score a past run against labels that arrived later (model_health.json)."""
    from obsidianchain.ml.delayed_labels import write_model_health
    out = write_model_health(run_dir, labels)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(app())
