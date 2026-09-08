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

DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))


def _provenance(
    kind: str,
    dataset_id: str,
    *,
    world=None,
    synthetic_network=None,
    manifest=None,
    config=None,
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
        graph, oracle, thresholds=levels, production_config=config
    )
    typer.echo(funnel.format_funnel(result))

    from obsidianchain.network import boundary as _boundary

    destination = out or (data_root / "processed" / "evidence_funnel.parquet")
    # PRODUCTION pipeline on the frozen dataset - but synthetic_network is
    # true, because the chain is real Elliptic++ and the announcements
    # behind every chi-square here are generated.
    rows = funnel.write_records(
        result,
        destination,
        provenance=_provenance(
            "PRODUCTION",
            dataset_id="elliptic++/frozen-september-2026",
            synthetic_network=True,
            manifest=_boundary.load_manifest(data_root / "processed"),
            config=config,
            notes=(
                "Network announcements are SYNTHETIC. Demonstrates the "
                "mechanism; validates nothing about Bitcoin.",
            ),
        ),
    )
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


if __name__ == "__main__":
    sys.exit(app())
