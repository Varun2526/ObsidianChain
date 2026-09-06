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
from pathlib import Path

import typer

from obsidianchain import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Offline Bitcoin forensics prototype (NTRO PS 26146).",
)

DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))


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


@app.command()
def cospend(
    data_root: Path = typer.Option(
        DATA_ROOT, "--data-root", help="Directory containing raw/."
    ),
    top: int = typer.Option(10, "--top", help="How many largest clusters to list."),
) -> None:
    """Cluster addresses by co-spend and report the cluster statistics.

    Co-spend is derived from AddrTx_edgelist.csv by grouping on txId: the
    input addresses sharing a transaction are controlled by one entity.
    AddrAddr_edgelist.csv is a money-flow graph and is never used here.
    """
    from obsidianchain.cluster.unionfind import UnionFind
    from obsidianchain.io import elliptic

    started = time.perf_counter()

    typer.echo("=" * 68)
    typer.echo("obsidianchain :: co-spend clustering")
    typer.echo("=" * 68)
    typer.echo(f"source           {elliptic.COSPEND_SOURCE} (grouped on txId)")
    typer.echo(f"excluded         {elliptic.ADDR_ADDR_DO_NOT_CLUSTER} (money flow)")
    typer.echo(f"data root        {data_root}")
    typer.echo("")

    typer.echo("loading ...")
    load_started = time.perf_counter()
    graph = elliptic.load_cospend_graph(data_root)
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
    typer.echo("")

    typer.echo("clustering ...")
    union_started = time.perf_counter()
    uf = UnionFind(graph.n_addresses)
    _require_implemented(uf)
    merges = uf.add_edges(graph.edges)
    sizes = uf.component_sizes()
    union_seconds = time.perf_counter() - union_started

    n_clusters = int(sizes.size)
    largest = int(sizes[0]) if n_clusters else 0
    singletons = int((sizes == 1).sum())
    clustered = int(sizes[sizes > 1].sum())
    coverage = (clustered / graph.n_addresses * 100) if graph.n_addresses else 0.0
    elapsed = time.perf_counter() - started

    typer.echo("")
    typer.echo("-- results " + "-" * 57)
    typer.echo(f"  total addresses          {graph.n_addresses:>14,}")
    typer.echo(f"  total edges processed    {graph.n_edges:>14,}")
    typer.echo(f"  edges that merged        {merges:>14,}")
    typer.echo(f"  redundant edges          {graph.n_edges - merges:>14,}")
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
    typer.echo(f"  load time                {load_seconds:>13.2f} s")
    typer.echo(f"  union-find time          {union_seconds:>13.2f} s")
    typer.echo(f"  wall time                {elapsed:>13.2f} s")
    typer.echo("=" * 68)


def _require_implemented(uf: object) -> None:
    """Fail clearly while find()/union() are still stubs.

    Remove once they are implemented; it is a scaffolding guard, not logic.
    """
    if uf.find(0) is None:  # type: ignore[attr-defined]
        typer.secho(
            "UnionFind.find() returned None - find() and union() are not yet "
            "implemented in src/obsidianchain/cluster/unionfind.py.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)


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
