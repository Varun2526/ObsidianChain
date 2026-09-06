"""obsidianchain command line entry point.

Runs fully air-gapped. No command here may perform network I/O.
"""

from __future__ import annotations

import os
import platform
import socket
import sys
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
