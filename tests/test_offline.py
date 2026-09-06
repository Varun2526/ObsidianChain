"""Air-gap invariants.

These assert the properties the demo depends on: deterministic hashing and
no reachable network. The network assertions only run inside the container,
where `--network none` is in force.
"""

from __future__ import annotations

import os
import socket

import pytest

from obsidianchain.cli import _network_state

IN_CONTAINER = os.path.exists("/.dockerenv") or os.environ.get("OBSIDIANCHAIN_DATA") == "/data"
container_only = pytest.mark.skipif(
    not IN_CONTAINER, reason="air-gap assertions only hold inside the container"
)


@container_only
def test_hash_seed_is_pinned() -> None:
    """Deterministic output requires a fixed hash seed."""
    assert os.environ.get("PYTHONHASHSEED") == "0"


@container_only
def test_thread_counts_pinned() -> None:
    """BLAS thread counts are pinned so numeric results are reproducible."""
    assert os.environ.get("OMP_NUM_THREADS") == "1"


@container_only
def test_no_network_route() -> None:
    assert _network_state().startswith("isolated")


@container_only
def test_outbound_tcp_fails() -> None:
    """A real connect attempt must fail rather than hang or succeed."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        with pytest.raises(OSError):
            sock.connect(("1.1.1.1", 53))
    finally:
        sock.close()


def test_network_state_is_a_string() -> None:
    """Runs everywhere; just checks the probe never raises."""
    assert isinstance(_network_state(), str)
