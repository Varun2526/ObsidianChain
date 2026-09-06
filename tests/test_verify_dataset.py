"""Tests for scripts/verify_dataset.py.

The script must fail loudly on a missing dataset and pass on a complete
one, and must never touch the network.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_dataset.py"

REQUIRED = [
    "AddrAddr_edgelist.csv",
    "AddrTx_edgelist.csv",
    "TxAddr_edgelist.csv",
    "wallets_features_classes_combined.csv",
]
OPTIONAL = ["txs_features.csv", "txs_classes.csv", "txs_edgelist.csv"]


def run_script(data_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--data-root", str(data_root)],
        capture_output=True,
        text=True,
        check=False,
    )


def write_csv(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["src,dst"] + [f"addr{i},addr{i + 1}" for i in range(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    return tmp_path


def test_fails_on_empty_raw_dir(data_root: Path) -> None:
    result = run_script(data_root)
    assert result.returncode == 1
    assert "SUMMARY: FAIL" in result.stdout


def test_fails_when_addraddr_missing(data_root: Path) -> None:
    for name in REQUIRED[1:] + OPTIONAL:
        write_csv(data_root / "raw" / name, 5)
    result = run_script(data_root)
    assert result.returncode == 1
    assert "AddrAddr_edgelist.csv is missing" in result.stdout
    assert "Co-spend clustering is IMPOSSIBLE" in result.stdout


def test_passes_on_complete_dataset(data_root: Path) -> None:
    for name in REQUIRED + OPTIONAL:
        write_csv(data_root / "raw" / name, 7)
    result = run_script(data_root)
    assert result.returncode == 0, result.stdout
    assert "SUMMARY: PASS" in result.stdout
    # row count excludes the header
    assert "7" in result.stdout


def test_reports_row_and_column_counts(data_root: Path) -> None:
    for name in REQUIRED + OPTIONAL:
        write_csv(data_root / "raw" / name, 12)
    result = run_script(data_root)
    assert "rows" in result.stdout and "columns" in result.stdout
    assert "12" in result.stdout


def test_fails_on_header_only_file(data_root: Path) -> None:
    for name in REQUIRED + OPTIONAL:
        write_csv(data_root / "raw" / name, 3)
    write_csv(data_root / "raw" / "AddrAddr_edgelist.csv", 0)
    result = run_script(data_root)
    assert result.returncode == 1
    assert "no data rows" in result.stdout


def test_finds_files_in_a_subdirectory(data_root: Path) -> None:
    for name in REQUIRED + OPTIONAL:
        write_csv(data_root / "raw" / "EllipticPlusPlus" / name, 4)
    result = run_script(data_root)
    assert result.returncode == 0, result.stdout


def test_script_makes_no_network_calls() -> None:
    """Guard the 'downloads nothing' promise at the source level."""
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("urllib.request", "requests", "httpx", "socket", "urlopen"):
        assert forbidden not in source.replace("no sockets", ""), forbidden
