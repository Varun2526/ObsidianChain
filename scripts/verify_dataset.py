#!/usr/bin/env python3
"""Verify that the Elliptic++ dataset is present in data/raw.

Read-only and offline: this script downloads nothing and opens no sockets.
It only inspects whatever is already on disk. Standard library only, so it
runs on the host as well as inside the air-gapped container.

Dataset: https://github.com/git-disl/EllipticPlusPlus  (fetch it yourself,
on a networked machine, then copy it into data/raw/).

Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Basename -> (required?, why it matters)
EXPECTED: dict[str, tuple[bool, str]] = {
    "AddrAddr_edgelist.csv": (
        True,
        "address-to-address edges; co-spend clustering is impossible without it",
    ),
    "AddrTx_edgelist.csv": (
        True,
        "address -> transaction edges; needed to attribute inputs to addresses",
    ),
    "TxAddr_edgelist.csv": (
        True,
        "transaction -> address edges; the output side of the same mapping",
    ),
    "wallets_features_classes_combined.csv": (
        True,
        "address-level features and licit/illicit labels",
    ),
    "txs_features.csv": (False, "transaction-level features (tx graph baseline)"),
    "txs_classes.csv": (False, "transaction-level labels"),
    "txs_edgelist.csv": (False, "transaction-to-transaction edges"),
}

CRITICAL = "AddrAddr_edgelist.csv"

_READ_CHUNK = 8 * 1024 * 1024
_SNIFF_ROWS = 2000


@dataclass
class CsvReport:
    path: Path
    size_bytes: int
    rows: int | None = None          # data rows, header excluded
    columns: int | None = None
    header: list[str] | None = None
    ragged_at: int | None = None     # 1-based data row with a bad field count
    error: str | None = None


def human_bytes(n: int) -> str:
    step = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if step < 1024 or unit == "TB":
            return f"{step:,.1f} {unit}" if unit != "B" else f"{int(step):,} B"
        step /= 1024
    return f"{step:,.1f} TB"


def count_data_rows(path: Path) -> int:
    """Count newline-delimited rows, excluding the header.

    Byte-level newline counting: these files are flat numeric/ID CSVs with
    no quoted embedded newlines. inspect_csv separately checks that the
    field count is consistent, which would catch a file where that
    assumption does not hold.
    """
    newlines = 0
    trailing_newline = True
    with path.open("rb") as fh:
        while chunk := fh.read(_READ_CHUNK):
            newlines += chunk.count(b"\n")
            trailing_newline = chunk.endswith(b"\n")
    total_lines = newlines if trailing_newline else newlines + 1
    return max(total_lines - 1, 0)  # drop the header


def inspect_csv(path: Path) -> CsvReport:
    report = CsvReport(path=path, size_bytes=path.stat().st_size)
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                report.error = "file is empty"
                return report
            report.header = header
            report.columns = len(header)
            for i, row in enumerate(reader, start=1):
                if i > _SNIFF_ROWS:
                    break
                if row and len(row) != report.columns:
                    report.ragged_at = i
                    break
        report.rows = count_data_rows(path)
    except OSError as exc:
        report.error = f"unreadable: {exc}"
    return report


def find_csvs(raw_dir: Path) -> dict[str, list[Path]]:
    """Map basename -> paths, searching data/raw recursively."""
    found: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".csv":
            found.setdefault(path.name, []).append(path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data")),
        help="directory containing raw/ (default: $OBSIDIANCHAIN_DATA or /data)",
    )
    args = parser.parse_args()

    raw_dir: Path = args.data_root / "raw"
    print("=" * 72)
    print("obsidianchain :: Elliptic++ dataset verification")
    print("=" * 72)
    print(f"data root : {args.data_root}")
    print(f"raw dir   : {raw_dir}")
    print("mode      : read-only, offline (nothing is downloaded)")
    print()

    if not raw_dir.is_dir():
        print(f"[!] {raw_dir} does not exist.")
        print()
        print_hint()
        print("SUMMARY: FAIL - no raw data directory")
        return 1

    found = find_csvs(raw_dir)
    all_entries = [p for p in sorted(raw_dir.rglob("*")) if p.is_file()]
    visible = [p for p in all_entries if p.name != ".gitkeep"]

    print(f"-- files present under raw/ ({len(visible)}) " + "-" * 30)
    if not visible:
        print("  (none)")
    for path in visible:
        rel = path.relative_to(raw_dir)
        print(f"  {human_bytes(path.stat().st_size):>12}  {rel}")
    print()

    reports: dict[str, CsvReport] = {}
    unexpected = sorted(set(found) - set(EXPECTED))

    print("-- CSV contents " + "-" * 55)
    for name in list(EXPECTED) + unexpected:
        paths = found.get(name)
        if not paths:
            continue
        if len(paths) > 1:
            locations = ", ".join(str(p.relative_to(raw_dir)) for p in paths)
            print(f"  [!] {name}: found in several places ({locations}); using the first")
        report = inspect_csv(paths[0])
        reports[name] = report

        if report.error:
            print(f"  [!] {name}: {report.error}")
            continue

        rows = f"{report.rows:,}" if report.rows is not None else "?"
        print(f"  [ok] {name}")
        print(f"       rows {rows:>14}   columns {report.columns:>4}   {human_bytes(report.size_bytes)}")
        if report.header:
            preview = ", ".join(report.header[:6])
            if len(report.header) > 6:
                preview += f", ... (+{len(report.header) - 6} more)"
            print(f"       header: {preview}")
        if report.ragged_at is not None:
            print(f"       [!] inconsistent field count at data row {report.ragged_at}")
    if not reports:
        print("  (no CSV files found)")
    print()

    # ---- checks -------------------------------------------------------
    failures: list[str] = []
    warnings: list[str] = []

    print("-- required files " + "-" * 53)
    for name, (required, why) in EXPECTED.items():
        report = reports.get(name)
        ok = report is not None and report.error is None
        mark = "PASS" if ok else ("FAIL" if required else "WARN")
        print(f"  [{mark}] {name}")
        print(f"         {why}")
        if ok and report.rows == 0:
            msg = f"{name} has a header but no data rows"
            print(f"         [!] {msg}")
            (failures if required else warnings).append(msg)
        elif ok and report.ragged_at is not None:
            warnings.append(f"{name} has an inconsistent field count")
        elif not ok:
            (failures if required else warnings).append(f"{name} is missing")
    print()

    print("-- critical check " + "-" * 53)
    critical_report = reports.get(CRITICAL)
    if critical_report is not None and critical_report.error is None and critical_report.rows:
        print(f"  [PASS] {CRITICAL} is present with {critical_report.rows:,} edges.")
        print("         Address-level co-spend clustering is possible.")
    else:
        print(f"  [FAIL] {CRITICAL} is absent or empty.")
        print("         Co-spend clustering is IMPOSSIBLE without address-level")
        print("         data - the transaction-only Elliptic dataset is not enough.")
    print()

    if unexpected:
        print(f"-- extra CSVs not part of the expected set ({len(unexpected)}) " + "-" * 16)
        for name in unexpected:
            print(f"  {name}")
        print()

    print("=" * 72)
    if failures:
        print(f"SUMMARY: FAIL  ({len(failures)} blocking, {len(warnings)} warning)")
        for msg in failures:
            print(f"  FAIL  {msg}")
        for msg in warnings:
            print(f"  warn  {msg}")
        print()
        print_hint()
        print("=" * 72)
        return 1

    if warnings:
        print(f"SUMMARY: PASS with {len(warnings)} warning(s)")
        for msg in warnings:
            print(f"  warn  {msg}")
    else:
        print("SUMMARY: PASS - all required Elliptic++ files present and readable")
    print("=" * 72)
    return 0


def print_hint() -> None:
    print("How to fix (do this on a networked machine, not in the container):")
    print("  1. git clone https://github.com/git-disl/EllipticPlusPlus")
    print("  2. Follow that repo's instructions to obtain the dataset CSVs.")
    print("  3. Copy them into ./data/raw/ on the host.")
    print("  4. Re-run: make verify")
    print()


if __name__ == "__main__":
    sys.exit(main())
