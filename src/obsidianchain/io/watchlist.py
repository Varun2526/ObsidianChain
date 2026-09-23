"""Seed illicit wallets for risk propagation, from offline sources only.

Two sources, both files on disk - nothing is fetched:

* The OFAC SDN list (``data/raw/ofac_sdn/sdn_xml.zip``), whose entries carry
  ``Digital Currency Address - XBT`` identifiers. A sanctioned address is a
  published, attributable seed.
* Analyst watchlists: any ``*.csv`` under ``data/watchlists/`` with an
  ``address`` column (optional ``label`` column).

A seed is a claim about an address made by someone outside this system. The
source travels with every seed so an alert can say who made the claim.
"""

from __future__ import annotations

import csv
import functools
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

DEFAULT_OFAC_ZIP = Path("data") / "raw" / "ofac_sdn" / "sdn_xml.zip"
DEFAULT_WATCHLIST_DIR = Path("data") / "watchlists"

OFAC_BTC_ID_TYPE = "Digital Currency Address - XBT"


@dataclass(frozen=True)
class Seed:
    address: str
    source: str
    label: str = ""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


@functools.lru_cache(maxsize=4)
def _ofac_cached(path: str, mtime: float) -> tuple[Seed, ...]:
    seeds: list[Seed] = []
    with zipfile.ZipFile(path) as zf:
        name = next((n for n in zf.namelist() if n.lower().endswith(".xml")), None)
        if name is None:
            return ()
        with zf.open(name) as fh:
            entry_name = ""
            for _, elem in ElementTree.iterparse(fh, events=("end",)):
                tag = _local(elem.tag)
                if tag == "lastName":
                    entry_name = (elem.text or "").strip()
                elif tag == "id":
                    fields = {_local(c.tag): (c.text or "").strip() for c in elem}
                    if fields.get("idType") == OFAC_BTC_ID_TYPE and fields.get("idNumber"):
                        seeds.append(Seed(fields["idNumber"], "OFAC_SDN", entry_name))
                    elem.clear()
                elif tag == "sdnEntry":
                    entry_name = ""
                    elem.clear()
    return tuple(seeds)


def _data_root() -> Path:
    import os
    return Path(os.environ.get("OBSIDIANCHAIN_DATA") or "data")


def load_ofac(path: str | Path | None = None) -> list[Seed]:
    """Bitcoin addresses on the OFAC SDN list. Empty if the file is absent."""
    path = Path(path) if path is not None else _data_root() / "raw" / "ofac_sdn" / "sdn_xml.zip"
    if not path.is_file():
        return []
    return list(_ofac_cached(str(path.resolve()), path.stat().st_mtime))


def load_csv_watchlists(directory: str | Path | None = None) -> list[Seed]:
    directory = Path(directory) if directory is not None else _data_root() / "watchlists"
    if not directory.is_dir():
        return []
    seeds: list[Seed] = []
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                address = (row.get("address") or "").strip()
                if address:
                    seeds.append(Seed(address, f"WATCHLIST:{path.name}", (row.get("label") or "").strip()))
    return seeds


def load_default_seeds(ofac_zip: str | Path | None = None,
                       watchlist_dir: str | Path | None = None) -> list[Seed]:
    """Every seed from every offline source, first source wins on duplicates."""
    out: dict[str, Seed] = {}
    for seed in load_ofac(ofac_zip) + load_csv_watchlists(watchlist_dir):
        out.setdefault(seed.address, seed)
    return list(out.values())
