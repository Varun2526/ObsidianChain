"""Validate and normalise the entity-label dataset for entity resolution.

Source: ``data/raw/address_labels/addresses.csv`` - an aggregated set of
Bitcoin addresses tagged with an owning entity, a category, and a provenance
source. Only the ``entity`` column is ground truth for entity resolution:
two addresses sharing an entity are controlled by the same actor, which is a
must-link; two addresses with different entities are a cannot-link.

Why this needs a derived file
-----------------------------
The raw file cannot be used as-is. It carries a pandas index column, 12
malformed addresses, 2,425 duplicate address rows, and a handful of rows that
disagree with each other about which entity owns an address. Re-deriving those
decisions on every evaluation run would put the dedupe and conflict handling -
the two places a silent error would corrupt every downstream metric - inside
the measurement loop. They belong in one auditable pass with a printed report.

What is deliberately NOT done here
----------------------------------
No metrics, no pair generation, no clustering. This module only produces a
clean table. Bitcoin base58 addresses are case-sensitive and are never
casefolded; only *entity names* are normalised for grouping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from obsidianchain.io import elliptic

LABELS_SUBDIR = "address_labels"
LABELS_FILE = "addresses.csv"

#: Base58 P2PKH/P2SH, or lowercase bech32. Uppercase bech32 is legal in the
#: spec but does not occur here, and accepting it would create two spellings
#: of one address.
ADDRESS_PATTERN = re.compile(
    r"^(?:[13][A-HJ-NP-Za-km-z1-9]{25,34}|bc1[023456789ac-hj-np-z]{11,71})$"
)

OUTPUT_COLUMNS = [
    "address",
    "entity_norm",
    "entity_display",
    "category",
    "source",
    "in_elliptic",
]


@dataclass
class NormalisationReport:
    """Every row that entered, and where each one went."""

    rows_in: int = 0
    dropped_no_entity: int = 0
    dropped_malformed_address: int = 0
    dropped_conflicting_entity: int = 0
    duplicate_rows_collapsed: int = 0
    category_conflicts_resolved: int = 0
    casefold_collisions: int = 0

    rows_out: int = 0
    addresses_out: int = 0
    entities_out: int = 0
    entities_multi_address: int = 0

    elliptic_checked: bool = False
    in_elliptic: int = 0
    entities_in_elliptic: int = 0
    entities_in_elliptic_multi: int = 0
    same_entity_pairs_in_elliptic: int = 0

    def accounted(self) -> bool:
        """Every input row is either dropped, collapsed, or emitted."""
        return self.rows_in == (
            self.rows_out
            + self.dropped_no_entity
            + self.dropped_malformed_address
            + self.dropped_conflicting_entity
            + self.duplicate_rows_collapsed
        )


def find_labels_file(data_root: Path | None = None) -> Path:
    """Locate addresses.csv, preferring the organised subdirectory."""
    root = Path(data_root) if data_root is not None else elliptic.DEFAULT_DATA_ROOT
    direct = root / "raw" / LABELS_SUBDIR / LABELS_FILE
    if direct.is_file():
        return direct
    return elliptic.find_dataset_file(LABELS_FILE, data_root)


def load_raw(data_root: Path | None = None) -> pd.DataFrame:
    """Read the raw label file, discarding the unnamed pandas index column."""
    path = find_labels_file(data_root)
    frame = pd.read_csv(path)
    frame = frame.loc[:, [c for c in frame.columns if not c.startswith("Unnamed")]]
    missing = {"address", "entity", "category", "source"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing expected columns: {sorted(missing)}; "
            f"found {list(frame.columns)}"
        )
    return frame


def valid_address(series: pd.Series) -> pd.Series:
    """Boolean mask of well-formed Bitcoin addresses."""
    return series.astype("string").fillna("").map(
        lambda a: bool(ADDRESS_PATTERN.match(a))
    )


def normalise_entity(series: pd.Series) -> pd.Series:
    """Fold entity names for grouping: strip surrounding space, casefold.

    ``Kucoin`` and ``kucoin`` are one entity. The original spelling is kept
    separately for display; only this folded form is ever grouped on.
    """
    return series.astype("string").str.strip().str.casefold()


def load_elliptic_addresses(data_root: Path | None = None) -> set[str]:
    """The Elliptic++ address universe, for the ``in_elliptic`` flag."""
    path = elliptic.find_dataset_file(elliptic.WALLETS_CLASSES, data_root)
    return set(pd.read_csv(path, usecols=["address"])["address"])


def build(
    data_root: Path | None = None, check_elliptic: bool = True
) -> tuple[pd.DataFrame, NormalisationReport]:
    """Produce the clean entity-label table and a report of every decision."""
    frame = load_raw(data_root)
    report = NormalisationReport(rows_in=int(len(frame)))

    # 1. Only entity-labelled rows are ground truth for entity resolution.
    #    Category-only rows are dropped; they cannot form must-link pairs.
    labelled = frame[frame["entity"].notna()].copy()
    report.dropped_no_entity = report.rows_in - int(len(labelled))

    # 2. Malformed addresses are dropped, not repaired. The bad rows here are
    #    truncated (e.g. "2cX4MWcT..."), and guessing the missing prefix would
    #    invent ground truth.
    ok = valid_address(labelled["address"])
    report.dropped_malformed_address = int((~ok).sum())
    labelled = labelled[ok].copy()

    # 3. Fold entity names.
    labelled["entity_norm"] = normalise_entity(labelled["entity"])
    report.casefold_collisions = int(
        labelled["entity"].nunique() - labelled["entity_norm"].nunique()
    )

    # 4. An address claimed by two different entities is unusable: we cannot
    #    tell which is right, and either choice fabricates a must-link. Drop
    #    the address outright rather than pick.
    per_address = labelled.groupby("address")["entity_norm"].nunique()
    conflicted = set(per_address[per_address > 1].index)
    report.dropped_conflicting_entity = int(
        labelled["address"].isin(conflicted).sum()
    )
    labelled = labelled[~labelled["address"].isin(conflicted)].copy()

    # 5. Category disagreements are survivable - category is not the ground
    #    truth - so resolve deterministically and count how often it happened.
    cat_per_address = labelled.groupby("address")["category"].nunique(dropna=True)
    report.category_conflicts_resolved = int((cat_per_address > 1).sum())

    # 6. Collapse duplicate rows. Sort first so the surviving row is
    #    deterministic regardless of input order.
    before = int(len(labelled))
    labelled = labelled.sort_values(
        ["address", "entity_norm", "category", "source"],
        kind="stable",
        na_position="last",
    )
    labelled = labelled.drop_duplicates(subset="address", keep="first")
    report.duplicate_rows_collapsed = before - int(len(labelled))

    # 7. Display name: the most common original spelling of each folded name.
    display = (
        labelled.groupby("entity_norm")["entity"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("entity_display")
    )
    labelled = labelled.merge(display, on="entity_norm", how="left")

    # 8. Flag membership of the Elliptic++ universe. This is what makes the
    #    file usable against our clusters; addresses outside it can never be
    #    evaluated no matter how well labelled they are.
    if check_elliptic:
        universe = load_elliptic_addresses(data_root)
        labelled["in_elliptic"] = labelled["address"].isin(universe)
        report.elliptic_checked = True
    else:
        labelled["in_elliptic"] = pd.NA

    out = labelled.loc[:, OUTPUT_COLUMNS].sort_values(
        ["entity_norm", "address"], kind="stable"
    ).reset_index(drop=True)

    report.rows_out = int(len(out))
    report.addresses_out = int(out["address"].nunique())
    sizes = out.groupby("entity_norm")["address"].nunique()
    report.entities_out = int(len(sizes))
    report.entities_multi_address = int((sizes >= 2).sum())

    if check_elliptic:
        inside = out[out["in_elliptic"] == True]  # noqa: E712
        report.in_elliptic = int(len(inside))
        esizes = inside.groupby("entity_norm")["address"].nunique()
        report.entities_in_elliptic = int(len(esizes))
        report.entities_in_elliptic_multi = int((esizes >= 2).sum())
        report.same_entity_pairs_in_elliptic = int((esizes * (esizes - 1) // 2).sum())

    return out, report


def write(frame: pd.DataFrame, path: Path, provenance=None) -> int:
    """Write the processed table. Returns the row count.

    ``provenance`` attaches the record two ways - a marker column in every
    row and a sibling ``.meta.json``. Optional so the function keeps working
    for callers that only want the frame on disk; every CLI path passes it,
    and ``tests/test_provenance.py`` reads the artifacts back to check.
    """
    if provenance is not None:
        from obsidianchain import provenance as prov

        prov.write_frame(frame, path, provenance)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return int(len(frame))


def format_report(report: NormalisationReport, out_path: Path | None = None) -> str:
    """Render the normalisation report."""
    width = 74
    lines: list[str] = []
    add = lines.append
    add("=" * width)
    add("obsidianchain :: entity-label normalisation")
    add("=" * width)
    add(f"source   raw/{LABELS_SUBDIR}/{LABELS_FILE}  (never modified)")
    if out_path:
        add(f"output   {out_path}")
    add("")
    add("-- rows in " + "-" * (width - 12))
    add(f"  raw rows                             {report.rows_in:>10,}")
    add("")
    add("-- dropped " + "-" * (width - 12))
    add(f"  no entity label                      {report.dropped_no_entity:>10,}   not ground truth")
    add(f"  malformed address                    {report.dropped_malformed_address:>10,}   truncated, not repaired")
    add(f"  address claimed by 2+ entities       {report.dropped_conflicting_entity:>10,}   unusable, dropped whole")
    add(f"  duplicate rows collapsed             {report.duplicate_rows_collapsed:>10,}")
    add("")
    add("-- resolved " + "-" * (width - 13))
    add(f"  entity names folded together         {report.casefold_collisions:>10,}   e.g. Kucoin / kucoin")
    add(f"  category disagreements               {report.category_conflicts_resolved:>10,}   category is not ground truth")
    add("")
    add("-- rows out " + "-" * (width - 13))
    add(f"  rows                                 {report.rows_out:>10,}")
    add(f"  distinct addresses                   {report.addresses_out:>10,}")
    add(f"  distinct entities                    {report.entities_out:>10,}")
    add(f"  entities with >=2 addresses          {report.entities_multi_address:>10,}   can form must-links")
    add("")
    if report.elliptic_checked:
        add("-- usable against Elliptic++ clusters " + "-" * (width - 39))
        add(f"  addresses inside Elliptic++          {report.in_elliptic:>10,}")
        add(f"  entities represented                 {report.entities_in_elliptic:>10,}")
        add(f"  ...with >=2 addresses inside         {report.entities_in_elliptic_multi:>10,}")
        add(f"  same-entity pairs available          {report.same_entity_pairs_in_elliptic:>10,}")
        add("")
        add("  Only these rows can be scored against our clustering. The rest")
        add("  are retained for later work on other chain data.")
        add("")
    status = "OK" if report.accounted() else "MISMATCH - rows unaccounted for"
    add(f"row accounting: {status}")
    add("=" * width)
    return "\n".join(lines)
