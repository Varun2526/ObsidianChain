"""Tests for entity-label validation and normalisation.

Synthetic fixture exercising every documented decision:

    A1, A2, A3   -> entity "Kraken"      three addresses, one entity
    A4           -> entity "kraken"      case variant, must fold into Kraken
    A5           -> entity "Bitfinex"
    A6           -> no entity            dropped, not ground truth
    A7 (bad)     -> malformed address    dropped, not repaired
    A8           -> two entities         conflicting, address dropped whole
    A9           -> duplicated row       collapsed
    A10          -> two categories       resolved, kept
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from obsidianchain.io import elliptic
from obsidianchain.io import entity_labels as el

GOOD = "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
ROWS = [
    (GOOD[:-1] + "a", "Kraken", "EXCHANGE", "Src"),
    (GOOD[:-1] + "b", "Kraken", "EXCHANGE", "Src"),
    (GOOD[:-1] + "c", "Kraken", "EXCHANGE", "Src"),
    (GOOD[:-1] + "d", "kraken", "EXCHANGE", "Src"),      # case variant
    (GOOD[:-1] + "e", "Bitfinex", "EXCHANGE", "Src"),
    (GOOD[:-1] + "f", None, "GAMBLING", "Src"),          # no entity
    ("2cX4MWcTFbmKgPQX1irMiDsU84dXB6LFBv", "APT", "RANSOMWARE", "Src"),  # malformed
    (GOOD[:-1] + "h", "Alpha", "MIXER", "Src"),          # conflicting pair 1/2
    (GOOD[:-1] + "h", "Beta", "MIXER", "Src"),           # conflicting pair 2/2
    (GOOD[:-1] + "i", "Gamma", "PONZI", "Src"),          # duplicate pair 1/2
    (GOOD[:-1] + "i", "Gamma", "PONZI", "Src"),          # duplicate pair 2/2
    (GOOD[:-1] + "j", "Delta", "MINING", "Src"),         # category conflict 1/2
    (GOOD[:-1] + "j", "Delta", "EXCHANGE", "Src"),       # category conflict 2/2
]


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    raw = tmp_path / "raw" / el.LABELS_SUBDIR
    raw.mkdir(parents=True)
    frame = pd.DataFrame(ROWS, columns=["address", "entity", "category", "source"])
    frame.insert(0, "Unnamed: 0", range(len(frame)))
    frame.to_csv(raw / el.LABELS_FILE, index=False)

    # Elliptic++ universe: only the three Kraken addresses are inside.
    (tmp_path / "raw").mkdir(exist_ok=True)
    universe = [GOOD[:-1] + c for c in "abc"]
    (tmp_path / "raw" / elliptic.WALLETS_CLASSES).write_text(
        "address,class\n" + "".join(f"{a},2\n" for a in universe), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture()
def built(dataset: Path):
    return el.build(dataset)


# ---- address validation ------------------------------------------------


def test_valid_address_accepts_real_formats() -> None:
    good = pd.Series([
        "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
        "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",
        "bc1qygg2x02cfy0e6r7798v4qrcjjkzm8tl5t0xkwf",
    ])
    assert el.valid_address(good).all()


def test_valid_address_rejects_malformed() -> None:
    bad = pd.Series([
        "2cX4MWcTFbmKgPQX1irMiDsU84dXB6LFBv",  # truncated
        "NBazWh9xNVf2SgmvLv8pc3Uc9CCXtXMu",     # truncated
        "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN0",   # '0' not in base58
        "", "not-an-address",
    ])
    assert not el.valid_address(bad).any()


def test_base58_addresses_are_never_casefolded(built) -> None:
    """Base58 is case-sensitive; folding it would merge distinct addresses."""
    frame, _ = built
    assert any(a != a.lower() for a in frame["address"]), "case must be preserved"


# ---- entity normalisation ----------------------------------------------


def test_entity_case_is_folded(built) -> None:
    frame, report = built
    kraken = frame[frame["entity_norm"] == "kraken"]
    assert len(kraken) == 4, "Kraken and kraken are one entity"
    assert report.casefold_collisions == 1


def test_original_spelling_is_preserved_for_display(built) -> None:
    frame, _ = built
    kraken = frame[frame["entity_norm"] == "kraken"]
    assert set(kraken["entity_display"]) == {"Kraken"}, "most common spelling wins"


def test_normalise_entity_strips_whitespace() -> None:
    got = el.normalise_entity(pd.Series(["  Kraken ", "KRAKEN"]))
    assert list(got) == ["kraken", "kraken"]


# ---- drops and conflicts -----------------------------------------------


def test_rows_without_an_entity_are_dropped(built) -> None:
    _, report = built
    assert report.dropped_no_entity == 1


def test_malformed_addresses_are_dropped_not_repaired(built) -> None:
    frame, report = built
    assert report.dropped_malformed_address == 1
    assert not frame["address"].str.startswith("2c").any()


def test_address_claimed_by_two_entities_is_dropped_whole(built) -> None:
    """Picking either side would fabricate a must-link."""
    frame, report = built
    assert report.dropped_conflicting_entity == 2, "both rows go"
    assert "alpha" not in set(frame["entity_norm"])
    assert "beta" not in set(frame["entity_norm"])


def test_duplicate_rows_are_collapsed(built) -> None:
    frame, report = built
    assert report.duplicate_rows_collapsed == 2  # Gamma dup + Delta cat conflict
    assert not frame["address"].duplicated().any()


def test_category_conflict_is_resolved_and_counted(built) -> None:
    """Category is not ground truth, so the address survives."""
    frame, report = built
    assert report.category_conflicts_resolved == 1
    delta = frame[frame["entity_norm"] == "delta"]
    assert len(delta) == 1
    assert delta.iloc[0]["category"] in {"MINING", "EXCHANGE"}


def test_resolution_is_deterministic(dataset: Path) -> None:
    first, _ = el.build(dataset)
    second, _ = el.build(dataset)
    pd.testing.assert_frame_equal(first, second)


# ---- output shape ------------------------------------------------------


def test_output_columns(built) -> None:
    frame, _ = built
    assert list(frame.columns) == el.OUTPUT_COLUMNS


def test_one_row_per_address(built) -> None:
    frame, _ = built
    assert frame["address"].is_unique


def test_row_accounting_balances(built) -> None:
    """Every input row is dropped, collapsed, or emitted - nothing vanishes."""
    _, report = built
    assert report.accounted(), (
        f"{report.rows_in} in vs {report.rows_out} out plus drops"
    )


def test_elliptic_flag(built) -> None:
    frame, report = built
    assert report.elliptic_checked is True
    assert report.in_elliptic == 3, "only the three Kraken addresses"
    assert report.entities_in_elliptic == 1
    assert report.entities_in_elliptic_multi == 1
    assert report.same_entity_pairs_in_elliptic == 3  # C(3,2)


def test_elliptic_check_can_be_skipped(dataset: Path) -> None:
    frame, report = el.build(dataset, check_elliptic=False)
    assert report.elliptic_checked is False
    assert frame["in_elliptic"].isna().all()


def test_entity_counts(built) -> None:
    _, report = built
    assert report.entities_out == 4      # kraken, bitfinex, gamma, delta
    assert report.entities_multi_address == 1  # only kraken


# ---- io ----------------------------------------------------------------


def test_write_creates_parent_directory(built, tmp_path: Path) -> None:
    frame, _ = built
    out = tmp_path / "processed" / "entity_labels.csv"
    assert el.write(frame, out) == len(frame)
    assert out.is_file()
    assert list(pd.read_csv(out).columns) == el.OUTPUT_COLUMNS


def test_missing_columns_are_reported(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / el.LABELS_SUBDIR
    raw.mkdir(parents=True)
    (raw / el.LABELS_FILE).write_text("address,entity\nabc,X\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        el.load_raw(tmp_path)


def test_report_renders(built) -> None:
    _, report = built
    text = el.format_report(report, Path("/data/processed/entity_labels.csv"))
    for expected in ("malformed", "2+ entities", "row accounting", "Elliptic++"):
        assert expected in text
