"""The Phase 6 leakage boundary, enforced rather than documented.

Three things are asserted here, and each one exists because getting it wrong
produces a number that looks like success:

1. **No feature builder reads a label.** Labels are joined once, in
   ``features/dataset.py``, after every feature is built. A builder that
   opened ``wallets_classes.csv`` could encode the answer into a feature and
   nothing downstream would be able to tell.

2. **No feature builder reads ``wallets_features.csv``.** SPEC 0.3 measured
   that 52 of its 55 columns are whole-life aggregates replicated across
   timesteps - at t=25 a row already states how many transactions the address
   will ever make. Reading it under a chronological split leaks the future.

3. **The splits are address-disjoint and temporally ordered.** Labels are
   static, so an address appearing in both train and test would be memorised
   rather than predicted.

The source-level checks are deliberately structural rather than behavioural:
a test that merely exercised one code path would pass while an unexercised
branch read a label.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from obsidianchain.features import dataset

# Located through the IMPORTED package, not through this file's parents: the
# container installs obsidianchain into site-packages while the tests live in
# /app, so a path relative to the test file finds nothing and every
# parametrised check below would silently collapse to zero cases.
import obsidianchain.features as _features
import obsidianchain.ml as _ml

FEATURES_DIR = Path(_features.__file__).resolve().parent
ML_DIR = Path(_ml.__file__).resolve().parent

#: Files that carry an outcome. A feature builder may not name one.
LABEL_FILES = ("wallets_classes", "txs_classes")

#: The file SPEC 0.3 found leaks the future. Never read, by anything.
LEAKING_FILE = "wallets_features"

#: Truth accessors and quarantined directories, as in test_truth_isolation.
TRUTH_PATTERNS = (
    "FOR_EVALUATION_ONLY", "network_truth", "worlds_truth",
    "true_entity_id", "true_origin_id",
)


def builder_modules() -> list[Path]:
    """Every feature module except the one that is allowed to join labels."""
    return sorted(
        p for p in FEATURES_DIR.rglob("*.py")
        if p.name not in {"dataset.py", "__init__.py"}
    )


def code_only(source: str) -> str:
    """Source with docstrings stripped.

    A module may legitimately DISCUSS a label file in prose - peel.py explains
    at length that it must not read one - so only executable references count.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant
            ) and isinstance(body[0].value.value, str):
                body[0].value.value = ""
    return ast.unparse(tree)


def test_the_builder_set_is_not_empty() -> None:
    """Guards against a vacuous suite if the package is moved."""
    assert builder_modules(), f"no feature modules found under {FEATURES_DIR}"


@pytest.mark.parametrize("path", builder_modules(), ids=lambda p: p.name)
def test_no_feature_builder_reads_a_label_file(path: Path) -> None:
    executable = code_only(path.read_text(encoding="utf-8"))
    offenders = [name for name in LABEL_FILES if name in executable]
    assert not offenders, (
        f"{path.name} references {offenders}. Features are built before "
        f"labels are joined precisely so a builder cannot encode the answer."
    )


@pytest.mark.parametrize(
    "path",
    sorted(FEATURES_DIR.rglob("*.py")) + sorted(ML_DIR.rglob("*.py")),
    ids=lambda p: p.name,
)
def test_nothing_reads_the_whole_life_wallet_features(path: Path) -> None:
    """SPEC 0.3. Not even the three columns that survive.

    Deriving first-appearance from ``min(Time step)`` costs nothing and means
    the file with 52 leaking columns is never opened at all, which is a
    stronger guarantee than opening it and taking three columns carefully.
    """
    executable = code_only(path.read_text(encoding="utf-8"))
    assert LEAKING_FILE not in executable, (
        f"{path.name} reads {LEAKING_FILE}.csv, whose columns are whole-life "
        f"aggregates replicated across timesteps (SPEC 0.3)."
    )


@pytest.mark.parametrize(
    "path",
    sorted(FEATURES_DIR.rglob("*.py")) + sorted(ML_DIR.rglob("*.py")),
    ids=lambda p: p.name,
)
def test_no_phase6_module_touches_ground_truth(path: Path) -> None:
    executable = code_only(path.read_text(encoding="utf-8"))
    offenders = [m for m in TRUTH_PATTERNS if m in executable]
    assert not offenders, f"{path.name} references truth access: {offenders}"


def test_the_excluded_feature_blocks_are_absent_from_the_contract() -> None:
    """Aggregate_feature_*, Local_feature_* and final cluster size.

    All three are excluded by the spec for different reasons: unresolved
    provenance, anonymity, and "final membership is not a predictive input".
    A name appearing in the contract would put it in the matrix.
    """
    columns = [c for group in dataset.FEATURE_GROUPS.values() for c in group]
    joined = " ".join(columns)
    assert "Aggregate_feature" not in joined
    assert "Local_feature" not in joined
    assert "cluster_size_final" not in joined
    assert "cluster_size_asof_t" in joined, (
        "the as-of-t cluster size is the permitted form and should be present"
    )


def test_the_transaction_column_allowlist_excludes_the_anonymised_blocks() -> None:
    from obsidianchain.features import incidence

    joined = " ".join(incidence.TX_COLUMNS)
    assert "Aggregate_feature" not in joined
    assert "Local_feature" not in joined


# ---- split integrity, on the real dataset ------------------------------


import os
DATA_ROOT = Path(os.environ.get("OBSIDIANCHAIN_DATA", "/data"))
DATASET = DATA_ROOT / "processed" / "phase6_dataset_d5.parquet"


@pytest.fixture(scope="module")
def built() -> pd.DataFrame:
    """The generated dataset. A FAILURE, never a skip, when absent.

    This is the only place the leakage controls are checked against real
    rows rather than against source text, so a suite that skipped it would
    be reporting green on the checks that matter most.
    """
    assert DATASET.is_file(), (
        f"{DATASET} is absent, so the split's address-disjointness and "
        f"temporal ordering have not been verified. Generate it with "
        f"'make run ARGS=\"phase6-dataset\"'."
    )
    return pd.read_parquet(DATASET)


def test_no_address_appears_in_two_splits(built) -> None:
    """Labels are static, so a repeated address would be memorised."""
    per_address = built.groupby("address")["split"].nunique()
    offenders = per_address[per_address > 1]
    assert offenders.empty, (
        f"{len(offenders):,} addresses appear in more than one split"
    )


def test_addresses_are_unique_rows(built) -> None:
    assert built["address"].is_unique


def test_each_split_lies_inside_its_timestep_window(built) -> None:
    """The whole point of dropping boundary-spanning addresses.

    An address's observation point must not cross the boundary its first
    appearance placed it behind, or its features would aggregate transactions
    from the next split.
    """
    windows = {
        dataset.SPLIT_TRAIN: (1, dataset.TRAIN_END),
        dataset.SPLIT_VALIDATION: (dataset.TRAIN_END + 1, dataset.VALIDATION_END),
        dataset.SPLIT_TEST: (dataset.VALIDATION_END + 1, dataset.TEST_END),
    }
    for split, (low, high) in windows.items():
        rows = built[built["split"] == split]
        assert not rows.empty, f"{split} is empty"
        assert rows["first_t"].min() >= low, split
        assert rows["observed_at_t"].max() <= high, (
            f"{split} observes a timestep beyond its window: "
            f"{rows['observed_at_t'].max()} > {high}"
        )


def test_the_splits_are_temporally_ordered(built) -> None:
    train = built[built["split"] == dataset.SPLIT_TRAIN]["observed_at_t"].max()
    validation = built[built["split"] == dataset.SPLIT_VALIDATION]
    test = built[built["split"] == dataset.SPLIT_TEST]
    assert train < validation["first_t"].min()
    assert validation["observed_at_t"].max() < test["first_t"].min()


def test_unknown_labels_never_entered_the_dataset(built) -> None:
    """SPEC 1. class 3 is excluded, never mapped to licit."""
    assert set(built["y"].unique()) <= {0, 1}
    labels = pd.read_csv(DATA_ROOT / "raw" / "wallets_classes.csv")
    unknown = set(labels.loc[labels["class"] == 3, "address"])
    assert not (set(built["address"]) & unknown), (
        "an unknown-class address reached the supervised dataset"
    )


def test_the_prevalence_matches_the_label_file(built) -> None:
    """Guards against a silently mis-joined label column."""
    labels = pd.read_csv(DATA_ROOT / "raw" / "wallets_classes.csv")
    expected = dict(zip(labels["address"], labels["class"]))
    sample = built.sample(n=2000, random_state=7)
    for address, y in zip(sample["address"], sample["y"]):
        assert expected[address] == (1 if y == 1 else 2)
