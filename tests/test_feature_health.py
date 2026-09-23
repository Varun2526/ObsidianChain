"""Degenerate features must not return silently.

Nine of the thirty features in ``ps_native_v1`` carry no information another
feature does not already carry, and the pipeline shipped, documented and
ablated over them without noticing. ``is_peeling_candidate`` is constant zero
because the condition it tests is arithmetically unsatisfiable on synthesised
amounts - so "Group D: Structural Patterns" was never actually measured, and
the PS requirement for peeling detection is not met by that feature.

These tests do two jobs. The detector tests pin the diagnostics themselves on
fixtures whose answers are known by construction. The artifact tests pin the
known findings on the real frozen dataset, so the list shrinking is a visible
change rather than a silent one - which is what fixing the generator should
look like.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain.ml import diagnostics

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
PS_DATASETS = DATA_ROOT / "models" / "ps_native" / "datasets"
PS_MANIFEST = DATA_ROOT / "models" / "ps_native" / "v1" / "manifest.json"


# ---- the detectors, on fixtures with known answers ---------------------


@pytest.fixture
def rigged() -> pd.DataFrame:
    """Every degeneracy the real dataset has, reproduced deliberately."""
    rng = np.random.default_rng(0)
    base = rng.integers(1, 40, size=400).astype("float64")
    return pd.DataFrame({
        "real_a": rng.normal(size=400),
        "real_b": base,
        "always_zero": np.zeros(400),
        "always_seven": np.full(400, 7.0),
        "float_noise": rng.normal(size=400) * 1e-15,
        "copy_of_b": base.copy(),
        "log_of_b": np.log2(base),
    })


def test_a_constant_feature_is_found(rigged) -> None:
    found = {f.feature: f for f in diagnostics.find_constants(
        rigged, list(rigged.columns))}
    assert found["always_zero"].kind == diagnostics.CONSTANT
    assert found["always_seven"].kind == diagnostics.CONSTANT


def test_floating_point_residue_is_not_called_a_measurement(rigged) -> None:
    """The shape ``input_amount_std`` actually has: 2.8e-14, not zero."""
    found = {f.feature: f for f in diagnostics.find_constants(
        rigged, list(rigged.columns))}
    assert found["float_noise"].kind == diagnostics.NOISE
    assert "noise floor" in found["float_noise"].detail


def test_a_real_feature_is_not_flagged(rigged) -> None:
    flagged = {f.feature for f in diagnostics.audit_features(
        rigged, list(rigged.columns))}
    assert "real_a" not in flagged
    assert "real_b" not in flagged


def test_an_exact_duplicate_is_found(rigged) -> None:
    found = {f.feature: f for f in diagnostics.find_duplicates(
        rigged, list(rigged.columns))}
    assert "copy_of_b" in found
    assert found["copy_of_b"].related == ("real_b",)


def test_only_one_of_a_duplicate_pair_is_flagged(rigged) -> None:
    """Flagging both would delete the information along with the redundancy."""
    flagged = {f.feature for f in diagnostics.find_duplicates(
        rigged, list(rigged.columns))}
    assert ("real_b" in flagged) != ("copy_of_b" in flagged)


def test_a_log_transform_is_found(rigged) -> None:
    """The shape ``output_entropy`` has: log2 of another column."""
    found = {f.feature: f for f in diagnostics.find_dependencies(
        rigged, list(rigged.columns))}
    assert "log_of_b" in found
    assert found["log_of_b"].kind == diagnostics.DEPENDENT


def test_each_feature_is_reported_at_most_once(rigged) -> None:
    findings = diagnostics.audit_features(rigged, list(rigged.columns))
    names = [f.feature for f in findings]
    assert len(names) == len(set(names)), names


def test_a_missing_feature_is_reported_not_ignored(rigged) -> None:
    findings = diagnostics.audit_features(
        rigged, [*rigged.columns, "never_generated"])
    assert any(f.feature == "never_generated" for f in findings)


def test_healthy_features_preserve_declared_order(rigged) -> None:
    declared = list(rigged.columns)
    healthy = diagnostics.healthy_features(rigged, declared)
    assert healthy == [c for c in declared if c in set(healthy)]


def test_assert_healthy_refuses_a_degenerate_set(rigged) -> None:
    """The guard a NEW candidate must pass."""
    with pytest.raises(ValueError, match="carry no independent information"):
        diagnostics.assert_healthy(rigged, list(rigged.columns))


def test_assert_healthy_accepts_a_clean_set(rigged) -> None:
    diagnostics.assert_healthy(rigged, ["real_a", "real_b"])


def test_the_error_names_every_offender(rigged) -> None:
    with pytest.raises(ValueError) as caught:
        diagnostics.assert_healthy(rigged, list(rigged.columns))
    message = str(caught.value)
    for offender in ("always_zero", "float_noise", "copy_of_b", "log_of_b"):
        assert offender in message


def test_detection_is_deterministic(rigged) -> None:
    first = [f.as_dict() for f in diagnostics.audit_features(
        rigged, list(rigged.columns))]
    second = [f.as_dict() for f in diagnostics.audit_features(
        rigged, list(rigged.columns))]
    assert first == second


# ---- the frozen artifact -----------------------------------------------


@pytest.fixture(scope="module")
def ps_train() -> pd.DataFrame:
    path = PS_DATASETS / "train.parquet"
    if not path.is_file():
        pytest.skip(f"{path} not generated")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def ps_features() -> list:
    """The LIVE schema, not the frozen model manifest.

    ``v1/manifest.json`` still lists the 30 v1 features because the model was
    deliberately not retrained. The development data is v2, so auditing it
    against v1 would report six columns as "missing" rather than as removed.
    """
    from obsidianchain.pipeline import features_ps

    return list(features_ps.CORE_PS_FEATURE_COLUMNS)


@pytest.fixture(scope="module")
def frozen_v1_features() -> list:
    if not PS_MANIFEST.is_file():
        pytest.skip(f"{PS_MANIFEST} not generated")
    return json.loads(PS_MANIFEST.read_text())["features"]


def test_the_frozen_model_still_declares_the_v1_schema(frozen_v1_features) -> None:
    """The model was not retrained, so its manifest must not have moved.

    A v1 model served against v2 features would be reading different columns
    under the same names - which is why ps_model.py validates the schema
    version before scoring.
    """
    assert len(frozen_v1_features) == 30
    assert "output_entropy" in frozen_v1_features


def test_the_v2_development_data_has_no_degenerate_feature(
    ps_train, ps_features
) -> None:
    """The fix, asserted on the regenerated development data.

    v1 carried nine. The builder now passes the REAL per-transaction value
    summary instead of synthesising per-output amounts, and the six columns
    that only existed to summarise that synthesis are gone from the schema.
    """
    findings = diagnostics.audit_features(ps_train, ps_features)
    assert findings == [], [f.as_dict() for f in findings]


def test_the_v1_degeneracies_are_kept_as_a_historical_record() -> None:
    """The list is history now, not a live finding.

    Kept because it is the specification of what the v2 schema had to fix,
    and because a regression that reintroduced any of these columns should
    be recognisable by name rather than rediscovered.
    """
    assert len(diagnostics.KNOWN_DEGENERATE_PS_NATIVE) == 9
    for column in diagnostics.KNOWN_DEGENERATE_PS_NATIVE:
        assert isinstance(column, str)


def test_the_peeling_flag_can_now_fire(ps_train) -> None:
    """v1's flag required ``out_max >= 0.8 * total_out`` on two outputs that
    the builder had made exactly equal, i.e. ``0.5 >= 0.8``. It was constant
    zero across all 262,433 rows, so Group D detected no structural pattern
    and the PS peeling requirement was not met by it."""
    fired = int(ps_train["is_peeling_candidate"].sum())
    assert fired > 0, "the peeling flag is still incapable of firing"
    share = ps_train["is_peeling_candidate"].mean()
    assert 0.001 < share < 0.30, (
        f"peeling fires on {share:.1%} of rows; a structural shape this "
        f"common is a cardinality rule, not a detector"
    )


def test_the_mixing_flag_is_no_longer_a_bare_cardinality_rule(ps_train) -> None:
    """v1 reduced to ``input_count>=3 AND output_count>=3`` and fired on 20-34%
    of addresses. v2 additionally requires uniform outputs and varied inputs,
    which is what separates a collaborative spend from a payout batch."""
    cardinality_only = (
        (ps_train["input_count"] >= 3) & (ps_train["output_count"] >= 3)
    )
    assert not (
        ps_train["is_mixing_candidate"].astype(bool) == cardinality_only
    ).all(), "the mixing flag is still equivalent to a cardinality rule"
    assert ps_train["is_mixing_candidate"].mean() < 0.05, (
        "a mixing-like structure firing on more than 5% of addresses is "
        "measuring fan-out, not mixing"
    )


def test_the_schema_shrank_to_what_the_source_supports(ps_features) -> None:
    # 24 in /3; /4 adds the seven group-G upstream columns (exp20).
    assert len(ps_features) == 31


def test_the_fix_holds_over_the_pooled_development_data(ps_features) -> None:
    """Schema health is a property of the POOLED data, not of one window.

    Audited over train+validation together. A genuinely rare structural
    signal can be absent from any single window - see the test below - and
    that is a fact about the period, not a defect in the schema. Auditing
    per-window would conflate the two.

    The holdout is deliberately absent: it was NOT regenerated, so it still
    carries the v1 schema. That asymmetry is the seal working.
    """
    frames = []
    for name in ("train", "validation"):
        path = PS_DATASETS / f"{name}.parquet"
        if not path.is_file():
            pytest.skip(f"{path} not generated")
        frames.append(pd.read_parquet(path))
    pooled = pd.concat(frames, ignore_index=True)
    assert diagnostics.audit_features(pooled, ps_features) == []
    assert int(pooled["is_peeling_candidate"].sum()) > 0


def test_the_mixing_signal_is_absent_from_the_validation_window() -> None:
    """Recorded because it changes how a fold result must be read.

    ``is_mixing_candidate`` fires on 0.31% of the training period and NOT AT
    ALL in t35-41. That is what a rare structure looks like, and it is the
    honest shape of the data - but it means any comparison scored on a fold
    inside that window is not testing the mixing feature at all. A model that
    appears not to benefit from it there has not been shown anything.

    This is an observation to carry into interpretation, not a defect. It is
    pinned so that it stops being true visibly rather than silently.
    """
    train_path = PS_DATASETS / "train.parquet"
    val_path = PS_DATASETS / "validation.parquet"
    if not (train_path.is_file() and val_path.is_file()):
        pytest.skip("development parquets not generated")
    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(val_path)

    assert train["is_mixing_candidate"].sum() > 0
    assert train["is_mixing_candidate"].mean() < 0.05
    assert validation["is_mixing_candidate"].sum() == 0, (
        "mixing-like structure has appeared in t35-41; the interpretation "
        "note attached to fold results there needs updating"
    )


def test_the_sealed_holdout_was_not_regenerated() -> None:
    """It still carries the v1 schema, and that is correct.

    Regenerating it would have replaced the one dataset that has never
    informed a development decision - which is the only thing that makes a
    final measurement worth anything.
    """
    path = PS_DATASETS / "test.parquet"
    if not path.is_file():
        pytest.skip(f"{path} not generated")
    columns = set(pd.read_parquet(path).columns)
    assert "output_entropy" in columns, (
        "the holdout appears to have been regenerated under the v2 schema"
    )
    assert "output_spread" not in columns
