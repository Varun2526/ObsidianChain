"""M4 - the mixing / CoinJoin-like pattern detector.

The detector's value is not that it fires on a CoinJoin. Any two-line rule
does that. Its value is that it does NOT fire on the four benign shapes that
produce the same fan-in, fan-out or equal values, so most of this file is
about the things it must refuse.

Nothing here asserts that a MIXING_PATTERN transaction is unlawful. The
classification is about transaction STRUCTURE, and the tests check that the
module's own language says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obsidianchain.features import mixing
from obsidianchain.features.mixing import (
    INSUFFICIENT_DATA,
    MIXING_LIKELIHOOD,
    MIXING_PATTERN,
    NO_MIXING_SIGNAL,
)


def tx(txid, *, n_in, n_out, in_values, out_values, timestep=10):
    """One transaction row in the shape ``txs_features.csv`` supplies.

    The individual output values are NOT available in that file - only the
    summary - so the fixtures give the summary, exactly as the real loader
    does. Passing the values here and reducing them keeps the fixtures
    readable without pretending the detector sees more than it does.
    """
    in_values = np.asarray(in_values, dtype="float64")
    out_values = np.asarray(out_values, dtype="float64")
    return {
        "txId": txid,
        "Time step": timestep,
        "num_input_addresses": n_in,
        "num_output_addresses": n_out,
        "in_BTC_min": in_values.min(), "in_BTC_max": in_values.max(),
        "in_BTC_mean": in_values.mean(),
        "out_BTC_min": out_values.min(), "out_BTC_max": out_values.max(),
        "out_BTC_mean": out_values.mean(),
    }


# ---- the five required fixtures ----------------------------------------


COINJOIN = tx(
    "coinjoin", n_in=8, n_out=8,
    # Eight independent funders bring unrelated amounts...
    in_values=[0.31, 1.84, 0.52, 2.10, 0.79, 1.22, 3.05, 0.44],
    # ...and every one receives the same denomination back.
    out_values=[0.10] * 8,
)

EXCHANGE_BATCH = tx(
    "batch", n_in=1, n_out=180,
    in_values=[92.4],
    # A withdrawal batch: one hot wallet paying many different amounts.
    out_values=[0.004, 0.51, 1.9, 0.07, 12.0, 0.3, 2.2, 0.011],
)

CONSOLIDATION = tx(
    "consolidation", n_in=140, n_out=1,
    in_values=[0.002, 0.31, 0.05, 1.2, 0.009, 0.44],
    out_values=[2.0],
)

ORDINARY = tx(
    "ordinary", n_in=1, n_out=2,
    in_values=[1.5],
    out_values=[0.4, 1.09],  # payment plus change
)

INSUFFICIENT = {
    "txId": "unknown", "Time step": 10,
    "num_input_addresses": 6, "num_output_addresses": 6,
    "in_BTC_min": np.nan, "in_BTC_max": np.nan, "in_BTC_mean": np.nan,
    "out_BTC_min": np.nan, "out_BTC_max": np.nan, "out_BTC_mean": np.nan,
}

# ---- further benign shapes that resemble mixing ------------------------

UNIFORM_PAYOUT = tx(
    "payout", n_in=1, n_out=4,
    in_values=[4.1],
    out_values=[1.0] * 4,   # equal outputs, but ONE payer
)

MERCHANT_BATCH = tx(
    "merchant", n_in=2, n_out=40,
    in_values=[10.0, 8.0],
    out_values=[0.02, 0.5, 1.1, 0.3],
)

RAPID_BENIGN = tx(
    "rapid", n_in=2, n_out=2,
    in_values=[0.9, 1.1],
    out_values=[0.5, 1.48],
)

BIG_BUT_VARIED = tx(
    "big-varied", n_in=25, n_out=25,
    in_values=[0.1, 5.0, 0.3, 9.9],
    out_values=[0.2, 7.4, 0.05, 3.1],   # large both ways, values NOT uniform
)


def classify(*rows) -> pd.DataFrame:
    return mixing.score_transactions(pd.DataFrame(list(rows)))


def one(row) -> pd.Series:
    return classify(row).iloc[0]


# ---- A. the pattern is recognised --------------------------------------


def test_a_coinjoin_like_transaction_is_a_mixing_pattern() -> None:
    result = one(COINJOIN)
    assert result["mixing_class"] == MIXING_PATTERN
    assert result["mixing_score"] >= mixing.DEFAULT_CONFIG.pattern_threshold
    assert result["suppressor"] == mixing.SUPPRESSOR_NONE


def test_the_pattern_is_carried_by_more_than_one_signal() -> None:
    """A single dominant signal would be the rule this detector replaces."""
    result = one(COINJOIN)
    for signal in ("output_uniformity", "input_heterogeneity",
                   "participant_symmetry", "cardinality_signal"):
        assert result[signal] > 0.5, f"{signal} contributed nothing"


# ---- B-E. the benign shapes are refused --------------------------------


@pytest.mark.parametrize("row,expected_suppressor", [
    (EXCHANGE_BATCH, mixing.SUPPRESSOR_BATCH),
    (CONSOLIDATION, mixing.SUPPRESSOR_CONSOLIDATION),
    (ORDINARY, mixing.SUPPRESSOR_ORDINARY),
    (UNIFORM_PAYOUT, mixing.SUPPRESSOR_UNIFORM_PAYOUT),
    (MERCHANT_BATCH, mixing.SUPPRESSOR_BATCH),
])
def test_a_benign_shape_is_suppressed_with_its_reason(row, expected_suppressor) -> None:
    result = one(row)
    assert result["mixing_class"] != MIXING_PATTERN
    assert result["suppressor"] == expected_suppressor, (
        "the reason a transaction was not called a mixing pattern must be "
        "recorded, not left to be inferred from a low score"
    )


def test_an_unmeasurable_transaction_is_insufficient_data_not_clean() -> None:
    result = one(INSUFFICIENT)
    assert result["mixing_class"] == INSUFFICIENT_DATA
    assert pd.isna(result["mixing_score"]), (
        "an unmeasured transaction must not be scored zero; that would read "
        "as 'measured and clean'"
    )


def test_an_ordinary_transaction_shows_no_signal() -> None:
    assert one(ORDINARY)["mixing_class"] in (NO_MIXING_SIGNAL,)


def test_size_alone_does_not_make_a_mixing_pattern() -> None:
    """25-in/25-out with varied values is a large ordinary transaction."""
    result = one(BIG_BUT_VARIED)
    assert result["mixing_class"] != MIXING_PATTERN


def test_a_rapid_benign_transaction_is_not_flagged() -> None:
    assert one(RAPID_BENIGN)["mixing_class"] != MIXING_PATTERN


def test_no_benign_fixture_reaches_the_pattern_class() -> None:
    """The headline false-positive guarantee, over every benign fixture."""
    benign = [EXCHANGE_BATCH, CONSOLIDATION, ORDINARY, UNIFORM_PAYOUT,
              MERCHANT_BATCH, RAPID_BENIGN, BIG_BUT_VARIED]
    classes = classify(*benign)["mixing_class"].tolist()
    assert MIXING_PATTERN not in classes, dict(zip(
        [row["txId"] for row in benign], classes))


# ---- partial evidence is reported as partial ---------------------------


def test_partial_structure_is_a_likelihood_not_a_pattern() -> None:
    """Uniform outputs and enough parties, but uniform inputs too.

    Consistent with a collaborative spend and equally consistent with a
    service paying a fixed amount. The honest answer is partial.
    """
    partial = tx("partial", n_in=4, n_out=4,
                 in_values=[1.0] * 4, out_values=[0.25] * 4)
    result = one(partial)
    assert result["mixing_class"] == MIXING_LIKELIHOOD
    assert result["input_heterogeneity"] == 0.0


def test_two_parties_are_never_a_pattern() -> None:
    """Two participants learn each other's output by elimination, so there
    is no anonymity set to detect."""
    pair = tx("pair", n_in=2, n_out=2,
              in_values=[0.4, 1.6], out_values=[0.5, 0.5])
    assert one(pair)["mixing_class"] != MIXING_PATTERN


# ---- determinism -------------------------------------------------------


def test_the_detector_is_deterministic() -> None:
    rows = [COINJOIN, EXCHANGE_BATCH, CONSOLIDATION, ORDINARY, INSUFFICIENT]
    first = classify(*rows)
    for _ in range(5):
        pd.testing.assert_frame_equal(first, classify(*rows))


def test_row_order_does_not_change_a_classification() -> None:
    forward = classify(COINJOIN, EXCHANGE_BATCH, ORDINARY)
    backward = classify(ORDINARY, EXCHANGE_BATCH, COINJOIN)
    by_id = dict(zip(backward["txId"], backward["mixing_class"]))
    for txid, klass in zip(forward["txId"], forward["mixing_class"]):
        assert by_id[txid] == klass


def test_the_config_label_changes_when_a_threshold_changes() -> None:
    """Two runs under different thresholds are not comparable, so the label
    that travels into provenance must distinguish them."""
    assert (
        mixing.MixingConfig().label
        != mixing.MixingConfig(uniform_spread=0.05).label
    )


def test_the_signal_weights_sum_to_one() -> None:
    assert sum(mixing.SIGNAL_WEIGHTS.values()) == pytest.approx(1.0)


# ---- the contract requires value columns, and says so ------------------


def test_counts_alone_are_refused() -> None:
    """Without values the detector would be the rule it exists to replace."""
    with pytest.raises(KeyError, match="value structure"):
        mixing.score_transactions(pd.DataFrame([{
            "txId": "x", "num_input_addresses": 9, "num_output_addresses": 9,
        }]))


# ---- language ----------------------------------------------------------


def test_the_module_states_what_the_pattern_is_not() -> None:
    assert "not proof" in mixing.MEANING.lower()
    assert "not itself unlawful" in mixing.MEANING.lower()


def test_no_class_name_asserts_criminality() -> None:
    for name in mixing.CLASSES:
        lowered = name.lower()
        for forbidden in ("mixer", "criminal", "launder", "fraud", "illegal"):
            assert forbidden not in lowered, (
                f"{name} names a conclusion the structure cannot support"
            )


def test_every_suppressor_explains_itself() -> None:
    for code, reason in mixing.SUPPRESSORS.items():
        assert len(reason) > 60, f"{code} is unexplained"


# ---- address-level aggregation and as-of-t safety ----------------------


class FakeIncidence:
    """The two attributes ``mixing.build`` reads, and nothing else."""

    def __init__(self, transactions: pd.DataFrame, frame: pd.DataFrame):
        self.transactions = transactions.set_index("txId", drop=False)
        self.frame = frame


def build_fixture():
    transactions = pd.DataFrame([
        tx("cj-early", n_in=6, n_out=6,
           in_values=[0.3, 1.8, 0.5, 2.1, 0.8, 1.2],
           out_values=[0.1] * 6, timestep=5),
        tx("cj-late", n_in=6, n_out=6,
           in_values=[0.3, 1.8, 0.5, 2.1, 0.8, 1.2],
           out_values=[0.1] * 6, timestep=40),
        tx("plain", n_in=1, n_out=2,
           in_values=[1.0], out_values=[0.4, 0.59], timestep=5),
    ])
    rows = pd.DataFrame([
        # address 0 is in the early mixing transaction and a plain one
        {"code": 0, "txId": "cj-early", "role": 0, "Time step": 5},
        {"code": 0, "txId": "plain", "role": 1, "Time step": 5},
        # address 1 is in the LATE one only
        {"code": 1, "txId": "cj-late", "role": 0, "Time step": 40},
        # address 2 is in the plain transaction only
        {"code": 2, "txId": "plain", "role": 0, "Time step": 5},
    ])
    return FakeIncidence(transactions, rows)


def test_address_features_count_the_patterns_it_took_part_in() -> None:
    incidence = build_fixture()
    cutoff = pd.Series({0: 5, 1: 40, 2: 5})
    features = mixing.build(incidence, cutoff)

    assert features.loc[0, "mixing_tx_count_asof_t"] == 1
    assert features.loc[0, "mixing_tx_share_asof_t"] == pytest.approx(0.5)
    assert features.loc[2, "mixing_tx_count_asof_t"] == 0
    assert features.loc[2, "mixing_signal_available"] == 1


def test_a_feature_is_not_available_before_its_timestep() -> None:
    """The as-of-t guarantee: a transaction at t=40 must be invisible to an
    address observed at t=10, or the model sees the future."""
    incidence = build_fixture()
    early = mixing.build(incidence, pd.Series({0: 5, 1: 10, 2: 5}))
    assert 1 not in early.index or early.loc[1, "mixing_tx_count_asof_t"] == 0

    late = mixing.build(incidence, pd.Series({0: 5, 1: 40, 2: 5}))
    assert late.loc[1, "mixing_tx_count_asof_t"] == 1


def test_one_address_in_both_roles_counts_one_transaction() -> None:
    """An address that funds and receives from the same transaction took
    part in one transaction, not two."""
    incidence = build_fixture()
    incidence.frame = pd.concat([
        incidence.frame,
        pd.DataFrame([{"code": 0, "txId": "cj-early", "role": 1,
                       "Time step": 5}]),
    ], ignore_index=True)
    features = mixing.build(incidence, pd.Series({0: 5, 1: 40, 2: 5}))
    assert features.loc[0, "mixing_tx_count_asof_t"] == 1
    assert features.loc[0, "mixing_tx_share_asof_t"] == pytest.approx(0.5)


def test_the_address_build_is_deterministic() -> None:
    incidence = build_fixture()
    cutoff = pd.Series({0: 5, 1: 40, 2: 5})
    first = mixing.build(incidence, cutoff)
    for _ in range(3):
        pd.testing.assert_frame_equal(first, mixing.build(incidence, cutoff))


def test_the_columns_are_exactly_the_declared_contract() -> None:
    features = mixing.build(build_fixture(), pd.Series({0: 5, 1: 40, 2: 5}))
    assert list(features.columns) == mixing.M4_COLUMNS


def test_no_m4_column_names_a_label() -> None:
    """LABEL-BLIND, like every other feature group."""
    for column in mixing.M4_COLUMNS:
        for forbidden in ("illicit", "licit", "class", "y_", "label", "truth"):
            assert forbidden not in column.lower()
