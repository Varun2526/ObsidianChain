"""Tests for change-address detection and confidence scoring.

Synthetic dataset, 13 addresses and 5 transactions:

    tx 100  in {A1}    out {A2, A3}    A2 fresh, A3 reused  -> A2 is change
    tx 101  in {A3}    out {A4, A5}    both fresh           -> ambiguous
    tx 102  in {A6}    out {A7,A8,A9}  three outputs        -> batching, skipped
    tx 103  in {A10}   out {A10, A11}  A10 is its own input -> self-reference
    tx 104  in {}      out {A12, A13}  no inputs            -> coinbase, skipped

first_block_appeared_in is set so that A2 > A3 and A11 > A10, while A4 == A5.

Tests that exercise select_change_rows FAIL until the merge decision is
implemented; that is intentional and they carry a clear message.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsidianchain.cluster import change as ch
from obsidianchain.io import elliptic

INPUTS = [("A1", 100), ("A3", 101), ("A6", 102), ("A10", 103)]
OUTPUTS = [
    (100, "A2"), (100, "A3"),
    (101, "A4"), (101, "A5"),
    (102, "A7"), (102, "A8"), (102, "A9"),
    (103, "A10"), (103, "A11"),
    (104, "A12"), (104, "A13"),
]
UNIVERSE = [f"A{i}" for i in range(1, 14)]
FIRST_BLOCK = {
    "A1": 1, "A2": 10, "A3": 5, "A4": 10, "A5": 10, "A6": 1,
    "A7": 10, "A8": 10, "A9": 10, "A10": 1, "A11": 10, "A12": 1, "A13": 10,
}


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    (raw / elliptic.WALLETS_CLASSES).write_text(
        "address,class\n" + "".join(f"{a},2\n" for a in UNIVERSE), encoding="utf-8"
    )
    (raw / elliptic.ADDR_TX).write_text(
        "input_address,txId\n" + "".join(f"{a},{t}\n" for a, t in INPUTS),
        encoding="utf-8",
    )
    (raw / elliptic.TX_ADDR).write_text(
        "txId,output_address\n" + "".join(f"{t},{a}\n" for t, a in OUTPUTS),
        encoding="utf-8",
    )
    (raw / "wallets_features.csv").write_text(
        "address,Time step,first_block_appeared_in,last_block_appeared_in\n"
        + "".join(f"{a},1,{b},{b + 2}\n" for a, b in FIRST_BLOCK.items()),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def candidates(dataset: Path) -> ch.ChangeCandidates:
    graph = elliptic.load_cospend_graph(dataset, keep_labels=True)
    return ch.build_candidates(graph, dataset)


def code_of(dataset: Path, address: str) -> int:
    graph = elliptic.load_cospend_graph(dataset, keep_labels=True)
    return int(np.flatnonzero(np.asarray(graph.addresses) == address)[0])


def row_for(candidates: ch.ChangeCandidates, dataset: Path, address: str) -> pd.Series:
    code = code_of(dataset, address)
    match = candidates.table[candidates.table["code"] == code]
    assert len(match) == 1, f"expected exactly one row for {address}"
    return match.iloc[0]


# ---- weight calibration ----------------------------------------------


def test_weights_are_calibrated_against_the_default_threshold() -> None:
    """Only the corroborated band clears 0.7; one_fresh alone must not."""
    w = ch.ChangeWeights()
    assert w.self_reference >= ch.DEFAULT_THRESHOLD
    # one_fresh always drags peer_reused along (see test below), so 0.65 is
    # the real "one-fresh only" score, and it must not clear the threshold.
    assert w.one_fresh + w.peer_reused < ch.DEFAULT_THRESHOLD
    # Adding first-block corroboration must clear it.
    assert (
        w.one_fresh + w.peer_reused + w.fresher_first_block >= ch.DEFAULT_THRESHOLD
    )


def test_one_fresh_structurally_implies_peer_reused(
    candidates: ch.ChangeCandidates,
) -> None:
    """Not a coincidence: 'peer is not fresh' means 'peer has degree >= 2'.

    This is why scores of 0.50 and 0.70 are unreachable, and why the
    threshold can only be tuned in the steps 0.65 / 0.85 / 0.98.
    """
    table = candidates.table
    assert not (table["one_fresh"] & ~table["peer_reused"]).any()


def test_reachable_confidence_bands(candidates: ch.ChangeCandidates) -> None:
    reachable = {0.0, 0.15, 0.20, 0.35, 0.65, 0.85, 0.98}
    got = {round(float(v), 2) for v in candidates.table["confidence"].unique()}
    assert got <= reachable, f"unexpected band: {got - reachable}"
    assert 0.50 not in got and 0.70 not in got


def test_score_is_additive_and_bounded() -> None:
    table = pd.DataFrame(
        {
            "self_reference": [False, False, False, True],
            "one_fresh": [True, True, False, True],
            "fresher_first_block": [True, False, False, True],
            "peer_reused": [True, False, False, True],
        }
    )
    got = ch.score(table)
    assert got[0] == pytest.approx(0.85)
    assert got[1] == pytest.approx(0.50)
    assert got[2] == pytest.approx(0.0)
    assert got[3] == pytest.approx(0.98), "self-reference short-circuits"
    assert ((got >= 0) & (got <= 1)).all()


def test_custom_weights_are_respected() -> None:
    table = pd.DataFrame(
        {
            "self_reference": [False],
            "one_fresh": [True],
            "fresher_first_block": [False],
            "peer_reused": [False],
        }
    )
    assert ch.score(table, ch.ChangeWeights(one_fresh=0.9))[0] == pytest.approx(0.9)


# ---- gates ------------------------------------------------------------


def test_only_two_output_transactions_are_scored(
    candidates: ch.ChangeCandidates,
) -> None:
    assert set(candidates.table["tx_id"]) == {100, 101, 103}
    assert 102 not in set(candidates.table["tx_id"]), "3 outputs is batching"
    assert 104 not in set(candidates.table["tx_id"]), "no inputs to merge with"


def test_funnel_counts(candidates: ch.ChangeCandidates) -> None:
    assert candidates.n_transactions_with_outputs == 5
    assert candidates.n_two_output == 4          # 100, 101, 103, 104
    assert candidates.n_skipped_output_count == 1  # tx 102
    assert candidates.n_skipped_no_inputs == 1     # tx 104
    assert candidates.n_candidates == 6            # 3 transactions x 2 outputs


def test_every_scored_transaction_has_an_input_anchor(
    candidates: ch.ChangeCandidates,
) -> None:
    assert candidates.table["input_rep"].notna().all()
    assert (candidates.table["n_inputs"] >= 1).all()


# ---- individual signals ----------------------------------------------


def test_self_reference_is_detected(
    candidates: ch.ChangeCandidates, dataset: Path
) -> None:
    row = row_for(candidates, dataset, "A10")
    assert bool(row["self_reference"]) is True
    assert row["confidence"] == pytest.approx(0.98)


def test_self_reference_does_not_leak_to_the_peer(
    candidates: ch.ChangeCandidates, dataset: Path
) -> None:
    assert bool(row_for(candidates, dataset, "A11")["self_reference"]) is False


def test_one_fresh_rule_fires_for_the_fresh_output(
    candidates: ch.ChangeCandidates, dataset: Path
) -> None:
    a2 = row_for(candidates, dataset, "A2")
    a3 = row_for(candidates, dataset, "A3")
    assert bool(a2["fresh"]) is True and bool(a2["one_fresh"]) is True
    assert bool(a3["fresh"]) is False and bool(a3["one_fresh"]) is False


def test_one_fresh_rule_does_not_fire_when_both_are_fresh(
    candidates: ch.ChangeCandidates, dataset: Path
) -> None:
    """tx 101 has two fresh outputs; guessing between them is how collapse starts."""
    for address in ("A4", "A5"):
        row = row_for(candidates, dataset, address)
        assert bool(row["fresh"]) is True
        assert bool(row["one_fresh"]) is False
        assert row["confidence"] < ch.DEFAULT_THRESHOLD


def test_fresher_first_block(candidates: ch.ChangeCandidates, dataset: Path) -> None:
    assert bool(row_for(candidates, dataset, "A2")["fresher_first_block"]) is True
    assert bool(row_for(candidates, dataset, "A3")["fresher_first_block"]) is False


def test_equal_first_block_is_not_fresher(
    candidates: ch.ChangeCandidates, dataset: Path
) -> None:
    for address in ("A4", "A5"):
        assert bool(row_for(candidates, dataset, address)["fresher_first_block"]) is False


def test_peer_reused(candidates: ch.ChangeCandidates, dataset: Path) -> None:
    assert bool(row_for(candidates, dataset, "A2")["peer_reused"]) is True
    assert bool(row_for(candidates, dataset, "A3")["peer_reused"]) is False


# ---- combined confidence ---------------------------------------------


def test_confidence_of_the_clear_change_output(
    candidates: ch.ChangeCandidates, dataset: Path
) -> None:
    """A2: one_fresh + fresher + peer_reused = 0.50 + 0.20 + 0.15."""
    assert row_for(candidates, dataset, "A2")["confidence"] == pytest.approx(0.85)


def test_ambiguous_transaction_scores_below_threshold(
    candidates: ch.ChangeCandidates,
) -> None:
    tx101 = candidates.table[candidates.table["tx_id"] == 101]
    assert (tx101["confidence"] < ch.DEFAULT_THRESHOLD).all()


def test_above_threshold_selection(candidates: ch.ChangeCandidates) -> None:
    above = candidates.above(ch.DEFAULT_THRESHOLD)
    assert set(above["tx_id"]) == {100, 103}
    assert len(above) == 2


def test_features_can_be_disabled(dataset: Path) -> None:
    graph = elliptic.load_cospend_graph(dataset, keep_labels=True)
    without = ch.build_candidates(graph, dataset, use_features=False)
    assert without.features_used is False
    assert not without.table["fresher_first_block"].any()
    # A2 loses the 0.20 contribution and drops from 0.85 to the 0.65 band,
    # which is below the default threshold.
    code = code_of(dataset, "A2")
    row = without.table[without.table["code"] == code].iloc[0]
    assert row["confidence"] == pytest.approx(0.65)
    assert row["confidence"] < ch.DEFAULT_THRESHOLD


def test_requires_address_labels(dataset: Path) -> None:
    graph = elliptic.load_cospend_graph(dataset, keep_labels=False)
    with pytest.raises(ValueError, match="keep_labels"):
        ch.build_candidates(graph, dataset)


def test_signal_breakdown(candidates: ch.ChangeCandidates) -> None:
    breakdown = ch.signal_breakdown(candidates.table)
    assert set(breakdown["signal"]) == {
        "self_reference", "one_fresh", "fresher_first_block", "peer_reused"
    }
    fired = dict(zip(breakdown["signal"], breakdown["fired"]))
    assert fired["self_reference"] == 1  # A10
    assert fired["one_fresh"] == 1       # A2


def test_output_schema_is_validated(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / elliptic.TX_ADDR).write_text("txId,wrong_name\n1,A1\n")
    with pytest.raises(ValueError, match="expected"):
        ch.load_output_edges(tmp_path)


# ---- edge construction ------------------------------------------------


def test_edges_from_selection_links_change_to_an_input() -> None:
    selected = pd.DataFrame(
        {"code": [7, 9], "input_rep": [1, 2], "confidence": [0.85, 0.98]}
    )
    edges = ch.edges_from_selection(selected)
    assert edges.dtype == np.int32
    assert edges.tolist() == [[1, 7], [2, 9]]


def test_edges_from_empty_selection() -> None:
    assert ch.edges_from_selection(pd.DataFrame()).shape == (0, 2)
    assert ch.edges_from_selection(None).shape == (0, 2)


# ---- the merge decision (unimplemented) -------------------------------


@pytest.fixture()
def _decision_implemented(candidates: ch.ChangeCandidates) -> None:
    if ch.select_change_rows(candidates, ch.DEFAULT_THRESHOLD) is None:
        pytest.fail(
            "change.select_change_rows() returned None - implement the merge "
            "decision in src/obsidianchain/cluster/change.py."
        )


@pytest.mark.usefixtures("_decision_implemented")
def test_selection_respects_the_threshold(candidates: ch.ChangeCandidates) -> None:
    selected = ch.select_change_rows(candidates, ch.DEFAULT_THRESHOLD)
    assert (selected["confidence"] >= ch.DEFAULT_THRESHOLD).all()


@pytest.mark.usefixtures("_decision_implemented")
def test_at_most_one_change_output_per_transaction(
    candidates: ch.ChangeCandidates,
) -> None:
    selected = ch.select_change_rows(candidates, ch.DEFAULT_THRESHOLD)
    assert not selected["tx_id"].duplicated().any()


@pytest.mark.usefixtures("_decision_implemented")
def test_selection_picks_the_expected_outputs(
    candidates: ch.ChangeCandidates, dataset: Path
) -> None:
    """Only tx 100 survives.

    tx 101 is below threshold (both outputs fresh, so ambiguous). tx 103's
    winner is A10, which is the transaction's own single input - a self-loop,
    covered by the test below.
    """
    selected = ch.select_change_rows(candidates, ch.DEFAULT_THRESHOLD)
    assert set(selected["tx_id"]) == {100}
    assert set(selected["code"]) == {code_of(dataset, "A2")}


@pytest.mark.usefixtures("_decision_implemented")
def test_self_referencing_winner_yields_nothing_rather_than_the_runner_up(
    candidates: ch.ChangeCandidates, dataset: Path
) -> None:
    """tx 103: A10 is both the input and an output, so A10 is the change.

    A10 scores 0.98 and wins, but code == input_rep so the edge is a no-op
    and the transaction contributes nothing. The runner-up A11 must NOT be
    promoted in its place - that would assert A11 is change when A10
    demonstrably is, turning a harmless no-op into a false merge.
    """
    selected = ch.select_change_rows(candidates, ch.DEFAULT_THRESHOLD)
    assert 103 not in set(selected["tx_id"])
    assert code_of(dataset, "A11") not in set(selected["code"])


@pytest.mark.usefixtures("_decision_implemented")
def test_ties_are_dropped_rather_than_guessed(
    candidates: ch.ChangeCandidates,
) -> None:
    """Both outputs scoring identically means the signals cannot separate them."""
    table = candidates.table.copy()
    table.loc[table["tx_id"] == 101, "confidence"] = 0.9
    tied = ch.ChangeCandidates(table=table, weights=candidates.weights)
    selected = ch.select_change_rows(tied, ch.DEFAULT_THRESHOLD)
    assert 101 not in set(selected["tx_id"]), "a tie must not be guessed"


@pytest.mark.usefixtures("_decision_implemented")
def test_self_loops_are_never_selected(candidates: ch.ChangeCandidates) -> None:
    table = candidates.table.copy()
    table["input_rep"] = table["code"]
    table["confidence"] = 0.99
    degenerate = ch.ChangeCandidates(table=table, weights=candidates.weights)
    selected = ch.select_change_rows(degenerate, ch.DEFAULT_THRESHOLD)
    assert len(selected) == 0


@pytest.mark.usefixtures("_decision_implemented")
def test_self_loop_dropped_even_with_a_unique_winner(
    candidates: ch.ChangeCandidates,
) -> None:
    """Isolate the self-loop rule from the tie rule.

    One clear winner per transaction, no ties anywhere, but every winner is
    its own input anchor. Nothing may be selected.
    """
    table = candidates.table.copy()
    table["confidence"] = np.where(
        table.groupby("tx_id").cumcount() == 0, 0.99, 0.10
    )
    table["input_rep"] = table["code"]
    degenerate = ch.ChangeCandidates(table=table, weights=candidates.weights)
    assert len(ch.select_change_rows(degenerate, ch.DEFAULT_THRESHOLD)) == 0

    # Same table, anchors distinct: now the unique winners are selected,
    # which proves the previous assertion was the self-loop rule at work.
    table["input_rep"] = table["code"] + 1
    ok = ch.ChangeCandidates(table=table, weights=candidates.weights)
    assert len(ch.select_change_rows(ok, ch.DEFAULT_THRESHOLD)) == 3
