"""Phase 6 feature builders: shape, as-of-t semantics, and PEEL-1.

These use small hand-built fixtures rather than the real dataset, so they run
on a machine with no Elliptic++ download and so a failure points at one
behaviour instead of at 262,433 rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obsidianchain.features import behaviour, graph, peel
from obsidianchain.features.incidence import ROLE_IN, ROLE_OUT, Incidence


def make_incidence(rows, transactions) -> Incidence:
    """``rows`` is (code, txId, role); ``transactions`` is a dict of dicts."""
    frame = pd.DataFrame(rows, columns=["code", "txId", "role"])
    tx = pd.DataFrame.from_dict(transactions, orient="index")
    tx.index.name = "txId"
    frame["Time step"] = frame["txId"].map(tx["Time step"]).astype(np.int16)
    n = int(frame["code"].max()) + 1
    return Incidence(
        frame=frame, transactions=tx,
        addresses=np.array([f"addr{i}" for i in range(n)], dtype=object),
    )


def tx(step, *, outputs=2, inputs=1, total=1.0, in_total=1.0, out_total=0.9,
       fees=0.01, size=250.0, out_max=0.8):
    return {
        "Time step": step, "total_BTC": total, "fees": fees, "size": size,
        "num_input_addresses": inputs, "num_output_addresses": outputs,
        "in_BTC_total": in_total, "in_BTC_max": in_total, "in_BTC_mean": in_total,
        "out_BTC_total": out_total, "out_BTC_max": out_max,
        "out_BTC_mean": out_total / max(outputs, 1),
    }


# ---- M0 ----------------------------------------------------------------


def test_m0_separates_sending_from_receiving() -> None:
    inc = make_incidence(
        [(0, "t1", ROLE_IN), (0, "t2", ROLE_OUT)],
        {"t1": tx(1, in_total=5.0), "t2": tx(2, out_total=3.0)},
    )
    m0 = behaviour.build(inc)
    assert m0.loc[0, "n_txs_asof_t"] == 2
    assert m0.loc[0, "n_txs_as_sender_asof_t"] == 1
    assert m0.loc[0, "n_txs_as_receiver_asof_t"] == 1
    assert m0.loc[0, "btc_sent_total_asof_t"] == pytest.approx(5.0)
    assert m0.loc[0, "btc_received_total_asof_t"] == pytest.approx(3.0)


def test_m0_leaves_an_absent_role_null_not_zero() -> None:
    """"Never sent" and "sent nothing" are different and must stay so."""
    inc = make_incidence([(0, "t1", ROLE_OUT)], {"t1": tx(1)})
    m0 = behaviour.build(inc)
    assert pd.isna(m0.loc[0, "btc_sent_total_asof_t"])


def test_m0_gap_is_null_for_a_single_timestep_address() -> None:
    """Zero would merge "came back immediately" with "never came back"."""
    inc = make_incidence([(0, "t1", ROLE_OUT)], {"t1": tx(3)})
    m0 = behaviour.build(inc)
    assert pd.isna(m0.loc[0, "gap_mean_timesteps_asof_t"])


def test_m0_counts_distinct_active_timesteps() -> None:
    inc = make_incidence(
        [(0, "a", ROLE_OUT), (0, "b", ROLE_OUT), (0, "c", ROLE_OUT)],
        {"a": tx(1), "b": tx(1), "c": tx(5)},
    )
    m0 = behaviour.build(inc)
    assert m0.loc[0, "active_timesteps_asof_t"] == 2
    assert m0.loc[0, "timesteps_since_first_seen_asof_t"] == 4
    assert m0.loc[0, "gap_max_timesteps_asof_t"] == 4


# ---- M1 ----------------------------------------------------------------


def test_counterparty_edges_link_inputs_to_outputs() -> None:
    inc = make_incidence(
        [(0, "t1", ROLE_IN), (1, "t1", ROLE_OUT), (2, "t1", ROLE_OUT)],
        {"t1": tx(1, outputs=2)},
    )
    edges = graph.counterparty_edges(inc)
    assert set(map(tuple, edges[["code_a", "code_b"]].to_numpy())) == {(0, 1), (0, 2)}


def test_an_address_is_not_its_own_counterparty() -> None:
    inc = make_incidence(
        [(0, "t1", ROLE_IN), (0, "t1", ROLE_OUT)], {"t1": tx(1)}
    )
    assert graph.counterparty_edges(inc).empty


def test_a_very_wide_transaction_contributes_no_counterparty_edges() -> None:
    """A 69x1001 payout implies 69,069 pairs; the recipients share a payer,
    not each other. Capped as a declared DESIGN choice, not silently."""
    rows = [(0, "wide", ROLE_IN)]
    rows += [(i, "wide", ROLE_OUT) for i in range(1, 3)]
    inc = make_incidence(rows, {"wide": tx(1, outputs=2)})
    assert not graph.counterparty_edges(inc).empty

    big = [(i, "wide", ROLE_IN) for i in range(200)]
    big += [(200 + i, "wide", ROLE_OUT) for i in range(200)]
    wide = make_incidence(big, {"wide": tx(1, outputs=200, inputs=200)})
    assert len(graph.counterparty_edges(wide)) == 0, (
        "200x200 = 40,000 pairs exceeds MAX_PAIRS_PER_TX and must be dropped"
    )


def test_clustering_coefficient_excludes_a_future_closing_edge() -> None:
    """The one place the as-of-t rule actually bites in M1.

    Address 0 has neighbours 1 and 2 from a transaction at t=1. They become
    connected to each other only at t=9. As of t=1 that triangle does not
    exist yet, and counting it would be future information about 0.
    """
    inc = make_incidence(
        [
            (0, "early", ROLE_IN), (1, "early", ROLE_OUT), (2, "early", ROLE_OUT),
            (1, "late", ROLE_IN), (2, "late", ROLE_OUT),
        ],
        {"early": tx(1, outputs=2), "late": tx(9, outputs=2)},
    )
    at_one = graph.build(inc, pd.Series({0: 1, 1: 9, 2: 9}))
    assert at_one.loc[0, "local_clustering_coefficient_asof_t"] == 0.0

    at_nine = graph.build(inc, pd.Series({0: 9, 1: 9, 2: 9}))
    assert at_nine.loc[0, "local_clustering_coefficient_asof_t"] == 1.0


def test_clustering_coefficient_is_null_below_two_neighbours() -> None:
    """Undefined, not zero: no pair exists that could have been connected."""
    inc = make_incidence(
        [(0, "t1", ROLE_IN), (1, "t1", ROLE_OUT)], {"t1": tx(1)}
    )
    m1 = graph.build(inc, pd.Series({0: 1, 1: 1}))
    assert pd.isna(m1.loc[0, "local_clustering_coefficient_asof_t"])


def test_m1_never_emits_a_final_cluster_size() -> None:
    assert "cluster_size_final" not in graph.M1_COLUMNS
    assert "cluster_size_asof_t" in graph.M1_COLUMNS


# ---- M2 / PEEL-1 -------------------------------------------------------


def chain_incidence(steps) -> Incidence:
    """A linear chain: tx_i's output address is tx_{i+1}'s input."""
    rows, transactions = [], {}
    for i, step in enumerate(steps):
        name = f"c{i}"
        transactions[name] = tx(step, outputs=2, inputs=1)
        rows.append((i, name, ROLE_IN))
        rows.append((i + 1, name, ROLE_OUT))
    return make_incidence(rows, transactions)


def test_peel_requires_strictly_forward_timesteps() -> None:
    """SPEC 5.4(C): Elliptic++ has no intra-timestep ordering."""
    same = chain_incidence([3, 3, 3, 3, 3, 3])
    assert len(peel.build_chain_graph(same).hops) == 0

    forward = chain_incidence([1, 2, 3, 4, 5, 6])
    assert len(peel.build_chain_graph(forward).hops) > 0


def test_peel_depth_grows_along_a_chain() -> None:
    inc = chain_incidence([1, 2, 3, 4, 5, 6])
    chains = peel.build_chain_graph(inc)
    assert int(chains.depth_at[49].max()) == 6


def test_peel_depth_snapshots_respect_the_cutoff() -> None:
    """An address observed early must not see the chain's later hops."""
    inc = chain_incidence([1, 2, 3, 4, 5, 6])
    chains = peel.build_chain_graph(inc)
    assert int(chains.depth_at[3].max()) == 3
    assert int(chains.depth_at[6].max()) == 6


def test_peel_admits_only_small_fanout() -> None:
    """Small fan-out is a necessary filter; it is NOT a signature."""
    wide = make_incidence(
        [(0, "a", ROLE_IN), (1, "a", ROLE_OUT), (1, "b", ROLE_IN), (2, "b", ROLE_OUT)],
        {"a": tx(1, outputs=9), "b": tx(2, outputs=9)},
    )
    assert len(peel.build_chain_graph(wide).hops) == 0


def test_peel_applies_no_dominance_filter() -> None:
    """SPEC 5.4(E): 61.3% of all transactions have a dominant output, so
    dominance is the ordinary payment-plus-change shape, not a signature."""
    source = (peel.__file__)
    text = open(source, encoding="utf-8").read()
    assert "out_BTC_max" not in text.split('"""')[-1], (
        "PEEL-1 must not filter on output dominance"
    )


def test_below_min_depth_no_address_is_in_a_chain() -> None:
    inc = chain_incidence([1, 2, 3])
    cutoff = pd.Series({i: 49 for i in range(4)})
    m2 = peel.build(inc, cutoff, min_depth=5)
    assert not m2["in_chain"].any()
    assert m2["value_retention_ratio"].isna().all()


def test_value_retention_is_null_when_not_attributable() -> None:
    """SPEC 5.6: only single-input downstream transactions attribute value."""
    rows, transactions = [], {}
    for i, step in enumerate([1, 2, 3, 4, 5, 6]):
        name = f"c{i}"
        # Two input addresses downstream: value can no longer be attributed.
        transactions[name] = tx(step, outputs=2, inputs=2)
        rows.append((i, name, ROLE_IN))
        rows.append((i + 1, name, ROLE_OUT))
    inc = make_incidence(rows, transactions)
    cutoff = pd.Series({i: 49 for i in range(7)})
    m2 = peel.build(inc, cutoff, min_depth=5)
    assert m2["in_chain"].any(), "the chain itself should still be detected"
    assert not m2["value_retention_available"].any()
    assert m2["value_retention_ratio"].isna().all()


def test_value_retention_is_available_on_a_single_input_chain() -> None:
    inc = chain_incidence([1, 2, 3, 4, 5, 6])
    cutoff = pd.Series({i: 49 for i in range(7)})
    m2 = peel.build(inc, cutoff, min_depth=5)
    assert m2["value_retention_available"].any()
    assert m2.loc[m2["value_retention_available"], "value_retention_ratio"].notna().any()
