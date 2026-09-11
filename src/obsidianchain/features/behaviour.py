"""M0 - address behavioural features, as-of-t.

SPEC 3.2 M0. Every quantity is an aggregate over the transactions an address
touched with ``Time step <= t``, where ``t`` is that address's last active
timestep (SPEC 4.4). Because a boundary-spanning address is dropped before
this runs, "all of its transactions" and "its transactions inside its split"
are the same set, and the aggregate cannot reach across a split boundary.

These replace the 52 leaking columns of ``wallets_features.csv``. Where a name
matches one of those columns the quantity is deliberately the same idea,
recomputed honestly - ``btc_sent_total_asof_t`` is what
``btc_sent_total`` would have meant if it had been computed at ``t`` instead
of at the end of the dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from obsidianchain.features.incidence import ROLE_IN, ROLE_OUT, Incidence

#: The M0 block. Order is fixed so a feature matrix is reproducible.
M0_COLUMNS = [
    "n_txs_asof_t", "n_txs_as_sender_asof_t", "n_txs_as_receiver_asof_t",
    "btc_sent_total_asof_t", "btc_sent_mean_asof_t", "btc_sent_max_asof_t",
    "btc_received_total_asof_t", "btc_received_mean_asof_t",
    "btc_received_max_asof_t",
    "fees_total_asof_t", "fees_mean_asof_t",
    "tx_size_mean_asof_t",
    "input_fanin_mean_asof_t", "output_fanout_mean_asof_t",
    "active_timesteps_asof_t", "timesteps_since_first_seen_asof_t",
    "tx_per_active_timestep_asof_t",
    "gap_mean_timesteps_asof_t", "gap_max_timesteps_asof_t",
    "sent_share_of_tx_asof_t",
]


def build(incidence: Incidence) -> pd.DataFrame:
    """One row per address code, indexed by ``code``.

    The transaction-level BTC columns are attributed by role: a transaction
    where the address was an input contributes to "sent", one where it was an
    output contributes to "received". ``total_BTC`` is the whole transaction's
    value and is NOT attributed to the address - it is not the address's
    share, and treating it as such would inflate every total.
    """
    frame = incidence.frame
    tx = incidence.transactions

    joined = frame.join(
        tx[["total_BTC", "fees", "size", "num_input_addresses",
            "num_output_addresses", "in_BTC_total", "out_BTC_total"]],
        on="txId",
    )
    grouped = joined.groupby("code")

    out = pd.DataFrame(index=grouped.size().index)
    out.index.name = "code"
    out["n_txs_asof_t"] = grouped.size()

    sent = joined[joined["role"] == ROLE_IN].groupby("code")
    received = joined[joined["role"] == ROLE_OUT].groupby("code")

    out["n_txs_as_sender_asof_t"] = sent.size().reindex(out.index, fill_value=0)
    out["n_txs_as_receiver_asof_t"] = received.size().reindex(out.index, fill_value=0)

    # in_BTC_total is what the transaction consumed, so it is the natural
    # magnitude for a transaction the address helped fund. out_BTC_total is
    # what it produced, the magnitude for one the address received from.
    out["btc_sent_total_asof_t"] = sent["in_BTC_total"].sum().reindex(out.index)
    out["btc_sent_mean_asof_t"] = sent["in_BTC_total"].mean().reindex(out.index)
    out["btc_sent_max_asof_t"] = sent["in_BTC_total"].max().reindex(out.index)
    out["btc_received_total_asof_t"] = received["out_BTC_total"].sum().reindex(out.index)
    out["btc_received_mean_asof_t"] = received["out_BTC_total"].mean().reindex(out.index)
    out["btc_received_max_asof_t"] = received["out_BTC_total"].max().reindex(out.index)

    out["fees_total_asof_t"] = grouped["fees"].sum()
    out["fees_mean_asof_t"] = grouped["fees"].mean()
    out["tx_size_mean_asof_t"] = grouped["size"].mean()
    out["input_fanin_mean_asof_t"] = grouped["num_input_addresses"].mean()
    out["output_fanout_mean_asof_t"] = grouped["num_output_addresses"].mean()

    steps = grouped["Time step"]
    first, last = steps.min(), steps.max()
    out["active_timesteps_asof_t"] = steps.nunique()
    out["timesteps_since_first_seen_asof_t"] = (last - first).astype(np.int32)
    out["tx_per_active_timestep_asof_t"] = (
        out["n_txs_asof_t"] / out["active_timesteps_asof_t"].replace(0, np.nan)
    )

    gaps = _timestep_gaps(frame)
    out["gap_mean_timesteps_asof_t"] = gaps["mean"].reindex(out.index)
    out["gap_max_timesteps_asof_t"] = gaps["max"].reindex(out.index)

    out["sent_share_of_tx_asof_t"] = (
        out["n_txs_as_sender_asof_t"] / out["n_txs_asof_t"].replace(0, np.nan)
    )
    return out[M0_COLUMNS]


def _timestep_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean and max gap between an address's consecutive active timesteps.

    A single-timestep address has no gap to measure. It gets NaN, not zero:
    "never came back" and "came back immediately" are different behaviours and
    zero would merge them.
    """
    distinct = (
        frame[["code", "Time step"]].drop_duplicates()
        .sort_values(["code", "Time step"])
    )
    delta = distinct.groupby("code")["Time step"].diff()
    distinct = distinct.assign(gap=delta)
    agg = distinct.groupby("code")["gap"].agg(["mean", "max"])
    return agg
