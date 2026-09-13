"""Network-layer to blockchain-layer correlation, for the investigation view.

What this joins
---------------
``observations.parquet`` carries ``txid, observer_id, peer_ip, peer_port,
peer_asn, timestamp_ms``. ``AddrTx``/``TxAddr`` carry ``address <-> txid``.
Joining them on ``txid`` produces the three-node relation the problem
statement asks for::

    IP  --announced-->  TRANSACTION  --involves-->  WALLET

Every edge above is an observed fact. None of them is an ownership claim.

The correction that shapes this module
--------------------------------------
Measured on the frozen dataset: **170,899 of 202,804 transactions (84.3%)
were announced by MORE THAN ONE peer IP.** That is what gossip relay looks
like - each observer hears a transaction from whichever peer reached it
first, and those peers differ.

So ``peer_ip`` is a RELAYING VANTAGE POINT, not an originator. A panel that
labelled it "sender" would be asserting the single thing this project has
spent five phases refusing to assert. The artifact therefore records, per
transaction, how many distinct peers announced it and how many observers
saw it, and the API wording says what that means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Transactions surfaced per alert, most recently announced first. A cluster
#: can touch thousands; an investigator can read tens.
TRANSACTIONS_PER_ALERT = 40

#: Announcing peers surfaced per transaction. Most transactions here have
#: several; showing all of them for all transactions would bury the alert.
PEERS_PER_TRANSACTION = 6

COLUMNS = [
    "alert_id", "txid", "peer_ip", "peer_port", "peer_asn",
    "first_seen_ms", "last_seen_ms", "observers", "announcing_peers",
    "address", "address_role",
]

ROLE_INPUT = "input"
ROLE_OUTPUT = "output"


def build(data_root, joined: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    """One row per (alert, transaction, announcing peer, member address).

    Args:
        data_root: the data root, for the raw edgelists and observations.
        joined: scored test-split members with ``address`` and ``cluster_id``.
        alerts: the alert index, for ``cluster_id -> alert_id``.
    """
    from obsidianchain.features.incidence import ROLE_IN, load_incidence

    incidence = load_incidence(data_root)
    observations = pd.read_parquet(
        data_root / "processed" / "network" / "observations.parquet"
    )

    lookup = alerts.set_index("cluster_id")["alert_id"]
    membership = joined[["address", "cluster_id"]].copy()
    membership["alert_id"] = membership["cluster_id"].map(lookup)

    code_of = pd.Series(
        np.arange(len(incidence.addresses), dtype=np.int64),
        index=incidence.addresses,
    )
    membership["code"] = code_of.reindex(membership["address"]).to_numpy()
    membership = membership.dropna(subset=["code", "alert_id"])
    membership["code"] = membership["code"].astype(np.int64)

    # address -> transactions it took part in, with the role it played
    touching = incidence.frame.merge(
        membership[["code", "alert_id", "address"]], on="code", how="inner"
    )
    if touching.empty:
        return pd.DataFrame(columns=COLUMNS)
    touching["address_role"] = np.where(
        touching["role"] == ROLE_IN, ROLE_INPUT, ROLE_OUTPUT
    )

    # Per (txid, peer_ip): when it was first and last seen, and by how many
    # observers. Rolled up here rather than in the API so a request never
    # touches 1.59 million observation rows.
    wanted = set(touching["txId"].unique())
    relevant = observations[observations["txid"].isin(wanted)]
    if relevant.empty:
        return pd.DataFrame(columns=COLUMNS)

    per_peer = relevant.groupby(["txid", "peer_ip"], as_index=False).agg(
        peer_port=("peer_port", "first"),
        peer_asn=("peer_asn", "first"),
        first_seen_ms=("timestamp_ms", "min"),
        last_seen_ms=("timestamp_ms", "max"),
        observers=("observer_id", "nunique"),
    )
    # How many DISTINCT peers announced each transaction. The number that
    # stops a reader treating one peer as "the" source.
    peers_per_tx = per_peer.groupby("txid")["peer_ip"].transform("size")
    per_peer["announcing_peers"] = peers_per_tx.to_numpy()

    per_peer = per_peer.sort_values(
        ["txid", "first_seen_ms"]
    ).groupby("txid", group_keys=False).head(PEERS_PER_TRANSACTION)

    out = touching[["alert_id", "txId", "address", "address_role"]].rename(
        columns={"txId": "txid"}
    ).drop_duplicates().merge(per_peer, on="txid", how="inner")

    # Cap transactions per alert, newest announcement first.
    ranked = out.groupby(["alert_id", "txid"], as_index=False)["first_seen_ms"].min()
    ranked = ranked.sort_values(
        ["alert_id", "first_seen_ms"], ascending=[True, False]
    )
    keep = ranked.groupby("alert_id", group_keys=False).head(TRANSACTIONS_PER_ALERT)
    keep = set(map(tuple, keep[["alert_id", "txid"]].to_numpy()))
    mask = [
        (a, t) in keep for a, t in zip(out["alert_id"], out["txid"])
    ]
    out = out[mask]

    out = out.sort_values(
        ["alert_id", "first_seen_ms", "txid"], ascending=[True, False, True]
    )
    return out[COLUMNS].reset_index(drop=True)


def summarise_alert(frame: pd.DataFrame, data_root=None) -> dict:
    """Network footprint of one alert, with GeoIP/ASN facts attached."""
    from obsidianchain import geoip

    if frame.empty:
        return {
            "transactions": 0, "announcing_peers": 0, "asns": 0,
            "observers": 0, "geo": None,
        }
    facts = geoip.summarise(
        frame["peer_ip"].dropna().tolist(),
        frame["peer_asn"].dropna().tolist(),
        data_root=data_root,
    )
    return {
        "transactions": int(frame["txid"].nunique()),
        "announcing_peers": int(frame["peer_ip"].nunique()),
        "asns": int(frame["peer_asn"].nunique()),
        "observers": int(frame["observers"].max()),
        "geo": facts,
    }
