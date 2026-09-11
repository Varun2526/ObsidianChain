"""Generate the five Phase 7 alert artifacts. Offline, never at request time.

Why this runs offline
---------------------
The API's guarantee is that a response is a file the pipeline already wrote.
Scoring inside a request handler would produce numbers no manifest describes,
and would put a LightGBM booster one call away from an HTTP route. So every
number an investigator sees is computed here, once, and written down.

What is reused rather than rebuilt
----------------------------------
Scoring goes through :func:`obsidianchain.ml.experiment.fit_full` - Phase 6's
own function, unchanged. Phase 6 computed per-address risk and SHAP and then
persisted only the aggregates, so this materialises what it already knew
rather than reimplementing it. No Phase 6 modelling decision is reopened
here: not the features, not the split, not the calibration, not the bands.

Test split only
---------------
Alerts are generated from the TEST split. Train and validation scores are
optimistic by construction - the trees were fitted on one and the calibrator
on the other - and ranking them beside test scores would put the most
confidently wrong rows at the top of an investigator's queue.

No labels, anywhere
-------------------
``y`` is dropped before anything is written. An investigation tool that
displayed the ground-truth label would be a label viewer, and on real data
there is no label to display. ``tests/test_phase7_alerts.py`` asserts no
artifact and no response carries one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from obsidianchain.alerts import contract
from obsidianchain.features import dataset as ds

#: Ranking aggregation. AGG-TOPK is robust to a single outlier dominating a
#: very large cluster, which is the super-cluster stress case SPEC 0.6
#: records. All four are served on every alert regardless; Phase 6 measured
#: their cluster PR-AUC within 0.140-0.152 of each other, which is too narrow
#: to call a winner, so this is a stated default rather than a finding.
RANKING_AGGREGATION = "AGG-TOPK"
TOPK = 3
QUANTILE = 0.90

#: Explanations kept per member address. Eight is enough to show why a row
#: scored where it did without turning the artifact into a full SHAP dump.
EXPLANATIONS_PER_MEMBER = 8

#: Relationship rows kept per alert, highest timestep first. A cluster of
#: 11,001 addresses would otherwise emit millions of edges for a panel that
#: can render a few hundred.
RELATIONSHIPS_PER_ALERT = 200

LABEL_COLUMNS = ("y", "class", "label", "truth", "is_illicit")


@dataclass(frozen=True)
class AlertArtifacts:
    alerts: pd.DataFrame
    members: pd.DataFrame
    explanations: pd.DataFrame
    timeline: pd.DataFrame
    relationships: pd.DataFrame

    def named(self) -> dict[str, pd.DataFrame]:
        return {
            "alerts": self.alerts, "alert_members": self.members,
            "alert_explanations": self.explanations,
            "alert_timeline": self.timeline,
            "alert_relationships": self.relationships,
        }


def feature_group_of(column: str) -> str | None:
    for group, columns in ds.FEATURE_GROUPS.items():
        if column in columns:
            return group
    return None


def build(data_root, dataset_frame: pd.DataFrame, run_fingerprint: str,
          ) -> tuple[AlertArtifacts, dict]:
    """Score the test split and shape the five artifacts."""
    from obsidianchain.ml import experiment, model

    trained, scored = experiment.fit_full(dataset_frame)
    scored = scored.reset_index(drop=True)

    index = pd.read_parquet(
        data_root / "processed" / "address_clusters.parquet",
        columns=["address", "cluster_id"],
    )
    sizes = pd.read_parquet(
        data_root / "processed" / "clusters.parquet",
        columns=["cluster_id", "size"],
    ).set_index("cluster_id")["size"]

    joined = scored.merge(index, on="address", how="inner")
    contributions = model.shap_contributions(trained, joined)

    alerts = _alerts(joined, sizes, run_fingerprint, contributions)
    members = _members(joined, alerts)
    explanations = _explanations(joined, alerts, contributions)
    timeline = _timeline(data_root, joined, alerts)
    relationships = _relationships(data_root, joined, alerts)

    summary = {
        "n_alerts": int(len(alerts)),
        "n_members": int(len(members)),
        "ranking_aggregation": RANKING_AGGREGATION,
        "severity_counts": (
            alerts["severity"].value_counts().to_dict() if len(alerts) else {}
        ),
        "bands": [
            {"band": b.name, "target": b.target, "threshold": b.threshold,
             "support": b.support, "validation_precision": b.precision,
             "populated": b.populated}
            for b in trained.bands
        ],
        "n_features": len(trained.features),
    }
    return AlertArtifacts(
        alerts=alerts, members=members, explanations=explanations,
        timeline=timeline, relationships=relationships,
    ), summary


def _aggregate(grouped) -> pd.DataFrame:
    out = pd.DataFrame(index=grouped.size().index)
    out["AGG-MAX"] = grouped["risk"].max()
    out["AGG-TOPK"] = grouped["risk"].apply(
        lambda s: s.nlargest(min(TOPK, len(s))).mean()
    )
    out["AGG-QUANT"] = grouped["risk"].quantile(QUANTILE)
    return out


def _alerts(joined, sizes, run_fingerprint, contributions) -> pd.DataFrame:
    grouped = joined.groupby("cluster_id")
    out = _aggregate(grouped)

    weights = joined["n_txs_asof_t"].fillna(0.0) + 1.0
    weighted = joined.assign(_w=weights, _wr=joined["risk"] * weights)
    out["AGG-WMEAN"] = (
        weighted.groupby("cluster_id")["_wr"].sum()
        / weighted.groupby("cluster_id")["_w"].sum()
    )

    out["risk_score"] = out[RANKING_AGGREGATION]
    out["n_members_scored"] = grouped.size()
    out["n_members_total"] = sizes.reindex(out.index).fillna(
        out["n_members_scored"]
    ).astype("int64")
    out["first_t"] = grouped["first_t"].min().astype("int16")
    out["last_t"] = grouped["observed_at_t"].max().astype("int16")
    out["n_transactions"] = grouped["n_txs_asof_t"].sum()
    out["btc_sent_total"] = grouped["btc_sent_total_asof_t"].sum()
    out["btc_received_total"] = grouped["btc_received_total_asof_t"].sum()
    out["unique_counterparties"] = grouped["unique_counterparties_asof_t"].sum()
    out["peel_chain_members"] = grouped["in_chain"].sum().astype("int32")
    out["net_evidence_members"] = grouped["net_has_evidence"].sum().astype("int32")
    out["net_minimum_members"] = (
        grouped["net_reaches_production_minimum"].sum().astype("int32")
    )

    # Severity of the CLUSTER is the highest severity any member reached.
    # Taking the aggregate score's band instead would let a cluster holding a
    # CRITICAL address be labelled MEDIUM, which is the wrong way for this
    # error to point on an investigator's queue.
    order = {name: i for i, name in enumerate(contract.SEVERITIES)}
    worst = joined.assign(_rank=joined["severity"].map(order)).groupby(
        "cluster_id"
    )["_rank"].min()
    inverse = {i: name for name, i in order.items()}
    out["severity"] = worst.reindex(out.index).map(inverse)
    out["top_member_risk"] = grouped["risk"].max()

    signals = _top_cluster_signals(joined, contributions)
    for i in range(3):
        out[f"top_signal_{i + 1}"] = signals.get(i).reindex(out.index)

    out = out.reset_index()
    out["alert_id"] = [
        contract.make_alert_id(run_fingerprint, c) for c in out["cluster_id"]
    ]
    out = out.sort_values(
        ["risk_score", "cluster_id"], ascending=[False, True]
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    out["ranking_aggregation"] = RANKING_AGGREGATION
    return out


def _top_cluster_signals(joined, contributions) -> dict:
    """The three features that moved the cluster's members most.

    Mean absolute contribution across the cluster's members, so one extreme
    address cannot name the whole cluster's explanation.
    """
    features = [c for c in contributions.columns if c != "base_value"]
    frame = contributions[features].abs()
    frame = frame.assign(cluster_id=joined["cluster_id"].to_numpy())
    means = frame.groupby("cluster_id").mean()
    ordered = means.apply(lambda row: row.nlargest(3).index.tolist(), axis=1)
    return {
        i: ordered.map(lambda names, i=i: names[i] if len(names) > i else None)
        for i in range(3)
    }


def _members(joined, alerts) -> pd.DataFrame:
    lookup = alerts.set_index("cluster_id")["alert_id"]
    features = [c for group in ds.FEATURE_GROUPS.values() for c in group]
    keep = ["address", "risk", "severity", "observed_at_t", "first_t"] + features
    out = joined[keep].copy()
    out.insert(0, "alert_id", joined["cluster_id"].map(lookup).to_numpy())
    out = out.rename(columns={"risk": "risk_score"})
    return out.sort_values(
        ["alert_id", "risk_score"], ascending=[True, False]
    ).reset_index(drop=True)


def _explanations(joined, alerts, contributions) -> pd.DataFrame:
    """Top-K SHAP contributions per member, categorised.

    ``category`` distinguishes what KIND of claim each row is. A contribution
    is always a MODEL_SIGNAL - it describes the model. The feature's own
    value is separately BLOCKCHAIN_CONTEXT or NETWORK_CONTEXT, or
    INSUFFICIENT_EVIDENCE when it was never computable for this address.
    """
    lookup = alerts.set_index("cluster_id")["alert_id"]
    features = [c for c in contributions.columns if c != "base_value"]
    values = contributions[features].to_numpy()
    order = np.argsort(-np.abs(values), axis=1)[:, :EXPLANATIONS_PER_MEMBER]

    n = len(joined)
    rows = {
        "alert_id": np.repeat(
            joined["cluster_id"].map(lookup).to_numpy(), order.shape[1]
        ),
        "address": np.repeat(joined["address"].to_numpy(), order.shape[1]),
        "rank": np.tile(
            np.arange(1, order.shape[1] + 1, dtype=np.int32), n
        ),
        "feature": np.asarray(features, dtype=object)[order].reshape(-1),
        "contribution": np.take_along_axis(values, order, axis=1).reshape(-1),
    }
    out = pd.DataFrame(rows)

    feature_values = joined[features].to_numpy(dtype=float, na_value=np.nan)
    out["feature_value"] = np.take_along_axis(
        feature_values, order, axis=1
    ).reshape(-1)

    group_of = {f: feature_group_of(f) for f in features}
    out["feature_group"] = out["feature"].map(group_of)
    out["signal_category"] = contract.MODEL_SIGNAL
    context = out["feature_group"].map(contract.GROUP_CATEGORY)
    out["value_category"] = np.where(
        out["feature_value"].isna(), contract.INSUFFICIENT_EVIDENCE, context
    )
    out["base_value"] = np.repeat(
        contributions["base_value"].to_numpy(), order.shape[1]
    )
    return out


def _timeline(data_root, joined, alerts) -> pd.DataFrame:
    """Per-timestep activity for each alert, from the incidence substrate.

    Rebuilt here rather than carried through the dataset because the Phase 6
    matrix is one row per address with no time axis left on it - the
    aggregation to ``observed_at_t`` is what made it a feature matrix.
    """
    from obsidianchain.features.incidence import ROLE_IN, ROLE_OUT, load_incidence

    incidence = load_incidence(data_root)
    lookup = alerts.set_index("cluster_id")["alert_id"]
    membership = joined[["address", "cluster_id"]].copy()
    membership["alert_id"] = membership["cluster_id"].map(lookup)

    code_of = pd.Series(
        np.arange(len(incidence.addresses), dtype=np.int64),
        index=incidence.addresses,
    )
    membership["code"] = code_of.reindex(membership["address"]).to_numpy()
    membership = membership.dropna(subset=["code"])
    membership["code"] = membership["code"].astype(np.int64)

    frame = incidence.frame.merge(
        membership[["code", "alert_id"]], on="code", how="inner"
    )
    tx = incidence.transactions
    frame = frame.join(tx[["in_BTC_total", "out_BTC_total"]], on="txId")

    grouped = frame.groupby(["alert_id", "Time step"])
    out = pd.DataFrame({
        "n_transactions": grouped["txId"].nunique(),
        "n_active_addresses": grouped["code"].nunique(),
    })
    sent = frame[frame["role"] == ROLE_IN].groupby(["alert_id", "Time step"])
    received = frame[frame["role"] == ROLE_OUT].groupby(["alert_id", "Time step"])
    out["btc_sent"] = sent["in_BTC_total"].sum().reindex(out.index)
    out["btc_received"] = received["out_BTC_total"].sum().reindex(out.index)
    out = out.reset_index().rename(columns={"Time step": "timestep"})
    return out.sort_values(["alert_id", "timestep"]).reset_index(drop=True)


def _relationships(data_root, joined, alerts) -> pd.DataFrame:
    """Only the two relationships this data supports.

    ``CO_SPEND_COMPONENT`` is the heuristic that formed the cluster.
    ``FUNDED_VIA_TRANSACTION`` is an observed ledger fact. Nothing derived
    from network observations appears here: the network layer emits
    cannot-link only and can never say two addresses are the same party.
    """
    from obsidianchain.features.graph import counterparty_edges
    from obsidianchain.features.incidence import ROLE_IN, load_incidence

    incidence = load_incidence(data_root)
    edges = counterparty_edges(incidence)
    addresses = incidence.addresses

    lookup = alerts.set_index("cluster_id")["alert_id"]
    membership = joined[["address", "cluster_id"]].copy()
    membership["alert_id"] = membership["cluster_id"].map(lookup)
    code_of = pd.Series(
        np.arange(len(addresses), dtype=np.int64), index=addresses
    )
    membership["code"] = code_of.reindex(membership["address"]).to_numpy()
    membership = membership.dropna(subset=["code"])
    alert_of_code = dict(
        zip(membership["code"].astype(np.int64), membership["alert_id"])
    )

    frames = []

    # FUNDED_VIA_TRANSACTION: one member was an input to a transaction that
    # paid another member. An observed ledger fact.
    a_alert = edges["code_a"].map(alert_of_code)
    b_alert = edges["code_b"].map(alert_of_code)
    inside = edges[a_alert.notna() & (a_alert.to_numpy() == b_alert.to_numpy())]
    if not inside.empty:
        frames.append(pd.DataFrame({
            "alert_id": inside["code_a"].map(alert_of_code).to_numpy(),
            "address_a": addresses[inside["code_a"].to_numpy()],
            "address_b": addresses[inside["code_b"].to_numpy()],
            "relationship": contract.FUNDED,
            "timestep": inside["Time step"].to_numpy(),
        }))

    # CO_SPEND_COMPONENT: two members were inputs to the SAME transaction.
    # This is the relationship that actually formed the cluster, and without
    # it the graph panel would be almost empty - a co-spend cluster groups
    # INPUTS, so its members are rarely the input and output of one payment.
    #
    # Emitted as STAR edges, exactly as Phase 1 builds the union: the
    # lowest-coded input of each transaction linked to every other input,
    # k-1 edges rather than k(k-1)/2. A 1,001-input transaction would
    # otherwise contribute half a million edges describing one fact.
    members = set(alert_of_code)
    spends = incidence.frame[
        (incidence.frame["role"] == ROLE_IN)
        & incidence.frame["code"].isin(members)
    ]
    grouped = spends.groupby("txId")
    rows = []
    for _, block in grouped:
        codes = sorted(set(block["code"].tolist()))
        if len(codes) < 2:
            continue
        hub = codes[0]
        alert_id = alert_of_code.get(hub)
        step = int(block["Time step"].iloc[0])
        for other in codes[1:]:
            if alert_of_code.get(other) != alert_id:
                continue
            rows.append((alert_id, addresses[hub], addresses[other],
                         contract.CO_SPEND, step))
    if rows:
        frames.append(pd.DataFrame(rows, columns=[
            "alert_id", "address_a", "address_b", "relationship", "timestep",
        ]))

    if not frames:
        return pd.DataFrame(columns=[
            "alert_id", "address_a", "address_b", "relationship", "timestep",
        ])
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["alert_id", "timestep"], ascending=[True, False])
    return out.groupby("alert_id", group_keys=False).head(
        RELATIONSHIPS_PER_ALERT
    ).reset_index(drop=True)


def assert_no_labels(frames: dict[str, pd.DataFrame]) -> None:
    """Refuse to publish an alert artifact carrying a ground-truth label.

    Checked here rather than only in a test, because the failure is silent:
    a label column would render as just another number in a dashboard and
    nobody looking at the page could tell it was the answer.
    """
    for name, frame in frames.items():
        offenders = [c for c in frame.columns if c in LABEL_COLUMNS]
        if offenders:
            raise RuntimeError(
                f"{name} carries label column(s) {offenders}. An alert "
                f"artifact must not contain ground truth: on real data there "
                f"is no label, and displaying one turns an investigation "
                f"tool into a label viewer."
            )
