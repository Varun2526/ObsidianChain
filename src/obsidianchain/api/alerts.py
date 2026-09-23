"""GET /api/alerts and /api/alerts/{alert_id} - the investigation surface.

Reads five precomputed parquet artifacts and slices them. Nothing here
scores, clusters, pools, aggregates or evaluates: every number was written by
``phase7-alerts``, and ``api/boundary.py`` makes both the feature builders and
the model unreachable by import so that stays true.

What this endpoint refuses to say
---------------------------------
No response asserts that an IP owns a wallet, that an IP sent a transaction,
or that two addresses belong to one person. The network block is labelled
NETWORK_CONTEXT, carries the synthetic warning, and states in its own body
that the network layer can only ever say two groups look DIFFERENT. Where a
quantity was not computable the response says INSUFFICIENT_EVIDENCE rather
than returning a zero that would read as a measurement.
"""

from __future__ import annotations

import pandas as pd

from obsidianchain.alerts import contract
from obsidianchain.api import artifacts, provenance_gate

#: Members and explanations surfaced on a detail response. A cluster of
#: 11,001 addresses cannot be rendered, and paging it would hide the reason
#: the alert exists - so the highest-risk members are shown and the count of
#: what was withheld is stated in the response.
MEMBERS_IN_DETAIL = 25
EXPLANATIONS_PER_MEMBER = 8
RELATIONSHIPS_IN_DETAIL = 200

DEFAULT_LIMIT = 50
MAX_LIMIT = 500


def current_run_fingerprint(sidecar: dict) -> str:
    full = sidecar.get("run_fingerprint")
    if not isinstance(full, str) or len(full) < 16:
        raise provenance_gate.ProvenanceRefusedError(
            "the alert sidecar carries no usable run_fingerprint, so an "
            "alert id cannot be checked against the run that produced it."
        )
    return full[:16]


def _clean(value):
    """Missing -> None, numpy scalar -> Python scalar.

    NaN is not representable in JSON, and in this layer it always means "not
    computable for this address" - which the caller is told explicitly via
    INSUFFICIENT_EVIDENCE rather than by receiving a zero.
    """
    import pandas as pd

    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def _provenance_block(sidecar: dict) -> dict:
    """Model, dataset and as-of semantics, on every response."""
    artifact = sidecar.get("artifact") or {}
    return {
        "provenance_type": sidecar.get("provenance_type"),
        "dataset_id": sidecar.get("dataset_id"),
        "dataset_sha256": sidecar.get("dataset_sha256"),
        "synthetic_network": sidecar.get("synthetic_network"),
        "run_fingerprint": sidecar.get("run_fingerprint"),
        "public_run_fingerprint": current_run_fingerprint(sidecar),
        "phase6_dataset_fingerprint": (sidecar.get("inputs") or {}).get(
            "phase6_dataset_sha256"
        ),
        "artifact_schema": artifact.get("artifact_schema"),
        "model": artifact.get("model"),
        "feature_semantics": artifact.get("feature_semantics"),
        "split": artifact.get("split"),
        "scored_split": artifact.get("scored_split"),
        "severity_bands": artifact.get("severity_bands"),
        "ranking_aggregation": artifact.get("ranking_aggregation"),
        "git_revision": sidecar.get("git_revision"),
        "notes": sidecar.get("notes"),
    }


def list_alerts(root=None, *, severity=None, min_risk=None, max_risk=None,
                first_t=None, last_t=None, limit=DEFAULT_LIMIT, offset=0) -> dict:
    """Ranked alerts, filtered. A slice of one parquet file."""
    frame, sidecar = artifacts.load_alerts(root)
    fingerprint = current_run_fingerprint(sidecar)

    total = int(len(frame))
    if severity:
        wanted = {s.upper() for s in severity}
        unknown = wanted - set(contract.SEVERITIES)
        if unknown:
            raise AlertFilterError(
                f"unknown severity {sorted(unknown)}; expected a subset of "
                f"{list(contract.SEVERITIES)}"
            )
        frame = frame[frame["severity"].isin(wanted)]
    if min_risk is not None:
        frame = frame[frame["risk_score"] >= float(min_risk)]
    if max_risk is not None:
        frame = frame[frame["risk_score"] <= float(max_risk)]
    if first_t is not None:
        frame = frame[frame["last_t"] >= int(first_t)]
    if last_t is not None:
        frame = frame[frame["first_t"] <= int(last_t)]

    matched = int(len(frame))
    limit = max(1, min(int(limit), MAX_LIMIT))
    offset = max(0, int(offset))
    page = frame.sort_values("rank").iloc[offset:offset + limit]

    return {
        "run_fingerprint": fingerprint,
        "alert_count_total": total,
        "alert_count_matched": matched,
        "offset": offset,
        "limit": limit,
        "alerts": [_summary(row) for row in page.to_dict("records")],
        "meaning": contract.ALERT_MEANING,
        "score_scope": contract.SCORE_SCOPE,
        "provenance": _provenance_block(sidecar),
    }


class AlertFilterError(ValueError):
    """A filter value the endpoint cannot honour."""


def _summary(row: dict) -> dict:
    """Shape one alert row.

    Takes a MAPPING, never a row object. ``row.rank`` on a pandas Series
    resolves to ``Series.rank``, the method - so attribute access here works
    for a namedtuple from ``itertuples`` and silently returns a bound method
    for a Series from ``.iloc``. Dict access has one meaning for both.
    """
    return {
        "alert_id": row["alert_id"],
        "rank": int(row["rank"]),
        "cluster_id": int(row["cluster_id"]),
        "risk_score": _clean(row["risk_score"]),
        "severity": row["severity"],
        "ranking_aggregation": row["ranking_aggregation"],
        "members_scored": int(row["n_members_scored"]),
        "members_total": int(row["n_members_total"]),
        "first_timestep": int(row["first_t"]),
        "last_timestep": int(row["last_t"]),
        "top_signals": [
            s for s in (row["top_signal_1"], row["top_signal_2"],
                        row["top_signal_3"])
            if isinstance(s, str)
        ],
    }


def get_alert(raw_id: str, root=None) -> dict:
    """One alert, with everything an investigator needs to open it."""
    import pandas as pd

    requested_run, cluster_id = contract.parse_alert_id(raw_id)
    frame, sidecar = artifacts.load_alerts(root)
    current = current_run_fingerprint(sidecar)
    if requested_run != current:
        raise contract.AlertIdStaleError(
            f"alert id {raw_id!r} was minted against run {requested_run!r}; "
            f"the artifact on disk is run {current!r}. Alert ids do not "
            f"survive a regeneration, because the cluster a number refers to "
            f"can change. Re-open the alert from the current list."
        )

    match = frame[frame["cluster_id"] == cluster_id]
    if match.empty:
        raise contract.AlertNotFoundError(
            f"no alert for cluster {cluster_id} in run {current}."
        )
    alert = match.iloc[0].to_dict()

    members = artifacts.load_alert_table(root, "alert_members")
    members = members[members["alert_id"] == raw_id].sort_values(
        "risk_score", ascending=False
    )
    shown = members.head(MEMBERS_IN_DETAIL)

    explanations = artifacts.load_alert_table(root, "alert_explanations")
    explanations = explanations[
        explanations["address"].isin(set(shown["address"]))
    ]
    network_rows = artifacts.load_alert_table(root, "alert_network")
    network_rows = network_rows[network_rows["alert_id"] == raw_id]

    timeline = artifacts.load_alert_table(root, "alert_timeline")
    timeline = timeline[timeline["alert_id"] == raw_id].sort_values("timestep")
    relationships = artifacts.load_alert_table(root, "alert_relationships")
    relationships = relationships[
        relationships["alert_id"] == raw_id
    ].head(RELATIONSHIPS_IN_DETAIL)

    return {
        "alert_id": raw_id,
        "run_fingerprint": current,
        "summary": _summary(alert),
        "meaning": contract.ALERT_MEANING,
        "score_scope": contract.SCORE_SCOPE,
        "risk": {
            "score": _clean(alert["risk_score"]),
            "severity": alert["severity"],
            "ranking_aggregation": alert["ranking_aggregation"],
            "aggregations": {
                name: _clean(alert[name]) for name in contract.AGGREGATIONS
            },
            "top_member_risk": _clean(alert["top_member_risk"]),
        },
        "why_flagged": _why(shown, explanations),
        "evidence": _evidence(alert, shown),
        "relationships": _relationships(relationships),
        "timeline": _timeline(timeline),
        "network_context": _network(alert, network_rows, root),
        "correlation": _correlation(network_rows, root),
        "members": {
            "shown": int(len(shown)),
            "total_scored": int(len(members)),
            "withheld": int(max(0, len(members) - len(shown))),
            "rows": [
                {
                    "address": r.address,
                    "risk_score": _clean(r.risk_score),
                    "severity": r.severity,
                    "observed_at_timestep": int(r.observed_at_t),
                    "first_timestep": int(r.first_t),
                }
                for r in shown.itertuples(index=False)
            ],
        },
        "provenance": _provenance_block(sidecar),
    }


def _why(shown, explanations) -> dict:
    """The model's own explainability output, per surfaced member.

    Contributions are SHAP values in log-odds. The response says so rather
    than letting a reader take +0.27 for 27 percentage points of risk.
    """
    rows = []
    for address in shown["address"]:
        subset = explanations[explanations["address"] == address]
        subset = subset.sort_values("rank").head(EXPLANATIONS_PER_MEMBER)
        rows.append({
            "address": address,
            "base_value": _clean(
                subset["base_value"].iloc[0] if len(subset) else None
            ),
            "contributions": [
                {
                    "feature": r.feature,
                    "feature_group": r.feature_group,
                    "contribution": _clean(r.contribution),
                    "feature_value": _clean(r.feature_value),
                    "signal_category": r.signal_category,
                    "value_category": r.value_category,
                }
                for r in subset.itertuples(index=False)
            ],
        })
    return {
        "units": "log-odds (SHAP contribution to the model's raw margin)",
        "meaning": contract.MODEL_SIGNAL_MEANING,
        "insufficient_evidence_meaning": contract.INSUFFICIENT_EVIDENCE_MEANING,
        "categories": list(contract.CATEGORIES),
        "per_member": rows,
    }


def _evidence(alert, shown) -> dict:
    """M0-M3 evidence, grouped, with each group's category named."""
    from obsidianchain.alerts import contract as c

    groups = {}
    for group, category in c.GROUP_CATEGORY.items():
        columns = [col for col in shown.columns if _group_of(col) == group]
        if not columns:
            # The artifact predates this group, or was built without it.
            # Emitting an empty group would show a reader an evidence tab
            # with nothing in it, which reads as "measured, found nothing"
            # rather than "this run did not compute it".
            continue
        groups[group] = {
            "category": category,
            "features": columns,
            "members": [
                {
                    "address": r.address,
                    "values": {
                        col: _clean(getattr(r, col)) for col in columns
                    },
                    "unavailable": [
                        col for col in columns if _clean(getattr(r, col)) is None
                    ],
                }
                for r in shown.itertuples(index=False)
            ],
        }
    return {
        "aggregate": {
            "category": c.BLOCKCHAIN_CONTEXT,
            "transactions": _clean(alert["n_transactions"]),
            "btc_sent_total": _clean(alert["btc_sent_total"]),
            "btc_received_total": _clean(alert["btc_received_total"]),
            "unique_counterparties": _clean(alert["unique_counterparties"]),
            "peel_chain_members": int(alert["peel_chain_members"]),
        },
        "groups": groups,
        "groups_available": sorted(groups),
        "structural_meanings": {
            "M2": c.PEEL_STRUCTURE_MEANING,
            "M4": c.MIXING_PATTERN_MEANING,
        },
        "insufficient_evidence_meaning": c.INSUFFICIENT_EVIDENCE_MEANING,
    }


_GROUP_CACHE: dict[str, str | None] = {}


def _group_of(column: str) -> str | None:
    """Feature -> M0/M1/M2/M3, from the contract's own column lists.

    Resolved through a local table rather than by importing the feature
    package, which ``api/boundary.py`` forbids the API to reach.
    """
    if not _GROUP_CACHE:
        _GROUP_CACHE.update(_load_group_table())
    return _GROUP_CACHE.get(column)


def _load_group_table() -> dict:
    from obsidianchain.alerts import feature_groups

    return dict(feature_groups.GROUP_OF)


def _relationships(frame) -> dict:
    """Only relationships the data supports, each with its own caveat."""
    return {
        "supported": {k: v for k, v in contract.RELATIONSHIPS.items()},
        "count": int(len(frame)),
        "edges": [
            {
                "address_a": r.address_a,
                "address_b": r.address_b,
                "relationship": r.relationship,
                "timestep": int(r.timestep),
                "category": contract.BLOCKCHAIN_CONTEXT,
            }
            for r in frame.itertuples(index=False)
        ],
        "note": (
            "Co-spend membership is a HEURISTIC about spending, not proof of "
            "shared ownership. No relationship here is derived from network "
            "observations: the network layer emits cannot-link only and can "
            "never establish that two addresses are the same party."
        ),
    }


def _timeline(frame) -> dict:
    return {
        "category": contract.BLOCKCHAIN_CONTEXT,
        "unit": "Elliptic timestep (about two weeks each)",
        "points": [
            {
                "timestep": int(r.timestep),
                "transactions": int(r.n_transactions),
                "active_addresses": int(r.n_active_addresses),
                "btc_sent": _clean(r.btc_sent),
                "btc_received": _clean(r.btc_received),
            }
            for r in frame.itertuples(index=False)
        ],
    }


def _correlation(frame, root) -> dict:
    """IP <-> transaction <-> wallet, the three-node relation the PS asks for.

    Every row is an observed announcement: an observer heard transaction T
    from peer P at time t, and address A took part in T. The peer is a RELAY
    VANTAGE POINT - 84.3% of transactions in this dataset were announced by
    more than one peer - so ``announcing_peers`` travels with every row and
    the meaning text says plainly that a peer is not a sender.
    """
    from obsidianchain import geoip
    from obsidianchain.alerts import contract as c

    if frame is None or len(frame) == 0:
        return {
            "category": c.NETWORK_CONTEXT,
            "status": c.INSUFFICIENT_EVIDENCE,
            "available": False,
            "transactions": [],
            "summary": None,
            "meaning": c.ANNOUNCING_PEER_MEANING,
            "insufficient_evidence_meaning": c.INSUFFICIENT_EVIDENCE_MEANING,
        }

    transactions = []
    for txid, block in frame.groupby("txid", sort=False):
        first = block.iloc[0]
        peers = block[[
            "peer_ip", "peer_port", "peer_asn", "first_seen_ms",
            "last_seen_ms", "observers",
        ]].drop_duplicates(subset=["peer_ip"])
        transactions.append({
            "txid": str(txid),
            "announcing_peers_total": int(first["announcing_peers"]),
            "peers_shown": int(len(peers)),
            "first_seen_ms": _clean(block["first_seen_ms"].min()),
            "addresses": [
                {"address": r.address, "role": r.address_role}
                for r in block[["address", "address_role"]]
                .drop_duplicates().itertuples(index=False)
            ],
            "peers": [
                {
                    "ip": r.peer_ip,
                    "port": _clean(r.peer_port),
                    "asn": _clean(r.peer_asn),
                    "first_seen_ms": _clean(r.first_seen_ms),
                    "last_seen_ms": _clean(r.last_seen_ms),
                    "observers": int(r.observers),
                }
                for r in peers.itertuples(index=False)
            ],
        })

    return {
        "category": c.NETWORK_CONTEXT,
        "status": c.NETWORK_CONTEXT,
        "available": True,
        "transactions": transactions,
        "summary": {
            "transactions": int(frame["txid"].nunique()),
            "announcing_peers": int(frame["peer_ip"].nunique()),
            "asns": int(frame["peer_asn"].nunique()),
            "observers": int(frame["observers"].max()),
            "geo": geoip.summarise(
                frame["peer_ip"].dropna().tolist(),
                frame["peer_asn"].dropna().tolist(),
                data_root=root,
            ),
        },
        "meaning": c.ANNOUNCING_PEER_MEANING,
        "synthetic_warning": c.SYNTHETIC_NETWORK_WARNING,
        "insufficient_evidence_meaning": c.INSUFFICIENT_EVIDENCE_MEANING,
    }


def _network(alert, network_rows=None, root=None) -> dict:
    """Investigative context. Never an ownership or identity claim."""
    members_with = int(alert["net_evidence_members"])
    return {
        "category": contract.NETWORK_CONTEXT,
        "members_with_observations": members_with,
        "members_reaching_production_minimum": int(alert["net_minimum_members"]),
        "available": members_with > 0,
        "status": (
            contract.NETWORK_CONTEXT if members_with > 0
            else contract.INSUFFICIENT_EVIDENCE
        ),
        "meaning": contract.NETWORK_CONTEXT_MEANING,
        "synthetic_warning": contract.SYNTHETIC_NETWORK_WARNING,
        "insufficient_evidence_meaning": (
            contract.INSUFFICIENT_EVIDENCE_MEANING if members_with == 0 else None
        ),
    }


def get_related_alerts(raw_id: str, root=None, *, limit: int = 20) -> dict:
    """Identify real relationships between alerts based only on supported artifact data (T2).
    
    Why related:
    - shared on-chain transaction (txid)
    - shared peer observation (with gossip relay caveat)
    Never inferred from similar risk scores.
    """
    requested_run, cluster_id = contract.parse_alert_id(raw_id)
    frame, sidecar = artifacts.load_alerts(root)
    current = current_run_fingerprint(sidecar)
    if requested_run != current:
        raise contract.AlertIdStaleError(
            f"Alert id {raw_id!r} is from run {requested_run}, current run is {current}"
        )
    row = frame[frame["alert_id"] == raw_id]
    if row.empty:
        raise contract.AlertNotFoundError(f"alert {raw_id!r} not found in run {current}")
    
    net_table = artifacts.load_alert_table(root, "alert_network")
    this_net = net_table[net_table["alert_id"] == raw_id]
    
    related = []
    seen_alerts = set()
    
    # 1. Check shared transactions
    if not this_net.empty and "txid" in this_net.columns:
        txids = set(this_net["txid"].dropna().unique())
        if txids:
            tx_matches = net_table[
                (net_table["txid"].isin(txids)) & (net_table["alert_id"] != raw_id)
            ]
            for other_id, grp in tx_matches.groupby("alert_id"):
                if other_id in seen_alerts:
                    continue
                seen_alerts.add(other_id)
                match_row = frame[frame["alert_id"] == other_id]
                if match_row.empty:
                    continue
                m = match_row.iloc[0]
                shared_tx_list = [int(x) for x in grp["txid"].unique()[:3]]
                related.append({
                    "alert_id": other_id,
                    "cluster_id": int(m["cluster_id"]),
                    "rank": int(m["rank"]),
                    "severity": str(m["severity"]),
                    "risk_score": _clean(m["risk_score"]),
                    "top_signals": _summary(m.to_dict())["top_signals"],
                    "relationship_type": "SHARED_TRANSACTION",
                    "connecting_identifier": f"txids: {', '.join(str(t) for t in shared_tx_list)}",
                    "detail": f"Cluster members participated in {len(grp['txid'].unique())} shared on-chain transaction(s).",
                    "limitation": "Shared transactions demonstrate on-chain interaction or counterparty flow; they do not prove identical ownership.",
                })
                if len(related) >= limit:
                    break
                    
    # 2. Check shared peers (if still room)
    if len(related) < limit and not this_net.empty and "peer_ip" in this_net.columns:
        peers = set(this_net["peer_ip"].dropna().unique())
        if peers:
            peer_matches = net_table[
                (net_table["peer_ip"].isin(peers)) & (net_table["alert_id"] != raw_id)
            ]
            for other_id, grp in peer_matches.groupby("alert_id"):
                if other_id in seen_alerts:
                    continue
                seen_alerts.add(other_id)
                match_row = frame[frame["alert_id"] == other_id]
                if match_row.empty:
                    continue
                m = match_row.iloc[0]
                shared_peers_list = [str(x) for x in grp["peer_ip"].unique()[:2]]
                related.append({
                    "alert_id": other_id,
                    "cluster_id": int(m["cluster_id"]),
                    "rank": int(m["rank"]),
                    "severity": str(m["severity"]),
                    "risk_score": _clean(m["risk_score"]),
                    "top_signals": _summary(m.to_dict())["top_signals"],
                    "relationship_type": "OBSERVED_SHARED_PEER",
                    "connecting_identifier": f"peers: {', '.join(shared_peers_list)}",
                    "detail": f"Both alerts have transactions observed via announcing peer(s) {', '.join(shared_peers_list)}.",
                    "limitation": "An announcing peer is a relay vantage point in the Bitcoin p2p network, NOT an originator or proof of shared ownership.",
                })
                if len(related) >= limit:
                    break
                    
    return {
        "alert_id": raw_id,
        "run_fingerprint": current,
        "count": len(related),
        "related_alerts": related,
        "meaning": "Related alerts identified strictly through shared on-chain transactions or observed network relay peers. Similarity in risk score is never used to infer a relationship.",
    }


def get_transaction_drilldown(txid: int, root=None) -> dict:
    """One transaction: observed inputs and outputs, alert links, relay peers, mixing scan.

    The inputs and outputs come from the chain index, so every Elliptic++
    transaction resolves, not only the ones that touch an alert. Announcing
    peers exist only for alert transactions (alert_network) and are a relay
    vantage point, never a sender.
    """
    from obsidianchain.api import investigation

    try:
        tx = investigation.get_transaction(txid, root)
    except (artifacts.ArtifactMissingError, investigation.TransactionNotFoundError):
        tx = None
    try:
        net_table = artifacts.load_alert_table(root, "alert_network")
    except artifacts.ArtifactMissingError:
        if tx is None:
            raise investigation.TransactionNotFoundError(
                f"transaction {txid} does not appear in the chain index") from None
        net_table = None
    rows = (net_table[net_table["txid"] == txid] if net_table is not None
            else pd.DataFrame(columns=["alert_id", "address", "address_role", "peer_ip", "peer_port",
                                       "peer_asn", "observers", "announcing_peers"]))
    if tx is None and rows.empty:
        raise investigation.TransactionNotFoundError(
            f"transaction {txid} is in neither the chain index nor the alert network correlation")

    alert_of: dict[str, str] = {}
    clusters = set()
    for r in rows.itertuples():
        alert_id = str(r.alert_id)
        alert_of.setdefault(str(r.address), alert_id)
        if ":" in alert_id:
            clusters.add(alert_id.split(":")[1])

    if tx is not None:
        def side(entries):
            out = []
            for e in entries:
                model = e.get("model") or {}
                alert_id = alert_of.get(e["address"]) or model.get("alert_id")
                out.append({"address": e["address"], "alert_id": alert_id,
                            "model": e.get("model"), "cluster": e.get("cluster"),
                            "watchlist": e.get("watchlist", [])})
                if alert_id and ":" in alert_id:
                    clusters.add(alert_id.split(":")[1])
            return out
        inputs, outputs = side(tx["inputs"]), side(tx["outputs"])
    else:
        inputs, outputs, seen = [], [], set()
        for r in rows.itertuples():
            addr = str(r.address)
            if addr in seen:
                continue
            seen.add(addr)
            (inputs if str(r.address_role) == "input" else outputs).append(
                {"address": addr, "alert_id": str(r.alert_id)})

    peers, seen_peers = [], set()
    for r in rows.itertuples():
        peer_key = (r.peer_ip, r.peer_port)
        if peer_key in seen_peers or r.peer_ip is None:
            continue
        seen_peers.add(peer_key)
        peers.append({
            "ip": str(r.peer_ip),
            "port": int(r.peer_port) if _clean(r.peer_port) is not None else None,
            "asn": int(r.peer_asn) if _clean(r.peer_asn) is not None else None,
            "observers": int(r.observers) if _clean(r.observers) is not None else 1,
            "announcing_peers": int(r.announcing_peers) if _clean(r.announcing_peers) is not None else 1,
        })

    mixing_info = None
    from obsidianchain.api import patterns
    tx_mixing_df, _meta = patterns.load_tx_mixing(root)
    if tx_mixing_df is not None:
        m_row = tx_mixing_df[tx_mixing_df["txId"] == txid]
        if not m_row.empty:
            m = m_row.iloc[0]
            mixing_info = {
                "available": True,
                "mixing_class": str(m["mixing_class"]),
                "mixing_score": _clean(m["mixing_score"]),
                "signals": {k: _clean(m[k]) for k in (
                    "output_uniformity", "participant_symmetry", "cardinality_signal",
                    "input_heterogeneity", "suppressor") if k in m.index},
                "n_inputs": int(m["n_inputs"]),
                "n_outputs": int(m["n_outputs"]),
                "meaning": "Structural heuristic over input/output shape. A pattern, not an attribution.",
            }
    if mixing_info is None:
        mixing_info = {
            "available": False,
            "meaning": ("Transaction mixing classification scan is not loaded for this deployment."
                        if tx_mixing_df is None else
                        "The mixing scan has no row for this transaction."),
        }

    out = {
        "txid": txid,
        "input_count": len(inputs),
        "output_count": len(outputs),
        "inputs": inputs,
        "outputs": outputs,
        "associated_clusters": sorted(clusters),
        "announcing_peers": peers,
        "mixing": mixing_info,
        "limitation": "A peer announcing a transaction is a network relay observer. It does not identify the sender, wallet owner, or person who created the transaction.",
    }
    if tx is not None:
        out.update({k: tx[k] for k in ("timestep", "fee_btc", "in_btc", "out_btc", "n_inputs", "n_outputs")
                    if k in tx})
        out["provenance"] = tx["provenance"]
    return out


MAX_GRAPH_SEEDS = 40
MAX_GRAPH_PEERS = 60


def get_alert_graph(raw_id: str, root=None, hops: int = 2, *, direction: str = "both",
                    max_nodes: int = 400) -> dict:
    """The observed money flow around an alert's members, plus its cluster and relay peers.

    Seeds are the alert's members (the highest-scored ``MAX_GRAPH_SEEDS`` for
    very large clusters; the response says how many were left out). The flow
    is traced over the chain index for ``hops`` address-transaction-address
    steps. Every edge's two ends are in ``nodes``: nothing dangles.

    Layers and what they are:
      addr  --MEMBER_OF-->    cluster   entity resolution (heuristic)
      addr  --SPENDS-->       tx        observed on-chain
      tx    --PAYS-->         addr      observed on-chain
      tx    --ANNOUNCED_BY--> ip        relay vantage point, never a sender
    """
    from obsidianchain.api import investigation

    requested_run, cluster_id = contract.parse_alert_id(raw_id)
    frame, sidecar = artifacts.load_alerts(root)
    current = current_run_fingerprint(sidecar)
    if requested_run != current:
        raise contract.AlertIdStaleError(
            f"Alert id {raw_id!r} is from run {requested_run}, current run is {current}"
        )
    row = frame[frame["alert_id"] == raw_id]
    if row.empty:
        raise contract.AlertNotFoundError(f"alert {raw_id!r} not found in run {current}")
    summary = _summary(row.iloc[0].to_dict())

    members = artifacts.load_alert_table(root, "alert_members")
    mine = members[members["alert_id"] == raw_id].sort_values("risk_score", ascending=False)
    seeds = [str(a) for a in mine["address"].head(MAX_GRAPH_SEEDS)]
    result = investigation.trace(seeds, root, direction=direction, hops=hops, max_nodes=max_nodes)
    graph = result["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = {e["id"]: e for e in graph["edges"]}

    cluster_key = f"cluster:{summary['cluster_id']}"
    nodes[cluster_key] = {"id": cluster_key, "kind": "cluster", "label": f"Cluster {summary['cluster_id']}",
                          "data": {"cluster_id": summary["cluster_id"], "alert_id": raw_id,
                                   "severity": summary["severity"], "risk_score": summary["risk_score"],
                                   "members_total": summary["members_total"]}}
    member_set = set(str(a) for a in mine["address"])
    for key, n in list(nodes.items()):
        if n["kind"] == "address" and n["data"]["address"] in member_set:
            n["data"]["alert_member"] = True
            eid = f"MEMBER_OF:{key}->{cluster_key}"
            edges[eid] = {"id": eid, "source": key, "target": cluster_key, "kind": "MEMBER_OF",
                          "data": {"basis": "entity resolution heuristic"}}

    net = artifacts.load_alert_table(root, "alert_network")
    in_graph = {int(n["data"]["txid"]) for n in nodes.values() if n["kind"] == "transaction"}
    peers_seen = 0
    net = net[net["txid"].isin(in_graph) & net["peer_ip"].notna()]
    for r in net.drop_duplicates(["txid", "peer_ip"]).itertuples():
        ip_key = f"ip:{r.peer_ip}"
        if ip_key not in nodes:
            if peers_seen >= MAX_GRAPH_PEERS:
                continue
            peers_seen += 1
            nodes[ip_key] = {"id": ip_key, "kind": "ip", "label": str(r.peer_ip),
                             "data": {"ip": str(r.peer_ip),
                                      "asn": int(r.peer_asn) if _clean(r.peer_asn) is not None else None,
                                      "synthetic_network": _clean(getattr(r, "synthetic_network", None))}}
        eid = f"ANNOUNCED_BY:tx:{int(r.txid)}->{ip_key}"
        edges[eid] = {"id": eid, "source": f"tx:{int(r.txid)}", "target": ip_key, "kind": "ANNOUNCED_BY",
                      "data": {"port": int(r.peer_port) if _clean(r.peer_port) is not None else None,
                               "first_seen_ms": _clean(r.first_seen_ms)}}

    assert all(e["source"] in nodes and e["target"] in nodes for e in edges.values())
    return {
        "alert_id": raw_id,
        "summary": summary,
        "seeds": {"used": len(seeds), "members_total": int(len(mine)),
                  "omitted": max(0, int(len(mine)) - len(seeds)),
                  "rule": f"highest-scored {MAX_GRAPH_SEEDS} members"},
        "direction": result["direction"], "hops": result["hops"], "max_nodes": result["max_nodes"],
        "graph": {"node_count": len(nodes), "edge_count": len(edges),
                  "nodes": list(nodes.values()), "edges": list(edges.values())},
        "truncated": result["truncated"],
        "hub_transactions_skipped": result["hub_transactions_skipped"],
        "hub_threshold": result["hub_threshold"],
        "peers_omitted": peers_seen >= MAX_GRAPH_PEERS,
        "layers": {
            "MEMBER_OF": "Entity resolution heuristic (multi-input and change). Not proof of one owner.",
            "SPENDS": "Observed on-chain: the address funded the transaction.",
            "PAYS": "Observed on-chain: the transaction paid the address.",
            "ANNOUNCED_BY": ("The peer relayed the transaction to an observer. A relay vantage point, "
                             "never the sender. Peer data in this deployment is a synthetic network overlay."),
        },
        "meaning": result["meaning"],
        "provenance": {**result["provenance"], "alert_run_fingerprint": current},
    }
