"""Investigation read path: addresses, money-flow tracing, transactions, search.

Presentation code over precomputed artifacts only (see api/boundary.py):

* ``chain_edges`` / ``chain_transactions`` / ``watchlist_seeds``, written by
  ``build-chain-index``: OBSERVED on-chain structure and per-transaction
  value totals, plus EXTERNAL attributions (OFAC SDN, analyst watchlists).
* The Phase 7 alert tables: MODEL output of the Phase 6 reference model.
* ``address_clusters`` / ``clusters``: co-spend CLUSTER inference.

Every response keeps the three kinds of claim apart. An address's model
score always travels with the alert run and model it came from; a watchlist
hit always names its source; chain facts carry the chain-index fingerprint.
Elliptic++ class labels are not in any of these artifacts and are never
served.

Indexes are built once per artifact version (path, size, mtime) and cached,
so a lookup is a binary search on a sorted index rather than a scan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from obsidianchain.alerts import contract
from obsidianchain.api import artifacts

#: Transactions with more participants than this are not expanded while
#: tracing: an exchange sweep or batch payout links thousands of strangers.
#: Skipped hubs are counted and named in the response, never hidden.
DEFAULT_MAX_TX_DEGREE = 200
MAX_HOPS = 4
MAX_NODES_CEILING = 1500

MODEL_SCOPE = ("Risk scores here come from the Phase 7 reference alert run (Phase 6 LightGBM over "
               "Elliptic++). They are model associations, not findings, and exist only for addresses "
               "that run scored.")
CHAIN_SCOPE = ("Observed Elliptic++ structure: which addresses spent into and were paid by each "
               "transaction, and per-transaction totals. Per-address amounts are not in the source.")
WATCHLIST_SCOPE = "External attribution by the named source. Not a finding of this system."


class AddressNotFoundError(LookupError):
    pass


class TransactionNotFoundError(LookupError):
    pass


class TraceRequestError(ValueError):
    pass


# ---- cached indexes ---------------------------------------------------------


def _stamp(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return str(path), st.st_size, st.st_mtime_ns


@dataclass
class ChainIndex:
    by_address: pd.DataFrame      # index: address (sorted, non-unique)
    by_txid: pd.DataFrame         # index: txid (sorted, non-unique)
    tx: pd.DataFrame              # index: txid (unique)
    addresses_sorted: pd.Index    # unique addresses, sorted, for prefix search
    fingerprint: str


@lru_cache(maxsize=2)
def _chain(edges_stamp, tx_stamp, fingerprint) -> ChainIndex:
    edges = pd.read_parquet(edges_stamp[0], columns=["address", "txid", "role", "timestep"])
    tx = pd.read_parquet(tx_stamp[0], columns=["txid", "timestep", "fee_btc", "in_btc", "out_btc",
                                               "n_inputs", "n_outputs"])
    # Plain object/numpy columns, not Arrow-backed ones: row lookups below are
    # positional slices of a sorted frame, and Arrow ``take`` made each one
    # cost milliseconds.
    edges = edges.astype({"address": object, "role": object})
    edges["timestep"] = edges["timestep"].astype("float64")
    by_address = edges.set_index("address").sort_index(kind="stable")
    by_txid = edges.set_index("txid").sort_index(kind="stable")
    return ChainIndex(by_address=by_address, by_txid=by_txid, tx=tx.set_index("txid"),
                      addresses_sorted=pd.Index(by_address.index.unique()), fingerprint=fingerprint)


def chain(root=None) -> ChainIndex:
    e_path, e_meta = artifacts.chain_artifact(root, artifacts.CHAIN_EDGES)
    t_path, _ = artifacts.chain_artifact(root, artifacts.CHAIN_TRANSACTIONS)
    return _chain(_stamp(e_path), _stamp(t_path), str(e_meta.get("run_fingerprint")))


@lru_cache(maxsize=2)
def _watchlist(stamp) -> dict[str, list[dict]]:
    frame = pd.read_parquet(stamp[0], columns=["address", "source", "label"])
    out: dict[str, list[dict]] = {}
    for r in frame.itertuples(index=False):
        out.setdefault(str(r.address), []).append({"source": str(r.source), "label": str(r.label or "")})
    return out


def watchlist(root=None) -> dict[str, list[dict]]:
    path, _ = artifacts.chain_artifact(root, artifacts.WATCHLIST_SEEDS)
    return _watchlist(_stamp(path))


@lru_cache(maxsize=2)
def _members(stamp, run) -> pd.DataFrame:
    frame = pd.read_parquet(stamp[0], columns=["alert_id", "address", "risk_score", "severity",
                                               "observed_at_t", "first_t"])
    frame = frame.astype({"address": object})
    return frame.set_index("address").sort_index(kind="stable")


def members(root=None) -> tuple[pd.DataFrame, str]:
    _, sidecar = artifacts.load_alerts(root, columns=["alert_id"])
    run = contract.make_alert_id(str(sidecar.get("run_fingerprint")), 0).split(":")[0]
    path = artifacts.data_root(root) / artifacts.ALERT_TABLES["alert_members"]
    artifacts.load_alert_table(root, "alert_members", columns=["alert_id"])  # provenance + same-run gate
    return _members(_stamp(path), run), run


@lru_cache(maxsize=2)
def _clusters(stamp_a, stamp_c) -> tuple[pd.Series, pd.Series]:
    ac = pd.read_parquet(stamp_a[0], columns=["address", "cluster_id"]).set_index("address")["cluster_id"]
    cs = pd.read_parquet(stamp_c[0], columns=["cluster_id", "size"]).set_index("cluster_id")["size"]
    return ac, cs


def clusters(root=None) -> tuple[pd.Series, pd.Series]:
    a = artifacts.address_clusters_path(root)
    c = artifacts.clusters_path(root)
    from obsidianchain.api import provenance_gate
    provenance_gate.require_production(a)
    provenance_gate.require_production(c)
    return _clusters(_stamp(a), _stamp(c))


#: An annotation layer that is absent or refused is left out of a response
#: (and the response says what is missing), never allowed to fail the whole
#: read. Anything else - a bug - still raises.
def _optional_errors() -> tuple[type[BaseException], ...]:
    from obsidianchain.api import provenance_gate
    return (artifacts.ArtifactMissingError, artifacts.ArtifactInvalidError,
            provenance_gate.ProvenanceRefusedError)


_OPTIONAL = _optional_errors()


# ---- annotation helpers ---------------------------------------------------------


def _clean(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


def _rows(frame: pd.DataFrame, key) -> pd.DataFrame:
    """Rows whose (sorted, non-unique) index equals ``key``: a binary search and a slice."""
    index = frame.index
    lo = index.searchsorted(key, side="left")
    hi = index.searchsorted(key, side="right")
    return frame.iloc[lo:hi]


def _annotate_addresses(addresses: list[str], root=None) -> dict[str, dict]:
    """Model, cluster and watchlist annotations for a set of addresses."""
    try:
        mem, _run = members(root)
    except _OPTIONAL:
        mem = None
    try:
        ac, cs = clusters(root)
    except _OPTIONAL:
        ac, cs = None, None
    wl = watchlist(root)
    out = {}
    wanted = pd.Index(addresses)
    m_rows = mem[mem.index.isin(wanted)] if mem is not None else None
    for a in addresses:
        info: dict[str, Any] = {"model": None, "cluster": None, "watchlist": wl.get(a, [])}
        if m_rows is not None and a in m_rows.index:
            r = m_rows.loc[[a]].sort_values("risk_score", ascending=False).iloc[0]
            info["model"] = {"risk_score": _clean(r["risk_score"]), "severity": str(r["severity"]),
                             "alert_id": str(r["alert_id"]), "alerts": int((m_rows.index == a).sum()),
                             "scope": "PHASE7_REFERENCE_RUN"}
        if ac is not None and a in ac.index:
            cid = int(ac.loc[a])
            info["cluster"] = {"cluster_id": cid, "size": int(cs.get(cid, 0)) if cs is not None else None}
        out[a] = info
    return out


def _tx_row(idx: ChainIndex, txid: int) -> dict:
    try:
        r = idx.tx.loc[txid]
    except KeyError:
        return {"txid": int(txid)}
    ts = _clean(r["timestep"])
    return {"txid": int(txid), "timestep": int(ts) if ts is not None else None, "fee_btc": _clean(r["fee_btc"]),
            "in_btc": _clean(r["in_btc"]), "out_btc": _clean(r["out_btc"]),
            "n_inputs": int(r["n_inputs"]), "n_outputs": int(r["n_outputs"])}


def _provenance(idx: ChainIndex) -> dict:
    return {"chain_index_fingerprint": idx.fingerprint[:16], "chain_scope": CHAIN_SCOPE,
            "model_scope": MODEL_SCOPE, "watchlist_scope": WATCHLIST_SCOPE,
            "labels_served": False}


# ---- address ------------------------------------------------------------------------


def get_address(address: str, root=None, *, counterparty_limit: int = 25,
                transaction_limit: int = 200) -> dict:
    idx = chain(root)
    rows = _rows(idx.by_address, address)
    if rows.empty:
        raise AddressNotFoundError(f"address {address!r} does not appear in the Elliptic++ chain index")
    rows = rows.reset_index()
    as_in = rows[rows.role == "input"]
    as_out = rows[rows.role == "output"]
    steps = rows["timestep"].dropna().astype(int)

    timeline = (rows.groupby(["timestep", "role"]).size().unstack(fill_value=0)
                .reindex(columns=["input", "output"], fill_value=0))
    timeline_points = [{"timestep": int(t), "as_input": int(v["input"]), "as_output": int(v["output"])}
                       for t, v in timeline.iterrows()]

    hubs_skipped = 0

    def counterparts(txids, want_role):
        nonlocal hubs_skipped
        counts: dict[str, int] = {}
        for t in txids:
            tr = _rows(idx.by_txid, int(t))
            if len(tr) > DEFAULT_MAX_TX_DEGREE:
                hubs_skipped += 1
                continue
            for a in tr[tr.role == want_role]["address"]:
                if a != address:
                    counts[a] = counts.get(a, 0) + 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:counterparty_limit]
        return top, len(counts)

    paid_to, n_paid_to = counterparts(as_in.txid.unique(), "output")
    paid_by, n_paid_by = counterparts(as_out.txid.unique(), "input")
    names = [a for a, _ in paid_to] + [a for a, _ in paid_by] + [address]
    notes = _annotate_addresses(list(dict.fromkeys(names)), root)

    tx_ids = rows.sort_values(["timestep", "txid"], ascending=False)["txid"].drop_duplicates().head(transaction_limit)
    txs = []
    for t in tx_ids:
        role = set(rows.loc[rows.txid == t, "role"])
        txs.append({**_tx_row(idx, int(t)), "role": "both" if len(role) == 2 else next(iter(role))})

    explanations = []
    try:
        exp = artifacts.load_alert_table(root, "alert_explanations",
                                         columns=["alert_id", "address", "feature", "contribution",
                                                  "feature_value", "feature_group", "signal_category"])
        mine = exp[exp.address == address].sort_values("contribution", key=abs, ascending=False)
        explanations = [{k: _clean(v) for k, v in r.items()} for r in mine.head(12).to_dict("records")]
    except _OPTIONAL:
        explanations = []

    alerts = []
    try:
        mem, _run = members(root)
        for r in _rows(mem, address).reset_index().to_dict("records"):
            alerts.append({"alert_id": str(r["alert_id"]), "risk_score": _clean(r["risk_score"]),
                           "severity": str(r["severity"]), "observed_at_timestep": _clean(r["observed_at_t"])})
    except _OPTIONAL:
        pass

    network = []
    try:
        net = artifacts.load_alert_table(root, "alert_network",
                                         columns=["txid", "peer_ip", "peer_asn", "first_seen_ms", "address",
                                                  "address_role"])
        mine = net[net.address == address]
        network = [{"txid": int(t), "peers": int(g.peer_ip.nunique()), "asns": int(g.peer_asn.nunique()),
                    "first_seen_ms": _clean(g.first_seen_ms.min())} for t, g in mine.groupby("txid")][:50]
    except _OPTIONAL:
        network = []

    me = notes[address]
    return {
        "address": address,
        "observed": {
            "transactions": int(rows.txid.nunique()),
            "as_input": int(as_in.txid.nunique()),
            "as_output": int(as_out.txid.nunique()),
            "first_timestep": int(steps.min()) if len(steps) else None,
            "last_timestep": int(steps.max()) if len(steps) else None,
            "active_timesteps": int(steps.nunique()),
            "timeline": timeline_points,
            "transactions_list": txs,
            "transactions_shown": len(txs),
        },
        "counterparties": {
            "paid_to": [{"address": a, "shared_transactions": n, **notes.get(a, {})} for a, n in paid_to],
            "paid_to_total": n_paid_to,
            "paid_by": [{"address": a, "shared_transactions": n, **notes.get(a, {})} for a, n in paid_by],
            "paid_by_total": n_paid_by,
            "hub_transactions_skipped": hubs_skipped,
            "hub_threshold": DEFAULT_MAX_TX_DEGREE,
        },
        "cluster": me["cluster"],
        "watchlist": me["watchlist"],
        "model": {"alerts": alerts, "explanations": explanations,
                  "scope": "PHASE7_REFERENCE_RUN", "meaning": MODEL_SCOPE},
        "network": {"observations": network,
                    "meaning": "Synthetic announcement observations; an announcing peer is not a sender."},
        "provenance": _provenance(idx),
    }


# ---- tracing --------------------------------------------------------------------------


def trace(seeds: list[str], root=None, *, direction: str = "both", hops: int = 1,
          max_nodes: int = 300, max_tx_degree: int = DEFAULT_MAX_TX_DEGREE,
          min_timestep: int | None = None, max_timestep: int | None = None,
          seed_txids: list[int] | None = None) -> dict:
    """Directed money flow around ``seeds``.

    One hop is address -> transaction -> address. Downstream follows value
    out of an address (it spends into a transaction, which pays others);
    upstream follows it back (the address was paid by a transaction, which
    others funded). Edges are directed as value moves: SPENDS address->tx,
    PAYS tx->address.
    """
    if direction not in ("upstream", "downstream", "both"):
        raise TraceRequestError("direction must be upstream, downstream or both")
    hops = max(1, min(int(hops), MAX_HOPS))
    max_nodes = max(10, min(int(max_nodes), MAX_NODES_CEILING))
    idx = chain(root)
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    hubs: dict[int, int] = {}
    truncated = False

    def in_time(ts) -> bool:
        if ts is None or pd.isna(ts):
            return True
        return (min_timestep is None or ts >= min_timestep) and (max_timestep is None or ts <= max_timestep)

    def add_addr(a, hop, seed=False):
        key = f"addr:{a}"
        if key not in nodes:
            nodes[key] = {"id": key, "kind": "address", "label": a[:8] + "…" + a[-4:] if len(a) > 14 else a,
                          "data": {"address": a, "hop": hop, "seed": seed}}
        elif seed:
            nodes[key]["data"]["seed"] = True
        return key

    def add_tx(t, hop):
        key = f"tx:{t}"
        if key not in nodes:
            nodes[key] = {"id": key, "kind": "transaction", "label": str(t),
                          "data": {**_tx_row(idx, int(t)), "hop": hop}}
        return key

    def add_edge(src, dst, kind, ts):
        eid = f"{kind}:{src}->{dst}"
        if eid not in edges:
            edges[eid] = {"id": eid, "source": src, "target": dst, "kind": kind, "data": {"timestep": None if _clean(ts) is None else int(ts)}}

    frontier = []
    for a in seeds:
        if not _rows(idx.by_address, a).empty:
            add_addr(a, 0, seed=True)
            frontier.append(a)
    for t in seed_txids or []:
        tr = _rows(idx.by_txid, int(t))
        if tr.empty:
            continue
        tk = add_tx(int(t), 0)
        nodes[tk]["data"]["seed"] = True
        for r in tr.reset_index().itertuples():
            ak = add_addr(r.address, 0)
            if r.role == "input":
                add_edge(ak, tk, "SPENDS", r.timestep)
            else:
                add_edge(tk, ak, "PAYS", r.timestep)
            frontier.append(r.address)
    if not nodes:
        raise AddressNotFoundError("none of the requested seeds appears in the chain index")

    directions = ["downstream", "upstream"] if direction == "both" else [direction]
    visited: set[tuple[str, str]] = set()
    for hop in range(1, hops + 1):
        nxt = []
        for a in frontier:
            for d in directions:
                if (a, d) in visited:
                    continue
                visited.add((a, d))
                mine = _rows(idx.by_address, a).reset_index()
                mine = mine[mine.role == ("input" if d == "downstream" else "output")]
                for r in mine.itertuples():
                    if not in_time(r.timestep):
                        continue
                    t = int(r.txid)
                    tr = _rows(idx.by_txid, t)
                    if len(tr) > max_tx_degree:
                        hubs[t] = len(tr)
                        continue
                    if len(nodes) >= max_nodes:
                        truncated = True
                        break
                    tk = add_tx(t, hop)
                    ak = f"addr:{a}"
                    if d == "downstream":
                        add_edge(ak, tk, "SPENDS", r.timestep)
                        others = tr[tr.role == "output"].reset_index()
                    else:
                        add_edge(tk, ak, "PAYS", r.timestep)
                        others = tr[tr.role == "input"].reset_index()
                    for o in others.itertuples():
                        if len(nodes) >= max_nodes and f"addr:{o.address}" not in nodes:
                            truncated = True
                            break
                        ok = add_addr(o.address, hop)
                        if d == "downstream":
                            add_edge(tk, ok, "PAYS", o.timestep)
                        else:
                            add_edge(ok, tk, "SPENDS", o.timestep)
                        nxt.append(o.address)
                if truncated:
                    break
            if truncated:
                break
        frontier = list(dict.fromkeys(nxt))
        if truncated or not frontier:
            break

    notes = _annotate_addresses([n["data"]["address"] for n in nodes.values() if n["kind"] == "address"], root)
    for n in nodes.values():
        if n["kind"] == "address":
            n["data"].update(notes.get(n["data"]["address"], {}))
    return {
        "seeds": {"addresses": list(seeds), "txids": [int(t) for t in seed_txids or []]},
        "direction": direction, "hops": hops, "max_nodes": max_nodes,
        "time_window": {"min_timestep": min_timestep, "max_timestep": max_timestep},
        "graph": {"node_count": len(nodes), "edge_count": len(edges),
                  "nodes": list(nodes.values()), "edges": list(edges.values())},
        "truncated": truncated,
        "hub_transactions_skipped": [{"txid": t, "participants": n} for t, n in sorted(hubs.items())][:50],
        "hub_threshold": max_tx_degree,
        "meaning": ("Edges follow value: SPENDS = an address funded the transaction, PAYS = the "
                    "transaction paid the address. Sharing a transaction is not common ownership."),
        "provenance": _provenance(idx),
    }


# ---- transaction -------------------------------------------------------------------------


def get_transaction(txid: int, root=None) -> dict:
    idx = chain(root)
    rows = _rows(idx.by_txid, int(txid))
    if rows.empty:
        raise TransactionNotFoundError(f"transaction {txid} does not appear in the Elliptic++ chain index")
    rows = rows.reset_index()
    ins = sorted(set(rows[rows.role == "input"].address))
    outs = sorted(set(rows[rows.role == "output"].address))
    notes = _annotate_addresses(ins + outs, root)
    return {**_tx_row(idx, int(txid)),
            "inputs": [{"address": a, **notes[a]} for a in ins],
            "outputs": [{"address": a, **notes[a]} for a in outs],
            "provenance": _provenance(idx)}


# ---- search ---------------------------------------------------------------------------------


_ALERT = re.compile(r"^[0-9a-f]{16}:\d+$")


def search(q: str, root=None, limit: int = 10) -> dict:
    q = (q or "").strip()
    results: list[dict] = []
    if not q:
        return {"query": q, "results": []}
    idx = chain(root)
    if _ALERT.match(q):
        results.append({"kind": "alert", "id": q, "label": q})
    if q.isdigit():
        t = int(q)
        if t in idx.tx.index:
            results.append({"kind": "transaction", "id": str(t), "label": f"Transaction {t}",
                            "detail": _tx_row(idx, t)})
    if len(q) >= 4 and not q.isdigit():
        arr = idx.addresses_sorted
        lo = arr.searchsorted(q, side="left")
        hi = arr.searchsorted(q + "￿", side="left")
        for a in arr[lo:min(hi, lo + limit)]:
            results.append({"kind": "address", "id": a, "label": a})
    return {"query": q, "results": results[:limit],
            "meaning": "Exact transaction id, alert id, or address prefix (at least 4 characters)."}
