"""exp-net2b (secondary, reported not gating): do cross-layer flows join clusters of one owner?

After exp-net2 showed relay coherence is not a risk signal, it is used only
to link clusters (``cross_layer.relay_flows``). This measures what such a
link is worth on the v2 worlds, where the true owner of every address is
known: link precision = share of linked cluster pairs whose addresses belong
to the same world entity. Chance precision = the same for random cluster
pairs, and (the fair baseline) for every pair of clusters joined by an
on-chain hop whatever the relay. SYNTHETIC_CONTROL. In the v2 generator each
well-connected operator broadcasts from its own node, so precision here is
an upper bound for real traffic, where operators share relays.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from obsidianchain.correlation import cross_layer  # noqa: E402
from obsidianchain.correlation.engine import correlate_blockchain_and_network  # noqa: E402
from obsidianchain.io import ingest  # noqa: E402
from obsidianchain.network import propagation  # noqa: E402
from obsidianchain.pipeline.blockchain import build_blockchain_layer  # noqa: E402

OUT = Path(__file__).parent / "results" / "exp_net2b_flow_precision.json"


def world(name: str) -> dict:
    root = ROOT / "data" / "synthetic_world_v2" / name
    frame, _ = ingest.ingest(root / "capture.csv")
    bg, clusters = build_blockchain_layer(frame)
    net = propagation.analyse(correlate_blockchain_and_network(frame, bg))
    xl = cross_layer.analyse(bg, net)
    flows = xl.relay_flows(bg, clusters.address_to_cluster)
    truth = pd.read_csv(root / "world_truth" / "labels.csv")  # joined after linking
    owner = dict(zip(truth.address, truth.entity))
    cl_owner = {c: {owner.get(a) for a in addrs} - {None} for c, addrs in clusters.cluster_to_addresses.items()}
    pairs = {tuple(sorted(e)) for f in flows for e in f["cluster_edges"]}
    same = sum(bool(cl_owner.get(a, set()) & cl_owner.get(b, set())) for a, b in pairs)
    rng = random.Random(20260925)
    cids = list(cl_owner)
    rand = [tuple(rng.sample(cids, 2)) for _ in range(20000)]
    chance = sum(bool(cl_owner[a] & cl_owner[b]) for a, b in rand) / len(rand)
    def spender(t):
        fact = bg.transactions.get(t)
        return clusters.address_to_cluster.get(fact.inputs[0][0]) if fact and fact.inputs else None
    adjacent = {tuple(sorted((spender(p.parent), spender(p.child)))) for p in xl.pairs
                if spender(p.parent) and spender(p.child) and spender(p.parent) != spender(p.child)}
    adj_same = sum(bool(cl_owner.get(a, set()) & cl_owner.get(b, set())) for a, b in adjacent)
    return {"world": name, "hop_adjacent_cluster_pairs": len(adjacent),
            "hop_adjacent_precision": round(adj_same / len(adjacent), 4) if adjacent else None, "flows": len(flows), "linked_cluster_pairs": len(pairs),
            "same_owner": same, "precision": round(same / len(pairs), 4) if pairs else None,
            "chance_precision": round(chance, 5), "relays_significant": len({f["relay"] for f in flows})}


if __name__ == "__main__":
    res = {n: world(n) for n in ("signal", "null")}
    for r in res.values():
        print(r)
    OUT.write_text(json.dumps({"experiment_id": "exp_net2b_flow_precision", "provenance_type": "SYNTHETIC_CONTROL",
                               "results": res}, indent=2))
