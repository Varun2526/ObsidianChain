"""Blockchain <-> network coherence (correlation/cross_layer.py)."""

from __future__ import annotations

import random
from types import SimpleNamespace

from obsidianchain.correlation import cross_layer
from obsidianchain.pipeline.blockchain import BlockchainGraph, TransactionFact


def _chain_graph(n_chain: int, n_background: int) -> BlockchainGraph:
    """A peel-like chain c0 -> c1 -> ... plus unrelated background payments."""
    g = BlockchainGraph()
    for i in range(n_chain):
        g.transactions[f"c{i}"] = TransactionFact(
            txid=f"c{i}", timestamp=None, inputs=[(f"hold{i}", 10.0 - i)],
            outputs=[(f"hold{i + 1}", 9.0 - i), (f"pay{i}", 0.9)], fee=0.1, script_type="p2pkh")
    for j in range(n_background):
        g.transactions[f"b{j}"] = TransactionFact(
            txid=f"b{j}", timestamp=None, inputs=[(f"w{j}", 1.0)],
            outputs=[(f"m{j}", 0.5), (f"wc{j}", 0.49)], fee=0.01, script_type="p2pkh")
    return g


def _propagation(assign: dict[str, str]):
    return SimpleNamespace(transactions={
        t: SimpleNamespace(first_seen_peers=[peer], first_seen_ms=1_700_000_000_000 + i * 1000)
        for i, (t, peer) in enumerate(assign.items())
    })


def _peers(n: int) -> list[str]:
    return [f"203.0.113.{i}" for i in range(1, n + 1)]


def test_a_chain_first_seen_from_one_rare_relay_is_coherent():
    g = _chain_graph(6, 40)
    rng = random.Random(1)
    assign = {f"b{j}": rng.choice(_peers(30)) for j in range(40)}
    assign.update({f"c{i}": "198.51.100.77" for i in range(6)})
    result = cross_layer.analyse(g, _propagation(assign))
    assert len(result.pairs) == 5 and len(result.coherent_pairs) == 5
    test = result.evaluate({f"c{i}" for i in range(6)})
    assert test["status"] == "PRESENT" and test["p_value"] < 1e-4
    assert test["relays"][0]["peer_ip"] == "198.51.100.77"
    assert test["score"] == 1.0


def test_independent_relays_are_flagged_at_about_the_nominal_rate():
    """Under independence the per-cluster test is a 5% test: across many
    null captures it must flag about 5% of chains, not more. A single seed
    can and does flag by chance (2 of 5 hops matching via two different
    relays), which is why this is a rate, not one case."""
    flagged = 0
    trials = 400
    for seed in range(trials):
        g = _chain_graph(6, 40)
        rng = random.Random(seed)
        assign = {t: rng.choice(_peers(30)) for t in g.transactions}
        test = cross_layer.analyse(g, _propagation(assign)).evaluate({f"c{i}" for i in range(6)})
        flagged += test["status"] == "PRESENT"
    assert flagged / trials <= 0.08, flagged / trials


def test_a_relay_that_announces_everything_is_not_a_link():
    """Coherence at the chance rate is not evidence, however many pairs share it."""
    g = _chain_graph(6, 40)
    assign = {t: "8.8.8.8" for t in g.transactions}
    result = cross_layer.analyse(g, _propagation(assign))
    assert result.chance_rate == 1.0
    test = result.evaluate({f"c{i}" for i in range(6)})
    assert test["coherent_pairs"] == 5 and test["status"] == "NO_EVIDENCE"


def test_unobserved_transactions_are_left_out_not_counted_as_incoherent():
    g = _chain_graph(6, 10)
    assign = {f"c{i}": "198.51.100.77" for i in (0, 1, 2)}
    assign.update({f"b{j}": f"203.0.113.{j}" for j in range(10)})
    result = cross_layer.analyse(g, _propagation(assign))
    assert {(p.parent, p.child) for p in result.pairs} == {("c0", "c1"), ("c1", "c2")}


def test_no_network_data_is_reported_as_such():
    g = _chain_graph(4, 0)
    result = cross_layer.analyse(g, None)
    assert result.chance_rate is None
    assert result.evaluate({"c0"})["status"] == "NO_EVIDENCE"


def test_a_relay_coherent_chain_forms_one_flow_across_clusters():
    g = _chain_graph(6, 40)
    rng = random.Random(1)
    assign = {f"b{j}": rng.choice(_peers(30)) for j in range(40)}
    assign.update({f"c{i}": "198.51.100.77" for i in range(6)})
    result = cross_layer.analyse(g, _propagation(assign))
    # every holding address is its own singleton cluster, as in a peel chain
    a2c = {f"hold{i}": f"k{i}" for i in range(7)}
    flows = result.relay_flows(g, a2c)
    assert len(flows) == 1
    assert flows[0]["relay"] == "198.51.100.77"
    assert flows[0]["clusters"] == [f"k{i}" for i in range(6)]
    assert flows[0]["status"] == "LEAD_ONLY"


def test_cross_layer_evidence_is_not_fused_and_does_not_corroborate():
    """exp-net2: NOT_DEMONSTRATED, so the line must not move the score."""
    from obsidianchain.pipeline import alerts
    assert alerts.FUSE_CROSS_LAYER is False
    assert alerts.CROSS_LAYER_CONTEXT not in alerts.FUSION_WEIGHTS


def test_relay_flows_are_rare_when_relays_ignore_the_money_flow():
    """Null calibration for flows: random first relays over a capture with
    several chains must almost never produce a flow."""
    with_flow = 0
    trials = 200
    for seed in range(trials):
        g = _chain_graph(8, 60)
        rng = random.Random(1000 + seed)
        assign = {t: rng.choice(_peers(12)) for t in g.transactions}
        a2c = {f"hold{i}": f"k{i}" for i in range(9)}
        with_flow += bool(cross_layer.analyse(g, _propagation(assign)).relay_flows(g, a2c))
    assert with_flow / trials <= 0.05, with_flow / trials
