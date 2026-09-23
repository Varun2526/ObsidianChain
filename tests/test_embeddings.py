"""Counterparty-profile embeddings and cross-cluster link suggestions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from obsidianchain.ml import embeddings as emb


def _world(n_groups: int = 60, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Groups of two addresses that pay the same private merchants.

    Within a group the two addresses never co-spend (different clusters) but
    share counterparties - the case co-spend clustering cannot see.
    """
    rng = np.random.default_rng(seed)
    rows, cluster = [], {}
    for g in range(n_groups):
        merchants = [f"m{g}_{k}" for k in range(4)]
        for member in ("a", "b"):
            addr = f"g{g}{member}"
            cluster[addr] = addr
            for m in rng.choice(merchants, size=3, replace=False):
                rows.append((addr, str(m)))
    return pd.DataFrame(rows, columns=["payer", "payee"]), cluster


def test_shared_counterparties_make_addresses_similar() -> None:
    flows, _ = _world()
    e = emb.embed(flows, dim=32)
    same = np.mean([e.similarity(f"g{g}a", f"g{g}b") for g in range(60)])
    other = np.mean([e.similarity(f"g{g}a", f"g{(g + 1) % 60}b") for g in range(60)])
    assert same > 0.5 > other


def test_suggestions_cross_clusters_only_and_never_merge() -> None:
    flows, cluster = _world()
    e = emb.embed(flows, dim=32)
    s = emb.suggest_links(e, cluster, threshold=0.5)
    assert s.status == "RUN" and s.pairs
    assert all(p["cluster_a"] != p["cluster_b"] for p in s.pairs)
    # Mostly the true group partner.
    partner = [p for p in s.pairs if p["address_a"][:-1] == p["address_b"][:-1]]
    assert len(partner) / len(s.pairs) > 0.5
    # Two addresses already in one cluster are never suggested.
    merged = dict(cluster, g0b="g0a")
    s2 = emb.suggest_links(e, merged, threshold=0.5)
    assert not any({p["address_a"], p["address_b"]} == {"g0a", "g0b"} for p in s2.pairs)
    assert "nothing is merged" in s.summary()["meaning"]


def test_a_tiny_graph_produces_no_suggestion_and_says_why() -> None:
    flows = pd.DataFrame([("a", "m"), ("b", "m"), ("c", "n")], columns=["payer", "payee"])
    s = emb.suggest_links(emb.embed(flows), {})
    assert s.status == "GRAPH_TOO_SMALL"
    assert s.pairs == []


def test_hub_counterparties_are_dropped() -> None:
    flows, _ = _world()
    hub = pd.DataFrame([(f"g{g}{m}", "exchange") for g in range(60) for m in "ab"],
                       columns=["payer", "payee"])
    e = emb.embed(pd.concat([flows, hub]), dim=16, max_counterparty_degree=50)
    assert e.dropped_hub_counterparties >= 1


def test_an_address_seen_in_one_transaction_is_never_suggested() -> None:
    """Every payee of one transaction shares its payers; similarity built from
    a single transaction says nothing about ownership."""
    flows, cluster = _world()
    e = emb.embed(flows, dim=32)
    once = {a: 1 for a in e.addresses}
    s = emb.suggest_links(e, cluster, threshold=0.5, tx_counts=once)
    assert s.pairs == []
    twice = {a: 2 for a in e.addresses}
    assert emb.suggest_links(e, cluster, threshold=0.5, tx_counts=twice).pairs
