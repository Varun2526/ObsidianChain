"""Blockchain <-> network coherence: does the P2P layer agree with the money flow?

TXID matching (``correlation/engine.py``) says which transactions were
observed. This module asks the question that matching cannot: when value
moves on chain from one transaction to the next, were those two
transactions first announced to the observers by the same relay peer more
often than chance allows?

An operator who broadcasts their own chain from their own node leaks
exactly that: consecutive hops are first seen from one peer. A large public
relay that first-announces a big share of everything does not, because the
chance rate is taken from the capture itself.

Definitions (pre-registered in
``research/network_2026_09_25/PREREGISTRATION_exp_net2.md``)
--------------------------------------------------------------------------
- **Hop pair**: (t1, t2) where t2 spends an output of t1, both observed.
- **Coherent**: the two first-seen peer sets intersect (ties kept).
- **Chance rate** ``q = sum_p f_p^2``, with ``f_p`` the share of observed
  transactions whose first-seen set contains peer p.
- **Per cluster**: m hop pairs touching the cluster's members, k coherent,
  ``p = P(X >= k)`` for X ~ Binomial(m, q). PRESENT iff k >= 2 and
  p <= 0.05; score ``min(1, -log10(p) / 4)``.

A relay is a vantage point. Nothing here identifies a sender.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from scipy.stats import binom

SCHEMA = "obsidianchain.cross_layer/1"
MIN_COHERENT = 2
P_MAX = 0.05

MEANING = (
    "Cross-layer coherence: on-chain hops (a transaction spending another's "
    "output) whose two transactions were first announced by the same relay "
    "peer, compared with the chance rate of that happening in this capture. "
    "A relay is where the observers first heard a transaction, never the sender."
)


@dataclass(frozen=True)
class HopPair:
    parent: str
    child: str
    via_address: str
    coherent: bool
    shared_peers: tuple[str, ...]
    delta_ms: int | None

    def as_dict(self) -> dict:
        return {"parent": self.parent, "child": self.child, "via_address": self.via_address,
                "coherent": self.coherent, "shared_peers": list(self.shared_peers),
                "delta_ms": self.delta_ms}


@dataclass
class CrossLayerResult:
    chance_rate: float | None
    observed_transactions: int
    pairs: list[HopPair] = field(default_factory=list)
    _by_tx: dict[str, list[int]] = field(default_factory=dict, repr=False)
    _peer_share: dict[str, float] = field(default_factory=dict, repr=False)
    _first: dict[str, frozenset] = field(default_factory=dict, repr=False)

    @property
    def coherent_pairs(self) -> list[HopPair]:
        return [p for p in self.pairs if p.coherent]

    def evaluate(self, txids) -> dict:
        """The cluster-level test over hop pairs touching ``txids``."""
        idx = sorted({i for t in txids for i in self._by_tx.get(t, [])})
        pairs = [self.pairs[i] for i in idx]
        m = len(pairs)
        coherent = [p for p in pairs if p.coherent]
        k = len(coherent)
        q = self.chance_rate
        if q is None or m == 0:
            return {"status": "NO_EVIDENCE", "score": 0.0, "hop_pairs": m, "coherent_pairs": k,
                    "chance_rate": q, "p_value": None, "relays": [], "pairs": [],
                    "reason": "no observed on-chain hops" if m == 0 else "no network observations"}
        p_value = float(binom.sf(k - 1, m, q)) if k > 0 else 1.0
        present = k >= MIN_COHERENT and p_value <= P_MAX
        score = min(1.0, -math.log10(max(p_value, 1e-300)) / 4.0) if present else 0.0
        relays = Counter(peer for pair in coherent for peer in pair.shared_peers).most_common(3)
        return {
            "status": "PRESENT" if present else "NO_EVIDENCE",
            "score": round(score, 4),
            "hop_pairs": m,
            "coherent_pairs": k,
            "chance_rate": round(q, 4),
            "expected_coherent": round(m * q, 2),
            "p_value": p_value,
            "relays": [{"peer_ip": ip, "coherent_pairs": n} for ip, n in relays],
            "pairs": [p.as_dict() for p in coherent[:20]],
        }

    def relay_flows(self, blockchain_graph, address_to_cluster: dict[str, str]) -> list[dict]:
        """Flows the network layer ties together across clusters.

        For each relay r: of the m_r hops whose parent transaction r first
        announced, how many had the child first announced by r too (k_r),
        against r's overall first-announcement share f_r, which is what the
        child would show if r's announcements ignored the money flow.
        ``p = P(X >= k_r)``, X ~ Binomial(m_r, f_r). Relays significant after
        Bonferroni over the relays tested join the spending clusters of their
        coherent hops into one flow.

        Conditioning on the parent (rather than testing k_r against all hops
        in the capture) keeps unrelated hops, such as an exchange re-spending
        its own change, from diluting the test.

        An ownership lead, not a risk claim: exp-net2 showed relay coherence
        marks operator-controlled flow, which benign services produce too.
        """
        q_of = self._peer_share
        per_relay: dict[str, list[HopPair]] = {}
        from_relay: Counter = Counter()
        for pair in self.pairs:
            for peer in self._first.get(pair.parent, ()):
                from_relay[peer] += 1
        for pair in self.coherent_pairs:
            for peer in pair.shared_peers:
                per_relay.setdefault(peer, []).append(pair)
        tested = {r: ps for r, ps in per_relay.items() if len(ps) >= MIN_COHERENT}
        if not tested:
            return []
        alpha = P_MAX / len(tested)

        def spender(txid: str) -> str | None:
            fact = blockchain_graph.transactions.get(txid)
            return address_to_cluster.get(fact.inputs[0][0]) if fact is not None and fact.inputs else None

        flows = []
        for relay, pairs in tested.items():
            f = q_of.get(relay, 0.0)
            m_r = from_relay[relay]
            p_value = float(binom.sf(len(pairs) - 1, m_r, f))
            if p_value > alpha:
                continue
            parent: dict[str, str] = {}

            def find(x: str) -> str:
                while parent.setdefault(x, x) != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            hops: dict[str, list[HopPair]] = {}
            for pair in pairs:
                a, b = spender(pair.parent), spender(pair.child)
                if a is None or b is None:
                    continue
                parent[find(a)] = find(b)
                hops.setdefault(a, []).append(pair)
            groups: dict[str, set[str]] = {}
            for c in list(parent):
                groups.setdefault(find(c), set()).add(c)
            for members in groups.values():
                if len(members) < 2:
                    continue
                edges = sorted({(spender(p.parent), spender(p.child)) for c in members for p in hops.get(c, [])
                                if spender(p.parent) != spender(p.child)})
                flows.append({"relay": relay, "clusters": sorted(members), "cluster_edges": edges,
                              "relay_coherent_pairs": len(pairs), "relay_parent_hops": m_r,
                              "expected": round(m_r * f, 3),
                              "p_value": p_value, "alpha": alpha, "status": "LEAD_ONLY"})
        flows.sort(key=lambda x: (x["p_value"], -len(x["clusters"])))
        return flows

    def summary(self) -> dict:
        return {
            "schema": SCHEMA,
            "observed_transactions": self.observed_transactions,
            "hop_pairs": len(self.pairs),
            "coherent_pairs": len(self.coherent_pairs),
            "chance_rate": None if self.chance_rate is None else round(self.chance_rate, 4),
            "meaning": MEANING,
        }


def analyse(blockchain_graph, network_propagation) -> CrossLayerResult:
    """Hop pairs over the capture, each marked coherent or not."""
    first: dict[str, tuple[frozenset, int | None]] = {}
    if network_propagation is not None:
        for txid, row in network_propagation.transactions.items():
            if row.first_seen_peers:
                first[txid] = (frozenset(row.first_seen_peers), row.first_seen_ms)
    counts = Counter(peer for peers, _ in first.values() for peer in peers)
    n = len(first)
    q = sum((c / n) ** 2 for c in counts.values()) if n else None

    spenders: dict[str, list[str]] = {}
    for txid, fact in blockchain_graph.transactions.items():
        for address, _amount in fact.inputs:
            spenders.setdefault(address, []).append(txid)

    result = CrossLayerResult(chance_rate=q, observed_transactions=n,
                              _peer_share={peer: c / n for peer, c in counts.items()} if n else {},
                              _first={t: peers for t, (peers, _ms) in first.items()})
    seen: set[tuple[str, str]] = set()
    for parent, fact in blockchain_graph.transactions.items():
        if parent not in first:
            continue
        for address, _amount in fact.outputs:
            for child in spenders.get(address, []):
                if child == parent or child not in first or (parent, child) in seen:
                    continue
                seen.add((parent, child))
                a_peers, a_ms = first[parent]
                b_peers, b_ms = first[child]
                shared = tuple(sorted(a_peers & b_peers))
                pair = HopPair(parent=parent, child=child, via_address=address,
                               coherent=bool(shared), shared_peers=shared,
                               delta_ms=None if a_ms is None or b_ms is None else b_ms - a_ms)
                index = len(result.pairs)
                result.pairs.append(pair)
                result._by_tx.setdefault(parent, []).append(index)
                result._by_tx.setdefault(child, []).append(index)
    return result
