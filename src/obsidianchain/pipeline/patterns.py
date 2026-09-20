"""Pattern detection adapters for peeling chains and CoinJoin/mixing transactions.

Conforms to SPEC 5 (peel) and M4 (mixing) concepts, adapted to operate directly
on canonical Problem Statement transaction records with discrete input/output amounts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from obsidianchain.pipeline.blockchain import BlockchainGraph, TransactionFact


@dataclass
class PeelingChain:
    """A detected sequential peeling chain."""

    chain_id: str
    txids: list[str]
    addresses: list[str]
    depth: int
    total_peeled_amount: float
    retained_amount: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "txids": self.txids,
            "addresses": self.addresses,
            "depth": self.depth,
            "total_peeled_amount": round(self.total_peeled_amount, 6),
            "retained_amount": round(self.retained_amount, 6),
        }


@dataclass
class PeelingResult:
    """Outcome of peeling chain analysis across the graph."""

    chains: list[PeelingChain] = field(default_factory=list)
    tx_to_chain: dict[str, str] = field(default_factory=dict)
    tx_to_depth: dict[str, int] = field(default_factory=dict)

    @property
    def detected_count(self) -> int:
        return len(self.chains)

    def is_peeling(self, txid: str) -> bool:
        return txid in self.tx_to_chain


@dataclass
class MixingFact:
    """Forensic evaluation of one transaction for CoinJoin / mixing structure."""

    txid: str
    classification: str
    """'MIXING_PATTERN', 'MIXING_LIKELIHOOD', or 'NO_MIXING_SIGNAL'."""
    input_count: int
    output_count: int
    equal_output_count: int
    equal_amount: float | None
    change_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "txid": self.txid,
            "classification": self.classification,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "equal_output_count": self.equal_output_count,
            "equal_amount": self.equal_amount,
            "change_count": self.change_count,
        }


@dataclass
class MixingResult:
    """Outcome of mixing / CoinJoin analysis across the graph."""

    transactions: dict[str, MixingFact] = field(default_factory=dict)

    @property
    def pattern_count(self) -> int:
        return sum(1 for f in self.transactions.values() if f.classification == "MIXING_PATTERN")

    @property
    def likelihood_count(self) -> int:
        return sum(1 for f in self.transactions.values() if f.classification == "MIXING_LIKELIHOOD")

    def classification_for_tx(self, txid: str) -> str:
        rec = self.transactions.get(txid)
        return rec.classification if rec else "NO_MIXING_SIGNAL"


def detect_peeling_chains(graph: BlockchainGraph, min_depth: int = 2) -> PeelingResult:
    """Detect sequential peeling chains through change-address propagation.

    A peeling chain is a sequence of 2- or 3-output transactions where one output
    serves as change that is spent in the next transaction, while smaller amounts
    are peeled off.
    """
    res = PeelingResult()
    txs = graph.transactions

    # Map address -> tx where it is spent as input: dict[address, list[txid]]
    spending_txs: dict[str, list[str]] = {}
    for txid, fact in txs.items():
        for addr, _ in fact.inputs:
            spending_txs.setdefault(addr, []).append(txid)

    # Sort transactions by timestamp if available
    def _tx_time(t_fact: TransactionFact) -> float:
        if t_fact.timestamp is None:
            return 0.0
        try:
            return float(t_fact.timestamp)
        except (ValueError, TypeError):
            return 0.0

    visited_in_chain: set[str] = set()

    for start_txid, start_fact in txs.items():
        if start_txid in visited_in_chain:
            continue
        if len(start_fact.outputs) not in (2, 3):
            continue

        # Try to trace a chain forward
        current_txid = start_txid
        current_fact = start_fact
        chain_txs = [current_txid]
        chain_addrs: list[str] = []
        peeled_sum = 0.0

        while True:
            # Look for an output of current_fact that is spent by an admissible next tx
            next_hop = None
            for out_addr, out_amt in current_fact.outputs:
                spenders = spending_txs.get(out_addr, [])
                for candidate_txid in spenders:
                    if candidate_txid == current_txid or candidate_txid in chain_txs:
                        continue
                    cand_fact = txs.get(candidate_txid)
                    if cand_fact and len(cand_fact.outputs) in (2, 3):
                        # Strict or non-decreasing time order
                        if _tx_time(cand_fact) >= _tx_time(current_fact):
                            next_hop = (candidate_txid, cand_fact, out_addr, out_amt)
                            break
                if next_hop:
                    break

            if next_hop:
                next_txid, next_fact, hop_addr, hop_amt = next_hop
                chain_txs.append(next_txid)
                chain_addrs.append(hop_addr)
                # Peeled amount is the output amount that did NOT continue
                other_outs = [amt for a, amt in current_fact.outputs if a != hop_addr]
                peeled_sum += sum(other_outs)
                current_txid = next_txid
                current_fact = next_fact
            else:
                break

        if len(chain_txs) >= min_depth:
            cid = f"peel_chain_{start_txid}"
            retained = current_fact.outputs[0][1] if current_fact.outputs else 0.0
            chain = PeelingChain(
                chain_id=cid,
                txids=chain_txs,
                addresses=chain_addrs,
                depth=len(chain_txs),
                total_peeled_amount=peeled_sum,
                retained_amount=retained,
            )
            res.chains.append(chain)
            for depth_idx, tx_in_c in enumerate(chain_txs):
                res.tx_to_chain[tx_in_c] = cid
                res.tx_to_depth[tx_in_c] = depth_idx + 1
                visited_in_chain.add(tx_in_c)

    return res


def detect_mixing_patterns(graph: BlockchainGraph) -> MixingResult:
    """Detect CoinJoin / collaborative spend patterns in transactions.

    Characteristics:
        - Multiple inputs from distinct addresses
        - Equal denomination outputs (at least 2 or 3 outputs sharing identical amount)
    """
    res = MixingResult()

    for txid, fact in graph.transactions.items():
        in_addrs = {addr for addr, _ in fact.inputs}
        n_inputs = len(in_addrs)
        n_outputs = len(fact.outputs)

        if n_inputs < 2 or n_outputs < 2:
            res.transactions[txid] = MixingFact(
                txid=txid,
                classification="NO_MIXING_SIGNAL",
                input_count=n_inputs,
                output_count=n_outputs,
                equal_output_count=0,
                equal_amount=None,
                change_count=n_outputs,
            )
            continue

        # Count frequencies of output amounts (rounded to 8 decimal satoshi precision)
        amt_counts: dict[float, int] = {}
        for _, amt in fact.outputs:
            if amt > 0:
                r_amt = round(amt, 8)
                amt_counts[r_amt] = amt_counts.get(r_amt, 0) + 1

        max_equal_count = 0
        equal_val = None
        for val, count in amt_counts.items():
            if count > max_equal_count:
                max_equal_count = count
                equal_val = val

        classification = "NO_MIXING_SIGNAL"
        if n_inputs >= 3 and max_equal_count >= 3:
            classification = "MIXING_PATTERN"
        elif n_inputs >= 2 and max_equal_count >= 2:
            classification = "MIXING_LIKELIHOOD"

        change_count = n_outputs - max_equal_count if max_equal_count >= 2 else n_outputs

        res.transactions[txid] = MixingFact(
            txid=txid,
            classification=classification,
            input_count=n_inputs,
            output_count=n_outputs,
            equal_output_count=max_equal_count if max_equal_count >= 2 else 0,
            equal_amount=equal_val if max_equal_count >= 2 else None,
            change_count=change_count,
        )

    return res
