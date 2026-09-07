"""Change-address detection: a second, deliberately weaker merge signal.

Why weaker
----------
Multi-input co-spend is backed by cryptography: whoever built the transaction
held the private key for every input, so those addresses share an owner. That
is close to proof.

Change detection is not. It is an inference from behavioural regularities in
how wallet software builds transactions, and getting it wrong merges two
unrelated entities. It is the documented cause of cluster collapse, where one
bad change guess chains into a super-cluster spanning much of the graph. So
every candidate here carries a confidence, nothing merges below a threshold,
and change-derived merges are accounted for separately from co-spend ones for
as long as they exist.

Signals
-------
Scored per output of a transaction, never decided from one alone:

``self_reference``
    The output address is also an input to the same transaction. Then it is
    change with near-certainty - the sender is paying themselves. Note this
    yields no *new* merges, since co-spend has already unioned that address
    with the transaction's other inputs; it is a diagnostic, not evidence.
``one_fresh``
    Exactly one of the two outputs is fresh and the other is not. The fresh
    one is the likely change address, since wallets typically derive a new
    address for change while the payment goes somewhere already in use.
``fresher_first_block``
    The output's first appearance is later than its peer's, from
    wallets_features.csv. A finer-grained version of the same intuition.
``peer_reused``
    The other output has been seen in several transactions, which makes it
    look like a payment destination and strengthens the case for this one.

Gates
-----
Only transactions with **exactly two outputs** are considered. Anything with
more is batching - an exchange paying many customers at once - where the
notion of "the change output" does not apply and guessing is meaningless.
Transactions with no recorded inputs (coinbase) are skipped as well, since
there is nothing to merge the change address with.

The merge decision itself - which scored candidates actually become edges -
lives in :func:`select_change_rows`, which is intentionally unimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.io import elliptic

#: Nothing merges below this confidence unless the caller overrides it.
DEFAULT_THRESHOLD = 0.7

#: Rows read per chunk when scanning the 578 MB feature file.
_FEATURE_CHUNK = 250_000

_NO_BLOCK = np.iinfo(np.int32).max


@dataclass(frozen=True)
class ChangeWeights:
    """Additive confidence contributions per signal.

    The signals are **not independent**, so not every arithmetic sum is
    reachable. ``one_fresh`` means this output is fresh and its peer is not;
    "not fresh" means the peer has degree >= 2, which is exactly
    ``peer_reused``. So ``one_fresh`` always implies ``peer_reused``, and
    scores of 0.50 or 0.70 can never occur.

    The bands that actually occur, measured on Elliptic++:

    ====== ===================================================== =========
    score  signals                                               candidates
    ====== ===================================================== =========
    0.00   none                                                     95,103
    0.15   peer_reused                                              78,222
    0.35   peer_reused + fresher_first_block                        26,577
    0.65   one_fresh + peer_reused                                  57,403
    0.85   one_fresh + peer_reused + fresher_first_block            24,569
    0.98   self_reference                                           43,476
    ====== ===================================================== =========

    The default threshold of 0.7 therefore means, in practice, "require the
    one-fresh rule *and* corroboration from first-block ordering". Dropping
    it to 0.6 admits the 0.65 band, which roughly triples the change merges
    and lifts coverage to about 50%. There is no threshold between 0.65 and
    0.85 that does anything different, so tune in those steps, not smoothly.

    Note that ``self_reference`` scores highest but contributes **no new
    merges**: an output that is also an input to its own transaction has
    already been unioned with the other inputs by co-spend. It is kept as a
    diagnostic and a sanity check, not as a source of evidence.
    """

    self_reference: float = 0.98
    one_fresh: float = 0.50
    fresher_first_block: float = 0.20
    peer_reused: float = 0.15


@dataclass
class ChangeCandidates:
    """Scored change candidates, two rows per eligible transaction."""

    table: pd.DataFrame
    """Columns: tx_id, code, peer_code, input_rep, n_inputs, confidence,
    plus one boolean column per signal."""

    weights: ChangeWeights = field(default_factory=ChangeWeights)

    n_transactions_with_outputs: int = 0
    n_two_output: int = 0
    n_skipped_output_count: int = 0
    n_skipped_no_inputs: int = 0
    n_unmapped_outputs: int = 0
    features_used: bool = True

    @property
    def n_candidates(self) -> int:
        return int(len(self.table))

    def above(self, threshold: float) -> pd.DataFrame:
        """Candidate rows scoring at or above ``threshold``."""
        return self.table[self.table["confidence"] >= threshold]


# ---- inputs -----------------------------------------------------------


def load_output_edges(data_root: Path | None = None) -> pd.DataFrame:
    """Read TxAddr_edgelist.csv - the output side (txId, output_address)."""
    path = elliptic.find_dataset_file(elliptic.TX_ADDR, data_root)
    try:
        return pd.read_csv(path, usecols=["txId", "output_address"])
    except ValueError as exc:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        raise ValueError(
            f"{path.name} does not have the expected columns.\n"
            f"  expected: ['txId', 'output_address']\n"
            f"  found:    {header}\n"
            f"Original error: {exc}"
        ) from exc


def load_first_block_by_code(
    index: pd.Index, n_addresses: int, data_root: Path | None = None
) -> np.ndarray:
    """First block each address appeared in, indexed by address code.

    wallets_features.csv is address x time step, so an address has several
    rows and we take the minimum. The file is 578 MB and reading it whole
    peaks over 1.2 GB, so it is streamed in chunks and reduced straight into
    a code-indexed int32 array. Addresses with no feature row keep the
    sentinel ``_NO_BLOCK``.
    """
    path = elliptic.find_dataset_file("wallets_features.csv", data_root)
    first_block = np.full(n_addresses, _NO_BLOCK, dtype=np.int32)

    reader = pd.read_csv(
        path,
        usecols=["address", "first_block_appeared_in"],
        chunksize=_FEATURE_CHUNK,
    )
    for chunk in reader:
        codes = index.get_indexer(chunk["address"])
        keep = codes >= 0
        if not keep.any():
            continue
        grouped = (
            pd.DataFrame(
                {
                    "code": codes[keep],
                    "block": chunk["first_block_appeared_in"].to_numpy()[keep],
                }
            )
            .groupby("code", sort=False)["block"]
            .min()
        )
        target = first_block[grouped.index.to_numpy()]
        first_block[grouped.index.to_numpy()] = np.minimum(
            target, grouped.to_numpy().astype(np.int32)
        )
    return first_block


def address_tx_degree(
    input_pairs: pd.DataFrame, output_pairs: pd.DataFrame, n_addresses: int
) -> np.ndarray:
    """Distinct transactions each address takes part in, input or output.

    An address with degree 1 has been seen exactly once in the whole dataset,
    which is the strongest available proxy for "never seen before this
    transaction" - the edge lists carry no timestamps of their own.
    """
    both = pd.concat(
        [input_pairs[["tx_id", "code"]], output_pairs[["tx_id", "code"]]],
        ignore_index=True,
    ).drop_duplicates()
    degree = np.bincount(both["code"].to_numpy(), minlength=n_addresses)
    return degree.astype(np.int32)


# ---- candidate construction -------------------------------------------


def build_candidates(
    graph: elliptic.CoSpendGraph,
    data_root: Path | None = None,
    weights: ChangeWeights | None = None,
    use_features: bool = True,
) -> ChangeCandidates:
    """Score every output of every two-output transaction.

    ``graph`` must have been loaded with ``keep_labels=True``; the address
    strings are needed to map the output side into the same code space.
    """
    if graph.addresses is None:
        raise ValueError(
            "change detection needs address labels; "
            "call load_cospend_graph(keep_labels=True)"
        )
    weights = weights or ChangeWeights()
    index = pd.Index(graph.addresses)

    inputs = elliptic.load_input_edges(data_root)
    input_pairs = pd.DataFrame(
        {
            "tx_id": inputs["txId"].to_numpy(),
            "code": index.get_indexer(inputs["input_address"]).astype(np.int32),
        }
    )
    del inputs
    input_pairs = input_pairs[input_pairs["code"] >= 0].drop_duplicates()

    outputs = load_output_edges(data_root)
    out_codes = index.get_indexer(outputs["output_address"])
    n_unmapped = int((out_codes < 0).sum())
    output_pairs = pd.DataFrame(
        {"tx_id": outputs["txId"].to_numpy(), "code": out_codes.astype(np.int32)}
    )
    del outputs, out_codes
    output_pairs = output_pairs[output_pairs["code"] >= 0].drop_duplicates()

    degree = address_tx_degree(input_pairs, output_pairs, graph.n_addresses)

    first_block = (
        load_first_block_by_code(index, graph.n_addresses, data_root)
        if use_features
        else np.full(graph.n_addresses, _NO_BLOCK, dtype=np.int32)
    )
    del index

    # ---- gate: exactly two outputs -----------------------------------
    per_tx = output_pairs["tx_id"].value_counts()
    n_tx_with_outputs = int(len(per_tx))
    two_output_ids = per_tx.index[per_tx == 2]
    n_two_output = int(len(two_output_ids))

    pairs = output_pairs[output_pairs["tx_id"].isin(two_output_ids)]
    # Deterministic ordering so "first" and "second" output are stable.
    pairs = pairs.sort_values(["tx_id", "code"], kind="stable")

    tx_ids = pairs["tx_id"].to_numpy().reshape(-1, 2)[:, 0]
    codes = pairs["code"].to_numpy().reshape(-1, 2)
    del pairs, output_pairs, per_tx, two_output_ids

    left, right = codes[:, 0], codes[:, 1]

    # ---- one row per (transaction, output) ---------------------------
    table = pd.DataFrame(
        {
            "tx_id": np.repeat(tx_ids, 2),
            "code": codes.reshape(-1),
            "peer_code": np.column_stack([right, left]).reshape(-1),
        }
    )

    fresh = degree[table["code"].to_numpy()] == 1
    peer_fresh = degree[table["peer_code"].to_numpy()] == 1
    peer_degree = degree[table["peer_code"].to_numpy()]

    own_block = first_block[table["code"].to_numpy()]
    peer_block = first_block[table["peer_code"].to_numpy()]
    known = (own_block != _NO_BLOCK) & (peer_block != _NO_BLOCK)

    table["fresh"] = fresh
    table["peer_fresh"] = peer_fresh
    table["one_fresh"] = fresh & ~peer_fresh
    table["fresher_first_block"] = known & (own_block > peer_block)
    table["peer_reused"] = peer_degree >= 2

    # ---- inputs of the same transaction -------------------------------
    marker = input_pairs.assign(self_reference=True)
    table = table.merge(marker, on=["tx_id", "code"], how="left")
    table["self_reference"] = table["self_reference"].fillna(False).astype(bool)
    del marker

    input_rep = (
        input_pairs.sort_values(["tx_id", "code"], kind="stable")
        .groupby("tx_id", sort=False)["code"]
        .agg(["first", "size"])
        .rename(columns={"first": "input_rep", "size": "n_inputs"})
    )
    table = table.merge(input_rep, on="tx_id", how="left")
    del input_pairs, input_rep

    has_inputs = table["input_rep"].notna()
    n_skipped_no_inputs = int(
        table.loc[~has_inputs, "tx_id"].nunique()
    )
    table = table[has_inputs].copy()
    table["input_rep"] = table["input_rep"].astype(np.int32)
    table["n_inputs"] = table["n_inputs"].astype(np.int32)

    table["confidence"] = score(table, weights)
    table = table.sort_values(
        ["confidence", "tx_id", "code"], ascending=[False, True, True]
    ).reset_index(drop=True)

    return ChangeCandidates(
        table=table,
        weights=weights,
        n_transactions_with_outputs=n_tx_with_outputs,
        n_two_output=n_two_output,
        n_skipped_output_count=n_tx_with_outputs - n_two_output,
        n_skipped_no_inputs=n_skipped_no_inputs,
        n_unmapped_outputs=n_unmapped,
        features_used=use_features,
    )


def score(table: pd.DataFrame, weights: ChangeWeights | None = None) -> np.ndarray:
    """Combine the boolean signals into a confidence in [0, 1].

    Self-reference short-circuits: an output that is also an input to its own
    transaction is change regardless of what the other signals say. Otherwise
    the contributions add, which keeps the score readable - you can always say
    which signals produced a given number.
    """
    weights = weights or ChangeWeights()
    additive = (
        table["one_fresh"].to_numpy() * weights.one_fresh
        + table["fresher_first_block"].to_numpy() * weights.fresher_first_block
        + table["peer_reused"].to_numpy() * weights.peer_reused
    )
    confidence = np.where(
        table["self_reference"].to_numpy(), weights.self_reference, additive
    )
    return np.clip(confidence, 0.0, 1.0)


# ---- the merge decision (yours) ---------------------------------------


def select_change_rows(
    candidates: ChangeCandidates, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """Choose which scored candidates are accepted as change outputs.

    This is the merge decision, and it is deliberately left unimplemented.
    Everything upstream only scores; nothing upstream commits to a merge.

    The contract the rest of the pipeline relies on:

    * Keep only rows with ``confidence >= threshold``. Below it, the evidence
      is too weak to risk a false merge.
    * **At most one change output per transaction.** A transaction has one
      change address, not two. Where both outputs clear the threshold, keep
      the higher-scoring one; where they tie, keep neither - a tie means the
      signals cannot distinguish them, and guessing is exactly how cluster
      collapse starts.
    * Never select a row whose ``code`` equals its ``input_rep``; that edge
      would be a self-loop and merges nothing.
    * Return a subset of ``candidates.table`` with its columns intact, so the
      confidence of every accepted merge stays reportable.

    Args:
        candidates: Scored candidates from :func:`build_candidates`.
        threshold: Minimum confidence to accept.

    Returns:
        The accepted rows, a subset of ``candidates.table``.
    """
    table = candidates.table
    if table is None or len(table) == 0:
        return table.iloc[0:0] if table is not None else pd.DataFrame()

    # 1. Threshold. Everything below this is evidence we decline to act on.
    eligible = table[table["confidence"] >= threshold]
    if eligible.empty:
        return eligible

    # 2. One change output per transaction: keep each transaction's best row.
    best = eligible.groupby("tx_id")["confidence"].transform("max")
    winners = eligible[eligible["confidence"] == best]

    # 3. Ties are dropped, not broken. If both outputs score identically the
    #    signals cannot tell them apart, and picking one at random is how a
    #    single bad guess chains into a super-cluster.
    winners = winners[winners.groupby("tx_id")["code"].transform("size") == 1]

    # 4. Discard self-loops last, not first. The winner is the identified
    #    change output; if it happens to equal the input anchor the merge is a
    #    no-op and the transaction contributes nothing. Filtering these before
    #    step 2 would instead promote the runner-up, which asserts that the
    #    *other* output is change when the self-referencing one demonstrably
    #    is - a false merge dressed up as a saving.
    return winners[winners["code"] != winners["input_rep"]]


# ---- built on top of the decision -------------------------------------


def edges_from_selection(selected: pd.DataFrame) -> np.ndarray:
    """Turn accepted change rows into (m, 2) int32 merge edges.

    Each edge links the change output to one input address of the same
    transaction. Any input serves as the anchor: co-spend has already unioned
    all inputs of a transaction, so they share a component regardless.
    """
    if selected is None or len(selected) == 0:
        return np.empty((0, 2), dtype=np.int32)
    edges = np.empty((len(selected), 2), dtype=np.int32)
    edges[:, 0] = selected["input_rep"].to_numpy(dtype=np.int32)
    edges[:, 1] = selected["code"].to_numpy(dtype=np.int32)
    return edges


def signal_breakdown(table: pd.DataFrame) -> pd.DataFrame:
    """Count how often each signal fires, and its mean confidence."""
    rows = []
    for signal in ("self_reference", "one_fresh", "fresher_first_block", "peer_reused"):
        mask = table[signal].to_numpy()
        rows.append(
            {
                "signal": signal,
                "fired": int(mask.sum()),
                "share": float(mask.mean()) if len(table) else 0.0,
                "mean_confidence": (
                    float(table.loc[mask, "confidence"].mean()) if mask.any() else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)
