"""Graph embeddings of addresses, for entity-link SUGGESTIONS.

Problem-statement focus area "Entity Clustering: group wallets likely owned
by one entity using common-input-ownership + graph embeddings".

Common-input ownership (``cluster/``) merges addresses that co-spend. It is
precise and misses every same-entity pair that never co-spends - on
Elliptic++ its structural recall ceiling is 32.7% (HANDOFF §8). An
embedding looks at a different relation: WHO an address transacts with. Two
addresses of one wallet tend to pay the same merchants and be paid by the
same sources even when they never appear together as inputs.

Construction
------------
* A counterparty profile per address from money flow (payer -> payee): the
  payees it sent to and the payers it received from, as two blocks of one
  sparse row.
* TF-IDF weighting, so a counterparty everyone uses (an exchange hot wallet)
  says little and a rare shared counterparty says a lot.
* Truncated SVD to ``dim`` components, then L2-normalised rows, so cosine
  similarity is a dot product.

How it is used, and how it is NOT
---------------------------------
Similar addresses in DIFFERENT co-spend clusters become ``SUGGESTED_LINK``
items for an analyst. They are never merged automatically: a false merge is
permanent and propagates (HANDOFF §1), and a similarity score is a much
weaker kind of evidence than co-spending. SVD coordinates are not comparable
between runs (sign and rotation are arbitrary), so embeddings are never a
model feature - only within-run similarities are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

DEFAULT_DIM = 32

#: Counterparties with more distinct partners than this are dropped: a
#: counterparty shared with thousands of addresses links strangers.
DEFAULT_MAX_COUNTERPARTY_DEGREE = 1000

#: Default similarity at or above which a cross-cluster pair is suggested.
#: From the Elliptic++ labelled-entity evaluation (exp16), with both support
#: rules applied: among cross-cluster pairs of eligible labelled addresses
#: (base rate 1.5% same-entity), cosine >= 0.90 at dim 32 gave 132
#: suggestions at 12% precision (8x the base rate) and 8% recall; AUC 0.61.
#: 0.99 gave 12 suggestions and none correct. An earlier run without the
#: transaction rule reported 80% precision at 0.99; that was 85 payout
#: addresses of one mining pool, each seen in ONE transaction - the artefact
#: MIN_TRANSACTIONS now removes. This is a weak signal and is presented as one.
DEFAULT_SUGGEST_THRESHOLD = 0.90

#: Stated on every suggestion, because the number alone invites a merge.
SUGGESTION_CAVEAT = (
    "Weak hint. Counterparty-profile similarity; measured on labelled Elliptic++ "
    "entities at this threshold, about 1 suggestion in 8 was the same owner "
    "(12% precision, 8x chance). Never merged automatically; verify before linking."
)

#: Below this many addresses, or this many SVD components, cosine similarity
#: is degenerate - with one or two components nearly every pair reads ~1.0 -
#: so no suggestion is produced and the result says why.
MIN_ADDRESSES = 200
MIN_DIM = 8

#: Distinct (non-hub) counterparties an address needs before it may be
#: suggested. With one counterparty, every payee of the same payer has an
#: identical profile and reads cosine 1.0 - thirty recipients of one batch
#: payout would all be "the same entity".
MIN_SUPPORT = 2

#: Transactions an address must appear in before it may be suggested. Every
#: output of one transaction has the same payers, so a profile built from a
#: single transaction makes all of that transaction's payees look like one
#: owner. On world v2 that produced thousands of false "same entity" pairs.
MIN_TRANSACTIONS = 2



@dataclass
class AddressEmbedding:
    addresses: pd.Index
    vectors: np.ndarray
    dim: int
    explained_variance: float
    dropped_hub_counterparties: int
    support: np.ndarray | None = None
    """Distinct non-hub counterparties per address."""

    def vector(self, address: str) -> np.ndarray | None:
        i = self.addresses.get_indexer([address])[0]
        return None if i < 0 else self.vectors[i]

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.vector(a), self.vector(b)
        if va is None or vb is None:
            return float("nan")
        return float(va @ vb)


def flows_from_capture(frame: pd.DataFrame) -> pd.DataFrame:
    """``(payer, payee)`` pairs from a canonical capture: every input address
    to every output address of the same transaction."""
    rows: list[tuple[str, str]] = []
    for _, row in frame.drop_duplicates(subset="txid").iterrows():
        ins = row.get("input_addresses")
        outs = row.get("output_addresses")
        if not isinstance(ins, (list, tuple)) or not isinstance(outs, (list, tuple)):
            continue
        ins = [str(a).strip() for a in ins if a is not None and not pd.isna(a) and str(a).strip()]
        outs = [str(a).strip() for a in outs if a is not None and not pd.isna(a) and str(a).strip()]
        rows.extend((i, o) for i in ins for o in outs if i != o)
    return pd.DataFrame(rows, columns=["payer", "payee"]).drop_duplicates()


def embed(flows: pd.DataFrame, *, dim: int = DEFAULT_DIM,
          max_counterparty_degree: int = DEFAULT_MAX_COUNTERPARTY_DEGREE,
          seed: int = 0) -> AddressEmbedding:
    """Counterparty-profile SVD embedding over ``flows`` (payer, payee)."""
    flows = flows[["payer", "payee"]].astype(str).drop_duplicates()
    addresses = pd.Index(sorted(set(flows.payer) | set(flows.payee)))
    n = len(addresses)
    p = addresses.get_indexer(flows.payer)
    q = addresses.get_indexer(flows.payee)
    # Block 1: row = payer, column = payee it paid. Block 2: row = payee,
    # column = payer it was paid by (offset by n).
    rows = np.concatenate([p, q])
    cols = np.concatenate([q, p + n])
    x = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, 2 * n))
    x.data[:] = 1.0

    df = np.asarray((x > 0).sum(axis=0)).ravel()
    keep = (df > 0) & (df <= max_counterparty_degree)
    dropped = int(((df > max_counterparty_degree)).sum())
    x = x[:, np.flatnonzero(keep)]
    support = np.asarray((x > 0).sum(axis=1)).ravel()
    idf = np.log((1 + n) / (1 + df[keep])) + 1.0
    x = x @ sparse.diags(idf)

    k = int(max(1, min(dim, min(x.shape) - 1)))
    if x.shape[1] == 0 or k < 1 or n < 3:
        return AddressEmbedding(addresses, np.zeros((n, 1)), 0, 0.0, dropped, support)
    svd = TruncatedSVD(n_components=k, algorithm="randomized", random_state=seed)
    z = svd.fit_transform(x)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = np.divide(z, norms, out=np.zeros_like(z), where=norms > 0)
    return AddressEmbedding(addresses, z, k, float(svd.explained_variance_ratio_.sum()), dropped, support)


@dataclass
class LinkSuggestions:
    pairs: list[dict[str, Any]] = field(default_factory=list)
    threshold: float = DEFAULT_SUGGEST_THRESHOLD
    embedding_dim: int = 0
    status: str = "RUN"
    """'RUN', or 'GRAPH_TOO_SMALL' when similarity would be degenerate."""

    def summary(self) -> dict[str, Any]:
        return {"status": self.status, "suggested_links": len(self.pairs), "threshold": self.threshold,
                "embedding_dim": self.embedding_dim,
                "meaning": "analyst suggestions only; nothing is merged",
                "caveat": SUGGESTION_CAVEAT}

    def for_cluster(self, cluster_id: str) -> list[dict[str, Any]]:
        return [p for p in self.pairs if cluster_id in (p["cluster_a"], p["cluster_b"])]


def suggest_links(embedding: AddressEmbedding, address_to_cluster: dict[str, str], *,
                  threshold: float = DEFAULT_SUGGEST_THRESHOLD,
                  top_k: int = 3, max_pairs: int = 500,
                  tx_counts: dict[str, int] | None = None) -> LinkSuggestions:
    """Top cross-cluster neighbours per address above ``threshold``.

    Brute force over the run's addresses, in blocks, so memory stays bounded.
    Pairs inside one co-spend cluster are skipped: that link is already made
    by stronger evidence.
    """
    out = LinkSuggestions(threshold=threshold, embedding_dim=embedding.dim)
    if embedding.dim < MIN_DIM or len(embedding.addresses) < MIN_ADDRESSES:
        out.status = "GRAPH_TOO_SMALL"
        return out
    addrs = list(embedding.addresses)
    clusters = np.array([address_to_cluster.get(a, f"singleton:{a}") for a in addrs])
    z = embedding.vectors
    eligible = (embedding.support >= MIN_SUPPORT) if embedding.support is not None else np.ones(len(addrs), bool)
    if tx_counts is not None:
        eligible &= np.array([tx_counts.get(a, 0) >= MIN_TRANSACTIONS for a in addrs])
    seen: set[tuple[int, int]] = set()
    block = 2048
    for start in range(0, len(addrs), block):
        sims = z[start:start + block] @ z.T
        for local, row in enumerate(sims):
            i = start + local
            if not eligible[i]:
                continue
            row[clusters == clusters[i]] = -np.inf
            row[~eligible] = -np.inf
            cand = np.argpartition(-row, min(top_k, len(row) - 1))[:top_k]
            for j in cand:
                if row[j] < threshold:
                    continue
                key = (min(i, int(j)), max(i, int(j)))
                if key in seen:
                    continue
                seen.add(key)
                out.pairs.append({
                    "address_a": addrs[key[0]], "address_b": addrs[key[1]],
                    "cluster_a": str(clusters[key[0]]), "cluster_b": str(clusters[key[1]]),
                    "similarity": round(float(row[j]), 4),
                })
    out.pairs.sort(key=lambda p: -p["similarity"])
    del out.pairs[max_pairs:]
    return out
