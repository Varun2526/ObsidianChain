"""EXPERIMENT exp16_embedding_entity_links.  PRODUCTION data (Elliptic++).

Question: among labelled Elliptic++ addresses, does counterparty-profile
embedding similarity (``ml/embeddings.py``) identify same-entity pairs that
common-input clustering leaves in different clusters?

Design (fixed before running):
* Embedding over the full Elliptic++ money-flow graph (AddrAddr_edgelist:
  payer -> payee). Labels are never an input.
* Evaluation set: the 361 addresses in ``entity_labels.csv`` with
  in_elliptic = True (117 entities), all pairs among them.
* Pairs already in one co-spend cluster (Phase 1 ``address_clusters``) are
  REMOVED: the question is what the embedding adds beyond co-spend.
* Metrics on the remaining cross-cluster pairs: ROC-AUC of cosine for
  same-entity vs different-entity, precision at K (K = 25, 50, 100, 200)
  against the base rate, and precision/recall at similarity thresholds, used
  to set the production suggestion threshold.
* Support rule (added after the first run, which is kept in the log below):
  pairs where either address has fewer than ``MIN_SUPPORT`` distinct non-hub
  counterparties are reported separately, because such profiles are
  near-identical by construction (every payee of one payer).
* Transaction rule (added after world v2 showed payees of one transaction
  all reading as one owner): an address must appear in at least
  ``MIN_TRANSACTIONS`` transactions to be eligible; pairs failing it are
  counted with the unsupported ones.
* Two dimensions (16, 32) and one control: the same evaluation with entity
  labels permuted across addresses (expected AUC ~0.5).
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from obsidianchain.ml.embeddings import MIN_SUPPORT, MIN_TRANSACTIONS, embed

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/autoresearch_2026_09_23/results/exp16_embedding_entity_links.json"


def main() -> None:
    t0 = time.time()
    flows = pd.read_csv(ROOT / "data/raw/AddrAddr_edgelist.csv").rename(
        columns={"input_address": "payer", "output_address": "payee"})
    labels = pd.read_csv(ROOT / "data/processed/entity_labels.csv")
    labels = labels[labels.in_elliptic].drop_duplicates("address")
    clusters = pd.read_parquet(ROOT / "data/processed/address_clusters.parquet")
    cluster_of = dict(zip(clusters.address, clusters.cluster_id))
    ins = pd.read_csv(ROOT / "data/raw/AddrTx_edgelist.csv").rename(columns={"input_address": "address"})
    outs = pd.read_csv(ROOT / "data/raw/TxAddr_edgelist.csv").rename(columns={"output_address": "address"})
    tx_count = pd.concat([ins[["address", "txId"]], outs[["address", "txId"]]]).drop_duplicates() \
        .groupby("address").size()
    print(f"flows {len(flows):,}  labelled {len(labels)}  loaded {time.time()-t0:.0f}s")

    results = {}
    for dim in (16, 32):
        t1 = time.time()
        emb = embed(flows, dim=dim)
        lab = labels[labels.address.isin(emb.addresses)].reset_index(drop=True)
        idx = emb.addresses.get_indexer(lab.address)
        z = emb.vectors[idx]
        supported = (emb.support[idx] >= MIN_SUPPORT) & \
            (lab.address.map(tx_count).fillna(0).to_numpy() >= MIN_TRANSACTIONS)
        rows = []
        for i, j in itertools.combinations(range(len(lab)), 2):
            a, b = lab.address[i], lab.address[j]
            ca, cb = cluster_of.get(a, f"s:{a}"), cluster_of.get(b, f"s:{b}")
            if ca == cb:
                continue
            rows.append((i, j, float(z[i] @ z[j]), lab.entity_norm[i] == lab.entity_norm[j],
                         bool(supported[i] and supported[j])))
        all_pairs = pd.DataFrame(rows, columns=["i", "j", "sim", "same", "supported"])
        unsupported = all_pairs[~all_pairs.supported]
        pairs = all_pairs[all_pairs.supported].reset_index(drop=True)
        rng = np.random.default_rng(0)
        perm = rng.permutation(lab.entity_norm.to_numpy())
        pairs["same_permuted"] = perm[pairs.i] == perm[pairs.j]
        ranked = pairs.sort_values("sim", ascending=False)
        at_k = {k: round(float(ranked.same.head(k).mean()), 4) for k in (25, 50, 100, 200)}
        thresholds = {}
        for th in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
            sel = pairs.sim >= th
            thresholds[str(th)] = {
                "suggested": int(sel.sum()),
                "precision": round(float(pairs.same[sel].mean()), 4) if sel.any() else None,
                "recall": round(float(pairs.same[sel].sum() / max(pairs.same.sum(), 1)), 4),
            }
        top_entities = pairs[(pairs.sim >= 0.99) & pairs.same].i.map(lab.entity_norm).value_counts().to_dict()
        results[f"dim{dim}"] = {
            "unsupported_pairs_excluded": int(len(unsupported)),
            "unsupported_same_entity": int(unsupported.same.sum()),
            "same_entity_at_0.99_by_entity": top_entities,
            "labelled_in_graph": int(len(lab)),
            "cross_cluster_pairs": int(len(pairs)),
            "same_entity_cross_cluster_pairs": int(pairs.same.sum()),
            "base_rate": round(float(pairs.same.mean()), 5),
            "auc": round(float(roc_auc_score(pairs.same, pairs.sim)), 4),
            "auc_permuted_control": round(float(roc_auc_score(pairs.same_permuted, pairs.sim)), 4),
            "precision_at_k": at_k,
            "thresholds": thresholds,
            "explained_variance": round(emb.explained_variance, 4),
            "dropped_hub_counterparties": emb.dropped_hub_counterparties,
            "seconds": round(time.time() - t1, 1),
        }
        r = results[f"dim{dim}"]
        print(f"dim {dim}: pairs {r['cross_cluster_pairs']:,} (same {r['same_entity_cross_cluster_pairs']}, "
              f"base {r['base_rate']:.4f})  AUC {r['auc']:.3f}  control {r['auc_permuted_control']:.3f}  "
              f"P@k {at_k}  t={r['seconds']}s")
        for th, v in thresholds.items():
            print(f"    sim>={th}: n={v['suggested']:>6} precision={v['precision']} recall={v['recall']}")
    OUT.write_text(json.dumps({"experiment_id": "exp16_embedding_entity_links",
                               "provenance_type": "PRODUCTION", "design": __doc__,
                               "results": results}, indent=2))


if __name__ == "__main__":
    main()
