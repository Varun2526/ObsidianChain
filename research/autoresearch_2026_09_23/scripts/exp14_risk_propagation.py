"""EXPERIMENT exp14_risk_propagation.

Question: does personalized-PageRank risk propagation from seed illicit
wallets (``ml/propagation.py``) rank development-period addresses, and does
it add to the v2 LightGBM model?

Design (fixed before running):
* 12 rolling-origin folds from ``ml/protocol.py``. Rows are the v3
  development parquets; ``last_t`` is each row's own timestep.
* Graph: every Elliptic++ address-transaction edge (AddrTx inputs, TxAddr
  outputs) with transaction timestep <= the fold's eval end. Structure is
  observable at inference time; labels are not.
* Seeds: development addresses with y = 1 and last_t <= train_end. No label
  from the evaluation window is ever a seed.
* Candidates, all scored on the same eval rows:
    A  LightGBM v2 configuration alone
    P  propagated risk alone
    S  stack: logistic regression on [logit(A), log1p(P scaled)], fitted on a
       calibration window (t in (train_end-2, train_end]) where both inputs
       are produced exactly as at inference - LightGBM trained on
       t <= train_end-2, seeds from t <= train_end-2.
* Primary metric nAP; secondary transaction-weighted nAP. Paired verdicts
  S vs A through ``protocol.paired_verdict``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "reproduction"))

from obsidianchain.ml import protocol  # noqa: E402
from obsidianchain.ml.propagation import propagate  # noqa: E402
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS  # noqa: E402
from train_ps_model_v2 import (  # noqa: E402
    _logit, _model, load_development, transaction_weighted_nap,
)

OUT = ROOT / "research/autoresearch_2026_09_23/results/exp14_risk_propagation.json"


def load_edges() -> pd.DataFrame:
    raw = ROOT / "data" / "raw"
    steps = pd.read_csv(raw / "txs_features.csv", usecols=["txId", "Time step"])
    ins = pd.read_csv(raw / "AddrTx_edgelist.csv").rename(columns={"input_address": "address"})
    outs = pd.read_csv(raw / "TxAddr_edgelist.csv").rename(columns={"output_address": "address"})
    edges = pd.concat([ins[["address", "txId"]], outs[["address", "txId"]]], ignore_index=True)
    edges = edges.merge(steps, on="txId")
    # Dataset txids are float strings ("12660686.0"); match that spelling.
    edges["txid"] = edges["txId"].astype(float).astype(str)
    return edges[["address", "txid", "Time step"]].rename(columns={"Time step": "step"})


def prop_scores(edges, dev, graph_end, seed_end, targets) -> np.ndarray:
    seeds = dev.loc[(dev.y == 1) & (dev.last_t <= seed_end), "address"]
    res = propagate(edges[edges.step <= graph_end], seeds)
    return res.scores.reindex(targets).fillna(0.0).to_numpy()


def stack_features(lgbm_raw, prop) -> np.ndarray:
    return np.column_stack([_logit(lgbm_raw), np.log1p(1000 * prop)])


def main() -> None:
    t0 = time.time()
    dev = load_development()
    edges = load_edges()
    features = list(CORE_PS_FEATURE_COLUMNS)
    print(f"dev rows {len(dev):,}; edges {len(edges):,}; loaded in {time.time()-t0:.0f}s")

    rows = []
    for fold in protocol.rolling_origin_folds():
        te, es, ee = fold.train_end, fold.eval_start, fold.eval_end
        train = dev[dev.last_t <= te]
        ev = dev[(dev.last_t >= es) & (dev.last_t <= ee)]

        a_eval = _model().fit(train[features], train.y).predict_proba(ev[features])[:, 1]
        p_eval = prop_scores(edges, dev, ee, te, ev.address)

        # Stacker trained on a window that mimics inference exactly.
        cal_end = te - 2
        fit = dev[dev.last_t <= cal_end]
        cal = dev[(dev.last_t > cal_end) & (dev.last_t <= te)]
        a_cal = _model().fit(fit[features], fit.y).predict_proba(cal[features])[:, 1]
        p_cal = prop_scores(edges, dev, te, cal_end, cal.address)
        stacker = LogisticRegression().fit(stack_features(a_cal, p_cal), cal.y)
        s_eval = stacker.predict_proba(stack_features(a_eval, p_eval))[:, 1]

        r = {"fold": fold.label, "prevalence": round(float(ev.y.mean()), 4),
             "prop_coverage": round(float((p_eval > 0).mean()), 4),
             "prop_coverage_positives": round(float((p_eval[ev.y.to_numpy() == 1] > 0).mean()), 4),
             "stacker_coef": [round(float(c), 3) for c in stacker.coef_[0]]}
        for name, s in (("A", a_eval), ("P", p_eval), ("S", s_eval)):
            r[f"nap_{name}"] = round(protocol.normalised_average_precision(ev.y, s)[0], 4)
            r[f"txw_{name}"] = round(transaction_weighted_nap(ev.y, s, ev.txid), 4)
        rows.append(r)
        print(f"{fold.label:>16}  A {r['nap_A']:.3f}  P {r['nap_P']:.3f}  S {r['nap_S']:.3f}  "
              f"| txw A {r['txw_A']:.3f} S {r['txw_S']:.3f}  cov {r['prop_coverage']:.2f} "
              f"(pos {r['prop_coverage_positives']:.2f})", flush=True)

    frame = pd.DataFrame(rows)
    verdicts = {
        "S_vs_A_nap": protocol.paired_verdict((frame.nap_S - frame.nap_A).to_numpy(), a="stack", b="lgbm"),
        "S_vs_A_txw": protocol.paired_verdict((frame.txw_S - frame.txw_A).to_numpy(), a="stack", b="lgbm"),
        "P_vs_A_nap": protocol.paired_verdict((frame.nap_P - frame.nap_A).to_numpy(), a="prop", b="lgbm"),
    }
    summary = {k: round(float(frame[k].mean()), 4) for k in frame.columns if k.startswith(("nap_", "txw_", "prop_cov"))}
    print(json.dumps(summary, indent=1))
    for k, v in verdicts.items():
        print(k, v["verdict"], v.get("favours"), round(v["mean_difference"], 4), round(v["p_value"], 4))
    OUT.write_text(json.dumps({
        "experiment_id": "exp14_risk_propagation",
        "design": __doc__,
        "folds": rows, "summary": summary, "verdicts": verdicts,
        "runtime_seconds": round(time.time() - t0, 1),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
