"""EXPERIMENT exp18_temporal_stability.  Pre-registered before running.

Problem: fold nAP ranges 0.24-0.86 (sd 0.173). exp17 traced the worst folds'
address-level P@100 collapse (0.36 at t39-40, 0.58 at t17-18) to a few licit
transactions each filling 10-29 of the top-100 slots with identical rows.

Candidates (all LightGBM, v2 configuration, same 12 folds, seed fixed):
  B  baseline v2
  R  recency weight: sample_weight = 0.5 ** ((train_end - t) / 8)
  W  sliding window: train on the last 16 timesteps only
  N  drift-robust ranks: per-timestep percentile rank of fee, fee_ratio,
     total_input_amount, input_amount_mean, output_amount_mean,
     btc_recv_total_asof_t, net_flow_asof_t (replacing the raw columns)
  T  transaction weight: sample_weight = 1 / (rows sharing the txid)
  D  B's scores, queue diversified: at most 3 addresses per txid are
     ranked before every address beyond the cap (a ranking POLICY)

Decision rule, fixed now:
  A model candidate (R, W, N, T) is MORE STABLE than B iff
    (1) fold-nAP sd < B's, AND
    (2) worst-fold nAP > B's, AND
    (3) the Holm-corrected paired verdict on nAP does not favour B.
  D is evaluated on address-level P@K / R@K only (it reorders, it does not
  re-score); it is adopted iff its mean AND worst-fold address P@100 exceed
  B's without lowering mean address R@500.
Secondary metrics reported for all: address P@100, R@100, R@500, R@1000
(mean and min), tx-weighted nAP.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "reproduction"))

from obsidianchain.ml import protocol  # noqa: E402
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS  # noqa: E402
from train_ps_model_v2 import _model, load_development, transaction_weighted_nap  # noqa: E402

OUT = ROOT / "research/autoresearch_2026_09_23/results/exp18_temporal_stability.json"
DRIFTING = ["fee", "fee_ratio", "total_input_amount", "input_amount_mean", "output_amount_mean",
            "btc_recv_total_asof_t", "net_flow_asof_t"]
HALF_LIFE = 8
WINDOW = 16
CAP = 3


def diversified_order(txids: np.ndarray, scores: np.ndarray, cap: int = CAP) -> np.ndarray:
    """Descending score, but the (cap+1)-th and later address of a txid go
    after every address within its txid's cap."""
    order = np.argsort(-scores, kind="stable")
    seen: dict = {}
    first, rest = [], []
    for i in order:
        n = seen.get(txids[i], 0)
        (first if n < cap else rest).append(i)
        seen[txids[i]] = n + 1
    return np.array(first + rest)


def addr_metrics(y: np.ndarray, order: np.ndarray) -> dict:
    out = {}
    total = max(int(y.sum()), 1)
    for k in (100, 500, 1000):
        tp = int(y[order[:k]].sum())
        out[f"P@{k}"] = tp / min(k, len(y))
        out[f"R@{k}"] = tp / total
    return out


def main() -> None:
    dev = load_development()
    base = list(CORE_PS_FEATURE_COLUMNS)
    for c in DRIFTING:
        dev[c + "_rank"] = dev.groupby("last_t")[c].rank(pct=True)
    ranked = [c for c in base if c not in DRIFTING] + [c + "_rank" for c in DRIFTING]
    dev["txsize"] = dev.txid.map(dev.txid.value_counts())

    rows = {k: [] for k in "BRWNTD"}
    for fold in protocol.rolling_origin_folds():
        te = fold.train_end
        tr = dev[dev.last_t <= te]
        ev = dev[(dev.last_t >= fold.eval_start) & (dev.last_t <= fold.eval_end)]
        y, tx = ev.y.to_numpy(), ev.txid.to_numpy()
        runs = {
            "B": (tr, base, None),
            "R": (tr, base, 0.5 ** ((te - tr.last_t) / HALF_LIFE)),
            "W": (tr[tr.last_t > te - WINDOW], base, None),
            "N": (tr, ranked, None),
            "T": (tr, base, 1.0 / tr.txsize),
        }
        for name, (train, feats, w) in runs.items():
            s = _model().fit(train[feats], train.y, sample_weight=w).predict_proba(ev[feats])[:, 1]
            order = np.argsort(-s, kind="stable")
            rows[name].append({"fold": fold.label,
                               "nap": protocol.normalised_average_precision(y, s)[0],
                               "txw": transaction_weighted_nap(y, s, tx), **addr_metrics(y, order)})
            if name == "B":
                d_order = diversified_order(tx, s)
                d_score = np.empty(len(s)); d_score[d_order] = -np.arange(len(s))
                rows["D"].append({"fold": fold.label,
                                  "nap": protocol.normalised_average_precision(y, d_score)[0],
                                  "txw": transaction_weighted_nap(y, d_score, tx), **addr_metrics(y, d_order)})
        print(fold.label, " ".join(f"{k}:{rows[k][-1]['nap']:.3f}/{rows[k][-1]['P@100']:.2f}" for k in rows), flush=True)

    frames = {k: pd.DataFrame(v) for k, v in rows.items()}
    b = frames["B"]
    raw_p = {}
    verdicts = {}
    for k in "RWNT":
        v = protocol.paired_verdict((frames[k].nap - b.nap).to_numpy(), a=k, b="B")
        verdicts[k] = v
        raw_p[k] = v["p_value"]
    holm = protocol.holm(raw_p)
    summary = {}
    for k, f in frames.items():
        summary[k] = {m: {"mean": round(float(f[m].mean()), 4), "sd": round(float(f[m].std(ddof=1)), 4),
                          "min": round(float(f[m].min()), 4)} for m in ("nap", "txw", "P@100", "R@100", "R@500", "R@1000")}
    decisions = {}
    for k in "RWNT":
        s, sb = summary[k]["nap"], summary["B"]["nap"]
        favours_b = verdicts[k].get("favours") == "B" and holm[k] < protocol.ALPHA
        decisions[k] = {
            "sd_lower": s["sd"] < sb["sd"], "worst_higher": s["min"] > sb["min"],
            "holm": holm.get(k), "mean_diff": round(float(verdicts[k]["mean_difference"]), 4),
            "MORE_STABLE": bool(s["sd"] < sb["sd"] and s["min"] > sb["min"] and not favours_b),
        }
    d, bb = summary["D"], summary["B"]
    decisions["D"] = {"ADOPT": bool(d["P@100"]["mean"] > bb["P@100"]["mean"] and d["P@100"]["min"] > bb["P@100"]["min"]
                                    and d["R@500"]["mean"] >= bb["R@500"]["mean"])}
    for k in summary:
        s = summary[k]
        print(f"{k}: nAP {s['nap']['mean']:.3f} sd {s['nap']['sd']:.3f} min {s['nap']['min']:.3f} | "
              f"P@100 {s['P@100']['mean']:.2f} (min {s['P@100']['min']:.2f}) R@100 {s['R@100']['mean']:.3f} "
              f"R@500 {s['R@500']['mean']:.3f} R@1000 {s['R@1000']['mean']:.3f} | txw {s['txw']['mean']:.3f}  "
              f"{decisions.get(k, '')}")
    OUT.write_text(json.dumps({"experiment_id": "exp18_temporal_stability", "design": __doc__,
                               "per_fold": rows, "summary": summary, "verdicts": verdicts,
                               "holm": holm, "decisions": decisions}, indent=2, default=str))


if __name__ == "__main__":
    main()
