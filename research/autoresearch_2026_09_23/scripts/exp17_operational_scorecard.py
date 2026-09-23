"""EXPERIMENT exp17_operational_scorecard.

The investigator view of ps_native_v2: what a fixed workload of K alerts
buys, per rolling fold, instead of one ranking summary.

Per fold (the 12 protocol folds, v2 configuration including its 16-step
training window - see exp18 - raw score ranks):
* Address level: P@K, Recall@K, false positives among the top K, for
  K = 50, 100, 500, 1000. Recall denominator = all positive addresses in
  the window.
* Transaction level (one entry per txid, positive if ANY scored address in
  it is positive; score = max over its addresses): the same metrics. This is
  the lenient view the earlier P@100 = 0.88 used; both are reported.
* Calibration: Platt fitted exactly as in production - LightGBM trained on
  t <= train_end-2, scored on t in (train_end-2, train_end], Platt fitted on
  those scores - then applied to the window. Reliability table on deciles of
  the predicted probability, ECE, Brier.

Also a diagnosis of the worst fold (t39-40): concentration of positives in
transactions, cold-start share, and which behaviours of positives were
missed, against the same statistics on the other folds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "reproduction"))

from obsidianchain.ml import protocol  # noqa: E402
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS  # noqa: E402
from train_ps_model_v2 import _logit, _model, load_development, window  # noqa: E402

OUT = ROOT / "research/autoresearch_2026_09_23/results/exp17_operational_scorecard.json"
KS = (50, 100, 500, 1000)


def at_k(y: np.ndarray, s: np.ndarray, k: int) -> dict:
    order = np.argsort(-s, kind="stable")[:k]
    tp = int(y[order].sum())
    return {"k": int(min(k, len(y))), "precision": round(tp / max(len(order), 1), 4),
            "recall": round(tp / max(int(y.sum()), 1), 4), "tp": tp, "fp": int(len(order) - tp)}


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[list, float]:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    table, ece = [], 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        table.append({"bin": b, "n": int(m.sum()), "mean_predicted": round(float(p[m].mean()), 4),
                      "observed_rate": round(float(y[m].mean()), 4)})
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return table, round(float(ece), 4)


def main() -> None:
    dev = load_development()
    features = list(CORE_PS_FEATURE_COLUMNS)
    folds_out = []
    for fold in protocol.rolling_origin_folds():
        te = fold.train_end
        # Training rows exactly as the frozen v2 model: the recent window.
        train = window(dev, te)
        ev = dev[(dev.last_t >= fold.eval_start) & (dev.last_t <= fold.eval_end)].reset_index(drop=True)
        s = _model().fit(train[features], train.y).predict_proba(ev[features])[:, 1]
        y = ev.y.to_numpy()

        fit = window(dev, te - 2)
        cal = dev[(dev.last_t > te - 2) & (dev.last_t <= te)]
        sc = _model().fit(fit[features], fit.y).predict_proba(cal[features])[:, 1]
        platt = LogisticRegression().fit(_logit(sc).reshape(-1, 1), cal.y)
        p = platt.predict_proba(_logit(s).reshape(-1, 1))[:, 1]
        table, ece = reliability(y, p)

        tx = pd.DataFrame({"tx": ev.txid, "y": y, "s": s}).groupby("tx").agg(y=("y", "max"), s=("s", "max"))
        pos = ev[ev.y == 1]
        top_pos_tx_share = float(pos.txid.value_counts().head(5).sum() / max(len(pos), 1))
        order = np.argsort(-s)
        missed = ev.iloc[order[min(500, len(ev)):]]
        missed_pos = missed[missed.y == 1]
        folds_out.append({
            "fold": fold.label,
            "n_addresses": int(len(ev)), "n_positive_addresses": int(y.sum()),
            "n_transactions": int(len(tx)), "n_positive_transactions": int(tx.y.sum()),
            "prevalence": round(float(y.mean()), 4),
            "nap": round(protocol.normalised_average_precision(y, s)[0], 4),
            "address": {k: at_k(y, s, k) for k in KS},
            "transaction": {k: at_k(tx.y.to_numpy(), tx.s.to_numpy(), k) for k in KS},
            "calibration": {"brier_raw": round(float(brier_score_loss(y, s)), 4),
                            "brier_platt": round(float(brier_score_loss(y, p)), 4),
                            "ece_platt": ece, "calibration_window_prevalence": round(float(cal.y.mean()), 4),
                            "reliability": table},
            "diagnosis": {
                "share_of_positives_in_top5_txids": round(top_pos_tx_share, 4),
                "cold_start_share_positives": round(float((pos.n_txs_asof_t == 0).mean()), 4),
                "cold_start_share_all": round(float((ev.n_txs_asof_t == 0).mean()), 4),
                "positives_missed_beyond_top500": int(len(missed_pos)),
                "missed_positive_median_output_count": float(missed_pos.output_count.median()) if len(missed_pos) else None,
                "positive_median_output_count": float(pos.output_count.median()),
                "missed_positive_sender_share": round(float(missed_pos.addr_is_sender.mean()), 4) if len(missed_pos) else None,
            },
        })
        f = folds_out[-1]
        print(f"{f['fold']:>16} prev {f['prevalence']:.3f} nAP {f['nap']:.3f} | addr P@100 {f['address'][100]['precision']:.2f} "
              f"R@100 {f['address'][100]['recall']:.3f} P@500 {f['address'][500]['precision']:.2f} R@500 {f['address'][500]['recall']:.3f} "
              f"R@1000 {f['address'][1000]['recall']:.3f} | tx P@100 {f['transaction'][100]['precision']:.2f} "
              f"| ECE {ece:.3f}", flush=True)

    summary = {}
    for level in ("address", "transaction"):
        for k in KS:
            for m in ("precision", "recall", "fp"):
                vals = [f[level][k][m] for f in folds_out]
                summary[f"{level}_{m}@{k}"] = {"mean": round(float(np.mean(vals)), 4),
                                               "min": round(float(np.min(vals)), 4),
                                               "max": round(float(np.max(vals)), 4)}
    summary["ece_platt"] = {"mean": round(float(np.mean([f["calibration"]["ece_platt"] for f in folds_out])), 4)}
    summary["brier_platt"] = {"mean": round(float(np.mean([f["calibration"]["brier_platt"] for f in folds_out])), 4)}
    OUT.write_text(json.dumps({"experiment_id": "exp17_operational_scorecard", "design": __doc__,
                               "folds": folds_out, "summary": summary}, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if "@" in k and "fp" not in k}, indent=0))


if __name__ == "__main__":
    main()
