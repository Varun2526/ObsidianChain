"""The evaluation framework: one place that defines every reported metric.

Every function takes plain arrays, so the same definitions serve research
folds, the sealed holdout, shadow runs and the delayed-label loop. Nothing
here fits a model or chooses a threshold.

Ranking (address level unless stated)
    ap, nap, roc_auc, P@K, R@K, nR@K (recall over its ceiling min(1, K/pos)),
    R-precision, NDCG@K (binary gains), lift@K = P@K / prevalence,
    FP@K and FN@K (positives outside the top K).
Transaction level
    one entry per txid: positive if any scored address is, score = max.
Calibration (on a probability, not the ranking score)
    Brier, ECE (equal-mass deciles), reliability table, calibration slope and
    intercept (logistic regression of y on logit(p); 1 and 0 are perfect).
Slices
    the same scorecard on caller-supplied boolean masks; slices with no
    positive or no negative are reported as not measurable, never as zero.
Aggregation over folds
    mean, median, sd, min, max and a t-based 95% CI of the mean.

Ceilings: R@K cannot exceed K/positives, and P@K cannot exceed positives/K.
The ceilings are reported beside the values so no target is set above them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_KS = (10, 25, 50, 100, 250, 500, 1000)


def _order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")


def ranking_metrics(y, scores, ks=DEFAULT_KS) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    n, pos = len(y), int(y.sum())
    prevalence = pos / n if n else float("nan")
    out: dict[str, Any] = {"n": n, "positives": pos, "prevalence": prevalence}
    if pos == 0 or pos == n:
        out["measurable"] = False
        return out
    out["measurable"] = True
    ap = float(average_precision_score(y, s))
    out.update(ap=ap, nap=(ap - prevalence) / (1 - prevalence), roc_auc=float(roc_auc_score(y, s)))
    order = _order(s)
    ranked = y[order]
    cum = np.cumsum(ranked)
    out["r_precision"] = float(cum[pos - 1] / pos)
    discounts = 1.0 / np.log2(np.arange(2, n + 2))
    for k in ks:
        kk = min(k, n)
        tp = int(cum[kk - 1])
        ideal = discounts[:min(kk, pos)].sum()
        out[f"P@{k}"] = tp / kk
        out[f"R@{k}"] = tp / pos
        out[f"nR@{k}"] = (tp / pos) / min(1.0, kk / pos)
        out[f"lift@{k}"] = (tp / kk) / prevalence
        out[f"NDCG@{k}"] = float((ranked[:kk] * discounts[:kk]).sum() / ideal) if ideal else float("nan")
        out[f"FP@{k}"] = kk - tp
        out[f"FN@{k}"] = pos - tp
        out[f"ceiling_R@{k}"] = min(1.0, kk / pos)
        out[f"ceiling_P@{k}"] = min(1.0, pos / kk)
    return out


def transaction_metrics(y, scores, txids, ks=DEFAULT_KS) -> dict[str, Any]:
    frame = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(scores), "tx": np.asarray(txids)})
    per_tx = frame.groupby("tx").agg(y=("y", "max"), s=("s", "max"))
    return ranking_metrics(per_tx.y.to_numpy(), per_tx.s.to_numpy(), ks)


def transaction_weighted_nap(y, scores, txids) -> float:
    """nAP where each transaction carries total weight one."""
    y = np.asarray(y)
    tx = pd.Series(np.asarray(txids))
    w = 1.0 / tx.map(tx.value_counts()).to_numpy()
    prevalence = float(np.average(y, weights=w))
    if prevalence in (0.0, 1.0):
        return float("nan")
    return (float(average_precision_score(y, scores, sample_weight=w)) - prevalence) / (1 - prevalence)


def calibration_metrics(y, prob, bins: int = 10) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    table, ece = [], 0.0
    for b in range(max(len(edges) - 1, 1)):
        m = idx == b
        if not m.any():
            continue
        table.append({"bin": b, "n": int(m.sum()), "mean_predicted": float(p[m].mean()),
                      "observed_rate": float(y[m].mean())})
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    out = {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece), "reliability": table,
           "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())}
    if 0 < y.sum() < len(y):
        from sklearn.linear_model import LogisticRegression
        z = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6).fit(z, y)
        out["calibration_slope"] = float(lr.coef_[0][0])
        out["calibration_intercept"] = float(lr.intercept_[0])
    return out


def slice_metrics(y, scores, masks: dict[str, np.ndarray], ks=(50, 100, 500)) -> dict[str, Any]:
    out = {}
    y = np.asarray(y)
    s = np.asarray(scores)
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        m = ranking_metrics(y[mask], s[mask], ks) if mask.any() else {"n": 0, "measurable": False}
        m["share_of_rows"] = float(mask.mean())
        out[name] = m
    return out


def standard_slices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """The slices every scorecard reports, from PS-native feature columns."""
    n_txs = frame["n_txs_asof_t"].to_numpy()
    value = frame["total_input_amount"].to_numpy()
    q = np.nanquantile(value, [0.25, 0.75]) if len(value) else [0, 0]
    upstream = frame["upstream_funded_share"].to_numpy() if "upstream_funded_share" in frame else np.zeros(len(frame))
    return {
        "first_seen": n_txs == 0,
        "known_address": n_txs > 0,
        "high_history": n_txs >= 5,
        "low_volume_q1": value <= q[0],
        "high_volume_q4": value >= q[1],
        "no_upstream_funding": upstream == 0,
        "has_upstream_funding": upstream > 0,
        "sender": frame["addr_is_sender"].to_numpy() == 1,
        "receiver_only": frame["addr_is_sender"].to_numpy() == 0,
    }


def aggregate(values) -> dict[str, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not v.size:
        return {"n": 0}
    out = {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
           "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "min": float(v.min()), "max": float(v.max())}
    if v.size > 1:
        half = stats.t.ppf(0.975, v.size - 1) * out["sd"] / np.sqrt(v.size)
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def temporal_trend(values) -> dict[str, float]:
    """OLS slope of a per-fold metric over fold index, with its p-value."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return {"slope": float("nan"), "p_value": float("nan")}
    res = stats.linregress(np.arange(len(v))[ok], v[ok])
    return {"slope_per_fold": float(res.slope), "p_value": float(res.pvalue)}


def full_scorecard(y, scores, txids, prob=None, frame: pd.DataFrame | None = None,
                   ks=DEFAULT_KS) -> dict[str, Any]:
    """Everything above for one evaluation window."""
    out = {"address": ranking_metrics(y, scores, ks),
           "transaction": transaction_metrics(y, scores, txids, ks),
           "tx_weighted_nap": transaction_weighted_nap(y, scores, txids)}
    if prob is not None:
        out["calibration"] = calibration_metrics(y, prob)
    if frame is not None:
        out["slices"] = slice_metrics(y, scores, standard_slices(frame))
    return out
