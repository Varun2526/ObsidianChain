"""Analysis-only threshold sweep for ps_native_v5.

No model, feature schema, registry, holdout, or deployment is modified.
Threshold selection uses ONLY the development validation set
(data/models/ps_native/datasets/validation.parquet, t35-41). The sealed
t42-49 holdout is scored only AFTER the threshold candidates are frozen.

Metrics are computed from the model's CALIBRATED probabilities
(PsNativeRiskModel.scores), which is the production decision signal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "reproduction"))

from obsidianchain.ml import evaluation, registry  # noqa: E402
from obsidianchain.ml.ps_model import PsNativeRiskModel  # noqa: E420

OUT = Path(__file__).resolve().parent
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
              0.50, 0.55, 0.60, 0.65, 0.70]
TOP_KS = (100, 500, 1000)


def threshold_metrics(y: np.ndarray, prob: np.ndarray, t: float) -> dict:
    pred = (prob >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n_pos_pred = int(pred.sum())
    precision = tp / n_pos_pred if n_pos_pred else float("nan")
    recall = tp / int(y.sum()) if y.sum() else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    bal_acc = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) if (tp + fn) and (tn + fp) else float("nan")
    return {
        "threshold": t,
        "predicted_positives": n_pos_pred,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "false_positive_rate": fp / max(tn + fp, 1),
    }


def topk_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    ranked = y[order]
    cum = np.cumsum(ranked)
    pos = int(y.sum())
    out = {}
    for k in TOP_KS:
        kk = min(k, len(y))
        tp = int(cum[kk - 1])
        out[f"P@{k}"] = tp / kk
        out[f"R@{k}"] = tp / pos
        out[f"FP@{k}"] = kk - tp
        out[f"FN@{k}"] = pos - tp
    return out


def main() -> None:
    reg = registry.Registry.open(ROOT / "data" / "models" / "ps_native")
    model = PsNativeRiskModel.load(reg.model_dir("ps_native_v5"))

    val = pd.read_parquet(ROOT / "data" / "models" / "ps_native" / "datasets" / "validation.parquet")
    y = val["y"].to_numpy(dtype=int)
    scores = model.raw_scores(val)            # ranking score
    prob = model.calibrate(scores)            # calibrated probability (decision signal)

    rows = [threshold_metrics(y, prob, t) for t in THRESHOLDS]
    sweep = pd.DataFrame(rows)
    topk = topk_metrics(y, scores)

    # ---- validation reference at the current production threshold 0.50 ----
    base = sweep[sweep["threshold"] == 0.50].iloc[0]

    out = {
        "validation_dataset": "data/models/ps_native/datasets/validation.parquet",
        "validation_rows": int(len(val)),
        "validation_positives": int(y.sum()),
        "validation_prevalence": float(y.mean()),
        "model_version": model.version,
        "model_sha256": model._sha256,
        "decision_signal": "calibrated probability (PsNativeRiskModel.calibrate)",
        "threshold_sweep": rows,
        "topk": topk,
        "base_threshold_0_50": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in base.items()},
    }
    (OUT / "validation_sweep.json").write_text(json.dumps(out, indent=2, default=str))
    print(sweep.to_string(index=False))
    print()
    print("TOP-K (ranked by raw score):")
    for k in TOP_KS:
        print(f"  P@{k}={topk[f'P@{k}']:.4f}  R@{k}={topk[f'R@{k}']:.4f}  "
              f"FP@{k}={topk[f'FP@{k}']}  FN@{k}={topk[f'FN@{k}']}")


if __name__ == "__main__":
    main()