"""Threshold / operating-point analysis of the frozen ps_native_v5 (PLAN.md, commit b124dbe).

  python3 threshold_analysis.py validation   -> validation.json + selection.json
  python3 threshold_analysis.py holdout      -> holdout_check.json (REUSED_HOLDOUT)

Read-only with respect to the model, registry, schema and holdout artifacts.
The holdout mode refuses to run unless selection.json exists and is committed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "reproduction"))

from build_ps_dataset import build_canonical_frame  # noqa: E402
from obsidianchain.ml import protocol  # noqa: E402
from obsidianchain.ml.ps_model import PsNativeRiskModel  # noqa: E402
from obsidianchain.pipeline.features_ps import PsTemporalFeatureEngine  # noqa: E402

MODEL_DIR = ROOT / "data" / "models" / "ps_native" / "v5"
MODEL_SHA = "974d37f22e2f1e7df4e03cb5a13fdb35bf487e43f4d5db3ea3bb7191ebd3e607"
GRID = [round(0.10 + 0.05 * i, 2) for i in range(13)]          # 0.10 .. 0.70
VALIDATION_WINDOWS = [(2, 9), (10, 17), (18, 25)]
HOLDOUT_WINDOW = (42, 49)
TOP_K = (100, 500, 1000)


def load_model() -> PsNativeRiskModel:
    sha = hashlib.sha256((MODEL_DIR / "model.joblib").read_bytes()).hexdigest()
    assert sha == MODEL_SHA, f"model.joblib sha256 {sha} is not the registered v5"
    return PsNativeRiskModel.load(MODEL_DIR)


def window(model, lo: int, hi: int) -> pd.DataFrame:
    """Addresses first seen in [lo, hi], each at its last event <= hi (protocol B)."""
    frame = build_canonical_frame(ROOT / "data" / "raw", max_step=hi)
    feats = PsTemporalFeatureEngine().process_records(frame)
    feats["_step"] = feats.txid.map(dict(zip(frame.txid, frame["_step"]))).astype(int)
    feats["_seq"] = np.arange(len(feats))
    feats["_first"] = feats.address.map(feats.groupby("address")._step.min())
    w = feats[(feats._first >= lo) & (feats._first <= hi)]
    w = w.sort_values("_seq").drop_duplicates("address", keep="last")
    labels = pd.read_csv(ROOT / "data" / "raw" / "wallets_classes.csv").set_index("address")["class"]
    w = w.assign(cls=w.address.map(labels))
    w = w[w.cls.isin([1, 2])].assign(y=lambda d: (d.cls == 1).astype(int)).reset_index(drop=True)
    raw = model.raw_scores(w)
    return pd.DataFrame({"window": f"t{lo}-{hi}", "address": w.address, "y": w.y.to_numpy(),
                         "raw": raw, "prob": model.calibrate(raw)})


def at_threshold(y, prob, t) -> dict:
    pred = prob >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {"threshold": t, "precision": precision, "recall": recall, "f1": f1,
            "balanced_accuracy": (recall + tnr) / 2, "fpr": fp / (fp + tn) if fp + tn else float("nan"),
            "accuracy": (tp + tn) / len(y), "predicted_positive": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def top_k(y, score, ks=TOP_K) -> dict:
    order = np.argsort(-score, kind="stable")
    out = {}
    for k in ks:
        hit = int(y[order[:k]].sum())
        out[f"P@{k}"] = hit / k
        out[f"R@{k}"] = hit / int(y.sum())
    return out


def summary(frame: pd.DataFrame) -> dict:
    y, prob = frame.y.to_numpy(), frame.prob.to_numpy()
    return {"n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
            "nap": float(protocol.normalised_average_precision(y, frame.raw.to_numpy())[0]),
            "grid": [at_threshold(y, prob, t) for t in GRID],
            "reference_0_50": at_threshold(y, prob, 0.50),
            "top_k": top_k(y, prob)}


def select(pooled: dict) -> dict:
    """The rules in PLAN.md, applied mechanically."""
    g = pooled["grid"]
    best_f1 = max(r["f1"] for r in g)
    balanced = max(r["threshold"] for r in g if r["f1"] == best_f1)
    hr = [r["threshold"] for r in g if r["precision"] >= 0.50]
    high_recall = min(hr) if hr else None
    crit = [r["threshold"] for r in g if r["precision"] >= 0.90]
    critical = min(crit) if crit else None
    bands = {"Critical": critical, "High": balanced, "Medium": high_recall}
    merged = []
    names = list(bands)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if bands[a] is not None and bands[a] == bands[b]:
                merged.append(f"{a}={b}")
    return {"balanced_point": balanced, "high_recall_point": high_recall, "bands": bands,
            "band_rule": "Critical: lowest grid t with validation precision >= 0.90; High: balanced (max F1); "
                         "Medium: lowest grid t with precision >= 0.50; Low: below Medium",
            "merged_bands": merged}


def committed(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    out = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%h", "--", str(rel)],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(rel)],
                           capture_output=True, text=True).stdout.strip()
    return bool(out) and not dirty


def main(mode: str) -> None:
    model = load_model()
    if mode == "validation":
        frames = [window(model, lo, hi) for lo, hi in VALIDATION_WINDOWS]
        pooled = pd.concat(frames, ignore_index=True)
        result = {"result_type": "VALIDATION (addresses first seen t2-25; not in v5 training set)",
                  "model_sha256": MODEL_SHA, "pooled": summary(pooled),
                  "per_window": {f["window"].iloc[0]: summary(f) for f in frames},
                  "calibrated_prob_range": [float(pooled.prob.min()), float(pooled.prob.max())]}
        (HERE / "validation.json").write_text(json.dumps(result, indent=1))
        sel = select(result["pooled"])
        (HERE / "selection.json").write_text(json.dumps(sel, indent=1))
        print(json.dumps(sel, indent=1))
    elif mode == "holdout":
        sel_path = HERE / "selection.json"
        if not sel_path.exists() or not committed(sel_path):
            raise SystemExit("selection.json must exist and be committed before the holdout check")
        sel = json.loads(sel_path.read_text())
        hold = window(model, *HOLDOUT_WINDOW)
        y, prob, raw = hold.y.to_numpy(), hold.prob.to_numpy(), hold.raw.to_numpy()
        points = {"current_0_50": 0.50, "balanced": sel["balanced_point"], "high_recall": sel["high_recall_point"]}
        res = {"result_type": "REUSED_HOLDOUT (t42-49 is SEEN under ADR 0004; a check, not an estimate)",
               "model_sha256": MODEL_SHA, "n": int(len(y)), "positives": int(y.sum()),
               "prevalence": float(y.mean()),
               "nap": float(protocol.normalised_average_precision(y, raw)[0]),
               "top_k": top_k(y, prob),
               "points": {k: at_threshold(y, prob, t) for k, t in points.items() if t is not None},
               "band_counts": {b: int((prob >= t).sum()) for b, t in sel["bands"].items() if t is not None}}
        (HERE / "holdout_check.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ("n", "positives", "nap", "top_k")}, indent=1))
        for k, v in res["points"].items():
            print(k, {m: (round(v[m], 4) if isinstance(v[m], float) else v[m])
                      for m in ("threshold", "precision", "recall", "f1", "balanced_accuracy", "fpr",
                                "accuracy", "predicted_positive")})
    else:
        raise SystemExit("mode: validation | holdout")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
