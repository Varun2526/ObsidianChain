"""EXPERIMENT exp03_error_analysis.

Phase 7. Uses the LAST development fold (t<=38 -> eval t39-40) - the fold
with the most training history, most representative of what a deployed
model would actually see - to fit once (not 12x) and inspect predictions in
detail: score distribution, precision/recall at realistic alert budgets,
and false positive / false negative breakdown by feature-group values.

Single fold, single seed: this is DIAGNOSTIC, not a claim about ranking
quality (that claim is exp01/exp02's job, under the full 12-fold protocol).
Labelled as such throughout.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from obsidianchain.ml import diagnostics, protocol  # noqa: E402
from obsidianchain.pipeline.features_ps import (  # noqa: E402
    CORE_PS_FEATURE_COLUMNS, PS_FEATURE_GROUPS,
)

DS = ROOT / "data/models/ps_native/datasets"
OUT = ROOT / "research/autoresearch_2026_09_23/results/exp03_error_analysis.json"
BASE, STEP = 1400000000, 1209600
SEED = 20260919


def load_dev() -> pd.DataFrame:
    dev = pd.concat(
        [pd.read_parquet(DS / "train.parquet"), pd.read_parquet(DS / "validation.parquet")],
        ignore_index=True,
    )
    dev["first_t"] = ((dev["timestamp"] - BASE) // STEP + 1).astype(int)
    return dev


def main() -> None:
    dev = load_dev()
    declared = list(CORE_PS_FEATURE_COLUMNS)
    healthy = diagnostics.healthy_features(dev, declared)
    folds = protocol.rolling_origin_folds()
    fold = folds[-1]  # t<=38 -> t39-40, the largest-history fold
    print(f"using fold {fold.label}")

    usable = protocol.development(dev)
    train = usable[usable["first_t"] <= fold.train_end]
    ev = usable[(usable["first_t"] >= fold.eval_start) & (usable["first_t"] <= fold.eval_end)].copy()

    from lightgbm import LGBMClassifier
    X = np.nan_to_num(train[healthy].to_numpy("float32"))
    y = train["y"].to_numpy("int8")
    Xe = np.nan_to_num(ev[healthy].to_numpy("float32"))
    m = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(X, y)
    ev["score"] = m.predict_proba(Xe)[:, 1]

    n = len(ev)
    positives = int(ev["y"].sum())
    print(f"eval rows {n}  positives {positives}  prevalence {positives/n:.4f}")

    nap, prevalence = protocol.normalised_average_precision(ev["y"], ev["score"])
    print(f"single-fold nAP {nap:.4f} (diagnostic only, not the protocol's headline number)")

    # ---- precision/recall at realistic alert budgets ----
    budgets = [20, 50, 100, 200, 500]
    budget_rows = []
    ranked = ev.sort_values("score", ascending=False).reset_index(drop=True)
    for k in budgets:
        k = min(k, n)
        top = ranked.iloc[:k]
        prec = float(top["y"].mean())
        rec = float(top["y"].sum() / positives) if positives else float("nan")
        budget_rows.append({"k": k, "precision": prec, "recall": rec})
        print(f"  budget K={k:<4} precision={prec:.3f}  recall={rec:.3f}")

    # ---- false positives / false negatives at K=100 ----
    K = min(100, n)
    top_k = ranked.iloc[:K]
    fp = top_k[top_k["y"] == 0]
    fn_pool = ranked.iloc[K:]
    fn = fn_pool[fn_pool["y"] == 1]
    print(f"\nat K={K}: {len(fp)} false positives, {len(fn)} false negatives (of {positives} total positives)")

    def describe_group(frame: pd.DataFrame, group_cols: list[str]) -> dict:
        cols = [c for c in group_cols if c in frame.columns]
        if not cols or frame.empty:
            return {}
        return {c: {
            "mean": float(frame[c].mean()) if frame[c].notna().any() else None,
            "median": float(frame[c].median()) if frame[c].notna().any() else None,
        } for c in cols}

    comparison = {}
    for group_name, cols in PS_FEATURE_GROUPS.items():
        if group_name == "E_network":
            continue
        comparison[group_name] = {
            "false_positives_at_K100": describe_group(fp, cols),
            "false_negatives_beyond_K100": describe_group(fn, cols),
            "true_positives_at_K100": describe_group(top_k[top_k["y"] == 1], cols),
            "all_positives_eval": describe_group(ev[ev["y"] == 1], cols),
            "all_negatives_eval": describe_group(ev[ev["y"] == 0], cols),
        }

    # score distribution for positives vs negatives
    score_dist = {
        "positives": {
            "mean": float(ev.loc[ev.y == 1, "score"].mean()),
            "median": float(ev.loc[ev.y == 1, "score"].median()),
            "p10": float(ev.loc[ev.y == 1, "score"].quantile(0.10)),
        },
        "negatives": {
            "mean": float(ev.loc[ev.y == 0, "score"].mean()),
            "median": float(ev.loc[ev.y == 0, "score"].median()),
            "p90": float(ev.loc[ev.y == 0, "score"].quantile(0.90)),
        },
    }
    print("\nscore distribution (positives vs negatives):")
    print(json.dumps(score_dist, indent=1))

    # false negatives: are they low-activity addresses that simply lack signal?
    fn_txcount = fn["n_txs_asof_t"].describe().to_dict() if "n_txs_asof_t" in fn.columns else {}
    tp_txcount = top_k.loc[top_k.y == 1, "n_txs_asof_t"].describe().to_dict() if "n_txs_asof_t" in top_k.columns else {}
    print("\nn_txs_asof_t: false negatives vs true positives (do FNs have less history?)")
    print(" FN  n_txs_asof_t describe:", {k: round(v, 3) for k, v in fn_txcount.items()})
    print(" TP  n_txs_asof_t describe:", {k: round(v, 3) for k, v in tp_txcount.items()})

    payload = {
        "experiment_id": "exp03_error_analysis",
        "note": "SINGLE-FOLD DIAGNOSTIC (fold t<=38 -> t39-40). Not a ranking-quality claim.",
        "fold": fold.label,
        "n_eval": n,
        "positives": positives,
        "prevalence": prevalence,
        "single_fold_nap": nap,
        "budgets": budget_rows,
        "k_for_error_breakdown": K,
        "n_false_positives": int(len(fp)),
        "n_false_negatives": int(len(fn)),
        "feature_group_comparison": comparison,
        "score_distribution": score_dist,
        "false_negative_n_txs_asof_t": fn_txcount,
        "true_positive_n_txs_asof_t": tp_txcount,
    }
    OUT.write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
