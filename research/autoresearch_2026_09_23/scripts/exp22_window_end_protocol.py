"""EXPERIMENT exp22_window_end_protocol.  Pre-registered before running.

Why a second evaluation unit
----------------------------
The development parquets represent each address by its FINAL event in the
whole dataset. exp21's decomposition showed that choice matters: after the
whole-row snapshot fix, 10.9% of addresses moved to a later same-step event,
and those addresses are 30.7% illicit (5.6% overall) - mostly pass-through
wallets whose final event is the spend. Scoring at the final event is
hindsight: a deployed capture ends at a window boundary, and an address whose
activity continues is scored mid-life.

Protocol B (this experiment, the PRIMARY unit from here on):
* Event-level features come from the unchanged production engine over the
  causal Elliptic++ stream (``build_canonical_frame`` + PsTemporalFeatureEngine).
* Fold (train_end te, window es..ee), the 12 protocol folds:
    train  addresses first seen in (te - W, te], snapshot = last event <= te
    eval   addresses first seen in [es, ee],     snapshot = last event <= ee
  An address is in at most one of the two, so no label is ever seen in
  training for an evaluated address. Labels: wallets_classes (1 / 2 only).
* Tuning folds 1-6 choose; confirmation folds 7-12 only report.

Stage 1 (tuning folds): LightGBM, window W in {12, 16, 24, 41},
min_child_samples in {50, 200}, 31 leaves, CORE features. Pick the best mean
tuning nAP; ties within 0.005 go to the smaller W then larger min_child.
Stage 2 (all folds): A = production config (W 16, min_child 50) vs the
winner; and the group-G ablation (CORE vs CORE minus G) at the adopted config.
Adoption rules: as exp19 (winner adopted iff confirmation mean >= A and
confirmation worst >= A); G kept iff it improves tuning mean and does not
lower confirmation mean or worst.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "reproduction"))
sys.path.insert(0, str(ROOT / "research" / "autoresearch_2026_09_23" / "scripts"))

from build_ps_dataset import build_canonical_frame  # noqa: E402
from obsidianchain.ml import evaluation, protocol  # noqa: E402
from obsidianchain.pipeline.features_ps import (  # noqa: E402
    CORE_PS_FEATURE_COLUMNS, GROUP_G_UPSTREAM, PsTemporalFeatureEngine,
)
from exp19_tune_and_confirm import lgbm  # noqa: E402

OUT = ROOT / "research/autoresearch_2026_09_23/results/exp22_window_end_protocol.json"
CACHE = ROOT / "research/autoresearch_2026_09_23/results/exp22_event_features.parquet"
FOLDS = protocol.rolling_origin_folds()
CORE = list(CORE_PS_FEATURE_COLUMNS)
NO_G = [c for c in CORE if c not in GROUP_G_UPSTREAM]
PRODUCTION = {"train_window": 16, "num_leaves": 31, "min_child_samples": 50}


def event_features() -> pd.DataFrame:
    if CACHE.is_file():
        return pd.read_parquet(CACHE)
    frame = build_canonical_frame(ROOT / "data" / "raw")
    feats = PsTemporalFeatureEngine().process_records(frame)
    feats["_step"] = feats.txid.map(dict(zip(frame.txid, frame["_step"]))).astype(int)
    feats["_seq"] = np.arange(len(feats))  # causal processing order
    labels = pd.read_csv(ROOT / "data" / "raw" / "wallets_classes.csv").set_index("address")["class"]
    feats["y"] = feats.address.map(labels)
    feats = feats[feats.y.isin([1, 2])].copy()
    feats["y"] = (feats.y == 1).astype(np.int8)
    first = feats.groupby("address")._step.transform("min")
    feats["_first"] = first
    feats = feats[feats._first <= protocol.DEVELOPMENT_END]  # the holdout period is never read
    feats = feats[feats._step <= protocol.DEVELOPMENT_END]
    feats.to_parquet(CACHE, index=False)
    return feats


def snapshot(ev: pd.DataFrame, first_lo: int, first_hi: int, as_of: int) -> pd.DataFrame:
    rows = ev[(ev._first >= first_lo) & (ev._first <= first_hi) & (ev._step <= as_of)]
    return rows.sort_values("_seq").drop_duplicates("address", keep="last")


def fold_sets(ev, fold, window):
    te = fold.train_end
    train = snapshot(ev, te - window + 1, te, te)
    test = snapshot(ev, fold.eval_start, fold.eval_end, fold.eval_end)
    return train, test


def score(ev, fold, cfg, feats):
    train, test = fold_sets(ev, fold, cfg["train_window"])
    s = lgbm(cfg).fit(train[feats], train.y).predict_proba(test[feats])[:, 1]
    return test, s


def row_metrics(test, s) -> dict:
    y = test.y.to_numpy()
    m = evaluation.ranking_metrics(y, s, ks=(50, 100, 500, 1000))
    return {"n": m["n"], "positives": m["positives"], "prevalence": m["prevalence"], "nap": m["nap"],
            "roc_auc": m["roc_auc"], "r_precision": m["r_precision"], "P@50": m["P@50"], "P@100": m["P@100"],
            "P@500": m["P@500"], "R@500": m["R@500"], "nR@500": m["nR@500"], "R@1000": m["R@1000"],
            "txw": evaluation.transaction_weighted_nap(y, s, test.txid.to_numpy())}


def summarise(rows):
    f = pd.DataFrame(rows)
    return {k: {"mean": round(float(f[k].mean()), 4), "min": round(float(f[k].min()), 4)}
            for k in f.columns if k not in ("fold", "n", "positives")}


def main() -> None:
    t0 = time.time()
    ev = event_features()
    print(f"event rows {len(ev):,}  addresses {ev.address.nunique():,}  ({time.time()-t0:.0f}s)", flush=True)
    grid = []
    for w, mc in itertools.product((12, 16, 24, 41), (50, 200)):
        cfg = {"train_window": w, "num_leaves": 31, "min_child_samples": mc}
        naps = [protocol.normalised_average_precision(t.y, s)[0] for t, s in (score(ev, f, cfg, CORE) for f in FOLDS[:6])]
        grid.append({**cfg, "tune_nap": float(np.mean(naps))})
        print("grid", cfg, round(float(np.mean(naps)), 4), flush=True)
    best = max(g["tune_nap"] for g in grid)
    winner = sorted([g for g in grid if g["tune_nap"] >= best - 0.005],
                    key=lambda g: (g["train_window"], -g["min_child_samples"]))[0]
    w_cfg = {k: winner[k] for k in PRODUCTION}

    rows = {"A": [], "WINNER": []}
    for fold in FOLDS:
        for name, cfg in (("A", PRODUCTION), ("WINNER", w_cfg)):
            test, s = score(ev, fold, cfg, CORE)
            rows[name].append({"fold": fold.label, **row_metrics(test, s)})
        print(fold.label, {k: round(v[-1]["nap"], 3) for k, v in rows.items()}, flush=True)
    split = {n: {"tune": summarise(r[:6]), "confirm": summarise(r[6:]), "all": summarise(r)} for n, r in rows.items()}
    c, a = split["WINNER"]["confirm"]["nap"], split["A"]["confirm"]["nap"]
    adopt_winner = w_cfg != PRODUCTION and c["mean"] >= a["mean"] and c["min"] >= a["min"]
    adopted = w_cfg if adopt_winner else PRODUCTION

    g_rows = {"CORE": [], "NO_G": []}
    for fold in FOLDS:
        for name, feats in (("CORE", CORE), ("NO_G", NO_G)):
            test, s = score(ev, fold, adopted, feats)
            g_rows[name].append({"fold": fold.label, **row_metrics(test, s)})
    g_split = {n: {"tune": summarise(r[:6]), "confirm": summarise(r[6:]), "all": summarise(r)} for n, r in g_rows.items()}
    gc, gn = g_split["CORE"], g_split["NO_G"]
    keep_g = (gc["tune"]["nap"]["mean"] > gn["tune"]["nap"]["mean"]
              and gc["confirm"]["nap"]["mean"] >= gn["confirm"]["nap"]["mean"]
              and gc["confirm"]["nap"]["min"] >= gn["confirm"]["nap"]["min"])
    g_verdict = protocol.paired_verdict(np.array([x["nap"] for x in g_rows["CORE"]])
                                        - np.array([x["nap"] for x in g_rows["NO_G"]]), a="CORE", b="NO_G")
    for n, s in {**split, **{f"G:{k}": v for k, v in g_split.items()}}.items():
        for part in ("tune", "confirm", "all"):
            m = s[part]
            print(f"{n:>10} {part:>7}: nAP {m['nap']['mean']:.3f} (min {m['nap']['min']:.3f}) "
                  f"P@100 {m['P@100']['mean']:.2f} (min {m['P@100']['min']:.2f}) Rprec {m['r_precision']['mean']:.3f} "
                  f"nR@500 {m['nR@500']['mean']:.3f} R@1000 {m['R@1000']['mean']:.3f} AUC {m['roc_auc']['mean']:.3f}")
    print("winner", w_cfg, "adopt", adopt_winner, "| keep G", keep_g, g_verdict["verdict"],
          round(g_verdict["mean_difference"], 4), round(g_verdict["p_value"], 4))
    OUT.write_text(json.dumps({
        "experiment_id": "exp22_window_end_protocol", "design": __doc__, "grid": grid, "winner": winner,
        "per_fold": rows, "split": split, "adopt_winner": adopt_winner, "adopted_config": adopted,
        "g_per_fold": g_rows, "g_split": g_split, "keep_g": keep_g, "g_verdict": g_verdict,
        "runtime_seconds": round(time.time() - t0)}, indent=2, default=str))


if __name__ == "__main__":
    main()
