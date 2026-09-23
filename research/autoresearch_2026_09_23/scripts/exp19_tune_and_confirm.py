"""EXPERIMENT exp19_tune_and_confirm.  Pre-registered before running.

Goal: raise ranking quality without fooling ourselves. Every choice is made
on TUNING folds 1-6 (eval t17-28) and reported untouched on CONFIRMATION
folds 7-12 (eval t29-40). Selecting on the same folds one reports is how a
number gets inflated; here the confirmation folds never influence a choice.

Metrics per fold (address level unless stated):
  nAP, tx-weighted nAP, P@50, P@100, P@500,
  R-precision  = precision at K = number of positives (= recall there),
  nR@500       = R@500 / min(1, 500 / positives), i.e. recall as a share of
                 the most a 500-alert budget could possibly catch.

Stage 1 - grid (LightGBM), tuning folds only:
  train_window in {12, 16, 24}, num_leaves in {15, 31, 63},
  min_child_samples in {20, 50, 200}; other v2 settings fixed.
  Selection score: mean tuning-fold nAP. Ties within 0.005 go to the
  simpler model (fewer leaves, then larger min_child_samples).

Stage 2 - candidates, all at the selected configuration:
  A   current production v2 (window 16, 31 leaves, min_child 50)
  G   grid winner
  E   rank-average of G-LightGBM, HistGradientBoosting and RandomForest
      (the statistically tied tier from exp01), same window
  Candidate chosen for production = best mean tuning-fold nAP among A, G, E,
  adopted only if on CONFIRMATION folds its mean nAP >= A's and its worst
  confirmation-fold nAP >= A's.

Reported separately, never used for selection: the seeded scenario (+
propagation stacker, seeds = illicit labels with t <= train_end), which
applies to production only when seed wallets are present.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "reproduction"))
sys.path.insert(0, str(ROOT / "research" / "autoresearch_2026_09_23" / "scripts"))

from obsidianchain.ml import protocol  # noqa: E402
from obsidianchain.ml.stacking import stack_features  # noqa: E402
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS  # noqa: E402
from train_ps_model_v2 import HYPERPARAMETERS, SEED, load_development, transaction_weighted_nap  # noqa: E402

import os  # noqa: E402

from obsidianchain.pipeline.features_ps import GROUP_G_UPSTREAM  # noqa: E402

#: exp21 re-runs this exact design on the causally ordered data with group G
#: (EXP19_FEATURES=core_g EXP19_TAG=exp21_...). The default reproduces exp19.
_TAG = os.environ.get("EXP19_TAG", "exp19_tune_and_confirm")
OUT = ROOT / f"research/autoresearch_2026_09_23/results/{_TAG}.json"
FOLDS = protocol.rolling_origin_folds()
TUNE, CONFIRM = FOLDS[:6], FOLDS[6:]
FEATURES = list(CORE_PS_FEATURE_COLUMNS)
if os.environ.get("EXP19_FEATURES") == "core_g":
    FEATURES = FEATURES + [c for c in GROUP_G_UPSTREAM if c not in FEATURES]
WINDOWS = tuple(int(w) for w in os.environ.get("EXP19_WINDOWS", "12,16,24").split(","))
PRODUCTION = {"train_window": 16, "num_leaves": 31, "min_child_samples": 50}


def lgbm(cfg):
    params = dict(HYPERPARAMETERS, num_leaves=cfg["num_leaves"], min_child_samples=cfg["min_child_samples"])
    return LGBMClassifier(**params, random_state=SEED, verbose=-1, n_jobs=-1)


def train_rows(dev, fold, w):
    return dev[(dev.last_t <= fold.train_end) & (dev.last_t > fold.train_end - w)]


def eval_rows(dev, fold):
    return dev[(dev.last_t >= fold.eval_start) & (dev.last_t <= fold.eval_end)]


def metrics(y, s, tx) -> dict:
    order = np.argsort(-s, kind="stable")
    pos = int(y.sum())
    out = {"nap": protocol.normalised_average_precision(y, s)[0],
           "txw": transaction_weighted_nap(y, s, tx),
           "r_precision": float(y[order[:pos]].mean())}
    for k in (50, 100, 500):
        out[f"P@{k}"] = float(y[order[:k]].mean())
    r500 = y[order[:500]].sum() / max(pos, 1)
    out["nR@500"] = float(r500 / min(1.0, 500 / max(pos, 1)))
    return out


def summarise(rows: list[dict]) -> dict:
    f = pd.DataFrame(rows)
    return {m: {"mean": round(float(f[m].mean()), 4), "min": round(float(f[m].min()), 4)}
            for m in f.columns if m != "fold"}


def main() -> None:
    t0 = time.time()
    dev = load_development()

    # ---- stage 1: grid on tuning folds ---------------------------------
    grid = []
    for w, nl, mc in itertools.product(WINDOWS, (15, 31, 63), (20, 50, 200)):
        cfg = {"train_window": w, "num_leaves": nl, "min_child_samples": mc}
        naps = []
        for fold in TUNE:
            tr, ev = train_rows(dev, fold, w), eval_rows(dev, fold)
            s = lgbm(cfg).fit(tr[FEATURES], tr.y).predict_proba(ev[FEATURES])[:, 1]
            naps.append(protocol.normalised_average_precision(ev.y, s)[0])
        grid.append({**cfg, "tune_nap": float(np.mean(naps))})
        print(f"grid {cfg} tune nAP {np.mean(naps):.4f}", flush=True)
    best = max(g["tune_nap"] for g in grid)
    near = [g for g in grid if g["tune_nap"] >= best - 0.005]
    winner = sorted(near, key=lambda g: (g["num_leaves"], -g["min_child_samples"], g["train_window"]))[0]
    g_cfg = {k: winner[k] for k in PRODUCTION}
    print("grid winner", winner, flush=True)

    # ---- stage 2: candidates on all folds ------------------------------
    rows = {"A": [], "G": [], "E": [], "A_seeded": [], "E_seeded": []}
    from exp14_risk_propagation import load_edges, prop_scores
    edges = load_edges()
    for fold in FOLDS:
        ev = eval_rows(dev, fold)
        y, tx = ev.y.to_numpy(), ev.txid.to_numpy()
        scores = {}
        for name, cfg in (("A", PRODUCTION), ("G", g_cfg)):
            tr = train_rows(dev, fold, cfg["train_window"])
            scores[name] = lgbm(cfg).fit(tr[FEATURES], tr.y).predict_proba(ev[FEATURES])[:, 1]
        tr = train_rows(dev, fold, g_cfg["train_window"])
        hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED) \
            .fit(tr[FEATURES], tr.y).predict_proba(ev[FEATURES])[:, 1]
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED) \
            .fit(tr[FEATURES].fillna(-1), tr.y).predict_proba(ev[FEATURES].fillna(-1))[:, 1]
        scores["E"] = (rankdata(scores["G"]) + rankdata(hgb) + rankdata(rf)) / (3 * len(y))
        for name in ("A", "G", "E"):
            rows[name].append({"fold": fold.label, **metrics(y, scores[name], tx)})

        # Seeded scenario: stacker fitted on the calibration window, as exp14.
        te = fold.train_end
        cal = dev[(dev.last_t > te - 2) & (dev.last_t <= te)]
        p_eval = prop_scores(edges, dev, fold.eval_end, te, ev.address)
        p_cal = prop_scores(edges, dev, te, te - 2, cal.address)
        for name, cfg in (("A", PRODUCTION), ("E", g_cfg)):
            fit = dev[(dev.last_t <= te - 2) & (dev.last_t > te - 2 - cfg["train_window"])]
            a_cal = lgbm(cfg).fit(fit[FEATURES], fit.y).predict_proba(cal[FEATURES])[:, 1]
            base = scores["A"] if name == "A" else scores["G"]
            st = LogisticRegression().fit(stack_features(a_cal, p_cal), cal.y)
            s = st.predict_proba(stack_features(base, p_eval))[:, 1]
            if name == "E":
                s = (rankdata(s) + rankdata(hgb) + rankdata(rf)) / (3 * len(y))
            rows[f"{name}_seeded"].append({"fold": fold.label, **metrics(y, s, tx)})
        print(fold.label, " ".join(f"{k}:{rows[k][-1]['nap']:.3f}/{rows[k][-1]['P@100']:.2f}" for k in rows), flush=True)

    split = {}
    for name, r in rows.items():
        split[name] = {"tune": summarise(r[:6]), "confirm": summarise(r[6:]), "all": summarise(r)}
    chosen = max(("A", "G", "E"), key=lambda n: split[n]["tune"]["nap"]["mean"])
    c, a = split[chosen]["confirm"]["nap"], split["A"]["confirm"]["nap"]
    adopt = chosen != "A" and c["mean"] >= a["mean"] and c["min"] >= a["min"]
    verdict = protocol.paired_verdict(
        np.array([x["nap"] for x in rows[chosen][6:]]) - np.array([x["nap"] for x in rows["A"][6:]]),
        a=chosen, b="A") if chosen != "A" else None
    for name, s in split.items():
        for part in ("tune", "confirm"):
            m = s[part]
            print(f"{name:>9} {part:>7}: nAP {m['nap']['mean']:.3f} (min {m['nap']['min']:.3f}) "
                  f"P@100 {m['P@100']['mean']:.2f} (min {m['P@100']['min']:.2f}) P@500 {m['P@500']['mean']:.2f} "
                  f"Rprec {m['r_precision']['mean']:.3f} nR@500 {m['nR@500']['mean']:.3f} txw {m['txw']['mean']:.3f}")
    print("chosen", chosen, "adopt", adopt, verdict and (verdict["verdict"], round(verdict["p_value"], 3)))
    OUT.write_text(json.dumps({"experiment_id": "exp19_tune_and_confirm", "design": __doc__,
                               "grid": grid, "grid_winner": winner, "per_fold": rows, "split": split,
                               "chosen": chosen, "adopt": adopt, "confirm_verdict": verdict,
                               "runtime_seconds": round(time.time() - t0)}, indent=2, default=str))


if __name__ == "__main__":
    main()
