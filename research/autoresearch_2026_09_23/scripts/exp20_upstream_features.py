"""EXPERIMENT exp20_upstream_features.  Pre-registered before running.

Question: does candidate group G (upstream flow - a summary of the
transactions that funded this transaction's inputs, pipeline/features_ps.py)
improve ranking over CORE?

Design:
* Model configuration: the one exp19 adopted (read from its results file;
  production v2 if exp19 adopted nothing). Same seed, same 12 folds.
* Candidates: CORE vs CORE + G.
* Decision rule, same shape as exp19: G is adopted iff on TUNING folds 1-6
  mean nAP improves, AND on CONFIRMATION folds 7-12 mean nAP >= CORE's and
  worst-fold nAP >= CORE's. The full 12-fold paired verdict is reported, not
  used as a gate (with MDE 0.164 it would reject almost any real feature).
* Same metrics as exp19 (nAP, tx-weighted nAP, P@50/100/500, R-precision,
  ceiling-normalised R@500).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "reproduction"))
sys.path.insert(0, str(ROOT / "research" / "autoresearch_2026_09_23" / "scripts"))

from obsidianchain.ml import diagnostics, protocol  # noqa: E402
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS, GROUP_G_UPSTREAM  # noqa: E402
from exp19_tune_and_confirm import (  # noqa: E402
    FOLDS, PRODUCTION, eval_rows, lgbm, metrics, summarise, train_rows,
)
from train_ps_model_v2 import load_development  # noqa: E402

OUT = ROOT / "research/autoresearch_2026_09_23/results/exp20_upstream_features.json"
EXP19 = ROOT / "research/autoresearch_2026_09_23/results/exp19_tune_and_confirm.json"


def main() -> None:
    dev = load_development()
    cfg = dict(PRODUCTION)
    r19 = json.loads(EXP19.read_text())
    if r19.get("adopt") and r19["chosen"] in ("G", "E"):
        # E's LightGBM member uses the grid winner's configuration.
        cfg = {k: r19["grid_winner"][k] for k in PRODUCTION}
    core = list(CORE_PS_FEATURE_COLUMNS)
    with_g = core + [c for c in GROUP_G_UPSTREAM if c not in core]
    diagnostics.assert_healthy(dev, with_g)
    rows = {"CORE": [], "CORE_G": []}
    for fold in FOLDS:
        tr, ev = train_rows(dev, fold, cfg["train_window"]), eval_rows(dev, fold)
        y, tx = ev.y.to_numpy(), ev.txid.to_numpy()
        for name, feats in (("CORE", core), ("CORE_G", with_g)):
            s = lgbm(cfg).fit(tr[feats], tr.y).predict_proba(ev[feats])[:, 1]
            rows[name].append({"fold": fold.label, **metrics(y, s, tx)})
        print(fold.label, rows["CORE"][-1]["nap"].__round__(3), rows["CORE_G"][-1]["nap"].__round__(3), flush=True)
    split = {n: {"tune": summarise(r[:6]), "confirm": summarise(r[6:]), "all": summarise(r)} for n, r in rows.items()}
    c, g = split["CORE"], split["CORE_G"]
    adopt = (g["tune"]["nap"]["mean"] > c["tune"]["nap"]["mean"]
             and g["confirm"]["nap"]["mean"] >= c["confirm"]["nap"]["mean"]
             and g["confirm"]["nap"]["min"] >= c["confirm"]["nap"]["min"])
    verdict = protocol.paired_verdict(np.array([x["nap"] for x in rows["CORE_G"]])
                                      - np.array([x["nap"] for x in rows["CORE"]]), a="CORE_G", b="CORE")
    for n, s in split.items():
        for part in ("tune", "confirm", "all"):
            m = s[part]
            print(f"{n:>7} {part:>7}: nAP {m['nap']['mean']:.3f} (min {m['nap']['min']:.3f}) P@100 {m['P@100']['mean']:.2f} "
                  f"(min {m['P@100']['min']:.2f}) P@500 {m['P@500']['mean']:.2f} Rprec {m['r_precision']['mean']:.3f} "
                  f"nR@500 {m['nR@500']['mean']:.3f} txw {m['txw']['mean']:.3f}")
    print("config", cfg, "adopt G", adopt, verdict["verdict"], round(verdict["mean_difference"], 4), round(verdict["p_value"], 4))
    OUT.write_text(json.dumps({"experiment_id": "exp20_upstream_features", "design": __doc__, "config": cfg,
                               "per_fold": rows, "split": split, "adopt": adopt, "verdict": verdict},
                              indent=2, default=str))


if __name__ == "__main__":
    main()
