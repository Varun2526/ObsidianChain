"""Fit the frozen ML + risk-propagation stacker for the current PS-native model.

exp14 showed that a logistic stack of [logit(LightGBM), log1p(1000 * propagated
risk)] beats LightGBM alone on transaction-weighted nAP (+0.053, p = 0.0496,
12 folds) and by +0.044 row nAP (p = 0.061, not significant). This script
freezes that stacker for production.

Training rows are OUT-OF-FOLD: for each of the four most recent rolling
folds, LightGBM is fitted on t <= train_end and propagation is seeded with
illicit labels from t <= train_end only, exactly as at inference. The stacker
is fitted on the pooled evaluation rows of those folds.

Transfer caveat, recorded in the output: on Elliptic++ the seeds are every
labelled illicit address before the window (thousands). In an uploaded
capture the seeds are OFAC / watchlist addresses (usually few). The
propagated score of a direct neighbour of a seed is similar in both regimes;
the score of a distant address is smaller with sparse seeds, so the stacker
leans on it less. It is applied only when at least one seed is present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "autoresearch_2026_09_23" / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "reproduction"))

from obsidianchain.ml import protocol  # noqa: E402
from obsidianchain.ml.stacking import PROP_SCALE, stack_features  # noqa: E402
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS  # noqa: E402
from exp14_risk_propagation import load_edges, prop_scores  # noqa: E402
from train_ps_model_v2 import OUT, _model, load_development, window  # noqa: E402

FOLDS_USED = 4


def main() -> dict:
    dev = load_development()
    edges = load_edges()
    features = list(CORE_PS_FEATURE_COLUMNS)
    parts = []
    for fold in protocol.rolling_origin_folds()[-FOLDS_USED:]:
        train = window(dev, fold.train_end)
        ev = dev[(dev.last_t >= fold.eval_start) & (dev.last_t <= fold.eval_end)]
        a = _model().fit(train[features], train.y).predict_proba(ev[features])[:, 1]
        p = prop_scores(edges, dev, fold.eval_end, fold.train_end, ev.address)
        parts.append(pd.DataFrame({"a": a, "p": p, "y": ev.y.to_numpy()}))
        print(f"{fold.label}: {len(ev):,} rows")
    oof = pd.concat(parts, ignore_index=True)
    lr = LogisticRegression().fit(stack_features(oof.a.to_numpy(), oof.p.to_numpy()), oof.y)
    out = {
        "schema": "obsidianchain.ps_stacker/1",
        "inputs": ["logit(lgbm_raw_score)", f"log1p({PROP_SCALE} * propagated_risk)"],
        "coef": [float(c) for c in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "reference_prevalence": float(oof.y.mean()),
        "fitted_on": f"out-of-fold rows of the last {FOLDS_USED} rolling folds",
        "n_rows": int(len(oof)),
        "evidence": "research/autoresearch_2026_09_23/results/exp14_risk_propagation.json",
        "applied_when": "at least one seed wallet is present in the capture",
        "transfer_caveat": __doc__.split("Transfer caveat, recorded in the output: ")[1].strip(),
    }
    (OUT / "stacker.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("coef", "intercept", "n_rows")}))
    return out


if __name__ == "__main__":
    main()
