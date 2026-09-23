"""Fit the frozen model + risk-propagation stacker for the production model.

Protocol B folds (see train_ps_production_model.py). For each of the last
four folds the production LightGBM configuration is trained on the fold's
training set; propagation is seeded with the illicit labels of addresses
first seen at or before the fold's train_end, over the spend graph up to the
window end; the stacker is fitted on the pooled evaluation rows. Every input
is therefore produced exactly as it would be at inference.

Also reported (never used for fitting): stacked vs unstacked nAP for folds
2-12, each fold's stacker fitted on the PREVIOUS fold's rows only, so the
seeded-scenario gain is measured out of sample.

Transfer caveat, recorded in the output: on Elliptic++ the seeds are every
labelled illicit address known before the window (thousands). In an uploaded
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
from obsidianchain.ml import registry as model_registry  # noqa: E402
from obsidianchain.ml.propagation import propagate  # noqa: E402
from obsidianchain.ml.stacking import PROP_SCALE, stack_features  # noqa: E402
from exp14_risk_propagation import load_edges  # noqa: E402
from exp22_window_end_protocol import event_features, fold_sets  # noqa: E402
from train_ps_production_model import FEATURES, MODEL_VERSION, OUT, TRAIN_WINDOW, _model  # noqa: E402

FOLDS_USED = 4


def fold_inputs(ev, edges, fold):
    train, test = fold_sets(ev, fold, TRAIN_WINDOW)
    raw = _model().fit(train[FEATURES], train.y).predict_proba(test[FEATURES])[:, 1]
    seeds = ev.loc[(ev._first <= fold.train_end) & (ev.y == 1), "address"].unique()
    prop = propagate(edges[edges.step <= fold.eval_end], seeds).scores.reindex(test.address).fillna(0.0).to_numpy()
    return pd.DataFrame({"a": raw, "p": prop, "y": test.y.to_numpy()})


def main() -> dict:
    if MODEL_VERSION in model_registry.Registry.open(OUT.parent).data["models"]:
        raise SystemExit(f"{MODEL_VERSION} is registered and immutable; its stacker cannot be refitted")
    ev = event_features()
    edges = load_edges()
    folds = protocol.rolling_origin_folds()
    parts = [fold_inputs(ev, edges, f) for f in folds]
    oos = []
    for i in range(1, len(folds)):
        st = LogisticRegression().fit(stack_features(parts[i - 1].a, parts[i - 1].p), parts[i - 1].y)
        s = st.predict_proba(stack_features(parts[i].a, parts[i].p))[:, 1]
        oos.append({"fold": folds[i].label,
                    "nap_model": protocol.normalised_average_precision(parts[i].y, parts[i].a)[0],
                    "nap_stacked": protocol.normalised_average_precision(parts[i].y, s)[0]})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in oos[-1].items()}, flush=True)
    diff = np.array([r["nap_stacked"] - r["nap_model"] for r in oos])
    verdict = protocol.paired_verdict(diff, a="stacked", b="model")
    pooled = pd.concat(parts[-FOLDS_USED:], ignore_index=True)
    lr = LogisticRegression().fit(stack_features(pooled.a, pooled.p), pooled.y)
    out = {
        "schema": "obsidianchain.ps_stacker/2",
        "model_version": MODEL_VERSION,
        "inputs": ["logit(lgbm_raw_score)", f"log1p({PROP_SCALE} * propagated_risk)"],
        "coef": [float(c) for c in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "reference_prevalence": float(pooled.y.mean()),
        "fitted_on": f"protocol-B evaluation rows of the last {FOLDS_USED} folds ({len(pooled)} rows)",
        "out_of_sample_check": {"per_fold": oos, "verdict": verdict,
                                "mean_nap_model": float(np.mean([r["nap_model"] for r in oos])),
                                "mean_nap_stacked": float(np.mean([r["nap_stacked"] for r in oos]))},
        "applied_when": "at least one seed wallet is present in the capture",
        "transfer_caveat": __doc__.split("Transfer caveat, recorded in the output: ")[1].strip(),
    }
    (OUT / "stacker.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("coef", "intercept")}), verdict["verdict"],
          round(verdict["p_value"], 4), round(out["out_of_sample_check"]["mean_nap_model"], 3), "->",
          round(out["out_of_sample_check"]["mean_nap_stacked"], 3))
    return out


if __name__ == "__main__":
    main()
