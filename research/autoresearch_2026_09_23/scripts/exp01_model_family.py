"""EXPERIMENT exp01_model_family.

Hypothesis (locked before running): none of {LogReg-L2, LogReg-L1, RandomForest,
ExtraTrees, HistGradientBoosting, LightGBM} is assumed superior. We measure all
six under the existing canonical protocol (ml/protocol.py: 12 rolling-origin
folds, width=2, min_train=16, paired t-test + Holm-Bonferroni across the full
family, dev-only i.e. train+validation, holdout untouched) on the CURRENT
v2 PS-native feature schema (24 healthy columns, no network group - it is not
present in this dataset; see 01_repo_audit.md).

This extends research/protocol_2026_09_22/run_protocol.py (which already
covered RandomForest/LightGBM/LogisticRegression-L2 and found RF vs LightGBM
INDISTINGUISHABLE, both >> LogReg) by adding LogReg-L1, ExtraTrees and
HistGradientBoosting, and by running the comparison as ONE family (6 choose 2
= 15 pairwise tests, Holm-corrected together) rather than as separate
three-way runs. XGBoost is not tested: not installed and not in the vendored
offline wheel set (HANDOFF.md, ml/protocol.py's own offline constraint) -
recorded as NOT_TESTED, not as a negative result.

Same dataset, same folds, same feature list, same preprocessing (float32,
nan_to_num) for every candidate - required by the protocol's own ground rule
that a comparison spanning different preprocessing is not a model comparison.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from obsidianchain.ml import diagnostics, protocol  # noqa: E402
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS  # noqa: E402

DS = ROOT / "data/models/ps_native/datasets"
OUT = ROOT / "research/autoresearch_2026_09_23/results/exp01_model_family.json"
LEDGER = ROOT / "research/autoresearch_2026_09_23/experiments.jsonl"
BASE, STEP = 1400000000, 1209600
SEED = 20260919


def load_dev() -> pd.DataFrame:
    dev = pd.concat(
        [pd.read_parquet(DS / "train.parquet"), pd.read_parquet(DS / "validation.parquet")],
        ignore_index=True,
    )
    dev["first_t"] = ((dev["timestamp"] - BASE) // STEP + 1).astype(int)
    return dev


def make_fit_predict(kind: str):
    def fit_predict(train, ev, features, seed):
        X = np.nan_to_num(train[features].to_numpy("float32"))
        y = train["y"].to_numpy("int8")
        Xe = np.nan_to_num(ev[features].to_numpy("float32"))

        if kind == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_leaf=20,
                random_state=seed, n_jobs=-1,
            )
        elif kind == "ExtraTrees":
            from sklearn.ensemble import ExtraTreesClassifier
            m = ExtraTreesClassifier(
                n_estimators=100, max_depth=12, min_samples_leaf=20,
                random_state=seed, n_jobs=-1,
            )
        elif kind == "HistGradientBoosting":
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(
                max_depth=6, learning_rate=0.05, max_iter=150,
                l2_regularization=0.0, random_state=seed,
            )
        elif kind == "LightGBM":
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(
                n_estimators=100, learning_rate=0.05, random_state=seed,
                n_jobs=-1, verbose=-1,
            )
        elif kind == "LogisticRegression_L2":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(penalty="l2", max_iter=1000, random_state=seed),
            )
        elif kind == "LogisticRegression_L1":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="l1", solver="liblinear", max_iter=1000, random_state=seed,
                ),
            )
        else:
            raise ValueError(kind)

        m.fit(X, y)
        return m.predict_proba(Xe)[:, 1]

    return fit_predict


CANDIDATES = [
    "LogisticRegression_L2",
    "LogisticRegression_L1",
    "RandomForest",
    "ExtraTrees",
    "HistGradientBoosting",
    "LightGBM",
]


def main() -> None:
    dev = load_dev()
    declared = list(CORE_PS_FEATURE_COLUMNS)
    healthy = diagnostics.healthy_features(dev, declared)
    print(f"development rows {len(dev):,}  timesteps {dev.first_t.min()}-{dev.first_t.max()}")
    print(f"features declared {len(declared)} -> healthy {len(healthy)}")

    results = {}
    timings = {}
    for kind in CANDIDATES:
        t0 = time.time()
        r = protocol.evaluate_candidate(
            dev, make_fit_predict(kind), name=kind, seed=SEED, features=healthy,
        )
        timings[kind] = time.time() - t0
        results[kind] = r
        s = r.summary()
        print(
            f"{kind:<24} nAP mean {s['nap_mean']:.4f}  sd {s['nap_sd']:.4f}  "
            f"min {s['nap_min']:.4f}  max {s['nap_max']:.4f}  "
            f"({s['folds']} folds, {timings[kind]:.1f}s)"
        )

    family = protocol.compare_family([results[n] for n in CANDIDATES])
    print(f"\nfamily_size={family['family_size']}  alpha={family['alpha']}  correction={family['correction']}")
    for c in family["comparisons"]:
        lo, hi = c["ci95"]
        flag = " !UNDERPOWERED" if c.get("underpowered_for") else ""
        print(
            f"  {c['label']:<48} diff {c['mean_difference']:>+8.4f} "
            f"[{lo:>+7.3f},{hi:>+7.3f}]  p={c['p_value']:.3f} p_holm={c['p_holm']:.3f}  "
            f"{c['verdict']}{' -> ' + c['favours'] if c['favours'] else ''}{flag}"
        )

    payload = {
        "experiment_id": "exp01_model_family",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "scope": protocol.SCOPE_PS_NATIVE,
        "feature_schema_version": "ps_native_features/2",
        "seed": SEED,
        "candidates": {k: v.summary() for k, v in results.items()},
        "family_comparison": family,
        "timings_sec": timings,
        "notes": (
            "XGBoost not tested: not installed / not in the vendored offline "
            "wheel set. Dataset has no network-layer (Group E) columns - "
            "include_network=False in build_ps_dataset.py - so this "
            "experiment covers blockchain-only PS-native features only."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT}")

    ledger_entry = {
        "experiment_id": "exp01_model_family",
        "hypothesis": (
            "No candidate family among {LogReg-L2, LogReg-L1, RandomForest, "
            "ExtraTrees, HistGradientBoosting, LightGBM} is assumed superior "
            "a priori; measure all six under the same protocol/features/folds."
        ),
        "baseline": "none (first family-wide comparison of these six)",
        "single_change": "model family only; features/folds/preprocessing/seed held fixed",
        "dataset_version": "ps_native_features/2 (train+validation, dev-only)",
        "feature_version": "ps_native_features/2 CORE (24 healthy columns)",
        "model_version": "n/a - exploratory, no artifact frozen",
        "evaluation_protocol": protocol.PROTOCOL_VERSION,
        "seeds": [SEED],
        "folds": 12,
        "metrics": {k: v.summary()["nap_mean"] for k, v in results.items()},
        "runtime_sec": timings,
        "statistical_comparison": "paired t-test, Holm-Bonferroni over 15 pairwise comparisons",
        "decision": "SEE_RESULTS_JSON",
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(ledger_entry) + "\n")


if __name__ == "__main__":
    main()
