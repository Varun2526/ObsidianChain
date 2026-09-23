"""EXPERIMENT exp02_feature_ablation.

Hypothesis (locked before running): given exp01's finding that RandomForest,
HistGradientBoosting and LightGBM are statistically indistinguishable (see
05_model_comparison.md), the open bottleneck question is NOT which tree
ensemble to ship but WHICH FEATURE GROUPS actually carry ranking signal.
This experiment isolates that with leave-one-group-out and single-group-only
ablations against the full v2 CORE feature set (Groups A/B/C/D; Group E
network is absent from this dataset - see 01_repo_audit.md/02_data_audit.md
- so it cannot be ablated here).

Single representative model: LightGBM (fastest of the three statistically
tied top-tier candidates from exp01, 7.3s for the full 12-fold family run vs
11.7s for RF/HGB) - repeating every ablation across all three tied models
would be a >3x cost multiplier for no expected additional information, since
exp01 already established they behave the same on the full feature set.
Same protocol, same folds, same preprocessing, same seed as exp01.

Variants:
  full            - all 24 healthy CORE columns (the exp01 baseline)
  minus_A / .. D  - full set minus one group (leave-one-group-out)
  only_A / .. D   - only one group's columns (isolates each group alone)

Compared as one Holm-corrected family via protocol.compare_family.
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
from obsidianchain.pipeline.features_ps import (  # noqa: E402
    CORE_PS_FEATURE_COLUMNS, PS_FEATURE_GROUPS,
)

DS = ROOT / "data/models/ps_native/datasets"
OUT = ROOT / "research/autoresearch_2026_09_23/results/exp02_feature_ablation.json"
LEDGER = ROOT / "research/autoresearch_2026_09_23/experiments.jsonl"
BASE, STEP = 1400000000, 1209600
SEED = 20260919

CORE_GROUPS = {
    k: v for k, v in PS_FEATURE_GROUPS.items() if k != "E_network"
}


def load_dev() -> pd.DataFrame:
    dev = pd.concat(
        [pd.read_parquet(DS / "train.parquet"), pd.read_parquet(DS / "validation.parquet")],
        ignore_index=True,
    )
    dev["first_t"] = ((dev["timestamp"] - BASE) // STEP + 1).astype(int)
    return dev


def fit_predict(train, ev, features, seed):
    from lightgbm import LGBMClassifier
    X = np.nan_to_num(train[features].to_numpy("float32"))
    y = train["y"].to_numpy("int8")
    Xe = np.nan_to_num(ev[features].to_numpy("float32"))
    m = LGBMClassifier(
        n_estimators=100, learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(X, y)
    return m.predict_proba(Xe)[:, 1]


def build_variants(healthy: list[str]) -> dict[str, list[str]]:
    healthy_set = set(healthy)
    variants = {"full": list(healthy)}
    for group, cols in CORE_GROUPS.items():
        group_healthy = [c for c in cols if c in healthy_set]
        variants[f"minus_{group}"] = [c for c in healthy if c not in group_healthy]
        variants[f"only_{group}"] = group_healthy
    return variants


def main() -> None:
    dev = load_dev()
    declared = list(CORE_PS_FEATURE_COLUMNS)
    healthy = diagnostics.healthy_features(dev, declared)
    variants = build_variants(healthy)

    print(f"development rows {len(dev):,}  healthy features {len(healthy)}")
    for name, cols in variants.items():
        print(f"  {name:<14} {len(cols)} cols: {cols}")

    results = {}
    timings = {}
    for name, cols in variants.items():
        if not cols:
            print(f"skipping {name}: zero usable columns")
            continue
        t0 = time.time()
        r = protocol.evaluate_candidate(
            dev, fit_predict, name=name, seed=SEED, features=cols,
        )
        timings[name] = time.time() - t0
        results[name] = r
        s = r.summary()
        print(
            f"{name:<14} nAP mean {s['nap_mean']:.4f}  sd {s['nap_sd']:.4f}  "
            f"({s['folds']} folds, {timings[name]:.1f}s)"
        )

    names = list(results)
    family = protocol.compare_family([results[n] for n in names])
    print(f"\nfamily_size={family['family_size']}")
    for c in family["comparisons"]:
        # Only print comparisons that involve 'full' - the leave-one-out /
        # only-one signal is what this experiment is measuring; the
        # minus-X-vs-only-Y cross terms are noise for this question.
        if "full" not in c["label"]:
            continue
        lo, hi = c["ci95"]
        flag = " !UNDERPOWERED" if c.get("underpowered_for") else ""
        print(
            f"  {c['label']:<32} diff {c['mean_difference']:>+8.4f} "
            f"[{lo:>+7.3f},{hi:>+7.3f}]  p={c['p_value']:.3f} p_holm={c['p_holm']:.3f}  "
            f"{c['verdict']}{' -> ' + c['favours'] if c['favours'] else ''}{flag}"
        )

    payload = {
        "experiment_id": "exp02_feature_ablation",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "scope": protocol.SCOPE_PS_NATIVE,
        "model": "LightGBM (fixed, representative of exp01's tied top tier)",
        "seed": SEED,
        "variants": {k: v for k, v in variants.items() if v},
        "results": {k: v.summary() for k, v in results.items()},
        "family_comparison": family,
        "timings_sec": timings,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT}")

    ledger_entry = {
        "experiment_id": "exp02_feature_ablation",
        "hypothesis": (
            "Given exp01 (RF/HGB/LightGBM statistically tied), the "
            "bottleneck question is which feature GROUP carries ranking "
            "signal, not which tree ensemble. Leave-one-group-out and "
            "single-group-only ablation over Groups A/B/C/D (Group E absent "
            "from this dataset) isolates each group's contribution."
        ),
        "baseline": "exp01 LightGBM full-feature nAP mean 0.5914",
        "single_change": "feature set only; model=LightGBM fixed, folds/seed/preprocessing fixed",
        "dataset_version": "ps_native_features/2 (train+validation, dev-only)",
        "feature_version": "ps_native_features/2 CORE, group ablations",
        "model_version": "LightGBM, same hyperparameters as exp01",
        "evaluation_protocol": protocol.PROTOCOL_VERSION,
        "seeds": [SEED],
        "folds": 12,
        "metrics": {k: v.summary()["nap_mean"] for k, v in results.items()},
        "runtime_sec": timings,
        "statistical_comparison": "paired t-test, Holm-Bonferroni across full ablation family",
        "decision": "SEE_RESULTS_JSON",
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(ledger_entry) + "\n")


if __name__ == "__main__":
    main()
