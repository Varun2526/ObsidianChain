"""EXPERIMENT exp07_ablation_rf_confirmation.

Directly closes the scope limitation 13_red_team_review.md flagged against
exp02_feature_ablation: "Group A dominant / Group C,D weak" was measured on
LightGBM only. This repeats the same leave-one-group-out design (full,
minus_A, minus_B, minus_C, minus_D) on RandomForest - a different tree
mechanism (bagging vs boosting) - to check whether the finding is a property
of the DATA or an artefact of one model family.
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
OUT = ROOT / "research/autoresearch_2026_09_23/results/exp07_ablation_rf_confirmation.json"
LEDGER = ROOT / "research/autoresearch_2026_09_23/experiments.jsonl"
BASE, STEP = 1400000000, 1209600
SEED = 20260919

CORE_GROUPS = {k: v for k, v in PS_FEATURE_GROUPS.items() if k != "E_network"}


def load_dev() -> pd.DataFrame:
    dev = pd.concat(
        [pd.read_parquet(DS / "train.parquet"), pd.read_parquet(DS / "validation.parquet")],
        ignore_index=True,
    )
    dev["first_t"] = ((dev["timestamp"] - BASE) // STEP + 1).astype(int)
    return dev


def fit_predict(train, ev, features, seed):
    from sklearn.ensemble import RandomForestClassifier
    X = np.nan_to_num(train[features].to_numpy("float32"))
    y = train["y"].to_numpy("int8")
    Xe = np.nan_to_num(ev[features].to_numpy("float32"))
    m = RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=20,
        random_state=seed, n_jobs=-1,
    )
    m.fit(X, y)
    return m.predict_proba(Xe)[:, 1]


def build_variants(healthy: list[str]) -> dict[str, list[str]]:
    healthy_set = set(healthy)
    variants = {"full": list(healthy)}
    for group, cols in CORE_GROUPS.items():
        group_healthy = [c for c in cols if c in healthy_set]
        variants[f"minus_{group}"] = [c for c in healthy if c not in group_healthy]
    return variants


def main() -> None:
    dev = load_dev()
    declared = list(CORE_PS_FEATURE_COLUMNS)
    healthy = diagnostics.healthy_features(dev, declared)
    variants = build_variants(healthy)

    print(f"development rows {len(dev):,}  healthy features {len(healthy)}  model=RandomForest")

    results = {}
    timings = {}
    for name, cols in variants.items():
        t0 = time.time()
        r = protocol.evaluate_candidate(
            dev, fit_predict, name=name, seed=SEED, features=cols,
            scope=protocol.SCOPE_PS_NATIVE,
        )
        timings[name] = time.time() - t0
        results[name] = r
        s = r.summary()
        print(f"{name:<14} nAP mean {s['nap_mean']:.4f}  sd {s['nap_sd']:.4f}  ({s['folds']} folds, {timings[name]:.1f}s)")

    names = list(results)
    family = protocol.compare_family([results[n] for n in names])
    print(f"\nfamily_size={family['family_size']}")
    for c in family["comparisons"]:
        if "full" not in c["label"]:
            continue
        lo, hi = c["ci95"]
        flag = " !UNDERPOWERED" if c.get("underpowered_for") else ""
        print(
            f"  {c['label']:<24} diff {c['mean_difference']:>+8.4f} "
            f"[{lo:>+7.3f},{hi:>+7.3f}]  p={c['p_value']:.3f} p_holm={c['p_holm']:.3f}  "
            f"{c['verdict']}{' -> ' + c['favours'] if c['favours'] else ''}{flag}"
        )

    # cross-model comparison: does RF's minus_C/minus_D pattern match LightGBM's (exp02)?
    exp02 = json.loads((ROOT / "research/autoresearch_2026_09_23/results/exp02_feature_ablation.json").read_text())
    print("\ncross-model comparison (RandomForest here vs LightGBM in exp02):")
    for variant in ["full", "minus_A_transaction", "minus_B_address_history", "minus_C_graph", "minus_D_patterns"]:
        rf_nap = results[variant].summary()["nap_mean"] if variant in results else None
        lgbm_nap = exp02["results"].get(variant, {}).get("nap_mean")
        print(f"  {variant:<26} RF={rf_nap:.4f}  LightGBM(exp02)={lgbm_nap:.4f}  diff={rf_nap-lgbm_nap:+.4f}")

    payload = {
        "experiment_id": "exp07_ablation_rf_confirmation",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "scope": protocol.SCOPE_PS_NATIVE,
        "model": "RandomForest (confirmation model, exp02 used LightGBM)",
        "seed": SEED,
        "results": {k: v.summary() for k, v in results.items()},
        "family_comparison": family,
        "timings_sec": timings,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT}")

    ledger_entry = {
        "experiment_id": "exp07_ablation_rf_confirmation",
        "hypothesis": (
            "13_red_team_review.md flagged exp02's Group-A-dominant / "
            "Group-C,D-weak finding as measured on LightGBM only. Repeat the "
            "same leave-one-group-out design on RandomForest to check "
            "whether the finding is data-driven or a LightGBM-specific artefact."
        ),
        "baseline": "exp02 LightGBM full/minus_X results",
        "single_change": "model=RandomForest instead of LightGBM; features/folds/seed fixed",
        "dataset_version": "ps_native_features/2", "feature_version": "CORE 24 healthy, group ablations",
        "model_version": "RandomForest, same hyperparameters as exp01/exp06",
        "evaluation_protocol": protocol.PROTOCOL_VERSION,
        "seeds": [SEED], "folds": 12,
        "metrics": {k: v.summary()["nap_mean"] for k, v in results.items()},
        "runtime_sec": timings,
        "statistical_comparison": "paired t-test, Holm-Bonferroni across ablation family",
        "decision": "SEE_RESULTS_JSON",
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(ledger_entry) + "\n")


if __name__ == "__main__":
    main()
