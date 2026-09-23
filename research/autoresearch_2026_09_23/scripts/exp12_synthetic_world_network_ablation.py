"""EXPERIMENT exp12_synthetic_world_network_ablation.

Follow-up to exp11. On the ONLY dataset in this entire program where chain
structure and network structure share a real generative cause (the
synthetic world), measure whether network telemetry (Group E) adds
anything on top of Groups A-D - for the first time this program has been
able to ask this question at all (real Elliptic++ has no network layer
joined into the PS-native dataset; H2).

Explicit statistical caveat, stated before any result: exp11 found only 3
of the standard protocol's 12 rolling folds are usable at this world's
current size (944 usable rows total). This experiment reports descriptive,
exploratory results on those 3 folds - NOT a KEEP/REJECT claim under the
program's normal MDE/power standard, which requires far more folds than
this world currently provides. Treated explicitly as a pilot for whether a
larger purpose-built instance would be worth generating.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from obsidianchain.ml import diagnostics, protocol  # noqa: E402
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS  # noqa: E402

RESULTS = ROOT / "research/autoresearch_2026_09_23/results"
DEV_PATH = RESULTS / "exp11_synthetic_world_dev.parquet"
OUT = RESULTS / "exp12_synthetic_world_network_ablation.json"
LEDGER = ROOT / "research/autoresearch_2026_09_23/experiments.jsonl"
SEED = 20260919


def fit_predict(train, ev, features, seed):
    from sklearn.ensemble import RandomForestClassifier
    # RandomForest, not LightGBM: at ~700-900 total rows, LightGBM's default
    # leaf/split minimums are a poor match for this scale, and this
    # experiment does not need model-family generalization (exp01/exp06/
    # exp07 already established that question for the main dataset) - it
    # needs ONE reasonable, well-understood model to isolate the FEATURE
    # question at this small scale.
    X = np.nan_to_num(train[features].to_numpy("float32"))
    y = train["y"].to_numpy("int8")
    Xe = np.nan_to_num(ev[features].to_numpy("float32"))
    m = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=3,
        random_state=seed, n_jobs=-1,
    )
    m.fit(X, y)
    return m.predict_proba(Xe)[:, 1]


def main() -> None:
    dev = pd.read_parquet(DEV_PATH)
    declared = list(CORE_PS_FEATURE_COLUMNS)
    healthy_core = diagnostics.healthy_features(dev, declared)

    net_correct_cols = ["network_observation_count", "observer_diversity", "peer_count", "asn_count"]
    net_stub_cols = ["network_observation_count_STUB", "observer_diversity_STUB",
                      "peer_count_STUB", "asn_count_STUB"]
    healthy_with_correct_net = diagnostics.healthy_features(dev, declared + net_correct_cols)
    healthy_with_stub_net = diagnostics.healthy_features(dev, declared + net_stub_cols)

    print(f"n rows {len(dev)}  CORE healthy {len(healthy_core)}  "
          f"CORE+correct-net healthy {len(healthy_with_correct_net)}  "
          f"CORE+stub-net healthy {len(healthy_with_stub_net)}")
    print(f"correct-net columns surviving health check: "
          f"{[c for c in net_correct_cols if c in healthy_with_correct_net]}")
    print(f"stub-net columns surviving health check: "
          f"{[c for c in net_stub_cols if c in healthy_with_stub_net]} "
          f"(expected: fewer or none - constants/duplicates get filtered)")

    folds_all = protocol.rolling_origin_folds()

    variants = {
        "CORE_only": healthy_core,
        "CORE_plus_correct_network": healthy_with_correct_net,
        "CORE_plus_stub_network": healthy_with_stub_net,
    }
    results = {}
    for name, cols in variants.items():
        r = protocol.evaluate_candidate(
            dev, fit_predict, name=name, seed=SEED, features=cols,
            folds=folds_all, scope=protocol.SCOPE_UNDECLARED,
            # UNDECLARED, deliberately: this is neither SCOPE_PHASE6 nor
            # SCOPE_PS_NATIVE - it is a pilot on a third, synthetic-only
            # dataset, and protocol.py's own scope machinery exists
            # precisely to stop a number like this from being pooled with
            # either real-data scope by accident.
        )
        results[name] = r
        s = r.summary()
        print(f"\n{name}: {s['folds']} usable folds")
        for f in r.folds:
            print(f"  {f.fold.label:<20} n_eval={f.n_eval:<4} prevalence={f.prevalence:.3f} nap={f.nap:.4f}")
        print(f"  MEAN nap = {s['nap_mean']:.4f}  sd = {s['nap_sd']:.4f}")

    # descriptive paired comparison only - explicitly NOT run through
    # compare_family's Holm correction, which assumes enough folds for the
    # permutation/MDE machinery to mean something. Reported as raw paired
    # differences with a stated caveat, not a verdict.
    print("\n=== descriptive paired comparison (NOT a KEEP/REJECT verdict - too few folds) ===")
    names = list(results)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            a_naps = {f.fold.label: f.nap for f in results[a].folds}
            b_naps = {f.fold.label: f.nap for f in results[b].folds}
            common = sorted(set(a_naps) & set(b_naps))
            diffs = [a_naps[k] - b_naps[k] for k in common]
            print(f"  {a} vs {b}: per-fold diffs = {[round(d,4) for d in diffs]}  "
                  f"mean = {np.mean(diffs):+.4f}  (n={len(diffs)} folds - too few for a significance test)")

    payload = {
        "experiment_id": "exp12_synthetic_world_network_ablation",
        "caveat": (
            "DESCRIPTIVE / EXPLORATORY ONLY. Only 3 folds usable at this "
            "world's current size - far below this program's normal "
            "12-fold, Holm-corrected standard. No KEEP/REJECT verdict is "
            "issued; see 16_synthetic_world_investigation.md for the full "
            "reasoning and recommendation."
        ),
        "results": {k: v.summary() for k, v in results.items()},
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT}")

    ledger_entry = {
        "experiment_id": "exp12_synthetic_world_network_ablation",
        "hypothesis": (
            "On the synthetic world (the only dataset in this program where "
            "chain and network structure share a real generative cause), "
            "does a PROPERLY aggregated Group E network feature set add "
            "value over Groups A-D, and does it differ from the current "
            "production STUB implementation (confirmed constant in exp11)?"
        ),
        "baseline": "CORE_only (Groups A-D, unmodified production feature engine)",
        "single_change": "network feature variant: none / properly-aggregated / production-stub",
        "dataset_version": "SYNTHETIC_CONTROL world (data/synthetic_world/), NOT ps_native_features scope",
        "feature_version": "CORE + Group E variants (research-only aggregation for the 'correct' variant)",
        "model_version": "RandomForest (200 trees, depth 5, leaf 3) - scaled down for ~900-row dataset",
        "evaluation_protocol": "standard rolling_origin_folds(), only 3 of 12 usable at this data volume",
        "seeds": [SEED], "folds": 3,
        "metrics": {k: v.summary()["nap_mean"] for k, v in results.items()},
        "statistical_comparison": "descriptive only - too few folds for compare_family's Holm/permutation machinery",
        "decision": "SEE_RESULTS_JSON / 16_synthetic_world_investigation.md",
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(ledger_entry) + "\n")


if __name__ == "__main__":
    main()
