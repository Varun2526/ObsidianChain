"""EXPERIMENT exp06_model_search_round2.

Continuation of Phase 5 from the 2026-09-23 checkpoint. exp01 already
established (see 05_model_comparison.md, re-verified by reading
results/exp01_model_family.json before writing this script):
  - RandomForest / HistGradientBoosting / LightGBM: statistically tied
    (pairwise diffs 0.009-0.019 nAP, MDE=0.164) -> worth deeper variant
    investigation, NOT worth re-running as-is.
  - ExtraTrees, LogisticRegression (L1/L2): decisively beaten
    (p_holm<0.03 in every pairwise comparison against the tied tier) ->
    deprioritized as sole candidates. ExtraTrees left one open question in
    05_model_comparison.md: "whether the gap closes at higher n_estimators
    - not tested." Addressed here rather than ignored.

This experiment does NOT re-run LogisticRegression or plain LightGBM/RF/HGB
at their exp01 settings. Instead it:
  1. Reconstructs exp01's LightGBM/RandomForest/HistGradientBoosting/
     ExtraTrees CandidateResults from the SAVED per-fold JSON (not
     refitting), so new candidates can be compared against them with a
     valid paired-by-fold statistical test without wasting compute
     re-deriving numbers already on record.
  2. Trains and evaluates justified new variants, each addressing a
     specific open question:
       - LightGBM_balanced, RandomForest_balanced, HistGradientBoosting_balanced:
         class-weighted variants - none of exp01's candidates used class
         weighting despite 5.5% prevalence; a specific, justified gap.
       - LightGBM_deeper: tests the model-CAPACITY bottleneck hypothesis
         directly (more estimators/leaves, lower learning rate) - exp01's
         tied-tier finding used fixed, modest capacity for all three;
         whether capacity is the ceiling (vs. features, per 06_ablation_
         results.md) was never tested by varying capacity within one family.
       - LightGBM_lambdarank: a genuine ranking-oriented objective (NDCG-style
         listwise loss), justified because the problem itself is framed as
         RANKING, not classification - every other candidate to date
         optimizes a classification loss (log-loss / Gini) and only gets
         evaluated as a ranker after the fact.
       - ExtraTrees_500trees: directly answers 05_model_comparison.md's
         open question about ExtraTrees at higher n_estimators.
  3. XGBoost: NOT TESTED. Confirmed via `import xgboost` -> ModuleNotFoundError
     in this environment; not in the vendored offline wheel set (matches
     01_repo_audit.md). Recorded as NOT_TESTED (environment constraint),
     never as a negative result.

Compared as ONE Holm-corrected family spanning old (reconstructed) +
new candidates, so the multiplicity correction covers the full comparison
set, not just the new ones in isolation.
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
EXP01 = ROOT / "research/autoresearch_2026_09_23/results/exp01_model_family.json"
OUT = ROOT / "research/autoresearch_2026_09_23/results/exp06_model_search_round2.json"
LEDGER = ROOT / "research/autoresearch_2026_09_23/experiments.jsonl"
BASE, STEP = 1400000000, 1209600
SEED = 20260919

# Reconstructed (not re-run), per the explicit instruction not to repeat
# LogisticRegression/LightGBM without a specific unresolved hypothesis.
RECONSTRUCT_FROM_EXP01 = ["RandomForest", "HistGradientBoosting", "LightGBM", "ExtraTrees"]


def load_dev() -> pd.DataFrame:
    dev = pd.concat(
        [pd.read_parquet(DS / "train.parquet"), pd.read_parquet(DS / "validation.parquet")],
        ignore_index=True,
    )
    dev["first_t"] = ((dev["timestamp"] - BASE) // STEP + 1).astype(int)
    return dev


def reconstruct_candidate(name: str, summary: dict, folds_by_label: dict) -> protocol.CandidateResult:
    """Rebuild a CandidateResult from exp01's saved summary() JSON.

    Exact, not approximate: every per-fold nap/prevalence/n/precision_at_k
    value is read back verbatim from the file exp01 wrote; nothing is
    recomputed or refit.
    """
    fold_results = []
    for row in summary["per_fold"]:
        fold = folds_by_label[row["fold"]]
        fold_results.append(protocol.FoldResult(
            fold=fold,
            n_train=row["n_train"],
            n_eval=row["n_eval"],
            prevalence=row["prevalence"],
            nap=row["nap"],
            p_at_k_worst=row["precision_at_k_worst"],
            p_at_k_best=row["precision_at_k_best"],
        ))
    return protocol.CandidateResult(
        name=f"{name}(exp01)", seed=summary["seed"], folds=fold_results,
        scope=protocol.SCOPE_PS_NATIVE,
    )


def make_fit_predict(kind: str):
    def fit_predict(train, ev, features, seed):
        X = np.nan_to_num(train[features].to_numpy("float32"))
        y = train["y"].to_numpy("int8")
        Xe = np.nan_to_num(ev[features].to_numpy("float32"))

        if kind == "LightGBM_balanced":
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(
                n_estimators=100, learning_rate=0.05, random_state=seed,
                n_jobs=-1, verbose=-1, is_unbalance=True,
            )
        elif kind == "LightGBM_deeper":
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=63,
                min_child_samples=20, random_state=seed, n_jobs=-1, verbose=-1,
            )
        elif kind == "LightGBM_lambdarank":
            from lightgbm import LGBMRanker
            # Listwise NDCG-style ranking objective. No natural "query" unit
            # in this data, and a single group spanning the whole fold hits
            # LightGBM's hard per-query row cap (10,000) - confirmed by a
            # failed first attempt (see 05_model_comparison.md addendum).
            # Chunking into contiguous ~9,000-row groups is a documented
            # engineering compromise (optimizes NDCG WITHIN each chunk, not
            # globally), not a claim that this is the "correct" way to apply
            # LambdaRank here - reported as such.
            CHUNK = 9000
            n = len(y)
            groups = [CHUNK] * (n // CHUNK)
            remainder = n % CHUNK
            if remainder:
                groups.append(remainder)
            m = LGBMRanker(
                objective="lambdarank", n_estimators=100, learning_rate=0.05,
                random_state=seed, n_jobs=-1, verbose=-1,
            )
            m.fit(X, y, group=groups)
            return m.predict(Xe)
        elif kind == "RandomForest_balanced":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_leaf=20,
                random_state=seed, n_jobs=-1, class_weight="balanced_subsample",
            )
        elif kind == "HistGradientBoosting_balanced":
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(
                max_depth=6, learning_rate=0.05, max_iter=150,
                random_state=seed, class_weight="balanced",
            )
        elif kind == "ExtraTrees_500trees":
            from sklearn.ensemble import ExtraTreesClassifier
            m = ExtraTreesClassifier(
                n_estimators=500, max_depth=12, min_samples_leaf=20,
                random_state=seed, n_jobs=-1,
            )
        else:
            raise ValueError(kind)

        m.fit(X, y)
        return m.predict_proba(Xe)[:, 1]

    return fit_predict


NEW_CANDIDATES = [
    "LightGBM_balanced",
    "LightGBM_deeper",
    "LightGBM_lambdarank",
    "RandomForest_balanced",
    "HistGradientBoosting_balanced",
    "ExtraTrees_500trees",
]


def main() -> None:
    exp01 = json.loads(EXP01.read_text())
    print("=== Step 1: read back exp01 (not re-run) ===")
    for name in RECONSTRUCT_FROM_EXP01:
        s = exp01["candidates"][name]
        print(f"  {name:<22} nAP mean={s['nap_mean']:.4f} sd={s['nap_sd']:.4f} (from exp01, reconstructed)")

    dev = load_dev()
    declared = list(CORE_PS_FEATURE_COLUMNS)
    healthy = diagnostics.healthy_features(dev, declared)
    folds = protocol.rolling_origin_folds()
    folds_by_label = {f.label: f for f in folds}

    reconstructed = {
        name: reconstruct_candidate(name, exp01["candidates"][name], folds_by_label)
        for name in RECONSTRUCT_FROM_EXP01
    }

    print("\n=== Step 2: run new, justified candidates ===")
    results = dict(reconstructed)
    timings = {}
    for kind in NEW_CANDIDATES:
        t0 = time.time()
        try:
            r = protocol.evaluate_candidate(
                dev, make_fit_predict(kind), name=kind, seed=SEED, features=healthy,
                scope=protocol.SCOPE_PS_NATIVE,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {kind:<28} FAILED: {exc}")
            continue
        timings[kind] = time.time() - t0
        results[kind] = r
        s = r.summary()
        print(
            f"  {kind:<28} nAP mean {s['nap_mean']:.4f}  sd {s['nap_sd']:.4f}  "
            f"min {s['nap_min']:.4f}  max {s['nap_max']:.4f}  "
            f"({s['folds']} folds, {timings[kind]:.1f}s)"
        )

    names = list(results)
    family = protocol.compare_family([results[n] for n in names])
    print(f"\n=== Step 3: family comparison (size={family['family_size']}) ===")
    # Only print comparisons touching a NEW candidate - the old-vs-old
    # entries are already on record in 05_model_comparison.md.
    new_labels = set(NEW_CANDIDATES)
    for c in family["comparisons"]:
        left, right = c["label"].split(" vs ")
        left_is_new = left in new_labels
        right_is_new = right in new_labels
        if not (left_is_new or right_is_new):
            continue
        lo, hi = c["ci95"]
        flag = " !UNDERPOWERED" if c.get("underpowered_for") else ""
        print(
            f"  {c['label']:<58} diff {c['mean_difference']:>+8.4f} "
            f"[{lo:>+7.3f},{hi:>+7.3f}]  p={c['p_value']:.3f} p_holm={c['p_holm']:.3f}  "
            f"{c['verdict']}{' -> ' + c['favours'] if c['favours'] else ''}{flag}"
        )

    payload = {
        "experiment_id": "exp06_model_search_round2",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "scope": protocol.SCOPE_PS_NATIVE,
        "reconstructed_from_exp01": RECONSTRUCT_FROM_EXP01,
        "new_candidates": NEW_CANDIDATES,
        "not_tested": {"XGBoost": "ModuleNotFoundError - not installed, not in vendored offline wheel set"},
        "seed": SEED,
        "results": {k: v.summary() for k, v in results.items()},
        "family_comparison": family,
        "timings_sec": timings,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT}")

    ledger_entry = {
        "experiment_id": "exp06_model_search_round2",
        "hypothesis": (
            "Given exp01's tied tree tier {RF,HGB,LightGBM} and rejected "
            "{ExtraTrees,LogReg}, do justified variants (class-weighting, "
            "capacity, a native ranking objective, more ExtraTrees estimators) "
            "produce a defensible improvement over the tied tier, compared "
            "without re-running exp01's original candidates?"
        ),
        "baseline": "exp01 LightGBM/RandomForest/HistGradientBoosting/ExtraTrees (reconstructed from saved per-fold results, not refit)",
        "single_change": "model variant only; features/folds/preprocessing/seed fixed",
        "dataset_version": "ps_native_features/2 (train+validation, dev-only)",
        "feature_version": "ps_native_features/2 CORE (24 healthy)",
        "model_version": "n/a - exploratory",
        "evaluation_protocol": protocol.PROTOCOL_VERSION,
        "seeds": [SEED],
        "folds": 12,
        "metrics": {k: v.summary()["nap_mean"] for k, v in results.items()},
        "runtime_sec": timings,
        "statistical_comparison": "paired t-test, Holm-Bonferroni across full (reconstructed+new) family",
        "decision": "SEE_RESULTS_JSON",
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(ledger_entry) + "\n")


if __name__ == "__main__":
    main()
