"""EXPERIMENT exp09_alert_policy_redesign.

Phase 8 continuation. 09_alert_policy_analysis.md found the frozen v1
severity bands' recorded validation_precision matches their own optimization
target exactly (90.0/75.0/50.0) - i.e. thresholds were picked and reported
on the SAME sample. This experiment builds and tests the corrected
procedure: three genuinely separate splits -
  fit_set          - trains the model
  threshold_set     - held out from fitting, used ONLY to pick band cutoffs
  eval (the fold's actual eval window) - reports precision, touched by
                      neither fitting nor threshold selection

and directly demonstrates the flaw by ALSO picking thresholds on the eval
window itself (reproducing the v1 pattern) and comparing the two resulting
precision numbers on a THIRD, never-touched slice - showing which one
generalizes.
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

DS = ROOT / "data/models/ps_native/datasets"
OUT = ROOT / "research/autoresearch_2026_09_23/results/exp09_alert_policy_redesign.json"
LEDGER = ROOT / "research/autoresearch_2026_09_23/experiments.jsonl"
BASE, STEP = 1400000000, 1209600
SEED = 20260919
TARGETS = {"CRITICAL": 0.90, "HIGH": 0.75, "MEDIUM": 0.50}


def load_dev() -> pd.DataFrame:
    dev = pd.concat(
        [pd.read_parquet(DS / "train.parquet"), pd.read_parquet(DS / "validation.parquet")],
        ignore_index=True,
    )
    dev["first_t"] = ((dev["timestamp"] - BASE) // STEP + 1).astype(int)
    return dev


def threshold_for_target(y_true: np.ndarray, scores: np.ndarray, target_precision: float):
    """Highest-score-first cutoff whose precision-at-and-above first reaches
    target_precision. Returns (threshold, achieved_precision, support) or
    None if unreachable."""
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    s_sorted = scores[order]
    cum_pos = np.cumsum(y_sorted)
    counts = np.arange(1, len(y_sorted) + 1)
    precision_at_n = cum_pos / counts
    reachable = precision_at_n >= target_precision
    if not reachable.any():
        return None
    # largest n (most support) that still clears the target - matches the
    # "as many alerts as possible at this confidence tier" framing of a
    # severity band.
    n = int(np.max(np.where(reachable)[0])) + 1
    return float(s_sorted[n - 1]), float(precision_at_n[n - 1]), int(n)


def precision_of_band(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    mask = scores >= threshold
    if mask.sum() == 0:
        return float("nan"), 0
    return float(y_true[mask].mean()), int(mask.sum())


def main() -> None:
    dev = load_dev()
    declared = list(CORE_PS_FEATURE_COLUMNS)
    healthy = diagnostics.healthy_features(dev, declared)
    folds = protocol.rolling_origin_folds()
    fold = folds[-1]
    usable = protocol.development(dev)
    train_full = usable[usable["first_t"] <= fold.train_end]
    eval_window = usable[(usable["first_t"] >= fold.eval_start) & (usable["first_t"] <= fold.eval_end)].copy()

    # Three-way split of TRAIN only: fit / threshold-selection / a held-out
    # "second sample" to prove generalization. eval_window is the fold's own
    # eval window and is used only for final reporting, never for thresholds.
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(train_full))
    n = len(idx)
    fit_end = int(n * 0.70)
    thresh_end = int(n * 0.85)
    fit_set = train_full.iloc[idx[:fit_end]]
    threshold_set = train_full.iloc[idx[fit_end:thresh_end]]
    holdout_check_set = train_full.iloc[idx[thresh_end:]]  # never used for thresholds

    from lightgbm import LGBMClassifier
    Xf = np.nan_to_num(fit_set[healthy].to_numpy("float32"))
    yf = fit_set["y"].to_numpy("int8")
    m = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(Xf, yf)

    def score(frame):
        X = np.nan_to_num(frame[healthy].to_numpy("float32"))
        return m.predict_proba(X)[:, 1]

    s_threshold_set = score(threshold_set)
    y_threshold_set = threshold_set["y"].to_numpy("int8")
    s_holdout_check = score(holdout_check_set)
    y_holdout_check = holdout_check_set["y"].to_numpy("int8")
    s_eval = score(eval_window)
    y_eval = eval_window["y"].to_numpy("int8")

    print(f"fit_set={len(fit_set)}  threshold_set={len(threshold_set)}  "
          f"holdout_check_set={len(holdout_check_set)}  eval_window={len(eval_window)}")

    print("\n=== CORRECTED procedure: thresholds picked on threshold_set (never fit, never eval) ===")
    corrected_bands = {}
    for band, target in TARGETS.items():
        result = threshold_for_target(y_threshold_set, s_threshold_set, target)
        if result is None:
            print(f"  {band}: target {target:.0%} UNREACHABLE on threshold_set")
            continue
        thr, achieved, support = result
        # report on THREE different samples: the set it was picked on
        # (in-sample, expected to hit target by construction), a held-out
        # slice never used for anything (generalization check #1), and the
        # actual fold eval window (generalization check #2, what production
        # would see).
        p_holdout, n_holdout = precision_of_band(y_holdout_check, s_holdout_check, thr)
        p_eval, n_eval_band = precision_of_band(y_eval, s_eval, thr)
        corrected_bands[band] = {
            "threshold": thr, "target": target,
            "precision_on_threshold_set_insample": achieved, "support_on_threshold_set": support,
            "precision_on_holdout_check_set": p_holdout, "support_on_holdout_check_set": n_holdout,
            "precision_on_eval_window": p_eval, "support_on_eval_window": n_eval_band,
        }
        print(f"  {band:<10} threshold={thr:.4f}  in-sample={achieved:.3f}  "
              f"holdout-check={p_holdout:.3f} (n={n_holdout})  eval-window={p_eval:.3f} (n={n_eval_band})")

    print("\n=== REPRODUCING THE FLAW: thresholds picked directly on eval_window (matches v1's own-sample pattern) ===")
    flawed_bands = {}
    for band, target in TARGETS.items():
        result = threshold_for_target(y_eval, s_eval, target)
        if result is None:
            print(f"  {band}: target {target:.0%} UNREACHABLE on eval_window")
            continue
        thr, achieved, support = result
        # report the SAME threshold against holdout_check_set - a sample
        # this threshold has never seen either, for a fair apples-to-apples
        # generalization comparison against the corrected procedure above.
        p_holdout, n_holdout = precision_of_band(y_holdout_check, s_holdout_check, thr)
        flawed_bands[band] = {
            "threshold": thr, "target": target,
            "precision_on_eval_window_insample": achieved, "support_on_eval_window": support,
            "precision_on_holdout_check_set": p_holdout, "support_on_holdout_check_set": n_holdout,
        }
        print(f"  {band:<10} threshold={thr:.4f}  in-sample(eval)={achieved:.3f}  "
              f"holdout-check={p_holdout:.3f} (n={n_holdout})  "
              f"gap={achieved - p_holdout:+.3f}")

    payload = {
        "experiment_id": "exp09_alert_policy_redesign",
        "fold": fold.label,
        "split_sizes": {"fit": len(fit_set), "threshold": len(threshold_set),
                         "holdout_check": len(holdout_check_set), "eval_window": len(eval_window)},
        "corrected_procedure": corrected_bands,
        "flawed_procedure_reproduction": flawed_bands,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT}")

    ledger_entry = {
        "experiment_id": "exp09_alert_policy_redesign",
        "hypothesis": (
            "A severity-band procedure that selects thresholds on a genuinely "
            "held-out split (never used for fitting or for the final reported "
            "precision) will show a smaller in-sample-to-generalization gap "
            "than reproducing the flawed same-sample procedure identified in "
            "09_alert_policy_analysis.md."
        ),
        "baseline": "v1 frozen calibration.json bands (thresholds and reported precision from the same sample)",
        "single_change": "threshold-selection split: corrected (separate) vs flawed (same-sample) procedure, same model/fold",
        "dataset_version": "ps_native_features/2", "feature_version": "CORE 24 healthy",
        "model_version": "LightGBM, exp01 hyperparameters, fit on 70pct of fold training data",
        "evaluation_protocol": "single-fold diagnostic, three-way split",
        "seeds": [SEED], "folds": 1,
        "metrics": {"see results json for per-band precision under both procedures": True},
        "statistical_comparison": "none - descriptive demonstration",
        "decision": "SEE_RESULTS_JSON",
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(ledger_entry) + "\n")


if __name__ == "__main__":
    main()
