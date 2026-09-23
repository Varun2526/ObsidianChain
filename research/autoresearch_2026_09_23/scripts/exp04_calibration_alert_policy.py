"""EXPERIMENT exp04_calibration_alert_policy.

Phase 8. Investigates calibration SEPARATE from ranking (RULE 9), and
alert-policy performance SEPARATE from model score (RULE 10), directly
targeting the already-known problem from HANDOFF.md: "the current system
has evidence that rank-derived cutoffs are being used as value thresholds
and ties distort advertised alert counts" and "isotonic calibration
degrades ranking... resolution collapses ~10,000 distinct scores to ~55."

This experiment reproduces that isotonic-collapse finding on the PS-native
v2 scope specifically (the prior finding was reported without stating which
scope/dataset it was measured on) and separately evaluates alert-policy
precision/recall at fixed budgets under raw score vs isotonic-calibrated
score, on the same single diagnostic fold as exp03 (t<=38 -> t39-40) so the
two experiments are directly comparable.
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
OUT = ROOT / "research/autoresearch_2026_09_23/results/exp04_calibration_alert_policy.json"
BASE, STEP = 1400000000, 1209600
SEED = 20260919


def load_dev() -> pd.DataFrame:
    dev = pd.concat(
        [pd.read_parquet(DS / "train.parquet"), pd.read_parquet(DS / "validation.parquet")],
        ignore_index=True,
    )
    dev["first_t"] = ((dev["timestamp"] - BASE) // STEP + 1).astype(int)
    return dev


def main() -> None:
    dev = load_dev()
    declared = list(CORE_PS_FEATURE_COLUMNS)
    healthy = diagnostics.healthy_features(dev, declared)
    folds = protocol.rolling_origin_folds()
    fold = folds[-1]

    usable = protocol.development(dev)
    train_full = usable[usable["first_t"] <= fold.train_end]
    ev = usable[(usable["first_t"] >= fold.eval_start) & (usable["first_t"] <= fold.eval_end)].copy()

    # Split train into a fit set and a calibration set (isotonic needs
    # held-out predictions - fitting and calibrating on the same rows would
    # bias the calibrator toward overconfidence).
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(train_full))
    cut = int(len(idx) * 0.85)
    fit_idx, cal_idx = idx[:cut], idx[cut:]
    fit_set = train_full.iloc[fit_idx]
    cal_set = train_full.iloc[cal_idx]

    from lightgbm import LGBMClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss

    Xf = np.nan_to_num(fit_set[healthy].to_numpy("float32"))
    yf = fit_set["y"].to_numpy("int8")
    m = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(Xf, yf)

    Xc = np.nan_to_num(cal_set[healthy].to_numpy("float32"))
    raw_cal_scores = m.predict_proba(Xc)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal_scores, cal_set["y"].to_numpy("int8"))

    Xe = np.nan_to_num(ev[healthy].to_numpy("float32"))
    ev["raw_score"] = m.predict_proba(Xe)[:, 1]
    ev["cal_score"] = iso.predict(ev["raw_score"])

    n_distinct_raw = ev["raw_score"].nunique()
    n_distinct_cal = ev["cal_score"].nunique()
    print(f"distinct scores: raw={n_distinct_raw}  isotonic-calibrated={n_distinct_cal}  (eval n={len(ev)})")

    brier_raw = brier_score_loss(ev["y"], ev["raw_score"])
    brier_cal = brier_score_loss(ev["y"], ev["cal_score"])
    print(f"Brier: raw={brier_raw:.5f}  calibrated={brier_cal:.5f}")

    nap_raw, _ = protocol.normalised_average_precision(ev["y"], ev["raw_score"])
    nap_cal, _ = protocol.normalised_average_precision(ev["y"], ev["cal_score"])
    print(f"nAP:   raw={nap_raw:.4f}  calibrated={nap_cal:.4f}  (isotonic is rank-preserving up to ties, so any gap is a TIE artefact)")

    # ---- alert policy comparison: top-K vs percentile-band vs fixed threshold ----
    positives = int(ev["y"].sum())
    n = len(ev)
    budgets = [50, 100, 200, 500]

    def topk_policy(scores, k):
        k = min(k, n)
        idx_top = np.argsort(-scores.to_numpy())[:k]
        y_top = ev["y"].to_numpy()[idx_top]
        return {"k": k, "precision": float(y_top.mean()), "recall": float(y_top.sum() / positives)}

    policy_results = {"raw_score_topK": [], "calibrated_score_topK": []}
    for k in budgets:
        policy_results["raw_score_topK"].append(topk_policy(ev["raw_score"], k))
        policy_results["calibrated_score_topK"].append(topk_policy(ev["cal_score"], k))

    # Precision@K bounds under ties (protocol's own tie-aware metric) -
    # this is where isotonic collapse should show up as a WIDER bound.
    tie_bounds = {}
    for label, scores in (("raw", ev["raw_score"]), ("calibrated", ev["cal_score"])):
        tie_bounds[label] = {}
        for k in budgets:
            worst, best, n_tied = protocol.precision_at_k_bounds(ev["y"], scores, k)
            tie_bounds[label][f"K={k}"] = {"worst": worst, "best": best, "n_tied_at_cutoff": n_tied}

    print("\nprecision@K bounds (worst/best over tie orderings) - width shows tie severity:")
    for label in tie_bounds:
        for k, v in tie_bounds[label].items():
            width = v["best"] - v["worst"]
            print(f"  {label:<11} {k}: [{v['worst']:.3f}, {v['best']:.3f}]  width={width:.3f}  n_tied={v['n_tied_at_cutoff']}")

    # ---- current severity-band policy, replayed against this fold ----
    v1_manifest = json.loads((ROOT / "data/models/ps_native/v1/manifest.json").read_text())
    calib = json.loads((ROOT / "data/models/ps_native/v1/calibration.json").read_text())
    print("\nv1 frozen calibration.json bands (for reference, NOT applied to this v2-schema model):")
    print(json.dumps(calib, indent=1)[:800])

    payload = {
        "experiment_id": "exp04_calibration_alert_policy",
        "fold": fold.label,
        "n_eval": n,
        "positives": positives,
        "distinct_scores": {"raw": int(n_distinct_raw), "isotonic_calibrated": int(n_distinct_cal)},
        "brier": {"raw": float(brier_raw), "calibrated": float(brier_cal)},
        "nap": {"raw": nap_raw, "calibrated": nap_cal},
        "policy_topk": policy_results,
        "precision_at_k_tie_bounds": tie_bounds,
        "v1_frozen_calibration_reference": calib,
    }
    OUT.write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
