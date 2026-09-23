"""EXPERIMENT exp05_explanation_audit.

Phase 9. Compares ps_model.py's CURRENT explanation method
(global feature_importances_ * |raw value|, direction from whether the
calibrated probability >= 0.5) against real per-prediction TreeSHAP
attributions, on the SAME fitted model and SAME rows - isolating the
explanation METHOD as the only variable.

The external `shap` package is not installed / not vendored (see
01_repo_audit.md). LightGBM ships its own exact TreeSHAP implementation via
Booster.predict(pred_contrib=True) (this is the same algorithm the `shap`
package would call for a LightGBM model - not an approximation), which is
what HANDOFF.md refers to as "correct" for the Phase 6 path. This lets Phase
9 run without the missing dependency.
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
OUT = ROOT / "research/autoresearch_2026_09_23/results/exp05_explanation_audit.json"
BASE, STEP = 1400000000, 1209600
SEED = 20260919


def load_dev() -> pd.DataFrame:
    dev = pd.concat(
        [pd.read_parquet(DS / "train.parquet"), pd.read_parquet(DS / "validation.parquet")],
        ignore_index=True,
    )
    dev["first_t"] = ((dev["timestamp"] - BASE) // STEP + 1).astype(int)
    return dev


def current_method_top3(model, features, row_vals):
    """Replicates ps_model.py's PsNativeRiskModel.predict_address_features exactly."""
    importances = model.feature_importances_.astype("float64")
    importances = importances / (importances.sum() or 1.0)
    contribs = importances * np.abs(row_vals)
    top3 = np.argsort(-contribs)[:3]
    return [(features[i], float(contribs[i])) for i in top3]


def treeshap_top3(shap_row, features):
    # shap_row excludes the bias/expected-value column (last column from
    # pred_contrib output); rank by |contribution|, matching how an analyst
    # would read "what mattered most", positive or negative.
    order = np.argsort(-np.abs(shap_row))[:3]
    return [(features[i], float(shap_row[i])) for i in order]


def main() -> None:
    dev = load_dev()
    declared = list(CORE_PS_FEATURE_COLUMNS)
    healthy = diagnostics.healthy_features(dev, declared)
    folds = protocol.rolling_origin_folds()
    fold = folds[-1]

    usable = protocol.development(dev)
    train = usable[usable["first_t"] <= fold.train_end]
    ev = usable[(usable["first_t"] >= fold.eval_start) & (usable["first_t"] <= fold.eval_end)].copy()

    from lightgbm import LGBMClassifier
    X = np.nan_to_num(train[healthy].to_numpy("float32"))
    y = train["y"].to_numpy("int8")
    m = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(X, y)

    Xe = np.nan_to_num(ev[healthy].to_numpy("float32"))
    scores = m.predict_proba(Xe)[:, 1]

    # exact TreeSHAP via the booster's own pred_contrib
    booster = m.booster_
    shap_values = booster.predict(Xe, pred_contrib=True)  # (n, n_features+1); last col = expected value
    shap_feature_part = shap_values[:, :-1]

    # sanity check: SHAP values must sum (+ bias) to the raw margin/logit,
    # not just "look plausible" - this IS local fidelity, checked directly.
    raw_margin = booster.predict(Xe, raw_score=True)
    reconstructed = shap_feature_part.sum(axis=1) + shap_values[:, -1]
    fidelity_max_abs_err = float(np.max(np.abs(reconstructed - raw_margin)))
    print(f"TreeSHAP local fidelity check: max|sum(shap)+bias - raw_margin| = {fidelity_max_abs_err:.2e} (should be ~1e-6, i.e. exact)")

    # top-3 feature overlap between the two methods, per row
    n_check = min(500, len(ev))
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(ev), size=n_check, replace=False)

    overlap_counts = []
    direction_agreements = []
    sign_flip_cases = 0
    for i in sample_idx:
        row_vals = Xe[i]
        cur = current_method_top3(m, healthy, row_vals)
        shp = treeshap_top3(shap_feature_part[i], healthy)
        cur_names = {f for f, _ in cur}
        shp_names = {f for f, _ in shp}
        overlap_counts.append(len(cur_names & shp_names))

        # direction check: current method assigns direction from whether
        # cal_p>=0.5 GLOBALLY (same for every one of that row's top-3
        # features), never from the feature's own SHAP sign - check whether
        # that global-direction assumption agrees with each of THIS row's
        # own top SHAP-signed features.
        row_direction = "INCREASES_RISK" if scores[i] >= 0.5 else "DECREASES_RISK"
        for f, shap_val in shp:
            true_direction = "INCREASES_RISK" if shap_val > 0 else "DECREASES_RISK"
            direction_agreements.append(row_direction == true_direction)
            if row_direction != true_direction:
                sign_flip_cases += 1

    overlap_counts = np.array(overlap_counts)
    direction_agreements = np.array(direction_agreements)

    print(f"\ntop-3 feature overlap (current method vs TreeSHAP), n={n_check} rows sampled:")
    print(f"  mean features in common (of 3): {overlap_counts.mean():.2f}")
    print(f"  rows with 0 overlap: {(overlap_counts==0).sum()} ({100*(overlap_counts==0).mean():.1f}%)")
    print(f"  rows with full (3/3) overlap: {(overlap_counts==3).sum()} ({100*(overlap_counts==3).mean():.1f}%)")

    print(f"\ncurrent method's direction label vs TreeSHAP's per-feature sign, "
          f"n={len(direction_agreements)} (feature, row) pairs from {n_check} rows:")
    print(f"  agreement rate: {direction_agreements.mean():.3f}")
    print(f"  disagreements (current method's global direction contradicts this feature's actual SHAP sign): "
          f"{(~direction_agreements).sum()} of {len(direction_agreements)}")

    payload = {
        "experiment_id": "exp05_explanation_audit",
        "fold": fold.label,
        "treeshap_local_fidelity_max_abs_err": fidelity_max_abs_err,
        "n_rows_sampled": int(n_check),
        "top3_overlap": {
            "mean_of_3": float(overlap_counts.mean()),
            "pct_zero_overlap": float((overlap_counts == 0).mean()),
            "pct_full_overlap": float((overlap_counts == 3).mean()),
        },
        "direction_agreement": {
            "n_pairs": int(len(direction_agreements)),
            "agreement_rate": float(direction_agreements.mean()),
            "n_disagreements": int((~direction_agreements).sum()),
        },
    }
    OUT.write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
