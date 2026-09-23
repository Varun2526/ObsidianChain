"""EXPERIMENT / ARTIFACT exp10_v2_dev_candidate_artifact.

NOT a production promotion. This trains and saves a DEVELOPMENT CANDIDATE
artifact reflecting everything this program has confirmed so far, clearly
labeled as provisional:

  - Model: LightGBM, exp01's hyperparameters (the tied tier's fastest
    member and the only one with an exact, dependency-free TreeSHAP path -
    exp05/exp10 in the research deliverables).
  - Features: the 24 healthy v2 CORE columns only. Does NOT include the
    counterparty-history feature (exp08) - that result is INCONCLUSIVE
    (p=0.199, sub-MDE), and RULE 6 forbids silently changing the feature
    set on an unconfirmed result.
  - Explanations: bundled with a verified TreeSHAP-based explain function
    (scripts/explain_treeshap.py) as a drop-in prototype for what
    ps_model.py's explanation path should become - NOT wired into the
    actual production ps_model.py file by this program (that integration
    decision belongs to whoever promotes this candidate).
  - Alert policy: deliberately NOT included. exp09 showed static severity
    bands are unreliable across time even when correctly derived; this
    program's evidenced recommendation is top-K, which is an operational
    choice (what K) rather than a model-artifact property, so no bands are
    baked into this artifact.
  - Trained on ALL development data (train+validation combined) - the
    correct final-candidate practice (a model that will be evaluated once
    against the holdout should be fit on everything the holdout excludes),
    but the SEALED HOLDOUT IS NOT READ OR EVALUATED by this script. No
    holdout metric appears anywhere in this artifact's manifest.

Saved under research/autoresearch_2026_09_23/v2_dev_candidate/ - explicitly
NOT under data/models/ps_native/v2/, so it cannot be mistaken for a promoted
production artifact by any path-based check in the codebase
(api/provenance_gate.py and ps_model.py both look under data/models/).
"""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from obsidianchain.ml import diagnostics, protocol  # noqa: E402
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS  # noqa: E402

DS = ROOT / "data/models/ps_native/datasets"
OUT_DIR = ROOT / "research/autoresearch_2026_09_23/v2_dev_candidate"
BASE, STEP = 1400000000, 1209600
SEED = 20260919


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(DS / "train.parquet")
    val = pd.read_parquet(DS / "validation.parquet")
    dev = pd.concat([train, val], ignore_index=True)
    dev["first_t"] = ((dev["timestamp"] - BASE) // STEP + 1).astype(int)

    declared = list(CORE_PS_FEATURE_COLUMNS)
    healthy = diagnostics.healthy_features(dev, declared)
    print(f"training on {len(dev):,} rows (train+validation combined), {len(healthy)} healthy CORE features")

    from lightgbm import LGBMClassifier
    X = np.nan_to_num(dev[healthy].to_numpy("float32"))
    y = dev["y"].to_numpy("int8")
    model = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1)
    model.fit(X, y)

    model_path = OUT_DIR / "model.joblib"
    joblib.dump({"model": model, "features": healthy}, model_path)
    model_sha256 = _sha256(model_path)

    # global TreeSHAP importance, for the manifest and for
    # explain_treeshap.py's consumers to sanity-check against.
    booster = model.booster_
    sample = dev.sample(n=min(5000, len(dev)), random_state=SEED)
    Xs = np.nan_to_num(sample[healthy].to_numpy("float32"))
    shap_vals = booster.predict(Xs, pred_contrib=True)[:, :-1]
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance_ranked = sorted(
        zip(healthy, mean_abs_shap.tolist()), key=lambda kv: -kv[1]
    )

    train_hash = _sha256(DS / "train.parquet")
    val_hash = _sha256(DS / "validation.parquet")

    manifest = {
        "schema": "obsidianchain.ps_model_manifest/2_dev_candidate",
        "status": "DEVELOPMENT_CANDIDATE_NOT_PROMOTED",
        "provenance_type": "RESEARCH_DEVELOPMENT_CANDIDATE",
        "warning": (
            "This is NOT a production artifact. It is saved outside "
            "data/models/ps_native/ specifically so no provenance_gate or "
            "ps_model.py path check can mistake it for one. Promotion "
            "requires a human decision plus the Phase 16 checklist in "
            "14_final_holdout.md."
        ),
        "model_version": "ps_native_v2_dev_candidate",
        "model_type": "LightGBM",
        "feature_schema_version": "ps_native_features/2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "research/autoresearch_2026_09_23/scripts/exp10_v2_dev_candidate_artifact.py",
        "training_data": {
            "train.parquet_sha256": train_hash,
            "validation.parquet_sha256": val_hash,
            "n_rows": int(len(dev)), "n_positive": int(y.sum()),
        },
        "model_sha256": model_sha256,
        "features": healthy,
        "hyperparameters": {
            "n_estimators": 100, "learning_rate": 0.05, "random_state": SEED,
        },
        "development_evaluation_reference": (
            "See research/autoresearch_2026_09_23/results/exp01_model_family.json "
            "and exp06_model_search_round2.json for the 12-fold rolling-origin "
            "evaluation this model's family and hyperparameters were selected "
            "under. This manifest intentionally contains NO holdout metric — "
            "the sealed holdout has not been read."
        ),
        "global_treeshap_importance_top10": importance_ranked[:10],
        "explanation_method": (
            "Use scripts/explain_treeshap.py (TreeSHAP via "
            "Booster.predict(pred_contrib=True)), NOT the feature_importances_ "
            "x |value| proxy in the current production ps_model.py — that "
            "method is REJECTED, see 10_explanation_audit.md "
            "(33.9% direction agreement with true TreeSHAP, worse than chance)."
        ),
        "alert_policy": (
            "NOT included. 09_alert_policy_analysis.md and its exp09 addendum "
            "found static severity-band thresholds unreliable across time "
            "even when correctly derived. Recommended: top-K ranking at an "
            "operationally-chosen budget, re-evaluated periodically — not "
            "baked into this artifact."
        ),
        "known_open_items": [
            "Counterparty-history feature (exp08) is INCONCLUSIVE (+0.024 nAP, "
            "p=0.199, sub-MDE) and is NOT included in this feature set. "
            "Revisit if a richer counterparty feature or more folds resolve it.",
            "Model choice within the tied tier {RandomForest, HistGradientBoosting, "
            "LightGBM} is not statistically forced — LightGBM chosen here for its "
            "exact, dependency-free TreeSHAP path, not for superior ranking quality.",
            "make test has not been re-run as part of this program.",
        ],
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {model_path} (sha256={model_sha256[:16]}...)")
    print(f"wrote {manifest_path}")
    print("\ntop-10 features by mean |TreeSHAP|:")
    for name, val_ in importance_ranked[:10]:
        print(f"  {name:<30} {val_:.4f}")


if __name__ == "__main__":
    main()
