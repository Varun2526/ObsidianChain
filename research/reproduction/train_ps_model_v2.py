"""Train and freeze the PS-native v2 risk model (feature schema /3).

What changed from v1, and why
-----------------------------
v1 (``train_ps_model.py``) picked RandomForest over LightGBM on ONE validation
window. Under the 12-fold protocol that difference reverses sign and is far
inside the design's minimum detectable effect (``ml/protocol.py``). v2
therefore does not select a family on a point estimate. LightGBM is used
because the tied tier {RandomForest, HistGradientBoosting, LightGBM} does not
separate on ranking (exp01) and LightGBM alone gives exact per-row TreeSHAP
(``pred_contrib``) and native missing-value handling without a new dependency.

The script, in order:

1. Loads the development frame (t1-41) through ``protocol.development``.
   The sealed holdout is never opened.
2. Refuses to continue unless the feature set passes
   ``diagnostics.assert_healthy`` (invariant 12).
3. Evaluates the model over the 12 rolling-origin folds and records per-fold
   nAP, transaction-weighted nAP and distinct-transaction precision@100.
   Also runs one paired ablation: the v3 role group (F) against the same
   model without it.
4. Fits a Platt calibrator on OUT-OF-FOLD scores from the four most recent
   folds. Isotonic is not used: across all 12 folds it did not beat the raw
   score on Brier and collapsed scores to 25-64 levels
   (``17_dataset_metrics_audit.md``).
5. Refits on the whole development period and writes
   ``data/models/ps_native/v2/``.

The artifact states plainly that the holdout has NOT been evaluated. The
holdout is still at schema /1 and must be rebuilt under a written reason
before a final measurement is possible.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from obsidianchain.ml import diagnostics, protocol
from obsidianchain.ml.monitoring import build_reference_profile
from obsidianchain.pipeline.features_ps import (
    CORE_PS_FEATURE_COLUMNS,
    GROUP_F_ROLE,
    PS_FEATURE_GROUPS,
    PS_FEATURE_SCHEMA_VERSION,
)

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "data" / "models" / "ps_native" / "datasets"
#: This script produces the CURRENT PS-native model. Its name is kept from
#: the v2 introduction. Schema /5 (simultaneous events, whole-row
#: snapshots) writes ps_native_v4; v3 (/4) stays frozen and immutable.
MODEL_VERSION = "ps_native_v4"
OUT = ROOT / "data" / "models" / "ps_native" / "v4"

BASE_TIMESTAMP = 1400000000
TIMESTEP_SECONDS = 1209600
SEED = 20260919

#: Recorded in the manifest so the fold evaluation and the frozen model
#: provably share one configuration.
HYPERPARAMETERS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
)

#: Training window in timesteps: every fit uses only the most recent
#: TRAIN_WINDOW steps up to its cutoff. exp18 (pre-registered) compared this
#: against expanding-window training, recency weighting, per-step ranks and
#: transaction weighting: the 16-step window lowered fold-nAP sd 0.173 ->
#: 0.160, raised the worst fold 0.243 -> 0.409 (t39-40, the fold nearest
#: deployment; address P@100 there 0.36 -> 0.87) and the mean 0.620 -> 0.648.
#: The mean difference is not significant (Holm p ~0.52). Older history hurts
#: more than it helps because the fee market drifts ~50x over the period.
#: Re-selected on causally ordered data with group G (exp21): the same
#: configuration was kept, since neither the grid winner nor the
#: three-model ensemble beat it on the held-back confirmation folds.
TRAIN_WINDOW = 16

#: Folds whose out-of-fold scores train the calibrator: the most recent
#: four, because prevalence drifts and the calibrator should describe the
#: period closest to deployment.
CALIBRATION_FOLDS = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(seed: int = SEED) -> LGBMClassifier:
    return LGBMClassifier(**HYPERPARAMETERS, random_state=seed, verbose=-1, n_jobs=-1)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_development() -> pd.DataFrame:
    frames = [pd.read_parquet(DATASETS / f"{s}.parquet") for s in ("train", "validation")]
    dev = pd.concat(frames, ignore_index=True)
    # The timestep of the row's own observation, i.e. the address's LAST
    # snapshot. Earlier research scripts called this ``first_t``; it is not.
    dev["last_t"] = ((dev["timestamp"] - BASE_TIMESTAMP) // TIMESTEP_SECONDS + 1).astype(int)
    return protocol.development(dev, timestep="last_t")


def transaction_weighted_nap(y, scores, txids) -> float:
    """nAP where every transaction carries total weight one.

    83% of development rows duplicate another row's feature vector because
    group-A features belong to the transaction; one transaction holds 434
    positives. Row nAP lets a single transaction move a fold; this does not.
    """
    y = np.asarray(y)
    counts = pd.Series(txids).map(pd.Series(txids).value_counts()).to_numpy()
    w = 1.0 / counts
    prevalence = float(np.average(y, weights=w))
    if prevalence in (0.0, 1.0):
        return float("nan")
    ap = float(average_precision_score(y, scores, sample_weight=w))
    return (ap - prevalence) / (1 - prevalence)


def distinct_tx_precision_at_k(y, scores, txids, k: int = 100) -> tuple[float, int]:
    """Precision over the top ``k`` DISTINCT transactions.

    A transaction is positive if any of its scored addresses is. Returns the
    precision and how many distinct transactions were available.
    """
    frame = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(scores), "tx": np.asarray(txids)})
    per_tx = frame.groupby("tx").agg(s=("s", "max"), y=("y", "max")).sort_values("s", ascending=False)
    top = per_tx.head(k)
    return float(top["y"].mean()), int(len(per_tx))


def window(frame: pd.DataFrame, end: int) -> pd.DataFrame:
    """Training rows: the TRAIN_WINDOW timesteps ending at ``end``."""
    return frame[(frame.last_t <= end) & (frame.last_t > end - TRAIN_WINDOW)]


def evaluate(dev: pd.DataFrame, features: list[str]) -> tuple[list[dict], pd.DataFrame]:
    rows, oof = [], []
    for fold in protocol.rolling_origin_folds():
        train = window(dev, fold.train_end)
        ev = dev[(dev.last_t >= fold.eval_start) & (dev.last_t <= fold.eval_end)]
        model = _model().fit(train[features], train.y)
        s = model.predict_proba(ev[features])[:, 1]
        nap, prevalence = protocol.normalised_average_precision(ev.y, s)
        p100, n_tx = distinct_tx_precision_at_k(ev.y, s, ev.txid)
        rows.append({
            "fold": fold.label,
            "n_eval": int(len(ev)),
            "prevalence": round(prevalence, 4),
            "nap": round(nap, 4),
            "nap_tx_weighted": round(transaction_weighted_nap(ev.y, s, ev.txid), 4),
            "precision_at_100_distinct_tx": round(p100, 4),
            "distinct_tx": n_tx,
        })
        oof.append(pd.DataFrame({"fold": fold.index, "y": ev.y.to_numpy(), "raw": s}))
    return rows, pd.concat(oof, ignore_index=True)


def lineage(ds_manifest: dict, evaluation_sha: str) -> dict:
    """Everything needed to say exactly what produced this artifact."""
    import platform

    import lightgbm
    import sklearn

    from obsidianchain.provenance import git_revision
    code = {rel: _sha256(ROOT / rel) for rel in (
        "src/obsidianchain/pipeline/features_ps.py",
        "src/obsidianchain/ml/ps_model.py",
        "research/reproduction/build_ps_dataset.py",
        "research/reproduction/train_ps_model_v2.py",
    )}
    config = {"hyperparameters": HYPERPARAMETERS, "seed": SEED, "train_window": TRAIN_WINDOW,
              "calibration_folds": CALIBRATION_FOLDS, "features": list(CORE_PS_FEATURE_COLUMNS),
              "feature_schema_version": PS_FEATURE_SCHEMA_VERSION}
    return {
        "source_commit": git_revision(ROOT),
        "source_commit_note": "commit the artifact was trained from; uncommitted edits are "
                              "visible as a code_sha256 mismatch against that commit",
        "code_sha256": code,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "config": config,
        "training_data_sha256": {k: v for k, v in ds_manifest["artifact_hashes"].items() if k != "test.parquet"},
        "raw_source_sha256": ds_manifest.get("source_dataset_hashes"),
        "evaluation_sha256": evaluation_sha,
        "environment": {"python": platform.python_version(), "lightgbm": lightgbm.__version__,
                        "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform()},
    }


def main() -> dict:
    dev = load_development()
    features = list(CORE_PS_FEATURE_COLUMNS)
    diagnostics.assert_healthy(dev, features)
    print(f"development rows {len(dev):,}, positives {int(dev.y.sum()):,}, "
          f"timesteps {dev.last_t.min()}-{dev.last_t.max()}")

    print("12-fold evaluation, full v3 feature set ...")
    folds, oof = evaluate(dev, features)
    without_f = [c for c in features if c not in GROUP_F_ROLE]
    print("12-fold evaluation, without group F (role) ...")
    folds_nof, _ = evaluate(dev, without_f)

    nap = np.array([f["nap"] for f in folds])
    nap_nof = np.array([f["nap"] for f in folds_nof])
    txw = np.array([f["nap_tx_weighted"] for f in folds])
    txw_nof = np.array([f["nap_tx_weighted"] for f in folds_nof])
    ablation = {
        "row_nap": protocol.paired_verdict(nap - nap_nof, a="with_F", b="without_F"),
        "tx_weighted_nap": protocol.paired_verdict(txw - txw_nof, a="with_F", b="without_F"),
    }
    for f in folds:
        print(f"  {f['fold']:>8}  nAP {f['nap']:.3f}  txw {f['nap_tx_weighted']:.3f}  "
              f"P@100tx {f['precision_at_100_distinct_tx']:.2f}  prev {f['prevalence']:.3f}")
    print(f"mean nAP {nap.mean():.3f} (sd {nap.std(ddof=1):.3f}); "
          f"tx-weighted {txw.mean():.3f}; without F {nap_nof.mean():.3f}")
    print("group F ablation:", ablation["row_nap"]["verdict"], ablation["tx_weighted_nap"]["verdict"])

    # Calibrator: Platt on out-of-fold scores of the most recent folds.
    recent = oof[oof.fold >= oof.fold.max() - CALIBRATION_FOLDS + 1]
    platt = LogisticRegression().fit(_logit(recent.raw.to_numpy()).reshape(-1, 1), recent.y)
    calibrated = platt.predict_proba(_logit(recent.raw.to_numpy()).reshape(-1, 1))[:, 1]
    calibration = {
        "method": "Platt (logistic on logit of the raw score)",
        "fitted_on": f"out-of-fold scores, last {CALIBRATION_FOLDS} rolling folds",
        "n_rows": int(len(recent)),
        "reference_prevalence": float(recent.y.mean()),
        "coef": float(platt.coef_[0][0]),
        "intercept": float(platt.intercept_[0]),
        "brier_raw_oof": float(brier_score_loss(recent.y, recent.raw)),
        "brier_calibrated_in_sample": float(brier_score_loss(recent.y, calibrated)),
        "note": ("The raw score ranks; the calibrated probability is for display. "
                 "Calibration is monotone, so it never changes the order."),
    }

    print("refitting on the full development period ...")
    recent = window(dev, int(dev.last_t.max()))
    final = _model().fit(recent[features], recent.y)
    # Score reference from OUT-OF-FOLD scores. In-sample scores of a model
    # with train AUC ~1.0 are far more extreme than anything it produces on
    # new data, so every real run would read as a major shift.
    reference = build_reference_profile(recent, features, oof.raw.to_numpy())

    from obsidianchain.ml import registry as model_registry
    reg = model_registry.Registry.open(OUT.parent)
    if MODEL_VERSION in reg.data["models"]:
        raise SystemExit(f"{MODEL_VERSION} is registered and immutable; bump MODEL_VERSION to retrain")
    OUT.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": "LightGBM",
        "model": final,
        "calibrator": {"coef": calibration["coef"], "intercept": calibration["intercept"],
                       "reference_prevalence": calibration["reference_prevalence"]},
        "features": features,
        "feature_schema_version": PS_FEATURE_SCHEMA_VERSION,
        "reference_profile": reference,
    }
    joblib.dump(artifact, OUT / "model.joblib")
    model_sha = _sha256(OUT / "model.joblib")

    evaluation = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "mde_80_power": protocol.MDE_80_POWER,
        "folds": folds,
        "summary": {
            "nap_mean": round(float(nap.mean()), 4),
            "nap_sd": round(float(nap.std(ddof=1)), 4),
            "nap_tx_weighted_mean": round(float(txw.mean()), 4),
            "precision_at_100_distinct_tx_mean": round(
                float(np.mean([f["precision_at_100_distinct_tx"] for f in folds])), 4),
        },
        "group_F_ablation": ablation,
        "folds_without_group_F": folds_nof,
    }
    (OUT / "evaluation.json").write_text(json.dumps(evaluation, indent=2, default=str))
    (OUT / "calibration.json").write_text(json.dumps(calibration, indent=2))
    (OUT / "feature_schema.json").write_text(json.dumps({
        "schema_version": PS_FEATURE_SCHEMA_VERSION,
        "feature_count": len(features),
        "feature_list": features,
        "feature_groups": {k: v for k, v in PS_FEATURE_GROUPS.items() if k != "E_network"},
    }, indent=2))

    ds_manifest = json.loads((DATASETS / "manifest.json").read_text())
    manifest = {
        "schema": "obsidianchain.ps_model_manifest/2",
        "model_version": MODEL_VERSION,
        "model_type": "LightGBM",
        "status": "PRODUCTION_HOLDOUT_PENDING",
        "holdout_evaluated": False,
        "holdout_note": (
            "The sealed holdout (t42-49) is still schema ps_native_features/1 and "
            "was not opened. Every number in evaluation.json is a development-period "
            "rolling-fold estimate, not a final measurement."
        ),
        "feature_schema_version": PS_FEATURE_SCHEMA_VERSION,
        "features": features,
        "hyperparameters": HYPERPARAMETERS,
        "train_window_timesteps": TRAIN_WINDOW,
        "final_fit_timesteps": f"{int(recent.last_t.min())}-{int(recent.last_t.max())}",
        "random_state": SEED,
        "training_data": {
            "train.parquet": ds_manifest["artifact_hashes"]["train.parquet"],
            "validation.parquet": ds_manifest["artifact_hashes"]["validation.parquet"],
            "n_rows": int(len(recent)),
            "n_positive": int(recent.y.sum()),
        },
        "explanation_method": "TreeSHAP via LightGBM Booster.predict(pred_contrib=True), per-feature sign",
        "ranking_score": "raw model probability",
        "display_probability": "Platt-calibrated raw score, see calibration.json",
        "severity_policy": "rank percentile within a run, see ml/ps_model.py SEVERITY_BUDGET",
        "model_sha256": model_sha,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generated_by": "research/reproduction/train_ps_model_v2.py",
    }
    for name in ("evaluation.json", "calibration.json", "feature_schema.json"):
        manifest.setdefault("artifacts", {})[name] = _sha256(OUT / name)
    manifest["artifacts"]["model.joblib"] = model_sha
    manifest["lineage"] = lineage(ds_manifest, evaluation_sha=manifest["artifacts"]["evaluation.json"])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT}  model sha256 {model_sha[:12]}")
    return manifest


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

