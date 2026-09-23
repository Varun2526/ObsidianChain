"""Train, evaluate and freeze the PS-native production model under protocol B.

Supersedes ``train_ps_model_v2.py`` (kept for the audit trail; it evaluates
on the final-event snapshot, which exp22 showed to be a hindsight unit).

Protocol B - the evaluation unit this model is validated and trained under
(exp22, research/autoresearch_2026_09_23/results/exp22_window_end_protocol.json):
  fold (train_end te, window es..ee), the 12 ml/protocol.py folds
    train  addresses first seen in (te - W, te], snapshot = last event <= te
    eval   addresses first seen in [es, ee],     snapshot = last event <= ee
  i.e. "rank addresses NEW in this window, as of the window's end", with no
  address in both sets. The sealed holdout period (t >= 42) is never read.

Steps
  1. Event-level features from the production engine (exp22 cache).
  2. Feature health (invariant 12) and the feature contract on every set.
  3. 12-fold evaluation with the full ml/evaluation.py scorecard, split into
     tuning folds 1-6 and confirmation folds 7-12, with slices.
  4. Calibration evaluated honestly per fold: Platt fitted on the raw
     out-of-fold scores of the four PREVIOUS folds only (folds 5-12).
  5. Frozen Platt: fitted on the out-of-fold scores of folds 9-12.
  6. Final model: protocol-B training set at te = 41.
  7. Performance: load time, scoring latency and throughput, artifact size.
  8. Manifest with full lineage. No registry role is assigned here.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "reproduction"))
sys.path.insert(0, str(ROOT / "research" / "autoresearch_2026_09_23" / "scripts"))

from obsidianchain.contracts.features import FEATURE_CONTRACT_VERSION, validate_feature_frame  # noqa: E402
from obsidianchain.ml import diagnostics, evaluation, protocol  # noqa: E402
from obsidianchain.ml import registry as model_registry  # noqa: E402
from obsidianchain.ml.monitoring import build_reference_profile  # noqa: E402
from obsidianchain.pipeline.features_ps import (  # noqa: E402
    CORE_PS_FEATURE_COLUMNS, PS_FEATURE_GROUPS, PS_FEATURE_SCHEMA_VERSION,
)
from exp22_window_end_protocol import event_features, fold_sets, snapshot  # noqa: E402

import os  # noqa: E402

from obsidianchain.pipeline.features_ps import GROUP_G_UPSTREAM  # noqa: E402

#: FEATURE_SET=no_g builds the registered FALLBACK: the same configuration
#: without group G (exp22: nAP 0.761 vs 0.808). A genuinely different, simpler
#: model on the same schema, so a defect in the upstream-flow computation can
#: be answered by a rollback that needs no code change.
FEATURE_SET = os.environ.get("FEATURE_SET", "core")
#: v5, not v4: ps_native_v4 was trained before this script's last edit, so no
#: commit holds its exact training code and its lineage cannot be attested.
#: v5 is the same configuration trained from committed code.
if FEATURE_SET == "no_g":
    MODEL_VERSION = "ps_native_v5_fallback_no_g"
    OUT = ROOT / "data" / "models" / "ps_native" / "v5_fallback_no_g"
else:
    MODEL_VERSION = "ps_native_v5"
    OUT = ROOT / "data" / "models" / "ps_native" / "v5"
PROTOCOL_ID = "protocol_B_new_addresses_at_window_end/1"
SEED = 20260919
TRAIN_WINDOW = 16
HYPERPARAMETERS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
FEATURES = list(CORE_PS_FEATURE_COLUMNS)
if FEATURE_SET == "no_g":
    FEATURES = [c for c in FEATURES if c not in GROUP_G_UPSTREAM]
CAL_FOLDS = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model() -> LGBMClassifier:
    return LGBMClassifier(**HYPERPARAMETERS, random_state=SEED, verbose=-1, n_jobs=-1)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _platt(raw, y) -> LogisticRegression:
    return LogisticRegression().fit(_logit(raw).reshape(-1, 1), y)


def _check(frame: pd.DataFrame, where: str) -> None:
    report = validate_feature_frame(frame, FEATURES)
    if not report.ok:
        raise SystemExit(f"feature contract violated in {where}: {report.violations[:3]}")


def _slim(card: dict) -> dict:
    """Drop the per-bin reliability tables from the per-fold records."""
    out = dict(card)
    if "calibration" in out:
        out["calibration"] = {k: v for k, v in out["calibration"].items() if k != "reliability"}
    return out


def lineage(evaluation_sha: str) -> dict:
    import platform

    import lightgbm
    import sklearn

    from obsidianchain.provenance import git_revision
    code = {rel: _sha256(ROOT / rel) for rel in (
        "src/obsidianchain/pipeline/features_ps.py", "src/obsidianchain/ml/ps_model.py",
        "src/obsidianchain/contracts/features.py", "research/reproduction/build_ps_dataset.py",
        "research/reproduction/train_ps_production_model.py",
        "research/autoresearch_2026_09_23/scripts/exp22_window_end_protocol.py")}
    config = {"hyperparameters": HYPERPARAMETERS, "seed": SEED, "train_window": TRAIN_WINDOW,
              "features": FEATURES, "feature_schema_version": PS_FEATURE_SCHEMA_VERSION,
              "protocol": PROTOCOL_ID, "calibration": f"Platt on OOF of folds 9-12"}
    raw = ROOT / "data" / "raw"
    return {
        "source_commit": git_revision(ROOT),
        "code_sha256": code,
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "raw_source_sha256": {f: _sha256(raw / f) for f in (
            "AddrTx_edgelist.csv", "TxAddr_edgelist.csv", "txs_features.csv", "txs_edgelist.csv",
            "wallets_classes.csv")},
        "evaluation_sha256": evaluation_sha,
        "environment": {"python": platform.python_version(), "lightgbm": lightgbm.__version__,
                        "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform()},
    }


def main() -> dict:
    if MODEL_VERSION in model_registry.Registry.open(OUT.parent).data["models"]:
        raise SystemExit(f"{MODEL_VERSION} is registered and immutable; bump MODEL_VERSION to retrain")
    t0 = time.time()
    ev = event_features()
    folds = protocol.rolling_origin_folds()

    fold_records, oof = [], []
    for i, fold in enumerate(folds):
        train, test = fold_sets(ev, fold, TRAIN_WINDOW)
        if i == 0:
            diagnostics.assert_healthy(train, FEATURES)
        _check(train, f"train {fold.label}")
        _check(test, f"eval {fold.label}")
        raw = _model().fit(train[FEATURES], train.y).predict_proba(test[FEATURES])[:, 1]
        y = test.y.to_numpy()
        prob = None
        if i >= CAL_FOLDS:
            prior = pd.concat(oof[i - CAL_FOLDS:i])
            prob = _platt(prior.raw, prior.y).predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        card = evaluation.full_scorecard(y, raw, test.txid.to_numpy(), prob=prob, frame=test)
        fold_records.append({"fold": fold.label, "part": "tune" if i < 6 else "confirm", **_slim(card),
                             "reliability": card.get("calibration", {}).get("reliability")})
        oof.append(pd.DataFrame({"raw": raw, "y": y}))
        a = card["address"]
        print(f"{fold.label:>16} n {a['n']:>6} prev {a['prevalence']:.3f} nAP {a['nap']:.3f} "
              f"P@100 {a['P@100']:.2f} Rprec {a['r_precision']:.3f} nR@500 {a['nR@500']:.3f}"
              + (f" ECE {card['calibration']['ece']:.3f}" if prob is not None else ""), flush=True)

    def agg(part: str | None, level: str, key: str):
        vals = [r[level][key] for r in fold_records if part is None or r["part"] == part]
        return evaluation.aggregate(vals)

    keys = ["nap", "ap", "roc_auc", "r_precision", "P@10", "P@25", "P@50", "P@100", "P@250", "P@500",
            "P@1000", "R@100", "R@500", "R@1000", "nR@100", "nR@500", "NDCG@100", "lift@100", "FP@100", "FP@500"]
    summary = {part or "all": {level: {k: agg(part, level, k) for k in keys} for level in ("address", "transaction")}
               for part in ("tune", "confirm", None)}
    for part in ("tune", "confirm", None):
        rows = [r for r in fold_records if part is None or r["part"] == part]
        summary[part or "all"]["tx_weighted_nap"] = evaluation.aggregate([r["tx_weighted_nap"] for r in rows])
        cal = [r["calibration"] for r in rows if "calibration" in r]
        summary[part or "all"]["calibration"] = {k: evaluation.aggregate([c[k] for c in cal if k in c])
                                                 for k in ("ece", "brier", "calibration_slope", "calibration_intercept")}
    summary["temporal_trend_nap"] = evaluation.temporal_trend([r["address"]["nap"] for r in fold_records])
    slice_names = list(fold_records[0]["slices"])
    summary["slices_nap"] = {s: evaluation.aggregate([r["slices"][s].get("nap") for r in fold_records
                                                      if r["slices"][s].get("measurable")]) for s in slice_names}

    # Frozen calibrator and final fit.
    recent = pd.concat(oof[-CAL_FOLDS:])
    platt = _platt(recent.raw, recent.y)
    final_train = snapshot(ev, protocol.DEVELOPMENT_END - TRAIN_WINDOW + 1, protocol.DEVELOPMENT_END,
                           protocol.DEVELOPMENT_END)
    _check(final_train, "final training set")
    final = _model().fit(final_train[FEATURES], final_train.y)
    reference = build_reference_profile(final_train, FEATURES, pd.concat(oof).raw.to_numpy())

    OUT.mkdir(parents=True, exist_ok=True)
    calibrator = {"coef": float(platt.coef_[0][0]), "intercept": float(platt.intercept_[0]),
                  "reference_prevalence": float(recent.y.mean())}
    joblib.dump({"model_name": "LightGBM", "model": final, "calibrator": calibrator, "features": FEATURES,
                 "feature_schema_version": PS_FEATURE_SCHEMA_VERSION, "reference_profile": reference},
                OUT / "model.joblib")

    # Performance: a cold load and scoring latency on a realistic batch.
    from obsidianchain.ml.ps_model import PsNativeRiskModel
    manifest_stub = {"model_version": MODEL_VERSION, "model_sha256": _sha256(OUT / "model.joblib"),
                     "feature_schema_version": PS_FEATURE_SCHEMA_VERSION, "features": FEATURES}
    (OUT / "manifest.json").write_text(json.dumps(manifest_stub))
    t_load = time.perf_counter()
    served = PsNativeRiskModel.load(OUT, require_live_schema=True)
    load_seconds = time.perf_counter() - t_load
    batch = final_train.sample(min(10000, len(final_train)), random_state=0)
    lat = []
    for n in (1, 100, 1000, 10000):
        part = batch.head(n)
        t = time.perf_counter()
        for _ in range(3):
            served.raw_scores(part)
        lat.append({"rows": n, "seconds_per_call": (time.perf_counter() - t) / 3})
    single = []
    for i in range(200):
        t = time.perf_counter()
        served.raw_scores(batch.iloc[[i]])
        single.append(time.perf_counter() - t)
    performance = {"load_seconds": round(load_seconds, 4),
                   "single_row_latency_ms": {q: round(float(np.quantile(single, p)) * 1000, 3)
                                             for q, p in (("p50", .5), ("p95", .95), ("p99", .99))},
                   "batch_latency": lat,
                   "rows_per_second_at_10k": round(10000 / lat[-1]["seconds_per_call"], 0),
                   "artifact_bytes": (OUT / "model.joblib").stat().st_size}

    evaluation_record = {
        "protocol": PROTOCOL_ID, "protocol_version": protocol.PROTOCOL_VERSION,
        "result_type": "DEVELOPMENT (tune folds 1-6) / CONFIRMATION (folds 7-12); NOT a holdout result",
        "selection_evidence": "research/autoresearch_2026_09_23/results/exp22_window_end_protocol.json",
        "folds": fold_records, "summary": summary, "performance": performance,
        "final_training_rows": int(len(final_train)), "final_training_positives": int(final_train.y.sum()),
    }
    (OUT / "evaluation.json").write_text(json.dumps(evaluation_record, indent=2, default=str))
    (OUT / "calibration.json").write_text(json.dumps({
        **calibrator, "version": f"{MODEL_VERSION}/platt-1", "method": "Platt on logit(raw)",
        "fitted_on": f"out-of-fold raw scores of folds 9-12 ({len(recent)} rows)",
        "per_fold_honest_evaluation": "folds 5-12 in evaluation.json, each calibrated on the 4 previous folds",
    }, indent=2))
    (OUT / "feature_schema.json").write_text(json.dumps({
        "schema_version": PS_FEATURE_SCHEMA_VERSION, "feature_contract": FEATURE_CONTRACT_VERSION,
        "features": FEATURES, "groups": {k: v for k, v in PS_FEATURE_GROUPS.items() if k != "E_network"}}, indent=2))
    manifest = {
        "schema": "obsidianchain.ps_model_manifest/3", "model_version": MODEL_VERSION, "model_type": "LightGBM",
        "status": "FROZEN_PENDING_HOLDOUT", "holdout_evaluated": False,
        "feature_schema_version": PS_FEATURE_SCHEMA_VERSION, "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "features": FEATURES, "hyperparameters": HYPERPARAMETERS, "random_state": SEED,
        "train_window_timesteps": TRAIN_WINDOW, "evaluation_protocol": PROTOCOL_ID,
        "final_training_set": f"addresses first seen in t{protocol.DEVELOPMENT_END - TRAIN_WINDOW + 1}-"
                              f"{protocol.DEVELOPMENT_END}, snapshot as of t{protocol.DEVELOPMENT_END}",
        "ranking_score": "raw model probability", "display_probability": "calibration.json",
        "explanation_method": "TreeSHAP via LightGBM pred_contrib, per-feature sign",
        "severity_policy": "rank budget per run (ml/ps_model.py SEVERITY_BUDGET)",
        "model_sha256": _sha256(OUT / "model.joblib"),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generated_by": "research/reproduction/train_ps_production_model.py",
    }
    manifest["artifacts"] = {n: _sha256(OUT / n) for n in ("model.joblib", "evaluation.json",
                                                            "calibration.json", "feature_schema.json")}
    manifest["lineage"] = lineage(manifest["artifacts"]["evaluation.json"])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    s = summary
    print(f"all folds: nAP {s['all']['address']['nap']['mean']:.3f} (min {s['all']['address']['nap']['min']:.3f}); "
          f"confirm nAP {s['confirm']['address']['nap']['mean']:.3f}; P@100 {s['all']['address']['P@100']['mean']:.3f}; "
          f"ECE {s['all']['calibration']['ece'].get('mean', float('nan')):.3f}; load {load_seconds:.2f}s; "
          f"p95 single-row {performance['single_row_latency_ms']['p95']} ms; {time.time()-t0:.0f}s total")
    return manifest


if __name__ == "__main__":
    main()
