"""Research-only exp15-protocol test of the current product design.

This runner never loads, changes, or writes a production model artifact or
registry entry.  It uses the current production feature engine and the frozen
v5 training configuration to fit one disposable LightGBM per exp15 fold.
"""

from __future__ import annotations

from pathlib import Path
import statistics

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from obsidianchain.contracts.features import validate_feature_frame
from obsidianchain.io import ingest
from obsidianchain.ml import protocol
from obsidianchain.pipeline.features_ps import (
    CORE_PS_FEATURE_COLUMNS,
    PS_FEATURE_SCHEMA_VERSION,
    extract_ps_features,
    last_snapshot_per_address,
)
from obsidianchain.world.noisy import BASE_TIMESTAMP, TIMESTEP_SECONDS


ROOT = Path(__file__).resolve().parents[3]
CAPTURE = ROOT / "data" / "synthetic_world_v2" / "signal" / "capture.csv"
LABELS = ROOT / "data" / "synthetic_world_v2" / "signal" / "world_truth" / "labels.csv"
RESULT = Path(__file__).with_name("RESULT.md")

# Exact ps_native_v5 configuration from
# research/reproduction/train_ps_production_model.py.
MODEL_CONFIGURATION = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 20260919,
    "verbose": -1,
    "n_jobs": -1,
}
EXP15_NAP = 0.9460


def product_feature_frame() -> pd.DataFrame:
    """Run the serving feature path before loading any evaluation labels."""
    capture, report = ingest.ingest(CAPTURE)
    if not report.ok:
        raise RuntimeError(f"ingest failed: {report.errors[:3]}")

    # This is the exact default serving call in pipeline/orchestrator.py.
    product_features = extract_ps_features(capture)
    snapshots = last_snapshot_per_address(product_features)

    # Labels are joined only after all product features have been materialized.
    labels = pd.read_csv(LABELS, usecols=["address", "y"])
    snapshots = snapshots.merge(labels, on="address", how="left", validate="one_to_one")
    if snapshots["y"].isna().any():
        raise RuntimeError("evaluation labels are missing for one or more address snapshots")
    snapshots["last_t"] = (
        (snapshots["timestamp"] - BASE_TIMESTAMP) // TIMESTEP_SECONDS + 1
    ).astype(int)
    return protocol.development(snapshots, timestep="last_t")


def run() -> list[dict[str, float | int | str]]:
    data = product_feature_frame()
    features = list(CORE_PS_FEATURE_COLUMNS)
    if PS_FEATURE_SCHEMA_VERSION != "ps_native_features/5" or len(features) != 31:
        raise RuntimeError("current product schema is not the expected 31-feature ps_native_features/5 contract")

    results: list[dict[str, float | int | str]] = []
    for fold in protocol.rolling_origin_folds():
        # Exact exp15 research unit and boundaries: last snapshot per address,
        # train through train_end, evaluate only the next two timesteps.
        train = data[data["last_t"] <= fold.train_end]
        test = data[(data["last_t"] >= fold.eval_start) & (data["last_t"] <= fold.eval_end)]
        if train["y"].nunique() != 2 or test["y"].nunique() != 2:
            raise RuntimeError(f"{fold.label}: both train and test require two label classes")
        for name, subset in (("train", train), ("test", test)):
            contract = validate_feature_frame(subset, features)
            if not contract.ok:
                raise RuntimeError(f"{fold.label} {name} feature contract violation: {contract.violations[:3]}")

        # A new, in-memory research model for this fold only. No production
        # artifact is loaded or changed, and no test label is passed to fit.
        model = LGBMClassifier(**MODEL_CONFIGURATION)
        model.fit(train[features], train["y"])
        raw_scores = model.predict_proba(test[features])[:, 1]
        nap, prevalence = protocol.normalised_average_precision(test["y"], raw_scores)
        results.append({
            "fold": fold.label,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "prevalence": float(prevalence),
            "nap": float(nap),
        })
    return results


def render(rows: list[dict[str, float | int | str]]) -> str:
    naps = [float(row["nap"]) for row in rows]
    mean = float(np.mean(naps))
    std = float(statistics.stdev(naps))
    table = "\n".join(
        f"| {r['fold']} | {r['n_train']} | {r['n_test']} | {float(r['prevalence']):.4f} | {float(r['nap']):.4f} |"
        for r in rows
    )
    config = "\n".join(f"- `{key}`: `{value}`" for key, value in MODEL_CONFIGURATION.items())
    conclusion = (
        "A. Current product design reproduces research-level performance"
        if mean >= EXP15_NAP else
        "B. Current product design does not reproduce research-level performance"
    )
    return f"""# Current Product Feature/Model Design on SIGNAL

## Result

- **Mean address-level nAP:** `{mean:.4f}`
- **Sample standard deviation across 12 folds:** `{std:.4f}`
- **exp15 research nAP:** `{EXP15_NAP:.4f}`
- **Difference from exp15:** `{mean - EXP15_NAP:+.4f}`

| Fold | Train addresses | Test addresses | Test prevalence | Address-level nAP |
| --- | ---: | ---: | ---: | ---: |
{table}

## Exact feature schema

- `ps_native_features/5`
- `31` features: the current `CORE_PS_FEATURE_COLUMNS` (A/B/C/D/F/G) from `src/obsidianchain/pipeline/features_ps.py`.
- Feature extraction: current serving path, `extract_ps_features(capture)` then `last_snapshot_per_address(...)`.

## Exact temporary research model configuration

LightGBM, copied exactly from the current `ps_native_v5` training configuration in `research/reproduction/train_ps_production_model.py`:

{config}

Each fold creates one in-memory model and discards it after scoring. The frozen `data/models/ps_native/v5/model.joblib` artifact and the model registry are not loaded, written, or changed.

## Exact folds and metric

- `obsidianchain.ml.protocol.rolling_origin_folds()` — 12 expanding, time-ordered folds: train through `t16`, `t18`, …, `t38`; evaluate `t17-18`, `t19-20`, …, `t39-40`.
- Dataset: `data/synthetic_world_v2/signal/capture.csv`.
- Evaluation labels: `data/synthetic_world_v2/signal/world_truth/labels.csv`.
- Metric: `obsidianchain.ml.protocol.normalised_average_precision` on address-level raw LightGBM probabilities, matching exp15.

## Label-separation confirmation

Features are fully generated from the capture through the current product feature pipeline before `labels.csv` is read or joined. For every fold, `fit(...)` receives only that fold's training features and training labels; test labels are used only by the nAP calculation after test scores are produced. No future-fold row or test label is used during training.

## Conclusion

**{conclusion}.**
"""


if __name__ == "__main__":
    RESULT.write_text(render(run()), encoding="utf-8")
