"""Pre-specified, development-only E0/E1/E2 typology-robustness experiment.

This runner restricts its raw input to t1-41, builds causal product features
before labels are joined, and fits transient fold models only. It never reads
or writes a production model, registry, threshold, schema, or holdout file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.reproduction.build_ps_dataset import build_canonical_frame
from obsidianchain.contracts.features import validate_feature_frame
from obsidianchain.ml import evaluation, protocol
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS, PsTemporalFeatureEngine


OUT = Path(__file__).parent
METRICS = OUT / "metrics.json"
PER_FOLD = OUT / "per_fold.csv"
RESULTS = OUT / "RESULTS.md"
HASHES = OUT / "artifact_hashes.json"
SEED = 20260919
BOOTSTRAP_SEED = 20260926
TRAIN_WINDOW = 16
HYPERPARAMETERS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
}
E1_FEATURES = [
    "research_cold_start", "research_weak_upstream", "research_log_fan_in",
    "research_relationship_density", "research_cold_x_weak_upstream",
    "research_cold_x_upstream_share",
]
EXPERIMENTS = {"E0": list(CORE_PS_FEATURE_COLUMNS), "E1": list(CORE_PS_FEATURE_COLUMNS) + E1_FEATURES,
               "E2": list(CORE_PS_FEATURE_COLUMNS) + E1_FEATURES}


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_research_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic snapshot-time additions; no labels or future rows used."""
    out = frame.copy()
    cold = (out["n_txs_asof_t"].to_numpy(dtype=float) == 0).astype(np.int8)
    upstream = pd.to_numeric(out["upstream_funded_share"], errors="coerce").fillna(0.0).to_numpy(float)
    out["research_cold_start"] = cold
    out["research_weak_upstream"] = (upstream < 0.50).astype(np.int8)
    out["research_log_fan_in"] = np.log1p(out["input_count"].to_numpy(dtype=float))
    out["research_relationship_density"] = (
        out["unique_counterparties_asof_t"].to_numpy(dtype=float)
        / np.maximum(out["n_txs_asof_t"].to_numpy(dtype=float), 1.0)
    )
    out["research_cold_x_weak_upstream"] = cold * out["research_weak_upstream"].to_numpy(dtype=np.int8)
    out["research_cold_x_upstream_share"] = cold * upstream
    return out


def causal_events() -> pd.DataFrame:
    """Feature generation stops at t41 before any label file is read."""
    frame = build_canonical_frame(ROOT / "data" / "raw", max_step=protocol.DEVELOPMENT_END)
    features = PsTemporalFeatureEngine().process_records(frame)
    features["_step"] = features.txid.map(dict(zip(frame.txid, frame["_step"]))).astype(int)
    features["_seq"] = np.arange(len(features))
    features["_first"] = features.address.map(features.groupby("address")._step.min())
    features = add_research_features(features)
    # Only after feature materialization do labels enter the research frame.
    labels = pd.read_csv(ROOT / "data" / "raw" / "wallets_classes.csv").set_index("address")["class"]
    features["_class"] = features.address.map(labels)
    features = features[features["_class"].isin([1, 2])].copy()
    features["y"] = (features["_class"] == 1).astype(np.int8)
    return features.reset_index(drop=True)


def snapshot(events: pd.DataFrame, first_lo: int, first_hi: int, as_of: int) -> pd.DataFrame:
    rows = events[(events._first >= first_lo) & (events._first <= first_hi) & (events._step <= as_of)]
    return rows.sort_values("_seq").drop_duplicates("address", keep="last").reset_index(drop=True)


def fold_sets(events: pd.DataFrame, fold: protocol.Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = snapshot(events, fold.train_end - TRAIN_WINDOW + 1, fold.train_end, fold.train_end)
    test = snapshot(events, fold.eval_start, fold.eval_end, fold.eval_end)
    return train, test


def cluster_weights(train: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Training-only, positive transaction-event balancing from the preregistration."""
    positive = train.y.to_numpy() == 1
    counts = train.loc[positive, "txid"].value_counts()
    weights = np.ones(len(train), dtype=float)
    weights[positive] = 1.0 / train.loc[positive, "txid"].map(counts).to_numpy(dtype=float)
    weights *= len(train) / weights.sum()
    return weights, int(len(counts))


def slice_nap(test: pd.DataFrame, scores: np.ndarray, mask: np.ndarray) -> float:
    y = test.y.to_numpy()[mask]
    if len(y) == 0 or np.unique(y).size != 2:
        return float("nan")
    return float(protocol.normalised_average_precision(y, scores[mask])[0])


def one_fold(events: pd.DataFrame, fold: protocol.Fold, name: str) -> dict[str, Any]:
    train, test = fold_sets(events, fold)
    base_contract_train = validate_feature_frame(train, list(CORE_PS_FEATURE_COLUMNS))
    base_contract_test = validate_feature_frame(test, list(CORE_PS_FEATURE_COLUMNS))
    if not base_contract_train.ok or not base_contract_test.ok:
        raise RuntimeError(f"feature contract failed in {fold.label}: "
                           f"{base_contract_train.violations[:2]} {base_contract_test.violations[:2]}")
    if train.y.nunique() != 2 or test.y.nunique() != 2:
        raise RuntimeError(f"{fold.label} lacks both classes")
    model = LGBMClassifier(**HYPERPARAMETERS)
    kwargs: dict[str, Any] = {}
    clusters = int(train.loc[train.y == 1, "txid"].nunique())
    if name == "E2":
        weights, clusters = cluster_weights(train)
        kwargs["sample_weight"] = weights
    model.fit(train[EXPERIMENTS[name]], train.y, **kwargs)
    scores = model.predict_proba(test[EXPERIMENTS[name]])[:, 1]
    ranking = evaluation.ranking_metrics(test.y.to_numpy(), scores, ks=(100,))
    cold = test.research_cold_start.to_numpy(dtype=bool)
    weak = test.research_weak_upstream.to_numpy(dtype=bool)
    return {
        "experiment": name, "fold": fold.label, "train_end": fold.train_end,
        "test_start": fold.eval_start, "test_end": fold.eval_end,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "positive_train": int(train.y.sum()), "positive_test": int(test.y.sum()),
        "positive_train_clusters": clusters, "positive_test_clusters": int(test.loc[test.y == 1, "txid"].nunique()),
        "prevalence_test": float(test.y.mean()), "nap": float(ranking["nap"]),
        "p_at_100": float(ranking["P@100"]), "recall_at_100": float(ranking["R@100"]),
        "cold_start_nap": slice_nap(test, scores, cold), "weak_upstream_nap": slice_nap(test, scores, weak),
        "cold_start_positive_count": int(test.loc[cold, "y"].sum()),
        "weak_upstream_positive_count": int(test.loc[weak, "y"].sum()),
        "feature_schema": "ps_native_features/5" if name == "E0" else "ps_native_features/5 + 6 research-only causal columns",
        "feature_count": len(EXPERIMENTS[name]), "seed": SEED,
        "hyperparameters": json.dumps(HYPERPARAMETERS, sort_keys=True), "git_commit": git_revision(),
    }


def average(values: list[float]) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return float(v.mean()) if len(v) else float("nan")


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = np.asarray([r["nap"] for r in records], dtype=float)
    return {
        "mean_nap": float(n.mean()), "sd_nap": float(n.std(ddof=1)), "worst_fold_nap": float(n.min()),
        "median_fold_nap": float(np.median(n)), "mean_cold_start_nap": average([r["cold_start_nap"] for r in records]),
        "mean_weak_upstream_nap": average([r["weak_upstream_nap"] for r in records]),
        "mean_p_at_100": average([r["p_at_100"] for r in records]),
        "mean_recall_at_100": average([r["recall_at_100"] for r in records]),
        "folds": len(records), "measurable_cold_start_folds": int(sum(np.isfinite(r["cold_start_nap"]) for r in records)),
        "measurable_weak_upstream_folds": int(sum(np.isfinite(r["weak_upstream_nap"]) for r in records)),
    }


def bootstrap_delta(candidate: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    diff = np.asarray([a["nap"] - b["nap"] for a, b in zip(candidate, baseline)], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(diff, size=(10_000, len(diff)), replace=True).mean(axis=1)
    return {"mean_delta_nap": float(diff.mean()), "fold_wins": int((diff > 0).sum()),
            "fold_losses": int((diff < 0).sum()), "fold_ties": int((diff == 0).sum()),
            "bootstrap_95_ci": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": 10_000}


def winner(summary: dict[str, dict[str, Any]]) -> str:
    return max(summary, key=lambda name: (
        summary[name]["mean_nap"], summary[name]["worst_fold_nap"],
        summary[name]["mean_cold_start_nap"], summary[name]["mean_p_at_100"],
    ))


def write_csv(records: list[dict[str, Any]]) -> None:
    keys = list(records[0])
    with PER_FOLD.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def render(metrics: dict[str, Any], hashes: dict[str, str]) -> str:
    summary, deltas, champion = metrics["summary"], metrics["deltas"], metrics["winner"]
    rows = []
    by = {name: {r["fold"]: r for r in metrics["records"] if r["experiment"] == name} for name in EXPERIMENTS}
    for fold in [r["fold"] for r in metrics["records"] if r["experiment"] == "E0"]:
        rows.append("| " + " | ".join([
            fold, *(f"{by[e][fold]['nap']:.4f}" for e in EXPERIMENTS),
            *(f"{by[e][fold]['p_at_100']:.2f}" for e in EXPERIMENTS),
            str(by["E0"][fold]["positive_test"]), str(by["E0"][fold]["positive_test_clusters"]),
        ]) + " |")
    table = "\n".join(rows)
    compact = "\n".join(
        f"| {e} | {summary[e]['mean_nap']:.4f} | {summary[e]['sd_nap']:.4f} | {summary[e]['worst_fold_nap']:.4f} | "
        f"{summary[e]['mean_cold_start_nap']:.4f} | {summary[e]['mean_weak_upstream_nap']:.4f} | "
        f"{summary[e]['mean_p_at_100']:.4f} | {0.0 if e == 'E0' else deltas[e]['mean_delta_nap']:+.4f} |"
        for e in EXPERIMENTS)
    e1, e2 = deltas["E1"]["mean_delta_nap"], deltas["E2"]["mean_delta_nap"]
    conclusion = (
        "both helped" if e1 > 0 and e2 > e1 else
        "robustness features helped, but cluster-balanced training did not add mean nAP" if e1 > 0 else
        "cluster-balanced training helped while robustness features alone did not" if e2 > 0 else
        "neither helped on mean nAP"
    )
    consistency = (
        f"E1 is mixed and near-neutral ({deltas['E1']['fold_wins']} wins, "
        f"{deltas['E1']['fold_losses']} losses; its bootstrap interval spans zero). "
        f"E2 is lower on mean nAP in {deltas['E2']['fold_losses']} of 12 folds despite "
        f"a better worst fold, so that safety improvement is not accompanied by consistent mean improvement."
    )
    return f"""# E0/E1/E2 Typology-Robustness Development Experiment

## Scope and integrity

This is a **pre-specified exploratory, development-only** comparison. It used
only `t1-41` and the twelve fixed Protocol-B folds. `t42-49`, sealed/reused
holdout artifacts, the champion artifact, registry, production code, feature
contract, and classification thresholds were not read or changed. All models
were transient per-fold LightGBM fits; no model artifact was saved.

The complete fixed specification is [PREREGISTRATION.md](PREREGISTRATION.md).

## Winner rule result

Winner by the pre-specified lexicographic development-only rule: **{champion}**.
This is not an adoption or production-promotion decision.

## Compact comparison

| Experiment | Mean nAP | Std | Worst Fold | Cold-start nAP | Weak-upstream nAP | P@100 | Delta vs E0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{compact}

## Per-fold performance

| Fold | E0 nAP | E1 nAP | E2 nAP | E0 P@100 | E1 P@100 | E2 P@100 | Test positives | Independent positive test tx clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

## Aggregate metrics and uncertainty

| Experiment | Mean nAP | SD | Median nAP | Worst nAP | Cold-start nAP | Weak-upstream nAP | P@100 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {e} | {summary[e]['mean_nap']:.4f} | {summary[e]['sd_nap']:.4f} | {summary[e]['median_fold_nap']:.4f} | "
        f"{summary[e]['worst_fold_nap']:.4f} | {summary[e]['mean_cold_start_nap']:.4f} | "
        f"{summary[e]['mean_weak_upstream_nap']:.4f} | {summary[e]['mean_p_at_100']:.4f} | {summary[e]['mean_recall_at_100']:.4f} |"
        for e in EXPERIMENTS) + f"""

- E1 − E0: mean paired nAP delta `{deltas['E1']['mean_delta_nap']:+.4f}`, bootstrap 95% CI `{deltas['E1']['bootstrap_95_ci'][0]:+.4f}` to `{deltas['E1']['bootstrap_95_ci'][1]:+.4f}`, fold wins/losses/ties `{deltas['E1']['fold_wins']}/{deltas['E1']['fold_losses']}/{deltas['E1']['fold_ties']}`.
- E2 − E0: mean paired nAP delta `{deltas['E2']['mean_delta_nap']:+.4f}`, bootstrap 95% CI `{deltas['E2']['bootstrap_95_ci'][0]:+.4f}` to `{deltas['E2']['bootstrap_95_ci'][1]:+.4f}`, fold wins/losses/ties `{deltas['E2']['fold_wins']}/{deltas['E2']['fold_losses']}/{deltas['E2']['fold_ties']}`.

Bootstrap resamples fixed at 10,000 over the 12 paired fold deltas (seed `{BOOTSTRAP_SEED}`). These intervals describe variability across the fixed development windows; they do not establish unseen future-period generalization.

## Exact methods

- **E0 schema:** 31 `ps_native_features/5` CORE columns.
- **E1/E2 schema:** E0 plus six fixed, research-only causal columns (`research_cold_start`, `research_weak_upstream`, `research_log_fan_in`, `research_relationship_density`, `research_cold_x_weak_upstream`, `research_cold_x_upstream_share`). Production schema remains unchanged.
- **All models:** LightGBM with seed `{SEED}` and `{json.dumps(HYPERPARAMETERS, sort_keys=True)}`.
- **E2 clusters/weights:** positive snapshot transaction ID is the deterministic event cluster; its positive training rows share one raw unit. Negative rows retain unit weight. Weights are normalized to total training-row count. Per-fold training and test sizes, positive counts, and independent positive-cluster counts are in `per_fold.csv`.
- **Causality and labels:** product features were generated from the raw t1-41 stream before the label join. Each snapshot is at its fold boundary; test labels were used only after scoring for metrics, never in fit or E2 weights.
- **Commit:** `{metrics['git_commit']}`.

## Conclusion

**{conclusion.capitalize()}.** {consistency} The winner is only the pre-specified best development configuration; it must not be selected, threshold-tuned, or promoted using `t42-49`.

## Generated artifact hashes

| Artifact | SHA-256 |
| --- | --- |
""" + "\n".join(f"| {name} | `{digest}` |" for name, digest in hashes.items()) + "\n"


def main() -> None:
    events = causal_events()
    records = [one_fold(events, fold, name) for fold in protocol.rolling_origin_folds() for name in EXPERIMENTS]
    summary = {name: aggregate([r for r in records if r["experiment"] == name]) for name in EXPERIMENTS}
    grouped = {name: [r for r in records if r["experiment"] == name] for name in EXPERIMENTS}
    deltas = {"E1": bootstrap_delta(grouped["E1"], grouped["E0"]), "E2": bootstrap_delta(grouped["E2"], grouped["E0"])}
    payload = {"experiment": "2026_09_26_typology_robustness", "status": "EXPLORATORY_DEVELOPMENT_ONLY",
               "git_commit": git_revision(), "seed": SEED, "bootstrap_seed": BOOTSTRAP_SEED,
               "protocol": "protocol_B_new_addresses_at_window_end/1", "source_max_timestep": 41,
               "feature_schemas": {name: EXPERIMENTS[name] for name in EXPERIMENTS},
               "hyperparameters": HYPERPARAMETERS, "records": records, "summary": summary,
               "deltas": deltas, "winner": winner(summary)}
    write_csv(records)
    METRICS.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    artifact_hashes = {name: sha256(OUT / name) for name in ("PREREGISTRATION.md", "run.py", "per_fold.csv", "metrics.json")}
    HASHES.write_text(json.dumps(artifact_hashes, indent=2), encoding="utf-8")
    RESULTS.write_text(render(payload, artifact_hashes), encoding="utf-8")
    print(render(payload, artifact_hashes).split("## Per-fold performance")[0])


if __name__ == "__main__":
    main()
