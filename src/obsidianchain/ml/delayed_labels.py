"""Delayed-label evaluation: prediction -> later truth -> model-health report.

Labels for Bitcoin addresses arrive weeks or months after a run (case
outcomes, sanctions designations, exchange reports). This module joins a
run's ``predictions.parquet`` - written at scoring time, never modified - to
a labels file that arrived later, and scores what the model said THEN
against what is known NOW.

It is read-only with respect to the model: labels never flow back into a
scored run or into inference. They feed monitoring and, through a new
research cycle under ml/protocol.py, a future model.

Labels file: CSV with ``address`` and ``label`` (1 illicit, 0 licit);
anything else (unknown, blank) is excluded and counted, not treated as 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from obsidianchain.ml import evaluation

#: Below this many labelled scored addresses, or with fewer than this many
#: positives, a report says the sample is too small rather than print noise.
MIN_LABELLED = 50
MIN_POSITIVES = 5


def load_labels(path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(path, dtype={"address": str})
    if not {"address", "label"} <= set(raw.columns):
        raise ValueError("labels file needs 'address' and 'label' columns")
    label = pd.to_numeric(raw["label"], errors="coerce")
    usable = raw[label.isin([0, 1])].assign(label=label[label.isin([0, 1])].astype(int))
    conflicts = usable.groupby("address")["label"].nunique()
    conflicted = set(conflicts[conflicts > 1].index)
    usable = usable[~usable.address.isin(conflicted)].drop_duplicates("address")
    return usable[["address", "label"]], {
        "rows": int(len(raw)), "usable": int(len(usable)),
        "excluded_not_binary": int((~label.isin([0, 1])).sum()),
        "excluded_conflicting": len(conflicted),
    }


def model_health(run_dir: str | Path, labels_path: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    labels, label_stats = load_labels(labels_path)
    joined = preds.merge(labels, on="address", how="inner")
    report: dict[str, Any] = {
        "run_id": str(preds["run_id"].iloc[0]) if len(preds) else None,
        "model_versions": sorted(preds["model_version"].dropna().unique().tolist()),
        "predictions": int(len(preds)), "labelled_predictions": int(len(joined)),
        "label_coverage": round(len(joined) / len(preds), 4) if len(preds) else 0.0,
        "labels": label_stats,
        "result_type": "DELAYED_LABEL_PRODUCTION_RESULT",
        "caveat": ("Labels arrive for a non-random subset of addresses (those that "
                   "were investigated or designated), so these numbers describe that "
                   "subset and are biased towards alerted addresses."),
    }
    positives = int(joined["label"].sum()) if len(joined) else 0
    if len(joined) < MIN_LABELLED or positives < MIN_POSITIVES or positives == len(joined):
        report["status"] = "INSUFFICIENT_LABELS"
        return report
    report["status"] = "EVALUATED"
    report["scorecard"] = evaluation.full_scorecard(
        joined["label"].to_numpy(), joined["raw_score"].to_numpy(),
        joined["snapshot_txid"].fillna(joined["address"]).to_numpy(),
        prob=joined["calibrated_probability"].to_numpy(), ks=(10, 25, 50, 100, 250))
    return report


def write_model_health(run_dir: str | Path, labels_path: str | Path) -> Path:
    report = model_health(run_dir, labels_path)
    out = Path(run_dir) / "model_health.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return out
