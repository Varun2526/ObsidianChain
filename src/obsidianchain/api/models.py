"""Model intelligence: the PS-native registry and its recorded evaluations.

Reads the JSON records the ML pipeline wrote under
``<data_root>/models/ps_native/``: the registry, each version's manifest,
``evaluation.json``, ``calibration.json``, the production-gate report, the
locked holdout scorecard and the drift baseline. It never imports
``obsidianchain.ml`` (forbidden in this layer) and computes nothing: every
number is one a training or evaluation script already recorded.

Every metric block carries its RESULT TYPE (DEVELOPMENT / CONFIRMATION /
HOLDOUT) so a research number can never be read as a production one. There
is no PRODUCTION performance block, because no labelled production traffic
exists; the response says so rather than leaving a gap to be filled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from obsidianchain.api import artifacts

ADDRESS_KEYS = ("nap", "roc_auc", "r_precision", "P@10", "P@50", "P@100", "P@500", "R@100", "R@500",
                "R@1000", "nR@500", "FP@100")


class ModelNotFoundError(LookupError):
    pass


def _root(root=None) -> Path:
    return artifacts.data_root(root) / "models" / "ps_native"


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _registry(root=None) -> dict:
    reg = _json(_root(root) / "registry.json")
    if reg is None:
        raise artifacts.ArtifactMissingError(
            f"{_root(root) / 'registry.json'} not found; register a model with 'obsidianchain model register'")
    return reg


def list_models(root=None) -> dict:
    reg = _registry(root)
    roles = {v: r for r, v in reg["roles"].items() if v}
    rows = []
    for version, entry in sorted(reg["models"].items()):
        rows.append({
            "version": version,
            "role": roles.get(version),
            "feature_schema_version": entry.get("feature_schema_version"),
            "registered_at": entry.get("registered_at"),
            "attested_source_commit": entry.get("attested_source_commit"),
            "holdout": entry.get("holdout"),
            "notes": entry.get("notes"),
            "withdrawn": "WITHDRAWN" in str(entry.get("notes", "")),
        })
    return {"roles": reg["roles"], "models": rows,
            "history": reg.get("history", [])[-25:],
            "meaning": ("The registry is the only authority on which model serves. Champion serves, "
                        "candidate runs in shadow, fallback serves only if the champion fails verification.")}


def _slim_fold(f: dict) -> dict:
    a = f.get("address", {})
    cal = f.get("calibration") or {}
    return {"fold": f.get("fold"), "part": f.get("part"),
            "result_type": "DEVELOPMENT" if f.get("part") == "tune" else "CONFIRMATION",
            "n": a.get("n"), "positives": a.get("positives"), "prevalence": a.get("prevalence"),
            **{k: a.get(k) for k in ADDRESS_KEYS},
            "tx_weighted_nap": f.get("tx_weighted_nap"),
            "ece": cal.get("ece"), "brier": cal.get("brier")}


def get_model(version: str, root=None) -> dict:
    reg = _registry(root)
    entry = reg["models"].get(version)
    if entry is None:
        raise ModelNotFoundError(f"model {version!r} is not registered")
    base = _root(root)
    mdir = base / entry["path"]
    manifest = _json(mdir / "manifest.json") or {}
    evaluation = _json(mdir / "evaluation.json")
    calibration = _json(mdir / "calibration.json")
    stacker = _json(mdir / "stacker.json")
    gate = _json(base / "gates" / f"{version}.json")
    holdout = _json(base / "holdout" / f"{version}.json")
    baseline = _json(base / "monitoring" / f"{version}_baseline.json")
    roles = [r for r, v in reg["roles"].items() if v == version]

    out: dict[str, Any] = {
        "version": version, "roles": roles,
        "feature_schema_version": entry.get("feature_schema_version"),
        "attested_source_commit": entry.get("attested_source_commit"),
        "registered_at": entry.get("registered_at"),
        "notes": entry.get("notes"),
        "manifest": {k: manifest.get(k) for k in (
            "model_type", "status", "evaluation_protocol", "final_training_set", "hyperparameters",
            "train_window_timesteps", "features", "ranking_score", "display_probability",
            "explanation_method", "severity_policy", "model_sha256", "created_at")},
        "result_types": {
            "DEVELOPMENT": "rolling folds 1-6 (t17-28), used for selection",
            "CONFIRMATION": "rolling folds 7-12 (t29-40), never used for selection",
            "HOLDOUT": "t42-49, opened once for this version (ADR 0003)",
            "PRODUCTION": "no labelled production traffic exists yet",
        },
        "production": {"available": False,
                       "meaning": "No production performance exists until delayed labels arrive "
                                  "(obsidianchain model health)."},
    }
    if evaluation:
        s = evaluation.get("summary", {})
        out["evaluation"] = {
            "protocol": evaluation.get("protocol"),
            "summary": {part: {"result_type": {"tune": "DEVELOPMENT", "confirm": "CONFIRMATION",
                                               "all": "DEVELOPMENT+CONFIRMATION"}[part],
                               "address": {k: s[part]["address"].get(k) for k in ADDRESS_KEYS
                                           if k in s[part]["address"]},
                               "calibration": s[part].get("calibration")}
                        for part in ("tune", "confirm", "all") if part in s},
            "temporal_trend_nap": s.get("temporal_trend_nap"),
            "slices_nap": s.get("slices_nap"),
            "folds": [_slim_fold(f) for f in evaluation.get("folds", [])],
            "performance": evaluation.get("performance"),
        }
    if calibration:
        out["calibration"] = calibration
    if stacker:
        out["stacker"] = {k: stacker.get(k) for k in ("inputs", "coef", "intercept", "applied_when",
                                                       "out_of_sample_check", "transfer_caveat")}
    if gate:
        out["gate"] = {"decision": gate.get("decision"), "evaluated_at": gate.get("evaluated_at"),
                       "spec": gate.get("spec"),
                       "criteria": {k: {"status": v.get("status"), "evidence": v.get("evidence")}
                                    for k, v in gate.get("criteria", {}).items()}}
    if holdout:
        card = holdout["models"][version]["scorecard"]
        out["holdout"] = {
            "result_type": "HOLDOUT", "period": holdout.get("period"), "exception": holdout.get("exception"),
            "n_addresses": holdout.get("n_addresses"), "positives": holdout.get("positives"),
            "prevalence": holdout.get("prevalence"),
            "address": {k: card["address"].get(k) for k in ADDRESS_KEYS if k in card["address"]},
            "calibration": {k: v for k, v in card.get("calibration", {}).items() if k != "reliability"},
            "reliability": card.get("calibration", {}).get("reliability"),
            "slices_nap": {k: v.get("nap") for k, v in card.get("slices", {}).items()},
            "by_first_seen_step": holdout["models"][version].get("by_first_seen_step"),
            "sha256": (entry.get("holdout") or {}).get("sha256"),
        }
    if baseline:
        out["drift_baseline"] = {"calibrated_on": baseline.get("calibrated_on"),
                                 "quantile": baseline.get("quantile"),
                                 "score_psi_p95": baseline.get("score_psi"),
                                 "per_step": [{"step": p["step"], "score_psi": p["score_psi"],
                                               "n_major": p["n_major"]} for p in baseline.get("per_step", [])],
                                 "finding": ("Input drift does not detect the holdout collapse at t43/t45 "
                                             "(exp23); performance is verified only by delayed labels.")}
    return out
