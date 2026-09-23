"""EXPERIMENT exp15_noisy_world_network_ablation.  SYNTHETIC_CONTROL.

Question: when the network layer carries a signal the chain layer does not,
does the production pipeline extract it - and when it carries none, does it
stay silent?

Design (fixed before running):
* Two worlds from ``world/noisy.py``, identical seed and config except
  ``network_signal``: SIGNAL (RELAY_LAUNDERING broadcasts from well-connected
  nodes) and NULL (it broadcasts like NORMAL).
* Each capture goes through the production path unchanged: ``io.ingest``
  then ``PsTemporalFeatureEngine(include_network=True)``, collapsed to one
  row per address at its last observation (the production dataset unit).
* Labels are joined LAST from the quarantined ``world_truth/labels.csv``.
* The 12 protocol rolling folds over t1-41; the world's own t42-49 band is
  not used.
* Candidates: CORE (schema /3 chain features) vs CORE+E (plus the network
  group). LightGBM, the v2 configuration. Paired verdict per world.
* Pre-stated reading: a network contribution is claimed only if CORE+E beats
  CORE in SIGNAL and does not in NULL. CORE nAP must be below 0.95 in both
  worlds, otherwise the task is saturated and the comparison is void (the
  exp12 failure).
* Diagnostic (truth used post hoc only): AUC of RELAY_LAUNDERING addresses
  against all negatives in each fold, CORE vs CORE+E. (First run used recall
  in the top 10%; with 26% prevalence the top decile holds only the easiest
  positives, so it read 0.0 for both and measured nothing. Replaced, and the
  generator was NOT changed after seeing any result.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "reproduction"))

from obsidianchain.io import ingest  # noqa: E402
from obsidianchain.ml import diagnostics, protocol  # noqa: E402
from obsidianchain.pipeline.features_ps import (  # noqa: E402
    CORE_PS_FEATURE_COLUMNS, GROUP_E_NETWORK, extract_ps_features,
)
from obsidianchain.world.noisy import (  # noqa: E402
    BASE_TIMESTAMP, RELAY_LAUNDERING, TIMESTEP_SECONDS, NoisyWorldConfig, write_noisy_world,
)
from train_ps_model_v2 import _model, transaction_weighted_nap  # noqa: E402

WORLDS = ROOT / "data" / "synthetic_world_v2"
OUT = ROOT / "research/autoresearch_2026_09_23/results/exp15_noisy_world_network_ablation.json"


def dataset(root: Path) -> pd.DataFrame:
    frame, report = ingest.ingest(root / "capture.csv")
    assert report.ok, report.errors[:3]
    feats = extract_ps_features(frame, include_network=True)
    last = feats.sort_values("timestamp", kind="stable").groupby("address").last().reset_index()
    truth = pd.read_csv(root / "world_truth" / "labels.csv")
    behaviour = pd.read_csv(root / "world_truth" / "entities.csv").set_index("entity")["behaviour"]
    last = last.merge(truth[["address", "y", "entity"]], on="address", how="left")
    last["behaviour"] = last["entity"].map(behaviour).fillna("EXTERNAL")
    last["last_t"] = ((last["timestamp"] - BASE_TIMESTAMP) // TIMESTEP_SECONDS + 1).astype(int)
    return protocol.development(last, timestep="last_t")


def run_world(name: str, cfg: NoisyWorldConfig) -> dict:
    root = WORLDS / name
    manifest = write_noisy_world(root, cfg)
    dev = dataset(root)
    core = list(CORE_PS_FEATURE_COLUMNS)
    core = diagnostics.healthy_features(dev, core)
    network = [c for c in GROUP_E_NETWORK if c in dev.columns]
    folds = []
    for fold in protocol.rolling_origin_folds():
        tr = dev[dev.last_t <= fold.train_end]
        ev = dev[(dev.last_t >= fold.eval_start) & (dev.last_t <= fold.eval_end)]
        if ev.y.nunique() < 2 or tr.y.nunique() < 2:
            continue
        row = {"fold": fold.label, "n_eval": int(len(ev)), "prevalence": round(float(ev.y.mean()), 4)}
        relay = (ev.behaviour == RELAY_LAUNDERING).to_numpy()
        for tag, feats in (("CORE", core), ("CORE_E", core + network)):
            s = _model().fit(tr[feats], tr.y).predict_proba(ev[feats])[:, 1]
            row[f"nap_{tag}"] = round(protocol.normalised_average_precision(ev.y, s)[0], 4)
            row[f"txw_{tag}"] = round(transaction_weighted_nap(ev.y, s, ev.txid), 4)
            neg = (ev.y == 0).to_numpy()
            if relay.any() and neg.any():
                y_rel = np.r_[np.ones(relay.sum()), np.zeros(neg.sum())]
                row[f"relay_auc_{tag}"] = round(float(roc_auc_score(y_rel, np.r_[s[relay], s[neg]])), 4)
        folds.append(row)
    f = pd.DataFrame(folds)
    verdict = protocol.paired_verdict((f.nap_CORE_E - f.nap_CORE).to_numpy(), a="CORE_E", b="CORE")
    relay_verdict = protocol.paired_verdict(
        (f.relay_auc_CORE_E - f.relay_auc_CORE).dropna().to_numpy(), a="CORE_E", b="CORE")
    return {
        "world": name, "manifest_counts": manifest["counts"], "n_dev_rows": int(len(dev)),
        "dev_prevalence": round(float(dev.y.mean()), 4), "core_features": core,
        "network_features": network, "folds": folds,
        "mean": {c: round(float(f[c].mean()), 4) for c in f.columns if c not in ("fold",)},
        "verdict_CORE_E_vs_CORE": verdict,
        "verdict_relay_auc": relay_verdict,
    }


def main() -> None:
    results = {}
    for name, signal in (("signal", True), ("null", False)):
        res = run_world(name, NoisyWorldConfig(network_signal=signal))
        results[name] = res
        m, v = res["mean"], res["verdict_CORE_E_vs_CORE"]
        print(f"{name:>6}: folds {len(res['folds'])}  nAP CORE {m['nap_CORE']:.3f}  CORE+E {m['nap_CORE_E']:.3f}  "
              f"relay AUC {m['relay_auc_CORE']:.3f} -> {m['relay_auc_CORE_E']:.3f}  "
              f"verdict {v['verdict']} {v.get('favours')} diff {v['mean_difference']:.4f} p {v['p_value']:.4f}")
    saturated = any(results[w]["mean"]["nap_CORE"] >= 0.95 for w in results)
    s_v, n_v = results["signal"]["verdict_CORE_E_vs_CORE"], results["null"]["verdict_CORE_E_vs_CORE"]
    conclusion = (
        "VOID_SATURATED" if saturated else
        "NETWORK_SIGNAL_EXTRACTED" if s_v.get("favours") == "CORE_E" and n_v.get("favours") != "CORE_E" else
        "NOT_DEMONSTRATED"
    )
    print("conclusion:", conclusion)
    OUT.write_text(json.dumps({"experiment_id": "exp15_noisy_world_network_ablation",
                               "provenance_type": "SYNTHETIC_CONTROL", "design": __doc__,
                               "results": results, "conclusion": conclusion}, indent=2, default=str))


if __name__ == "__main__":
    main()
