"""exp-net1: CORE vs CORE+network features on the v2 SIGNAL and NULL worlds.

Design and decision rule: PREREGISTRATION.md (committed before this ran).
SYNTHETIC_CONTROL: every number here is a property of the generator.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from obsidianchain.io import ingest  # noqa: E402
from obsidianchain.ml import diagnostics, protocol  # noqa: E402
from obsidianchain.pipeline.features_ps import (  # noqa: E402
    CORE_PS_FEATURE_COLUMNS, GROUP_E_NETWORK, PS_FEATURE_SCHEMA_VERSION, extract_ps_features,
)
from obsidianchain.world.noisy import BASE_TIMESTAMP, RELAY_LAUNDERING, TIMESTEP_SECONDS  # noqa: E402

WORLDS = ROOT / "data" / "synthetic_world_v2"
OUT = Path(__file__).parent / "results" / "exp_net1_network_features.json"
WINDOW = 16
SEED = 20260925
V5 = json.loads((ROOT / "data/models/ps_native/v5/manifest.json").read_text())


def model() -> LGBMClassifier:
    return LGBMClassifier(**V5["hyperparameters"], random_state=SEED, verbose=-1, n_jobs=-1)


def events(root: Path) -> pd.DataFrame:
    frame, report = ingest.ingest(root / "capture.csv")
    assert report.ok, report.errors[:3]
    ev = extract_ps_features(frame, include_network=True).reset_index(drop=True)
    ev["_seq"] = np.arange(len(ev))
    ev["_step"] = ((pd.to_numeric(ev["timestamp"]) - BASE_TIMESTAMP) // TIMESTEP_SECONDS + 1).astype(int)
    ev["_first"] = ev.groupby("address")["_step"].transform("min")
    truth = pd.read_csv(root / "world_truth" / "labels.csv")
    behaviour = pd.read_csv(root / "world_truth" / "entities.csv").set_index("entity")["behaviour"]
    ev = ev.merge(truth[["address", "y", "entity"]], on="address", how="inner")  # labels joined last
    ev["behaviour"] = ev["entity"].map(behaviour).fillna("EXTERNAL")
    return ev[ev["_first"] <= protocol.DEVELOPMENT_END]


def snapshot(ev: pd.DataFrame, lo: int, hi: int, as_of: int) -> pd.DataFrame:
    rows = ev[(ev._first >= lo) & (ev._first <= hi) & (ev._step <= as_of)]
    return rows.sort_values("_seq").drop_duplicates("address", keep="last")


def run_world(name: str) -> dict:
    root = WORLDS / name
    manifest = json.loads((root / "world_manifest.json").read_text())
    capture_sha = hashlib.sha256((root / "capture.csv").read_bytes()).hexdigest()
    ev = events(root)
    core = diagnostics.healthy_features(snapshot(ev, 1, protocol.DEVELOPMENT_END, protocol.DEVELOPMENT_END),
                                        list(CORE_PS_FEATURE_COLUMNS))
    network = [c for c in GROUP_E_NETWORK if c in ev.columns]
    folds = []
    for fold in protocol.rolling_origin_folds():
        tr = snapshot(ev, fold.train_end - WINDOW + 1, fold.train_end, fold.train_end)
        te = snapshot(ev, fold.eval_start, fold.eval_end, fold.eval_end)
        if tr.y.nunique() < 2 or te.y.nunique() < 2:
            continue
        row = {"fold": fold.label, "part": "tune" if fold.index < 6 else "confirm",
               "n_train": int(len(tr)), "n_eval": int(len(te)), "prevalence": round(float(te.y.mean()), 4)}
        relay = (te.behaviour == RELAY_LAUNDERING).to_numpy()
        neg = (te.y == 0).to_numpy()
        for tag, feats in (("CORE", core), ("CORE_E", core + network)):
            s = model().fit(tr[feats], tr.y).predict_proba(te[feats])[:, 1]
            row[f"nap_{tag}"] = round(protocol.normalised_average_precision(te.y, s)[0], 4)
            if relay.any() and neg.any():
                y_rel = np.r_[np.ones(relay.sum()), np.zeros(neg.sum())]
                row[f"relay_auc_{tag}"] = round(float(roc_auc_score(y_rel, np.r_[s[relay], s[neg]])), 4)
        folds.append(row)
    f = pd.DataFrame(folds)
    verdict = protocol.paired_verdict((f.nap_CORE_E - f.nap_CORE).to_numpy(), a="CORE_E", b="CORE")
    rel = f.dropna(subset=["relay_auc_CORE", "relay_auc_CORE_E"])
    relay_verdict = protocol.paired_verdict((rel.relay_auc_CORE_E - rel.relay_auc_CORE).to_numpy(), a="CORE_E", b="CORE")
    return {
        "world": name, "network_signal": manifest["config"]["network_signal"],
        "world_config_sha256": manifest["config_sha256"], "capture_sha256": capture_sha,
        "core_features": core, "network_features": network, "folds": folds,
        "mean": {c: round(float(f[c].mean()), 4) for c in f.columns if c.startswith(("nap_", "relay_auc_"))},
        "verdict_nap": verdict, "verdict_relay_auc": relay_verdict,
    }


def main() -> None:
    t0 = time.time()
    results = {name: run_world(name) for name in ("signal", "null")}
    for name, r in results.items():
        v = r["verdict_nap"]
        print(f"{name:>6}: nAP CORE {r['mean']['nap_CORE']:.4f}  CORE+E {r['mean']['nap_CORE_E']:.4f}  "
              f"relay AUC {r['mean'].get('relay_auc_CORE', float('nan')):.3f} -> {r['mean'].get('relay_auc_CORE_E', float('nan')):.3f}  "
              f"verdict {v['verdict']} favours {v.get('favours')} diff {v['mean_difference']:+.4f} p {v['p_value']:.4f}")
    saturated = any(r["mean"]["nap_CORE"] >= 0.95 for r in results.values())
    s_fav = results["signal"]["verdict_nap"].get("favours")
    n_fav = results["null"]["verdict_nap"].get("favours")
    conclusion = ("VOID_SATURATED" if saturated else
                  "METHOD_MANUFACTURES_SIGNAL" if n_fav == "CORE_E" else
                  "MECHANISM_DEMONSTRATED" if s_fav == "CORE_E" else
                  "NOT_DEMONSTRATED")
    print("conclusion:", conclusion, f"({time.time() - t0:.0f}s)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment_id": "exp_net1_network_features", "provenance_type": "SYNTHETIC_CONTROL",
        "preregistration": "research/network_2026_09_25/PREREGISTRATION.md",
        "feature_schema": PS_FEATURE_SCHEMA_VERSION, "model_hyperparameters": V5["hyperparameters"], "seed": SEED,
        "results": results, "conclusion": conclusion,
        "production_consequence": "none: the production model is unchanged whatever this shows (see PREREGISTRATION.md)",
        "runtime_seconds": round(time.time() - t0),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
