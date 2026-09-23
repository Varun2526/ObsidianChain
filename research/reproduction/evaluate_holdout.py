"""The single sealed-holdout evaluation of ps_native_v5 (ADR 0003).

Refuses to run unless:
  * OBSIDIANCHAIN_HOLDOUT_EXCEPTION points at the committed ADR 0003,
  * no lock file exists for the model version (one opening only),
  * every model verifies against the registry,
  * test.parquet has its sealed MD5 before and after.

Writes data/models/ps_native/holdout/<version>.json (the immutable holdout
scorecard) and <version>.lock (its SHA-256).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "reproduction"))

from build_ps_dataset import build_canonical_frame  # noqa: E402
from obsidianchain.ml import evaluation, protocol, registry  # noqa: E402
from obsidianchain.ml.ps_model import PsNativeRiskModel  # noqa: E402
from obsidianchain.pipeline.features_ps import PsTemporalFeatureEngine  # noqa: E402

SEALED_MD5 = "a15500c94b9808cd42d584ad4b5c3017"
SEALED_FILE = ROOT / "data" / "models" / "ps_native" / "datasets" / "test.parquet"
MODELS_ROOT = ROOT / "data" / "models" / "ps_native"
OUT_DIR = MODELS_ROOT / "holdout"
EXCEPTION = "docs/decisions/0003-holdout-exception-protocol-b.md"
VERSIONS = ("ps_native_v5", "ps_native_v5_fallback_no_g")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def evaluate(models: dict, lo: int, hi: int) -> dict:
    """Score addresses first seen in [lo, hi], snapshotted at their last event <= hi."""
    frame = build_canonical_frame(ROOT / "data" / "raw", max_step=hi)
    feats = PsTemporalFeatureEngine().process_records(frame)
    feats["_step"] = feats.txid.map(dict(zip(frame.txid, frame["_step"]))).astype(int)
    feats["_seq"] = np.arange(len(feats))
    first_step = feats.groupby("address")._step.min()
    feats["_first"] = feats.address.map(first_step)
    hold = feats[(feats._first >= lo) & (feats._first <= hi)]
    hold = hold.sort_values("_seq").drop_duplicates("address", keep="last")
    labels = pd.read_csv(ROOT / "data" / "raw" / "wallets_classes.csv").set_index("address")["class"]
    hold = hold.assign(cls=hold.address.map(labels))
    hold = hold[hold.cls.isin([1, 2])].assign(y=lambda d: (d.cls == 1).astype(int)).reset_index(drop=True)
    out = {"schema": "obsidianchain.holdout_scorecard/1",
           "protocol": "protocol_B_new_addresses_at_window_end/1", "period": f"t{lo}-{hi}",
           "n_addresses": int(len(hold)), "positives": int(hold.y.sum()),
           "prevalence": float(hold.y.mean()), "models": {}}
    y = hold.y.to_numpy()
    for v, model in models.items():
        raw = model.raw_scores(hold)
        prob = model.calibrate(raw)
        card = evaluation.full_scorecard(y, raw, hold.txid.to_numpy(), prob=prob, frame=hold)
        steps = {}
        for s in range(lo, hi + 1):
            m = (hold._first == s).to_numpy()
            if m.sum() and 0 < y[m].sum() < m.sum():
                r = evaluation.ranking_metrics(y[m], raw[m], ks=(50, 100))
                steps[f"t{s}"] = {k: r[k] for k in ("n", "positives", "nap", "P@50", "P@100", "r_precision")}
        out["models"][v] = {"model_sha256": model._sha256, "scorecard": card, "by_first_seen_step": steps}
        a = card["address"]
        print(f"{v}: n {a['n']} pos {a['positives']} nAP {a['nap']:.3f} AUC {a['roc_auc']:.3f} "
              f"P@100 {a['P@100']:.2f} Rprec {a['r_precision']:.3f} nR@500 {a['nR@500']:.3f} "
              f"ECE {card['calibration']['ece']:.3f}", flush=True)
    return out


def main() -> None:
    if os.environ.get("OBSIDIANCHAIN_HOLDOUT_EXCEPTION") != EXCEPTION or not (ROOT / EXCEPTION).is_file():
        raise SystemExit(f"set OBSIDIANCHAIN_HOLDOUT_EXCEPTION={EXCEPTION}")
    for v in VERSIONS:
        if (OUT_DIR / f"{v}.lock").exists():
            raise SystemExit(f"{v} has already been evaluated on the holdout; it is not reopened")
    if _md5(SEALED_FILE) != SEALED_MD5:
        raise SystemExit("sealed test.parquet has changed; refusing")
    reg = registry.Registry.open(MODELS_ROOT)
    models = {}
    for v in VERSIONS:
        reg.verify(v)
        models[v] = PsNativeRiskModel.load(reg.model_dir(v), require_live_schema=True)

    protocol.break_seal(f"ADR 0003: single protocol-B evaluation of {', '.join(VERSIONS)}")
    t0 = time.time()
    results = evaluate(models, protocol.HOLDOUT_START, protocol.DATASET_END)
    results.update({"result_type": "HOLDOUT", "exception": EXCEPTION, "sealed_file_md5_before": SEALED_MD5})
    results["sealed_file_md5_after"] = _md5(SEALED_FILE)
    if results["sealed_file_md5_after"] != SEALED_MD5:
        raise SystemExit("sealed file changed during evaluation")
    results["runtime_seconds"] = round(time.time() - t0, 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for v in VERSIONS:
        body = json.dumps({**results, "models": {v: results["models"][v]}}, indent=2, default=str).encode()
        (OUT_DIR / f"{v}.json").write_bytes(body)
        (OUT_DIR / f"{v}.lock").write_text(hashlib.sha256(body).hexdigest() + "\n")
    print("holdout evaluated once; results locked")


if __name__ == "__main__":
    if "--dry-run-development" in sys.argv:
        # Code-path check on a DEVELOPMENT period (in-sample for the frozen
        # model, so its numbers mean nothing); the seal is not touched.
        reg = registry.Registry.open(MODELS_ROOT)
        ms = {v: PsNativeRiskModel.load(reg.model_dir(v), require_live_schema=True) for v in VERSIONS}
        r = evaluate(ms, 39, 41)
        print("dry run ok:", r["n_addresses"], "addresses,", list(r["models"]["ps_native_v5"]["by_first_seen_step"]))
    else:
        main()
