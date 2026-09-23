"""Development-calibrated drift baseline for a frozen model (exp23).

A fixed PSI threshold (0.25) read MAJOR_SHIFT on every single timestep of
Elliptic++, including windows where the model ranks at nAP 0.98: one two-week
window always differs from a 16-step training mix. An alert that is always
on carries no information.

This builds, from DEVELOPMENT timesteps only (t <= 41, protocol-B per-step
snapshots of addresses first seen in that step), the distribution of the
model's own drift statistics across ordinary windows, and stores its
upper quantiles. monitoring.compare_to_reference then reads a run relative to
that baseline: ABNORMAL only when it exceeds what normal windows show.

It never touches a model artifact or a score; it is a separate, versioned
monitoring artifact. The holdout period is not read.

usage: python research/reproduction/build_monitoring_baseline.py ps_native_v5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "reproduction"))
sys.path.insert(0, str(ROOT / "research" / "autoresearch_2026_09_23" / "scripts"))

from obsidianchain.ml import monitoring, protocol, registry  # noqa: E402
from obsidianchain.ml.ps_model import PsNativeRiskModel  # noqa: E402
from exp22_window_end_protocol import event_features  # noqa: E402

FIRST_STEP = 17
QUANTILE = 0.95


def main(version: str) -> Path:
    reg = registry.Registry.open()
    reg.verify(version)
    model = PsNativeRiskModel.load(reg.model_dir(version), require_live_schema=True)
    ev = event_features()  # development period only
    per_step = []
    for s in range(FIRST_STEP, protocol.DEVELOPMENT_END + 1):
        snap = ev[ev._first == s].sort_values("_seq").drop_duplicates("address", keep="last")
        r = model.drift_report(snap)
        per_step.append({"step": s, "score_psi": r["score_psi"], "n_major": len(r["major_shift_features"]),
                         "features": {k: v["psi"] for k, v in r["features"].items()}})
    feats = per_step[0]["features"].keys()
    baseline = {
        "schema": "obsidianchain.monitoring_baseline/1", "model_version": version,
        "model_sha256": model._sha256, "quantile": QUANTILE,
        "calibrated_on": f"development timesteps t{FIRST_STEP}-{protocol.DEVELOPMENT_END}, per-step snapshots",
        "score_psi": float(np.quantile([p["score_psi"] for p in per_step], QUANTILE)),
        "n_major": float(np.quantile([p["n_major"] for p in per_step], QUANTILE)),
        "feature_psi": {f: float(np.quantile([p["features"][f] for p in per_step], QUANTILE)) for f in feats},
        "per_step": per_step,
    }
    out = ROOT / "data" / "models" / "ps_native" / "monitoring" / f"{version}_baseline.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2))
    print(f"wrote {out}: score_psi p95 {baseline['score_psi']:.3f}, major-feature count p95 {baseline['n_major']:.1f}")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ps_native_v5")
