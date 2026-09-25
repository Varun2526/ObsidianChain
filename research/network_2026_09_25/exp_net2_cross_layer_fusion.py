"""exp-net2: BASE vs CROSS (cross-layer coherence line fused at W = 0.5).

Design and decision rule: PREREGISTRATION_exp_net2.md (committed before this
was implemented or run). SYNTHETIC_CONTROL: every number is a property of
the v2 generator.

One production pipeline run per world. The cross-layer line is computed but
not fused in production, so CROSS is formed exactly from BASE by noisy-OR:
fused_CROSS = 1 - (1 - fused_BASE) * (1 - W * s).
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from obsidianchain.ml import protocol  # noqa: E402
from obsidianchain.pipeline import alerts as alerts_mod  # noqa: E402
from obsidianchain.pipeline.orchestrator import run_pipeline  # noqa: E402
from obsidianchain.world.noisy import RELAY_LAUNDERING  # noqa: E402

WORLDS = ROOT / "data" / "synthetic_world_v2"
OUT = Path(__file__).parent / "results" / "exp_net2_cross_layer_fusion.json"
W = 0.5
SEED = 20260925
N_BOOT = 2000


def nap(y: np.ndarray, s: np.ndarray) -> float:
    return float(protocol.normalised_average_precision(y, s)[0])


def run_world(name: str) -> dict:
    root = WORLDS / name
    manifest = json.loads((root / "world_manifest.json").read_text())
    capture = root / "capture.csv"
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        outcome = run_pipeline(capture, runs_dir=Path(tmp) / "runs")
        run_dir = next((Path(tmp) / "runs").glob("*/"))
        alerts = json.loads((run_dir / "alerts.json").read_text())["alerts"]
        graph = json.loads((run_dir / "investigation_graph.json").read_text())
        members_of: dict[str, list[str]] = {}
        for e in graph["edges"]:
            if e["kind"] == "MEMBER_OF":
                members_of.setdefault(e["target"].split(":", 1)[1], []).append(e["source"].split(":", 1)[1])
        runtime = time.time() - t0
    truth = pd.read_csv(root / "world_truth" / "labels.csv")  # joined after scoring
    behaviour = pd.read_csv(root / "world_truth" / "entities.csv").set_index("entity")["behaviour"]
    y_of = dict(zip(truth.address, truth.y))
    relay_of = {a: behaviour.get(e) == RELAY_LAUNDERING for a, e in zip(truth.address, truth.entity)}
    rows = []
    for a in alerts:
        members = members_of.get(a["cluster_id"]) or [a["primary_address"]]
        assert len(members) == a["member_count"], a["cluster_id"]
        line = next((e for e in a["evidence"] if e["signal_name"] == "cross_layer_relay_coherence"), None)
        s = float(line["score"]) if line and line["status"] == "PRESENT" else 0.0
        rows.append({
            "base": float(a["fused_risk_score"]),
            "s": s,
            "y": int(any(y_of.get(m, 0) == 1 for m in members)),
            "relay": bool(any(relay_of.get(m, False) for m in members)),
        })
    f = pd.DataFrame(rows)
    f["cross"] = 1 - (1 - f.base) * (1 - W * f.s)
    y, base, cross = f.y.to_numpy(), f.base.to_numpy(), f.cross.to_numpy()
    rng = np.random.default_rng(SEED)
    deltas = []
    n = len(f)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        deltas.append(nap(y[i], cross[i]) - nap(y[i], base[i]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    def p_at(scores, k):
        return float(y[np.argsort(-scores, kind="stable")[:k]].mean())

    def relay_recall(scores, k=200):
        top = np.argsort(-scores, kind="stable")[:k]
        total = int(f.relay.sum())
        return None if total == 0 else round(float(f.relay.to_numpy()[top].sum() / total), 4)

    return {
        "world": name, "network_signal": manifest["config"]["network_signal"],
        "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
        "run_fingerprint": getattr(outcome, "run_fingerprint", None),
        "alerts": n, "positive_alerts": int(y.sum()),
        "line_present": int((f.s > 0).sum()), "line_present_positive": int(((f.s > 0) & (f.y == 1)).sum()),
        "nap_base": round(nap(y, base), 4), "nap_cross": round(nap(y, cross), 4),
        "delta_nap": round(nap(y, cross) - nap(y, base), 4),
        "delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_at_50": {"base": p_at(base, 50), "cross": p_at(cross, 50)},
        "p_at_200": {"base": p_at(base, 200), "cross": p_at(cross, 200)},
        "relay_recall_top200": {"base": relay_recall(base), "cross": relay_recall(cross)},
        "runtime_seconds": round(runtime),
    }


def main() -> None:
    assert not alerts_mod.FUSE_CROSS_LAYER, "BASE must be the unfused production score"
    results = {name: run_world(name) for name in ("signal", "null")}
    for r in results.values():
        print(f"{r['world']:>6}: alerts {r['alerts']} pos {r['positive_alerts']} line {r['line_present']} "
              f"(pos {r['line_present_positive']})  nAP {r['nap_base']} -> {r['nap_cross']}  "
              f"delta {r['delta_nap']:+.4f} CI {r['delta_ci95']}  relay@200 {r['relay_recall_top200']}")
    sig, nul = results["signal"], results["null"]
    if min(sig["positive_alerts"], nul["positive_alerts"]) < 50:
        conclusion = "VOID"
    elif nul["delta_ci95"][0] > 0:
        conclusion = "METHOD_MANUFACTURES_SIGNAL"
    elif sig["delta_ci95"][0] > 0:
        conclusion = "PROMOTE"
    else:
        conclusion = "NOT_DEMONSTRATED"
    print("conclusion:", conclusion)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "experiment_id": "exp_net2_cross_layer_fusion", "provenance_type": "SYNTHETIC_CONTROL",
        "preregistration": "research/network_2026_09_25/PREREGISTRATION_exp_net2.md",
        "weight": W, "seed": SEED, "bootstrap_resamples": N_BOOT,
        "results": results, "conclusion": conclusion,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
