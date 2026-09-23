"""Evaluate a model version against docs/production_gate_spec.json.

Each criterion is checked from evidence - test runs, artifact files, the
registry, the evaluation and holdout records - never from a hand-set flag.
Writes data/models/ps_native/gates/<version>.json with PASS / FAIL /
PENDING per criterion and an overall decision. PENDING (the holdout not yet
evaluated) never counts as PASS.

usage: python research/reproduction/production_gate.py ps_native_v4
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "reproduction"))

from obsidianchain.ml import protocol, registry  # noqa: E402

SPEC = ROOT / "docs" / "production_gate_spec.json"
MODELS = ROOT / "data" / "models" / "ps_native"
EXP22 = ROOT / "research/autoresearch_2026_09_23/results/exp22_window_end_protocol.json"


def _pytest(*paths: str) -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", *paths],
                          cwd=ROOT, capture_output=True, text=True)
    tail = (proc.stdout.strip().splitlines() or [""])[-1]
    return proc.returncode == 0, tail


def _res(ok: bool | None, evidence) -> dict:
    return {"status": "PENDING" if ok is None else ("PASS" if ok else "FAIL"), "evidence": evidence}


def main(version: str) -> dict:
    reg = registry.Registry.open(MODELS)
    entry = reg.entry(version)
    mdir = reg.model_dir(version)
    manifest = json.loads((mdir / "manifest.json").read_text())
    ev = json.loads((mdir / "evaluation.json").read_text())
    s = ev["summary"]
    conf, allf = s["confirm"]["address"], s["all"]["address"]
    cal = s["all"]["calibration"]
    out: dict = {}

    ok, tail = _pytest("tests/test_leakage_audit.py")
    out["1_leakage_safety"] = _res(ok, tail)

    folds = protocol.rolling_origin_folds()
    exp22 = json.loads(EXP22.read_text()) if EXP22.is_file() else {}
    out["2_temporal_validity"] = _res(
        all(f.eval_end < protocol.HOLDOUT_START for f in folds) and bool(exp22)
        and ev["protocol"].startswith("protocol_B"),
        {"max_eval_step": max(f.eval_end for f in folds), "selection_record": EXP22.name, "protocol": ev["protocol"]})

    ok, tail = _pytest("tests/test_registry_and_chaos.py::test_conflicting_and_invalid_transactions_are_quarantined",
                       "tests/test_leakage_audit.py::test_development_data_satisfies_the_feature_contract")
    out["3_data_quality"] = _res(ok, tail + "; feature contract enforced on every set by the training script")

    r4 = {"nap": conf["nap"]["mean"], "P@100": conf["P@100"]["mean"], "nR@500": conf["nR@500"]["mean"]}
    out["4_ranking_quality"] = _res(r4["nap"] >= 0.60 and r4["P@100"] >= 0.90 and r4["nR@500"] >= 0.75, r4)

    out["5_calibration"] = _res(cal["ece"]["mean"] <= 0.05 and cal["ece"]["max"] <= 0.10,
                                {"ece_mean": cal["ece"]["mean"], "ece_max": cal["ece"]["max"],
                                 "folds": cal["ece"]["n"]})

    trend = s["temporal_trend_nap"]
    down = trend["p_value"] < 0.05 and trend["slope_per_fold"] < 0
    out["6_stability"] = _res(allf["nap"]["sd"] <= 0.20 and not down, {"nap_sd": allf["nap"]["sd"], "trend": trend})

    out["7_worst_fold"] = _res(allf["nap"]["min"] >= 0.40 and allf["P@100"]["min"] >= 0.70,
                               {"worst_nap": allf["nap"]["min"], "worst_P@100": allf["P@100"]["min"]})

    sl = s["slices_nap"]
    weak = {k: v["mean"] for k, v in sl.items() if v.get("mean", 1) < 0.30}
    out["8_slice_performance"] = _res(sl["first_seen"]["mean"] >= 0.50 and sl["receiver_only"]["mean"] >= 0.50,
                                      {"first_seen": sl["first_seen"]["mean"], "receiver_only": sl["receiver_only"]["mean"],
                                       "warnings_below_0.30": weak,
                                       "all_slices": {k: v.get("mean") for k, v in sl.items()}})

    if not version.endswith("_no_g") and exp22:
        mine = [f["address"]["nap"] for f in ev["folds"]]
        ref = [f["nap"] for f in exp22["g_per_fold"]["CORE"]]
        out["9_reproducibility"] = _res(max(abs(a - b) for a, b in zip(mine, ref)) < 1e-9,
                                        {"max_abs_diff_vs_exp22": max(abs(a - b) for a, b in zip(mine, ref))})
    elif exp22:
        mine = [f["address"]["nap"] for f in ev["folds"]]
        ref = [f["nap"] for f in exp22["g_per_fold"]["NO_G"]]
        out["9_reproducibility"] = _res(max(abs(a - b) for a, b in zip(mine, ref)) < 1e-9,
                                        {"max_abs_diff_vs_exp22_no_g": max(abs(a - b) for a, b in zip(mine, ref))})

    ok, tail = _pytest("tests/test_registry_and_chaos.py::test_a_score_does_not_depend_on_the_batch",
                       "tests/test_registry_and_chaos.py::test_runs_are_reproducible",
                       "tests/test_leakage_audit.py::test_simultaneous_events_do_not_see_each_other")
    out["10_inference_determinism"] = _res(ok, tail)

    perf = ev["performance"]
    out["11_latency"] = _res(perf["single_row_latency_ms"]["p95"] <= 50 and perf["rows_per_second_at_10k"] >= 10000,
                             {"p95_ms": perf["single_row_latency_ms"]["p95"], "rows_per_s": perf["rows_per_second_at_10k"]})

    ok, tail = _pytest("tests/test_registry_and_chaos.py")
    out["12_reliability"] = _res(ok, tail)

    try:
        reg.verify(version)
        verified = True
    except registry.RegistryError as exc:
        verified = str(exc)
    ok, tail = _pytest("tests/test_console_auth.py", "tests/test_api_access.py")
    out["13_security"] = _res(verified is True and ok, {"registry_verify": verified, "auth_tests": tail})

    import joblib
    has_ref = joblib.load(mdir / "model.joblib").get("reference_profile") is not None
    ok, tail = _pytest("tests/test_ps_model_current.py::test_psi_is_zero_for_identical_and_large_for_disjoint_distributions")
    out["14_monitoring"] = _res(has_ref and ok, {"reference_profile": has_ref, "tests": tail})

    fb = reg.role("fallback")
    fb_ok = None
    if fb:
        try:
            reg.verify(fb)
            fb_ok = reg.entry(fb)["feature_schema_version"] == entry["feature_schema_version"] and fb != version
        except registry.RegistryError:
            fb_ok = False
    out["15_rollback"] = _res(bool(fb_ok), {"fallback": fb, "same_schema": fb_ok})

    lin = manifest.get("lineage", {})
    need = ("source_commit", "code_sha256", "config_sha256", "raw_source_sha256", "environment", "evaluation_sha256")
    # Presence is not enough: the recorded commit must provably contain the
    # training code (``obsidianchain model attest``).
    out["16_artifact_lineage"] = _res(all(lin.get(k) for k in need) and bool(entry.get("attested_source_commit")),
                                      {"missing": [k for k in need if not lin.get(k)],
                                       "source_commit": lin.get("source_commit"),
                                       "attested_commit": entry.get("attested_source_commit")})

    ok, tail = _pytest("tests/test_registry_and_chaos.py::test_every_prediction_is_traceable",
                       "tests/test_alert_fusion.py")
    out["17_auditability"] = _res(ok, tail)

    hold_file = MODELS / "holdout" / f"{version}.json"
    if hold_file.is_file():
        h = json.loads(hold_file.read_text())["models"][version]["scorecard"]
        a = h["address"]
        worst_conf = min(f["address"]["nap"] for f in ev["folds"] if f["part"] == "confirm")
        vals = {"nap": a["nap"], "P@100": a["P@100"], "nR@500": a["nR@500"], "ece": h["calibration"]["ece"],
                "worst_confirmation_nap": worst_conf}
        out["18_holdout"] = _res(a["nap"] >= 0.50 and a["P@100"] >= 0.80 and a["nR@500"] >= 0.60
                                 and h["calibration"]["ece"] <= 0.10 and a["nap"] >= worst_conf - 0.05, vals)
    else:
        out["18_holdout"] = _res(None, "holdout not yet evaluated")

    statuses = [c["status"] for c in out.values()]
    decision = "FAIL" if "FAIL" in statuses else ("PENDING" if "PENDING" in statuses else "PASS")
    report = {"schema": "obsidianchain.production_gate_report/1", "model_version": version,
              "spec": str(SPEC.relative_to(ROOT)), "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "decision": decision, "criteria": out}
    (MODELS / "gates").mkdir(exist_ok=True)
    (MODELS / "gates" / f"{version}.json").write_text(json.dumps(report, indent=2, default=str))
    for k, v in out.items():
        print(f"{k:<26} {v['status']:<8} {json.dumps(v['evidence'], default=str)[:150]}")
    print("DECISION:", decision)
    return report


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ps_native_v5")
