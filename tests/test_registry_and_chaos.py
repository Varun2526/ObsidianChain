"""Model registry semantics and failure behaviour (Phases 10-13, 17, 24).

Every test builds its own registry around a small model trained here on
synthetic features, so none depends on the real artifacts and all run in CI.

Failure scenarios covered: tampered artifact, missing artifact, schema
mismatch, no model at all, champion broken with a healthy fallback,
malformed and conflicting capture rows, duplicate observations, a capture
refused whole, a feature-contract violation, batch invariance of the score,
shadow scoring that must not touch alerts, and rollback.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from obsidianchain import geoip
from obsidianchain.ml import registry
from obsidianchain.ml.ps_model import PsNativeRiskModel
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS, PS_FEATURE_SCHEMA_VERSION
from obsidianchain.pipeline.orchestrator import CaptureContractError, run_pipeline

CAPTURE = Path("tests/data/synthetic_acceptance_capture.json")
FEATURES = list(CORE_PS_FEATURE_COLUMNS)


def _write_model(model_dir: Path, version: str, seed: int, schema: str = PS_FEATURE_SCHEMA_VERSION) -> None:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.uniform(0, 1, size=(400, len(FEATURES))), columns=FEATURES)
    y = (x[FEATURES[0]] + rng.normal(0, 0.3, 400) > 0.6).astype(int)
    model = LGBMClassifier(n_estimators=20, verbose=-1, random_state=seed).fit(x, y)
    model_dir.mkdir(parents=True)
    joblib.dump({"model_name": "LightGBM", "model": model, "features": FEATURES,
                 "calibrator": {"coef": 1.0, "intercept": 0.0, "reference_prevalence": 0.05},
                 "feature_schema_version": schema, "reference_profile": None}, model_dir / "model.joblib")
    sha = registry.file_sha256(model_dir / "model.joblib")
    (model_dir / "manifest.json").write_text(json.dumps({
        "model_version": version, "feature_schema_version": schema, "features": FEATURES,
        "model_sha256": sha, "holdout_evaluated": False}))


@pytest.fixture()
def reg_root(tmp_path) -> Path:
    root = tmp_path / "models"
    _write_model(root / "a", "test_a", 1)
    _write_model(root / "b", "test_b", 2)
    reg = registry.Registry.open(root)
    reg.register("test_a", "a", feature_schema_version=PS_FEATURE_SCHEMA_VERSION)
    reg.register("test_b", "b", feature_schema_version=PS_FEATURE_SCHEMA_VERSION)
    reg.assign("champion", "test_a", "test fixture")
    reg.assign("fallback", "test_b", "test fixture")
    reg.save()
    return root


def _run(tmp_path, root, capture=CAPTURE, **kw):
    return run_pipeline(capture, runs_dir=tmp_path / "runs", registry_root=root,
                        geoip_provider=geoip.TestFixtureProvider(), seed_addresses=[], **kw)


# ---- registry semantics ------------------------------------------------------

def test_artifacts_are_immutable(reg_root) -> None:
    reg = registry.Registry.open(reg_root)
    (reg_root / "a" / "extra.json").write_text("{}")
    with pytest.raises(registry.RegistryError, match="immutable"):
        reg.register("test_a", "a", feature_schema_version=PS_FEATURE_SCHEMA_VERSION)


def test_a_role_change_needs_a_reason(reg_root) -> None:
    with pytest.raises(registry.RegistryError, match="reason"):
        registry.Registry.open(reg_root).assign("candidate", "test_b", "  ")


def test_rollback_swaps_and_is_logged(reg_root) -> None:
    reg = registry.Registry.open(reg_root)
    assert reg.rollback("drill", PS_FEATURE_SCHEMA_VERSION) == "test_b"
    assert reg.role("champion") == "test_b" and reg.role("fallback") == "test_a"
    assert reg.data["history"][-1]["event"] == "ROLLBACK"


def test_rollback_across_schemas_is_refused(reg_root) -> None:
    reg = registry.Registry.open(reg_root)
    reg.data["models"]["test_b"]["feature_schema_version"] = "ps_native_features/0"
    with pytest.raises(registry.RegistryError, match="roll back the code"):
        reg.rollback("drill", PS_FEATURE_SCHEMA_VERSION)


# ---- serving failures ----------------------------------------------------------

def test_tampered_champion_is_refused_and_fallback_serves(tmp_path, reg_root) -> None:
    with open(reg_root / "a" / "model.joblib", "ab") as fh:
        fh.write(b"tamper")
    out = _run(tmp_path, reg_root)
    s10 = next(s for s in out.stages if s.stage_number == 10)
    assert s10.summary["model_load"]["role"] == "fallback"
    assert "does not match registered" in str(s10.summary["model_load"]["failures"])
    codes = {a["code"] for a in out.manifest["provenance"]["monitoring_alerts"]}
    assert "SERVED_BY_FALLBACK" in codes


def test_no_loadable_model_degrades_honestly(tmp_path, reg_root) -> None:
    (reg_root / "a" / "model.joblib").unlink()
    shutil.rmtree(reg_root / "b")
    out = _run(tmp_path, reg_root)
    assert out.manifest["provenance"]["ml_status"] == "MODEL_UNAVAILABLE_FOR_SCHEMA"
    assert out.is_success
    codes = {a["code"] for a in out.manifest["provenance"]["monitoring_alerts"]}
    assert "NO_MODEL_SERVED" in codes
    for alert in out.alert_result.alerts:
        model = next(e for e in alert.evidence if e.category == "MODEL_SIGNAL")
        assert model.status == "UNAVAILABLE"


def test_schema_mismatch_is_never_loaded(tmp_path) -> None:
    root = tmp_path / "models"
    _write_model(root / "old", "old", 3, schema="ps_native_features/4")
    reg = registry.Registry.open(root)
    reg.register("old", "old", feature_schema_version="ps_native_features/4")
    reg.data["roles"]["champion"] = "old"
    reg.save()
    with pytest.raises(registry.RegistryError, match="declares"):
        PsNativeRiskModel.from_registry("champion", root)


def test_direct_load_with_live_schema_refuses_a_legacy_artifact() -> None:
    v1 = Path("data/models/ps_native/v1")
    if not v1.is_dir():
        pytest.skip("v1 artifact absent")
    with pytest.raises(ValueError, match="feature schema"):
        PsNativeRiskModel.load(v1, require_live_schema=True)


# ---- determinism and batch invariance -------------------------------------

def test_a_score_does_not_depend_on_the_batch(reg_root) -> None:
    model = PsNativeRiskModel.from_registry("champion", reg_root)
    rng = np.random.default_rng(5)
    batch = pd.DataFrame(rng.uniform(0, 1, size=(300, len(FEATURES))), columns=FEATURES)
    together = model.raw_scores(batch)
    alone = np.array([model.raw_scores(batch.iloc[[i]])[0] for i in range(0, 300, 37)])
    np.testing.assert_array_equal(together[::37], alone)
    np.testing.assert_array_equal(model.raw_scores(batch.iloc[::-1])[::-1], together)


def test_runs_are_reproducible(tmp_path, reg_root) -> None:
    a = pd.read_parquet(_run(tmp_path / "1", reg_root).run_dir / "predictions.parquet")
    b = pd.read_parquet(_run(tmp_path / "2", reg_root).run_dir / "predictions.parquet")
    cols = ["address", "raw_score", "calibrated_probability", "rank", "feature_row_sha256", "model_sha256"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_every_prediction_is_traceable(tmp_path, reg_root) -> None:
    out = _run(tmp_path, reg_root)
    preds = pd.read_parquet(out.run_dir / "predictions.parquet")
    for col in ("run_id", "address", "model_version", "model_sha256", "feature_schema_version",
                "input_sha256", "feature_row_sha256", "raw_score", "rank"):
        assert preds[col].notna().all(), col
    assert set(preds.model_version) == {"test_a"}
    assert out.manifest["artifacts"]["predictions.parquet"]


# ---- shadow ---------------------------------------------------------------------

def test_shadow_candidate_never_changes_alerts(tmp_path, reg_root) -> None:
    base = _run(tmp_path / "base", reg_root)
    reg = registry.Registry.open(reg_root)
    reg.assign("candidate", "test_b", "shadow drill")
    reg.save()
    shadowed = _run(tmp_path / "shadow", reg_root)
    report = json.loads((shadowed.run_dir / "shadow.json").read_text())
    assert report["status"] == "SHADOW_ONLY" and report["candidate"] == "test_b"
    assert [a.fused_risk_score for a in base.alert_result.alerts] == \
        [a.fused_risk_score for a in shadowed.alert_result.alerts]


# ---- malformed input ----------------------------------------------------------

def _capture(tmp_path, records) -> Path:
    path = tmp_path / "capture.json"
    path.write_text(json.dumps({"records": records}))
    return path


def _rec(txid, **kw):
    base = {"timestamp": 1600000000, "src_ip": "198.51.100.10", "dst_ip": "198.51.100.1", "src_port": 8333,
            "dst_port": 8333, "txid": txid, "input_addresses": ["1In" + txid], "output_addresses": ["1Out" + txid],
            "input_amounts": [1.0], "output_amounts": [0.99], "fee": 0.01, "script_type": "p2pkh",
            "geo_country": "US", "asn": 64500}
    base.update(kw)
    return base


def test_conflicting_and_invalid_transactions_are_quarantined(tmp_path, reg_root) -> None:
    records = [_rec(f"ok{i}") for i in range(8)] + [
        _rec("neg", input_amounts=[-1.0]),
        _rec("fee", fee=5.0),
        _rec("dup"), _rec("dup", output_amounts=[0.5], src_ip="198.51.100.11"),
        _rec("badip", src_ip="999.1.1.1"),
    ]
    out = _run(tmp_path, reg_root, capture=_capture(tmp_path, records))
    contract = next(s for s in out.stages if s.stage_number == 2).summary["capture_contract"]
    kinds = contract["quarantine_kinds"]
    assert kinds["NEGATIVE_OR_NONFINITE_AMOUNT"] == 1 and kinds["FEE_EXCEEDS_INPUTS"] == 1
    assert kinds["CONFLICTING_CHAIN_FACTS"] == 1
    assert contract["rows_dropped"] == {"IP_INVALID": 1}
    scored = set(pd.read_parquet(out.run_dir / "predictions.parquet").address)
    assert not {"1Inneg", "1Infee", "1Indup"} & scored


def test_duplicate_observations_do_not_inflate_history(tmp_path, reg_root) -> None:
    records = [_rec("t1", src_ip=f"198.51.100.{i}") for i in range(1, 6)]
    out = _run(tmp_path, reg_root, capture=_capture(tmp_path, records))
    assert next(s for s in out.stages if s.stage_number == 2).summary["capture_contract"]["quarantined_transactions"] == 0


def test_a_mostly_invalid_capture_is_refused(tmp_path, reg_root) -> None:
    records = [_rec(f"bad{i}", input_amounts=[-1.0]) for i in range(5)] + [_rec("ok")]
    with pytest.raises(CaptureContractError):
        _run(tmp_path, reg_root, capture=_capture(tmp_path, records))


def test_a_feature_contract_violation_blocks_scoring(tmp_path, reg_root, monkeypatch) -> None:
    from obsidianchain.contracts import features as fc
    spec = fc.CATALOG["input_count"]
    monkeypatch.setitem(fc.CATALOG, "input_count", fc.FeatureSpec(**{**spec.__dict__, "high": 0.0}))
    out = _run(tmp_path, reg_root)
    assert out.manifest["provenance"]["ml_status"] == "FEATURE_CONTRACT_VIOLATION"
    codes = {a["code"] for a in out.manifest["provenance"]["monitoring_alerts"]}
    assert "FEATURE_CONTRACT_VIOLATION" in codes


def test_attestation_requires_the_exact_training_code(tiny_registry) -> None:
    reg = registry.Registry.open(tiny_registry)
    manifest_path = reg.model_dir("test_a") / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["lineage"] = {"code_sha256": {"x.py": __import__("hashlib").sha256(b"print(1)").hexdigest()}}
    manifest_path.write_text(json.dumps(m))
    with pytest.raises(registry.RegistryError, match="not the training code"):
        reg.attest("test_a", "c0ffee", lambda c, p: b"print(2)")
    reg.attest("test_a", "c0ffee", lambda c, p: b"print(1)")
    assert reg.entry("test_a")["attested_source_commit"] == "c0ffee"


def test_serving_finds_the_registry_under_obsidianchain_data(tmp_path, monkeypatch, tiny_registry) -> None:
    """The container runs from /app with data at /data: the registry must be
    found through OBSIDIANCHAIN_DATA, not the working directory."""
    data = tmp_path / "mounted_data"
    (data / "models").mkdir(parents=True)
    shutil.copytree(tiny_registry, data / "models" / "ps_native")
    monkeypatch.setenv("OBSIDIANCHAIN_DATA", str(data))
    monkeypatch.chdir(tmp_path)  # nothing under ./data here
    assert registry.default_root() == data / "models" / "ps_native"
    assert PsNativeRiskModel.from_registry("champion").version == "test_a"
