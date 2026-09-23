import os
from pathlib import Path

# Ensure OBSIDIANCHAIN_DATA defaults to local repo data directory during test runs
if not os.environ.get("OBSIDIANCHAIN_DATA"):
    repo_data = Path(__file__).resolve().parents[1] / "data"
    if repo_data.is_dir():
        os.environ["OBSIDIANCHAIN_DATA"] = str(repo_data)


# ---- a self-contained model registry for pipeline tests -------------------
import json as _json  # noqa: E402

import pytest as _pytest  # noqa: E402


def write_tiny_model(model_dir: Path, version: str, seed: int, schema: str | None = None,
                     stacker: bool = False) -> None:
    """A small LightGBM on synthetic features, in the production artifact
    layout, so pipeline tests never depend on the real frozen models."""
    import joblib
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMClassifier

    from obsidianchain.ml import registry
    from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS, PS_FEATURE_SCHEMA_VERSION
    schema = schema or PS_FEATURE_SCHEMA_VERSION
    features = list(CORE_PS_FEATURE_COLUMNS)
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.uniform(0, 1, size=(400, len(features))), columns=features)
    y = (x[features[0]] + rng.normal(0, 0.3, 400) > 0.6).astype(int)
    model = LGBMClassifier(n_estimators=20, verbose=-1, random_state=seed).fit(x, y)
    model_dir.mkdir(parents=True)
    joblib.dump({"model_name": "LightGBM", "model": model, "features": features,
                 "calibrator": {"coef": 1.0, "intercept": 0.0, "reference_prevalence": 0.05},
                 "feature_schema_version": schema, "reference_profile": None}, model_dir / "model.joblib")
    (model_dir / "manifest.json").write_text(_json.dumps({
        "model_version": version, "feature_schema_version": schema, "features": features,
        "model_sha256": registry.file_sha256(model_dir / "model.joblib"), "holdout_evaluated": False}))
    if stacker:
        (model_dir / "stacker.json").write_text(_json.dumps({"coef": [1.0, 1.0], "intercept": 0.0}))


@_pytest.fixture()
def tiny_registry(tmp_path) -> Path:
    """Registry with champion ``test_a`` (with a stacker) and fallback ``test_b``."""
    from obsidianchain.ml import registry
    from obsidianchain.pipeline.features_ps import PS_FEATURE_SCHEMA_VERSION
    root = tmp_path / "models"
    write_tiny_model(root / "a", "test_a", 1, stacker=True)
    write_tiny_model(root / "b", "test_b", 2)
    reg = registry.Registry.open(root)
    reg.register("test_a", "a", feature_schema_version=PS_FEATURE_SCHEMA_VERSION)
    reg.register("test_b", "b", feature_schema_version=PS_FEATURE_SCHEMA_VERSION)
    reg.assign("champion", "test_a", "test fixture")
    reg.assign("fallback", "test_b", "test fixture")
    reg.save()
    return root


# ---- tests that need locally generated artifacts -----------------------------
#: Test file -> the gitignored artifact it verifies. These tests fail loudly
#: when the artifact is absent (by design: absence is not success). CI, which
#: has no local data, deselects them explicitly with
#: ``-m "not requires_local_artifacts"`` and prints what it deselected;
#: ``make test`` on a machine with the data runs everything.
REQUIRES_LOCAL_ARTIFACTS = {
    "test_artifact_atomicity.py": "data/processed/evidence_funnel.parquet",
    "test_funnel_golden.py": "data/processed/evidence_funnel.parquet",
    "test_run_fingerprint.py": "data/processed/evidence_funnel.parquet",
    "test_phase6_leakage.py": "data/processed/phase6_dataset_d5.parquet",
    "test_phase7_alerts.py": "data/processed/alerts.parquet",
    "test_ps_dataset_and_eval.py": "data/models/ps_native/datasets/train.parquet",
    "test_ps_model_current.py": "data/models/ps_native/datasets/validation.parquet",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_local_artifacts(path): needs a gitignored, locally built artifact")


def pytest_collection_modifyitems(config, items):
    for item in items:
        needed = REQUIRES_LOCAL_ARTIFACTS.get(Path(str(item.fspath)).name)
        if needed:
            item.add_marker(_pytest.mark.requires_local_artifacts(needed))
