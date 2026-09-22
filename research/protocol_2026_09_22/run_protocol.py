"""Run the canonical protocol over the DEVELOPMENT period only.

Reads train.parquet and validation.parquet. It does not open test.parquet,
and the protocol's own fold construction would refuse a fold that reached it.
"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from obsidianchain.ml import diagnostics, protocol

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data/models/ps_native/datasets"
BASE, STEP = 1400000000, 1209600

dev = pd.concat([pd.read_parquet(DS/"train.parquet"),
                 pd.read_parquet(DS/"validation.parquet")], ignore_index=True)
dev["first_t"] = ((dev["timestamp"] - BASE) // STEP + 1).astype(int)
from obsidianchain.pipeline.features_ps import CORE_PS_FEATURE_COLUMNS
declared = list(CORE_PS_FEATURE_COLUMNS)
healthy = diagnostics.healthy_features(dev, declared)
print(f"development rows {len(dev):,}  timesteps {dev.first_t.min()}-{dev.first_t.max()}")
print(f"features declared {len(declared)} -> healthy {len(healthy)}")
print(f"folds: {[f.label for f in protocol.rolling_origin_folds()]}\n")

def make(kind, feats):
    def fit_predict(train, ev, features, seed):
        if kind == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(n_estimators=100, max_depth=12,
                                       min_samples_leaf=20, random_state=seed, n_jobs=-1)
        elif kind == "LightGBM":
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(n_estimators=100, learning_rate=0.05,
                               random_state=seed, n_jobs=-1, verbose=-1)
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000, random_state=seed))
        X = train[feats].to_numpy("float32"); X = np.nan_to_num(X)
        Xe = np.nan_to_num(ev[feats].to_numpy("float32"))
        m.fit(X, train["y"].to_numpy("int8"))
        return m.predict_proba(Xe)[:, 1]
    return fit_predict

results = {}
for kind in ("RandomForest", "LightGBM", "LogisticRegression"):
    r = protocol.evaluate_candidate(dev, make(kind, healthy), name=kind,
                                    seed=20260919, features=healthy)
    results[kind] = r
    s = r.summary()
    print(f"{kind:<20} nAP mean {s['nap_mean']:.4f}  sd {s['nap_sd']:.4f}  "
          f"min {s['nap_min']:.4f}  max {s['nap_max']:.4f}  ({s['folds']} folds)")

print()
names = list(results)
family = protocol.compare_family([results[n] for n in names])
print(f"{'comparison':<42} {'mean diff':>10} {'95% CI':>18} {'p':>7} {'p_holm':>8}  verdict")
for c in family["comparisons"]:
    lo, hi = c["ci95"]
    print(f"{c['label']:<42} {c['mean_difference']:>+10.4f} "
          f"[{lo:>+7.3f},{hi:>+7.3f}] {c['p_value']:>7.3f} {c['p_holm']:>8.3f}  "
          f"{c['verdict']}{' -> ' + c['favours'] if c['favours'] else ''}")
print()
for c in family["comparisons"]:
    if c.get("underpowered_for"):
        print(f"  ! {c['label']}: {c['underpowered_for']}")
    if not c["permutation_usable"]:
        print(f"  ! {c['label']}: permutation check unusable at n={c['n_folds']}")

out = ROOT/"research/protocol_2026_09_22/protocol_results.json"
out.write_text(json.dumps({k: v.summary() for k, v in results.items()}, indent=1))
print(f"\nwrote {out}")
