"""How many folds does this problem need to support a comparison at all?

n=5 detects only enormous effects. This measures the trade: narrower folds
give more of them, but each window must still hold enough positives for nAP
and precision@k to mean anything.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from obsidianchain.ml import protocol

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data/models/ps_native/datasets"
BASE, STEP = 1400000000, 1209600
dev = pd.concat([pd.read_parquet(DS/"train.parquet"),
                 pd.read_parquet(DS/"validation.parquet")], ignore_index=True)
dev["first_t"] = ((dev["timestamp"] - BASE) // STEP + 1).astype(int)

SD = 0.1850  # observed mean paired sd of fold differences
rng = np.random.default_rng(7)

def power(n, effect, alpha=0.05, trials=20000):
    d = rng.normal(effect, SD, size=(trials, n))
    return float((stats.ttest_1samp(d, 0.0, axis=1).pvalue < alpha).mean())

def mde(n, alpha=0.05, target=0.80):
    lo, hi = 0.0, 1.5
    for _ in range(40):
        mid = (lo + hi) / 2
        if power(n, mid, alpha) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2

print("=== fold designs available in the development period (t1-41) ===")
print(f"{'width':>6} {'min_train':>10} {'folds':>6} {'min rows':>9} {'min pos':>8} "
      f"{'MDE@80%':>8} {'power@0.10':>11}")
for width in (1, 2, 3, 4, 6):
    for min_train in (16, 20, 24):
        try:
            folds = protocol.rolling_origin_folds(width=width, min_train=min_train)
        except Exception:
            continue
        if len(folds) < 2: continue
        rows, pos = [], []
        for f in folds:
            w = dev[(dev.first_t >= f.eval_start) & (dev.first_t <= f.eval_end)]
            rows.append(len(w)); pos.append(int(w.y.sum()))
        n = len(folds)
        print(f"{width:>6} {min_train:>10} {n:>6} {min(rows):>9,} {min(pos):>8,} "
              f"{mde(n):>8.3f} {power(n, 0.10):>11.3f}")

print("\n=== what n would be needed ===")
for target_effect in (0.05, 0.10, 0.15):
    n = next((k for k in range(3, 400) if power(k, target_effect) >= 0.80), None)
    print(f"  to detect {target_effect:.2f} nAP at 80% power: n = {n} folds")

print("\n=== exact sign-flip permutation: smallest reachable two-sided p ===")
for n in (5, 7, 10, 14, 21):
    print(f"  n={n:>3}  min p = 2/2^{n} = {2/2**n:.5f}"
          f"  {'usable at .05' if 2/2**n < 0.05 else 'CANNOT reject at .05'}")
