"""Is the '|mean diff| > 1 sd' decision rule defensible? Measure it.

The rule in ml/protocol.py was chosen by reasoning, not by measurement. This
script measures its operating characteristics by simulation, using the REAL
fold-difference spread observed on the development folds.

scipy only: statsmodels is not in the vendored wheel set and adding it would
require re-vendoring, which touches the offline build guarantee.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
results = json.loads((ROOT / "research/protocol_2026_09_22/protocol_results.json").read_text())

naps = {k: np.array([f["nap"] for f in v["per_fold"]]) for k, v in results.items()}
names = list(naps)
n_folds = len(next(iter(naps.values())))
print(f"folds = {n_folds}\n")

# ---- 1. observed paired differences -------------------------------------
print("=== observed paired differences (nAP) ===")
pairs = {}
for i, a in enumerate(names):
    for b in names[i+1:]:
        d = naps[a] - naps[b]
        pairs[(a, b)] = d
        print(f"{a:<20} - {b:<20} mean {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  "
              f"per-fold {np.round(d, 3)}")

# ---- 2. is a paired t-test even licensed at n=5? ------------------------
print("\n=== normality of fold differences (Shapiro-Wilk) ===")
for (a, b), d in pairs.items():
    W, p = stats.shapiro(d)
    print(f"{a[:12]:<13}-{b[:12]:<13} W={W:.3f} p={p:.3f}"
          f"  {'no evidence against normality' if p > 0.05 else 'NON-NORMAL'}")
print("note: at n=5 Shapiro-Wilk has almost no power; a pass is weak evidence.")

# ---- 3. three tests on the same data ------------------------------------
print("\n=== three candidate tests, same differences ===")
print(f"{'comparison':<30} {'t-test p':>10} {'wilcoxon p':>11} {'perm p':>9}")
for (a, b), d in pairs.items():
    t_p = stats.ttest_1samp(d, 0.0).pvalue
    try:
        w_p = stats.wilcoxon(d).pvalue
    except ValueError:
        w_p = float("nan")
    # Exact sign-flip permutation: the only one of the three that needs no
    # distributional assumption at all. 2^5 = 32 sign assignments.
    signs = np.array(np.meshgrid(*[[-1, 1]] * len(d))).T.reshape(-1, len(d))
    null = (signs * d).mean(axis=1)
    perm_p = float((np.abs(null) >= abs(d.mean()) - 1e-12).mean())
    print(f"{a[:14]+' vs '+b[:12]:<30} {t_p:>10.4f} {w_p:>11.4f} {perm_p:>9.4f}")

# ---- 4. multiple comparisons --------------------------------------------
print("\n=== Holm-Bonferroni over the 3 pairwise comparisons ===")
raw = {f"{a} vs {b}": stats.ttest_1samp(d, 0.0).pvalue for (a, b), d in pairs.items()}
order = sorted(raw.items(), key=lambda kv: kv[1])
m = len(order)
running = 0.0
for i, (label, p) in enumerate(order):
    adj = min(1.0, max(running, (m - i) * p))
    running = adj
    print(f"  {label:<44} raw {p:.4f} -> Holm {adj:.4f}"
          f"  {'reject' if adj < 0.05 else 'retain H0'}")

# ---- 5. operating characteristics of the '1 sd' rule --------------------
print("\n=== simulated operating characteristics (10,000 trials) ===")
rng = np.random.default_rng(20260922)
sd_obs = float(np.mean([d.std(ddof=1) for d in pairs.values()]))
print(f"using observed mean paired sd = {sd_obs:.4f}, n = {n_folds}")
print(f"{'true effect':>12} {'1-sd rule':>11} {'t-test .05':>11} {'perm .05':>10}")
for true_effect in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
    d = rng.normal(true_effect, sd_obs, size=(10000, n_folds))
    means, sds = d.mean(axis=1), d.std(axis=1, ddof=1)
    rule = (np.abs(means) > sds).mean()
    t_rej = (stats.ttest_1samp(d, 0.0, axis=1).pvalue < 0.05).mean()
    signs = np.array(np.meshgrid(*[[-1, 1]] * n_folds)).T.reshape(-1, n_folds)
    sub = d[:2000]
    null = np.einsum('ij,kj->ik', sub, signs) / n_folds
    perm = (np.abs(null) >= np.abs(sub.mean(axis=1))[:, None] - 1e-12).mean(axis=1)
    print(f"{true_effect:>12.2f} {rule:>11.3f} {t_rej:>11.3f} {(perm < 0.05).mean():>10.3f}")

print("\nRow 0.00 is the FALSE POSITIVE rate. Rows below are power.")
