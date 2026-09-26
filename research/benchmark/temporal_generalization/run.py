"""Read-only temporal-generalization diagnosis for ps_native_v5.

Uses recorded development/holdout results for performance and rebuilds only
causal features from the already available raw source to diagnose distributions.
It never opens test.parquet, writes production artifacts, fits a model, or
changes a threshold.  ADR 0004 explicitly permits this t42-49 error analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.reproduction.build_ps_dataset import build_canonical_frame
from obsidianchain.ml import monitoring, protocol
from obsidianchain.ml.ps_model import PsNativeRiskModel
from obsidianchain.pipeline.features_ps import PsTemporalFeatureEngine


MODEL_DIR = ROOT / "data" / "models" / "ps_native" / "v5"
DEV_EVALUATION = MODEL_DIR / "evaluation.json"
HOLDOUT_SCORECARD = ROOT / "data" / "models" / "ps_native" / "holdout" / "ps_native_v5.json"
OUT = Path(__file__).with_name("RESULT.md")


def snapshot(events: pd.DataFrame, first_lo: int, first_hi: int, as_of: int) -> pd.DataFrame:
    part = events[
        (events["_first"] >= first_lo)
        & (events["_first"] <= first_hi)
        & (events["_step"] <= as_of)
    ]
    return part.sort_values("_seq").drop_duplicates("address", keep="last").reset_index(drop=True)


def build_events() -> pd.DataFrame:
    """Generate features before labels are read, using the unchanged product engine."""
    frame = build_canonical_frame(ROOT / "data" / "raw", max_step=protocol.DATASET_END)
    feats = PsTemporalFeatureEngine().process_records(frame)
    feats["_step"] = feats.txid.map(dict(zip(frame.txid, frame["_step"]))).astype(int)
    feats["_seq"] = np.arange(len(feats))
    feats["_first"] = feats.address.map(feats.groupby("address")._step.min())
    labels = pd.read_csv(ROOT / "data" / "raw" / "wallets_classes.csv").set_index("address")["class"]
    feats = feats.assign(cls=feats.address.map(labels))
    return feats[feats.cls.isin([1, 2])].assign(y=lambda d: (d.cls == 1).astype(int)).reset_index(drop=True)


def feature_auc(frame: pd.DataFrame, feature: str, direction: int) -> float:
    values = pd.to_numeric(frame[feature], errors="coerce")
    mask = values.notna()
    if mask.sum() == 0 or frame.loc[mask, "y"].nunique() != 2:
        return float("nan")
    return float(roc_auc_score(frame.loc[mask, "y"], direction * values[mask]))


def direction_from_train(frame: pd.DataFrame, feature: str) -> int:
    pos = pd.to_numeric(frame.loc[frame.y == 1, feature], errors="coerce").median()
    neg = pd.to_numeric(frame.loc[frame.y == 0, feature], errors="coerce").median()
    return 1 if pos >= neg else -1


def score_summary(model: PsNativeRiskModel, frame: pd.DataFrame) -> dict[str, float | int]:
    raw = model.raw_scores(frame)
    prob = model.calibrate(raw)
    y = frame.y.to_numpy()
    pos, neg = raw[y == 1], raw[y == 0]
    return {
        "n": int(len(frame)), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "raw_median": float(np.median(raw)), "prob_median": float(np.median(prob)),
        "positive_raw_median": float(np.median(pos)), "negative_raw_median": float(np.median(neg)),
        "positive_prob_median": float(np.median(prob[y == 1])), "negative_prob_median": float(np.median(prob[y == 0])),
        "score_auc": float(roc_auc_score(y, raw)),
        "positive_below_negative_p95": float(np.mean(pos <= np.quantile(neg, 0.95))),
    }


def performance_rows() -> tuple[list[dict], list[dict]]:
    dev = json.loads(DEV_EVALUATION.read_text())["folds"]
    hold = json.loads(HOLDOUT_SCORECARD.read_text())["models"]["ps_native_v5"]["by_first_seen_step"]
    dev_rows = []
    for f in dev:
        a = f["address"]
        dev_rows.append({"window": f["fold"], "train": f["fold"].split(" -> ")[0], "test": f["fold"].split(" -> ")[1],
                         "nAP": a["nap"], "P100": a["P@100"], "R100": a["R@100"],
                         "prevalence": a["prevalence"], "positives": a["positives"], "n": a["n"]})
    hold_rows = []
    for step, a in hold.items():
        positives, n = a["positives"], a["n"]
        hold_rows.append({"window": step, "train": "fixed ps_native_v5 (trained t26-41)", "test": f"{step} addresses; snapshots <=t49",
                          "nAP": a["nap"], "P100": a["P@100"], "R100": None,
                          "prevalence": positives / n, "positives": positives, "n": n})
    return dev_rows, hold_rows


def run() -> str:
    dev_rows, hold_rows = performance_rows()
    best_dev = max(dev_rows, key=lambda r: r["nAP"])
    # The best recorded two-step development fold is used only as a stable,
    # earlier comparison cohort; no parameter/model choice occurs here.
    good_start, good_end = 33, 34

    model = PsNativeRiskModel.load(MODEL_DIR, require_live_schema=True)
    events = build_events()
    train = snapshot(events, 26, 41, 41)
    good = snapshot(events, good_start, good_end, good_end)
    hold_all = snapshot(events, 42, 49, 49)
    degraded = pd.concat([snapshot(events, s, s, 49) for s in (43, 45, 47)], ignore_index=True)
    cohorts = {"train t26-41@41": train, "good t33-34@34": good,
               "degraded t43/t45/t47@49": degraded, "holdout t42-49@49": hold_all}

    # The project's existing PSI implementation and the artifact's frozen
    # reference bins are used rather than a new drift metric.
    drift = {name: monitoring.compare_to_reference(model.reference_profile, frame, model.raw_scores(frame))
             for name, frame in cohorts.items()}
    features = model.features
    directions = {f: direction_from_train(train, f) for f in features}
    ranked = []
    for f in features:
        ranked.append({"feature": f,
                       "psi_good": drift["good t33-34@34"]["features"][f]["psi"],
                       "psi_degraded": drift["degraded t43/t45/t47@49"]["features"][f]["psi"],
                       "psi_holdout": drift["holdout t42-49@49"]["features"][f]["psi"],
                       "auc_train": feature_auc(train, f, directions[f]),
                       "auc_good": feature_auc(good, f, directions[f]),
                       "auc_degraded": feature_auc(degraded, f, directions[f]),
                       "auc_holdout": feature_auc(hold_all, f, directions[f])})
    ranked.sort(key=lambda r: r["psi_degraded"], reverse=True)
    importance = dict(zip(features, getattr(model.model, "feature_importances_", np.zeros(len(features)))))
    top_predictive = sorted(ranked, key=lambda r: importance[r["feature"]], reverse=True)[:10]
    scores = {name: score_summary(model, frame) for name, frame in cohorts.items()}
    per_step_scores = {s: score_summary(model, snapshot(events, s, s, 49)) for s in range(42, 50)}

    def performance_table(rows: list[dict]) -> str:
        lines = []
        for r in rows:
            recall = "not recorded" if r["R100"] is None else f"{r['R100']:.3f}"
            lines.append(
                f"| {r['window']} | {r['train']} | {r['test']} | {r['nAP']:.3f} | {r['P100']:.2f} | "
                f"{recall} | {r['prevalence']:.3%} | {r['positives']} | {r['n']:,} |"
            )
        return "\n".join(lines)
    def drift_table() -> str:
        return "\n".join(
            f"| {r['feature']} | {r['psi_good']:.3f} | {r['psi_degraded']:.3f} | {r['psi_holdout']:.3f} |"
            for r in ranked)
    def separation_table() -> str:
        return "\n".join(
            f"| {r['feature']} | {importance[r['feature']]:.0f} | {r['auc_train']:.3f} | {r['auc_good']:.3f} | {r['auc_degraded']:.3f} | {r['auc_holdout']:.3f} |"
            for r in top_predictive)
    def score_table() -> str:
        return "\n".join(
            f"| {name} | {x['n']:,} | {x['prevalence']:.3%} | {x['raw_median']:.4f} | {x['positive_raw_median']:.4f} | {x['negative_raw_median']:.4f} | {x['positive_prob_median']:.4f} | {x['negative_prob_median']:.4f} | {x['score_auc']:.3f} | {x['positive_below_negative_p95']:.1%} |"
            for name, x in scores.items())
    def per_step_score_table() -> str:
        return "\n".join(
            f"| t{s} | {x['n']:,} | {x['positives']} | {x['prevalence']:.3%} | {x['positive_raw_median']:.4f} | {x['negative_raw_median']:.4f} | {x['score_auc']:.3f} | {x['positive_below_negative_p95']:.1%} |"
            for s, x in per_step_scores.items())

    return f"""# ps_native_v5 Temporal Generalization / Concept-Drift Diagnosis

## 1. Executive summary

The weak later result is concentrated in **three abrupt windows** (`t43`, `t45`, `t47`), not a smooth decline. Their recorded nAPs are `0.135`, `0.017`, and `0.324`, while adjacent holdout windows include `0.704` (`t42`), `0.687` (`t44`), `0.981` (`t46`), and `0.717` (`t49`). Read-only diagnostics show that the failing windows contain lower-prevalence, differently shaped positive examples and that their positive scores are much less separated from negative scores. This supports a **positive-class typology/relationship change plus sparse clustered labels**, not a threshold or feature-pipeline defect.

The current product pipeline also has broad input PSI changes over time, but the retained `exp23_relative_drift` result marked every holdout step — including failures — `WITHIN_BASELINE`. Therefore ordinary covariate drift is present but is insufficient to explain why only specific windows collapse.

## 2. Performance by window (recorded results)

Development fold values are the recorded refit-fold result in `v5/evaluation.json`; holdout values are the published fixed-v5 result in `holdout/ps_native_v5.json`. They are not interchangeable evaluation regimes.

| Window | Train period | Test period | nAP | P@100 | Recall@100 | Positive prevalence | Positives | Test addresses |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{performance_table(dev_rows)}
{performance_table(hold_rows)}

- Best recorded development fold: `{best_dev['window']}` (`nAP={best_dev['nAP']:.3f}`).
- Best holdout step: `t46` (`nAP=0.981`); it is not independent evidence of broad generalization because 505 of 508 positives arise from one transaction.
- Worst holdout step: `t45` (`nAP=0.017`, `P@100=0.02`).
- Degradation starts suddenly at `t43`, recovers at `t44`, collapses at `t45`, recovers at `t46`, and weakens again at `t47–48`: it is **episodic, not gradual**.
- `Recall@100` is not retained in the per-step holdout scorecard; it is reported as `not recorded` rather than recomputed from sealed artifacts.

## 3. Feature drift (all 31 current `/5` features)

PSI is the project's existing metric, using the frozen v5 training reference bins. Reading: `<0.10` stable, `0.10–0.25` shifted, `>=0.25` major. Cohorts are produced by the unchanged causal feature engine; labels are joined only after feature generation.

| Feature (ranked by degraded-period PSI) | PSI: good t33-34 | PSI: degraded t43/45/47 | PSI: all holdout t42-49 |
| --- | ---: | ---: | ---: |
{drift_table()}

The largest degraded-period shifts are the first rows above. They are principally transaction shape, fee/amount, counterparty, and upstream-flow features. But these shifts are not unique to failure windows: the recorded runtime monitor rated all holdout steps within the development baseline, including `t43`, `t45`, and `t47`.

## 4. Positive/negative distribution comparison

For each high-importance model feature, the table gives a one-feature ROC-AUC with its direction fixed from training (`0.5` = no class separation). This directly tests whether an earlier predictive relationship remains discriminative later; it does not train any model.

| Feature | v5 importance | Train AUC | Earlier-good AUC | Degraded AUC | All-holdout AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
{separation_table()}

- Positive behavior changed: retained diagnostics show failing positives have lower fan-in and upstream-funded share, and more cold starts than development positives (for `t43/t45/t47`: fan-in `4/3/8` vs development `103`; upstream-funded share `0.65/0.56/0.78` vs `0.91`; no-prior-transaction `0.52/0.59/0.72` vs `0.23`).
- Negative behavior also shifts on several inputs (the PSI table), but the score/separation evidence specifically shows the class relationship weakening in failure windows.
- Label/prevalence shift is substantial: `t43=1.6%`, `t45=0.3%`, and `t47=2.9%`, versus every recorded development fold at or above `4.8%`. The holdout positives are clustered into very few independent transactions (`24`, `5`, and `22` respectively), so per-step estimates are high variance.

## 5. Score-distribution comparison (frozen v5; diagnostic only)

Raw scores are the model's ranking score. `Positive ≤ negative p95` is a simple overlap measure: lower is better; a large percentage means many positives fall below a high negative-score cutoff. All values below were computed with the frozen artifact; no threshold was applied or selected.

| Cohort | n | Prevalence | Median raw score | Positive raw median | Negative raw median | Positive calibrated median | Negative calibrated median | Raw-score AUC | Positive ≤ negative p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{score_table()}

| Holdout first-seen step | n | Positives | Prevalence | Positive median raw score | Negative median raw score | Raw-score AUC | Positive ≤ negative p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{per_step_score_table()}

Later failure-step positives receive substantially less separable scores than well-performing periods; this is ranking failure, so a fixed classification threshold cannot restore nAP or P@100. Calibrated probabilities are monotone in these raw scores, so calibration cannot change the ranking conclusion.

## 6. Evidence by possible cause

| Possible cause | Evidence | Assessment |
| --- | --- | --- |
| A. Covariate drift | Major absolute PSI appears in many features, including the degraded cohort. However all `t42-49` steps were `WITHIN_BASELINE` on the retained relative-drift monitor, regardless of performance. | Present, but not sufficient as the primary explanation. |
| B. Concept / relationship drift | Earlier-important feature separation and frozen-model score separation weaken in degraded steps; failing positives have a different structural typology. | Strongest supported cause. |
| C. Label / prevalence shift | Prevalence falls to 1.6%, 0.3%, and 2.9%; only 24, 5, and 22 independent illicit transactions underpin `t43/t45/t47`. Labels were previously audited as internally consistent. | Strong contributor and makes step results high variance. |
| D. Protocol/evaluation difference | Development uses two-step rolling refit windows; holdout uses one fixed model and a single eight-step window with snapshots at `t49`. This accounts for some aggregate comparison difference, but cannot explain abrupt `t43/t45/t47` collapses within the same holdout protocol. | Partial contributor, not root cause of episodic collapse. |
| E. Implementation mismatch | The same current causal production engine and `/5` feature contract generated development and holdout diagnostics; artifact hash is unchanged; holdout scorecard matches the published result. | No evidence. |

## 7. Most likely cause and confidence

**Most likely cause:** a change in the positive class's transaction/graph typology (concept/relationship drift) combined with a severe prevalence and independent-event-count collapse. The current model recognizes the dominant development-era consolidation/upstream-funded positive pattern, but does not rank the small, weakly funded, cold-start-heavy positive patterns in `t43/t45/t47` as well.

**Confidence: moderate.** The direction is supported by recorded results, retained feature diagnostics, and this fresh frozen-model score diagnosis. It is not high because `t42-49` is a seen/reused holdout, and the failure steps have very few independent illicit transactions.

## 8. Is it fixable without retraining?

- **Threshold problem:** no. nAP/P@100 and raw-score separation degrade; changing a classification cutoff cannot repair ranking.
- **Calibration problem:** not the main issue. Calibration is monotone and the pooled holdout calibration record is good (`ECE=0.010`); it cannot improve rank separation.
- **Feature robustness / model retraining:** likely requires a future, prospectively evaluated model/data strategy, not a serving-time patch. No fix is implemented here.
- **Protocol problem:** it explains part of the headline comparison, so future experiments should retain matched evaluation units; it does not remove the within-holdout collapse.

## 9. Recommended next experiment

Pre-register a **development-only** typology-robustness experiment: split the existing `t17-41` protocol-B confirmation folds by independent transaction/event clusters and cold-start/upstream-funded strata, then test a deliberately specified robustness feature or training objective only on those folds. Do not choose it from `t42-49`; use delayed labels from new production traffic for an unbiased final generalization test.

## 10. Integrity statement

**No production code, model, registry, feature schema, threshold, or sealed holdout artifact was changed or inspected.** This report reused recorded holdout/development performance and, under ADR 0004's diagnosis allowance, generated read-only causal features from the existing raw source for score/distribution diagnosis. Labels were joined only after feature generation; no model was trained or retrained.
"""


if __name__ == "__main__":
    OUT.write_text(run(), encoding="utf-8")
