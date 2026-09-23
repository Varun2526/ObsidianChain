# 05 — Model Comparison (exp01_model_family, completed 2026-09-23)

Full results: `results/exp01_model_family.json`. Ledger entry:
`experiments.jsonl` (`experiment_id: exp01_model_family`).

## Setup (locked in `00_research_protocol.md` before running)

6 candidates, no preselected winner: `LogisticRegression_L2`,
`LogisticRegression_L1`, `RandomForest`, `ExtraTrees`,
`HistGradientBoosting`, `LightGBM`. Same dataset (train+validation,
`ps_native_features/2`), same 24 healthy features, same 12 rolling-origin
folds, same seed (20260919), same preprocessing (`float32`, `nan_to_num`)
for every candidate. Compared as **one** `protocol.compare_family` call (15
pairwise comparisons, Holm-corrected together) — not as separate scripts.

## Results

| candidate | nAP mean | nAP sd | min | max | fit+predict wall time (12 folds) |
|---|---|---|---|---|---|
| LogisticRegression_L2 | 0.2915 | 0.2533 | 0.0494 | 0.8139 | 3.2s |
| LogisticRegression_L1 | 0.3020 | 0.2539 | 0.0508 | 0.8224 | **679.7s** |
| RandomForest | 0.5723 | 0.1826 | 0.2955 | 0.8414 | 11.7s |
| ExtraTrees | 0.3103 | 0.2221 | 0.0398 | 0.8083 | 4.1s |
| HistGradientBoosting | 0.5814 | 0.1686 | 0.2459 | 0.8226 | 11.7s |
| LightGBM | 0.5914 | 0.1636 | 0.3033 | 0.8018 | 7.3s |

## Family comparison (Holm-corrected, α=0.05)

**Three-way statistical tie at the top:** RandomForest, HistGradientBoosting
and LightGBM are pairwise `INDISTINGUISHABLE` from each other — every
pairwise difference (0.009–0.019 nAP) is far below the design's MDE (0.164
nAP at 80% power), so this is **underpowered, not equal**, per the
protocol's own decision vocabulary.

**Decisive separation below that tier:** all three top-tier tree models beat
`ExtraTrees` and both `LogisticRegression` variants at `p_holm < 0.03`
(e.g. RandomForest vs ExtraTrees: diff +0.262, p_holm=0.008; RandomForest vs
LogReg-L2: diff +0.281, p_holm=0.021).

`LogisticRegression_L1` vs `LogisticRegression_L2`: `INDISTINGUISHABLE`
(diff −0.010, underpowered) — L1 regularization does not measurably change
ranking quality here, at 679.7s of runtime cost for the L1 (liblinear
solver) vs 3.2s for L2 (lbfgs). **Lesson logged**: do not default to
`liblinear` for L1 at this data scale (208k rows) in future experiments;
`saga` or a reduced `max_iter` would be the next thing to try if L1 is
wanted again.

## The one genuinely new, unexpected finding: ExtraTrees underperforms RandomForest by a wide, significant margin

Same `max_depth=12`, `min_samples_leaf=20`, `n_estimators=100` for both — the
**only** structural difference between RandomForest and ExtraTrees is split
selection (RF searches for the best threshold per candidate feature;
ExtraTrees picks a random threshold). ExtraTrees lands statistically in the
same tier as the linear models (indistinguishable from both LogReg
variants), decisively below RandomForest/HistGradientBoosting/LightGBM
(p_holm 0.007–0.013 against all three).

This was not predicted by the research protocol (RandomForest and
ExtraTrees are usually close cousins) and is flagged per RULE 8/investigating-anomalous-results
practice rather than accepted at face value:

- **Plausible mechanism**: at 5.5% base rate with skewed, heavy-tailed
  amount features (`total_input_amount`, `btc_sent_total_asof_t`, etc.), a
  *best*-split search can find the specific threshold that isolates a small
  cluster of positives; a *random* threshold, drawn from the feature's
  observed range, is far less likely to land near that informative cutoff,
  especially early in a tree with only 12 max depth and a 20-sample leaf
  floor to work with. This is a testable mechanism, not yet tested — a
  follow-up experiment (does ExtraTrees close the gap as `n_estimators`
  increases, which is the usual way its randomness is compensated for) is a
  candidate for a future inner-loop cycle, not run here.
- **Ruled out as an explanation, on inspection**: this is not a
  preprocessing artifact — both models saw identical `nan_to_num`'d
  `float32` arrays, identical features, identical folds.

## Interpretation for Phase 6 (bottleneck search)

Because three *architecturally different* best-split tree learners (bagging
/ RandomForest, boosting / LightGBM, boosting / HistGradientBoosting) land
in the same statistical tier, **the bottleneck to further gains is unlikely
to be "which specific advanced tree ensemble to use."** The candidate
explanations left standing are: feature quality/representation, labels, the
address-last-snapshot unit of evaluation (`02_data_audit.md`), or the
ceiling implied by the protocol's own measured power limits. Phase 6
(feature ablation) is designed around this reframing.

## Decision

- **KEEP**: RandomForest, HistGradientBoosting and LightGBM as the
  surviving candidate tier — any of the three is a defensible choice on
  ranking-quality grounds alone; this program does not pick one yet
  (calibration, runtime, explainability and alert-policy fit are separate,
  later axes — Phases 8/9).
- **REJECT**: ExtraTrees and both LogisticRegression variants as the sole
  production candidate, on this feature set, at these hyperparameters.
- **INCONCLUSIVE**: RandomForest vs HistGradientBoosting vs LightGBM
  ranking-quality ordering — the data cannot resolve a difference this
  small (RULE 15 applies).
- **NEEDS_MORE_DATA**: whether ExtraTrees' gap closes at higher
  `n_estimators` — not tested.

## What this reproduces vs. what is new

Reproduces (confirmatory, matches `research/protocol_2026_09_22/run_protocol.py`
and `HANDOFF.md`'s Trap table): RandomForest vs LightGBM
`INDISTINGUISHABLE`, both decisively beat LogisticRegression.

New in this experiment: HistGradientBoosting and ExtraTrees added to the
family; the three-way tree tie is now measured directly rather than inferred
from a two-model comparison; the ExtraTrees anomaly is a new finding not
previously in the project's record.

## Addendum 2026-09-23: round 2 — justified variants of the tied tier (exp06_model_search_round2)

Continuation from this checkpoint, per explicit instruction not to re-run
LogisticRegression or plain LightGBM without a specific unresolved
hypothesis. exp01's LightGBM/RandomForest/HistGradientBoosting/ExtraTrees
results were **reconstructed from the saved per-fold JSON, not refit**, and
compared against 6 new, individually-justified variants in one 45-comparison
Holm-corrected family (`results/exp06_model_search_round2.json`).

| New candidate | Justification | nAP mean | vs its unweighted/plain counterpart |
|---|---|---|---|
| LightGBM_balanced | class-weighting, untested in exp01 despite 5.5% prevalence | 0.5552 | −0.036 vs plain LightGBM (indistinguishable, underpowered) |
| RandomForest_balanced | same | 0.5140 | −0.058 vs plain RandomForest (indistinguishable, underpowered) |
| HistGradientBoosting_balanced | same | 0.5548 | −0.027 vs plain HGB (indistinguishable, underpowered) |
| LightGBM_deeper | tests model-capacity bottleneck directly (3× estimators, more leaves, lower LR) | 0.5727 | −0.019 vs plain LightGBM (indistinguishable, underpowered) |
| LightGBM_lambdarank | genuine ranking-objective (NDCG-style), justified by the problem's own ranking framing | 0.5034 | −0.088 vs plain LightGBM (nominal p=0.022, **p_holm=0.614 — not significant after correction**, also sub-MDE) |
| ExtraTrees_500trees | directly answers this file's own open question: does more averaging close the RF gap? | 0.3034 | +0.007 vs ExtraTrees@100trees (indistinguishable — **answer: no**) |

**Result: no new variant produced a statistically defensible improvement
over the tied tier.** Every one of the 45 pairwise comparisons in the
corrected family is either not significant after Holm correction, below the
0.164 MDE, or both. Class-weighting is numerically worse in all three
tested models (a consistent, non-decisive pattern — not claimed as
significant). LightGBM_lambdarank's chunking workaround (LightGBM enforces
a 10,000-row cap per ranking query group — the first fit attempt with one
whole-fold group failed outright with this exact error) numerically
underperforms; this is evidence against *this specific* engineering
compromise, not against ranking objectives generally, and is reported with
that scope limit explicit.

**Open question closed**: ExtraTrees at 500 trees does not close the gap to
RandomForest (0.3034 vs 0.3103 at 100 trees — statistically indistinguishable
from itself, both still far below RandomForest's 0.5723) — confirms the
mechanism identified in the original ExtraTrees finding is about
split-selection quality, not insufficient ensemble averaging.

**Decision**: KEEP the original tied tier {RandomForest, HistGradientBoosting,
LightGBM} exactly as evaluated in exp01 — no round-2 variant displaces it.
REJECT "more ExtraTrees trees closes the gap." INCONCLUSIVE on
class-weighting and the chunked ranking objective specifically (underpowered
to rule out a real but small effect, and — for lambdarank — confounded by
the chunking workaround forced by data scale). NOT_TESTED: XGBoost
(environment constraint, unchanged from exp01).
