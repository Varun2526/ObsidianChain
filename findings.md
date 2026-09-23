# Research Findings

## Research Question

Find the strongest defensible pipeline for ranking suspicious Bitcoin
activity / investigative leads under realistic time-ordered evaluation
(ObsidianChain PS-native scope), while preserving leakage prevention,
reproducibility, explainability, and operational usefulness.

## Current Understanding

*(updated 2026-09-23, after exp01 + exp02 — outer-loop cycle 1)*

Two completed experiments now anchor the program. **exp01** (6-model family
comparison) found that RandomForest, HistGradientBoosting and LightGBM are
statistically indistinguishable from each other (all pairwise differences
0.009–0.019 nAP, far below the 0.164 MDE), and all three decisively beat
LogisticRegression (both L1/L2) and — unexpectedly — ExtraTrees, which
underperforms RandomForest by 0.262 nAP (p_holm=0.008) despite identical
depth/leaf hyperparameters, isolating best-split-vs-random-split selection
as a real mechanism on this imbalanced, skewed-feature problem. **exp02**
(feature-group ablation, LightGBM only) then reframed the open question:
since model family doesn't matter within the tied tier, does feature group?
It found Group A (instantaneous transaction shape — counts, amounts, fee
ratio) carries the large majority of the ranking signal (removing it drops
nAP from 0.591 to 0.245; A alone reaches 0.472), Group B (address history)
adds a small but real increment, and Groups C (graph, 2 cols) and D
(patterns, 2 cols) do not survive the corrected test as marginal
contributors once A+B are present — though each is far better than nothing
in isolation, so this is a statement about their *current, thin*
engineering on top of A+B, not a claim that graph/behavioral signal is
inherently useless.

This project already had a rigorous, tested evaluation protocol
(`src/obsidianchain/ml/protocol.py`) before this research program started —
the program's job is to *use* it exhaustively across model families,
features, calibration and alert policy, not to invent a new one. The single
most important fact already established (and independently reproduced by
`exp01`, see below): **on this dataset, the fold-to-fold evaluation-window
variance (sd ≈ 0.18 nAP) is roughly 35× the seed-to-seed variance**, so any
comparison that doesn't pair by rolling fold and correct for multiple
comparisons is very likely reporting noise dressed as a finding. The
project's own history contains a concrete example: RandomForest was
originally selected over LightGBM on one validation window (0.5702 vs
0.5228 PR-AUC); under the correct protocol the difference is statistically
indistinguishable and below the design's minimum detectable effect (0.164
nAP).

## Key Results

- **exp01_model_family** (complete): RF/HGB/LightGBM tied (KEEP all three
  as candidates); LogReg-L1/L2 and ExtraTrees REJECTED as sole candidates
  (p_holm<0.03 against the tied tier). See `05_model_comparison.md`.
- **exp02_feature_ablation** (complete): Group A dominant, Group B a small
  real contributor, Groups C/D not defensible as marginal contributors atop
  A+B under this test's power (though each beats nothing in isolation). See
  `06_ablation_results.md`.
- **Data availability finding (confirmed, not pending):** the current
  PS-native development dataset (`train.parquet`/`validation.parquet`) was
  built with `include_network=False`. Group E (network) features are
  **absent**, not merely weak — Phase 4 Group F/G (network-layer,
  cross-layer ablation) cannot run against this dataset without regenerating
  it with network telemetry joined through `network/boundary.py`.
- **Feature health (confirmed):** all 24 declared v2 CORE feature columns
  pass `diagnostics.healthy_features` against the live dev frame — no
  constants, duplicates or functional dependencies, which is the expected
  and now-verified consequence of the v1→v2 fixes (fabricated
  even-division features removed, `is_peeling_candidate` / `is_mixing_candidate`
  redefined against real value summaries).

## Patterns and Insights

- **Architecture diversity among best-split trees doesn't matter here;
  split-selection mechanism does.** RF (bagging), HistGradientBoosting and
  LightGBM (both boosting, different internals) tie. ExtraTrees (same
  bagging family as RF, only differs in using random rather than best
  splits) loses decisively. The signal: on this imbalanced (5.5%), skewed-
  amount-feature problem, *finding* the right threshold matters far more
  than *which* ensembling strategy is layered on top of threshold-finding.
- **Simple, instantaneous features dominate engineered temporal/graph
  features.** This cuts against the intuitive project narrative (a lot of
  engineering effort has gone into Groups B/C/D and the network layer) and
  is the single most actionable finding so far for where to invest further
  feature engineering effort — or where to accept a simpler, faster,
  more explainable model is not leaving value on the table.

## Key Results (continued)

- **exp03_error_analysis** (single-fold diagnostic): false negatives are
  disproportionately cold-start addresses (median 0 prior transactions vs
  median 1 for true positives at K=100) — connects mechanistically to
  Group B's real-but-modest contribution from exp02.
- **exp04_calibration_alert_policy**: isotonic calibration collapses 2,185
  distinct raw scores to 50, and this produces a precision@50 tie-bound
  width of 0.26 (0.16–0.42) vs 0.02 for the raw score — a concrete,
  quantified confirmation (on PS-native v2 specifically) of the
  already-suspected severity-band/tie problem. Brier calibration result on
  this fold is an unresolved anomaly (got worse, not better — contradicts
  the frozen v1 model's own recorded improvement).
- **exp05_explanation_audit** (MOST IMPORTANT FINDING SO FAR): the current
  `ps_model.py` explanation method's direction label agrees with true
  TreeSHAP direction only 33.9% of the time (worse than chance), and top-3
  feature overlap with true TreeSHAP is only 1.4/3 on average. The current
  "explainability" is not faithful to the model — confirmed by exact
  comparison (TreeSHAP verified locally exact to 1.91e-14), not assumed.

- **A synthetic benchmark's difficulty must be checked, never assumed.** A
  deterministic, low-noise generator (exact shape parameters, tight
  tolerances) can produce a task that looks superficially adversarial
  (deliberately confusable behavior classes) while still being trivially
  saturated by the exact detectors it was designed to stress-test — because
  matched determinism on both sides leaves no ambiguity. Always check for
  a ceiling effect (e.g. exact nAP=1.0) before interpreting any comparison
  run on synthetic data, the same way this program checks fold spread
  before trusting a real-data comparison.

## Lessons and Constraints

- **Never compare model families outside `protocol.compare_family`.** A
  hand-rolled pairwise script (even with the right test) loses the Holm
  correction across the full comparison set. `exp01` runs all six
  candidates through one `compare_family` call for this reason.
- **xgboost is not available** in this environment/vendored wheel set —
  documented as NOT_TESTED in Phase 5, not silently dropped and not treated
  as a negative result.
- **Do not resurrect v1 feature columns** (`output_entropy`,
  `equal_output_count`, `input_amount_std`, `output_amount_std`,
  `in_degree_asof_t`, `out_degree_asof_t`) — each is either constant,
  numerically-zero noise, or an exact duplicate of a v2 column, confirmed by
  `ml/diagnostics.py` and re-confirmed live in this program's protocol-lock
  step.
- **Git commits are not made autonomously in this program** — the project's
  standing working agreement (HANDOFF.md §11) overrides the generic
  autoresearch skill's default commit-per-milestone protocol.

- **LightGBM has a concrete, evidenced explainability advantage over
  RandomForest/HistGradientBoosting**: it has an exact, built-in TreeSHAP
  path (`Booster.predict(pred_contrib=True)`) requiring no extra dependency;
  the `shap` package needed for sklearn tree models is not installed/vendored.
  This is the first evidence-backed tiebreaker among exp01's tied top tier.

## Key Results — cycle 2 (2026-09-23, continuation from checkpoint)

- **exp06_model_search_round2**: 6 justified variants (class-weighted
  RF/HGB/LightGBM, a deeper/higher-capacity LightGBM, a chunked LambdaRank
  objective, ExtraTrees at 500 trees) tested against exp01's tied tier
  **reconstructed from saved results, not re-run**. None produced a
  defensible improvement (all 45 pairwise comparisons in the corrected
  family are non-significant, underpowered, or both). Class-weighting is
  numerically (not significantly) worse everywhere tested. Closed an open
  question: more ExtraTrees estimators does NOT close its gap to
  RandomForest — confirms the mechanism is split-selection quality, not
  under-averaging.
- **exp07_ablation_rf_confirmation**: closed the red-team-flagged scope
  limitation on `06_ablation_results.md` — repeated the feature-group
  ablation on RandomForest (a different tree mechanism than LightGBM) and
  got the same qualitative pattern (Group A decisive, Group D negligible,
  B/C small/non-decisive) — **confirmed data-driven, not LightGBM-specific.**
- **exp08_counterparty_history_feature**: engineered and tested the
  cold-start-motivated counterparty-history feature. Result: +0.024 nAP,
  consistent direction, but INDISTINGUISHABLE (p=0.199, sub-MDE) — a
  genuinely promising but unconfirmed lead, reported at the correct
  confidence level rather than oversold. **Also surfaced an important
  correction**: 90.9% of all rows in the diagnostic fold are cold-start
  (not a rare subpopulation) — this dataset is structurally a cold-start
  problem, which raises the stakes of resolving this INCONCLUSIVE result.
- **exp09_alert_policy_redesign**: prototyped the corrected (properly
  held-out) threshold-selection procedure this program recommended. It
  fixes the same-sample bug (thresholds generalize almost exactly within
  the same time period), but revealed a DEEPER problem: correctly-derived
  thresholds still collapse against a genuinely new time window (CRITICAL
  90%→59%, HIGH 75%→48%, MEDIUM 50%→18%). Static-threshold severity bands
  are fundamentally in tension with this dataset's known temporal
  instability — strengthens the case for a top-K/rank-based policy beyond
  what the original tie-bound evidence alone showed.

## Key Results — cycle 3 (2026-09-23, new axis: does the dataset itself limit conclusions?)

- **Discovery**: `src/obsidianchain/world/` already exists — a coherent
  chain+network synthetic world generator built for Phase 2/3 clustering
  research, with a 120-entity instance already on disk
  (`data/synthetic_world/`). Its own docstring states almost exactly this
  program's Group F/G question as its original motivation.
- **exp11**: confirmed empirically (not just by reading source) that
  production `features_ps.py`'s Group E feature code is a **stub** —
  `observer_diversity`/`peer_count` hardcoded to 1.0 regardless of real
  underlying variation. A second, independent blocker on network-layer
  evaluation, on top of the already-known data-absence gap (H2).
- **exp12 (decisive)**: the full CORE feature set achieves **exact
  nAP=1.0000** in every usable fold on this synthetic world, **regardless of
  which network-feature variant is added** — a ceiling effect, not a
  network-value result. Investigated as an anomaly before trusting: ruled
  out "easy negatives dominate" (adversarial behaviors are the majority);
  confirmed even the generator's own stated hardest case (MERCHANT_SWEEP)
  separates from true positives by a ~7× score margin. Mechanistic, not a
  bug: deterministic, fixed-parameter behavior shapes leave existing
  detectors nothing to fail on.
- **H7 REFUTED**: the existing synthetic world instance is not suitable as
  a Group F/G benchmark, for two independent, compounding reasons (ceiling
  effect + only 3/12 folds usable at current scale). A concrete,
  evidenced specification for what a useful next-generation instance would
  require is documented (`16_synthetic_world_investigation.md`) but not
  built — substantial new-behavior-design effort, not yet justified.
- **Nothing about exp01–exp10 changed.** This was an additive, clearly
  separated investigation (own experiment IDs, own dataset paths,
  `SCOPE_UNDECLARED` to prevent accidental pooling).

## Open Questions

- Does any of the six Phase-5 candidate families beat RandomForest by more
  than the 0.164 nAP MDE? (exp01 pending)
- Which feature groups (A/B/C/D) actually carry the ranking signal, once
  measured by ablation rather than assumed? (Phase 6, not started)
- Is the current severity-band policy (rank-derived cutoffs used as value
  thresholds) defensible under a controlled alert-policy comparison, given
  HANDOFF.md already reports CRITICAL advertises 90% but measures 82.2%
  validation / 37.2% test precision? (Phase 8, not started)
- Are `ps_model.py`'s current "explanations" (global importance × |value|)
  meaningfully different from TreeSHAP once measured side by side? **Answered
  (exp05): yes, dramatically — 33.9% direction agreement, worse than chance.**
- Does the counterparty-history feature's +0.024 nAP become significant with
  more folds, a richer feature (diversity not just max/mean), or is it
  genuinely sub-MDE noise? (exp08, INCONCLUSIVE, open)
- Would a periodically-re-derived (rather than fixed) severity threshold
  recover acceptable out-of-time precision? (exp09, NEEDS_MORE_DATA, open)
- Would a next-generation synthetic world instance (noisy/overlapping
  behavior parameters, a genuinely chain-ambiguous behavior class, larger
  scale) actually show network evidence adding value, once the ceiling
  effect and power problems are engineered away? (exp11/exp12, specified
  but not built — open, substantial scope)

## Optimization Trajectory

| experiment | candidate | nAP mean | nAP sd | vs prior best | decision |
|---|---|---|---|---|---|
| (protocol_2026_09_22, prior work) | RandomForest | 0.5723 | 0.1826 | baseline | reference |
| (protocol_2026_09_22, prior work) | LightGBM | 0.5914 | 0.1636 | +0.0190 (INDISTINGUISHABLE, underpowered) | reference |
| (protocol_2026_09_22, prior work) | LogisticRegression | 0.2915 | 0.2533 | −0.28 (FAVOURS tree models, p_holm<0.05) | reference |
| exp01_model_family | LightGBM (best of tied tier) | 0.5914 | 0.1636 | ties RF | KEEP (tied tier) |
| exp01_model_family | ExtraTrees | 0.3103 | 0.2221 | −0.262 vs RF, p_holm=0.008 | REJECT |
| exp01_model_family | LogReg L1/L2 | 0.29-0.30 | ~0.25 | −0.28-0.30 vs tied tier, p_holm<0.03 | REJECT |
| exp02_feature_ablation | full (A+B+C+D) | 0.5914 | 0.1636 | reference | KEEP |
| exp02_feature_ablation | minus_A (B+C+D only) | 0.2454 | 0.1079 | −0.346, p_holm=0.000 | A is necessary |
| exp02_feature_ablation | only_A | 0.4722 | 0.1930 | −0.119, underpowered | A alone recovers 80% |
| exp02_feature_ablation | minus_D (A+B+C) | 0.5896 | 0.1637 | −0.002, p=0.653 | D not defensible atop A+B+C |
| exp02b | A_plus_B only | 0.5359 | 0.1667 | −0.055 vs full, nominal p=0.036, sub-MDE | suggestive C+D increment, unconfirmed |
