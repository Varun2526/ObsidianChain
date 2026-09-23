# 00 — Research Protocol (locked 2026-09-23, before extensive experimentation)

Scope: `research/autoresearch_2026_09_23/`. This document is the pre-registration
for the investigative-ranking research program. It is written from the actual
code in `src/obsidianchain/ml/protocol.py`, `ml/metrics.py`, `ml/diagnostics.py`
and `pipeline/features_ps.py` — not proposed abstractly — because the project
already has a canonical, tested evaluation protocol and this program's job is
to use it, not replace it.

## Research question

> Find the strongest defensible pipeline for ranking suspicious Bitcoin
> activity / investigative leads under realistic time-ordered evaluation,
> while preserving leakage prevention, reproducibility, explainability, and
> operational usefulness.

This is an **imbalanced ranking problem**, not a classification-accuracy
problem. Prevalence across development folds ranges 3.3%–21.0% (6.3× swing).
Accuracy is never reported (`ml/metrics.py` docstring: "not informative" at
this prevalence).

## Scope declaration (mandatory per `protocol.py`)

All experiments in this program run under **`SCOPE_PS_NATIVE`**
(`protocol.SCOPE_PS_NATIVE`): the PS-native feature schema
(`ps_native_features/2`), address-as-of-timestamp unit of analysis, over
`data/models/ps_native/datasets/{train,validation}.parquet`.

The Phase 6 LightGBM / Elliptic++ M0–M4 path (`SCOPE_PHASE6`) is a **separate,
frozen reference scope** (serves alert run `043ea584e99daf99`). This program
does **not** compare numbers across the two scopes — `protocol._require_same_scope`
raises `ScopeMismatchError` if attempted, and the reasoning is in
`docs/decisions/0001-two-production-model-paths.md`. Any apples-to-apples
comparison between the two paths would require rebuilding one under the
other's feature set and unit of analysis first, which is out of scope here
unless a later phase explicitly designs that controlled experiment.

## The sealed holdout — never touched during this program

- `HOLDOUT_START = 42`, `DATASET_END = 49`. `protocol.holdout()` raises
  `HoldoutSealError` unless `break_seal(reason)` has been called.
- **No experiment in this program calls `break_seal`.** The holdout is read
  exactly once, at the very end (Phase 16), after the candidate pipeline,
  features, preprocessing, model, alert policy and explanations are all
  frozen.
- `OBSIDIANCHAIN_REGENERATE_HOLDOUT` stays unset for the duration of this
  program. `test.parquet` (schema `ps_native_features/1`) is not regenerated.
- Every experiment script reads only `train.parquet` + `validation.parquet`
  (timesteps 1–41), matching `protocol.development()`.

## Evaluation design (already fixed by `ml/protocol.py` — not re-derived)

| Parameter | Value | Source |
|---|---|---|
| Fold scheme | Rolling-origin, expanding window | `protocol.rolling_origin_folds` |
| Fold width | 2 timesteps | measured design sweep, see `protocol.py` module docstring |
| Min training history | 16 timesteps | `MIN_TRAIN_TIMESTEPS` |
| Resulting folds | 12, all inside t1–t41 | reproduced live: `t≤16→t17-18` … `t≤38→t39-40` |
| Primary metric | normalised AP (nAP) | `(AP − prevalence) / (1 − prevalence)`, puts no-skill at 0 and perfect at 1 in every fold regardless of prevalence swing |
| Secondary ranking metric | Precision@K (K ∈ {100, 200, 500}) | reported as **[worst, best] bounds over tie orderings**, never a single point value — isotonic calibration collapses ~10,000 scores to ~55, so a point estimate at the tie boundary is an artefact of row order |
| Primary comparison | Paired t-test, per fold | `paired_verdict`, α = 0.05 |
| Multiplicity correction | Holm-Bonferroni | `protocol.holm`, applied whenever ≥3 candidates are compared as a family (`compare_family`) |
| Robustness check 1 | Exact sign-flip permutation | usable only at ≥7 folds (`MIN_FOLDS_FOR_PERMUTATION`); reported but does not override the t-test |
| Robustness check 2 | Wilcoxon signed-rank | to be added per-experiment where the paired-difference distribution looks non-normal (not built into `protocol.py`; run ad hoc via `scipy.stats.wilcoxon` and logged alongside) |
| MDE at 80% power | **0.164 nAP** | measured by simulation at observed paired spread (sd=0.185), 12 folds — see `POWER_LIMITS` |
| Permanently unresolvable | differences below ≈0.05 nAP | stated in `protocol.py`, not a caveat to rediscover each time |

**Consequence for this program:** a verdict of `INDISTINGUISHABLE` on a
difference below 0.164 nAP is reported as **underpowered**, not as "the
models are equal." RULE 3 and RULE 15 apply throughout: no claim of
superiority below the design's resolving power, and INCONCLUSIVE is a valid,
required verdict class alongside KEEP/REJECT/NEEDS_MORE_DATA.

## Primary objectives (tracked every experiment)

1. nAP (mean, sd, min, max, per-fold) — ranking quality, temporal stability, variance in one shot (the fold spread *is* the stability/variance measure)
2. Precision@K / Recall@K at realistic alert budgets (K=100/200/500 initially; alert-budget realism revisited in Phase 8 against actual investigator capacity)
3. Calibration quality — Brier, reliability curve (`ml/metrics.calibration_curve`) — measured **separately from ranking** (RULE 9)
4. Computational cost — wall-clock fit+predict time per fold, logged per experiment

## Secondary objectives

False-positive/false-negative concentration by feature group and score band, score distribution, alert volume under candidate policies, severity-band behavior, explanation fidelity (SHAP/TreeSHAP vs the current `feature_importances_ × |value|` proxy in `ps_model.py`), robustness to temporal changes (fold-to-fold spread already captures this), robustness to graph/structural changes (via Group C ablation), operational usefulness (does a change survive contact with the alert policy, not just the raw metric).

## What will NOT be optimized

- Ordinary accuracy (RULE 1) — never computed as a decision criterion, only ever reported if requested for context, always beside prevalence.
- A model chosen for architectural novelty (RULE 2, RULE 12) — GNN/Transformer/temporal-sequence approaches are only attempted if Phase 10 finds the current data/evaluation design can support a fair evaluation of them; otherwise documented as **not attempted, and why**.

## Feature schema boundary

All feature work in this program starts from **`ps_native_features/2`**
(`CORE_PS_FEATURE_COLUMNS`, `pipeline/features_ps.py`). The v1 schema is
frozen for provenance only (`data/models/ps_native/v1/`) and is **not**
resurrected — `output_entropy`, `equal_output_count`, `input_amount_std`,
`output_amount_std`, `in_degree_asof_t`, `out_degree_asof_t` are known-dead
v1 columns (see `ml/diagnostics.py` docstring and `01_repo_audit.md` §Known
leakage / fabrication risks). Any reintroduction requires a positive,
evidenced reason and a note in this file.

**Confirmed at protocol-lock time:** the current `train.parquet` /
`validation.parquet` were built with `include_network=False`
(`research/reproduction/build_ps_dataset.py:158`) — **Group E (network)
features are not present in this dataset.** Phase 4 Group F/G (network-layer,
cross-layer ablation) cannot run against this dataset as-is; it requires
either regenerating the PS-native dataset with network telemetry joined in,
or running the comparison against the separate Phase 6 M0–M4 path (different
scope, not comparable here). This is recorded as a **data availability gap**,
not an experiment result — see `12_pipeline_gap_analysis.md`.

## Decision vocabulary (used in every experiment record)

- **KEEP** — statistically defensible improvement under the family-wise
  correction, above MDE, replicated across the fold set.
- **REJECT** — no improvement, or a regression, under the same standard.
- **INCONCLUSIVE** — verdict is `INDISTINGUISHABLE` *and* the observed
  difference is below `MDE_80_POWER` (0.164 nAP) — the design cannot resolve
  it, full stop.
- **NEEDS_MORE_DATA** — the experiment could not run at all under the
  protocol (e.g. feature not present, fold count too low for a robustness
  check, holdout would have to be touched).

## Experiment ledger

Every experiment appends one record to
`research/autoresearch_2026_09_23/experiments.jsonl` (schema: experiment_id,
hypothesis, baseline, single_change, dataset_version, feature_version,
model_version, evaluation_protocol, seeds, folds, metrics, runtime_sec,
statistical_comparison, error_analysis pointer, decision). Protocol lock
(this file) precedes any result by construction — it is written before
`exp01` was run.

## Git discipline note

The project's own working agreement (`HANDOFF.md` §11): *"Nothing is
committed by Claude without being asked."* This overrides the generic
autoresearch skill's default git-commit-per-milestone protocol. This program
therefore does **not** auto-commit protocol/result milestones to git; all
state lives in the working tree and in `research-state.yaml` /
`research-log.md` / `findings.md` at the repo root, and commits happen only
when the user asks.
