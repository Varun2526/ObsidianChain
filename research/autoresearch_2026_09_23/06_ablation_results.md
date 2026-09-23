# 06 — Feature Ablation Results (exp02_feature_ablation, completed 2026-09-23)

Full results: `results/exp02_feature_ablation.json`,
`results/exp02b_A_plus_B.json`. Ledger: `experiments.jsonl`
(`exp02_feature_ablation`).

## Setup

Single representative model, **LightGBM**, fixed hyperparameters (identical
to exp01), fixed folds/seed/preprocessing — only the feature set varies.
Justified by exp01: RandomForest/HistGradientBoosting/LightGBM are
statistically tied on the full feature set, so repeating every ablation
across all three would triple the cost for no expected new information.
Leave-one-group-out (`minus_X`) and single-group-only (`only_X`) for Groups
A–D; Group E (network) is absent from the dataset (see `01_repo_audit.md`)
and cannot be ablated. A follow-up `A_plus_B` combination was run to
separate B's marginal contribution from C+D's.

## Results

| variant | cols | nAP mean | nAP sd | vs full | verdict (Holm-corrected) |
|---|---|---|---|---|---|
| **full** | 24 | 0.5914 | 0.1636 | — | reference |
| minus_A_transaction | 14 | 0.2454 | 0.1079 | −0.346 | **FAVOURS full, p_holm=0.000** |
| only_A_transaction | 10 | 0.4722 | 0.1930 | −0.119 | INDISTINGUISHABLE (underpowered, p_holm=0.053) |
| minus_B_address_history | 14 | 0.5694 | 0.1735 | −0.022 | FAVOURS full, p_holm=0.048 (effect size below MDE — see note) |
| only_B_address_history | 10 | 0.1929 | 0.1581 | −0.399 | **FAVOURS full, p_holm=0.000** |
| minus_C_graph | 22 | 0.5350 | 0.1776 | −0.056 | INDISTINGUISHABLE (p_holm=0.126) |
| only_C_graph | 2 | 0.1292 | 0.0907 | −0.462 | **FAVOURS full, p_holm=0.000** |
| minus_D_patterns | 22 | 0.5896 | 0.1637 | −0.002 | INDISTINGUISHABLE (p=0.653) |
| only_D_patterns | 2 | 0.0151 | 0.0258 | −0.576 | **FAVOURS full, p_holm=0.000** |
| A_plus_B (follow-up) | 20 | 0.5359 | 0.1667 | −0.055 | nominally significant uncorrected (p=0.036) but sub-MDE; not run through the corrected family — treat as suggestive, not confirmed |

## Interpretation

**Group A (instantaneous transaction shape) is the dominant carrier of
ranking signal.** Removing it collapses performance to near the weakest
tier (0.2454, barely above `only_C`/`only_D`); Group A *alone* (10 columns:
counts, totals, fee, fee_ratio, spreads) recovers 80% of full performance
(0.4722 of 0.5914) with no history, graph or pattern information at all.
This is a genuinely load-bearing finding for a project whose architecture
narrative emphasizes temporal/graph/behavioral sophistication (Groups B/C/D,
and the network layer): **the single transaction's own shape — mostly
amounts, counts and fee ratio — is doing most of the work.**

**Group B (address history) adds a small, borderline-significant
increment**, and is decisively insufficient alone (0.1929). The
`minus_B` result (p_holm=0.048, just inside significance) is a useful
illustration of the protocol's own honesty mechanism: it is *statistically
significant* under the corrected test yet *below the design's MDE*
(0.164 nAP) — meaning it is real enough to survive multiplicity correction
at this specific magnitude, but small enough that this program is not
claiming high confidence in its exact size. Both statements are reported
together, per `00_research_protocol.md`.

**Group C (graph: 2 columns) and Group D (patterns: 2 columns) do not
survive the corrected family test as marginal contributors** on top of
A+B (`minus_C` p_holm=0.126, `minus_D` p=0.653 — not even nominally
significant). Each is decisively better than nothing when used *alone*
(`only_C`=0.1292, `only_D`=0.0151, both far below `only_A`), so they are not
useless signals in isolation — but once A and B are already present, this
protocol cannot defend a claim that removing them costs anything.

**The A+B-only follow-up (0.5359) vs full (0.5914)** shows a small
nominally-significant gap (uncorrected p=0.036, diff 0.055 — still below
MDE) attributable to C+D together, suggesting a small combined contribution
exists even though neither survives alone under the stricter per-group test.
This is reported as suggestive, not confirmed — it was not run through
`compare_family` alongside the other 8 variants and should be treated as a
lead for a future, properly-powered follow-up rather than a claim.

## Answering Phase 6's bottleneck question (A–I)

| Candidate bottleneck | Verdict from this experiment |
|---|---|
| A. Model capacity | REJECTED as the primary bottleneck — exp01 already showed 3 architecturally different tree ensembles tie; more model sophistication is not where the next gain is. |
| B. Feature quality | **Implicated, but asymmetrically** — Group A is high quality and load-bearing; Groups C/D as currently engineered (2 columns each) are not pulling weight once A/B exist. Group B is a small but real contributor. |
| C. Labels | Not tested by this experiment; see `03_label_audit.md` for structural label findings (static per-address, no target leakage found). |
| D. Representation | **Directly implicated by `02_data_audit.md`**: the address-last-snapshot, boundary-spanner-dropping representation is unchanged across all ablations here — this experiment cannot separate "Group C/D features are weak" from "Group C/D features are weak *given this particular representation*." A graph/temporal feature computed on the full per-transaction stream (rather than collapsed to one last-snapshot row per address) might behave differently — untested, flagged for Phase 10. |
| E. Temporal formulation | Group B (temporal/address-history) already active-duration/velocity/gap features exist and contribute modestly; no evidence yet that a different temporal formulation would do better — untested directly. |
| F. Graph representation | Group C is only 2 columns (`unique_counterparties_asof_t`, `cluster_size_asof_t`) — a very thin graph representation. Its weak marginal contribution here is consistent with either "graph structure doesn't matter much" or "this particular thin graph representation doesn't capture what matters" — this experiment cannot distinguish the two. Phase 10 territory. |
| G. Network evidence | Not tested — absent from dataset (`01_repo_audit.md`, `02_data_audit.md`). |
| H. Calibration | Not tested here — Phase 8. |
| I. Alert policy | Not tested here — Phase 8. |

## Decision

- **KEEP**: Group A and Group B as necessary, evidenced contributors.
- **REJECT** (as currently engineered, on top of A+B, under this test's
  power): Group C and Group D as individually load-bearing marginal
  features — this is a claim about their *measured marginal contribution in
  this ablation*, not a claim that graph/pattern features are inherently
  useless (see representation caveat above).
- **INCONCLUSIVE**: the exact size of Group B's marginal contribution
  (significant but sub-MDE) and the combined C+D contribution (nominally
  significant, not run through the full corrected family).
- **NEEDS_MORE_DATA / NEEDS_MORE_ENGINEERING**: whether richer graph (Group
  C) or pattern (Group D) features — not just more of the current thin
  ones — would change this picture. Current Group C/D are 2 columns each;
  this experiment tested the *existing* engineering, not the *ceiling* of
  what graph/pattern signal could contribute if better engineered.

## Addendum 2026-09-23: confirmed on a second model family (exp07_ablation_rf_confirmation)

`13_red_team_review.md` flagged this experiment's single-model scope
(LightGBM only) as unconfirmed. `exp07_ablation_rf_confirmation.py` repeats
the same leave-one-group-out design on **RandomForest** — bagging, not
boosting, a genuinely different mechanism:

| Ablation | RandomForest p_holm | LightGBM p_holm (this file, above) | Agreement |
|---|---|---|---|
| minus_A | **0.001** (even more decisive) | 0.000 | Both: FAVOURS full, decisive |
| minus_B | 0.156 (not sig. after correction) | 0.048 (borderline sig.) | Both: small, non-decisive-to-borderline |
| minus_C | 0.156 (not sig.) | 0.126 (not sig.) | Both: not significant |
| minus_D | p=0.941 (essentially zero effect) | p=0.653 (essentially zero effect) | Both: no measurable effect |

**The pattern replicates closely enough (same ranking of group importance,
same qualitative verdicts, similar effect sizes — see the cross-model table
in `results/exp07_ablation_rf_confirmation.json`) to conclude this is a
property of the DATA, not an artefact of LightGBM's specific boosting
mechanism.** The scope limitation flagged in `13_red_team_review.md` is
now closed: Group A dominance and Group C/D's weak marginal contribution
(as currently engineered) hold across both bagging and boosting tree
ensembles.
