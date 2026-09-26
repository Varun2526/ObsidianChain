# Results register — PS-native ML (last updated 2026-09-23)

Every headline number produced by the PS-native research and production
program, with its status. **If a number is not in this table as VALID, do
not quote it.**

Status
- **VALID**: produced by the current, leakage-audited pipeline, under the
  stated unit and result type.
- **INVALID**: affected by a defect found later. Kept for the audit trail; never quote.
- **WITHDRAWN**: artifact or claim removed from use for a stated reason.
- **CORRECTED**: a factual claim that was wrong; the correction is given.
- **SUPERSEDED UNIT**: computed correctly, but in an evaluation unit later
  shown to be optimistic.

Result type
- **DEVELOPMENT**: tuning folds 1-6
- **CONFIRMATION**: folds 7-12, never used for selection
- **HOLDOUT**: Sealed Holdout Window (Timesteps 42–49 / t42-49), opened once (ADR 0003)
- **SHADOW**: candidate scored on production traffic
- **PRODUCTION**: labelled production traffic
- **SYNTHETIC**: Controlled Synthetic Benchmark; says nothing about Bitcoin

## Current production model: ObsidianChain Risk Model (champion: ps_native_v5)

| Result | Value | Type | Status |
|---|---|---|---|
| nAP, 12 folds, Time-Ordered Evaluation Protocol (Protocol B) | 0.808 mean, sd 0.176, worst 0.477 | DEV + CONFIRMATION | VALID |
| nAP, confirmation folds | 0.796 mean, worst 0.553 | CONFIRMATION | VALID |
| address P@100 | 0.971 mean, worst 0.76 | DEV + CONFIRMATION | VALID |
| ceiling-normalised R@500 | 0.869 mean, worst 0.532 | DEV + CONFIRMATION | VALID |
| R-precision | 0.759 mean | DEV + CONFIRMATION | VALID |
| ECE, honest per-fold Platt (folds 5-12) | 0.025 mean, 0.040 max | DEV + CONFIRMATION | VALID |
| single-row scoring latency p95 | 0.59 ms | local measurement | VALID |
| **nAP** | **0.548** | **HOLDOUT** | VALID |
| ROC-AUC | 0.946 | HOLDOUT | VALID |
| address P@100 (pooled Sealed Holdout t42-49) | 1.00 | HOLDOUT | VALID |
| address P@100 per first-seen step | later time windows (t43 0.18, t45 0.02, t47 0.28); others 0.44-1.00 | HOLDOUT | VALID |
| R-precision | 0.510 | HOLDOUT | VALID |
| ceiling-normalised R@500 | 0.950 | HOLDOUT | VALID |
| ECE / calibration slope | 0.010 / 1.04 | HOLDOUT | VALID |
| slice nAP: receiver-only / high-value q4 / first-seen | 0.168 / 0.135 / 0.425 | HOLDOUT | VALID |
| fallback Risk Model Fallback (ps_native_v5_fallback_no_g) nAP | 0.521 | HOLDOUT | VALID |
| seed-propagation stack, out of sample | nAP 0.807 -> 0.861 (p = 0.22, not significant) | DEVELOPMENT scenario | VALID |
| Group G (Upstream Flow Dynamics) ablation, Time-Ordered Evaluation Protocol (Protocol B) | +0.047 nAP (p = 0.013); confirmation worst 0.445 -> 0.553 | DEV + CONFIRMATION | VALID |
| input drift monitoring detects t43/t45 collapse | no (absolute PSI and baseline-relative reading both miss it) | HOLDOUT diagnostic | VALID (negative) |
| SHADOW result | none: no candidate has run on production traffic | SHADOW | — |
| PRODUCTION result | none: no labelled production traffic exists yet | PRODUCTION | — |

## Invalid, withdrawn, corrected

| Result | Status | Reason |
|---|---|---|
| ps_native_v2 12-fold nAP 0.619 / 0.648 | INVALID | txId order inside an Elliptic timestep let history features see later same-step transactions (about +0.06 nAP) |
| exp01-exp18 absolute metric values | INVALID (absolute); relative comparisons historical only | same txId-order leak; decisions from them were re-made in exp20-exp22 |
| exp20 first run, group G 0.771 | INVALID | same-step funding pairs in txId order; 1.4% ran backwards in time |
| CORE nAP 0.802 | INVALID | spend-DAG level written into the timestamp leaked chain depth into every "seconds" feature |
| ps_native_v3 scorecard (nAP 0.713, P@100 0.94, ECE 0.041) | WITHDRAWN | schema /4: simultaneous events visible to each other; per-column snapshot splice (groupby().last()) |
| last-snapshot (Retrospective Evaluation Protocol / Protocol A) nAP 0.829 on /5 | SUPERSEDED UNIT | final-event snapshot is hindsight; Time-Ordered Evaluation Protocol (Protocol B) is the primary unit |
| exp16 "80% precision at cosine >= 0.99" | INVALID | payout addresses seen in a single transaction; with the >= 2-transaction rule: 12% at 0.90, 0 of 12 at 0.99 |
| ps_native_v4, v4_fallback_no_g | WITHDRAWN | training script edited after training, so lineage not attestable; v5 is byte-identical and attested |
| doc 17: "7% fee==0 rows are missing fees encoded as zero" | CORRECTED | they are genuine zero-fee transactions in the source |
| exp12 Controlled Synthetic Benchmark nAP 1.000 | VALID as SYNTHETIC, uninformative | saturated generator |
| exp15 network ablation (relay AUC 0.803 -> 0.930) | VALID as SYNTHETIC only | the signal was placed in the generator on purpose |
