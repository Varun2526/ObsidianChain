# 20 — Production program: final engineering report (2026-09-23)

Branch `feat/ps-native-v3-production`. Commits, in order:

| Commit | Content |
|---|---|
| `e6f0afc` | research baseline (v3) |
| `5431241` | hardening, and the attested source of v5 |
| `92b6082` | v5 frozen before the holdout |
| `674f017` | holdout opened once; v5 promoted |
| `29ed148` | clean-checkout fixtures and markers |
| final commit | this report, CI, documentation, product and container fixes |

Result types are never mixed below: **DEV** (tuning folds 1-6),
**CONF** (confirmation folds 7-12), **HOLDOUT** (t42-49, opened once),
**SHADOW**, **PRODUCTION**.

## 1. What was discovered

Six leakage or validity defects (L1-L6) and one data defect (V1); details in
`19_leakage_audit.md`:

- L1: txId order inside Elliptic timesteps; every v2/v3 number inflated by
  about 0.06 nAP.
- L2: chain depth written into timestamps (invalid nAP 0.802).
- L4: simultaneous events visible to each other.
- L5: `groupby().last()` spliced columns from different transactions.
- L6: the final-event snapshot was a hindsight evaluation unit.
- L7: repeated network observations replayed as repeated transactions.
- V1: conflicting chain facts resolved silently.

Production and engineering defects:

- **Container served no model:** paths were resolved from the working
  directory, not `OBSIDIANCHAIN_DATA`. This predates the program; v1 had the
  same bug.
- A pandas copy-on-write read-only array would crash scoring.
- The login endpoint had no brute-force limit.
- A test fixture was never committed (the global `*.csv` ignore).
- An undefined `Path` in `console/runs.py`.
- The run page carried four misleading claims: toggles, a sensitivity value,
  a model-family ("Ensemble") selector and an analysis profile, none of them
  honoured; "Isolation Forest" where the code uses MAD; and "cryptographically
  signed" where artifacts are hashed.
- Uploaded-run alerts were never shown.
- Drift alerts fired on every window; the monitor gave no information.
- ps_native_v4's lineage could not be attested.

## 2. What was changed

- `contracts/`: capture contract (quarantine, never coerce) and feature
  catalog / feature contract enforced before scoring.
- Engine: causal `event_order`, simultaneous-event groups, whole-row
  snapshots, group G (upstream flow). Schema /5.
- Evaluation: `ml/evaluation.py` (every metric, slices, CIs, trend);
  protocol B as the primary unit.
- MLOps:
  - `ml/registry.py`: immutable versions, champion / candidate / fallback,
    reasoned role changes, rollback, attestation, holdout records;
  - serving only through the registry, with an exact schema check;
  - per-run `predictions.parquet` audit log, shadow scoring, monitoring
    alerts, system metrics;
  - delayed-label `model health` loop and model CLI.
- Governance: production gate spec (committed before the holdout), gate
  runner, ADR 0003 holdout exception, one-shot locked holdout script, results
  register.
- Security: login throttle, opt-in Secure cookie, registry-verified pickle
  loading, `pip-audit`.
- Product: case-scoped run-results endpoint and panel, with evidence classes
  and result-type labels; misleading controls and claims removed.
- CI (`.github/workflows/ci.yml`): tests on Python 3.11, lint, frontend,
  model integrity, dependency audit.

## 3. What was rejected

| Candidate | Why rejected |
|---|---|
| grid winners in exp19, exp21, exp22 | lost the confirmation worst fold |
| three-model rank-average ensemble | lost the confirmation worst fold, and was batch-dependent |
| 3-per-transaction queue cap | lowered R@500 0.523 -> 0.441 |
| transaction weighting | failed the stability rule |
| per-step rank features | lower mean |
| absolute PSI drift alerts | fire on every window |
| development-baseline drift alerts as a regime-change detector | missed t43 and t45 |
| embedding threshold 0.99 | 0 of 12 correct |

## 4. What was invalidated

See `docs/results_register.md`: v2 0.619 / 0.648, exp01-exp18 absolute values,
exp20 0.771, CORE 0.802, the v3 scorecard 0.713, embedding "80% precision",
the doc 17 fee claim, and the ps_native_v4 artifact.

## 5. What was researched

- Temporal validation with pre-registered tune/confirm splits.
- Point-in-time features under surrogate timestamps.
- Concept drift versus covariate drift: input monitors cannot detect a change
  in P(y|x), and delayed labels are required.
- The published Elliptic post-t43 regime change (Weber et al., 2019).
- PSI reading conventions.
- Champion / candidate / fallback registry semantics.

Decisions are recorded in ADR 0002, ADR 0003 and docs 19-20.

## 6. What was experimentally verified

| Exp | Finding |
|---|---|
| exp13-exp18 | historical (pre-L1 data) |
| exp19-exp21 | selection on intermediate data, re-done in exp22 |
| exp22 | protocol B: configuration kept; group G +0.047 nAP (p = 0.013) |
| exp23 | drift monitors cannot see the holdout collapse (negative result) |
| exp14 / stacker | seed propagation +0.054 nAP out of sample (p = 0.22, not significant) |

The placebo gives nAP 0.03, and training is byte-for-byte reproducible.

## 7-8. Final model and schema

LightGBM (300 trees, 31 leaves, min_child 50, 16-step window), 31 features,
`ps_native_features/5` (`docs/feature_catalog.md`), with Platt display
calibration and TreeSHAP explanations (`docs/model_card.md`).

## 9. Evaluation method

Protocol B: addresses new in a window, snapshotted at the window end, never
in training. 12 rolling folds: 1-6 tune, 7-12 confirm. Holdout t42-49 opened
once. Tests paired and Holm-corrected; gate thresholds fixed before the
holdout.

## 10-12. Temporal scorecard, worst cases, calibration

| | DEV (1-6) | CONF (7-12) | HOLDOUT |
|---|---|---|---|
| nAP | 0.820 (worst 0.477) | 0.796 (worst 0.553) | **0.548** |
| ROC-AUC | 0.965 | 0.954 | 0.946 |
| address P@100 | 0.98 (worst 0.91) | 0.96 (worst 0.76) | 1.00 pooled; per step 0.02-1.00 |
| R-precision | 0.773 | 0.746 | 0.510 |
| ceiling-normalised R@500 | 0.908 | 0.830 | 0.950 |
| ECE | 0.030 | 0.023 | 0.010 |
| calibration slope | 1.24 | 1.00 | 1.04 |

- Fold nAP sd 0.176; no temporal trend (p = 0.84).
- Holdout per first-seen step: t42 0.70, **t43 0.14**, t44 0.69, **t45 0.02**,
  t46 0.98, **t47 0.32**, t48 0.48, t49 0.72.
- Weak slices, holdout nAP: receive-only 0.17, high-value 0.14, first-seen 0.43.

## 13. Holdout result

nAP 0.548, AUC 0.946, P@100 1.00, ECE 0.010, 54,335 addresses, 2,518
positive. Production gate criterion 18: **PASS** (nAP >= 0.503 required).
The result is locked: sha256 `3e9e2f4b18a7a7adc03b7aef17e0d3e46d52a0f10593be8549f19984b028df93`.

## 14-16. Leakage, reproducibility, batch invariance

- Leakage: `tests/test_leakage_audit.py`, 12 tests, pass. Placebo nAP 0.03.
- Reproducibility: production training matches exp22 per fold to 0.0
  difference. v5 `model.joblib` is byte-identical to v4, trained separately.
  Two runs give identical prediction logs.
- Batch invariance: the score is invariant, tested. Severity and propagation
  are capture-relative by design, documented.

## 17-20. Security, performance, monitoring, rollback

- **Security:** `docs/security.md`; `pip-audit` found no known vulnerabilities.
- **Performance:** single-row p95 0.59 ms; about 460,000 rows/s at 10k-row
  batches; a 7,303-transaction capture runs end to end in about 24 s;
  artifact 1.05 MB.
- **Monitoring:**
  - covered: schema and feature contracts, per-feature PSI, unseen
    missingness, score PSI, a development-relative drift reading, cold-start
    share, serving role, capture quarantine, stage timings, memory;
  - not covered by input monitoring: concept drift, which needs the
    delayed-label loop;
  - not implemented: live p50/p95/p99 over time, queue depth, and CPU as time
    series. This is batch software without a metrics service.
- **Rollback:** `obsidianchain model rollback`, tested. The fallback on the
  same schema fails the champion gate and is an emergency mode.

## 21-22. Tests and CI

- Backend: 1,916 passed, 4 skipped on the full local suite. On a clean
  checkout: 1,509 passed, 162 skipped, and 243 deselected as
  `requires_local_artifacts`.
- Frontend: 103 passed.
- CI: workflow written, and each job's commands verified locally (tests,
  ruff, integrity, pip-audit, frontend). **Not yet run on GitHub**, because
  nothing has been pushed.

## 23. Production-readiness status

**Approved for MONITORED deployment**: champion ps_native_v5, production gate
18/18 PASS. **Not "production-proven"**:

- no SHADOW or PRODUCTION result exists;
- the holdout shows per-window collapse that no current monitor detects;
- the delayed-label review loop is mandatory (`docs/runbook.md` §4).

## 24-25. Remaining limitations and risks

- Regime change after t43 is undetectable before labels arrive.
- Weak slices: receive-only, high-value and first-seen addresses.
- Elliptic++ surrogate timestamps and even-split per-address amounts are a
  train/serve difference.
- A missing fee is scored as zero.
- The labels are a biased subset.
- The rule-line fusion weights are policy, not learned.
- Embeddings are a weak hint (12% precision).
- joblib is pickle: integrity-checked, but trusted once registered.
- The login throttle is in-process.
- The holdout t42-49 is spent for any future model.
- The run-results page was not checked visually in a browser: that needs a
  password typed into the login form, which was not done. Component and API
  tests cover it.

## 26. Recommended future research

1. A newer labelled period, as the next holdout and for regime-change study.
2. Label-efficient adaptation after a detected regime change, triggered by
   the delayed-label loop.
3. Features for receive-only and high-value addresses.
4. A labelled cluster-level set, so fusion weights can be learned.
5. Real-timestamp captures, to remove the surrogate-time difference.
6. A protocol-A holdout reading, if ever needed, under its own ADR.

## 27-29. Identifiers and reproduction

- Champion `ps_native_v5`: model sha256 in `data/models/ps_native/registry.json`,
  attested source commit `5431241a4d2beea200aa71ea685ef779d1f58c68`,
  schema `ps_native_features/5`, contracts `capture_contract/1` and
  `feature_contract/1`.
- Fallback `ps_native_v5_fallback_no_g`.
- Reproduction commands: `docs/runbook.md` §7. The final commit id is in
  `git log` on this branch.
