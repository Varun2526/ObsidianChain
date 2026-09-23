# 01 — Repository Audit (2026-09-23)

Baseline report per the research program's Phase 1. Grounded in direct
reading of the code and artifacts listed below, plus `HANDOFF.md` (last
updated 2026-09-23, same day) which already contains a maintained audit of
frozen results, invariants and traps — this document defers to it rather than
re-deriving what it already states, and focuses on what HANDOFF.md does not
cover in ML-research detail (exact protocol mechanics, dataset column
inventory, current test/tooling inventory).

## 1. Current pipeline (PS-native scope — the one this program targets)

`pipeline/orchestrator.py` — 17-stage deterministic pipeline. Stage 10 is
"Supervised ML Risk (or MODEL_UNAVAILABLE_FOR_SCHEMA)": the orchestrator
degrades honestly when the frozen model's declared schema doesn't match the
live feature schema (see §5 below) rather than crashing or silently scoring
misaligned columns.

Feature extraction: `pipeline/features_ps.py::PsTemporalFeatureEngine` — a
chronological state machine (`process_records`) that guarantees forward-only
temporal state per its own P0 constraints (see file docstring): information
at t_future never influences a feature at t; labels never enter the feature
matrix; missing network telemetry is NaN, never fabricated as zero.
Cluster-size-as-of-t is tracked via an incremental union-find replayed in
timestamp order (`IncrementalUnionFind`), not the global co-spend clustering
used elsewhere in the repo.

Model: `ml/ps_model.py::PsNativeRiskModel` — loads
`data/models/ps_native/v1/model.joblib`, verifies SHA-256 against
`manifest.json` before use, exposes `.scores()` (calibrated probabilities)
and `.predict_address_features()` (full prediction + explanation objects).

## 2. Current data schema

Source (Elliptic++, gitignored): `AddrTx_edgelist.csv`, `TxAddr_edgelist.csv`,
`txs_features.csv`, `wallets_classes.csv`. Hashed in
`data/models/ps_native/datasets/manifest.json`
(`source_dataset_hashes`).

Timestep→wall-clock surrogate: `t = 1400000000 + (step − 1) × 1209600`
(14-day steps from a fixed epoch) — confirmed in
`research/protocol_2026_09_22/run_protocol.py` and reproduced identically in
this program's `exp01_model_family.py`.

Split boundaries (from the dataset manifest): train = step 1–34, validation =
step 35–41, test = step 42–49 (the sealed holdout).

## 3. Current feature schema — v2, confirmed live

`pipeline/features_ps.py::PS_FEATURE_SCHEMA_VERSION = "ps_native_features/2"`.
24 columns in `CORE_PS_FEATURE_COLUMNS` across four groups actually present:

| Group | Columns | Count |
|---|---|---|
| A_transaction | input_count, output_count, total_input_amount, total_output_amount, fee, fee_ratio, input_amount_mean, output_amount_mean, input_spread, output_spread | 10 |
| B_address_history | n_txs_asof_t, n_sent_asof_t, n_recv_asof_t, btc_sent_total_asof_t, btc_recv_total_asof_t, net_flow_asof_t, mean_fee_ratio_asof_t, active_duration_seconds, tx_velocity_per_hour, gap_since_last_tx | 10 |
| C_graph | unique_counterparties_asof_t, cluster_size_asof_t | 2 |
| D_patterns | is_peeling_candidate, is_mixing_candidate | 2 |
| **E_network** (optional, `include_network=True`) | network_observation_count, observer_diversity, peer_count, asn_count | 4 (**not present in current dataset**, see §9) |

**Confirmed by direct inspection:** `train.parquet` / `validation.parquet`
have exactly 28 columns = 4 metadata (`address, txid, timestamp, y`) + the 24
CORE columns. `include_network=False` at `research/reproduction/build_ps_dataset.py:158`.
No network-layer feature currently reaches the PS-native development dataset.

v1→v2 changes (already executed, per `ml/diagnostics.py` and
`pipeline/features_ps.py` docstrings, and confirmed by the current
`git diff` on `tests/test_ps_feature_correctness.py`, `test_ps_features_leakage.py`,
`test_ps_model.py`):
- Removed as fabricated/dead: `input_amount_max`, `input_amount_std`,
  `output_amount_max`, `output_amount_std` (synthesised by even-division of
  totals — std ≈ 2.8e-14 floating-point noise, max ≡ mean),
  `equal_output_count` (≡ `output_count`), `output_entropy` (≡
  log2(output_count)), `in_degree_asof_t`/`out_degree_asof_t` (exact
  duplicates of `n_sent_asof_t`/`n_recv_asof_t`).
- Added as real replacements: `input_spread`, `output_spread` — computed from
  the transaction's actual min/max/mean summary columns (`in_BTC_min` etc.)
  when present, not synthesised.
- Redefined (not removed): `is_peeling_candidate` (v1 was mathematically
  incapable of firing — `0.5 >= 0.8` — now uses real out_min/out_max),
  `is_mixing_candidate` (v1 was a bare `n_in≥3 and n_out≥3` fan-out rule
  firing on 34% of addresses; v2 requires uniform outputs AND varied inputs).

**Feature health, confirmed live at protocol-lock time**
(`exp01_model_family.py` run): `diagnostics.healthy_features` passes 24/24
declared CORE columns as healthy against the current train+validation frame —
no constants, duplicates or functional dependencies detected in v2, which is
the expected consequence of the fixes above (this is a direct check, not an
inference from the docstrings).

## 4. Current label definition

Binary, from Elliptic++ wallet classes: class 2 → y=0 (licit), class 1 → y=1
(illicit), class 3 (unknown) excluded entirely from the usable frame (dataset
manifest `label_schema`). Development prevalence: train 5.47%, validation
6.26% (measured directly from the parquet files). Unit of analysis is
**address-as-of-transaction-timestamp**, not transaction or global address —
the same address appears once per transaction it participates in, with
features computed from state strictly before that transaction.

## 5. Current model paths (two declared, non-comparable scopes)

Per `docs/decisions/0001-two-production-model-paths.md` and `ml/protocol.py`'s
own `SCOPES` dict:

- **`SCOPE_PHASE6`** — Elliptic++ reference, M0–M4 features, LightGBM +
  isotonic calibration, TreeSHAP, serves frozen alert run `043ea584e99daf99`.
- **`SCOPE_PS_NATIVE`** — this program's target. Frozen artifact
  `data/models/ps_native/v1/`: RandomForest (from `manifest.json`:
  `model_type: RandomForest`), declares `ps_native_features/1` (30 columns).
  **Blocked**: the live pipeline emits `/2` (24 columns); `ps_model.py`
  refuses to score the mismatch rather than silently scoring misaligned
  columns — orchestrator stage 10 reports `MODEL_UNAVAILABLE_FOR_SCHEMA`,
  all 17 stages still complete. This is the standing blocker HANDOFF.md
  calls out as "the one item blocking the product."

`ml/protocol.py::_require_same_scope` enforces the non-comparability
mechanically (`ScopeMismatchError`), not just by convention.

## 6. Current evaluation protocol

Fully documented in `00_research_protocol.md`; summary: 12 rolling-origin
folds (width 2, min train 16) strictly inside t1–41, paired t-test primary,
Holm-Bonferroni across families, sign-flip permutation + (added by this
program) Wilcoxon as robustness checks, MDE 0.164 nAP at 80% power, holdout
sealed via `HoldoutSealError` raised at `Fold` **construction** (not at use)
so a leaking fold cannot be built in the first place — this is the fix for
the "folds reaching the holdout" trap in HANDOFF.md §7.

## 7. Current metrics

`ml/metrics.py` (Phase 6 scope): PR-AUC + its no-skill baseline (never
reported alone), normalised AP, ROC-AUC, precision/recall/F1 at threshold,
Brier, log-loss, precision@k / recall@k for k∈{10,50,100}.

`ml/protocol.py` (PS-native scope, this program's primary metric source):
normalised AP per fold + spread across folds (the spread is treated as a
first-class result, not a footnote), precision@k **bounds** (worst/best over
tie orderings) for k∈{100,200,500} by default.

Accuracy is never computed as a decision criterion in either path.

## 8. Known bugs / already-fixed defects relevant to this program

All from HANDOFF.md §7 (Traps), restated only where directly relevant to ML
research scope:
- **Single-window model selection** (already caught): RandomForest selected
  over LightGBM on one validation window (0.5702 vs 0.5228 PR-AUC) — under
  12 rolling folds the difference reverses sign (−0.019) and is
  statistically indistinguishable. **This program's `exp01` independently
  reproduces this exact finding** (§ below) as its first confirmatory
  result under the locked protocol.
- **Fabricated v1 features** (already fixed, v2 ships): see §3.
- **"1 sd" decision rule** (already replaced): a reasoned-not-measured rule
  had an 8.7% false-positive rate by simulation; replaced by the calibrated
  paired t-test (measured 5.1%), which is what this program uses.
- **Isotonic calibration degrades ranking** (already measured, not yet
  acted on): AP falls (0.2925→0.1642 on the cited comparison), resolution
  collapses ~10,000 distinct scores → ~55. This program's Phase 8
  (calibration/alert-policy analysis) inherits this as a starting finding
  to verify and extend, not rediscover from scratch.
- **`ps_model.py` explanations are not SHAP** (confirmed by direct read,
  §1/§5 of this doc and `ml/ps_model.py:150-187`): contribution =
  `global feature_importances_ × |raw value|`, direction derived from
  whether calibrated probability ≥ 0.5 — not a per-prediction attribution
  method, and direction is calibration-threshold-derived rather than
  feature-derived, so a feature can be reported "increasing risk" purely
  because the *overall* prediction crossed 0.5, independent of that
  feature's actual marginal effect. Phase 9 (explanation audit) targets
  this directly against TreeSHAP.

## 9. Known leakage risks

- **Structural, not observed:** the P0 constraints in
  `features_ps.py` (forward-only state, label-blind matrix) are enforced by
  construction in `PsTemporalFeatureEngine.process_records` (state updated
  strictly after a row's features are emitted) and pinned by
  `tests/test_ps_features_leakage.py` (temporal leakage safeguards A–E,
  including one test that explicitly constructs a "future" transaction and
  asserts it cannot change an earlier feature).
- **Group E network features, if reintroduced:** must go through
  `network/boundary.py` (Phase 2/3 architecture) which is "the only
  sanctioned load path" per HANDOFF invariant 1 — inference sees exactly six
  raw columns (`txid, observer_id, peer_ip, peer_port, peer_asn,
  timestamp_ms`), never truth accessors. Any experiment in this program that
  reintroduces Group E must build it through that boundary, not by joining
  a `*_FOR_EVALUATION_ONLY` truth accessor.
- **M3 network features are not as-of-t** (HANDOFF.md §9, item 5):
  `netfeat.build()` (Phase 6 scope) takes no cutoff, unlike M0/M1/M2.
  Currently benign because M3 contributes ~nothing to the frozen Phase 6
  model, but flagged as a fix-before-retrain item if any future model in
  either scope tries to exploit M3 harder.

## 10. Known synthetic-data artifacts

- All network-layer telemetry in the repo (`network/synthetic.py`,
  `SYNTHETIC_CONTROL` worlds A–E) is synthetic, generated by the same team
  as the analysis — "validates nothing" per HANDOFF §8, demonstrates
  mechanism only.
- The five `make demo` scenarios (`demo/scenarios.py`) are fixtures
  engineered to reach each engine state; namespace-guarded
  (`data/demo/`, `assert_demo_namespace`) and flag-enforced
  (`demo: true`, `provenance: SYNTHETIC_DEMONSTRATION`) so they cannot be
  mistaken for a result.
- PS-native training labels come from real Elliptic++ ground truth (not
  synthetic), but the **timestamp surrogate** (`t = 1400000000 + (step−1)×1209600`)
  is an assumption stated explicitly as such (HANDOFF §9 item 5: "Replace
  `DEFAULT_TX_MEDIAN_MS`... re-fit σ from real transactions" — mainnet
  capture planned for October, not yet available).

## 11. Current production/reference scope boundaries

Enforced in code, not just documentation:
- `api/provenance_gate.py` — refuses to serve any artifact whose sidecar
  does not declare `provenance_type = PRODUCTION` (schema
  `obsidianchain.provenance/2`). A missing sidecar is a refusal, not a
  default.
- `api/boundary.py` — `FORBIDDEN_FIELDS` / truth-column denylist as a second
  layer behind the provenance gate.
- `ml/protocol.py::_require_same_scope` — mechanical refusal to compare
  Phase 6 vs PS-native (§5).
- `console/` — SQLite stores only identifiers (`alert_id`, `run_fingerprint`);
  `tests/test_console_boundary.py` checks the DDL directly so a risk score
  cannot silently gain a second home (HANDOFF invariant 13).

## 12. Current reproducibility state

- `data/models/ps_native/datasets/manifest.json` records generator script,
  source-file SHA-256s, feature schema version, split boundaries, row/positive
  counts and output artifact hashes — a full provenance record for the
  current dev datasets.
- `data/models/ps_native/v1/manifest.json` records the frozen model's own
  SHA-256, training-dataset hash, and metrics summary — but that summary
  (`val_pr_auc: 0.5702`) is the **exact single-window number the protocol
  module was built to supersede** (§8). It remains in the manifest as
  historical record, not as this program's evidence standard.
- `research/README.md` §3 ("Empirical Research Findings & Production
  Consequences") currently states RandomForest was selected for "superior
  Validation PR-AUC (0.5702 vs 0.5228)" as a **production consequence**.
  **This is now known-superseded** by `ml/protocol.py`'s own reproduced
  finding (§8, and `exp01` below): under the correct rolling-fold protocol
  the RF/LightGBM difference is statistically indistinguishable and
  underpowered to resolve. `research/README.md` should be corrected as part
  of this program's Phase 17 documentation-update step; not done yet as of
  this audit.
- Offline build guarantee (`Dockerfile`, `Makefile`): vendored wheels in
  `vendor/linux-{amd64,arm64}/`, `docker build --network none`. **xgboost is
  not vendored** (confirmed: `ModuleNotFoundError` in this environment) —
  candidate family in Phase 5 is limited to what's already vendored
  (scikit-learn 1.9.0, lightgbm 4.7.0) plus scipy 1.17.1; adding xgboost,
  statsmodels or shap means re-vendoring, which HANDOFF.md flags as an open
  decision (§9 item 7), not yet made.
- Test suite: 65 files under `tests/`, spanning clustering, network,
  console/RBAC, API boundary/provenance, ML protocol
  (`test_ml_protocol.py`), PS feature correctness/leakage/schema
  (`test_ps_*.py`), feature health (`test_feature_health.py`), demo honesty
  contract, offline/frontend-offline checks. HANDOFF.md reports 1745 passed /
  4 skipped backend + 101 passed frontend as of its last full run; the full
  suite was not re-run by this program (would need the Docker
  `--network none` harness for a true reproduction). **Update, cycle 3**:
  ran the ML/PS-native-relevant subset directly with pytest (outside Docker)
  as a cheap verification that this program's extensive reading and
  experimentation left the codebase unchanged and passing —
  `test_ps_*.py`, `test_ml_protocol.py`, `test_feature_health.py`,
  `test_phase6_*.py` (excluding the container-only leakage test): **all
  pass**, confirming no production file was inadvertently modified across
  cycles 1–3. The full 1745-test suite (clustering, console, API layers
  outside this program's scope) remains un-re-run — a reasonable, explicitly
  stated scope limit, not an oversight.
- **Uncommitted working-tree state at audit time** (`git status`): `HANDOFF.md`,
  `research/protocol_2026_09_22/protocol_results.json`, and three
  `tests/test_ps_*.py` files are modified but not committed — these are the
  in-progress v1→v2 schema-fix work already described in HANDOFF.md, not
  changes made by this program.

## 13. What this audit does NOT cover yet (explicitly deferred, not skipped)

- Full `make test` re-run (noted above).
- `frontend/` and `console/` API surface audit beyond the boundary/provenance
  mechanisms already covered — deferred to Phase 13 (pipeline gap analysis),
  since it's an operational-platform question more than an ML-research one.
- Security/audit-trail review of `console/audit.py`, `console/rbac.py` beyond
  confirming their existence and the boundary tests — same deferral as above.

These are recorded here so they are not silently dropped, per RULE 7 (never
hide what wasn't done).
