# 12 — Pipeline Gap Analysis (2026-09-23)

Phase 13. What is missing for a serious investigative platform, and which
gaps are ML-result-relevant vs. product/engineering gaps.

## Gaps that materially affect the ML result (this program's direct concern)

| Gap | Evidence | Materially affects |
|---|---|---|
| **Explanation fidelity** | `10_explanation_audit.md`: 33.9% direction agreement with true TreeSHAP | Every ML result's investigative usefulness — an investigator cannot currently trust the "why" |
| **Severity-band threshold selection uses in-sample precision** | `09_alert_policy_analysis.md`: v1's own `calibration.json` numbers match the target exactly | Alert policy, not ranking — but is the layer investigators actually see |
| **Calibration tie-collapse** | `08_calibration_analysis.md`: precision@50 bound width 0.26 | Any reported precision number at low-K budgets |
| **Cold-start blind spot** | `07_error_analysis.md`: 511/565 false negatives, median 0 prior txs | Recall on exactly the addresses an investigator most wants an early warning on |
| **Network-layer data absent from PS-native dataset** | `01_repo_audit.md`, `02_data_audit.md` | Blocks Group F/G entirely — not a modeling failure, a data pipeline gap |
| **Production Group E feature code is a stub** (new, cycle 3) | `16_synthetic_world_investigation.md` / exp11: `observer_diversity`, `peer_count` hardcoded to 1.0 regardless of real underlying variation (confirmed against real multi-observer data, which has peer_count range 1–8, mean 5.67) | Even if network data were joined into the PS-native dataset today, the current feature code could not use it meaningfully — a second, independent blocker on top of the data-absence gap above |
| **Boundary-spanning address exclusion, unquantified** | `02_data_audit.md` | Unknown-size selection bias in both training and evaluation |
| **v1/v2 schema mismatch blocking inference** | `01_repo_audit.md` §5, HANDOFF.md | `MODEL_UNAVAILABLE_FOR_SCHEMA` on every uploaded-dataset run — the standing product blocker this program's Phase 15/16 exists to resolve |

## Gaps that are product/engineering, not ML-result-relevant (out of this program's scope, listed for completeness per RULE 7)

- **Data validation / schema validation**: exists for the trained artifact
  (`ps_model.py` refuses a feature mismatch) but not as a general
  upload-time contract beyond that. Not evaluated here.
- **Feature drift / model drift monitoring**: none observed in the
  codebase (`grep` for drift-monitoring code in `src/obsidianchain/` during
  `01_repo_audit.md` found none) — this is an operational-deployment gap,
  not something a research program produces without a live monitoring
  target to validate against.
- **Model registry / experiment tracking beyond this program's own
  `experiments.jsonl`**: `data/models/ps_native/v1/manifest.json` is a
  single-artifact manifest, not a registry across versions. Sufficient for
  the current one-frozen-artifact reality; would need design work before a
  v2 promotion introduces a second version to track.
- **Human feedback / analyst feedback loop, case management, evidence
  lineage**: `console/` already has casework, investigations, reports,
  audit tables (`01_repo_audit.md` §11) — this exists at the product layer;
  whether ML-model feedback (e.g. an analyst marking a prediction wrong)
  flows back into retraining is not evaluated here — no evidence of such a
  loop in the code read during this audit, flagged but not investigated
  further (out of ML-research scope).
- **Security / offline deployment / deterministic inference**: covered
  extensively by the project's own Docker/vendoring/provenance-gate
  mechanisms (`01_repo_audit.md` §11, §12) — already strong, not a gap this
  program identifies new issues in.
- **Rollback**: not evaluated — no promotion has happened yet to roll back
  from (v1 is frozen, v2 does not exist yet).

## Confidence semantics — a specific, concrete gap worth naming

`03_label_audit.md`: labels are static per-address ("ever illicit"), but
predictions are per-transaction-snapshot. Nothing in `ps_model.py`'s output
or the alert policy currently distinguishes "this address has a long
illicit-looking history" from "this is the address's first transaction and
it already looks illicit" — despite `07_error_analysis.md` showing these
are mechanistically very different cases for the model (one has real Group
B signal, the other doesn't). This is a genuine confidence-semantics gap:
the system does not currently tell an investigator *how much track record*
a score is based on.

## Priority ranking (this program's assessment)

1. **Explanation fidelity fix** — cheapest to fix (swap to TreeSHAP for
   LightGBM, already proven exact and dependency-free in
   `10_explanation_audit.md`), highest trust impact.
2. **Alert policy correction** — rank on raw score, recompute severity
   bands on a properly held-out split never used for threshold reporting
   (`09_alert_policy_analysis.md`).
3. **v2 model training + promotion** — resolves the standing
   `MODEL_UNAVAILABLE_FOR_SCHEMA` blocker (Phase 15/16 of this program).
4. **Counterparty-history feature engineering** — addresses the cold-start
   blind spot (`07_error_analysis.md`), the clearest evidenced lever for a
   further ranking-quality gain.
5. **Network-layer dataset regeneration** — needed before Group F/G can be
   evaluated at all; lower priority than 1–4 because there is no current
   evidence it would help (untested, not disconfirmed).
6. **Fix the Group E feature stub** (new, cycle 3) — cheap, mechanical,
   independent of whether a network dataset ever materializes; currently
   any network telemetry that DID reach the pipeline would be discarded
   down to a near-constant signal regardless.
7. **Next-generation synthetic Group F/G benchmark** (new, cycle 3,
   `16_synthetic_world_investigation.md`) — well-scoped (noisy/overlapping
   behavior parameters, a genuinely chain-ambiguous behavior class, larger
   scale for protocol-grade power) but a substantial new-behavior-design
   effort, not a quick follow-up. Lowest priority of the seven: no evidence
   yet that it would change any conclusion, and the existing world instance
   was shown unsuitable for this specific purpose (ceiling effect + low
   power), so building the next one is speculative infrastructure
   investment until a concrete need reasserts itself.
