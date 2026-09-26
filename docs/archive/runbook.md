# Production runbook — PS-native ML

## 1. What serves production

`data/models/ps_native/registry.json` is the only authority.

```bash
python -m obsidianchain model list
python -m obsidianchain model verify ps_native_v5
python scripts/check_model_integrity.py
```

Serving (`pipeline/orchestrator.py`) resolves `champion`, then `fallback`.
Each is verified against its registered SHA-256 and must declare the live
feature schema. If both fail, the run completes with
`MODEL_UNAVAILABLE_FOR_SCHEMA` and a `NO_MODEL_SERVED` alert; it never scores
with an unverified model.

## 2. Reading a run

Each run directory (`data/runs/<run_id>/`) holds:
- `manifest.json`: every artifact hash, stage summaries, `provenance.model_trust`
  and `provenance.monitoring_alerts`
- `alerts.json`: ranked entities with evidence, each item carrying `evidence_class`
- `predictions.parquet`: one audit record per scored address
- `monitoring.json`: drift, system metrics and alerts
- `shadow.json`: only when a candidate is assigned

| Monitoring alert | Meaning | Action |
|---|---|---|
| `NO_MODEL_SERVED` (CRITICAL) | champion and fallback both unloadable | see §5; results carry no model score |
| `FEATURE_CONTRACT_VIOLATION` (CRITICAL) | a feature value is outside the catalog | not scored; inspect the capture |
| `SERVED_BY_FALLBACK` (HIGH) | champion failed verification | §5; the fallback is weaker (holdout nAP 0.521, fails the champion gate) |
| `CAPTURE_QUARANTINE_ABOVE_5PCT` (HIGH) | many transactions violate the capture contract | check the export; quarantined txids are listed in stage 2 |
| `DRIFT_ABNORMAL_VS_DEVELOPMENT` (HIGH) | inputs outside the development baseline | read scores with caution |
| `UNSEEN_MISSINGNESS` (MEDIUM) | a feature is missing that never was in training (e.g. fee) | scored as if zero; treat fee-driven evidence with caution |
| `PERFORMANCE_UNVERIFIED_UNTIL_LABELS` (INFO) | always present | §4 |

## 3. Promotion (candidate -> champion)

1. Train from a **committed** tree, e.g.
   `python research/reproduction/train_ps_production_model.py`, with a new
   `MODEL_VERSION`. Registered versions are immutable.
2. `python -m obsidianchain model register <version> <dir>`
3. `python -m obsidianchain model attest <version> --commit <sha>`
4. `python -m obsidianchain model promote <version> --role candidate --reason "..."`
   It then runs in shadow on every production run (`shadow.json`).
5. `python research/reproduction/production_gate.py <version>`: every criterion
   of `docs/production_gate_spec.json` except the holdout.
6. The holdout is used once per version, under a written exception (ADR 0003
   is the template). A new model needs data NOT in any earlier holdout.
7. `python -m obsidianchain model promote <version> --role champion --gate-report data/models/ps_native/gates/<version>.json --reason "..."`
   This is refused without a PASS report.

## 4. Delayed-label review (mandatory)

Input monitoring cannot see concept drift (Research Experiment 23: the later time
windows t43/t45 collapse read as normal). When case outcomes or designations arrive
for a run's addresses:

```bash
python -m obsidianchain model health data/runs/<run_id> labels.csv   # address,label (1/0)
```

This writes `model_health.json` with P@K, R@K, nAP and calibration on the
labelled subset. **If P@50 on a run with at least 5 labelled positives falls
below 0.5, treat the model as degraded for that period:** raise an incident
(§6) and start a research cycle on the new period. The labelled subset is
biased towards alerted addresses; the report says so.

## 5. Rollback

```bash
python -m obsidianchain model rollback --reason "..."
```

This swaps champion and fallback after verifying the fallback, and records
the change in the registry history. It refuses a fallback on another feature
schema, because that needs the code of that schema too (the error names the
commit). The fallback is an emergency mode, not a peer: it fails the champion
gate (worst fold 0.31).

## 6. Incident response

| Symptom | First checks |
|---|---|
| model not served | `model verify <v>`; `git status data/models/ps_native` (tampering shows as a diff) |
| all runs flag drift | expected with absolute PSI; alerts use the development baseline (`data/models/ps_native/monitoring/`) |
| precision collapse in delayed labels | the known regime-change mode; document the period, do not retune on it, start a new version |
| capture refused | stage 2 `capture_contract` lists the violation kinds; fix the export, never the contract |

Every role change, rollback, attestation and holdout record is appended to
the registry history, with a reason.

## 7. Reproducing the production model

```bash
git checkout 5431241
python research/reproduction/build_ps_dataset.py        # protocol-A parquets (research history)
python research/autoresearch_2026_09_23/scripts/exp22_window_end_protocol.py
python research/reproduction/train_ps_production_model.py   # writes v5; byte-identical model.joblib
FEATURE_SET=no_g python research/reproduction/train_ps_production_model.py
python research/reproduction/train_ps_stacker_v2.py
```

The v5 `model.joblib` reproduced byte-for-byte (it is identical to the
withdrawn v4 trained earlier from the same configuration).
