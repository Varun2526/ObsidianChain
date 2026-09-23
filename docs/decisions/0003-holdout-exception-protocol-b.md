# ADR 0003 — Sealed-holdout exception: one evaluation of ps_native_v5 under protocol B

Date: 2026-09-23. Status: accepted before the holdout was opened.

## Why an exception is needed

The sealed holdout file `data/models/ps_native/datasets/test.parquet`
(MD5 `a15500c94b9808cd42d584ad4b5c3017`) was built with feature schema
`ps_native_features/1`, in the final-event snapshot unit, from a stream
processed in txId order. The candidate `ps_native_v5` declares schema `/5`
and was selected and validated under protocol B (new addresses, snapshot at
window end; exp22). The file therefore cannot be scored by the candidate:
its columns, values and evaluation unit are all different.

## What is done

- `test.parquet` is **not opened, not regenerated and not modified**. Its MD5
  is re-checked before and after the evaluation.
- A protocol-B holdout set is derived from the raw Elliptic++ files for the
  sealed period: addresses first seen in timesteps 42-49, each snapshotted at
  its last event at or before t49, features from the unchanged production
  engine over the full causal stream. Labels: `wallets_classes.csv`, classes
  1 and 2 only.
- The frozen models are scored exactly once, in one run of
  `research/reproduction/evaluate_holdout.py`:
  - `ps_native_v5`, the candidate
  - `ps_native_v5_fallback_no_g`, the registered fallback, declared here in
    advance so the fallback also has a holdout result

  (This ADR first named v4. v4 was never evaluated on the holdout: its
  lineage could not be attested, because its training script changed after
  training. v5 is the same configuration trained from committed code, and
  this ADR was amended before the holdout was opened.)
- `ml/protocol.break_seal(reason)` is called with this ADR's path.
- The script writes a lock file with the SHA-256 of its result and refuses
  to run again for the same model version.

## What cannot change

- Models, calibrators, stacker, features, schema and thresholds are frozen.
  They are verified against the registry before scoring.
- The production-gate thresholds are those in `docs/production_gate_spec.json`
  as committed before this evaluation.
- No hyperparameter, feature, threshold or calibration change may follow from
  the holdout result for these versions. A future change is a new version,
  and it must be evaluated on data this holdout does not include.

## Why this does not permit tuning

The result is used for two things only: the production gate's criterion 18,
and the published scorecard. If the gate fails, the model is not promoted.
Nothing is refitted against it. A later research cycle may use t42-49 as
ordinary development data only once a newer period exists to serve as that
cycle's holdout, and it must say so.

## Relationship to the protocol-A holdout

Protocol A (final-event snapshot) remains the historical unit of exp01-exp21.
No protocol-A holdout result exists for any model. `test.parquet` remains
sealed and can serve a protocol-A evaluation in the future, under its own
written exception.
