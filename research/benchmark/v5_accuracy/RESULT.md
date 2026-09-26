Accuracy: 96.28%
Precision: 81.12%
Recall: 25.77%
F1: 39.12%
Balanced accuracy: 62.74%
nAP: 0.5475

## Thresholded classification result

- Threshold: calibrated probability `>= 0.50` is positive/risky; `< 0.50` is negative/benign.
- Samples: `54,335`
- Positives: `2,518`
- Negatives: `51,817`
- Positive prevalence: `4.6342%`
- Confusion matrix (actual rows × predicted columns, `[negative, positive]`):

  |  | Predicted negative | Predicted positive |
  | --- | ---: | ---: |
  | Actual negative | 51,666 | 151 |
  | Actual positive | 1,869 | 649 |

## Holdout and inference provenance

- Holdout used: the registered v5 Protocol-B sealed holdout, `t42-49`, defined by `data/models/ps_native/datasets/test.parquet` and its manifest. Evaluation targets are addresses first seen in `t42-49`, each represented by its last event at or before `t49`.
- Labels: `data/raw/wallets_classes.csv`; class `1` is positive/risky, class `2` is negative/benign, and class `3` is excluded.
- Model: `data/models/ps_native/v5/model.joblib`, version `ps_native_v5`, SHA-256 `974d37f22e2f1e7df4e03cb5a13fdb35bf487e43f4d5db3ea3bb7191ebd3e607`.
- Inference: the current causal production feature pipeline (`build_canonical_frame(..., max_step=t49)` then `PsTemporalFeatureEngine().process_records(...)`) and the production model's calibrated probability (`PsNativeRiskModel.raw_scores(...)` followed by `calibrate(...)`). The computed address-level raw-score nAP exactly matches the registered holdout result: `0.5475262986`.
- Calibrated probabilities were finite and within `[0, 1]` (`0.0002187` to `0.9923491`), so the specified `0.50` threshold was valid. No other threshold was evaluated.

## Label separation and training

Labels were not provided to feature generation: causal features were materialized from the transaction stream first, then labels were joined only to the selected holdout address snapshots for evaluation. Earlier transactions were supplied only as causal history required by the production feature engine; no earlier address was included in the scored sample or metrics. No training, retraining, tuning, model replacement, or registry/model/data modification occurred.
