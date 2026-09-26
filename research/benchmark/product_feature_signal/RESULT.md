# Current Product Feature/Model Design on SIGNAL

## Result

- **Mean address-level nAP:** `0.9504`
- **Sample standard deviation across 12 folds:** `0.0312`
- **exp15 research nAP:** `0.9460`
- **Difference from exp15:** `+0.0044`

| Fold | Train addresses | Test addresses | Test prevalence | Address-level nAP |
| --- | ---: | ---: | ---: | ---: |
| t<=16 -> t17-18 | 3099 | 522 | 0.2146 | 0.9589 |
| t<=18 -> t19-20 | 3621 | 548 | 0.2318 | 0.9441 |
| t<=20 -> t21-22 | 4169 | 542 | 0.3081 | 0.9576 |
| t<=22 -> t23-24 | 4711 | 554 | 0.3736 | 0.9808 |
| t<=24 -> t25-26 | 5265 | 377 | 0.2414 | 0.9653 |
| t<=26 -> t27-28 | 5642 | 514 | 0.2082 | 0.9210 |
| t<=28 -> t29-30 | 6156 | 614 | 0.3111 | 0.9843 |
| t<=30 -> t31-32 | 6770 | 477 | 0.2411 | 0.9853 |
| t<=32 -> t33-34 | 7247 | 589 | 0.2411 | 0.8740 |
| t<=34 -> t35-36 | 7836 | 1277 | 0.2052 | 0.9399 |
| t<=36 -> t37-38 | 9113 | 1970 | 0.2147 | 0.9370 |
| t<=38 -> t39-40 | 11083 | 2382 | 0.3056 | 0.9569 |

## Exact feature schema

- `ps_native_features/5`
- `31` features: the current `CORE_PS_FEATURE_COLUMNS` (A/B/C/D/F/G) from `src/obsidianchain/pipeline/features_ps.py`.
- Feature extraction: current serving path, `extract_ps_features(capture)` then `last_snapshot_per_address(...)`.

## Exact temporary research model configuration

LightGBM, copied exactly from the current `ps_native_v5` training configuration in `research/reproduction/train_ps_production_model.py`:

- `n_estimators`: `300`
- `learning_rate`: `0.05`
- `num_leaves`: `31`
- `min_child_samples`: `50`
- `subsample`: `0.8`
- `subsample_freq`: `1`
- `colsample_bytree`: `0.8`
- `random_state`: `20260919`
- `verbose`: `-1`
- `n_jobs`: `-1`

Each fold creates one in-memory model and discards it after scoring. The frozen `data/models/ps_native/v5/model.joblib` artifact and the model registry are not loaded, written, or changed.

## Exact folds and metric

- `obsidianchain.ml.protocol.rolling_origin_folds()` — 12 expanding, time-ordered folds: train through `t16`, `t18`, …, `t38`; evaluate `t17-18`, `t19-20`, …, `t39-40`.
- Dataset: `data/synthetic_world_v2/signal/capture.csv`.
- Evaluation labels: `data/synthetic_world_v2/signal/world_truth/labels.csv`.
- Metric: `obsidianchain.ml.protocol.normalised_average_precision` on address-level raw LightGBM probabilities, matching exp15.

## Label-separation confirmation

Features are fully generated from the capture through the current product feature pipeline before `labels.csv` is read or joined. For every fold, `fit(...)` receives only that fold's training features and training labels; test labels are used only by the nAP calculation after test scores are produced. No future-fold row or test label is used during training.

## Conclusion

**A. Current product design reproduces research-level performance.**
