"""LightGBM training, validation-only calibration, severity thresholds.

SPEC 6.3 and 6.3.1. The test split is never consulted by anything here: the
model is fitted on train, the calibrator on validation, and the severity
thresholds on calibrated validation scores. Test data enters only in
:mod:`obsidianchain.ml.experiment`, after all three are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Fixed so a run is reproducible. Not tuned against the test split.
LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 400,
    "random_state": 20260912,
    "n_jobs": -1,
    "verbose": -1,
}

#: SPEC 6.3.1. Precision targets per band.
BAND_TARGETS = (("CRITICAL", 0.90), ("HIGH", 0.75), ("MEDIUM", 0.50))

#: SPEC 6.3.1 step 3. A band defined on fewer than this is not a band.
MIN_SUPPORT = 50

UNPOPULATED = None


@dataclass
class Band:
    name: str
    target: float
    threshold: float | None
    support: int
    precision: float

    @property
    def populated(self) -> bool:
        return self.threshold is not None


@dataclass
class TrainedModel:
    booster: object
    calibrator: object
    features: list[str]
    bands: list[Band] = field(default_factory=list)

    def raw_scores(self, frame: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(frame[self.features])[:, 1]

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated probabilities."""
        raw = self.raw_scores(frame)
        return self.calibrator.predict(raw)

    def severity(self, scores) -> np.ndarray:
        """Band name per score, assigned top-down (SPEC 6.3.1)."""
        out = np.full(len(scores), "LOW", dtype=object)
        assigned = np.zeros(len(scores), dtype=bool)
        for band in self.bands:
            if not band.populated:
                continue
            mask = (~assigned) & (np.asarray(scores) >= band.threshold)
            out[mask] = band.name
            assigned |= mask
        return out


def fit(train: pd.DataFrame, validation: pd.DataFrame,
        features: list[str]) -> TrainedModel:
    """Fit on train, calibrate on validation, derive bands on validation."""
    from lightgbm import LGBMClassifier
    from sklearn.isotonic import IsotonicRegression

    booster = LGBMClassifier(**LGBM_PARAMS)
    booster.fit(train[features], train["y"])

    raw_validation = booster.predict_proba(validation[features])[:, 1]
    # Calibrated on VALIDATION, never on the data the trees were fitted to:
    # a model's own training scores are optimistic and would produce a
    # calibrator that is confidently wrong on anything new.
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_validation, validation["y"].to_numpy())

    calibrated = calibrator.predict(raw_validation)
    bands = select_bands(calibrated, validation["y"].to_numpy())
    assert_band_order(bands)
    return TrainedModel(
        booster=booster, calibrator=calibrator, features=list(features),
        bands=bands,
    )


def select_bands(scores, y_true) -> list[Band]:
    """SPEC 6.3.1, exactly. Deterministic, validation only."""
    return [
        _select_one(name, target, scores, y_true)
        for name, target in BAND_TARGETS
    ]


def _select_one(name: str, target: float, scores, y_true) -> Band:
    """Lowest threshold whose every higher threshold also meets the target.

    The monotone-suffix constraint (step 2) is the important part. Precision
    is not monotone in the threshold, so the first point where it crosses the
    target is often a noisy local maximum that will not hold on new data.
    Requiring the target to hold for the whole suffix picks a threshold that
    is stable rather than lucky.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-scores, kind="stable")
    ranked = y_true[order]
    cumulative = np.cumsum(ranked)
    support = np.arange(1, len(ranked) + 1)
    precision = cumulative / support
    running_min = np.minimum.accumulate(precision)

    feasible = (running_min >= target) & (support >= MIN_SUPPORT)
    if not feasible.any():
        return Band(name=name, target=target, threshold=UNPOPULATED,
                    support=0, precision=float("nan"))
    last = int(np.flatnonzero(feasible)[-1])
    return Band(
        name=name, target=target, threshold=float(scores[order][last]),
        support=int(support[last]), precision=float(precision[last]),
    )


def assert_band_order(bands: list[Band]) -> None:
    """CRITICAL >= HIGH >= MEDIUM, or fail loudly.

    Guaranteed by the monotone-suffix rule, since a higher precision target
    can only shrink the feasible set. Asserted anyway: inverted bands would
    silently mislabel every alert, and a guarantee nobody checks is a hope.
    """
    populated = [b for b in bands if b.populated]
    for earlier, later in zip(populated, populated[1:]):
        if earlier.threshold < later.threshold:
            raise RuntimeError(
                f"severity bands are inverted: {earlier.name} threshold "
                f"{earlier.threshold:.6f} is below {later.name} "
                f"{later.threshold:.6f}. The monotone-suffix rule should make "
                f"this impossible; the selection is wrong."
            )


def shap_contributions(model: TrainedModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP contributions, per feature, per row.

    Uses LightGBM's own ``pred_contrib``, which implements TreeSHAP inside the
    booster. The standalone ``shap`` package is not vendored and this build is
    offline; the algorithm is the same one, so nothing is given up.

    Contributions are in raw margin space, not calibrated probability. That is
    stated wherever they are rendered: a contribution of +0.27 is a shift in
    log-odds, not 27 percentage points of risk.
    """
    values = model.booster.booster_.predict(
        frame[model.features], pred_contrib=True
    )
    columns = list(model.features) + ["base_value"]
    return pd.DataFrame(values, columns=columns, index=frame.index)
