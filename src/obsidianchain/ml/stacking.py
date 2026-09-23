"""The frozen ML + risk-propagation stacker (``data/models/ps_native/v3/stacker.json``).

Two inputs, both monotone: the LightGBM raw score (as a logit) and the
propagated seed risk (``log1p(PROP_SCALE * p)``, so the long tail of tiny
propagated values is spread out). Fitted out-of-fold by
``research/reproduction/train_ps_stacker_v2.py``; evidence in exp14.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PROP_SCALE = 1000.0
DEFAULT_STACKER = Path("data") / "models" / "ps_native" / "v3" / "stacker.json"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_features(model_raw, propagated) -> np.ndarray:
    return np.column_stack([_logit(model_raw), np.log1p(PROP_SCALE * np.asarray(propagated, dtype=float))])


def load_stacker(path: str | Path = DEFAULT_STACKER) -> dict[str, Any] | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def apply_stacker(stacker: dict[str, Any], model_raw, propagated) -> np.ndarray:
    """Probability from the stacked inputs."""
    z = stack_features(model_raw, propagated) @ np.asarray(stacker["coef"]) + stacker["intercept"]
    return 1.0 / (1.0 + np.exp(-z))
