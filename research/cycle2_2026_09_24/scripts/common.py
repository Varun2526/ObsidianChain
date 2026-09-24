"""Shared data layer for improvement cycle 2 (ADR 0004).

Two event-level caches, built by the unchanged production feature engine
over the causal Elliptic++ stream (the same path as exp22 and the ADR 0003
holdout):

* ``development`` - timesteps 1-41 only. Every selection decision in this
  cycle reads this and nothing else.
* ``full`` - timesteps 1-49, for the post-hoc DIAGNOSIS of v5's published
  holdout failure (ADR 0004 rule 2) and for the single reused-holdout veto
  (rule 4). Any number computed on t42-49 from it is REUSED_HOLDOUT.

Evaluation unit (protocol B): a fold's training set is addresses first seen
in (te - W, te], snapshotted at their last event <= te; its evaluation set is
addresses first seen in the fold window, snapshotted at the window end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CYCLE = ROOT / "research" / "cycle2_2026_09_24"
RESULTS = CYCLE / "results"
sys.path.insert(0, str(ROOT / "research" / "reproduction"))
sys.path.insert(0, str(ROOT / "src"))

from build_ps_dataset import build_canonical_frame  # noqa: E402
from obsidianchain.ml import protocol  # noqa: E402
from obsidianchain.pipeline.features_ps import (  # noqa: E402
    CORE_PS_FEATURE_COLUMNS, PsTemporalFeatureEngine,
)

CORE = list(CORE_PS_FEATURE_COLUMNS)
FOLDS = protocol.rolling_origin_folds()
TUNE, CONFIRM = FOLDS[:6], FOLDS[6:]
FULL_CACHE = RESULTS / "events_full_t1_49.parquet"
DEV_CACHE = ROOT / "research/autoresearch_2026_09_23/results/exp22_event_features.parquet"
HOLDOUT_LO, HOLDOUT_HI = 42, 49


def _labelled(feats: pd.DataFrame) -> pd.DataFrame:
    labels = pd.read_csv(ROOT / "data" / "raw" / "wallets_classes.csv").set_index("address")["class"]
    feats = feats.assign(y=feats.address.map(labels))
    feats = feats[feats.y.isin([1, 2])].copy()
    feats["y"] = (feats.y == 1).astype(np.int8)
    feats["_first"] = feats.groupby("address")._step.transform("min")
    return feats


def events_full() -> pd.DataFrame:
    """t1-49. Diagnosis and the reused-holdout veto only (ADR 0004)."""
    if FULL_CACHE.is_file():
        return pd.read_parquet(FULL_CACHE)
    frame = build_canonical_frame(ROOT / "data" / "raw")
    feats = PsTemporalFeatureEngine().process_records(frame)
    feats["_step"] = feats.txid.map(dict(zip(frame.txid, frame["_step"]))).astype(int)
    feats["_seq"] = np.arange(len(feats))
    feats = _labelled(feats)
    feats.to_parquet(FULL_CACHE, index=False)
    return feats


def events_dev() -> pd.DataFrame:
    """t1-41, the only data selection may use. Equal to the exp22 cache."""
    ev = pd.read_parquet(DEV_CACHE)
    return ev[(ev._first <= protocol.DEVELOPMENT_END) & (ev._step <= protocol.DEVELOPMENT_END)]


def snapshot(ev: pd.DataFrame, first_lo: int, first_hi: int, as_of: int) -> pd.DataFrame:
    rows = ev[(ev._first >= first_lo) & (ev._first <= first_hi) & (ev._step <= as_of)]
    return rows.sort_values("_seq").drop_duplicates("address", keep="last")


def fold_sets(ev: pd.DataFrame, fold, window: int = 16):
    te = fold.train_end
    return (snapshot(ev, te - window + 1, te, te),
            snapshot(ev, fold.eval_start, fold.eval_end, fold.eval_end))


def slice_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """The scorecard's own slice definitions (ml/evaluation.standard_slices).

    Note what they measure: ``first_seen`` is "no earlier transaction as of
    the snapshot event", ``high_volume_q4`` is the snapshot TRANSACTION's
    input amount, ``receiver_only`` is the address's role in that event.
    They describe the snapshot event, not the address's lifetime totals.
    """
    from obsidianchain.ml import evaluation
    return evaluation.standard_slices(frame)
