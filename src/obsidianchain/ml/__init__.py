"""Phase 6 modelling: LightGBM, calibration, thresholds, evaluation.

Nothing here reads a truth file or a network ground-truth directory. The only
labels it sees are the ``y`` column of the Phase 6 dataset, which
:mod:`obsidianchain.features.dataset` joined from ``wallets_classes.csv``
after every feature was built.
"""

from __future__ import annotations
