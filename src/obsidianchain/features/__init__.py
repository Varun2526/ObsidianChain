"""Phase 6 feature construction. Address level, as-of-t, leakage-controlled.

Every module here obeys SPEC 4.1: a feature for address ``a`` observed at
timestep ``t`` is computed only from transactions with ``Time step <= t``.

Nothing in this package reads a label. ``wallets_classes.csv`` and
``txs_classes.csv`` are joined in :mod:`obsidianchain.features.dataset`, once,
after every feature is built - so a feature builder cannot see an outcome even
by accident. ``tests/test_phase6_leakage.py`` asserts this at source level.
"""

from __future__ import annotations
