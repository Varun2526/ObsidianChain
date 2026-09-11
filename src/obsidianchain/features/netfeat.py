"""M3 - network features from the Phase 2/3 separation oracle.

SPEC 3.2 M3. These are behavioural summaries of SYNTHETIC announcement
observations. They are **not** ownership claims: the network layer emits
cannot-link only, has no mechanism for asserting sameness, and nothing here
maps an IP to a wallet. HANDOFF invariants 1, 5 and 7.

The oracle is loaded through :func:`obsidianchain.network.separation.build_oracle`,
which reads exclusively via ``network/boundary.py`` - the only sanctioned load
path - so this module cannot reach ground truth even by accident.

Reach, measured
---------------
Of 284,709 clustered addresses, 17.4% carry any pooled observation, 2.3%
reach pooled >= 5, and 1.64% reach the production minimum of 25. Most of
these columns are therefore NULL for most addresses. That is the honest
shape of the data and is why P1 predicts a small aggregate contribution -
though sparsity motivates that prediction rather than implying it (SPEC 6.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

M3_COLUMNS = [
    "net_pooled_observations", "net_has_evidence",
    "net_observer_coverage", "net_observers_present",
    "net_arrival_dispersion_mean", "net_arrival_dispersion_max",
    "net_reaches_production_minimum",
]


def build(data_root, incidence, config=None) -> pd.DataFrame:
    """One row per address code of the incidence factorisation.

    The oracle keys its statistics by the Phase 1 address code, which is a
    different factorisation from this package's, so results are mapped back
    through the address strings rather than by assuming the codes align.
    """
    from obsidianchain.io import elliptic
    from obsidianchain.network import separation

    config = config or separation.SeparationConfig()
    graph = elliptic.load_cospend_graph(data_root, keep_labels=True)
    oracle = separation.build_oracle(
        graph, processed_root=data_root / "processed", data_root=data_root,
        config=config,
    )

    index = pd.Index(sorted(incidence.frame["code"].unique()), name="code")
    phase1_of_string = pd.Series(
        np.arange(graph.n_addresses, dtype=np.int64), index=graph.addresses
    )
    mapped = phase1_of_string.reindex(incidence.addresses[index.to_numpy()])

    rows = []
    minimum = config.min_pooled_observations
    for phase1_code in mapped.to_numpy():
        if pd.isna(phase1_code):
            rows.append((0, False, np.nan, 0, np.nan, np.nan, False))
            continue
        # The public accessor, not the oracle's statistics dict. Phase 4.1
        # reserves direct access for the pooling implementation so there
        # stays exactly one of it; this module only summarises what has
        # already been pooled and never merges two groups.
        stats = oracle.stats_for(int(phase1_code))
        if stats is None or stats.count == 0:
            rows.append((0, False, np.nan, 0, np.nan, np.nan, False))
            continue
        present = np.asarray(stats.dim_count) > 0
        n_present = int(present.sum())
        dispersion = _dispersion(stats)
        # A dimension needs two observations before it has a variance, so an
        # address can have evidence on several observers and still yield an
        # all-NaN dispersion. That is "not estimable", not zero, and the
        # guard keeps numpy from warning about a slice it correctly cannot
        # summarise.
        estimable = np.isfinite(dispersion).any()
        rows.append((
            int(stats.count), True,
            n_present / max(len(present), 1), n_present,
            float(np.nanmean(dispersion)) if estimable else np.nan,
            float(np.nanmax(dispersion)) if estimable else np.nan,
            bool(stats.count >= minimum),
        ))

    out = pd.DataFrame(rows, index=index, columns=M3_COLUMNS)
    out["net_has_evidence"] = out["net_has_evidence"].astype(bool)
    out["net_reaches_production_minimum"] = (
        out["net_reaches_production_minimum"].astype(bool)
    )
    return out


def _dispersion(stats) -> np.ndarray:
    """Per-observer standard deviation of arrival offsets.

    Computed from the same mergeable sufficient statistics the oracle carries,
    so this is a summary of what the engine already pooled and not a second
    pass over the observations. A dimension with fewer than two observations
    has no variance to estimate and yields NaN rather than zero.
    """
    count = np.asarray(stats.dim_count, dtype=float)
    total = np.asarray(stats.dim_sum, dtype=float)
    sumsq = np.asarray(stats.dim_sumsq, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
        variance = np.where(count > 1, sumsq / count - mean**2, np.nan)
        variance = np.where(variance > 0, variance, np.nan)
    return np.sqrt(variance)
