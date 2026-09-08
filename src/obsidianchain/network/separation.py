"""Group-level separation evidence: network grounds for refusing a merge.

CANNOT-LINK ONLY
----------------
This module emits one kind of claim: "these two groups were not broadcast
from the same machine, so do not merge them." It never emits must-link, and
that is not an oversight. A shared origin does not imply a shared entity -
one Electrum server broadcasts for tens of thousands of unrelated users, and
treating co-origination as co-ownership would fabricate exactly the
super-cluster this work exists to prevent. The asymmetry is the design.

WHY GROUPS AND NOT TRANSACTION PAIRS
------------------------------------
The Phase 2 audit measured per-transaction separability at **1.003** - four
independent metrics agreed, so it is a property of the data rather than of a
chosen distance. A model that compares two single arrival vectors and returns
P(same origin) cannot work here, and its failure would measure the
propagation sigma rather than anything about the method.

The same audit measured per-origin centroids at **6.563 standard errors**
apart. The signal is entirely in the aggregate.

So evidence is computed between candidate *groups*. For a proposed merge of
clusters A and B: pool every usable transaction touching A, pool those
touching B, and compare the two centroids scaled by their standard errors.
Confidence rises with the number of pooled observations because the standard
error falls as their square root.

This makes the repetition requirement structural rather than bolted on. A
constraint cannot fire on a single observation because a single observation
cannot produce a pooled count above the minimum - there is no code path that
would allow it.

BELOW THE MINIMUM, THE ANSWER IS "NO EVIDENCE"
---------------------------------------------
Not "weak evidence", not a low confidence score. A group with four pooled
observations is not slightly separated from another with three; the question
is unanswerable at that sample size and the honest return is abstention. A
confidence number would invite a threshold sweep that finds signal in noise.

WHAT THIS COSTS ON THE FROZEN DATASET
-------------------------------------
Measured, not assumed: only about 0.9% of baseline components hold twenty or
more poolable transactions, and with sigma = 1.15 several hundred per side
are needed before a real centroid difference clears its own noise. So
constraints will fire on a small number of large components and abstain
everywhere else. That is the conservative behaviour asked for, but it means
the mechanism is demonstrated on few merges, and the count of evaluable
merges must be reported alongside any result.

BOUNDARY
--------
Everything is loaded through :mod:`obsidianchain.network.boundary`. Nothing
here reads ``processed/network_truth/``. Transactions classified NO_EVIDENCE
never reach a statistic: they are filtered at load, and
``assert_no_constraint_permitted`` re-checks it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from obsidianchain.network import arrivals, boundary


#: Share of a network dataset's transactions that must exist in the chain
#: before the pair is considered the same experiment. A correct pairing sits
#: at 1.0; a mismatched one measured 0.00095 (57 coincidentally equal integer
#: ids out of 60,000), which is worse than zero because those 57 are
#: different transactions that happen to share a number.
MIN_TXID_OVERLAP = 0.5


class DatasetMismatchError(RuntimeError):
    """Raised when a chain and a network dataset are not the same experiment.

    Lives here rather than in the evaluation harness because the guard has to
    fire on the *production* fusion path. It moved after an audit found the
    check was only ever called from ``eval/phase33.py``, leaving
    ``run --mode fused``, ``fusion-summary`` and ``evidence-funnel``
    unguarded - the three commands most likely to be pointed at a real
    capture paired with the wrong chain.
    """


def assert_txid_overlap(
    observed_txids,
    chain_txids,
    chain_description: str,
    network_description: str,
) -> float:
    """The compatibility rule, applied to two already-loaded id sets.

    Split out from :func:`assert_datasets_compatible` so that
    :func:`build_oracle` can apply the identical rule to the frames it has
    just read, without opening either file a second time. One rule, one
    implementation, two entry points.

    Returns the resolved fraction.
    """
    observed = set(observed_txids)
    if not observed:
        raise DatasetMismatchError(
            f"network dataset {network_description} has no transactions"
        )

    chain = set(chain_txids)
    resolved = len(observed & chain) / len(observed)

    if resolved < MIN_TXID_OVERLAP:
        raise DatasetMismatchError(
            f"chain {chain_description} and network dataset "
            f"{network_description} are not the same experiment: only "
            f"{resolved:.4%} of {len(observed):,} observed transactions exist "
            f"in the chain's {len(chain):,}. Pairing them would produce an "
            f"oracle with no evidence and a silent 100% abstention. Point "
            f"--chain-root at the blockchain this network data was generated "
            f"over."
        )
    return resolved


def assert_datasets_compatible(
    chain_root: Path, network_root: Path, world: str | None = None
) -> float:
    """Fail loudly when a chain and a network dataset do not belong together.

    Without this the failure is silent, not loud. :func:`build_oracle` maps
    observations onto the chain and drops whatever does not resolve, so an
    incompatible pair yields an oracle with no statistics, an abstention rate
    of 100%, and a report that looks like a clean negative result. That is
    the same shape of failure as an evaluation that cannot come out any way
    but "pass", and it has bitten this project twice already.

    Returns the resolved fraction so callers can report it.
    """
    from obsidianchain.io import elliptic

    observations = boundary.load_observations(
        Path(network_root) / "processed", world=world
    )
    return assert_txid_overlap(
        observations["txid"].unique().tolist(),
        elliptic.load_input_edges(chain_root)["txId"].unique().tolist(),
        chain_description=str(chain_root),
        network_description=(
            f"{network_root} regime {world}" if world is not None
            else str(network_root)
        ),
    )


class Verdict(str, Enum):
    """The three answers this module can give. Note the absent fourth."""

    SEPARATED = "SEPARATED"
    """Groups differ enough to refuse the merge: a cannot-link."""

    NOT_SEPARATED = "NOT_SEPARATED"
    """Enough data to look, and no meaningful difference found."""

    NO_EVIDENCE = "NO_EVIDENCE"
    """Too few pooled observations to answer. Abstention, not weak evidence."""

    # There is deliberately no SAME_ORIGIN / MUST_LINK verdict.


@dataclass(frozen=True)
class SeparationConfig:
    """Thresholds for declaring two groups separated.

    Two gates must both pass, and neither is redundant. Significance alone
    is not enough: with thousands of pooled observations a difference of no
    practical size becomes statistically certain, and blocking merges on that
    basis would be noise-chasing dressed up in a p-value. Effect size alone
    is not enough either: a large apparent difference between two groups of
    thirty transactions is mostly sampling error.
    """

    min_pooled_observations: int = 25
    """Pooled usable transactions required on EACH side before answering."""

    min_observer_observations: int = 5
    """Per-observer minimum before that observer contributes a dimension."""

    alpha: float = 1e-4
    """Significance floor. Deliberately strict - a false cannot-link keeps a
    correct merge from ever happening, and unlike a false merge it leaves no
    trace to find later."""

    min_effect: float = 0.05
    """Root-mean-square centroid difference floor, in unit-normalised
    arrival-offset units. Guards against significant-but-trivial."""


@dataclass
class SeparationEvidence:
    """The result of one group comparison, with its inputs on the record."""

    verdict: Verdict
    n_a: int = 0
    n_b: int = 0
    chi2: float = 0.0
    dof: int = 0
    p_value: float = 1.0
    effect: float = 0.0
    reason: str = ""

    @property
    def blocks_merge(self) -> bool:
        return self.verdict is Verdict.SEPARATED

    def describe(self) -> str:
        if self.verdict is Verdict.NO_EVIDENCE:
            return f"NO_EVIDENCE ({self.reason}; pooled {self.n_a}/{self.n_b})"
        return (
            f"{self.verdict.value} (pooled {self.n_a}/{self.n_b}, "
            f"chi2={self.chi2:.1f} dof={self.dof}, p={self.p_value:.2e}, "
            f"effect={self.effect:.4f})"
        )


# ---- sufficient statistics --------------------------------------------


@dataclass
class GroupStats:
    """Per-observer sufficient statistics for one group of transactions.

    Held per observer dimension rather than as whole vectors so that a
    transaction missing one observer still contributes its other seven. Two
    groups' statistics add elementwise, which is what lets a union-find carry
    them and merge them in constant time - the pooled centroid of a merged
    component costs one vector addition, not a re-scan of its members.
    """

    count: int
    dim_count: np.ndarray
    dim_sum: np.ndarray
    dim_sumsq: np.ndarray

    @classmethod
    def empty(cls, n_dims: int) -> GroupStats:
        return cls(
            count=0,
            dim_count=np.zeros(n_dims, dtype=np.int64),
            dim_sum=np.zeros(n_dims, dtype=np.float64),
            dim_sumsq=np.zeros(n_dims, dtype=np.float64),
        )

    def combine(self, other: GroupStats) -> GroupStats:
        return GroupStats(
            count=self.count + other.count,
            dim_count=self.dim_count + other.dim_count,
            dim_sum=self.dim_sum + other.dim_sum,
            dim_sumsq=self.dim_sumsq + other.dim_sumsq,
        )

    def mean(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(
                self.dim_count > 0, self.dim_sum / np.maximum(self.dim_count, 1), np.nan
            )

    def variance(self) -> np.ndarray:
        """Unbiased per-observer variance; NaN where fewer than two samples."""
        with np.errstate(invalid="ignore", divide="ignore"):
            n = self.dim_count
            mean = self.mean()
            raw = self.dim_sumsq / np.maximum(n, 1) - mean**2
            raw = np.maximum(raw, 0.0)  # floating-point safety
            unbiased = raw * n / np.maximum(n - 1, 1)
            return np.where(n >= 2, unbiased, np.nan)


def separation_evidence(
    a: GroupStats, b: GroupStats, config: SeparationConfig | None = None
) -> SeparationEvidence:
    """Compare two pooled groups and decide whether to refuse their merge.

    Per observer, a two-sample standardised difference of means; summed
    across observers into a chi-square statistic. The per-observer form is
    deliberate: it stays explainable out loud - "the two groups' average
    arrival pattern differs by more than the measurement precision allows" -
    and needs no covariance estimate that these sample sizes could not
    support anyway.
    """
    config = config or SeparationConfig()

    if a.count < config.min_pooled_observations or b.count < config.min_pooled_observations:
        return SeparationEvidence(
            verdict=Verdict.NO_EVIDENCE,
            n_a=a.count,
            n_b=b.count,
            reason=(
                f"pooled observations below minimum "
                f"{config.min_pooled_observations}"
            ),
        )

    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.variance(), b.variance()

    usable = (
        (a.dim_count >= config.min_observer_observations)
        & (b.dim_count >= config.min_observer_observations)
        & np.isfinite(var_a)
        & np.isfinite(var_b)
    )
    if not usable.any():
        return SeparationEvidence(
            verdict=Verdict.NO_EVIDENCE,
            n_a=a.count,
            n_b=b.count,
            reason="no observer has enough paired observations",
        )

    delta = mean_a[usable] - mean_b[usable]
    standard_error = np.sqrt(
        var_a[usable] / a.dim_count[usable] + var_b[usable] / b.dim_count[usable]
    )
    live = standard_error > 0
    if not live.any():
        return SeparationEvidence(
            verdict=Verdict.NO_EVIDENCE,
            n_a=a.count,
            n_b=b.count,
            reason="zero variance in every usable observer",
        )

    z = delta[live] / standard_error[live]
    chi2 = float(np.sum(z**2))
    dof = int(live.sum())
    effect = float(np.sqrt(np.mean(delta[live] ** 2)))

    from scipy.stats import chi2 as chi2_dist

    p_value = float(chi2_dist.sf(chi2, dof))

    if p_value < config.alpha and effect >= config.min_effect:
        verdict = Verdict.SEPARATED
        reason = ""
    else:
        verdict = Verdict.NOT_SEPARATED
        reason = (
            "effect below floor" if p_value < config.alpha else "not significant"
        )

    return SeparationEvidence(
        verdict=verdict,
        n_a=a.count,
        n_b=b.count,
        chi2=chi2,
        dof=dof,
        p_value=p_value,
        effect=effect,
        reason=reason,
    )


# ---- building the per-address statistics -------------------------------


@dataclass
class SeparationOracle:
    """Per-address pooled statistics plus the comparison rule.

    A clusterer holds one of these, seeds each address with its own
    statistics, and adds them as components merge. The oracle itself is
    stateless with respect to clustering: it answers questions about pooled
    statistics and knows nothing about clusters, which keeps the constraint
    rule independent of the order edges happen to arrive in.
    """

    n_dims: int
    observer_ids: list[str]
    address_stats: dict[int, GroupStats] = field(default_factory=dict)
    config: SeparationConfig = field(default_factory=SeparationConfig)
    n_usable_transactions: int = 0
    n_excluded_no_evidence: int = 0
    n_addresses_with_evidence: int = 0
    n_pooled_observations: int = 0
    """Total pooled observations. Must equal the number of usable
    transactions attributed to an address - if it exceeds that, transactions
    are being double-counted and every standard error is wrong."""

    def stats_for(self, code: int) -> GroupStats:
        """Pooled statistics for one address; empty if it has no usable tx."""
        return self.address_stats.get(int(code), GroupStats.empty(self.n_dims))

    def compare(self, a: GroupStats, b: GroupStats) -> SeparationEvidence:
        return separation_evidence(a, b, self.config)

    def empty_stats(self) -> GroupStats:
        return GroupStats.empty(self.n_dims)


def _unit_normalise(offsets: np.ndarray) -> np.ndarray:
    """Scale each arrival vector by its own largest offset.

    Absolute delay varies between transactions for reasons unrelated to
    origin - network weather, transaction size, which peer happened to relay
    first. Normalising to a shape leaves the pooled centroid describing
    *relative* arrival order, which is the part that carries origin
    information. Pooling raw milliseconds would instead let a handful of slow
    transactions dominate the mean.
    """
    scale = np.nanmax(np.where(np.isnan(offsets), -np.inf, offsets), axis=1)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return offsets / scale[:, None]


def build_oracle(
    graph,
    processed_root: Path | None = None,
    data_root: Path | None = None,
    config: SeparationConfig | None = None,
    world: str | None = None,
) -> SeparationOracle:
    """Load network evidence and accumulate it per address.

    Reads exclusively through :mod:`obsidianchain.network.boundary`, so this
    function cannot see ground truth even by accident.
    """
    from obsidianchain.io import elliptic

    config = config or SeparationConfig()
    processed_root = Path(processed_root) if processed_root is not None else None
    if processed_root is None:
        root = Path(data_root) if data_root is not None else elliptic.DEFAULT_DATA_ROOT
        processed_root = root / "processed"

    # ``world`` selects WHICH dataset to read - the frozen production set by
    # default, or a controlled-world regime. It does not affect how evidence
    # is judged: the thresholds live in SeparationConfig and are unchanged.
    inputs = boundary.load_phase3_inputs(processed_root, world=world)
    observer_ids = sorted(inputs.observers["observer_id"].tolist()) or sorted(
        inputs.observations["observer_id"].unique().tolist()
    )
    vectors = arrivals.build(inputs.observations, observer_ids=observer_ids)

    broadcaster_txids = set(
        inputs.observations.loc[
            inputs.observations["peer_ip"].isin(inputs.broadcaster_ips), "txid"
        ]
    )
    labels = arrivals.classify_evidence(vectors, broadcaster_txids)

    usable_mask = (labels["evidence"] == arrivals.Evidence.USABLE.value).to_numpy()
    usable_txids = labels.loc[usable_mask, "txid"].to_numpy()

    # Belt and braces: the filter above already removed them, and this
    # re-checks that no NO_EVIDENCE transaction survived into the pool.
    arrivals.assert_no_constraint_permitted(labels, usable_txids)

    unit = _unit_normalise(vectors.offsets_ms[usable_mask])
    tx_position = pd.Index(usable_txids)

    # Map addresses to codes through the graph's own code space.
    if graph.addresses is None:
        raise ValueError(
            "separation needs address labels; "
            "call load_cospend_graph(keep_labels=True)"
        )
    edges = elliptic.load_input_edges(data_root)

    # THE PAIRING GUARD. This is the production fusion path - every caller
    # that builds an oracle passes through here - and a mismatched
    # chain/network pair must not get past it.
    #
    # Without this the failure is silent: the attribution below drops every
    # transaction that does not resolve to a chain address, the oracle comes
    # back with no statistics, and the run reports a 100% abstention that
    # reads exactly like a clean negative result. Measured on a real
    # mismatch: 0.095% of ids resolved, all of them coincidentally equal
    # integers belonging to different transactions.
    #
    # Applied to the frames already in hand rather than by calling
    # assert_datasets_compatible, which would re-read both files. Same rule,
    # same exception, no second read - and it fires before a single
    # sufficient statistic is accumulated, so no verdict can be produced
    # from an incompatible pair.
    assert_txid_overlap(
        inputs.observations["txid"].unique().tolist(),
        edges["txId"].unique().tolist(),
        chain_description=str(data_root if data_root is not None else "<default>"),
        network_description=(
            f"{processed_root} world {world}" if world is not None
            else str(processed_root)
        ),
    )

    edges = edges.assign(
        _code=pd.Index(graph.addresses).get_indexer(edges["input_address"])
    )
    edges = edges[edges["_code"] >= 0]

    # ONE representative input address per transaction.
    #
    # Without this, a transaction is counted once per input address, and
    # because co-spend unions all of a transaction's inputs into the same
    # component, every component formed from a k-input transaction counts
    # that transaction k times. Measured on this dataset the inflation
    # reached 651x: one component pooled 6,035 "observations" drawn from 26
    # distinct transactions. That understates the standard error by roughly
    # the square root of the inflation, so the chi-square explodes and
    # constraints fire on what is effectively a single observation replicated
    # hundreds of times - precisely what the repetition requirement exists
    # to prevent.
    #
    # Attributing each transaction to its lowest-coded input is safe: since
    # co-spend merges all inputs of a transaction, the component holding the
    # representative holds the others too, so nothing is lost. In a
    # constrained run a blocked merge can leave inputs in different
    # components, in which case the transaction counts only toward the
    # representative's side - an undercount, never an inflation.
    edges = edges.sort_values(["txId", "_code"], kind="stable").drop_duplicates(
        subset="txId", keep="first"
    )

    rows = tx_position.get_indexer(edges["txId"])
    keep = rows >= 0
    codes = edges.loc[keep, "_code"].to_numpy()
    rows = rows[keep]

    n_dims = len(observer_ids)
    n_codes = int(graph.n_addresses)
    present = ~np.isnan(unit)
    filled = np.nan_to_num(unit, nan=0.0)

    dim_count = np.zeros((n_codes, n_dims), dtype=np.int64)
    dim_sum = np.zeros((n_codes, n_dims), dtype=np.float64)
    dim_sumsq = np.zeros((n_codes, n_dims), dtype=np.float64)
    for d in range(n_dims):
        dim_count[:, d] = np.bincount(
            codes, weights=present[rows, d].astype(np.float64), minlength=n_codes
        ).astype(np.int64)
        dim_sum[:, d] = np.bincount(
            codes, weights=filled[rows, d], minlength=n_codes
        )
        dim_sumsq[:, d] = np.bincount(
            codes, weights=filled[rows, d] ** 2, minlength=n_codes
        )
    tx_count = np.bincount(codes, minlength=n_codes)

    address_stats: dict[int, GroupStats] = {}
    for code in np.flatnonzero(tx_count > 0):
        address_stats[int(code)] = GroupStats(
            count=int(tx_count[code]),
            dim_count=dim_count[code].copy(),
            dim_sum=dim_sum[code].copy(),
            dim_sumsq=dim_sumsq[code].copy(),
        )

    return SeparationOracle(
        n_dims=n_dims,
        observer_ids=observer_ids,
        address_stats=address_stats,
        config=config,
        n_usable_transactions=int(usable_mask.sum()),
        n_excluded_no_evidence=int((~usable_mask).sum()),
        n_addresses_with_evidence=len(address_stats),
        n_pooled_observations=int(tx_count.sum()),
    )


def format_oracle_summary(oracle: SeparationOracle) -> str:
    """Describe the loaded evidence, warning banner included."""
    width = 78
    out: list[str] = []
    add = out.append
    add("-- network separation evidence " + "-" * (width - 32))
    add("  SYNTHETIC network data - demonstrates the mechanism, validates")
    add("  nothing. See network.synthetic for what that means.")
    add("")
    add(f"  observers                       {oracle.n_dims:>12,}")
    add(f"  usable transactions             {oracle.n_usable_transactions:>12,}")
    add(f"  excluded NO_EVIDENCE            {oracle.n_excluded_no_evidence:>12,}"
        f"   never reach a constraint")
    add(f"  addresses with any evidence     {oracle.n_addresses_with_evidence:>12,}")
    counts = np.array([s.count for s in oracle.address_stats.values()] or [0])
    add(f"  pooled observations total       {oracle.n_pooled_observations:>12,}"
        f"   one per transaction, no double count")
    add(f"  pooled per address  median {np.median(counts):>6.0f}   max {counts.max():>8,}")
    cfg = oracle.config
    add(f"  min pooled per side             {cfg.min_pooled_observations:>12,}")
    add(f"  alpha                           {cfg.alpha:>12.1e}")
    add(f"  min effect                      {cfg.min_effect:>12.3f}")
    add("  CANNOT-LINK only: shared origin never implies shared entity.")
    return "\n".join(out)
