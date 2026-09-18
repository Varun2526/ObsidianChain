"""M4 - mixing / CoinJoin-like transaction structure. LABEL-BLIND.

What this module detects, and what it does not
----------------------------------------------
It detects a STRUCTURAL PATTERN in a transaction: many participants on both
sides, outputs of near-identical value, inputs of varied value. That shape is
what a collaborative-spend protocol produces, and it is also what a handful
of unrelated things produce. The detector says the pattern is present. It
does not say the transaction is a mixer, that a mixer was used illegally, or
that anyone involved did anything wrong. Mixing is a legal privacy technique
with ordinary uses.

So the vocabulary is deliberately about the PATTERN:

``MIXING_PATTERN``
    Every structural condition holds and no benign shape explains it.

``MIXING_LIKELIHOOD``
    Some conditions hold. Partial evidence, reported as partial.

``NO_MIXING_SIGNAL``
    The structure was measurable and does not show the pattern.

``INSUFFICIENT_DATA``
    The quantities needed were not available for this transaction. Reported
    instead of a zero, because "not measured" and "measured as none" are
    different answers - the same rule the rest of this project follows.

Why not a rule
--------------
``if inputs > 5 and equal_outputs: mixer`` would fire on exchange payout
batches, on consolidations, on faucet runs and on any service that pays a
fixed amount to many people. This scores FOUR independent structural signals
and then applies explicit SUPPRESSORS for the benign shapes that resemble
mixing. A transaction that looks like a batch is classified as a batch, and
the reason is recorded.

What is observable here
-----------------------
``txs_features.csv`` gives per-transaction input/output counts and the min,
max, mean and median of the input and output values. It does NOT give the
individual output values, so "how many outputs share a value" cannot be
computed exactly. The min/max/mean/median summary is enough to measure
whether the outputs are near-identical, which is the signal that matters, and
this module claims nothing beyond it.

Determinism
-----------
Pure arithmetic on the transaction table. No sampling, no ordering
dependence, no state. The same input frame and the same config produce the
same classification, which
``tests/test_mixing.py::test_the_detector_is_deterministic`` pins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from obsidianchain.alerts import contract as _contract

#: Address-level columns this group contributes to the Phase 6 matrix.
M4_COLUMNS = [
    "mixing_tx_count_asof_t",
    "mixing_tx_share_asof_t",
    "mixing_score_max_asof_t",
    "mixing_score_mean_asof_t",
    "mixing_participants_max_asof_t",
    "mixing_signal_available",
]

#: Per-transaction columns the scan artifact carries.
TX_COLUMNS = [
    "txId", "timestep", "mixing_class", "mixing_score",
    "output_uniformity", "participant_symmetry", "cardinality_signal",
    "input_heterogeneity", "suppressor", "n_inputs", "n_outputs",
]

# The vocabulary lives in alerts/contract.py, which is standard library only
# and therefore importable by the API. Restating it here would let the two
# drift, and then the detector and the endpoint describing it would disagree.
MIXING_PATTERN = _contract.MIXING_PATTERN
MIXING_LIKELIHOOD = _contract.MIXING_LIKELIHOOD
NO_MIXING_SIGNAL = _contract.NO_MIXING_SIGNAL
INSUFFICIENT_DATA = _contract.MIXING_INSUFFICIENT_DATA

CLASSES = _contract.MIXING_CLASSES

SUPPRESSOR_NONE = _contract.SUPPRESSOR_NONE
SUPPRESSOR_BATCH = _contract.SUPPRESSOR_BATCH
SUPPRESSOR_CONSOLIDATION = _contract.SUPPRESSOR_CONSOLIDATION
SUPPRESSOR_ORDINARY = _contract.SUPPRESSOR_ORDINARY
SUPPRESSOR_UNIFORM_PAYOUT = _contract.SUPPRESSOR_UNIFORM_PAYOUT

SUPPRESSORS = _contract.MIXING_SUPPRESSORS


@dataclass(frozen=True)
class MixingConfig:
    """Every threshold, in one frozen place.

    Named and versioned because the classification is meaningless without
    them: two runs under different thresholds are not comparable, and the
    label travels into the artifact's provenance so that cannot be forgotten.
    """

    #: Fewest participants per side before an anonymity set is meaningful.
    #: Three is the smallest number for which a collaborative spend hides
    #: anything at all; two parties learn each other's output by elimination.
    min_participants: int = 3

    #: Output spread, ``(max-min)/mean``, at or below which outputs count as
    #: near-identical. 2% absorbs fee-adjusted change in an equal-value
    #: round without admitting genuinely different payment amounts.
    uniform_spread: float = 0.02

    #: Spread at or above which values count as genuinely varied. Used for
    #: the INPUT side: independent funders bring unrelated amounts.
    varied_spread: float = 0.25

    #: A side is "many" from here. Used by the batch and consolidation
    #: suppressors.
    many: int = 5

    #: At or below this a side is "few". Used by the ordinary-shape
    #: suppressor.
    few: int = 2

    #: Score at or above which the full pattern is asserted.
    pattern_threshold: float = 0.75

    #: Score at or above which partial evidence is reported.
    likelihood_threshold: float = 0.45

    @property
    def label(self) -> str:
        """Travels into the artifact sidecar. Changing a threshold changes
        this string, so a reader can never mistake one run's classification
        for another's."""
        return (
            f"mixing/1:min_participants={self.min_participants},"
            f"uniform_spread={self.uniform_spread},"
            f"varied_spread={self.varied_spread},"
            f"many={self.many},few={self.few},"
            f"pattern={self.pattern_threshold},"
            f"likelihood={self.likelihood_threshold}"
        )


DEFAULT_CONFIG = MixingConfig()

#: The four signals, and the weight each carries. They sum to 1.0.
#:
#: Output uniformity is weighted highest because it is the one signal no
#: benign shape reproduces WHILE also having many independent inputs.
#: Cardinality alone is nearly worthless - large transactions are common -
#: so it carries least.
SIGNAL_WEIGHTS = {
    "output_uniformity": 0.40,
    "input_heterogeneity": 0.25,
    "participant_symmetry": 0.20,
    "cardinality_signal": 0.15,
}

MEANING = _contract.MIXING_PATTERN_MEANING


INSUFFICIENT_DATA_MEANING = _contract.MIXING_INSUFFICIENT_DATA_MEANING



def _spread(low, high, mean) -> np.ndarray:
    """``(max - min) / mean``: 0 when every value is identical.

    Guarded against a zero or missing mean, which cannot be normalised by
    and is returned as NaN so the caller classifies it INSUFFICIENT_DATA
    rather than as a suspiciously perfect zero.
    """
    mean = np.asarray(mean, dtype="float64")
    span = np.asarray(high, dtype="float64") - np.asarray(low, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(mean > 0, span / mean, np.nan)
    return out


def _ramp(value, lo: float, hi: float) -> np.ndarray:
    """Linear 0->1 between ``lo`` and ``hi``, clamped. NaN stays NaN."""
    value = np.asarray(value, dtype="float64")
    with np.errstate(invalid="ignore"):
        return np.clip((value - lo) / (hi - lo), 0.0, 1.0)


def score_transactions(
    transactions: pd.DataFrame, config: MixingConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Classify every transaction. One row in, one row out, same order.

    ``transactions`` must carry ``num_input_addresses``,
    ``num_output_addresses`` and the min/max/mean of both value sides.
    Anything missing yields INSUFFICIENT_DATA for that row rather than an
    assumed value.
    """
    frame = transactions
    required = (
        "num_input_addresses", "num_output_addresses",
        "in_BTC_min", "in_BTC_max", "in_BTC_mean",
        "out_BTC_min", "out_BTC_max", "out_BTC_mean",
    )
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(
            f"the mixing detector needs {missing}; it measures value "
            f"structure and cannot infer it from counts alone"
        )

    n_in = pd.to_numeric(frame["num_input_addresses"], errors="coerce").to_numpy("float64")
    n_out = pd.to_numeric(frame["num_output_addresses"], errors="coerce").to_numpy("float64")

    out_spread = _spread(frame["out_BTC_min"], frame["out_BTC_max"],
                         frame["out_BTC_mean"])
    in_spread = _spread(frame["in_BTC_min"], frame["in_BTC_max"],
                        frame["in_BTC_mean"])

    # ---- the four signals ------------------------------------------------
    # Uniformity: 1 when the outputs are identical, falling away as they
    # differ. Measured, not thresholded, so a near-miss is visible as a
    # near-miss rather than collapsing to zero.
    output_uniformity = 1.0 - _ramp(
        out_spread, config.uniform_spread, config.varied_spread
    )

    # Heterogeneity: independent funders bring unrelated amounts. This is
    # what separates a collaborative spend from a service paying one amount
    # to many people, whose inputs are as uniform as its outputs.
    input_heterogeneity = _ramp(
        in_spread, config.uniform_spread, config.varied_spread
    )

    # Symmetry: a collaborative spend has comparable counts on both sides.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (n_in > 0) & (n_out > 0),
            np.minimum(n_in, n_out) / np.maximum(n_in, n_out),
            np.nan,
        )
    participant_symmetry = np.asarray(ratio, dtype="float64")

    # Cardinality: the anonymity-set precondition. Saturates quickly - the
    # difference between 3 and 5 participants matters, between 50 and 500
    # does not.
    smaller_side = np.minimum(n_in, n_out)
    cardinality_signal = _ramp(
        smaller_side, config.min_participants, config.min_participants * 3
    )

    score = (
        SIGNAL_WEIGHTS["output_uniformity"] * output_uniformity
        + SIGNAL_WEIGHTS["input_heterogeneity"] * input_heterogeneity
        + SIGNAL_WEIGHTS["participant_symmetry"] * participant_symmetry
        + SIGNAL_WEIGHTS["cardinality_signal"] * cardinality_signal
    )

    # ---- suppressors: the benign shapes that resemble mixing -------------
    # Evaluated in order of specificity. The first that matches is recorded,
    # so the reason a transaction was not called a mixing pattern is a fact
    # in the artifact rather than an inference from a low score.
    suppressor = np.full(len(frame), SUPPRESSOR_NONE, dtype=object)

    ordinary = (n_in <= config.few) & (n_out <= config.few + 1)
    batch = (n_in <= config.few) & (n_out >= config.many)
    consolidation = (n_in >= config.many) & (n_out <= config.few)
    uniform_payout = (
        (output_uniformity > 0.5) & (n_in < config.min_participants) & ~batch
    )

    suppressor[uniform_payout] = SUPPRESSOR_UNIFORM_PAYOUT
    suppressor[ordinary] = SUPPRESSOR_ORDINARY
    suppressor[consolidation] = SUPPRESSOR_CONSOLIDATION
    suppressor[batch] = SUPPRESSOR_BATCH
    suppressed = suppressor != SUPPRESSOR_NONE

    # ---- classification --------------------------------------------------
    measurable = (
        np.isfinite(out_spread) & np.isfinite(in_spread)
        & np.isfinite(n_in) & np.isfinite(n_out)
        & (n_in > 0) & (n_out > 0)
    )

    enough_participants = (
        (n_in >= config.min_participants) & (n_out >= config.min_participants)
    )

    mixing_class = np.full(len(frame), NO_MIXING_SIGNAL, dtype=object)
    mixing_class[
        measurable & ~suppressed & (score >= config.likelihood_threshold)
    ] = MIXING_LIKELIHOOD
    mixing_class[
        measurable & ~suppressed & enough_participants
        & (score >= config.pattern_threshold)
    ] = MIXING_PATTERN
    mixing_class[~measurable] = INSUFFICIENT_DATA

    # A score is a claim about measured structure. Where nothing could be
    # measured there is no claim, so the score is absent rather than zero.
    score = np.where(measurable, score, np.nan)

    result = pd.DataFrame({
        "mixing_class": mixing_class,
        "mixing_score": score,
        "output_uniformity": np.where(measurable, output_uniformity, np.nan),
        "input_heterogeneity": np.where(measurable, input_heterogeneity, np.nan),
        "participant_symmetry": np.where(measurable, participant_symmetry, np.nan),
        "cardinality_signal": np.where(measurable, cardinality_signal, np.nan),
        "suppressor": suppressor,
        "n_inputs": n_in,
        "n_outputs": n_out,
    }, index=frame.index)
    if "txId" in frame.columns:
        result.insert(0, "txId", frame["txId"].to_numpy())
    if "Time step" in frame.columns:
        result.insert(1, "timestep", frame["Time step"].to_numpy())
    return result


def build(incidence, cutoff: pd.Series,
          config: MixingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Per-address M4 features, each as of that address's own timestep.

    An address's mixing exposure is how many of the transactions it took part
    in - AT OR BEFORE its observation point - showed the pattern. The cutoff
    filter is applied explicitly rather than relied upon: it happens to be
    redundant today because ``cutoff`` is the address's own last active
    timestep, and a redundant guard that is tested is worth more than an
    invariant that is merely true.
    """
    scored = score_transactions(incidence.transactions, config)
    scored.index = incidence.transactions.index

    rows = incidence.frame
    limit = cutoff.reindex(rows["code"].to_numpy()).to_numpy()
    visible = rows[rows["Time step"].to_numpy() <= limit]

    # One (address, transaction) pair once: an address that both funds and
    # receives from a transaction took part in ONE transaction, not two.
    pairs = visible[["code", "txId"]].drop_duplicates()
    joined = pairs.join(
        scored[["mixing_class", "mixing_score", "n_inputs", "n_outputs"]],
        on="txId",
    )

    is_pattern = joined["mixing_class"].to_numpy() == MIXING_PATTERN
    measured = joined["mixing_class"].to_numpy() != INSUFFICIENT_DATA
    joined = joined.assign(
        _pattern=is_pattern.astype("float64"),
        _measured=measured.astype("float64"),
        _participants=np.where(
            is_pattern,
            np.minimum(joined["n_inputs"].to_numpy(),
                       joined["n_outputs"].to_numpy()),
            np.nan,
        ),
    )

    grouped = joined.groupby("code")
    features = pd.DataFrame(index=grouped.size().index)
    features.index.name = "code"

    features["mixing_tx_count_asof_t"] = grouped["_pattern"].sum()
    measured_n = grouped["_measured"].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        features["mixing_tx_share_asof_t"] = np.where(
            measured_n > 0,
            features["mixing_tx_count_asof_t"] / measured_n,
            np.nan,
        )
    features["mixing_score_max_asof_t"] = grouped["mixing_score"].max()
    features["mixing_score_mean_asof_t"] = grouped["mixing_score"].mean()
    features["mixing_participants_max_asof_t"] = grouped["_participants"].max()
    # Not a measurement: a flag saying whether the measurements above mean
    # anything for this address. An address all of whose transactions were
    # unmeasurable gets 0 here and NaN above, never a confident zero.
    features["mixing_signal_available"] = (measured_n > 0).astype("int8")

    return features.reindex(columns=M4_COLUMNS)
