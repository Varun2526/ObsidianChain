"""The canonical evaluation protocol. Model selection happens HERE or nowhere.

Why this module exists
----------------------
The audit measured three sources of variation in this problem:

    random seed, fixed window          sd = 0.005
    bootstrap within one window        95% CI half-width = 0.015
    CHOICE OF EVALUATION WINDOW        sd = 0.175

Window choice dominates the other two by roughly 35x. Every headline number
this project has quoted to four decimals - ``val_pr_auc 0.5702``,
``test_pr_auc 0.2079`` - is one draw from a distribution that wide. The
val-to-test "collapse" is well inside it and is not, by itself, evidence of
anything.

The consequence is blunt: **a single-window comparison cannot rank two
models on this data.** Across the development folds the ranking actually
flips - LightGBM leads on one fold and collapses to nAP 0.049 on another
where RandomForest holds 0.317. Anyone selecting on one window is selecting
on noise, and will report a number that does not survive contact with the
next window.

So this module makes the only defensible comparison the easy one to run:
repeated rolling-origin folds inside the development period, paired by fold,
reported with a spread.

The two rules
-------------
1. **The holdout is sealed.** Timesteps >= :data:`HOLDOUT_START` are not
   readable through the sanctioned accessors. Reaching them requires
   :func:`break_seal` with a written reason, which is the point: it turns an
   accident into a decision somebody made on the record.

2. **No model is superior on a point estimate.** :func:`compare` reports the
   paired per-fold difference, its spread and how many folds it won. A
   difference smaller than the fold-to-fold spread is reported as
   INDISTINGUISHABLE, and that is the honest answer far more often than not.

What this module does not do
----------------------------
It does not choose a model, tune a hyperparameter or touch an artifact. It
measures, and it refuses to measure on the holdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Last timestep available for ANY development activity - feature work,
#: hyperparameters, model choice, threshold selection, this protocol's own
#: folds. Matches the production split boundary so a model selected here can
#: be frozen against the same line.
DEVELOPMENT_END = 41

#: First timestep of the sealed holdout. Read once, at freeze time, after the
#: model and every threshold are final.
HOLDOUT_START = 42

#: Final timestep in the dataset.
DATASET_END = 49

#: Width of each evaluation fold, in timesteps.
#:
#: Two, measured rather than argued. The design trade was swept over the
#: development period:
#:
#:     width  folds  min rows  min positives  MDE@80% power
#:       4        5     9,325          1,174          0.311
#:       3        8     3,732            322          0.215
#:       2       12     1,379            141          0.164
#:       1       25       665             43          0.108
#:
#: Width 4 gave an MDE of 0.311 nAP - larger than any difference between the
#: candidate families, so the design could not have detected a real effect if
#: one existed. Width 1 has the most power but leaves windows too small for
#: precision@100 to mean anything (k would exceed 15% of the window). Width 2
#: is the point where the design can resolve a plausible effect and every
#: fold still holds enough positives to rank.
FOLD_WIDTH = 2

#: Timesteps of history required before the first fold. A model trained on
#: five timesteps is not a smaller version of the real thing, it is a
#: different and much worse thing, and folds that early would add variance
#: that says nothing about the candidate.
MIN_TRAIN_TIMESTEPS = 16

PROTOCOL_VERSION = "obsidianchain.eval_protocol/2"

#: Significance level, and the test the verdict is based on.
ALPHA = 0.05
TEST_NAME = "paired_t"

#: Smallest paired nAP difference this design detects at 80% power, measured
#: by simulation at the observed paired spread (sd = 0.185) over 12 folds.
MDE_80_POWER = 0.164

#: Fewest positives in any development fold under the default design. Below
#: about a hundred, precision@k is estimated on too few events to rank.
MIN_POSITIVES_PER_FOLD = 141

#: What this dataset can and cannot resolve, stated as fact rather than
#: buried as a caveat. Measured, not asserted.
POWER_LIMITS = (
    "At 12 folds and the observed paired spread (sd 0.185), this design "
    "detects a 0.164 nAP difference at 80% power. Detecting 0.10 would need "
    "29 folds and 0.05 would need 110 folds, neither of which the "
    "development period can supply. Differences below roughly 0.05 nAP are "
    "therefore PERMANENTLY unresolvable on this dataset, and a protocol run "
    "that reports INDISTINGUISHABLE for such a difference is reporting the "
    "limit of the data, not a property of the models."
)

#: Why the exact sign-flip permutation needs a minimum fold count: with n
#: folds there are 2^n sign assignments, so the smallest reachable two-sided
#: p-value is 2/2^n. At n=5 that is 0.0625 and the test cannot reject at .05
#: however large the effect.
MIN_FOLDS_FOR_PERMUTATION = 7

SEAL_MEANING = (
    "Timesteps 42-49 are the sealed holdout. They are read once, after the "
    "model, its hyperparameters, its features and its thresholds are all "
    "final. A number measured on them after any development decision was "
    "informed by them is not a held-out number, however it is labelled."
)

COMPARISON_MEANING = (
    "Differences are paired by fold and reported with their spread. On this "
    "data the fold-to-fold standard deviation is roughly 35x the seed-to-seed "
    "standard deviation, so a difference smaller than the fold spread is "
    "indistinguishable from the choice of window and is reported as such."
)


#: The two production paths, with the scope each one covers.
#:
#: Both are production. They describe different feature sets over different
#: unit definitions and different source data, so a number from one says
#: nothing about the other - and the protocol refuses to compare them rather
#: than leaving it to a reader to notice.
SCOPE_PHASE6 = "phase6_elliptic"
SCOPE_PS_NATIVE = "ps_native"
SCOPE_UNDECLARED = "undeclared"

SCOPES = {
    SCOPE_PHASE6: (
        "The Elliptic++ reference artifacts: M0-M3 features, LightGBM with "
        "isotonic calibration, TreeSHAP explanations, serving the frozen "
        "alert run 043ea584e99daf99."
    ),
    SCOPE_PS_NATIVE: (
        "Uploaded-dataset runs: the ps_native feature schema executed by the "
        "orchestrator for an AnalysisRun, over transactions a user supplied."
    ),
    SCOPE_UNDECLARED: (
        "No scope declared. Usable for exploration, and refused for any "
        "comparison against a declared scope so an undeclared result cannot "
        "be quietly pooled with a real one."
    ),
}


class ScopeMismatchError(ValueError):
    """Raised when two candidates from different scopes would be compared."""


class HoldoutSealError(RuntimeError):
    """Raised when the sealed holdout is reached without an explicit break."""


@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold. Train is everything before the window."""

    index: int
    train_end: int
    eval_start: int
    eval_end: int

    @property
    def label(self) -> str:
        return f"t<={self.train_end} -> t{self.eval_start}-{self.eval_end}"

    def __post_init__(self) -> None:
        # Enforced at construction rather than checked at use, so a fold that
        # reaches the holdout cannot exist to be passed around. The audit that
        # produced this protocol built folds 41-44 and 45-48 by hand and
        # evaluated on holdout timesteps without noticing.
        if self.eval_end > DEVELOPMENT_END:
            raise HoldoutSealError(
                f"fold {self.label} evaluates on t{self.eval_end}, which is "
                f"inside the sealed holdout (t>={HOLDOUT_START}). "
                f"{SEAL_MEANING}"
            )
        if self.eval_start <= self.train_end:
            raise ValueError(
                f"fold {self.label} evaluates on timesteps it also trains on"
            )


def rolling_origin_folds(
    *, width: int = FOLD_WIDTH, min_train: int = MIN_TRAIN_TIMESTEPS,
    development_end: int = DEVELOPMENT_END,
) -> list[Fold]:
    """Expanding-window folds, entirely inside the development period.

    Expanding rather than sliding: a deployed model is refitted on everything
    known so far, so a fold that discarded early history would measure a
    model nobody would ship.
    """
    folds = []
    train_end = min_train
    while train_end + width <= development_end:
        folds.append(Fold(
            index=len(folds), train_end=train_end,
            eval_start=train_end + 1, eval_end=train_end + width,
        ))
        train_end += width
    return folds


# ---- the seal -----------------------------------------------------------


def development(frame: pd.DataFrame, *, timestep: str = "first_t") -> pd.DataFrame:
    """Every row a model developer may look at. The default accessor.

    Filters on FIRST appearance, matching how the production split is
    assigned: an address that first appears at t40 belongs to development
    even though it stays active into the holdout window.
    """
    _require_column(frame, timestep)
    return frame[frame[timestep] <= DEVELOPMENT_END]


def holdout(frame: pd.DataFrame, *, timestep: str = "first_t") -> pd.DataFrame:
    """The sealed holdout. Raises unless the seal has been broken."""
    if not _seal.broken:
        raise HoldoutSealError(
            "the holdout is sealed. " + SEAL_MEANING + " If the model is "
            "genuinely final, call break_seal(reason=...) and say why in the "
            "record."
        )
    _require_column(frame, timestep)
    return frame[frame[timestep] >= HOLDOUT_START]


@dataclass
class _Seal:
    broken: bool = False
    reasons: list = field(default_factory=list)


_seal = _Seal()


def break_seal(reason: str) -> None:
    """Unseal the holdout, on the record.

    Deliberately awkward. There is no way to reach the holdout by accident,
    and every break carries a reason that a reader can judge. Breaking the
    seal to "just check" is exactly the thing that turns a holdout into a
    second validation set.
    """
    if not reason or len(reason.strip()) < 20:
        raise ValueError(
            "breaking the holdout seal requires a written reason of at least "
            "20 characters, naming what is final and why this read is the "
            "one-time freeze measurement"
        )
    _seal.broken = True
    _seal.reasons.append(reason.strip())


def reseal() -> None:
    """Restore the seal. Used by tests; harmless in a script."""
    _seal.broken = False


def seal_state() -> dict:
    return {"broken": _seal.broken, "reasons": list(_seal.reasons)}


def _require_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise KeyError(
            f"the evaluation protocol needs a {column!r} column to place a "
            f"row in time; without it the holdout cannot be sealed"
        )


# ---- metrics that survive ties -----------------------------------------


def normalised_average_precision(y_true, scores) -> tuple[float, float]:
    """``(nAP, prevalence)``. The primary metric, and why.

    Prevalence across the development folds ranges from 3.3% to 21.0% - a
    6.3x swing. Raw average precision has its floor AT the prevalence, so a
    no-skill model scores 0.03 on one fold and 0.21 on another, and an
    average over folds is then dominated by which windows were chosen.
    Normalising puts no-skill at 0 and perfect at 1 in every fold, so the
    ends of the scale mean the same thing everywhere and the folds can be
    averaged at all.

    What it does NOT do: make a fixed-separation signal score identically
    across prevalences. A window with more positives is genuinely easier to
    rank, and nAP leaves that real difference visible rather than
    normalising it away. Per-fold prevalence therefore travels with every
    fold result, so a reader can always see which is which.
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true)
    prevalence = float(y_true.mean()) if len(y_true) else float("nan")
    if not np.isfinite(prevalence) or prevalence in (0.0, 1.0):
        return float("nan"), prevalence
    ap = float(average_precision_score(y_true, scores))
    return (ap - prevalence) / (1.0 - prevalence), prevalence


def precision_at_k_bounds(y_true, scores, k: int) -> tuple[float, float, int]:
    """``(worst, best, n_tied)`` precision@k over all tie orderings.

    Isotonic calibration collapses ~10,000 distinct scores onto ~55, so the
    rank-k cutoff usually lands inside a block of identical scores and the
    reported precision depends on row order - an arbitrary artefact of how
    the frame was built. Reporting a single value there is false precision.

    The bounds are exact: within the tied block, the best case takes its
    positives first and the worst takes its negatives first.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype="float64")
    if len(y_true) == 0:
        return float("nan"), float("nan"), 0
    k = min(int(k), len(y_true))

    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    ranked_y = y_true[order]

    cutoff = ranked_scores[k - 1]
    strictly_above = int((ranked_scores > cutoff).sum())
    tied = int((scores == cutoff).sum())
    slots_in_tie = k - strictly_above

    certain_positives = int(ranked_y[:strictly_above].sum())
    tied_positives = int(y_true[scores == cutoff].sum())

    best = certain_positives + min(slots_in_tie, tied_positives)
    worst = certain_positives + max(
        0, slots_in_tie - (tied - tied_positives)
    )
    return worst / k, best / k, tied


# ---- running a candidate ------------------------------------------------


@dataclass
class FoldResult:
    fold: Fold
    n_train: int
    n_eval: int
    prevalence: float
    nap: float
    p_at_k_worst: dict
    p_at_k_best: dict

    def as_dict(self) -> dict:
        return {
            "fold": self.fold.label,
            "train_end": self.fold.train_end,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "prevalence": self.prevalence,
            "nap": self.nap,
            "precision_at_k_worst": self.p_at_k_worst,
            "precision_at_k_best": self.p_at_k_best,
        }


@dataclass
class CandidateResult:
    """One candidate, measured across every fold. Never one number."""

    name: str
    seed: int
    folds: list
    #: Which production path produced this. Declared, never inferred.
    scope: str = SCOPE_UNDECLARED

    @property
    def naps(self) -> np.ndarray:
        return np.array([f.nap for f in self.folds], dtype="float64")

    def summary(self) -> dict:
        naps = self.naps
        finite = naps[np.isfinite(naps)]
        return {
            "candidate": self.name,
            "scope": self.scope,
            "seed": self.seed,
            "folds": len(self.folds),
            "nap_mean": float(finite.mean()) if len(finite) else float("nan"),
            # The spread is not a footnote. It is the number that decides
            # whether any difference between two candidates means anything.
            "nap_sd": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
            "nap_min": float(finite.min()) if len(finite) else float("nan"),
            "nap_max": float(finite.max()) if len(finite) else float("nan"),
            "per_fold": [f.as_dict() for f in self.folds],
        }


def evaluate_candidate(
    frame: pd.DataFrame,
    fit_predict,
    *,
    name: str,
    seed: int = 0,
    scope: str = SCOPE_UNDECLARED,
    features: list | None = None,
    folds: list | None = None,
    k_values=(100, 200, 500),
    timestep: str = "first_t",
    label: str = "y",
) -> CandidateResult:
    """Run one candidate over every development fold.

    ``fit_predict(train_frame, eval_frame, features, seed) -> scores`` keeps
    this module free of any model: a candidate is a callable, so comparing
    two families needs no change here and no model is privileged.

    The frame is passed through :func:`development` first, so a caller who
    hands in the whole dataset still cannot reach the holdout.
    """
    usable = development(frame, timestep=timestep)
    folds = folds if folds is not None else rolling_origin_folds()
    features = features or [
        c for c in usable.columns
        if c not in (timestep, label) and usable[c].dtype.kind in "biufc"
    ]

    results = []
    for fold in folds:
        train = usable[usable[timestep] <= fold.train_end]
        evaluation = usable[
            (usable[timestep] >= fold.eval_start)
            & (usable[timestep] <= fold.eval_end)
        ]
        if len(train) == 0 or len(evaluation) == 0:
            continue
        if evaluation[label].nunique() < 2:
            # A fold with one class has no ranking to measure. Skipped and
            # counted, never scored as zero.
            continue

        scores = np.asarray(fit_predict(train, evaluation, features, seed))
        nap, prevalence = normalised_average_precision(
            evaluation[label], scores
        )
        worst, best = {}, {}
        for k in k_values:
            lo, hi, _tied = precision_at_k_bounds(evaluation[label], scores, k)
            worst[f"P@{k}"], best[f"P@{k}"] = lo, hi

        results.append(FoldResult(
            fold=fold, n_train=int(len(train)), n_eval=int(len(evaluation)),
            prevalence=prevalence, nap=nap,
            p_at_k_worst=worst, p_at_k_best=best,
        ))

    return CandidateResult(name=name, seed=seed, folds=results, scope=scope)


# ---- comparing candidates ----------------------------------------------

INDISTINGUISHABLE = "INDISTINGUISHABLE"
FAVOURS = "FAVOURS"


def compare(a: "CandidateResult", b: "CandidateResult") -> dict:
    """Paired fold-wise comparison of two candidates.

    Delegates to :func:`paired_verdict` so there is exactly one decision
    rule in this module. An earlier version applied a separate
    one-standard-deviation heuristic here, which measured an 8.7%
    false-positive rate against the t-test's 5.1%.

    Paired because the folds are the same windows for both candidates, so
    the fold-to-fold variation that dominates everything cancels in the
    difference. Comparing two unpaired means on this data would be measuring
    which windows each happened to be run on.

    For three or more candidates use :func:`compare_family`, which adds the
    multiplicity correction this function cannot apply on its own.
    """
    _require_same_scope([a, b])
    differences = _paired_differences(a, b)
    if differences.size == 0:
        return {
            "verdict": INDISTINGUISHABLE,
            "favours": None,
            "reason": "the two candidates share no fold",
            "meaning": COMPARISON_MEANING,
            "power_limits": POWER_LIMITS,
        }
    verdict = paired_verdict(differences, a=a.name, b=b.name)
    verdict["folds_compared"] = int(differences.size)
    verdict["mean_nap_difference"] = verdict["mean_difference"]
    verdict["sd_nap_difference"] = verdict["sd_difference"]
    return verdict


# ---- the decision rule, calibrated -------------------------------------
#
# The first version of this module compared |mean difference| against one
# standard deviation. That was reasoned, not measured, and simulation at the
# observed paired spread put its false-positive rate at 8.7% - it declares a
# winner about one time in eleven when the candidates are identical. A paired
# t-test at alpha=.05 measured 5.1%. The rule below is the calibrated one.


def holm(p_values: dict) -> dict:
    """Holm-Bonferroni step-down adjustment over a family of comparisons.

    Three pairwise comparisons among three candidates is a family: testing
    each at .05 gives roughly a 14% chance of at least one false positive.
    Holm controls that at .05 while being uniformly more powerful than
    plain Bonferroni, and it needs no independence assumption - which
    matters here, because the three comparisons share folds and candidates
    and are anything but independent.

    Implemented directly rather than taken from statsmodels, which is not in
    the vendored wheel set; adding it would mean re-vendoring and touching
    the offline build guarantee for ten lines of arithmetic.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    total = len(ordered)
    adjusted = {}
    running = 0.0
    for index, (label, value) in enumerate(ordered):
        # Step-down with the running maximum, which is what keeps the
        # adjusted values monotone in the raw ones.
        running = min(1.0, max(running, (total - index) * float(value)))
        adjusted[label] = running
    return adjusted


def _permutation_p(differences: np.ndarray) -> tuple[float, bool]:
    """Exact sign-flip permutation p, and whether it can reject at ALPHA.

    Assumption-free: under the null a paired difference is as likely to be
    negative as positive, so enumerating all 2^n sign assignments gives the
    exact null distribution of the mean. Its weakness is granularity, and
    the second return value reports it rather than hiding it.
    """
    n = len(differences)
    usable = n >= MIN_FOLDS_FOR_PERMUTATION
    if n == 0:
        return float("nan"), False
    if n > 22:
        # Enumeration costs 2^n; beyond this a deterministic subsample is
        # indistinguishable and finite.
        rng = np.random.default_rng(0)
        signs = rng.choice([-1.0, 1.0], size=(1 << 20, n))
    else:
        signs = np.array(
            np.meshgrid(*[[-1.0, 1.0]] * n)
        ).T.reshape(-1, n)
    null = (signs * differences).mean(axis=1)
    observed = abs(float(differences.mean()))
    return float((np.abs(null) >= observed - 1e-12).mean()), usable


def paired_verdict(differences, *, a: str = "a", b: str = "b") -> dict:
    """Decide whether a paired difference vector supports a winner.

    Reports the effect, its interval, the calibrated p-value and the
    assumption-free check together, because any one of them alone is how a
    number gets quoted without its uncertainty.
    """
    from scipy import stats

    differences = np.asarray(differences, dtype="float64")
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n < 2:
        return {
            "verdict": INDISTINGUISHABLE, "favours": None, "n_folds": n,
            "reason": "fewer than two comparable folds",
            "meaning": COMPARISON_MEANING, "power_limits": POWER_LIMITS,
        }

    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))

    if sd == 0.0:
        # Zero variance breaks the t statistic (0/0 -> nan). Two cases, and
        # they mean opposite things, so neither may be left as nan:
        # every fold identical at zero is the strongest possible evidence of
        # NO difference, and every fold identical at some non-zero value is a
        # difference with no sampling variation to argue about.
        p_value = 1.0 if mean == 0.0 else 0.0
    else:
        p_value = float(stats.ttest_1samp(differences, 0.0).pvalue)
    half_width = float(
        stats.t.ppf(1 - ALPHA / 2, n - 1) * sd / np.sqrt(n)
    ) if sd > 0 else 0.0
    permutation_p, permutation_usable = _permutation_p(differences)

    decisive = p_value < ALPHA
    return {
        "verdict": FAVOURS if decisive else INDISTINGUISHABLE,
        "favours": (a if mean > 0 else b) if decisive else None,
        "a": a, "b": b,
        "n_folds": n,
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95": (mean - half_width, mean + half_width),
        "test": TEST_NAME,
        "p_value": p_value,
        "permutation_p": permutation_p,
        # False below MIN_FOLDS_FOR_PERMUTATION: the exact test's smallest
        # reachable p is 2/2^n, so at five folds it cannot reject at .05
        # however large the effect. Reported so a non-significant
        # permutation p is never read as evidence of no difference.
        "permutation_usable": permutation_usable,
        "folds_won_by_a": int((differences > 0).sum()),
        "folds_won_by_b": int((differences < 0).sum()),
        "underpowered_for": (
            None if abs(mean) >= MDE_80_POWER
            else f"|difference| {abs(mean):.4f} is below this design's "
                 f"minimum detectable effect of {MDE_80_POWER:.3f} nAP, so a "
                 f"verdict of INDISTINGUISHABLE here reflects the study's "
                 f"power rather than the models"
        ),
        "meaning": COMPARISON_MEANING,
        "power_limits": POWER_LIMITS,
        "note": (
            "A verdict of INDISTINGUISHABLE does not mean the candidates are "
            "equal. It means this data cannot tell them apart, which is a "
            "different and more useful statement than a point estimate."
        ),
    }


def _require_same_scope(results: list) -> None:
    """Refuse a comparison that spans production paths.

    Phase 6 and PS-native are both production, over different features and a
    different unit of analysis. A difference between them would not be a
    difference between models, and there is no correction that makes the
    number mean something - so it is refused rather than caveated.
    """
    scopes = {getattr(r, "scope", SCOPE_UNDECLARED) for r in results}
    if len(scopes) > 1:
        raise ScopeMismatchError(
            f"cannot compare candidates from different scopes {sorted(scopes)}. "
            f"{SCOPES.get(SCOPE_PHASE6)} {SCOPES.get(SCOPE_PS_NATIVE)} "
            f"A number from one describes nothing about the other."
        )


def _paired_differences(a: "CandidateResult", b: "CandidateResult") -> np.ndarray:
    common = {f.fold.label: f.nap for f in a.folds}
    paired = [
        (common[f.fold.label], f.nap)
        for f in b.folds if f.fold.label in common
    ]
    if not paired:
        return np.array([], dtype="float64")
    left = np.array([p[0] for p in paired], dtype="float64")
    right = np.array([p[1] for p in paired], dtype="float64")
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask] - right[mask]


def compare_family(results: list) -> dict:
    """Every pairwise comparison in one family, Holm-corrected.

    The correction is the point. Comparing three candidates pairwise at .05
    each gives roughly a 14% chance of declaring at least one spurious
    winner, and "we ran three comparisons and one was significant" is how a
    model gets selected on noise.
    """
    _require_same_scope(results)
    comparisons = []
    for i, left in enumerate(results):
        for right in results[i + 1:]:
            differences = _paired_differences(left, right)
            verdict = paired_verdict(
                differences, a=left.name, b=right.name
            )
            verdict["label"] = f"{left.name} vs {right.name}"
            comparisons.append(verdict)

    raw = {
        c["label"]: c.get("p_value", 1.0)
        for c in comparisons if np.isfinite(c.get("p_value", np.nan))
    }
    adjusted = holm(raw) if raw else {}
    for comparison in comparisons:
        p_adj = adjusted.get(comparison["label"], float("nan"))
        comparison["p_holm"] = p_adj
        # The corrected value is what decides. An uncorrected verdict left
        # in place would be the family-wise error this function exists to
        # control.
        if np.isfinite(p_adj) and p_adj >= ALPHA:
            comparison["verdict"] = INDISTINGUISHABLE
            comparison["favours"] = None

    return {
        "protocol_version": PROTOCOL_VERSION,
        "alpha": ALPHA,
        "correction": "holm-bonferroni",
        "family_size": len(comparisons),
        "comparisons": comparisons,
        "power_limits": POWER_LIMITS,
    }
