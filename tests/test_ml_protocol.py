"""The canonical evaluation protocol, and the seal over the holdout.

What these tests defend
-----------------------
Two failures this project has already had once each, in a form nobody
noticed until the artifacts were measured.

The first is evaluating on the holdout during development. The audit that
produced this protocol built rolling folds by hand and two of them - 41-44
and 45-48 - evaluated on timesteps 42 onward. No test could have caught it,
because there was nothing that knew where the holdout began.

The second is declaring a winner on one window. Fold-to-fold spread on this
data is roughly 35x the seed-to-seed spread, the model ranking flips between
folds, and every four-decimal number quoted so far is one draw from a
distribution with sd around 0.175.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obsidianchain.ml import protocol


@pytest.fixture(autouse=True)
def sealed():
    """Every test starts with the holdout sealed, whatever ran before."""
    protocol.reseal()
    yield
    protocol.reseal()


@pytest.fixture
def frame() -> pd.DataFrame:
    """A small time-ordered frame with a signal that drifts.

    The signal weakens with time on purpose: a protocol that averages folds
    must be able to show a candidate getting worse, not just report a number.
    """
    rng = np.random.default_rng(11)
    rows = []
    for t in range(1, protocol.DATASET_END + 1):
        for _ in range(200):
            y = int(rng.random() < 0.08)
            strength = max(0.0, 1.5 - 0.03 * t)
            rows.append({
                "first_t": t,
                "x": rng.normal(y * strength, 1.0),
                "noise": rng.normal(),
                "y": y,
            })
    return pd.DataFrame(rows)


def linear_candidate(train, evaluation, features, seed):
    """A deterministic scorer. No model library, so the test measures the
    protocol rather than sklearn."""
    weights = {f: float(train[f].corr(train["y"]) or 0.0) for f in features}
    return sum(evaluation[f].to_numpy() * w for f, w in weights.items())


# ---- fold construction --------------------------------------------------


def test_no_fold_reaches_the_holdout() -> None:
    """The defect that motivated the module, made unrepresentable."""
    for fold in protocol.rolling_origin_folds():
        assert fold.eval_end <= protocol.DEVELOPMENT_END
        assert fold.eval_start <= protocol.DEVELOPMENT_END


def test_a_fold_that_reaches_the_holdout_cannot_be_constructed() -> None:
    with pytest.raises(protocol.HoldoutSealError, match="sealed holdout"):
        protocol.Fold(index=0, train_end=40, eval_start=41, eval_end=44)


def test_a_fold_never_evaluates_on_what_it_trained_on() -> None:
    for fold in protocol.rolling_origin_folds():
        assert fold.eval_start > fold.train_end
    with pytest.raises(ValueError, match="also trains on"):
        protocol.Fold(index=0, train_end=30, eval_start=28, eval_end=31)


def test_the_folds_expand_rather_than_slide() -> None:
    """A deployed model refits on everything known; folds must match."""
    folds = protocol.rolling_origin_folds()
    assert len(folds) >= 3, folds
    train_ends = [f.train_end for f in folds]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)


def test_the_folds_do_not_overlap_each_other() -> None:
    covered = []
    for fold in protocol.rolling_origin_folds():
        covered.extend(range(fold.eval_start, fold.eval_end + 1))
    assert len(covered) == len(set(covered))


def test_fold_construction_is_deterministic() -> None:
    assert protocol.rolling_origin_folds() == protocol.rolling_origin_folds()


# ---- the seal -----------------------------------------------------------


def test_development_never_returns_a_holdout_row(frame) -> None:
    visible = protocol.development(frame)
    assert visible["first_t"].max() <= protocol.DEVELOPMENT_END
    assert len(visible) < len(frame), "the fixture must span the boundary"


def test_the_holdout_is_refused_by_default(frame) -> None:
    with pytest.raises(protocol.HoldoutSealError, match="sealed"):
        protocol.holdout(frame)


def test_breaking_the_seal_requires_a_written_reason() -> None:
    for bad in ("", "   ", "because", "final now"):
        with pytest.raises(ValueError, match="written reason"):
            protocol.break_seal(bad)


def test_a_broken_seal_is_recorded_with_its_reason(frame) -> None:
    reason = ("model, features and thresholds are final; this is the "
              "one-time freeze measurement")
    protocol.break_seal(reason)
    held = protocol.holdout(frame)
    assert held["first_t"].min() >= protocol.HOLDOUT_START
    state = protocol.seal_state()
    assert state["broken"] is True
    assert reason in state["reasons"]


def test_evaluate_candidate_cannot_see_the_holdout_even_if_handed_it(frame) -> None:
    """The frame passed in spans everything; the protocol still must not
    score a holdout row."""
    result = protocol.evaluate_candidate(
        frame, linear_candidate, name="linear", features=["x", "noise"],
    )
    assert result.folds
    for fold_result in result.folds:
        assert fold_result.fold.eval_end <= protocol.DEVELOPMENT_END


def test_a_frame_without_a_timestep_is_refused(frame) -> None:
    """Without time there is no holdout to seal."""
    with pytest.raises(KeyError, match="place a row in time"):
        protocol.development(frame.drop(columns=["first_t"]))


# ---- metrics ------------------------------------------------------------


def test_normalised_ap_puts_no_skill_at_zero() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(4000) < 0.05).astype(int)
    nap, prevalence = protocol.normalised_average_precision(y, rng.random(4000))
    assert prevalence == pytest.approx(y.mean())
    assert abs(nap) < 0.05, nap


def test_normalised_ap_puts_perfect_ranking_at_one() -> None:
    y = np.array([1] * 50 + [0] * 950)
    nap, _ = protocol.normalised_average_precision(y, -np.arange(1000))
    assert nap == pytest.approx(1.0, abs=1e-9)


def test_normalised_ap_fixes_the_floor_at_every_prevalence() -> None:
    """The property that makes folds averageable - and only that property.

    Development-fold prevalence ranges 3.3% to 21.0%. Raw AP has its floor AT
    the prevalence, so a no-skill model scores 0.03 on one fold and 0.21 on
    another and the average is dominated by which windows were chosen. nAP
    puts no-skill at 0 and perfect at 1 in EVERY fold, so the ends of the
    scale mean the same thing everywhere.

    It does NOT make a fixed-separation signal score identically across
    prevalences, and this test deliberately does not claim that: a window
    with more positives is genuinely easier to rank, and no normalisation
    should hide a real difference in the task.
    """
    rng = np.random.default_rng(5)
    for prevalence in (0.03, 0.10, 0.21):
        y = (rng.random(20000) < prevalence).astype(int)

        no_skill, reported = protocol.normalised_average_precision(
            y, rng.random(len(y))
        )
        assert reported == pytest.approx(y.mean())
        assert abs(no_skill) < 0.03, (prevalence, no_skill)

        perfect, _ = protocol.normalised_average_precision(
            y, y + rng.random(len(y)) * 1e-6
        )
        assert perfect == pytest.approx(1.0, abs=1e-6), (prevalence, perfect)


def test_precision_at_k_bounds_are_exact_under_a_full_tie() -> None:
    """Every score identical: precision@k can be anything the order allows."""
    y = np.array([1] * 10 + [0] * 90)
    worst, best, tied = protocol.precision_at_k_bounds(y, np.ones(100), 10)
    assert tied == 100
    assert worst == pytest.approx(0.0)
    assert best == pytest.approx(1.0)


def test_precision_at_k_bounds_collapse_when_there_is_no_tie() -> None:
    y = np.array([1] * 10 + [0] * 90)
    scores = -np.arange(100, dtype=float)
    worst, best, tied = protocol.precision_at_k_bounds(y, scores, 10)
    assert tied == 1
    assert worst == pytest.approx(best) == pytest.approx(1.0)


def test_precision_at_k_bounds_bracket_the_naive_value() -> None:
    """Whatever row order gives, it must lie inside the reported band."""
    rng = np.random.default_rng(7)
    y = (rng.random(500) < 0.2).astype(int)
    scores = np.round(rng.random(500), 2)  # heavy ties by construction
    worst, best, tied = protocol.precision_at_k_bounds(y, scores, 50)
    assert tied > 1
    naive = y[np.argsort(-scores, kind="stable")[:50]].mean()
    assert worst - 1e-12 <= naive <= best + 1e-12


# ---- candidate evaluation and comparison --------------------------------


def test_a_candidate_is_reported_with_a_spread_not_a_number(frame) -> None:
    result = protocol.evaluate_candidate(
        frame, linear_candidate, name="linear", features=["x", "noise"],
    )
    summary = result.summary()
    assert summary["folds"] >= 3
    assert np.isfinite(summary["nap_mean"])
    assert np.isfinite(summary["nap_sd"])
    assert len(summary["per_fold"]) == summary["folds"]


def test_every_fold_reports_its_own_prevalence(frame) -> None:
    """Without it a reader cannot tell a prevalence shift from a model change."""
    result = protocol.evaluate_candidate(
        frame, linear_candidate, name="linear", features=["x"],
    )
    for entry in result.summary()["per_fold"]:
        assert 0.0 < entry["prevalence"] < 1.0


def test_precision_at_k_is_reported_as_a_band(frame) -> None:
    result = protocol.evaluate_candidate(
        frame, linear_candidate, name="linear", features=["x"],
    )
    for entry in result.summary()["per_fold"]:
        for key, worst in entry["precision_at_k_worst"].items():
            assert worst <= entry["precision_at_k_best"][key] + 1e-12


def test_evaluation_is_deterministic(frame) -> None:
    first = protocol.evaluate_candidate(
        frame, linear_candidate, name="a", features=["x", "noise"],
    ).summary()
    second = protocol.evaluate_candidate(
        frame, linear_candidate, name="a", features=["x", "noise"],
    ).summary()
    assert first == second


def test_two_identical_candidates_are_indistinguishable(frame) -> None:
    """The headline behaviour: no winner is declared on noise."""
    a = protocol.evaluate_candidate(
        frame, linear_candidate, name="a", features=["x"])
    b = protocol.evaluate_candidate(
        frame, linear_candidate, name="b", features=["x"])
    verdict = protocol.compare(a, b)
    assert verdict["verdict"] == protocol.INDISTINGUISHABLE
    assert verdict["favours"] is None
    assert verdict["mean_nap_difference"] == pytest.approx(0.0, abs=1e-12)


def test_a_genuinely_better_candidate_is_favoured(frame) -> None:
    """The protocol must still be able to detect a real difference.

    A test that only ever said INDISTINGUISHABLE would be useless, so this
    pins the other direction: signal beats pure noise on every fold.
    """
    signal = protocol.evaluate_candidate(
        frame, linear_candidate, name="signal", features=["x"])

    def noise_only(train, evaluation, features, seed):
        return evaluation["noise"].to_numpy()

    noise = protocol.evaluate_candidate(
        frame, noise_only, name="noise", features=["noise"])

    verdict = protocol.compare(signal, noise)
    assert verdict["verdict"] == protocol.FAVOURS
    assert verdict["favours"] == "signal"
    assert verdict["folds_won_by_a"] > verdict["folds_won_by_b"]


def test_a_small_difference_is_not_called_a_win(frame) -> None:
    """A difference inside the fold spread is a statement about windows."""
    a = protocol.evaluate_candidate(
        frame, linear_candidate, name="a", features=["x"])

    def barely_different(train, evaluation, features, seed):
        return linear_candidate(train, evaluation, features, seed) + 1e-9

    b = protocol.evaluate_candidate(
        frame, barely_different, name="b", features=["x"])
    assert protocol.compare(a, b)["verdict"] == protocol.INDISTINGUISHABLE


def test_the_comparison_is_paired_by_fold(frame) -> None:
    a = protocol.evaluate_candidate(
        frame, linear_candidate, name="a", features=["x"])
    b = protocol.evaluate_candidate(
        frame, linear_candidate, name="b", features=["x"])
    verdict = protocol.compare(a, b)
    assert verdict["folds_compared"] == len(a.folds)


def test_the_comparison_explains_what_indistinguishable_means(frame) -> None:
    a = protocol.evaluate_candidate(
        frame, linear_candidate, name="a", features=["x"])
    verdict = protocol.compare(a, a)
    assert "does not mean the candidates are equal" in verdict["note"]
    assert "35x" in verdict["meaning"]


def test_the_protocol_states_what_the_seal_is_for() -> None:
    assert "read once" in protocol.SEAL_MEANING
    assert "not a held-out number" in protocol.SEAL_MEANING


# ---- the corrected decision rule (measured, not reasoned) --------------
#
# The original rule was `|mean diff| > 1 sd`, chosen by argument. Simulation
# over 10,000 trials at the observed paired sd put its false-positive rate at
# 8.7% - it declares a winner roughly one time in eleven when there is no
# difference at all. A paired t-test at alpha=.05 measured 5.1%.


def test_the_decision_rule_is_a_calibrated_test() -> None:
    """Not a heuristic multiple of the standard deviation."""
    assert protocol.ALPHA == 0.05
    assert protocol.TEST_NAME == "paired_t"


def test_no_difference_is_not_called_a_win() -> None:
    zero = np.zeros(12)
    verdict = protocol.paired_verdict(zero)
    assert verdict["verdict"] == protocol.INDISTINGUISHABLE
    assert verdict["p_value"] > protocol.ALPHA


def test_a_large_consistent_difference_is_called_a_win() -> None:
    consistent = np.full(12, 0.30) + np.linspace(-0.01, 0.01, 12)
    verdict = protocol.paired_verdict(consistent)
    assert verdict["verdict"] == protocol.FAVOURS
    assert verdict["p_value"] < protocol.ALPHA


def test_the_verdict_reports_a_confidence_interval() -> None:
    """A point estimate alone is what got this project into trouble."""
    verdict = protocol.paired_verdict(np.full(12, 0.30))
    low, high = verdict["ci95"]
    assert low <= verdict["mean_difference"] <= high


def test_the_verdict_carries_the_permutation_check(  ) -> None:
    """Assumption-free companion. Needs n>=7 to be able to reject at all."""
    verdict = protocol.paired_verdict(np.full(12, 0.30) + np.linspace(-0.01, 0.01, 12))
    assert verdict["permutation_p"] < protocol.ALPHA
    assert verdict["permutation_usable"] is True


def test_the_permutation_check_declares_itself_unusable_below_seven_folds() -> None:
    """At n=5 the smallest reachable two-sided p is 2/32 = 0.0625.

    The test cannot reject at .05 no matter how large the effect, so it must
    say so rather than silently returning a non-significant p.
    """
    verdict = protocol.paired_verdict(np.full(5, 10.0))
    assert verdict["permutation_usable"] is False
    assert verdict["permutation_p"] >= 0.0625


def test_holm_correction_is_applied_across_a_family() -> None:
    """Three pairwise comparisons is a family; uncorrected p inflates."""
    raw = {"a_vs_b": 0.02, "a_vs_c": 0.03, "b_vs_c": 0.60}
    adjusted = protocol.holm(raw)
    assert adjusted["a_vs_b"] == pytest.approx(0.06)
    assert adjusted["a_vs_c"] == pytest.approx(0.06)
    assert adjusted["b_vs_c"] == pytest.approx(0.60)
    assert all(adjusted[k] >= raw[k] for k in raw)


def test_holm_is_monotone() -> None:
    adjusted = protocol.holm({"x": 0.001, "y": 0.02, "z": 0.04})
    values = [adjusted["x"], adjusted["y"], adjusted["z"]]
    assert values == sorted(values)


def test_compare_family_corrects_for_multiplicity(frame) -> None:
    results = [
        protocol.evaluate_candidate(
            frame, linear_candidate, name=n, features=["x"])
        for n in ("a", "b", "c")
    ]
    family = protocol.compare_family(results)
    assert len(family["comparisons"]) == 3
    for entry in family["comparisons"]:
        assert entry["p_holm"] >= entry["p_value"]


# ---- the protocol declares what it cannot detect ------------------------


def test_the_default_design_gives_at_least_twelve_folds() -> None:
    """Five folds had an MDE of 0.311 nAP - larger than any difference
    between these model families. Twelve halves it to 0.164."""
    assert len(protocol.rolling_origin_folds()) >= 12


def test_the_minimum_detectable_effect_is_declared(  ) -> None:
    assert protocol.MDE_80_POWER == pytest.approx(0.164, abs=0.02)
    assert "0.05" in protocol.POWER_LIMITS
    assert "110 folds" in protocol.POWER_LIMITS


def test_every_fold_still_holds_enough_positives_to_measure() -> None:
    """Narrower folds buy power and cost window size. This pins the floor
    that made width=2 the choice over width=1."""
    assert protocol.MIN_POSITIVES_PER_FOLD >= 100


# ---- two model paths, two scopes ---------------------------------------
#
# Phase 6 (Elliptic++ reference artifacts) and PS-native (uploaded-dataset
# runs) are both production, with declared scopes. Their numbers describe
# different feature sets over different unit definitions, so a comparison
# ACROSS them is meaningless and must be impossible rather than discouraged.


def test_a_candidate_carries_its_scope(frame) -> None:
    result = protocol.evaluate_candidate(
        frame, linear_candidate, name="a", features=["x"],
        scope=protocol.SCOPE_PS_NATIVE,
    )
    assert result.scope == protocol.SCOPE_PS_NATIVE


def test_the_default_scope_is_explicit_not_guessed(frame) -> None:
    result = protocol.evaluate_candidate(
        frame, linear_candidate, name="a", features=["x"])
    assert result.scope == protocol.SCOPE_UNDECLARED


def test_comparing_across_scopes_is_refused(frame) -> None:
    a = protocol.evaluate_candidate(
        frame, linear_candidate, name="a", features=["x"],
        scope=protocol.SCOPE_PS_NATIVE)
    b = protocol.evaluate_candidate(
        frame, linear_candidate, name="b", features=["x"],
        scope=protocol.SCOPE_PHASE6)
    with pytest.raises(protocol.ScopeMismatchError, match="different scopes"):
        protocol.compare(a, b)


def test_comparing_within_a_scope_is_allowed(frame) -> None:
    a = protocol.evaluate_candidate(
        frame, linear_candidate, name="a", features=["x"],
        scope=protocol.SCOPE_PS_NATIVE)
    b = protocol.evaluate_candidate(
        frame, linear_candidate, name="b", features=["x"],
        scope=protocol.SCOPE_PS_NATIVE)
    assert protocol.compare(a, b)["verdict"] == protocol.INDISTINGUISHABLE


def test_a_family_must_share_one_scope(frame) -> None:
    results = [
        protocol.evaluate_candidate(
            frame, linear_candidate, name=n, features=["x"], scope=s)
        for n, s in (("a", protocol.SCOPE_PS_NATIVE),
                     ("b", protocol.SCOPE_PS_NATIVE),
                     ("c", protocol.SCOPE_PHASE6))
    ]
    with pytest.raises(protocol.ScopeMismatchError):
        protocol.compare_family(results)


def test_the_scopes_are_named_and_documented() -> None:
    assert protocol.SCOPE_PHASE6 in protocol.SCOPES
    assert protocol.SCOPE_PS_NATIVE in protocol.SCOPES
    for scope, description in protocol.SCOPES.items():
        assert len(description) > 40, scope
