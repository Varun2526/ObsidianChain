"""The behaviour repertoire: transaction SHAPES, not claims about people.

Each behaviour emits transactions with a characteristic structure. The suite
splits into two halves that matter equally:

The recognisable half - PEELING, MIXING_LIKE, RAPID_MOVEMENT and so on -
exists so a detector can be shown to find a structure that is there.

The adversarial half - EXCHANGE_BATCH, CONSOLIDATION, UNIFORM_PAYOUT,
BENIGN_EQUAL_SPLIT, MERCHANT_SWEEP - exists so a detector can be shown NOT to
find a structure that is not there. Every one of them produces a shape that
superficially resembles mixing or suspicious fan-in, and a detector that
cannot tell them apart is a detector that will bury an investigator in false
positives.

Nothing here asserts that a behaviour is unlawful. ``POSITIVE_CLASS`` below
designates which behaviours this synthetic evaluation treats as the positive
label, and that is a property of the experiment, not of the world.
"""

from __future__ import annotations

from dataclasses import dataclass

NORMAL = "NORMAL"
EXCHANGE_BATCH = "EXCHANGE_BATCH"
CONSOLIDATION = "CONSOLIDATION"
PEELING = "PEELING"
MIXING_LIKE = "MIXING_LIKE"
RAPID_MOVEMENT = "RAPID_MOVEMENT"
MULTI_HOP = "MULTI_HOP"

# ---- adversarial: benign, and shaped like something else ----------------
UNIFORM_PAYOUT = "UNIFORM_PAYOUT"
BENIGN_EQUAL_SPLIT = "BENIGN_EQUAL_SPLIT"
MERCHANT_SWEEP = "MERCHANT_SWEEP"

BEHAVIOURS = (
    NORMAL, EXCHANGE_BATCH, CONSOLIDATION, PEELING, MIXING_LIKE,
    RAPID_MOVEMENT, MULTI_HOP,
    UNIFORM_PAYOUT, BENIGN_EQUAL_SPLIT, MERCHANT_SWEEP,
)

#: The behaviours whose shapes are deliberately confusable with something a
#: detector is looking for. Held as data so a test can assert that NONE of
#: them is classified as a mixing pattern.
ADVERSARIAL = (
    EXCHANGE_BATCH, CONSOLIDATION, UNIFORM_PAYOUT, BENIGN_EQUAL_SPLIT,
    MERCHANT_SWEEP,
)

#: Which behaviours this synthetic evaluation labels positive.
#:
#: A LABELLING CONVENTION for a controlled experiment. It is not a claim that
#: peeling, mixing-like structure or rapid movement are unlawful: all three
#: have ordinary uses, and mixing in particular is a legal privacy technique.
#: The convention exists so that precision and recall have a defined meaning
#: on generated data where a real label does not exist.
POSITIVE_CLASS = (PEELING, MIXING_LIKE, RAPID_MOVEMENT)

#: What each behaviour is expected to LOOK like, in observable terms.
#: Asserted by ``tests/test_world.py``, so a generator change that stops
#: producing the shape fails rather than quietly weakening every experiment
#: built on it.
EXPECTED_SHAPE = {
    NORMAL: "one or two inputs, a payment and a change output, varied values",
    EXCHANGE_BATCH: "one input, many outputs of unrelated value",
    CONSOLIDATION: "many inputs swept into one output",
    PEELING: "a forward chain of two-output transactions, one small peel and "
             "one large remainder, at strictly increasing timesteps",
    MIXING_LIKE: "equal participant counts, outputs of one denomination, "
                 "inputs of varied value",
    RAPID_MOVEMENT: "many small transfers in consecutive timesteps",
    MULTI_HOP: "a linear sequence of single-input single-output transfers",
    UNIFORM_PAYOUT: "one input paying many outputs of the SAME value",
    BENIGN_EQUAL_SPLIT: "two inputs split into two equal outputs",
    MERCHANT_SWEEP: "two inputs paying many outputs of unrelated value",
}

#: The detector outcome each behaviour SHOULD produce, where the behaviour
#: has a definite expectation. Used by the scenario tests. ``None`` means the
#: shape does not determine an outcome and the test asserts only that the
#: pattern class is not reached.
EXPECTED_MIXING_CLASS = {
    MIXING_LIKE: "MIXING_PATTERN",
    NORMAL: "NO_MIXING_SIGNAL",
    EXCHANGE_BATCH: "NO_MIXING_SIGNAL",
    CONSOLIDATION: "NO_MIXING_SIGNAL",
    UNIFORM_PAYOUT: "NO_MIXING_SIGNAL",
    MERCHANT_SWEEP: "NO_MIXING_SIGNAL",
    BENIGN_EQUAL_SPLIT: "NO_MIXING_SIGNAL",
}


@dataclass(frozen=True)
class BehaviourConfig:
    """Shape parameters. Every one of them is a generator knob, not a finding."""

    #: Participants in a MIXING_LIKE round. Above the detector's minimum so
    #: the round is recognisable, small enough to read in a table.
    mixing_participants: int = 6

    #: The single denomination every MIXING_LIKE output carries.
    mixing_denomination: float = 0.1

    #: Outputs in an EXCHANGE_BATCH or MERCHANT_SWEEP.
    batch_outputs: int = 12

    #: Inputs swept by a CONSOLIDATION.
    consolidation_inputs: int = 9

    #: Hops in a PEELING chain. Above PEEL-1's primary depth of 5 so the
    #: chain is detected rather than sitting on the threshold.
    peel_depth: int = 7

    #: Fraction of the remaining value peeled off at each hop.
    peel_fraction: float = 0.08

    #: Transfers a RAPID_MOVEMENT entity makes.
    rapid_transfers: int = 8

    #: Hops in a MULTI_HOP relay.
    multi_hop_length: int = 5

    #: Outputs in a UNIFORM_PAYOUT, and their shared value.
    payout_outputs: int = 6
    payout_value: float = 0.25


DEFAULT_BEHAVIOUR_CONFIG = BehaviourConfig()

MEANING = (
    "These are synthetic transaction SHAPES generated to exercise the "
    "detectors. They are not claims about how real actors behave, and the "
    "positive-class designation is a labelling convention for a controlled "
    "experiment rather than a statement that a behaviour is unlawful."
)
