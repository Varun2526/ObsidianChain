"""The Phase 7 alert contract: schemas, categories, and the frozen wordings.

Standard library only, so the API may import it without reaching a
recomputation path - the same reason :mod:`obsidianchain.evidence_contract`
exists for Phase 5.

What an alert is, and is not
----------------------------
An alert is a CLUSTER that the Phase 6 model scored highly. A cluster is an
inference produced by co-spend clustering, not a person and not an account.
Nothing in this layer asserts identity, ownership, or that two addresses
belong to the same human being.

The four explanation categories exist so that a reader can always tell what
kind of claim they are looking at:

``MODEL_SIGNAL``
    A SHAP contribution. "This feature moved the model's score." It is a
    statement about the model, never about Bitcoin.

``BLOCKCHAIN_CONTEXT``
    An observed on-chain quantity - value, counterparty count, chain depth.
    True of the ledger, independent of any model.

``NETWORK_CONTEXT``
    A summary of SYNTHETIC announcement observations. Investigative context
    only. It never says who sent anything.

``INSUFFICIENT_EVIDENCE``
    The quantity was not computable for this address. Returned instead of a
    zero, because "not measured" and "measured as none" are different
    answers and only one of them is honest here.
"""

from __future__ import annotations

import re

ALERT_SCHEMA = "obsidianchain.alerts/1"
MEMBER_SCHEMA = "obsidianchain.alert_members/1"
EXPLANATION_SCHEMA = "obsidianchain.alert_explanations/1"
TIMELINE_SCHEMA = "obsidianchain.alert_timeline/1"
RELATIONSHIP_SCHEMA = "obsidianchain.alert_relationships/1"
NETWORK_SCHEMA = "obsidianchain.alert_network/1"

#: ``<16 hex run fingerprint>:<cluster id>``. Anchored, no sign, no leading
#: zeros, so exactly one string addresses one alert.
ALERT_ID = re.compile(r"^([0-9a-f]{16}):(0|[1-9][0-9]*)$")

MODEL_SIGNAL = "MODEL_SIGNAL"
BLOCKCHAIN_CONTEXT = "BLOCKCHAIN_CONTEXT"
NETWORK_CONTEXT = "NETWORK_CONTEXT"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

CATEGORIES = (
    MODEL_SIGNAL, BLOCKCHAIN_CONTEXT, NETWORK_CONTEXT, INSUFFICIENT_EVIDENCE,
)

#: Feature-group prefix -> the context category its VALUES belong to.
#: A SHAP contribution over any of them is MODEL_SIGNAL regardless of group;
#: the group decides what the underlying VALUE is evidence of.
GROUP_CATEGORY = {
    "M0": BLOCKCHAIN_CONTEXT,
    "M1": BLOCKCHAIN_CONTEXT,
    "M2": BLOCKCHAIN_CONTEXT,
    "M3": NETWORK_CONTEXT,
    # M4 is mixing / CoinJoin-like STRUCTURE: an observed property of a
    # transaction's inputs and outputs, true of the ledger and independent of
    # any model - so BLOCKCHAIN_CONTEXT, exactly like M2's chain structure.
    "M4": BLOCKCHAIN_CONTEXT,
}

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW")

#: Aggregations carried on every alert. Phase 6 evaluated all four; the
#: artifact serves all four and names which one ranked it, so a reader is
#: never shown one number as though it were the only one.
AGGREGATIONS = ("AGG-MAX", "AGG-TOPK", "AGG-QUANT", "AGG-WMEAN")

#: Relationships this data actually supports. Nothing else may be emitted.
CO_SPEND = "CO_SPEND_COMPONENT"
FUNDED = "FUNDED_VIA_TRANSACTION"
RELATIONSHIPS = {
    CO_SPEND: (
        "Both addresses were inputs to a common transaction, directly or "
        "transitively, so the common-input-ownership heuristic groups them. "
        "This is a HEURISTIC about spending, not proof of shared ownership."
    ),
    FUNDED: (
        "One address was an input to a transaction that paid the other. "
        "This is an observed ledger fact and says nothing about who "
        "controls either address."
    ),
}


class AlertIdInvalidError(ValueError):
    """The id is not ``<16 hex>:<non-negative int>``."""


class AlertIdStaleError(ValueError):
    """Well-formed, but minted against a different run."""


class AlertNotFoundError(LookupError):
    """The run matches but no alert carries that cluster id."""


def parse_alert_id(raw: str) -> tuple[str, int]:
    match = ALERT_ID.match(raw or "")
    if match is None:
        raise AlertIdInvalidError(
            f"{raw!r} is not a valid alert id. Expected "
            f"'<16 hex run fingerprint>:<cluster id>'."
        )
    return match.group(1), int(match.group(2))


def make_alert_id(run_fingerprint: str, cluster_id: int) -> str:
    return f"{str(run_fingerprint)[:16]}:{int(cluster_id)}"


# ---- wording the API must not paraphrase --------------------------------

ALERT_MEANING = (
    "An alert is a co-spend CLUSTER whose member addresses the Phase 6 model "
    "scored highly. A cluster is an inference produced by the "
    "common-input-ownership heuristic, not a person, an account or a legal "
    "entity. A high score is a ranking signal for investigator attention. It "
    "is not a finding, not an accusation, and not evidence of any offence."
)

NETWORK_CONTEXT_MEANING = (
    "Network observations describe when and where transaction announcements "
    "were seen by passive observers. They are INVESTIGATIVE CONTEXT only. "
    "They do not establish that an IP address owns, controls or sent from a "
    "wallet, and two addresses sharing network characteristics are not "
    "thereby the same person: one server can broadcast for tens of thousands "
    "of unrelated users. The network layer can only ever say that two groups "
    "look DIFFERENT, never that they are the same."
)

ANNOUNCING_PEER_MEANING = (
    "An announcing peer is the node an OBSERVER first heard this transaction "
    "from. It is a relay vantage point, not the originator and not the "
    "sender. Measured on this dataset, 170,899 of 202,804 transactions "
    "(84.3%) were announced by more than one peer, which is what gossip "
    "relay looks like. Do not read a peer IP as the party who made the "
    "payment, and do not read two transactions sharing a peer as being sent "
    "by the same person."
)

SYNTHETIC_NETWORK_WARNING = (
    "All network observations in this deployment are SYNTHETIC. They "
    "demonstrate the mechanism and validate nothing about Bitcoin. Any "
    "network figure shown here is a property of a generated dataset."
)

MODEL_SIGNAL_MEANING = (
    "A model signal is a SHAP contribution in log-odds: how much this "
    "feature moved THIS model's score for THIS address, relative to the "
    "model's base value. It is a statement about the model's behaviour, not "
    "about Bitcoin, and a large contribution is not evidence of wrongdoing."
)

INSUFFICIENT_EVIDENCE_MEANING = (
    "The quantity could not be established for this address from the "
    "available data. It is reported as insufficient evidence rather than as "
    "zero, because 'not measured' and 'measured as none' are different "
    "answers and conflating them would invent a fact."
)

#: The mixing detector's four outcomes. Named here, not in the detector, so
#: the API can speak about them without importing a module that can compute
#: one - the same separation alerts/feature_groups.py exists for.
MIXING_PATTERN = "MIXING_PATTERN"
MIXING_LIKELIHOOD = "MIXING_LIKELIHOOD"
NO_MIXING_SIGNAL = "NO_MIXING_SIGNAL"
MIXING_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

MIXING_CLASSES = (
    MIXING_PATTERN, MIXING_LIKELIHOOD, NO_MIXING_SIGNAL,
    MIXING_INSUFFICIENT_DATA,
)

SUPPRESSOR_NONE = ""
SUPPRESSOR_BATCH = "BATCH_SHAPE"
SUPPRESSOR_CONSOLIDATION = "CONSOLIDATION_SHAPE"
SUPPRESSOR_ORDINARY = "ORDINARY_SHAPE"
SUPPRESSOR_UNIFORM_PAYOUT = "UNIFORM_PAYOUT_SHAPE"

#: Why a structurally-similar transaction was NOT called a mixing pattern.
#: Recorded per transaction so the reason is auditable rather than implied.
MIXING_SUPPRESSORS = {
    SUPPRESSOR_BATCH: (
        "One or two inputs paying many outputs. That is an exchange or "
        "merchant batch, not a collaborative spend: a mixing pattern needs "
        "many INDEPENDENT funders."
    ),
    SUPPRESSOR_CONSOLIDATION: (
        "Many inputs paying one or two outputs. That is a consolidation - a "
        "wallet sweeping its own UTXOs - and it produces the fan-in of a "
        "mixing transaction with none of the fan-out."
    ),
    SUPPRESSOR_ORDINARY: (
        "Few inputs and few outputs: the ordinary payment-plus-change shape, "
        "which is the majority of all transactions."
    ),
    SUPPRESSOR_UNIFORM_PAYOUT: (
        "Outputs are near-identical but the inputs are not numerous enough "
        "for the outputs to belong to different parties. A single payer "
        "sending one amount to many recipients looks like this."
    ),
}

MIXING_INSUFFICIENT_DATA_MEANING = (
    "The input and output value summaries needed to measure this "
    "transaction's structure were not available, so no classification was "
    "made. Reported as insufficient data rather than as no signal, because "
    "an unmeasured transaction and a measured-and-clean one are different "
    "answers."
)

MIXING_PATTERN_MEANING = (
    "A mixing-like pattern is a STRUCTURAL observation: several "
    "independent-looking inputs, several outputs of near-identical value, "
    "and input values that are not themselves uniform. It is not proof that "
    "a mixing service was used, and using one is not itself unlawful. "
    "Benign shapes that produce the same fan-in or fan-out - exchange "
    "batches, consolidations, uniform payouts - are detected separately and "
    "suppressed, with the reason recorded."
)

PEEL_STRUCTURE_MEANING = (
    "A peeling-chain-like structure is a repeated ordered sequence of "
    "transactions, each with two or three outputs, in which one address is "
    "an output of one and an input of the next at a strictly later timestep. "
    "It is structural candidate generation, not laundering classification: "
    "the depth of a chain is an observation about the ledger's shape and "
    "says nothing about intent or legality."
)

SCORE_SCOPE = (
    "Risk is the calibrated probability the Phase 6 model assigned to an "
    "address, aggregated to its cluster. Scores are out-of-sample: alerts "
    "are generated from the TEST split only, because train and validation "
    "scores are optimistic by construction and ranking them beside test "
    "scores would mislead."
)
