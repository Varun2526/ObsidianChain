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

SCORE_SCOPE = (
    "Risk is the calibrated probability the Phase 6 model assigned to an "
    "address, aggregated to its cluster. Scores are out-of-sample: alerts "
    "are generated from the TEST split only, because train and validation "
    "scores are optimistic by construction and ranking them beside test "
    "scores would mislead."
)
