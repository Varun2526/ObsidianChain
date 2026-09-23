"""Versioned data contracts: what the system accepts, and what it computes.

``capture``  - the canonical input capture (PS field set): types, ranges,
               null semantics, per-txid consistency. Violations quarantine
               the transaction; they are never coerced into a plausible value.
``features`` - the feature catalog: every model feature with its source,
               point-in-time semantics, same-step behaviour, null semantics
               and valid range. It is also the per-feature leakage audit:
               a feature without a catalog entry cannot reach a model.
"""
