"""The evidence artifact's public contract: schema, reason codes, row config.

Why a separate module
---------------------
Three parties need these definitions and no existing module can serve all
three. The artifact generator lives under ``eval/``; the API is forbidden from
importing ``eval`` or ``network.separation``; and the tests need to assert the
two agree. So the contract lives here, in a module that is standard-library
only and safe for the API to import - the same reasoning that put
:mod:`obsidianchain.run_fingerprint` at the top level in Phase 5.2.

What the artifact promises
--------------------------
``evidence_funnel.parquet`` stopped being an internal diagnostic the moment an
HTTP endpoint served it. Schema ``/1`` was the thirteen-column probe funnel.
Schema ``/2`` adds seven columns carrying the production evaluation, and the
API refuses to serve ``verdict``/``reason`` from a ``/1`` artifact rather than
silently omitting them.

The reason codes exist because the human strings must not become an API
contract. ``"pooled observations below minimum 25"`` embeds the threshold, so
a frontend matching on that string would break the day the threshold moved.
:data:`REASON_CODES` maps every reason the production rule can produce onto a
stable symbol, and it is a **total** function: an unmapped reason raises rather
than falling through to a default, because four of the eight branches have
zero occurrences on the frozen dataset and a silent default is how they would
stay untested.
"""

from __future__ import annotations

#: Content layout of ``evidence_funnel.parquet``. Distinct from - and
#: orthogonal to - ``provenance.PROVENANCE_SCHEMA``, which versions the
#: sidecar's provenance RECORD rather than the artifact's columns.
#:
#: /1  thirteen probe columns, Phase 3.1 through Phase 5.2
#: /2  adds the seven production-evaluation columns below
ARTIFACT_SCHEMA = "obsidianchain.evidence_funnel/2"
ARTIFACT_SCHEMA_V1 = "obsidianchain.evidence_funnel/1"


class UnknownReasonError(ValueError):
    """Raised when a production reason has no mapped code.

    Deliberately fatal to artifact generation. Four of the eight reason
    branches never fire on the frozen dataset, so a permissive default would
    let a genuinely new rationale reach an API response labelled as something
    it is not.
    """


#: Every reason ``separation_evidence`` and ``evaluate_cannot_link`` can
#: produce, mapped to a stable code. Total by construction - see
#: :func:`reason_code`.
#:
#: The two entries marked unreachable cannot occur in the production funnel:
#: it walks with ``veto=False`` (so no static cannot-link is ever consulted)
#: and always with an oracle present. They are here so the mapping is total
#: rather than merely sufficient.
REASON_CODES: dict[str, str] = {
    # --- NO_EVIDENCE, gate 1: the pooled minimum. The text embeds the
    # threshold, which is exactly why callers must match on the code.
    "pooled observations below minimum 25": "INSUFFICIENT_POOLED",
    # --- NO_EVIDENCE, gate 2: no observer dimension survived
    "no observer has enough paired observations": "INSUFFICIENT_OBSERVER",
    # --- NO_EVIDENCE, gate 3: every usable observer had zero variance
    "zero variance in every usable observer": "ZERO_VARIANCE",
    # --- SEPARATED carries an empty reason: nothing to explain, the
    # statistic cleared both gates.
    "": "SEPARATED",
    # --- NOT_SEPARATED, significant but too small to act on
    "effect below floor": "EFFECT_BELOW_FLOOR",
    # --- NOT_SEPARATED, not significant. Note the asymmetry frozen in
    # separation.py: p >= alpha yields THIS regardless of effect size.
    "not significant": "NOT_SIGNIFICANT",
    # --- unreachable in the production funnel; present for totality
    "static cannot-link": "STATIC_CANNOT_LINK",
    "no oracle": "NO_ORACLE",
}

#: Codes that cannot occur in the production evidence funnel, with the reason
#: they cannot. Asserted by test, so the claim stays true or fails loudly.
UNREACHABLE_CODES: dict[str, str] = {
    "STATIC_CANNOT_LINK": "the funnel walks with veto=False; no static "
                          "cannot-link is consulted",
    "NO_ORACLE": "the funnel always walks with an oracle present",
}

#: Threshold-bearing prefix of the gate-1 reason. The full string includes the
#: pooled minimum, so a lookup has to tolerate the number changing without
#: silently failing to a default.
_POOLED_PREFIX = "pooled observations below minimum"


def reason_code(reason: str) -> str:
    """Map a production reason onto its stable code.

    Total: raises :class:`UnknownReasonError` rather than returning a default.

    The gate-1 reason is matched by prefix because its text carries the
    configured minimum (``"...below minimum 25"``). Matching the whole string
    would break the moment the threshold moved, and breaking loudly there
    would be right for a *config* change but wrong here - the rationale is
    the same rationale whatever the number is.
    """
    if reason in REASON_CODES:
        return REASON_CODES[reason]
    if reason.startswith(_POOLED_PREFIX):
        return "INSUFFICIENT_POOLED"
    raise UnknownReasonError(
        f"reason {reason!r} has no mapped code. Add it to "
        f"evidence_contract.REASON_CODES with a stable symbol; do not let it "
        f"fall through to a default, which would label a new rationale as "
        f"something it is not."
    )


#: The seven columns schema /2 adds. Order is the on-disk order.
PRODUCTION_COLUMNS = [
    "verdict_production",
    "reason_code_production",
    "reason_production",
    "dof_production",
    "chi2_production",
    "p_value_production",
    "effect_production",
]

#: Production columns that are NULL wherever the evaluation short-circuited.
#:
#: All three NO_EVIDENCE gates return early with the dataclass defaults
#: (chi2=0.0, dof=0, p_value=1.0, effect=0.0), so those numbers are NOT
#: results. Persisting them raw would put a chi-square of exactly zero and a
#: p-value of exactly 1.0 on 253,389 rows that were never evaluated, reading
#: as "perfectly consistent" instead of "not evaluated".
#:
#: ``dof_production`` is nullable for the same reason and one more: because
#: all three gates are numerically identical, a stored 0 would conflate "gate
#: 1 never attempted" with "gates 2/3 attempted and found no usable
#: dimension". Null for all four; ``reason_code_production`` discriminates.
NULLABLE_WHEN_UNEVALUATED = [
    "dof_production",
    "chi2_production",
    "p_value_production",
    "effect_production",
]

#: Columns rounded before persisting, to the same 6 decimal places the probe
#: columns already use. Without this the two blocks are not comparable at
#: face value and a diff between them shows phantom differences of ~5e-07.
ROUNDED_COLUMNS = ["chi2_production", "effect_production"]
ROUNDING_DECIMALS = 6


def probe_row_config_label(min_pooled: int, min_observer: int) -> str:
    """Canonical label for the probe half of the row configuration."""
    return f"probe:min_pooled={int(min_pooled)},min_observer={int(min_observer)}"


def production_row_config_label(
    min_pooled: int, min_observer: int, alpha: float, min_effect: float
) -> str:
    """Canonical label for the production half.

    Float formatting is pinned rather than left to ``repr``: the label goes
    into the run fingerprint, so a formatting change would silently invalidate
    every evidence id. ``alpha`` uses scientific notation with no significant
    digits (1e-04) and ``min_effect`` uses ``%g`` (0.05).
    """
    return (
        f"production:min_pooled={int(min_pooled)},"
        f"min_observer={int(min_observer)},"
        f"alpha={float(alpha):.0e},"
        f"min_effect={float(min_effect):g}"
    )


def combined_row_config_label(
    *,
    probe_min_pooled: int,
    probe_min_observer: int,
    production_min_pooled: int,
    production_min_observer: int,
    production_alpha: float,
    production_min_effect: float,
) -> str:
    """The full ``row_config`` term for the evidence run fingerprint.

    Both configurations, because schema /2 rows are determined by both: the
    thirteen probe columns by the first, the seven production columns by the
    second. A fingerprint covering only the probe half would let the
    production rule change while every evidence id kept resolving - the
    silent re-point the composite fingerprint exists to prevent.
    """
    return (
        probe_row_config_label(probe_min_pooled, probe_min_observer)
        + "|"
        + production_row_config_label(
            production_min_pooled,
            production_min_observer,
            production_alpha,
            production_min_effect,
        )
    )


# ---- wording the API must not paraphrase --------------------------------

#: What NOT_SEPARATED means, and does not mean. Carried on every response
#: that surfaces it.
NOT_SEPARATED_MEANING = (
    "NOT_SEPARATED is not evidence that these two candidate components are "
    "the same entity. It means sufficient network observations were pooled on "
    "both sides to run the production separation test, and that test did not "
    "establish a statistically significant separation. Absence of separation "
    "evidence is not evidence of common ownership: the network layer emits "
    "cannot-link only and has no mechanism for asserting sameness."
)

#: The frozen-run limitation. Required on every response carrying a
#: production verdict, so a dashboard screenshot cannot be read as a claim
#: about the method.
FROZEN_RUN_LIMITATION = (
    "SEPARATED = 0 across this run is a property of this frozen dataset, not "
    "of the method. Of 253,429 proposed unions, 40 cleared the pooled "
    "minimum of 25 and none separated. That describes the reach and content "
    "of this SYNTHETIC network dataset - the median component pools zero "
    "usable observations - and is not evidence that the production rule never "
    "separates, nor that it would fail on real entities. Validating that "
    "needs mainnet capture with controlled ground truth."
)

#: What the verdict is attached to. Matters once transitive clustering
#: reaches a UI: the evidence belongs to a proposed union at a point in the
#: replay, not to a pair of final clusters.
VERDICT_SCOPE = (
    "This verdict describes the network evidence evaluated for THIS proposed "
    "union at THIS point in the replay trajectory. It must not be read as "
    "'these two final clusters are the same or different'. cluster_id is the "
    "final cluster; component_size_at_record is the size at the moment the "
    "union was proposed. They are different times."
)

#: Section 7. Why a chain-only artifact may be presented as the fused
#: trajectory: because a verification established the two coincide, for this
#: run, by comparing both forests element-wise.
#:
#: Frozen here rather than built in the CLI so that a test can assert the
#: SERVED string equals the constant. Asserting only that it starts with
#: "VERIFIED:" would pass against a sidecar somebody hand-edited, which is
#: the one thing a claim of verification must not do.
TRAJECTORY_EQUIVALENCE_VERIFIED = (
    "VERIFIED: no union was SEPARATED, none was refused, and the fused "
    "forest's roots equal the chain-only forest's. The fused trajectory is "
    "identical to the one recorded here."
)

#: The same field when the veto did fire. The artifact still publishes, but
#: it may no longer be described as the fused trajectory.
TRAJECTORY_EQUIVALENCE_NOT_ESTABLISHED = (
    "NOT established for this run: the veto fired, so the fused trajectory "
    "differs from the chain-only one recorded here."
)

#: How the production verdict was produced. Prevents a later reader treating
#: it as a reconstruction.
VERDICT_DEFINITION = (
    "verdict_production, reason_production and reason_code_production are the "
    "direct return values of separation_evidence() invoked with the frozen "
    "production configuration against the pooled statistics of the two "
    "components as they stood at this exact proposed merge. Not a "
    "reconstruction, not a re-derivation from persisted columns, and not a "
    "post-hoc classification."
)
