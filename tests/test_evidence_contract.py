"""The evidence artifact's public contract holds, in both directions.

Two things are being defended. First, that the codes a caller matches on are
stable while the human strings they came from are free to change - the reason
text embeds the threshold, so a frontend matching the text would break the day
the threshold moved. Second, that the mapping is **total**: four of the eight
branches never fire on the frozen dataset, and a permissive default is exactly
how a genuinely new rationale would reach a response labelled as something it
is not.

The mapping is checked against the real ``separation.py`` source rather than
against a list retyped here, because a list retyped here would agree with
itself forever.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from obsidianchain import evidence_contract as contract

SRC = Path(__file__).resolve().parents[1] / "src" / "obsidianchain"
SEPARATION = SRC / "network" / "separation.py"
BOUNDARY = SRC / "network" / "boundary.py"


def imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


# ---- the module is safe for the API to import ---------------------------


def test_the_contract_imports_nothing_but_the_standard_library() -> None:
    """It exists so the API can name these without importing eval/.

    If it grew a dependency on a computation module, importing it from a
    request handler would drag that module in and the whole reason for a
    separate top-level module would be gone.
    """
    offenders = sorted(
        name for name in imported_names(SRC / "evidence_contract.py")
        if name.split(".")[0] == "obsidianchain"
    )
    assert not offenders, f"evidence_contract imports {offenders}"


def test_the_contract_holds_no_computation() -> None:
    """Labels and lookups only. No statistic may be computed here."""
    source = (SRC / "evidence_contract.py").read_text(encoding="utf-8")
    for forbidden in ("import math", "scipy", "numpy", "pandas", "sklearn"):
        assert forbidden not in source, f"evidence_contract mentions {forbidden}"


# ---- the reason mapping is total ----------------------------------------


def _reason_strings(node) -> set[str]:
    """Every reason text an expression can evaluate to.

    Both branches of a conditional, because ``"effect below floor" if ... else
    "not significant"`` produces two distinct reasons from one expression and
    taking only the first would leave the other unmapped.

    An f-string collapses to its literal prefix, which is exactly what
    :func:`reason_code` matches on - the pooled reason interpolates the
    configured minimum and the prefix is the part that must stay stable.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _reason_strings(node.body) | _reason_strings(node.orelse)
    if isinstance(node, ast.JoinedStr):
        parts = [
            v.value for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        ]
        return {"".join(parts).strip()}
    return set()


def reasons_in_source(*paths: Path) -> set[str]:
    """Every reason text the given modules can attach to a verdict.

    Parsed from the AST rather than grepped, and covering both spellings the
    codebase actually uses: the ``reason=`` keyword at the construction site,
    and a local ``reason = ...`` bound a few lines earlier and passed through.
    Grepping for one would have missed the other.
    """
    found: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "reason":
                        found |= _reason_strings(keyword.value)
            elif isinstance(node, ast.Assign):
                if any(
                    isinstance(t, ast.Name) and t.id == "reason"
                    for t in node.targets
                ):
                    found |= _reason_strings(node.value)
    return found


def reasons_in_separation_source() -> set[str]:
    """Every reason reachable from the production evaluation path.

    ``constrained.py`` is included because the two codes declared unreachable
    are declared unreachable *relative to the funnel's walk*, not absent from
    the codebase - and the mapping has to be total over both.
    """
    return reasons_in_source(SEPARATION, SRC / "cluster" / "constrained.py")


def test_the_separation_source_actually_yields_reasons() -> None:
    """A vacuous scan would make the totality test below pass trivially."""
    found = reasons_in_separation_source()
    assert len(found) >= 7, sorted(found)
    # the two-branch conditional must have been unfolded, not half-read
    assert {"effect below floor", "not significant"} <= found


def test_every_reason_in_the_source_has_a_code() -> None:
    """Totality, measured against the real implementation."""
    unmapped = []
    for reason in sorted(reasons_in_separation_source()):
        try:
            contract.reason_code(reason)
        except contract.UnknownReasonError:
            unmapped.append(reason)
    assert not unmapped, (
        f"separation.py can emit {unmapped}, which have no code. Add them to "
        f"REASON_CODES rather than letting them fall through."
    )


def test_an_unmapped_reason_raises_rather_than_defaulting() -> None:
    """Section 12: an unexpected reason must fail loudly, not silently."""
    with pytest.raises(contract.UnknownReasonError) as excinfo:
        contract.reason_code("a rationale nobody has written yet")
    assert "REASON_CODES" in str(excinfo.value)


def test_the_pooled_reason_is_matched_by_prefix_not_by_threshold() -> None:
    """The threshold may move; the rationale is the same rationale."""
    for minimum in (1, 25, 250):
        reason = f"pooled observations below minimum {minimum}"
        assert contract.reason_code(reason) == "INSUFFICIENT_POOLED"


def test_the_empty_reason_is_separated_not_unknown() -> None:
    """SEPARATED carries no reason: there is nothing to explain."""
    assert contract.reason_code("") == "SEPARATED"


def test_the_codes_are_distinct_stable_symbols() -> None:
    codes = list(contract.REASON_CODES.values())
    assert len(codes) == len(set(codes)), "two reasons share a code"
    for code in codes:
        assert re.fullmatch(r"[A-Z][A-Z_]*[A-Z]", code), code


def test_no_code_embeds_a_threshold() -> None:
    """The whole point: a code must survive a configuration change."""
    for code in contract.REASON_CODES.values():
        assert not any(ch.isdigit() for ch in code), code


# ---- the unreachable branches are declared, not forgotten ---------------


def test_the_unreachable_codes_are_real_codes() -> None:
    for code in contract.UNREACHABLE_CODES:
        assert code in contract.REASON_CODES.values(), code


def test_the_unreachable_codes_carry_a_stated_reason() -> None:
    """A claim of unreachability with no justification is just a hope."""
    for code, why in contract.UNREACHABLE_CODES.items():
        assert len(why) > 20, f"{code} has no substantive justification"


def test_the_funnel_walks_without_the_veto_as_the_unreachability_claims() -> None:
    """STATIC_CANNOT_LINK is claimed unreachable because veto=False.

    Checked against the generator's source, so the claim stops being true the
    moment somebody changes the walk.
    """
    funnel = (SRC / "eval" / "evidence_funnel.py").read_text(encoding="utf-8")
    assert "veto=False" in funnel, (
        "the production funnel no longer walks with veto=False; "
        "STATIC_CANNOT_LINK may now be reachable and UNREACHABLE_CODES is "
        "stale"
    )


# ---- verdicts: cannot-link only -----------------------------------------


def test_no_verdict_asserts_sameness() -> None:
    """The network layer emits cannot-link only, by construction."""
    source = BOUNDARY.read_text(encoding="utf-8")
    for banned in ("MUST_LINK", "SAME_ORIGIN", "SAME_ENTITY"):
        assert banned not in source, (
            f"{banned} appeared in the verdict vocabulary; the network layer "
            f"has no mechanism for asserting sameness"
        )


# ---- the row configuration labels ---------------------------------------


def test_the_probe_label_is_canonical() -> None:
    assert contract.probe_row_config_label(1, 2) == (
        "probe:min_pooled=1,min_observer=2"
    )


def test_the_production_label_pins_float_formatting() -> None:
    """The label goes into the fingerprint; repr drift would invalidate ids."""
    assert contract.production_row_config_label(25, 5, 1e-4, 0.05) == (
        "production:min_pooled=25,min_observer=5,alpha=1e-04,min_effect=0.05"
    )


def test_the_production_label_is_stable_across_equal_values() -> None:
    """0.0001 and 1e-4 are the same number and must produce one label."""
    assert (
        contract.production_row_config_label(25, 5, 0.0001, 0.05)
        == contract.production_row_config_label(25, 5, 1e-4, 0.05)
    )


@pytest.mark.parametrize("changed", [
    {"production_min_pooled": 26},
    {"production_min_observer": 6},
    {"production_alpha": 1e-3},
    {"production_min_effect": 0.06},
    {"probe_min_pooled": 2},
    {"probe_min_observer": 3},
])
def test_every_configuration_term_moves_the_combined_label(changed) -> None:
    """Section 6. A term that does not move the label cannot move the
    fingerprint, and could change rows while every id kept resolving."""
    base = dict(
        probe_min_pooled=1, probe_min_observer=2,
        production_min_pooled=25, production_min_observer=5,
        production_alpha=1e-4, production_min_effect=0.05,
    )
    assert (
        contract.combined_row_config_label(**base)
        != contract.combined_row_config_label(**{**base, **changed})
    )


def test_the_combined_label_contains_both_halves() -> None:
    label = contract.combined_row_config_label(
        probe_min_pooled=1, probe_min_observer=2,
        production_min_pooled=25, production_min_observer=5,
        production_alpha=1e-4, production_min_effect=0.05,
    )
    assert label == (
        contract.probe_row_config_label(1, 2)
        + "|"
        + contract.production_row_config_label(25, 5, 1e-4, 0.05)
    )


# ---- the column lists ----------------------------------------------------


def test_the_production_columns_are_all_suffixed() -> None:
    """The suffix is what keeps a /2 column from shadowing a probe column."""
    for column in contract.PRODUCTION_COLUMNS:
        assert column.endswith("_production"), column


def test_the_nullable_columns_are_the_numeric_ones() -> None:
    """verdict, reason and reason_code are never null: they are the record
    of what happened, including that nothing was evaluated."""
    assert set(contract.NULLABLE_WHEN_UNEVALUATED) == {
        "dof_production", "chi2_production",
        "p_value_production", "effect_production",
    }
    for column in contract.NULLABLE_WHEN_UNEVALUATED:
        assert column in contract.PRODUCTION_COLUMNS


def test_the_rounded_columns_match_the_probe_precision() -> None:
    """Section 21. Without this a diff between the two blocks shows phantom
    differences of ~5e-07."""
    assert set(contract.ROUNDED_COLUMNS) <= set(contract.PRODUCTION_COLUMNS)
    assert contract.ROUNDING_DECIMALS == 6


# ---- the schema strings --------------------------------------------------


def test_the_two_schema_versions_are_distinct_and_ordered() -> None:
    assert contract.ARTIFACT_SCHEMA != contract.ARTIFACT_SCHEMA_V1
    assert contract.ARTIFACT_SCHEMA.endswith("/2")
    assert contract.ARTIFACT_SCHEMA_V1.endswith("/1")
    assert contract.ARTIFACT_SCHEMA.rsplit("/", 1)[0] == (
        contract.ARTIFACT_SCHEMA_V1.rsplit("/", 1)[0]
    )


def test_the_artifact_schema_is_not_the_provenance_schema() -> None:
    """Orthogonal versions: one names the columns, the other the record."""
    from obsidianchain import provenance as prov

    assert contract.ARTIFACT_SCHEMA != prov.PROVENANCE_SCHEMA


# ---- the frozen wordings -------------------------------------------------


@pytest.mark.parametrize("wording", [
    "NOT_SEPARATED_MEANING", "FROZEN_RUN_LIMITATION",
    "VERDICT_SCOPE", "VERDICT_DEFINITION",
    # Section 7. Frozen here, not built in the CLI, so a test can compare the
    # SERVED string against the constant rather than matching a prefix.
    "TRAJECTORY_EQUIVALENCE_VERIFIED",
    "TRAJECTORY_EQUIVALENCE_NOT_ESTABLISHED",
])
def test_each_frozen_wording_is_substantive(wording) -> None:
    text = getattr(contract, wording)
    assert isinstance(text, str) and len(text) > 100, wording


def test_the_two_trajectory_wordings_cannot_be_confused() -> None:
    """One asserts a verification, the other withholds it.

    They are served in the same field, so a reader distinguishes them by
    their text alone. The positive one must not be a prefix or substring of
    the negative one, and only one of them may claim verification.
    """
    verified = contract.TRAJECTORY_EQUIVALENCE_VERIFIED
    not_established = contract.TRAJECTORY_EQUIVALENCE_NOT_ESTABLISHED
    assert verified != not_established
    assert verified.startswith("VERIFIED:")
    assert not not_established.startswith("VERIFIED:")
    assert "NOT established" in not_established
    assert verified not in not_established


def test_not_separated_denies_sameness_explicitly() -> None:
    text = contract.NOT_SEPARATED_MEANING
    assert "is not evidence that" in text
    assert "cannot-link only" in text


def test_the_frozen_run_limitation_attributes_the_result_to_the_dataset() -> None:
    text = contract.FROZEN_RUN_LIMITATION
    assert "not" in text and "of the method" in text
    assert "SYNTHETIC" in text


def test_the_verdict_definition_denies_reconstruction() -> None:
    text = contract.VERDICT_DEFINITION
    for phrase in ("Not a reconstruction", "not a re-derivation",
                   "not a post-hoc classification"):
        assert phrase in text, phrase


def test_the_scope_separates_the_two_timings() -> None:
    text = contract.VERDICT_SCOPE
    assert "cluster_id is the" in text
    assert "component_size_at_record" in text
    assert "different times" in text
