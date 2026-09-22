"""Feature health: constants, duplicates and functional dependencies.

Why a whole module for this
---------------------------
Eight of the thirty features in ``ps_native_v1`` carry no information that
another feature does not already carry, and nothing in the pipeline noticed.
The mechanism was one line in the dataset builder:

    in_amts = [tot_in / len(ins)] * len(ins)

Per-input and per-output amounts are not in the source data, so they were
synthesised by dividing the total evenly. Every consequence follows
mechanically:

``input_amount_std``, ``output_amount_std``
    Standard deviation of identical values. Max observed magnitude 2.8e-14,
    which is floating-point noise, not a measurement.

``input_amount_mean`` == ``input_amount_max``, and likewise for outputs
    The mean and max of identical values are the same number.

``equal_output_count`` == ``output_count``
    Every output matches every other, so the count of matching outputs is
    the count of outputs.

``output_entropy`` == log2(``output_count``)
    The entropy of a uniform distribution over n values.

``is_peeling_candidate``
    Requires ``out_max >= 0.8 * total_out`` with exactly two outputs. Equal
    outputs make ``out_max`` exactly ``0.5 * total_out``, so the condition
    is ``0.5 >= 0.8``. **The flag is mathematically incapable of firing** and
    is constant zero across all 262,433 rows.

``is_mixing_candidate``
    Reduces to ``input_count >= 3 and output_count >= 3`` once the equality
    test is vacuous. It fires on 34% of holdout addresses - a fan-out
    threshold wearing a detector's name.

Why it matters beyond tidiness
------------------------------
A duplicated feature splits its own importance with its twin, so any
feature-importance ranking over this set is wrong. A constant feature makes
an ablation group look like it was tested when it was not: "Group D:
Structural Patterns" contributes one dead flag, one cardinality rule and two
restatements of ``output_count``, and removing the group changes nothing -
which reads as "structural patterns do not help" when the truth is that they
were never measured.

This module finds those relationships so they cannot return silently.
``tests/test_feature_health.py`` pins the known set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Magnitude below which a standard deviation is floating-point noise rather
#: than a measurement. Chosen well above the 2.8e-14 actually observed and
#: far below any real quantity in this data.
NOISE_FLOOR = 1e-10

#: Tolerance for calling two columns the same column.
IDENTITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Finding:
    """One feature that carries no independent information."""

    feature: str
    kind: str
    detail: str
    related: tuple = ()

    def as_dict(self) -> dict:
        return {
            "feature": self.feature, "kind": self.kind,
            "detail": self.detail, "related": list(self.related),
        }


CONSTANT = "CONSTANT"
NOISE = "NUMERICALLY_ZERO"
DUPLICATE = "EXACT_DUPLICATE"
DEPENDENT = "FUNCTIONALLY_DEPENDENT"


def _numeric(frame: pd.DataFrame, columns) -> dict:
    out = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        out[column] = series.to_numpy(dtype="float64")
    return out


def find_constants(frame: pd.DataFrame, features) -> list:
    """Features with one distinct value, or a spread inside the noise floor."""
    findings = []
    for name, values in _numeric(frame, features).items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            findings.append(Finding(
                name, CONSTANT, "every value is missing", ()))
            continue
        distinct = np.unique(finite)
        if distinct.size == 1:
            findings.append(Finding(
                name, CONSTANT,
                f"constant {distinct[0]!r} across all {finite.size:,} rows; "
                f"it can carry no information and any group containing it "
                f"was not actually tested",
                (),
            ))
        elif float(np.abs(finite).max()) < NOISE_FLOOR:
            findings.append(Finding(
                name, NOISE,
                f"largest magnitude {np.abs(finite).max():.2e} is below the "
                f"{NOISE_FLOOR:.0e} noise floor; this is floating-point "
                f"residue, not a measurement",
                (),
            ))
    return findings


def find_duplicates(frame: pd.DataFrame, features) -> list:
    """Pairs of features that are the same column under two names."""
    numeric = _numeric(frame, features)
    names = list(numeric)
    findings = []
    seen = set()
    for i, left in enumerate(names):
        if left in seen:
            continue
        for right in names[i + 1:]:
            if right in seen:
                continue
            if np.allclose(
                numeric[left], numeric[right],
                rtol=IDENTITY_TOLERANCE, atol=IDENTITY_TOLERANCE,
                equal_nan=True,
            ):
                findings.append(Finding(
                    right, DUPLICATE,
                    f"identical to {left!r} on every row; the two split one "
                    f"feature's importance between them, so any importance "
                    f"ranking over this set is wrong",
                    (left,),
                ))
                seen.add(right)
    return findings


#: Transforms checked for functional dependence. Deliberately short: these
#: are the shapes that actually arise from summarising a synthesised uniform
#: distribution, not a general search for any relationship.
_TRANSFORMS = (
    ("log2", lambda x: np.log2(np.where(x > 0, x, np.nan))),
    ("log1p", np.log1p),
    ("sqrt", lambda x: np.sqrt(np.where(x >= 0, x, np.nan))),
)


def find_dependencies(frame: pd.DataFrame, features) -> list:
    """Features that are a fixed transform of another feature."""
    numeric = _numeric(frame, features)
    names = list(numeric)
    findings = []
    reported = set()
    for left in names:
        if left in reported:
            continue
        for right in names:
            if left == right or left in reported:
                continue
            for label, transform in _TRANSFORMS:
                with np.errstate(invalid="ignore", divide="ignore"):
                    transformed = transform(numeric[right])
                mask = np.isfinite(transformed) & np.isfinite(numeric[left])
                if mask.sum() < max(10, 0.5 * len(mask)):
                    continue
                if np.allclose(
                    numeric[left][mask], transformed[mask],
                    rtol=1e-7, atol=1e-7,
                ):
                    findings.append(Finding(
                        left, DEPENDENT,
                        f"equals {label}({right}) on every comparable row; it "
                        f"adds no information a tree model could not already "
                        f"extract from {right!r}",
                        (right,),
                    ))
                    # One finding per feature: a column that restates two
                    # others is still one redundant column, and reporting it
                    # twice would inflate the count.
                    reported.add(left)
                    break
    return findings


def audit_features(frame: pd.DataFrame, features) -> list:
    """Every finding, in one pass. The function a training script should call.

    Returns findings rather than raising, because the right response differs:
    a new model must not ship with these, while the frozen ``ps_native_v1``
    has them recorded as known and is not being retrained here.
    """
    present = [f for f in features if f in frame.columns]
    missing = [f for f in features if f not in frame.columns]
    findings = []
    for name in missing:
        findings.append(Finding(
            name, CONSTANT, "declared in the schema but absent from the frame",
            (),
        ))
    findings += find_constants(frame, present)
    already = {f.feature for f in findings}
    findings += [
        f for f in find_duplicates(frame, present) if f.feature not in already
    ]
    already |= {f.feature for f in findings}
    findings += [
        f for f in find_dependencies(frame, present)
        if f.feature not in already
    ]
    return findings


def healthy_features(frame: pd.DataFrame, features) -> list:
    """The features worth giving a model, in the declared order."""
    degenerate = {f.feature for f in audit_features(frame, features)}
    return [f for f in features if f not in degenerate]


def assert_healthy(frame: pd.DataFrame, features) -> None:
    """Raise if any feature carries no independent information.

    For a NEW candidate. Shipping a model whose feature set contains a
    constant and three restatements of one column is how a capability gets
    claimed that the code does not implement.
    """
    findings = audit_features(frame, features)
    if findings:
        lines = "\n".join(
            f"  {f.feature}: {f.kind} - {f.detail}" for f in findings
        )
        raise ValueError(
            f"{len(findings)} feature(s) carry no independent information:\n"
            f"{lines}\n"
            f"Remove them, or fix the generator that made them degenerate. "
            f"A model fitted on this set cannot support a claim about what "
            f"its features detect."
        )


#: The findings on ``ps_native_v1``, recorded so a regression test can assert
#: they are still detected - and so that fixing the generator shows up as
#: this list shrinking rather than as a silent change in behaviour.
KNOWN_DEGENERATE_PS_NATIVE = (
    "is_peeling_candidate",
    "input_amount_std",
    "output_amount_std",
    "input_amount_max",
    "output_amount_max",
    "equal_output_count",
    "output_entropy",
    "in_degree_asof_t",
    "out_degree_asof_t",
)
